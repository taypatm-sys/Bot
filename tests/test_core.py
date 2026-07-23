import asyncio
import io
import sqlite3
import tempfile
import unittest
import random
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from PIL import Image

from app.analysis_coordinator import AnalysisCoordinator
from app.config import (
    Config,
    ConfigError,
    DEFAULT_GEMINI_IMAGE_MODEL,
    DEFAULT_GEMINI_MODEL,
    normalize_gemini_image_size,
    normalize_gemini_model,
)
from app.copywriter import ProductCopy, shorten_design_name
from app.formatting import (
    TemplateError,
    contact_link,
    normalize_price,
    normalize_size,
    render_caption,
    render_caption_text,
    validate_template,
)
from app.health import build_health_app
from app.mockup_generator import (
    MockupAnalysisError,
    MockupGeometryError,
    NormalizedBox,
    MockupGenerator,
    MockupSpec,
    _DetectedMockup,
    _DetectedMockupResponse,
    _ResponseBox,
    build_mockup_spec,
    build_model_photo_prompt,
    choose_photo_directions,
    ensure_mockup_spec_ready,
    inspect_print_file,
    normalize_detected_mockup,
    prepare_analysis_image,
)
from app.reference_catalog import (
    ReferenceCatalog,
    ReferenceCompatibility,
    normalize_reference_urls,
)
from app.scheduling import parse_local_datetime
from app.storage import PostRepository
from app.template_store import CaptionTemplateStore


class SchemaCompatibilityTests(unittest.TestCase):
    def test_reference_schema_has_no_exclusive_minimum(self) -> None:
        schema_text = str(ReferenceCompatibility.model_json_schema())
        self.assertNotIn("exclusiveMinimum", schema_text)

    def test_pixel_box_overflow_is_repaired(self) -> None:
        from app.mockup_generator import _box_from_response

        box = _box_from_response(
            _ResponseBox(x=220, y=180, width=1100, height=1180),
            label="garment_panel_box",
            image_width=1280,
            image_height=1280,
            minimum_width=10,
            minimum_height=10,
        )
        self.assertLessEqual(box.x + box.width, 100)
        self.assertLessEqual(box.y + box.height, 100)


class FormattingTests(unittest.TestCase):
    def test_numeric_price_gets_currency(self) -> None:
        self.assertEqual(normalize_price(" 240 "), "240 манат")

    def test_written_price_is_preserved(self) -> None:
        self.assertEqual(normalize_price("от 250 TMT"), "от 250 TMT")

    def test_contact_link_contains_prefilled_text(self) -> None:
        link = contact_link("taypa", "Принт Алабай")
        self.assertTrue(link.startswith("https://t.me/taypa?text="))
        self.assertIn("%D0%9F%D1%80%D0%B8%D0%BD%D1%82", link)

    def test_template_requires_all_fields(self) -> None:
        with self.assertRaises(TemplateError):
            validate_template("{title} {price}")

    def test_custom_size_is_normalized(self) -> None:
        self.assertEqual(normalize_size("  M - 3XL  "), "M - 3XL")

    def test_render_caption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "template.txt"
            path.write_text(
                "{title}\n{size}\n\nЦена: {price}\n\n{hashtags}",
                encoding="utf-8",
            )
            caption = render_caption(
                path,
                title="Название",
                description="Описание",
                size="S-XXL",
                price="240 TMT",
                garment_type="Футболка",
                design_name="Название",
                theme_hashtag="принт",
            )
            self.assertEqual(
                caption,
                "Название\nS-XXL\n\nЦена: 240 TMT\n\n#футболка #принт",
            )

    def test_russian_template_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "template.txt"
            path.write_text(
                '{Тип товара} "{Название принта}"\n\n'
                "{Короткое описание, передающее настроение принта}\n\n"
                "👕 {Размеры}\n\n💸 {Цена}\n\n"
                "#Taýpa #{тип товара} #{тематика принта}",
                encoding="utf-8",
            )
            caption = render_caption(
                path,
                title='Футболка "Welcome to Turkmenistan"',
                description="Футболка, которая передает атмосферу нашей страны ❤️",
                size="S-XXL",
                price="250 манат",
                garment_type="Футболка",
                design_name="Welcome to Turkmenistan",
                theme_hashtag="#Туркменистан",
            )
            self.assertIn('Футболка "Welcome to Turkmenistan"', caption)
            self.assertIn("👕 S-XXL", caption)
            self.assertIn("#Taýpa #футболка #туркменистан", caption)

    def test_render_caption_from_persistent_template_text(self) -> None:
        caption = render_caption_text(
            "{title}\n{size}\n{price}\n{hashtags}",
            title="Футболка Тест",
            description="Описание",
            size="S-XXL",
            price="250 манат",
            garment_type="Футболка",
            design_name="Тест",
            theme_hashtag="тест",
        )
        self.assertIn("#футболка #тест", caption)


class CopywriterTests(unittest.TestCase):
    def test_product_title_contains_garment_and_quoted_name(self) -> None:
        product = ProductCopy(
            garment_type="Худи",
            design_name="“Soul of Karakum”",
            mood_description="Стиль с характером",
            theme_hashtag="#Karakum Spirit",
        )
        self.assertEqual(product.title, 'Худи "Soul of Karakum"')
        self.assertEqual(product.description, "Стиль с характером")
        self.assertEqual(product.hashtags, "#худи #karakum_spirit")

    def test_long_generic_name_is_shortened_naturally(self) -> None:
        self.assertEqual(
            shorten_design_name("Стильная восточная красавица с котом"),
            "Красавица с котом",
        )

    def test_title_never_exceeds_three_words(self) -> None:
        product = ProductCopy(
            garment_type="Футболка",
            design_name="Очень длинное название красивого восточного принта",
            mood_description="Спокойный национальный образ для повседневного настроения",
            theme_hashtag="восток",
        )
        name = product.title.split('"', 2)[1]
        self.assertLessEqual(len(name.split()), 3)
        self.assertLessEqual(len(name), 28)


class MockupGeneratorTests(unittest.TestCase):
    def test_mockup_image_is_normalized_to_supported_jpeg(self) -> None:
        image = Image.new("RGB", (997, 767), (70, 70, 74))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=80)

        prepared = prepare_analysis_image(buffer.getvalue())

        self.assertEqual(prepared.mime_type, "image/jpeg")
        self.assertEqual((prepared.width, prepared.height), (997, 767))
        self.assertGreater(len(prepared.data), 1000)

    def test_loose_analysis_response_is_normalized_before_validation(self) -> None:
        raw = _DetectedMockupResponse(
            side="frontside",
            garment_type="T shirt",
            shirt_color="washed charcoal gray",
            fabric_finish="mineral washed cotton",
            fit="relaxed",
            target_gender="neutral",
            target_age_group="all adults",
            moods=["cute", "youthful"],
            print_theme="cartoon cat with lettering",
            construction_details="crew neck and short sleeves",
            garment_panel_box=_ResponseBox(x=22, y=8, width=56, height=90),
            print_box=_ResponseBox(x=40, y=30, width=21, height=24),
            analysis_confidence=92.4,
        )

        detected = normalize_detected_mockup(raw)

        self.assertEqual(detected.garment_type, "t-shirt")
        self.assertEqual(detected.side, "front")
        self.assertEqual(detected.target_gender, "unisex")
        self.assertEqual(detected.target_age_group, "adult-universal")
        self.assertEqual(detected.moods, ["playful", "youth"])
        self.assertEqual(detected.analysis_confidence, 92)

    def test_zero_to_one_confidence_is_converted_to_percent(self) -> None:
        raw = _DetectedMockupResponse(
            side="front",
            garment_type="t-shirt",
            shirt_color="charcoal gray",
            fabric_finish="acid wash",
            fit="oversized",
            target_gender="unisex",
            target_age_group="adult-universal",
            moods=["playful"],
            print_theme="cat graphic",
            construction_details="crew neck and drop shoulder",
            garment_panel_box=_ResponseBox(x=22, y=7, width=56, height=90),
            print_box=_ResponseBox(x=41, y=30, width=19, height=24),
            analysis_confidence=0.96,
        )

        detected = normalize_detected_mockup(raw)

        self.assertEqual(detected.analysis_confidence, 96)

    def test_mixed_percent_and_pixel_boxes_are_normalized_separately(self) -> None:
        raw = _DetectedMockupResponse(
            side="front",
            garment_type="t-shirt",
            shirt_color="charcoal gray",
            fabric_finish="acid wash",
            fit="oversized",
            target_gender="unisex",
            target_age_group="adult-universal",
            moods=["playful"],
            print_theme="cat wearing a traditional hat",
            construction_details="crew neck and drop shoulder",
            garment_panel_box=_ResponseBox(x=22, y=7, width=56, height=90),
            print_box=_ResponseBox(x=410, y=230, width=190, height=180),
            analysis_confidence=1.0,
        )

        detected = normalize_detected_mockup(
            raw,
            image_width=997,
            image_height=767,
        )
        spec = build_mockup_spec(
            detected,
            image_width=997,
            image_height=767,
        )

        self.assertEqual(detected.analysis_confidence, 100)
        self.assertTrue(30 <= spec.print_width_percent <= 36)
        self.assertTrue(24 <= spec.print_height_percent <= 28)
        self.assertTrue(48 <= spec.print_center_x_percent <= 53)
        self.assertTrue(spec.geometry_validated)

    def test_impossible_geometry_is_rejected_instead_of_clipped(self) -> None:
        raw = _DetectedMockupResponse(
            side="front",
            garment_type="t-shirt",
            shirt_color="gray",
            fabric_finish="cotton",
            fit="oversized",
            target_gender="unisex",
            target_age_group="adult-universal",
            moods=["playful"],
            print_theme="cat graphic",
            construction_details="crew neck",
            garment_panel_box=_ResponseBox(x=20, y=5, width=60, height=90),
            print_box=_ResponseBox(x=95, y=80, width=20, height=30),
            analysis_confidence=95,
        )

        with self.assertRaises(MockupGeometryError):
            normalize_detected_mockup(
                raw,
                image_width=997,
                image_height=767,
            )

    def test_old_unvalidated_analysis_cannot_start_generation(self) -> None:
        spec = MockupSpec(
            side="front",
            garment_type="t-shirt",
            shirt_color="charcoal gray",
            fabric_finish="acid wash",
            fit="oversized",
            print_width_percent=5,
            print_height_percent=3,
            print_top_offset_percent=80,
            print_left_offset_percent=95,
            print_center_x_percent=98,
            target_gender="unisex",
            target_age_group="adult-universal",
            moods=["playful"],
            print_theme="cat graphic",
            construction_details="crew neck and drop shoulder",
            analysis_confidence=1,
        )

        with self.assertRaises(MockupGeometryError):
            ensure_mockup_spec_ready(spec)

    def test_structured_parsed_response_works_without_response_text(self) -> None:
        raw = _DetectedMockupResponse(
            side="front",
            garment_type="t-shirt",
            shirt_color="washed dark gray",
            fabric_finish="washed cotton jersey",
            fit="relaxed",
            target_gender="unisex",
            target_age_group="18-24",
            moods=["playful", "youth"],
            print_theme="cartoon cat",
            construction_details="ribbed crew neck and short sleeves",
            garment_panel_box=_ResponseBox(x=22, y=8, width=56, height=90),
            print_box=_ResponseBox(x=40, y=30, width=21, height=24),
            analysis_confidence=94,
        )

        class FakeModels:
            def generate_content(self, **kwargs):
                return SimpleNamespace(parsed=raw, text="")

        generator = object.__new__(MockupGenerator)
        generator.client = SimpleNamespace(models=FakeModels())
        generator.analysis_model = "analysis-model"

        spec = generator._analyze_mockup_sync(
            b"normalized-jpeg",
            "image/jpeg",
            997,
            767,
        )

        self.assertEqual(spec.garment_type, "t-shirt")
        self.assertEqual(spec.print_width_percent, 38)
        self.assertEqual(spec.print_center_x_percent, 51)

    def test_analysis_rate_limit_message_does_not_blame_image_quality(self) -> None:
        class FakeRateLimit(Exception):
            code = 429

        error = MockupGenerator._friendly_analysis_error(
            FakeRateLimit("RESOURCE_EXHAUSTED")
        )

        self.assertIsInstance(error, MockupAnalysisError)
        self.assertIn("лимита анализа", error.user_message)
        self.assertNotIn("четк", error.user_message.casefold())

    def test_algorithm_derives_relative_print_placement_from_boxes(self) -> None:
        detected = _DetectedMockup(
            side="front",
            garment_type="t-shirt",
            shirt_color="beige",
            fabric_finish="cotton jersey",
            fit="relaxed",
            target_gender="women",
            target_age_group="18-24",
            moods=["playful", "youth"],
            print_theme="illustrated character",
            construction_details="crew neck and short sleeves",
            garment_panel_box=NormalizedBox(x=10, y=20, width=80, height=60),
            print_box=NormalizedBox(x=42, y=26, width=16, height=18),
            analysis_confidence=91,
        )

        spec = build_mockup_spec(
            detected,
            image_width=2000,
            image_height=2000,
        )

        self.assertEqual(spec.print_width_percent, 20)
        self.assertEqual(spec.print_height_percent, 30)
        self.assertEqual(spec.print_left_offset_percent, 40)
        self.assertEqual(spec.print_top_offset_percent, 10)
        self.assertEqual(spec.print_center_x_percent, 50)
        self.assertEqual(spec.source_image_width_px, 2000)

    def test_print_png_transparency_and_visible_bounds_are_measured(self) -> None:
        image = Image.new("RGBA", (100, 50), (0, 0, 0, 0))
        image.paste((255, 0, 0, 255), (10, 5, 90, 45))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        info = inspect_print_file(buffer.getvalue())

        self.assertTrue(info["has_transparency"])
        self.assertEqual(info["content_x_px"], 10)
        self.assertEqual(info["content_y_px"], 5)
        self.assertEqual(info["content_width_px"], 80)
        self.assertEqual(info["content_height_px"], 40)

    def test_prompt_distinguishes_placement_reference_and_exact_png(self) -> None:
        direction = choose_photo_directions(1, random.Random(18))[0]
        prompt = build_model_photo_prompt(
            None,
            direction,
            "two-sources",
            has_separate_print=True,
        )
        self.assertIn("Two source images are supplied", prompt)
        self.assertIn("second image is the exact high-quality print source", prompt)

    def test_prompt_requires_manual_reference_and_close_framing(self) -> None:
        direction = choose_photo_directions(1, random.Random(19))[0]
        prompt = build_model_photo_prompt(
            None,
            direction,
            "manual-reference",
            has_style_reference=True,
            style_reference_tags={
                "framing": "waist-up",
                "camera_angle": "front",
                "setting": "plain wall",
            },
        )
        self.assertIn("manually selected photographic reference", prompt)
        self.assertIn("No crowd", prompt)
        self.assertIn("75 to 90 percent", prompt)
        self.assertIn("waist-up", prompt)

    def test_batch_uses_distinct_people_and_scenes(self) -> None:
        directions = choose_photo_directions(4, random.Random(42))
        self.assertEqual(len(directions), 4)
        self.assertEqual(len({item.label for item in directions}), 4)
        self.assertEqual(len({item.person for item in directions}), 4)
        self.assertEqual(len({item.seed for item in directions}), 4)

    def test_prompt_locks_print_scale_and_safe_area(self) -> None:
        direction = choose_photo_directions(
            1,
            random.Random(7),
            target_gender="women",
        )[0]
        spec = MockupSpec(
            side="front",
            garment_type="t-shirt",
            shirt_color="washed dark gray",
            fabric_finish="acid washed cotton",
            fit="relaxed",
            print_width_percent=48,
            print_height_percent=27,
            print_top_offset_percent=18,
            target_gender="women",
            construction_details="ribbed crew neck and dropped shoulders",
        )
        prompt = build_model_photo_prompt(spec, direction, "batch123")
        self.assertIn("front", prompt)
        self.assertIn("48%", prompt)
        self.assertIn("27%", prompt)
        self.assertIn("18%", prompt)
        self.assertIn("Do not redraw", prompt)
        self.assertIn("Vertical 4:5", prompt)
        self.assertIn("at least 8%", prompt)
        self.assertIn("Action, not pose", prompt)
        self.assertIn("Never invent a rectangular backing", prompt)
        self.assertIn("different fictional non-celebrity adult", prompt)
        self.assertIn("DTF heat-transfer layer", prompt)
        self.assertIn("normal 24-35 mm equivalent phone lens", prompt)
        self.assertIn("intended wearer is women", prompt)

    def test_direction_library_contains_sitting_and_moving_scenes(self) -> None:
        directions = choose_photo_directions(10, random.Random(13))
        kinds = {item.pose_kind for item in directions}
        self.assertIn("sitting", kinds)
        self.assertTrue({"walking", "activity"}.intersection(kinds))

    def test_cap_prompt_preserves_seam_and_dtf_physics(self) -> None:
        direction = choose_photo_directions(
            1,
            random.Random(8),
            target_gender="women",
            garment_type="cap",
        )[0]
        spec = MockupSpec(
            side="front",
            garment_type="cap",
            shirt_color="washed charcoal",
            fabric_finish="washed cotton twill",
            fit="soft unstructured crown",
            print_width_percent=52,
            print_height_percent=28,
            print_top_offset_percent=22,
            target_gender="women",
            construction_details="six panels, center seam and stitched curved brim",
        )
        prompt = build_model_photo_prompt(spec, direction, "capbatch")
        self.assertIn("CAP-SPECIFIC DTF PHYSICS", prompt)
        self.assertIn("center vertical panel seam", prompt)
        self.assertIn("not embroidery", prompt)
        self.assertIn("rows of stitching on the brim", prompt)
        self.assertIn("52%", prompt)

    def test_feminine_print_selects_only_women(self) -> None:
        directions = choose_photo_directions(
            4,
            random.Random(21),
            target_gender="women",
        )
        self.assertTrue(all(item.gender == "women" for item in directions))

    def test_used_photo_direction_is_avoided(self) -> None:
        first = choose_photo_directions(
            1,
            random.Random(4),
            target_gender="women",
        )[0]
        second = choose_photo_directions(
            1,
            random.Random(4),
            target_gender="women",
            exclude_labels=[first.label],
        )[0]
        self.assertNotEqual(first.label, second.label)

    def test_billing_error_is_explained(self) -> None:
        error = MockupGenerator._friendly_error(
            RuntimeError("FAILED_PRECONDITION: billing is required")
        )
        self.assertIn("платный тариф", error.user_message)

    def test_generation_requests_4_by_5_image(self) -> None:
        class FakeModels:
            kwargs = None

            def generate_content(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            inline_data=SimpleNamespace(
                                data=b"jpeg-data",
                                mime_type="image/jpeg",
                            )
                        )
                    ]
                )

        generator = object.__new__(MockupGenerator)
        generator.client = SimpleNamespace(models=FakeModels())
        generator.image_model = "gemini-3.1-flash-image"
        generator.image_size = "1K"
        generator.aspect_ratio = "4:5"
        direction = choose_photo_directions(1, random.Random(11))[0]

        result = generator._generate_variant_sync(
            b"source-image",
            "image/jpeg",
            None,
            direction,
            "batch",
        )

        config = generator.client.models.kwargs["config"]
        dumped = config.model_dump(by_alias=True, exclude_none=True)
        self.assertEqual(result.data, b"jpeg-data")
        self.assertEqual(dumped["imageConfig"]["aspectRatio"], "4:5")
        self.assertEqual(dumped["imageConfig"]["imageSize"], "1K")
        self.assertEqual(dumped["responseModalities"], ["IMAGE"])
        self.assertNotIn("outputMimeType", dumped["imageConfig"])
        self.assertNotIn("outputCompressionQuality", dumped["imageConfig"])
        self.assertNotIn("seed", dumped)

    def test_generation_sends_exact_png_as_second_source(self) -> None:
        class FakeModels:
            kwargs = None

            def generate_content(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            inline_data=SimpleNamespace(
                                data=b"jpeg-data",
                                mime_type="image/jpeg",
                            )
                        )
                    ]
                )

        generator = object.__new__(MockupGenerator)
        generator.client = SimpleNamespace(models=FakeModels())
        generator.image_model = "gemini-3.1-flash-image"
        generator.image_size = "1K"
        generator.aspect_ratio = "4:5"
        direction = choose_photo_directions(1, random.Random(12))[0]

        generator._generate_variant_sync(
            b"garment-image",
            "image/jpeg",
            None,
            direction,
            "batch",
            b"exact-print",
            "image/png",
        )

        contents = generator.client.models.kwargs["contents"]
        self.assertEqual(len(contents), 4)
        self.assertIn("Two source images are supplied", contents[0])
        self.assertEqual(contents[3].inline_data.mime_type, "image/png")

    def test_generation_sends_manual_reference_as_final_source(self) -> None:
        class FakeModels:
            kwargs = None

            def generate_content(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            inline_data=SimpleNamespace(
                                data=b"jpeg-data",
                                mime_type="image/jpeg",
                            )
                        )
                    ]
                )

        generator = object.__new__(MockupGenerator)
        generator.client = SimpleNamespace(models=FakeModels())
        generator.image_model = "gemini-3.1-flash-image"
        generator.image_size = "1K"
        generator.aspect_ratio = "4:5"
        direction = choose_photo_directions(1, random.Random(14))[0]

        generator._generate_variant_sync(
            b"garment-image",
            "image/jpeg",
            None,
            direction,
            "batch",
            reference_image_bytes=b"manual-reference",
            reference_mime_type="image/jpeg",
            reference_tags={"framing": "waist-up"},
        )

        contents = generator.client.models.kwargs["contents"]
        self.assertEqual(len(contents), 4)
        self.assertIn("manually selected photographic reference", contents[0])
        self.assertEqual(contents[-1].inline_data.data, b"manual-reference")
        self.assertEqual(contents[-1].inline_data.mime_type, "image/jpeg")

    def test_invalid_argument_error_is_explained(self) -> None:
        class FakeInvalidArgument(Exception):
            code = 400

        error = FakeInvalidArgument("INVALID_ARGUMENT")
        friendly = MockupGenerator._friendly_error(error)
        self.assertIn("ошибка 400", friendly.user_message)


class MockupAnalysisRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_geometry_is_remeasured_automatically(self) -> None:
        invalid = _DetectedMockupResponse(
            side="front",
            garment_type="t-shirt",
            shirt_color="charcoal gray",
            fabric_finish="acid wash",
            fit="oversized",
            target_gender="unisex",
            target_age_group="adult-universal",
            moods=["playful"],
            print_theme="cat graphic",
            construction_details="crew neck and drop shoulder",
            garment_panel_box=_ResponseBox(x=20, y=5, width=60, height=90),
            print_box=_ResponseBox(x=95, y=80, width=20, height=30),
            analysis_confidence=1.0,
        )
        valid = invalid.model_copy(
            update={
                "print_box": _ResponseBox(x=40, y=28, width=20, height=24),
                "analysis_confidence": 0.96,
            }
        )

        class FakeModels:
            def __init__(self):
                self.calls = 0
                self.prompts: list[str] = []

            def generate_content(self, **kwargs):
                self.prompts.append(kwargs["contents"][0])
                response = invalid if self.calls == 0 else valid
                self.calls += 1
                return SimpleNamespace(parsed=response, text="")

        image = Image.new("RGB", (997, 767), (70, 70, 74))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        models = FakeModels()
        generator = object.__new__(MockupGenerator)
        generator.client = SimpleNamespace(models=models)
        generator.analysis_model = "analysis-model"
        generator.analysis_coordinator = None

        spec = await generator.analyze_mockup(buffer.getvalue(), "image/jpeg")

        self.assertEqual(models.calls, 2)
        self.assertTrue(spec.geometry_validated)
        self.assertEqual(spec.analysis_confidence, 96)
        self.assertIn("previous measurement", models.prompts[1].casefold())


class AnalysisCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_interactive_analysis_runs_before_next_background_job(self) -> None:
        coordinator = AnalysisCoordinator()
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        order: list[str] = []

        async def first_background() -> None:
            async with coordinator.background():
                order.append("background-1")
                first_started.set()
                await release_first.wait()

        async def interactive() -> None:
            async with coordinator.interactive():
                order.append("interactive")

        async def second_background() -> None:
            async with coordinator.background():
                order.append("background-2")

        first_task = asyncio.create_task(first_background())
        await first_started.wait()
        interactive_task = asyncio.create_task(interactive())
        await asyncio.sleep(0)
        second_task = asyncio.create_task(second_background())
        release_first.set()
        await asyncio.gather(first_task, interactive_task, second_task)

        self.assertEqual(
            order,
            ["background-1", "interactive", "background-2"],
        )


class ConfigTests(unittest.TestCase):
    def test_old_gemini_model_is_upgraded(self) -> None:
        self.assertEqual(
            normalize_gemini_model("gemini-2.5-flash"),
            DEFAULT_GEMINI_MODEL,
        )

    def test_custom_gemini_model_is_preserved(self) -> None:
        self.assertEqual(
            normalize_gemini_model("gemini-3.5-flash"),
            "gemini-3.5-flash",
        )

    def test_default_image_model_and_valid_size(self) -> None:
        self.assertEqual(
            DEFAULT_GEMINI_IMAGE_MODEL,
            "gemini-3.1-flash-lite-image",
        )
        self.assertEqual(normalize_gemini_image_size(" 2k "), "2K")

    def test_invalid_image_size_is_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            normalize_gemini_image_size("HD")

    def test_persistent_template_is_seeded_from_bundled_template(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = Config(
                telegram_bot_token="token",
                gemini_api_key="key",
                admin_telegram_id=123,
                channel_id="@channel",
                contact_username="contact",
                timezone_name="Asia/Ashgabat",
                gemini_model=DEFAULT_GEMINI_MODEL,
                button_text="Написать",
                copy_language="ru",
                database_path=root / "data" / "posts.sqlite3",
                caption_template_path=root / "data" / "caption_template.txt",
            )

            config.ensure_runtime_paths()

            self.assertTrue(config.caption_template_path.is_file())
            self.assertIn(
                "{Название принта}",
                config.caption_template_path.read_text(encoding="utf-8"),
            )


class HealthTests(unittest.TestCase):
    def test_render_health_routes_exist(self) -> None:
        paths = {
            route.resource.canonical for route in build_health_app().router.routes()
        }
        self.assertTrue({"/", "/health"}.issubset(paths))


class SchedulingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.timezone = ZoneInfo("Asia/Ashgabat")
        self.now = datetime(2026, 7, 21, 10, 0, tzinfo=self.timezone)

    def test_full_date(self) -> None:
        result = parse_local_datetime("25.07.2026 18:30", self.timezone, now=self.now)
        self.assertEqual(
            result.astimezone(self.timezone).strftime("%d.%m.%Y %H:%M"),
            "25.07.2026 18:30",
        )

    def test_time_only_uses_today_when_future(self) -> None:
        result = parse_local_datetime("18:30", self.timezone, now=self.now)
        self.assertEqual(
            result.astimezone(self.timezone).strftime("%d.%m.%Y %H:%M"),
            "21.07.2026 18:30",
        )

    def test_past_time_uses_tomorrow(self) -> None:
        result = parse_local_datetime("09:30", self.timezone, now=self.now)
        self.assertEqual(
            result.astimezone(self.timezone).strftime("%d.%m.%Y %H:%M"),
            "22.07.2026 09:30",
        )


class ReferenceCatalogTests(unittest.TestCase):
    def test_pinterest_links_are_canonicalized_and_deduplicated(self) -> None:
        urls = normalize_reference_urls(
            """
            https://ru.pinterest.com/pin/12345/?utm_source=test
            https://www.pinterest.com/pin/12345/
            https://pin.it/AbCdEf
            https://example.com/not-accepted
            """
        )
        self.assertEqual(
            urls,
            [
                "https://www.pinterest.com/pin/12345/",
                "https://pin.it/AbCdEf",
            ],
        )

    def test_catalog_selects_and_reserves_best_matching_reference(self) -> None:
        tags = {
            "garment_types": ["t-shirt", "sweatshirt"],
            "gender": "women",
            "moods": ["calm", "playful"],
            "pose_kind": "sitting",
            "action": "reading",
            "location_category": "home",
            "setting": "ordinary living room",
            "camera_angle": "three-quarter",
            "framing": "three-quarter",
            "lighting": "daylight",
            "season": "all-season",
            "print_side_visible": "front",
            "print_area_visibility": 92,
            "composition_notes": "relaxed seated phone photo",
            "usable": True,
            "unusable_reason": "",
        }
        with tempfile.TemporaryDirectory() as directory:
            repository = PostRepository(Path(directory) / "posts.sqlite3")
            repository.initialize()
            repository.enqueue_reference_urls(
                ["https://www.pinterest.com/pin/12345/"],
                source_name="test",
            )
            job = repository.claim_reference_import()
            repository.store_reference_image(
                job.id,
                pin_id="12345",
                resolved_image_url="https://i.pinimg.com/originals/test.jpg",
                image_bytes=b"reference-image",
                image_mime_type="image/jpeg",
                thumbnail_bytes=b"thumbnail",
                width=1200,
                height=1600,
                image_sha256="abc",
            )
            repository.mark_reference_ready(job.id, tags=tags)

            catalog = object.__new__(ReferenceCatalog)
            catalog.repository = repository
            catalog.min_pool_size = 20
            selected = catalog.select_reference(
                garment_type="t-shirt",
                target_gender="women",
                moods=["calm"],
                request_token="request-1",
                rng=random.Random(10),
            )

            self.assertIsNotNone(selected)
            self.assertEqual(selected.id, job.id)
            self.assertEqual(repository.reference_stats()["ready"], 1)
            self.assertEqual(repository.list_ready_reference_assets(), [])
            repository.finish_reference_usage("request-1", outcome="completed")

    def test_status_separates_queue_processing_and_delayed_retries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = PostRepository(Path(directory) / "posts.sqlite3")
            repository.initialize()
            repository.enqueue_reference_urls(
                [
                    "https://www.pinterest.com/pin/10001/",
                    "https://www.pinterest.com/pin/10002/",
                    "https://www.pinterest.com/pin/10003/",
                ],
                source_name="test",
            )
            processing = repository.claim_reference_import()
            retry_job = repository.claim_reference_import()
            repository.mark_reference_import_error(
                retry_job.id,
                error="Pinterest временно ограничил частоту запросов",
                retry_at_utc=datetime.now(timezone.utc) + timedelta(hours=1),
                max_attempts=5,
            )

            catalog = object.__new__(ReferenceCatalog)
            catalog.repository = repository
            catalog.min_pool_size = 20
            status = catalog.status_text()

            self.assertIsNotNone(processing)
            self.assertIn("В очереди: 1", status)
            self.assertIn("Сейчас обрабатывается: 1", status)
            self.assertIn("Ждут повторной попытки: 1", status)
            self.assertIn("временное ограничение Pinterest: 1", status)

    def test_resume_now_releases_delayed_and_stale_reference_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = PostRepository(Path(directory) / "posts.sqlite3")
            repository.initialize()
            repository.enqueue_reference_urls(
                [
                    "https://www.pinterest.com/pin/20001/",
                    "https://www.pinterest.com/pin/20002/",
                ],
                source_name="test",
            )
            retry_job = repository.claim_reference_import()
            repository.mark_reference_import_error(
                retry_job.id,
                error="temporary",
                retry_at_utc=datetime.now(timezone.utc) + timedelta(hours=8),
                max_attempts=5,
            )
            stale_job = repository.claim_reference_import()
            stale_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
            with repository._connect() as connection:
                repository._execute(
                    connection,
                    "UPDATE reference_assets SET updated_at_utc = ? WHERE id = ?",
                    (stale_time, stale_job.id),
                )

            counts = repository.resume_reference_imports(
                stale_after=timedelta(minutes=10)
            )

            self.assertEqual(counts["retry"], 1)
            self.assertEqual(counts["stale"], 1)
            self.assertEqual(repository.reference_stats().get("retry"), 2)


class StorageTests(unittest.TestCase):
    def test_recent_mockup_directions_persist_without_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = PostRepository(Path(directory) / "posts.sqlite3")
            repository.initialize()
            repository.remember_mockup_direction("На ступенях", limit=3)
            repository.remember_mockup_direction("В лифте с кофе", limit=3)
            repository.remember_mockup_direction("На ступенях", limit=3)

            self.assertEqual(
                repository.get_recent_mockup_directions(limit=3),
                ["В лифте с кофе", "На ступенях"],
            )

    def test_old_database_gets_size_column(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "posts.sqlite3"
            with sqlite3.connect(path) as connection:
                connection.execute(
                    """
                    CREATE TABLE scheduled_posts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        author_id INTEGER NOT NULL,
                        photo_file_id TEXT NOT NULL,
                        title TEXT NOT NULL,
                        description TEXT NOT NULL,
                        price TEXT NOT NULL,
                        scheduled_at_utc TEXT NOT NULL,
                        next_attempt_at_utc TEXT NOT NULL,
                        status TEXT NOT NULL,
                        attempts INTEGER NOT NULL DEFAULT 0,
                        last_error TEXT,
                        published_message_id INTEGER,
                        created_at_utc TEXT NOT NULL
                    )
                    """
                )

            repository = PostRepository(path)
            repository.initialize()
            with sqlite3.connect(path) as connection:
                columns = {
                    row[1]
                    for row in connection.execute("PRAGMA table_info(scheduled_posts)")
                }
            self.assertTrue(
                {"size", "garment_type", "design_name", "theme_hashtag"}.issubset(
                    columns
                )
            )

    def test_post_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = PostRepository(Path(directory) / "posts.sqlite3")
            repository.initialize()
            scheduled_at = datetime.now(timezone.utc) - timedelta(seconds=1)
            post_id = repository.create(
                author_id=123,
                photo_file_id="photo-id",
                title="Название",
                description="Описание",
                size="S-XXL",
                price="240 TMT",
                scheduled_at_utc=scheduled_at,
            )

            self.assertEqual(repository.due_ids(datetime.now(timezone.utc)), [post_id])
            self.assertTrue(repository.claim_for_publish(post_id))
            self.assertFalse(repository.claim_for_publish(post_id))
            repository.mark_published(post_id, 777)

            post = repository.get(post_id)
            self.assertIsNotNone(post)
            self.assertEqual(post.status, "published")
            self.assertEqual(post.published_message_id, 777)

    def test_cancel_pending_post(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = PostRepository(Path(directory) / "posts.sqlite3")
            repository.initialize()
            scheduled_at = datetime.now(timezone.utc) + timedelta(hours=1)
            post_id = repository.create(
                author_id=123,
                photo_file_id="photo-id",
                title="Название",
                description="Описание",
                size="XS-3XL",
                price="240 TMT",
                scheduled_at_utc=scheduled_at,
            )

            self.assertTrue(repository.cancel(post_id))
            self.assertFalse(repository.cancel(post_id))

    def test_template_is_persisted_in_database(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template_path = root / "caption.txt"
            template_path.write_text(
                "{title}\n{size}\n{price}\n{hashtags}", encoding="utf-8"
            )
            repository = PostRepository(root / "posts.sqlite3")
            repository.initialize()
            store = CaptionTemplateStore(
                repository=repository,
                fallback_path=template_path,
            )
            store.initialize()
            store.set("{title}\nРазмер: {size}\n{price}\n{hashtags}")

            second_store = CaptionTemplateStore(
                repository=repository,
                fallback_path=template_path,
            )
            second_store.initialize()
            self.assertIn("Размер:", second_store.get())

    def test_presets_can_be_seeded_added_and_removed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = PostRepository(Path(directory) / "posts.sqlite3")
            repository.initialize()
            repository.seed_presets((("Футболка", "S-XXL", "250 манат"),))
            repository.seed_presets((("Не добавлять", "M", "1 манат"),))
            presets = repository.list_presets()
            self.assertEqual([item.name for item in presets], ["Футболка"])

            preset_id = repository.create_preset(
                name="Худи", size="S-2XL", price="460 манат"
            )
            self.assertEqual(repository.get_preset(preset_id).price, "460 манат")
            self.assertTrue(repository.delete_preset(preset_id))
            self.assertIsNone(repository.get_preset(preset_id))
            restored_id = repository.create_preset(
                name="Худи", size="M-2XL", price="510 манат"
            )
            self.assertEqual(restored_id, preset_id)
            self.assertEqual(repository.get_preset(preset_id).price, "510 манат")

    def test_active_draft_survives_repository_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "posts.sqlite3"
            first = PostRepository(path)
            first.initialize()
            first.save_active_draft(
                123,
                {
                    "photo_file_id": "telegram-file",
                    "title": "Футболка Тест",
                    "description": "Описание",
                },
            )
            first.close()

            second = PostRepository(path)
            second.initialize()
            draft = second.get_active_draft(123)
            self.assertIsNotNone(draft)
            self.assertEqual(draft["photo_file_id"], "telegram-file")
            second.clear_active_draft(123)
            self.assertIsNone(second.get_active_draft(123))

    def test_model_analysis_draft_survives_repository_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "posts.sqlite3"
            first = PostRepository(path)
            first.initialize()
            first.save_model_draft(
                321,
                {
                    "model_source_file_id": "garment-file",
                    "model_print_file_id": "print-file",
                    "model_mockup_spec": {"garment_type": "t-shirt"},
                },
            )
            first.close()

            second = PostRepository(path)
            second.initialize()
            draft = second.get_model_draft(321)
            self.assertIsNotNone(draft)
            self.assertEqual(draft["model_source_file_id"], "garment-file")
            self.assertEqual(draft["model_print_file_id"], "print-file")
            second.clear_model_draft(321)
            self.assertIsNone(second.get_model_draft(321))

    def test_unfinished_model_input_survives_repository_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "posts.sqlite3"
            first = PostRepository(path)
            first.initialize()
            first.save_model_draft(
                654,
                {
                    "model_source_file_id": "garment-before-analysis",
                    "model_source_mime_type": "image/jpeg",
                    "model_mockup_spec": None,
                },
            )
            first.close()

            second = PostRepository(path)
            second.initialize()
            draft = second.get_model_draft(654)

            self.assertIsNotNone(draft)
            self.assertEqual(
                draft["model_source_file_id"],
                "garment-before-analysis",
            )
            self.assertIsNone(draft["model_mockup_spec"])

    def test_pending_post_can_be_edited_and_rescheduled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = PostRepository(Path(directory) / "posts.sqlite3")
            repository.initialize()
            original_time = datetime.now(timezone.utc) + timedelta(hours=1)
            post_id = repository.create(
                author_id=123,
                photo_file_id="photo-id",
                title="Старое название",
                description="Описание",
                size="S-XXL",
                price="250 манат",
                scheduled_at_utc=original_time,
            )
            self.assertTrue(
                repository.update_pending(
                    post_id,
                    title="Новое название",
                    price="290 манат",
                )
            )
            new_time = original_time + timedelta(days=1)
            self.assertTrue(repository.reschedule(post_id, new_time))
            post = repository.get(post_id)
            self.assertEqual(post.title, "Новое название")
            self.assertEqual(post.price, "290 манат")
            self.assertEqual(post.scheduled_at_utc, new_time)


if __name__ == "__main__":
    unittest.main()
