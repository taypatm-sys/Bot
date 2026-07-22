import sqlite3
import tempfile
import unittest
import random
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

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
    MockupGenerator,
    MockupSpec,
    build_model_photo_prompt,
    choose_photo_directions,
)
from app.reference_catalog import ReferenceCatalog, normalize_reference_urls
from app.scheduling import parse_local_datetime
from app.storage import PostRepository
from app.template_store import CaptionTemplateStore


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
                '{Короткое описание, передающее настроение принта}\n\n'
                '👕 {Размеры}\n\n💸 {Цена}\n\n'
                '#Taýpa #{тип товара} #{тематика принта}',
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
            design_name='“Soul of Karakum”',
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

    def test_invalid_argument_error_is_explained(self) -> None:
        class FakeInvalidArgument(Exception):
            code = 400

        error = FakeInvalidArgument("INVALID_ARGUMENT")
        friendly = MockupGenerator._friendly_error(error)
        self.assertIn("ошибка 400", friendly.user_message)


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
        paths = {route.resource.canonical for route in build_health_app().router.routes()}
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
                    row[1] for row in connection.execute(
                        "PRAGMA table_info(scheduled_posts)"
                    )
                }
            self.assertTrue(
                {"size", "garment_type", "design_name", "theme_hashtag"}
                .issubset(columns)
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
