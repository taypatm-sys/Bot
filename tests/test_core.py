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
from app.copywriter import ProductCopy
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


class MockupGeneratorTests(unittest.TestCase):
    def test_batch_uses_distinct_people_and_scenes(self) -> None:
        directions = choose_photo_directions(4, random.Random(42))
        self.assertEqual(len(directions), 4)
        self.assertEqual(len({item.label for item in directions}), 4)
        self.assertEqual(len({item.person for item in directions}), 4)
        self.assertEqual(len({item.seed for item in directions}), 4)

    def test_prompt_locks_print_scale_and_safe_area(self) -> None:
        direction = choose_photo_directions(1, random.Random(7))[0]
        spec = MockupSpec(
            side="front",
            garment_type="t-shirt",
            shirt_color="washed dark gray",
            fabric_finish="acid washed cotton",
            fit="relaxed",
            print_width_percent=48,
            print_height_percent=27,
            print_top_from_collar_percent=18,
        )
        prompt = build_model_photo_prompt(spec, direction, "batch123")
        self.assertIn("front", prompt)
        self.assertIn("48%", prompt)
        self.assertIn("27%", prompt)
        self.assertIn("18%", prompt)
        self.assertIn("Do not redraw", prompt)
        self.assertIn("Vertical 4:5", prompt)
        self.assertIn("central 80%", prompt)
        self.assertIn("different, fictional, non-celebrity adult", prompt)

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
        self.assertEqual(DEFAULT_GEMINI_IMAGE_MODEL, "gemini-3.1-flash-image")
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


class StorageTests(unittest.TestCase):
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
