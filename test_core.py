import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from app.config import DEFAULT_GEMINI_MODEL, normalize_gemini_model
from app.copywriter import ProductCopy
from app.formatting import (
    TemplateError,
    contact_link,
    normalize_price,
    normalize_size,
    render_caption,
    validate_template,
)
from app.scheduling import parse_local_datetime
from app.storage import PostRepository


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


if __name__ == "__main__":
    unittest.main()
