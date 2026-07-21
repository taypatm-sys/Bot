import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.models import ScheduledPost


UTC = timezone.utc


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("datetime must include timezone")
    return value.astimezone(UTC).isoformat()


def _from_iso(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)


def _row_to_post(row: sqlite3.Row) -> ScheduledPost:
    return ScheduledPost(
        id=row["id"],
        author_id=row["author_id"],
        photo_file_id=row["photo_file_id"],
        title=row["title"],
        description=row["description"],
        garment_type=row["garment_type"],
        design_name=row["design_name"],
        theme_hashtag=row["theme_hashtag"],
        size=row["size"],
        price=row["price"],
        scheduled_at_utc=_from_iso(row["scheduled_at_utc"]),
        next_attempt_at_utc=_from_iso(row["next_attempt_at_utc"]),
        status=row["status"],
        attempts=row["attempts"],
        last_error=row["last_error"],
        published_message_id=row["published_message_id"],
        created_at_utc=_from_iso(row["created_at_utc"]),
    )


class PostRepository:
    def __init__(self, path: Path):
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS scheduled_posts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    author_id INTEGER NOT NULL,
                    photo_file_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    garment_type TEXT NOT NULL DEFAULT '',
                    design_name TEXT NOT NULL DEFAULT '',
                    theme_hashtag TEXT NOT NULL DEFAULT '',
                    size TEXT NOT NULL DEFAULT 'S-XXL',
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
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(scheduled_posts)")
            }
            if "size" not in columns:
                connection.execute(
                    "ALTER TABLE scheduled_posts "
                    "ADD COLUMN size TEXT NOT NULL DEFAULT 'S-XXL'"
                )
            for name in ("garment_type", "design_name", "theme_hashtag"):
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE scheduled_posts "
                        f"ADD COLUMN {name} TEXT NOT NULL DEFAULT ''"
                    )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_scheduled_due
                ON scheduled_posts(status, next_attempt_at_utc)
                """
            )

    def recover_interrupted_posts(self) -> None:
        now = _iso(datetime.now(UTC))
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE scheduled_posts
                SET status = 'scheduled', next_attempt_at_utc = ?
                WHERE status = 'publishing'
                """,
                (now,),
            )

    def create(
        self,
        *,
        author_id: int,
        photo_file_id: str,
        title: str,
        description: str,
        size: str,
        price: str,
        scheduled_at_utc: datetime,
        garment_type: str = "",
        design_name: str = "",
        theme_hashtag: str = "",
    ) -> int:
        now = datetime.now(UTC)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO scheduled_posts (
                    author_id, photo_file_id, title, description,
                    garment_type, design_name, theme_hashtag, size, price,
                    scheduled_at_utc, next_attempt_at_utc, status, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'scheduled', ?)
                """,
                (
                    author_id,
                    photo_file_id,
                    title,
                    description,
                    garment_type,
                    design_name,
                    theme_hashtag,
                    size,
                    price,
                    _iso(scheduled_at_utc),
                    _iso(scheduled_at_utc),
                    _iso(now),
                ),
            )
            return int(cursor.lastrowid)

    def get(self, post_id: int) -> Optional[ScheduledPost]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM scheduled_posts WHERE id = ?", (post_id,)
            ).fetchone()
        return _row_to_post(row) if row else None

    def list_pending(self, limit: int = 20) -> list[ScheduledPost]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM scheduled_posts
                WHERE status IN ('scheduled', 'failed')
                ORDER BY scheduled_at_utc ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [_row_to_post(row) for row in rows]

    def due_ids(self, now_utc: datetime, limit: int = 10) -> list[int]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id FROM scheduled_posts
                WHERE status = 'scheduled' AND next_attempt_at_utc <= ?
                ORDER BY next_attempt_at_utc ASC
                LIMIT ?
                """,
                (_iso(now_utc), limit),
            ).fetchall()
        return [int(row["id"]) for row in rows]

    def claim_for_publish(self, post_id: int) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE scheduled_posts
                SET status = 'publishing'
                WHERE id = ? AND status = 'scheduled'
                """,
                (post_id,),
            )
            return cursor.rowcount == 1

    def mark_published(self, post_id: int, message_id: int) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE scheduled_posts
                SET status = 'published', published_message_id = ?, last_error = NULL
                WHERE id = ?
                """,
                (message_id, post_id),
            )

    def mark_publish_error(
        self,
        post_id: int,
        *,
        error: str,
        next_attempt_at_utc: datetime,
        max_attempts: int,
    ) -> str:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT attempts FROM scheduled_posts WHERE id = ?", (post_id,)
            ).fetchone()
            attempts = (int(row["attempts"]) if row else 0) + 1
            status = "failed" if attempts >= max_attempts else "scheduled"
            connection.execute(
                """
                UPDATE scheduled_posts
                SET status = ?, attempts = ?, last_error = ?, next_attempt_at_utc = ?
                WHERE id = ?
                """,
                (
                    status,
                    attempts,
                    error[:1000],
                    _iso(next_attempt_at_utc),
                    post_id,
                ),
            )
        return status

    def cancel(self, post_id: int) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE scheduled_posts
                SET status = 'cancelled'
                WHERE id = ? AND status IN ('scheduled', 'failed')
                """,
                (post_id,),
            )
            return cursor.rowcount == 1
