import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence, Union

from app.models import ProductPreset, ScheduledPost


UTC = timezone.utc
DatabaseSource = Union[Path, str]
POSTGRES_PREFIXES = ("postgres://", "postgresql://")


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("datetime must include timezone")
    return value.astimezone(UTC).isoformat()


def _from_iso(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _row_to_post(row: Mapping[str, Any]) -> ScheduledPost:
    return ScheduledPost(
        id=int(row["id"]),
        author_id=int(row["author_id"]),
        photo_file_id=str(row["photo_file_id"]),
        title=str(row["title"]),
        description=str(row["description"]),
        garment_type=str(row["garment_type"]),
        design_name=str(row["design_name"]),
        theme_hashtag=str(row["theme_hashtag"]),
        size=str(row["size"]),
        price=str(row["price"]),
        scheduled_at_utc=_from_iso(row["scheduled_at_utc"]),
        next_attempt_at_utc=_from_iso(row["next_attempt_at_utc"]),
        status=str(row["status"]),
        attempts=int(row["attempts"]),
        last_error=row["last_error"],
        published_message_id=(
            int(row["published_message_id"])
            if row["published_message_id"] is not None
            else None
        ),
        created_at_utc=_from_iso(row["created_at_utc"]),
    )


def _row_to_preset(row: Mapping[str, Any]) -> ProductPreset:
    return ProductPreset(
        id=int(row["id"]),
        name=str(row["name"]),
        size=str(row["size"]),
        price=str(row["price"]),
    )


class PostRepository:
    """Queue storage with PostgreSQL support and a local SQLite fallback."""

    def __init__(self, source: DatabaseSource):
        source_text = str(source)
        self.database_url = (
            source_text if source_text.startswith(POSTGRES_PREFIXES) else ""
        )
        self.path = None if self.database_url else Path(source)
        self._pool: Any = None

    @property
    def backend_name(self) -> str:
        return "PostgreSQL" if self.database_url else "SQLite"

    @property
    def is_persistent(self) -> bool:
        return bool(self.database_url)

    def _ensure_pool(self) -> None:
        if not self.database_url or self._pool is not None:
            return
        try:
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool
        except ImportError as error:
            raise RuntimeError(
                "Для DATABASE_URL установите зависимости из requirements.txt"
            ) from error

        self._pool = ConnectionPool(
            conninfo=self.database_url,
            min_size=0,
            max_size=3,
            open=False,
            kwargs={"row_factory": dict_row, "prepare_threshold": None},
            check=ConnectionPool.check_connection,
        )
        self._pool.open(wait=True, timeout=30)

    @contextmanager
    def _connect(self) -> Iterator[Any]:
        if self.database_url:
            self._ensure_pool()
            with self._pool.connection() as connection:
                yield connection
            return

        if self.path is None:
            raise RuntimeError("Путь к SQLite не указан")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path, timeout=10) as connection:
            connection.row_factory = sqlite3.Row
            yield connection

    def _sql(self, query: str) -> str:
        return query.replace("?", "%s") if self.database_url else query

    def _execute(
        self,
        connection: Any,
        query: str,
        params: Sequence[Any] = (),
    ) -> Any:
        return connection.execute(self._sql(query), tuple(params))

    def initialize(self) -> None:
        id_column = "BIGSERIAL PRIMARY KEY" if self.database_url else (
            "INTEGER PRIMARY KEY AUTOINCREMENT"
        )
        with self._connect() as connection:
            if not self.database_url:
                connection.execute("PRAGMA journal_mode=WAL")
            self._execute(
                connection,
                f"""
                CREATE TABLE IF NOT EXISTS scheduled_posts (
                    id {id_column},
                    author_id BIGINT NOT NULL,
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
                    published_message_id BIGINT,
                    created_at_utc TEXT NOT NULL
                )
                """,
            )
            self._migrate_post_columns(connection)
            self._execute(
                connection,
                """
                CREATE INDEX IF NOT EXISTS idx_scheduled_due
                ON scheduled_posts(status, next_attempt_at_utc)
                """,
            )
            self._execute(
                connection,
                """
                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL
                )
                """,
            )
            self._execute(
                connection,
                f"""
                CREATE TABLE IF NOT EXISTS product_presets (
                    id {id_column},
                    name TEXT NOT NULL UNIQUE,
                    size TEXT NOT NULL,
                    price TEXT NOT NULL,
                    position INTEGER NOT NULL DEFAULT 0,
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at_utc TEXT NOT NULL
                )
                """,
            )

    def _migrate_post_columns(self, connection: Any) -> None:
        definitions = {
            "size": "TEXT NOT NULL DEFAULT 'S-XXL'",
            "garment_type": "TEXT NOT NULL DEFAULT ''",
            "design_name": "TEXT NOT NULL DEFAULT ''",
            "theme_hashtag": "TEXT NOT NULL DEFAULT ''",
        }
        if self.database_url:
            for name, definition in definitions.items():
                self._execute(
                    connection,
                    f"ALTER TABLE scheduled_posts "
                    f"ADD COLUMN IF NOT EXISTS {name} {definition}",
                )
            return

        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(scheduled_posts)")
        }
        for name, definition in definitions.items():
            if name not in columns:
                self._execute(
                    connection,
                    f"ALTER TABLE scheduled_posts ADD COLUMN {name} {definition}",
                )

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    def get_setting(self, key: str) -> Optional[str]:
        with self._connect() as connection:
            row = self._execute(
                connection,
                "SELECT value FROM app_settings WHERE key = ?",
                (key,),
            ).fetchone()
        return str(row["value"]) if row else None

    def set_setting(self, key: str, value: str) -> None:
        now = _iso(datetime.now(UTC))
        with self._connect() as connection:
            self._execute(
                connection,
                """
                INSERT INTO app_settings(key, value, updated_at_utc)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at_utc = excluded.updated_at_utc
                """,
                (key, value, now),
            )

    def seed_setting(self, key: str, value: str) -> None:
        now = _iso(datetime.now(UTC))
        with self._connect() as connection:
            self._execute(
                connection,
                """
                INSERT INTO app_settings(key, value, updated_at_utc)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO NOTHING
                """,
                (key, value, now),
            )

    def seed_presets(self, presets: Sequence[tuple[str, str, str]]) -> None:
        if self.get_setting("presets_seeded") == "1":
            return
        now = _iso(datetime.now(UTC))
        with self._connect() as connection:
            for position, (name, size, price) in enumerate(presets, start=1):
                self._execute(
                    connection,
                    """
                    INSERT INTO product_presets(
                        name, size, price, position, active, created_at_utc
                    ) VALUES (?, ?, ?, ?, 1, ?)
                    ON CONFLICT(name) DO NOTHING
                    """,
                    (name, size, price, position, now),
                )
        self.set_setting("presets_seeded", "1")

    def list_presets(self) -> list[ProductPreset]:
        with self._connect() as connection:
            rows = self._execute(
                connection,
                """
                SELECT id, name, size, price
                FROM product_presets
                WHERE active = 1
                ORDER BY position ASC, id ASC
                """,
            ).fetchall()
        return [_row_to_preset(row) for row in rows]

    def get_preset(self, preset_id: int) -> Optional[ProductPreset]:
        with self._connect() as connection:
            row = self._execute(
                connection,
                """
                SELECT id, name, size, price
                FROM product_presets
                WHERE id = ? AND active = 1
                """,
                (preset_id,),
            ).fetchone()
        return _row_to_preset(row) if row else None

    def create_preset(self, *, name: str, size: str, price: str) -> int:
        now = _iso(datetime.now(UTC))
        with self._connect() as connection:
            position_row = self._execute(
                connection,
                "SELECT COALESCE(MAX(position), 0) + 1 AS value FROM product_presets",
            ).fetchone()
            position = int(position_row["value"])
            existing = self._execute(
                connection,
                "SELECT id FROM product_presets WHERE LOWER(name) = LOWER(?)",
                (name,),
            ).fetchone()
            if existing:
                preset_id = int(existing["id"])
                self._execute(
                    connection,
                    """
                    UPDATE product_presets
                    SET name = ?, size = ?, price = ?, position = ?, active = 1
                    WHERE id = ?
                    """,
                    (name, size, price, position, preset_id),
                )
                return preset_id
            if self.database_url:
                row = self._execute(
                    connection,
                    """
                    INSERT INTO product_presets(
                        name, size, price, position, active, created_at_utc
                    ) VALUES (?, ?, ?, ?, 1, ?)
                    RETURNING id
                    """,
                    (name, size, price, position, now),
                ).fetchone()
                return int(row["id"])
            cursor = self._execute(
                connection,
                """
                INSERT INTO product_presets(
                    name, size, price, position, active, created_at_utc
                ) VALUES (?, ?, ?, ?, 1, ?)
                """,
                (name, size, price, position, now),
            )
            return int(cursor.lastrowid)

    def delete_preset(self, preset_id: int) -> bool:
        with self._connect() as connection:
            cursor = self._execute(
                connection,
                "UPDATE product_presets SET active = 0 WHERE id = ? AND active = 1",
                (preset_id,),
            )
            return cursor.rowcount == 1

    def recover_interrupted_posts(self) -> None:
        now = _iso(datetime.now(UTC))
        with self._connect() as connection:
            self._execute(
                connection,
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
        params = (
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
        )
        with self._connect() as connection:
            query = """
                INSERT INTO scheduled_posts (
                    author_id, photo_file_id, title, description,
                    garment_type, design_name, theme_hashtag, size, price,
                    scheduled_at_utc, next_attempt_at_utc, status, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'scheduled', ?)
            """
            if self.database_url:
                row = self._execute(connection, query + " RETURNING id", params).fetchone()
                return int(row["id"])
            cursor = self._execute(connection, query, params)
            return int(cursor.lastrowid)

    def get(self, post_id: int) -> Optional[ScheduledPost]:
        with self._connect() as connection:
            row = self._execute(
                connection,
                "SELECT * FROM scheduled_posts WHERE id = ?",
                (post_id,),
            ).fetchone()
        return _row_to_post(row) if row else None

    def list_pending(self, limit: int = 20) -> list[ScheduledPost]:
        with self._connect() as connection:
            rows = self._execute(
                connection,
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
            rows = self._execute(
                connection,
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
            cursor = self._execute(
                connection,
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
            self._execute(
                connection,
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
            row = self._execute(
                connection,
                "SELECT attempts FROM scheduled_posts WHERE id = ?",
                (post_id,),
            ).fetchone()
            attempts = (int(row["attempts"]) if row else 0) + 1
            status = "failed" if attempts >= max_attempts else "scheduled"
            self._execute(
                connection,
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

    def update_pending(self, post_id: int, **fields: str) -> bool:
        allowed = {
            "title",
            "description",
            "garment_type",
            "design_name",
            "theme_hashtag",
            "size",
            "price",
        }
        updates = {name: value for name, value in fields.items() if name in allowed}
        if not updates:
            return False
        assignments = ", ".join(f"{name} = ?" for name in updates)
        params = (*updates.values(), post_id)
        with self._connect() as connection:
            cursor = self._execute(
                connection,
                f"""
                UPDATE scheduled_posts SET {assignments}
                WHERE id = ? AND status IN ('scheduled', 'failed')
                """,
                params,
            )
            return cursor.rowcount == 1

    def reschedule(self, post_id: int, scheduled_at_utc: datetime) -> bool:
        timestamp = _iso(scheduled_at_utc)
        with self._connect() as connection:
            cursor = self._execute(
                connection,
                """
                UPDATE scheduled_posts
                SET scheduled_at_utc = ?, next_attempt_at_utc = ?,
                    status = 'scheduled', attempts = 0, last_error = NULL
                WHERE id = ? AND status IN ('scheduled', 'failed')
                """,
                (timestamp, timestamp, post_id),
            )
            return cursor.rowcount == 1

    def cancel(self, post_id: int) -> bool:
        with self._connect() as connection:
            cursor = self._execute(
                connection,
                """
                UPDATE scheduled_posts
                SET status = 'cancelled'
                WHERE id = ? AND status IN ('scheduled', 'failed')
                """,
                (post_id,),
            )
            return cursor.rowcount == 1
