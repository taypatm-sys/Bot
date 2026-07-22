import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence, Union

from app.models import (
    ProductPreset,
    ReferenceAsset,
    ReferenceImportJob,
    ScheduledPost,
)


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


def _optional_datetime(value: Any) -> Optional[datetime]:
    return _from_iso(value) if value else None


def _row_to_reference_job(row: Mapping[str, Any]) -> ReferenceImportJob:
    image_bytes = row["image_bytes"]
    return ReferenceImportJob(
        id=int(row["id"]),
        source_url=str(row["source_url"]),
        resolved_image_url=(
            str(row["resolved_image_url"]) if row["resolved_image_url"] else None
        ),
        image_bytes=bytes(image_bytes) if image_bytes is not None else None,
        image_mime_type=(
            str(row["image_mime_type"]) if row["image_mime_type"] else None
        ),
        attempt_count=int(row["attempt_count"]),
    )


def _row_to_reference_asset(row: Mapping[str, Any]) -> ReferenceAsset:
    try:
        tags = json.loads(str(row["tags_json"]))
    except (TypeError, ValueError, json.JSONDecodeError):
        tags = {}
    if not isinstance(tags, dict):
        tags = {}
    return ReferenceAsset(
        id=int(row["id"]),
        source_url=str(row["source_url"]),
        resolved_image_url=str(row["resolved_image_url"] or ""),
        image_bytes=bytes(row["image_bytes"] or b""),
        image_mime_type=str(row["image_mime_type"] or "image/jpeg"),
        thumbnail_bytes=bytes(row["thumbnail_bytes"] or b""),
        width=int(row["width"] or 0),
        height=int(row["height"] or 0),
        tags=tags,
        use_count=int(row["use_count"] or 0),
        last_used_at_utc=_optional_datetime(row["last_used_at_utc"]),
        cooldown_until_utc=_optional_datetime(row["cooldown_until_utc"]),
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
        id_column = (
            "BIGSERIAL PRIMARY KEY"
            if self.database_url
            else ("INTEGER PRIMARY KEY AUTOINCREMENT")
        )
        binary_column = "BYTEA" if self.database_url else "BLOB"
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
            self._execute(
                connection,
                f"""
                CREATE TABLE IF NOT EXISTS reference_assets (
                    id {id_column},
                    source_url TEXT NOT NULL UNIQUE,
                    pin_id TEXT,
                    resolved_image_url TEXT,
                    image_bytes {binary_column},
                    image_mime_type TEXT,
                    thumbnail_bytes {binary_column},
                    width INTEGER,
                    height INTEGER,
                    image_sha256 TEXT,
                    tags_json TEXT NOT NULL DEFAULT '{{}}',
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    next_retry_at_utc TEXT,
                    last_error TEXT,
                    source_name TEXT NOT NULL DEFAULT '',
                    use_count INTEGER NOT NULL DEFAULT 0,
                    last_used_at_utc TEXT,
                    cooldown_until_utc TEXT,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL
                )
                """,
            )
            self._execute(
                connection,
                """
                CREATE INDEX IF NOT EXISTS idx_reference_import_queue
                ON reference_assets(status, next_retry_at_utc, id)
                """,
            )
            self._execute(
                connection,
                """
                CREATE INDEX IF NOT EXISTS idx_reference_selection
                ON reference_assets(status, cooldown_until_utc, use_count)
                """,
            )
            self._execute(
                connection,
                f"""
                CREATE TABLE IF NOT EXISTS reference_usages (
                    id {id_column},
                    reference_id BIGINT NOT NULL,
                    request_token TEXT NOT NULL UNIQUE,
                    garment_type TEXT NOT NULL,
                    target_gender TEXT NOT NULL,
                    moods_json TEXT NOT NULL DEFAULT '[]',
                    outcome TEXT NOT NULL DEFAULT 'reserved',
                    used_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL
                )
                """,
            )
            self._execute(
                connection,
                """
                CREATE INDEX IF NOT EXISTS idx_reference_usage_asset
                ON reference_usages(reference_id, used_at_utc)
                """,
            )
            self._execute(
                connection,
                """
                UPDATE reference_assets
                SET status = 'retry', next_retry_at_utc = ?,
                    last_error = 'Импорт был прерван перезапуском'
                WHERE status = 'processing'
                """,
                (_iso(datetime.now(UTC)),),
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

    @staticmethod
    def _active_draft_key(chat_id: int) -> str:
        return f"active_draft:{chat_id}"

    @staticmethod
    def _model_draft_key(chat_id: int) -> str:
        return f"model_draft:{chat_id}"

    def _save_expiring_draft(
        self,
        key: str,
        data: Mapping[str, Any],
        *,
        lifetime: timedelta,
    ) -> None:
        payload = {
            "expires_at_utc": _iso(datetime.now(UTC) + lifetime),
            "data": dict(data),
        }
        self.set_setting(
            key,
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        )

    def _get_expiring_draft(self, key: str) -> Optional[dict[str, Any]]:
        raw = self.get_setting(key)
        if not raw:
            return None
        try:
            payload = json.loads(raw)
            expires_at = _from_iso(payload["expires_at_utc"])
            data = payload["data"]
            if not isinstance(data, dict):
                raise ValueError("draft data must be an object")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            with self._connect() as connection:
                self._execute(
                    connection, "DELETE FROM app_settings WHERE key = ?", (key,)
                )
            return None
        if expires_at <= datetime.now(UTC):
            with self._connect() as connection:
                self._execute(
                    connection, "DELETE FROM app_settings WHERE key = ?", (key,)
                )
            return None
        return data

    def save_active_draft(
        self,
        chat_id: int,
        data: Mapping[str, Any],
        *,
        lifetime: timedelta = timedelta(hours=48),
    ) -> None:
        self._save_expiring_draft(
            self._active_draft_key(chat_id),
            data,
            lifetime=lifetime,
        )

    def get_active_draft(self, chat_id: int) -> Optional[dict[str, Any]]:
        return self._get_expiring_draft(self._active_draft_key(chat_id))

    def clear_active_draft(self, chat_id: int) -> None:
        with self._connect() as connection:
            self._execute(
                connection,
                "DELETE FROM app_settings WHERE key = ?",
                (self._active_draft_key(chat_id),),
            )

    def save_model_draft(
        self,
        chat_id: int,
        data: Mapping[str, Any],
        *,
        lifetime: timedelta = timedelta(hours=48),
    ) -> None:
        self._save_expiring_draft(
            self._model_draft_key(chat_id),
            data,
            lifetime=lifetime,
        )

    def get_model_draft(self, chat_id: int) -> Optional[dict[str, Any]]:
        return self._get_expiring_draft(self._model_draft_key(chat_id))

    def clear_model_draft(self, chat_id: int) -> None:
        with self._connect() as connection:
            self._execute(
                connection,
                "DELETE FROM app_settings WHERE key = ?",
                (self._model_draft_key(chat_id),),
            )

    def get_recent_mockup_directions(self, *, limit: int = 10) -> list[str]:
        raw = self.get_setting("mockup_recent_directions:v1")
        if not raw:
            return []
        try:
            values = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if not isinstance(values, list):
            return []
        labels = [str(value) for value in values if str(value).strip()]
        return labels[-max(1, limit) :]

    def remember_mockup_direction(self, label: str, *, limit: int = 10) -> None:
        clean_label = label.strip()
        if not clean_label:
            return
        labels = self.get_recent_mockup_directions(limit=limit)
        labels = [value for value in labels if value != clean_label]
        labels.append(clean_label)
        self.set_setting(
            "mockup_recent_directions:v1",
            json.dumps(labels[-max(1, limit) :], ensure_ascii=False),
        )

    def enqueue_reference_urls(
        self,
        urls: Sequence[str],
        *,
        source_name: str,
    ) -> tuple[int, int]:
        now = _iso(datetime.now(UTC))
        added = 0
        total = 0
        with self._connect() as connection:
            for url in urls:
                clean_url = str(url).strip()
                if not clean_url:
                    continue
                total += 1
                cursor = self._execute(
                    connection,
                    """
                    INSERT INTO reference_assets(
                        source_url, status, source_name,
                        created_at_utc, updated_at_utc
                    ) VALUES (?, 'pending', ?, ?, ?)
                    ON CONFLICT(source_url) DO NOTHING
                    """,
                    (clean_url, source_name[:200], now, now),
                )
                if cursor.rowcount == 1:
                    added += 1
        return added, total

    def claim_reference_import(self) -> Optional[ReferenceImportJob]:
        now = _iso(datetime.now(UTC))
        with self._connect() as connection:
            row = self._execute(
                connection,
                """
                SELECT id, source_url, resolved_image_url, image_bytes,
                       image_mime_type, attempt_count
                FROM reference_assets
                WHERE status IN ('pending', 'retry')
                  AND (next_retry_at_utc IS NULL OR next_retry_at_utc <= ?)
                ORDER BY
                    CASE WHEN image_bytes IS NOT NULL THEN 0 ELSE 1 END,
                    id ASC
                LIMIT 1
                """,
                (now,),
            ).fetchone()
            if not row:
                return None
            cursor = self._execute(
                connection,
                """
                UPDATE reference_assets
                SET status = 'processing', updated_at_utc = ?
                WHERE id = ? AND status IN ('pending', 'retry')
                """,
                (now, int(row["id"])),
            )
            if cursor.rowcount != 1:
                return None
        return _row_to_reference_job(row)

    def store_reference_image(
        self,
        reference_id: int,
        *,
        pin_id: str,
        resolved_image_url: str,
        image_bytes: bytes,
        image_mime_type: str,
        thumbnail_bytes: bytes,
        width: int,
        height: int,
        image_sha256: str,
    ) -> None:
        now = _iso(datetime.now(UTC))
        with self._connect() as connection:
            self._execute(
                connection,
                """
                UPDATE reference_assets
                SET pin_id = ?, resolved_image_url = ?, image_bytes = ?,
                    image_mime_type = ?, thumbnail_bytes = ?, width = ?,
                    height = ?, image_sha256 = ?, updated_at_utc = ?
                WHERE id = ? AND status = 'processing'
                """,
                (
                    pin_id,
                    resolved_image_url,
                    image_bytes,
                    image_mime_type,
                    thumbnail_bytes,
                    width,
                    height,
                    image_sha256,
                    now,
                    reference_id,
                ),
            )

    def mark_reference_ready(
        self,
        reference_id: int,
        *,
        tags: Mapping[str, Any],
    ) -> None:
        now = _iso(datetime.now(UTC))
        usable = bool(tags.get("usable", True))
        status = "ready" if usable else "disabled"
        error = None if usable else str(tags.get("unusable_reason", ""))[:1000]
        with self._connect() as connection:
            self._execute(
                connection,
                """
                UPDATE reference_assets
                SET tags_json = ?, status = ?, last_error = ?,
                    next_retry_at_utc = NULL, updated_at_utc = ?
                WHERE id = ?
                """,
                (
                    json.dumps(dict(tags), ensure_ascii=False, separators=(",", ":")),
                    status,
                    error,
                    now,
                    reference_id,
                ),
            )

    def mark_reference_import_error(
        self,
        reference_id: int,
        *,
        error: str,
        retry_at_utc: datetime,
        max_attempts: int,
    ) -> str:
        now = _iso(datetime.now(UTC))
        with self._connect() as connection:
            row = self._execute(
                connection,
                "SELECT attempt_count FROM reference_assets WHERE id = ?",
                (reference_id,),
            ).fetchone()
            attempts = (int(row["attempt_count"]) if row else 0) + 1
            status = "failed" if attempts >= max_attempts else "retry"
            self._execute(
                connection,
                """
                UPDATE reference_assets
                SET status = ?, attempt_count = ?, next_retry_at_utc = ?,
                    last_error = ?, updated_at_utc = ?
                WHERE id = ?
                """,
                (
                    status,
                    attempts,
                    _iso(retry_at_utc),
                    error[:1000],
                    now,
                    reference_id,
                ),
            )
        return status

    def retry_failed_references(self) -> int:
        now = _iso(datetime.now(UTC))
        with self._connect() as connection:
            cursor = self._execute(
                connection,
                """
                UPDATE reference_assets
                SET status = 'retry', attempt_count = 0,
                    next_retry_at_utc = ?, last_error = NULL,
                    updated_at_utc = ?
                WHERE status = 'failed'
                """,
                (now, now),
            )
            return int(cursor.rowcount)

    def resume_reference_imports(
        self,
        *,
        stale_after: timedelta = timedelta(minutes=10),
    ) -> dict[str, int]:
        """Make delayed retries due now and recover abandoned processing leases."""
        now_value = datetime.now(UTC)
        now = _iso(now_value)
        stale_before = _iso(now_value - stale_after)
        with self._connect() as connection:
            retry_cursor = self._execute(
                connection,
                """
                UPDATE reference_assets
                SET next_retry_at_utc = ?, updated_at_utc = ?
                WHERE status = 'retry'
                """,
                (now, now),
            )
            failed_cursor = self._execute(
                connection,
                """
                UPDATE reference_assets
                SET status = 'retry', attempt_count = 0,
                    next_retry_at_utc = ?, last_error = NULL,
                    updated_at_utc = ?
                WHERE status = 'failed'
                """,
                (now, now),
            )
            stale_cursor = self._execute(
                connection,
                """
                UPDATE reference_assets
                SET status = 'retry', next_retry_at_utc = ?,
                    last_error = 'Зависшая обработка автоматически восстановлена',
                    updated_at_utc = ?
                WHERE status = 'processing' AND updated_at_utc <= ?
                """,
                (now, now, stale_before),
            )
        return {
            "retry": int(retry_cursor.rowcount),
            "failed": int(failed_cursor.rowcount),
            "stale": int(stale_cursor.rowcount),
        }

    def recover_stale_reference_imports(
        self,
        *,
        stale_after: timedelta = timedelta(minutes=10),
    ) -> int:
        now_value = datetime.now(UTC)
        now = _iso(now_value)
        stale_before = _iso(now_value - stale_after)
        with self._connect() as connection:
            cursor = self._execute(
                connection,
                """
                UPDATE reference_assets
                SET status = 'retry', next_retry_at_utc = ?,
                    last_error = 'Зависшая обработка автоматически восстановлена',
                    updated_at_utc = ?
                WHERE status = 'processing' AND updated_at_utc <= ?
                """,
                (now, now, stale_before),
            )
            return int(cursor.rowcount)

    def reference_stats(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = self._execute(
                connection,
                """
                SELECT status, COUNT(*) AS amount
                FROM reference_assets
                GROUP BY status
                """,
            ).fetchall()
        stats = {str(row["status"]): int(row["amount"]) for row in rows}
        stats["total"] = sum(stats.values())
        return stats

    def reference_queue_details(self) -> dict[str, Any]:
        with self._connect() as connection:
            retry_row = self._execute(
                connection,
                """
                SELECT MIN(next_retry_at_utc) AS next_retry_at_utc
                FROM reference_assets
                WHERE status = 'retry'
                """,
            ).fetchone()
            reason_rows = self._execute(
                connection,
                """
                SELECT last_error, COUNT(*) AS amount
                FROM reference_assets
                WHERE status IN ('retry', 'failed')
                  AND last_error IS NOT NULL AND last_error <> ''
                GROUP BY last_error
                ORDER BY amount DESC
                LIMIT 3
                """,
            ).fetchall()
        next_retry = None
        if retry_row and retry_row["next_retry_at_utc"]:
            next_retry = _from_iso(retry_row["next_retry_at_utc"])
        return {
            "next_retry_at_utc": next_retry,
            "reasons": [
                (str(row["last_error"]), int(row["amount"])) for row in reason_rows
            ],
        }

    def list_ready_reference_assets(self, *, limit: int = 500) -> list[ReferenceAsset]:
        now = _iso(datetime.now(UTC))
        with self._connect() as connection:
            rows = self._execute(
                connection,
                """
                SELECT * FROM reference_assets
                WHERE status = 'ready'
                  AND image_bytes IS NOT NULL
                  AND (cooldown_until_utc IS NULL OR cooldown_until_utc <= ?)
                ORDER BY use_count ASC, COALESCE(last_used_at_utc, '') ASC, id ASC
                LIMIT ?
                """,
                (now, max(1, limit)),
            ).fetchall()
        return [_row_to_reference_asset(row) for row in rows]

    def reserve_reference(
        self,
        reference_id: int,
        *,
        request_token: str,
        garment_type: str,
        target_gender: str,
        moods: Sequence[str],
        cooldown: timedelta = timedelta(days=30),
    ) -> bool:
        now_value = datetime.now(UTC)
        now = _iso(now_value)
        cooldown_until = _iso(now_value + cooldown)
        with self._connect() as connection:
            existing = self._execute(
                connection,
                "SELECT id FROM reference_usages WHERE request_token = ?",
                (request_token,),
            ).fetchone()
            if existing:
                return False
            cursor = self._execute(
                connection,
                """
                UPDATE reference_assets
                SET use_count = use_count + 1, last_used_at_utc = ?,
                    cooldown_until_utc = ?, updated_at_utc = ?
                WHERE id = ? AND status = 'ready'
                  AND (cooldown_until_utc IS NULL OR cooldown_until_utc <= ?)
                """,
                (now, cooldown_until, now, reference_id, now),
            )
            if cursor.rowcount != 1:
                return False
            self._execute(
                connection,
                """
                INSERT INTO reference_usages(
                    reference_id, request_token, garment_type, target_gender,
                    moods_json, outcome, used_at_utc, updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, 'reserved', ?, ?)
                ON CONFLICT(request_token) DO NOTHING
                """,
                (
                    reference_id,
                    request_token,
                    garment_type,
                    target_gender,
                    json.dumps(list(moods), ensure_ascii=False),
                    now,
                    now,
                ),
            )
        return True

    def finish_reference_usage(self, request_token: str, *, outcome: str) -> None:
        now = _iso(datetime.now(UTC))
        with self._connect() as connection:
            self._execute(
                connection,
                """
                UPDATE reference_usages
                SET outcome = ?, updated_at_utc = ?
                WHERE request_token = ?
                """,
                (outcome[:30], now, request_token),
            )

    def release_reference_reservation(
        self,
        reference_id: int,
        request_token: str,
        *,
        outcome: str = "rejected_preflight",
    ) -> None:
        """Release a reference rejected before paid image generation."""
        now = _iso(datetime.now(UTC))
        with self._connect() as connection:
            self._execute(
                connection,
                """
                UPDATE reference_usages
                SET outcome = ?, updated_at_utc = ?
                WHERE request_token = ?
                """,
                (outcome[:30], now, request_token),
            )
            self._execute(
                connection,
                """
                UPDATE reference_assets
                SET use_count = CASE WHEN use_count > 0 THEN use_count - 1 ELSE 0 END,
                    cooldown_until_utc = NULL,
                    updated_at_utc = ?
                WHERE id = ?
                """,
                (now, reference_id),
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
                row = self._execute(
                    connection, query + " RETURNING id", params
                ).fetchone()
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
