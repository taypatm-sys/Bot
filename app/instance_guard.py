from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any


class SingleInstanceError(RuntimeError):
    pass


class SingleInstanceGuard:
    """Cross-process guard for Telegram long polling.

    PostgreSQL uses a session advisory lock, so overlapping Render deployments and
    duplicate services cannot poll the same bot at the same time. SQLite/local mode
    uses an OS file lock.
    """

    def __init__(self, *, database_url: str, bot_token: str) -> None:
        digest = hashlib.blake2b(bot_token.encode("utf-8"), digest_size=8).digest()
        self.lock_key = int.from_bytes(digest, "big") & 0x7FFF_FFFF_FFFF_FFFF
        self.database_url = database_url
        self._connection: Any = None
        self._file = None
        self._file_path = Path("/tmp/taypa_telegram_polling.lock")

    def acquire(self, timeout_seconds: float = 120.0) -> None:
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        while True:
            if self._try_acquire():
                return
            if time.monotonic() >= deadline:
                raise SingleInstanceError(
                    "Другой экземпляр бота уже выполняет Telegram polling"
                )
            time.sleep(2.0)

    def _try_acquire(self) -> bool:
        if self.database_url:
            return self._try_postgres()
        return self._try_file()

    def _try_postgres(self) -> bool:
        import psycopg

        if self._connection is None or self._connection.closed:
            self._connection = psycopg.connect(self.database_url, autocommit=True)
        with self._connection.cursor() as cursor:
            cursor.execute("SELECT pg_try_advisory_lock(%s)", (self.lock_key,))
            row = cursor.fetchone()
        acquired = bool(row and row[0])
        if not acquired:
            self._connection.close()
            self._connection = None
        return acquired

    def _try_file(self) -> bool:
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self._file_path, "a+", encoding="utf-8")
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            handle.close()
            return False
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        self._file = handle
        return True

    def close(self) -> None:
        if self._connection is not None:
            try:
                with self._connection.cursor() as cursor:
                    cursor.execute("SELECT pg_advisory_unlock(%s)", (self.lock_key,))
            finally:
                self._connection.close()
                self._connection = None
        if self._file is not None:
            try:
                if os.name == "nt":
                    import msvcrt

                    self._file.seek(0)
                    msvcrt.locking(self._file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
            finally:
                self._file.close()
                self._file = None
