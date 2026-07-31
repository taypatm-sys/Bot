import sqlite3
from pathlib import Path

import pytest

from app.config import ConfigError, normalize_database_url
from app.storage import PostRepository


def test_database_url_repairs_stray_sslmode_quotes() -> None:
    assert (
        normalize_database_url(
            '"postgresql://user:pass@example.com/db?sslmode=%22require%22"'
        )
        == "postgresql://user:pass@example.com/db?sslmode=require"
    )
    assert (
        normalize_database_url(
            "postgresql://user:pass@example.com/db?sslmode=require%22%22"
        )
        == "postgresql://user:pass@example.com/db?sslmode=require"
    )


def test_database_url_rejects_unknown_sslmode() -> None:
    with pytest.raises(ConfigError):
        normalize_database_url(
            "postgresql://user:pass@example.com/db?sslmode=required"
        )


def test_postgres_never_silently_falls_back_to_sqlite() -> None:
    repository = PostRepository(
        "postgresql://example.invalid/test",
        allow_sqlite_fallback=True,
    )
    with pytest.raises(RuntimeError, match="без перехода на SQLite"):
        repository._fallback_to_sqlite("connection failed")
    assert repository.database_url.startswith("postgresql://")


def test_legacy_sqlite_null_ids_are_backfilled_and_future_ids_populate(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute(
        "CREATE TABLE reference_assets (id BIGSERIAL PRIMARY KEY, source_url TEXT)"
    )
    connection.execute(
        "INSERT INTO reference_assets(source_url) VALUES ('https://example.com/1')"
    )
    assert connection.execute(
        "SELECT id FROM reference_assets"
    ).fetchone()["id"] is None

    repository = PostRepository(path)
    repository._repair_legacy_sqlite_ids(connection)
    connection.commit()

    first_id = connection.execute(
        "SELECT id FROM reference_assets"
    ).fetchone()["id"]
    assert first_id == 1

    connection.execute(
        "INSERT INTO reference_assets(source_url) VALUES ('https://example.com/2')"
    )
    connection.commit()
    ids = [
        row["id"]
        for row in connection.execute(
            "SELECT id FROM reference_assets ORDER BY rowid"
        ).fetchall()
    ]
    assert ids == [1, 2]
    connection.close()
