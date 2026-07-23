import tempfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from app.formatting import contact_link
from app.instance_guard import SingleInstanceError, SingleInstanceGuard
from app.storage import PostRepository


def _ready_reference(repository: PostRepository) -> int:
    repository.enqueue_reference_urls(
        ["https://www.pinterest.com/pin/123456/"],
        source_name="test",
    )
    job = repository.claim_reference_import()
    assert job is not None
    repository.store_reference_image(
        job.id,
        pin_id="123456",
        resolved_image_url="https://i.pinimg.com/example.jpg",
        image_bytes=b"image",
        image_mime_type="image/jpeg",
        thumbnail_bytes=b"thumb",
        width=100,
        height=120,
        image_sha256="hash",
    )
    repository.mark_reference_ready(
        job.id,
        tags={
            "usable": True,
            "garment_types": ["t-shirt"],
            "gender": "unisex",
            "print_area_visibility": 95,
            "print_side_visible": "front",
            "camera_angle": "front",
            "framing": "waist-up",
            "garment_is_plain": True,
        },
    )
    return job.id


def test_reference_lifecycle_and_learning() -> None:
    path = Path(tempfile.mkdtemp()) / "test.sqlite3"
    repository = PostRepository(path)
    repository.initialize()
    reference_id = _ready_reference(repository)

    repository.store_simple_reference_variant(
        reference_id,
        image_bytes=b"prepared",
        image_mime_type="image/jpeg",
        thumbnail_bytes=b"prepared-thumb",
        ready=True,
        reason="чистая зона принта готова",
        level="A",
        quality_score=94,
    )
    prepared = repository.get_reference_asset(reference_id)
    assert prepared is not None
    assert prepared.lifecycle_state == "prepared"
    assert prepared.simple_level == "A"
    assert prepared.simple_quality_score == 94

    assert repository.reserve_reference(
        reference_id,
        request_token="request-1",
        garment_type="t-shirt",
        target_gender="unisex",
        moods=["street"],
    )
    repository.update_reference_match(
        reference_id,
        score=91,
        reason="сторона и кадрирование совпадают",
    )
    repository.record_reference_result(
        reference_id,
        success=True,
        match_score=91,
        reason="успешная генерация",
    )
    successful = repository.get_reference_asset(reference_id)
    assert successful is not None
    assert successful.lifecycle_state == "successful"
    assert successful.success_count == 1
    assert successful.last_match_score == 91
    repository.close()


def test_single_instance_file_lock() -> None:
    first = SingleInstanceGuard(database_url="", bot_token="test-token-v60")
    second = SingleInstanceGuard(database_url="", bot_token="test-token-v60")
    first.acquire(0)
    with pytest.raises(SingleInstanceError):
        second.acquire(0)
    first.close()
    second.acquire(0)
    second.close()


def test_order_message_uses_correct_accusative() -> None:
    url = contact_link("taypa", 'Футболка "Degmäň"')
    message = parse_qs(urlparse(url).query)["text"][0]
    assert message == "Здравствуйте! Хочу заказать футболку «Degmäň»."


def test_postgres_lifecycle_backfill_parameterizes_unicode_like_pattern() -> None:
    class FakeConnection:
        def __init__(self) -> None:
            self.query = ""
            self.params = ()

        def execute(self, query, params=()):
            self.query = query
            self.params = params
            return self

    repository = PostRepository("postgresql://example.invalid/test")
    connection = FakeConnection()
    repository._backfill_reference_lifecycle(connection)

    assert "simple_reason LIKE %s" in connection.query
    assert "%чистая%" not in connection.query
    assert connection.params == ("%чистая%",)
