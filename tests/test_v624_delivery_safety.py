from pathlib import Path
from tempfile import TemporaryDirectory

from app.storage import PostRepository


ROOT = Path(__file__).resolve().parents[1]


def test_openai_request_is_not_orphaned_or_retried() -> None:
    source = (ROOT / "app" / "openai_mockup_generator.py").read_text(encoding="utf-8")
    assert "max_retries=0" in source
    assert "asyncio.wait_for(" not in source
    assert 'output_format="jpeg"' in source
    assert 'return "1024x1536"' in source


def test_generated_result_can_be_recovered_after_telegram_failure() -> None:
    with TemporaryDirectory() as directory:
        repository = PostRepository(Path(directory) / "bot.sqlite3")
        repository.initialize()
        artifact_id = repository.save_generation_artifact(
            chat_id=123,
            request_token="paid-request-1",
            provider="OpenAI",
            model="gpt-image-1.5",
            mime_type="image/jpeg",
            file_name="result.jpg",
            image_bytes=b"image-data",
            caption="ready",
        )
        stored = repository.get_latest_pending_generation_artifact(123)
        assert stored is not None
        assert int(stored["id"]) == artifact_id
        assert bytes(stored["image_bytes"]) == b"image-data"
        repository.delete_generation_artifact(artifact_id)
        assert repository.pending_generation_artifact_count(123) == 0
        repository.close()


def test_check_is_rendered_as_table() -> None:
    source = (ROOT / "app" / "handlers.py").read_text(encoding="utf-8")
    assert "def _check_table" in source
    assert 'parse_mode="HTML"' in source
    assert "<pre>" in source


def test_openai_output_is_real_four_by_five_jpeg() -> None:
    import io
    from PIL import Image
    from app.image_delivery import prepare_telegram_four_by_five

    source = Image.new("RGB", (1024, 1536), "white")
    raw = io.BytesIO()
    source.save(raw, format="PNG")
    result = prepare_telegram_four_by_five(raw.getvalue())
    with Image.open(io.BytesIO(result)) as image:
        assert image.size == (1024, 1280)
        assert image.format == "JPEG"


def test_tshirt_fit_is_forced_to_moderate_oversize() -> None:
    source = (ROOT / "app" / "handlers.py").read_text(encoding="utf-8")
    assert 'return "moderately oversized fit"' in source
    assert 'return "умеренный оверсайз"' in source


def test_all_static_keyboard_callbacks_have_handlers() -> None:
    import re

    source = (ROOT / "app" / "handlers.py").read_text(encoding="utf-8")
    buttons = {
        match.group(1)
        for match in re.finditer(
            r'callback_data\s*=\s*(?:f)?["\']([^"\']+)["\']', source
        )
        if "{" not in match.group(1)
    }
    exact = set(
        re.findall(
            r'@router\.callback_query\(F\.data\s*==\s*["\']([^"\']+)', source
        )
    )
    prefixes = set(
        re.findall(
            r'@router\.callback_query\(F\.data\.startswith\(["\']([^"\']+)',
            source,
        )
    )
    in_values: set[str] = set()
    for block in re.findall(
        r'@router\.callback_query\(\s*F\.data\.in_\(\{(.*?)\}\)\s*\)',
        source,
        re.S,
    ):
        in_values.update(re.findall(r'["\']([^"\']+)["\']', block))

    missing = sorted(
        value
        for value in buttons
        if value not in exact
        and value not in in_values
        and not any(value.startswith(prefix) for prefix in prefixes)
    )
    assert missing == []


def test_paid_fallback_recalculates_cost() -> None:
    source = (ROOT / "app" / "handlers.py").read_text(encoding="utf-8")
    marker = "Локальная обработка повышена до Gemini"
    block = source[source.index(marker) : source.index(marker) + 2500]
    assert "estimated_cost_usd = _estimated_generation_cost_usd" in block


def test_openai_default_medium_portrait_cost_estimate() -> None:
    import sys
    import types

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = object
    sys.modules.setdefault("openai", fake_openai)
    # The source-level value is also checked to avoid relying on an installed SDK.
    source = (ROOT / "app" / "openai_mockup_generator.py").read_text(encoding="utf-8")
    assert '"medium": (0.034, 0.050)' in source
    assert '"minimum image estimate"' in source


def test_billed_error_records_request_id_and_cost() -> None:
    generator_source = (ROOT / "app" / "openai_mockup_generator.py").read_text(
        encoding="utf-8"
    )
    handler_source = (ROOT / "app" / "handlers.py").read_text(encoding="utf-8")
    assert "estimated_cost_usd=cost" in generator_source
    assert 'getattr(error, "estimated_cost_usd"' in handler_source
    assert 'repository.set_setting("last_openai_request_id", request_id)' in handler_source
