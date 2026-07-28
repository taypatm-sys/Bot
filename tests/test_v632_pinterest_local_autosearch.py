from pathlib import Path


def test_local_generation_calls_product_pinterest_search():
    text = Path("app/handlers.py").read_text(encoding="utf-8")
    assert 'requested_generation_mode == "local"' in text
    assert "reference_catalog.discover_for_product(" in text
    assert "import_now=3" in text
    assert 'preferred_source_name=dynamic_source_name' in text
    assert "last_pinterest_product_prepared" in text


def test_token_enables_search_by_default():
    text = Path("app/config.py").read_text(encoding="utf-8")
    assert "pinterest_search_enabled = bool(pinterest_access_token)" in text
    assert "pinterest_search_enabled=pinterest_search_enabled" in text
