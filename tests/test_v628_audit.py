from pathlib import Path

from app.config import normalize_openai_image_quality, normalize_openai_image_size


ROOT = Path(__file__).resolve().parents[1]


def test_create_post_and_model_are_separate_flows() -> None:
    source = (ROOT / "app" / "handlers.py").read_text(encoding="utf-8")
    post_block = source[source.index('async def request_post_photo') : source.index('async def request_model_mockup')]
    model_block = source[source.index('async def request_model_mockup') : source.index('async def accept_model_mockup')]
    assert 'Отправьте готовую фотографию товара' in post_block
    assert 'DraftStates.waiting_model_mockup' not in post_block
    assert 'DraftStates.waiting_model_mockup' in model_block


def test_legacy_openai_four_by_five_env_is_repaired() -> None:
    assert normalize_openai_image_size("1024x1280") == "1024x1536"
    assert normalize_openai_image_quality("standard") == "medium"


def test_health_server_starts_before_database_initialization() -> None:
    source = (ROOT / "bot.py").read_text(encoding="utf-8")
    assert source.index("health_runner = await start_health_server()") < source.index("repository.initialize()")


def test_railway_config_and_dockerignore_exist() -> None:
    assert (ROOT / "railway.toml").is_file()
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert "__pycache__" in dockerignore
    assert ".env" in dockerignore


def test_admin_messages_use_html_parse_mode() -> None:
    source = (ROOT / "app" / "handlers.py").read_text(encoding="utf-8")
    assert 'render_admins_text(config, repository),\n            parse_mode="HTML"' in source


def test_deploy_check_includes_openai_and_version_modules() -> None:
    source = (ROOT / "deploy_check.py").read_text(encoding="utf-8")
    assert '"app/openai_mockup_generator.py"' in source
    assert '"app/image_delivery.py"' in source
    assert '"app/version.py"' in source


def test_paid_generation_never_forces_incompatible_reference() -> None:
    source = (ROOT / "app" / "handlers.py").read_text(encoding="utf-8")
    assert 'compatibility.model_copy(update={"compatible": True})' not in source
    assert "confirmed_rejected_preflight" in source
    assert "Подходящий референс не найден. Генерация не запускалась." in source


def test_preview_delivery_failure_releases_reference() -> None:
    source = (ROOT / "app" / "handlers.py").read_text(encoding="utf-8")
    assert 'outcome="preview_delivery_failed"' in source
    assert "Платная генерация не запускалась" in source


def test_publisher_notifies_database_admins_too() -> None:
    source = (ROOT / "app" / "publisher.py").read_text(encoding="utf-8")
    assert "def admin_ids" in source
    assert "for admin_id in self.admin_ids()" in source


def test_docker_build_imports_every_module_before_deploy() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    deploy_check = (ROOT / "deploy_check.py").read_text(encoding="utf-8")
    assert "python deploy_check.py" in dockerfile
    assert '"bot"' in deploy_check


def test_pinterest_search_does_not_enable_itself_from_blank_env() -> None:
    source = (ROOT / "app" / "config.py").read_text(encoding="utf-8")
    block = source[source.index("pinterest_search_enabled=_bool_env") :]
    assert "default=False" in block[:220]


def test_check_table_caps_long_values() -> None:
    source = (ROOT / "app" / "handlers.py").read_text(encoding="utf-8")
    assert "if len(value) > 240" in source


def test_settings_admin_button_fits_telegram_width() -> None:
    source = (ROOT / "app" / "handlers.py").read_text(encoding="utf-8")
    assert 'text="👥 Админы"' in source


def test_reference_navigation_edits_existing_photo() -> None:
    source = (ROOT / "app" / "handlers.py").read_text(encoding="utf-8")
    assert "InputMediaPhoto" in source
    assert "await callback.message.edit_media(" in source


def test_postgres_failure_does_not_silently_use_temporary_sqlite() -> None:
    import pytest
    from app.storage import PostRepository

    repository = PostRepository("postgresql://invalid", allow_sqlite_fallback=False)
    with pytest.raises(RuntimeError):
        repository._fallback_to_sqlite("connection failed")
