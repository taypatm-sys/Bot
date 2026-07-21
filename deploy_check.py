from importlib import import_module
from pathlib import Path


REQUIRED_FILES = (
    "bot.py",
    "requirements.txt",
    "caption_template.txt",
    "app/__init__.py",
    "app/config.py",
    "app/copywriter.py",
    "app/formatting.py",
    "app/handlers.py",
    "app/health.py",
    "app/models.py",
    "app/publisher.py",
    "app/scheduling.py",
    "app/storage.py",
)


def main() -> None:
    root = Path(__file__).resolve().parent
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        names = ", ".join(missing)
        raise SystemExit(f"Incomplete deploy. Missing files: {names}")

    for module_name in (
        "app.config",
        "app.copywriter",
        "app.formatting",
        "app.handlers",
        "app.health",
        "app.models",
        "app.publisher",
        "app.scheduling",
        "app.storage",
    ):
        import_module(module_name)

    print("Render deploy check passed.")


if __name__ == "__main__":
    main()
