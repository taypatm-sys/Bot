from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_openai_provider_is_wired():
    config = (ROOT / "app" / "config.py").read_text(encoding="utf-8")
    handlers = (ROOT / "app" / "handlers.py").read_text(encoding="utf-8")
    bot = (ROOT / "bot.py").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert "OPENAI_API_KEY" in config
    assert "model:generate:openai" in handlers
    assert (
        'F.data.in_({"model:generate", "model:generate:local", "model:generate:gemini", "model:generate:openai"})'
        in handlers
    )
    assert "OpenAIMockupGenerator" in bot
    assert "openai>=" in requirements


def test_three_generation_modes_are_visible():
    handlers = (ROOT / "app" / "handlers.py").read_text(encoding="utf-8")
    assert "Простой - бесплатно" in handlers
    assert 'text="🔵 Gemini"' in handlers
    assert 'text="🟣 OpenAI"' in handlers
