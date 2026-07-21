import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.types import BotCommand

from app.config import Config, ConfigError
from app.copywriter import ImageCopywriter
from app.handlers import build_router
from app.health import start_health_server
from app.publisher import Publisher
from app.storage import PostRepository
from app.template_store import CaptionTemplateStore


DEFAULT_PRODUCT_PRESETS = (
    ("Футболка", "S-XXL", "250 манат"),
    ("A4", "S-XXL", "210 манат"),
    ("A3", "S-XXL", "240 манат"),
    ("A2", "S-XXL", "290 манат"),
)


async def main() -> None:
    config = Config.from_env()
    config.ensure_runtime_paths()

    bot = Bot(token=config.telegram_bot_token)
    repository = PostRepository(config.database_source)
    repository.initialize()
    repository.recover_interrupted_posts()
    repository.seed_presets(DEFAULT_PRODUCT_PRESETS)

    template_store = CaptionTemplateStore(
        repository=repository,
        fallback_path=config.caption_template_path,
    )
    template_store.initialize()

    copywriter = ImageCopywriter(
        api_key=config.gemini_api_key,
        model=config.gemini_model,
        language=config.copy_language,
    )
    publisher = Publisher(
        bot=bot,
        config=config,
        repository=repository,
        template_store=template_store,
    )

    dispatcher = Dispatcher()
    dispatcher.include_router(
        build_router(
            config=config,
            repository=repository,
            copywriter=copywriter,
            publisher=publisher,
            template_store=template_store,
        )
    )

    health_runner = await start_health_server()
    scheduler_task = asyncio.create_task(publisher.run_scheduler())
    try:
        await bot.set_my_commands(
            [
                BotCommand(command="start", description="Главное меню"),
                BotCommand(command="queue", description="Запланированные посты"),
                BotCommand(command="template", description="Шаблон подписи"),
                BotCommand(command="settemplate", description="Изменить шаблон"),
                BotCommand(command="presets", description="Готовые пресеты"),
                BotCommand(command="check", description="Проверить настройки"),
                BotCommand(command="cancel", description="Отменить черновик"),
            ]
        )
        await dispatcher.start_polling(
            bot,
            allowed_updates=dispatcher.resolve_used_update_types(),
        )
    finally:
        scheduler_task.cancel()
        await asyncio.gather(scheduler_task, return_exceptions=True)
        if health_runner is not None:
            await health_runner.cleanup()
        repository.close()
        await bot.session.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    try:
        asyncio.run(main())
    except ConfigError as error:
        raise SystemExit(f"Ошибка настройки: {error}") from error
