import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.types import BotCommand

from app.config import Config, ConfigError
from app.copywriter import ImageCopywriter
from app.handlers import build_router
from app.publisher import Publisher
from app.storage import PostRepository


async def main() -> None:
    config = Config.from_env()
    config.ensure_runtime_paths()

    bot = Bot(token=config.telegram_bot_token)
    repository = PostRepository(config.database_path)
    repository.initialize()
    repository.recover_interrupted_posts()

    copywriter = ImageCopywriter(
        api_key=config.gemini_api_key,
        model=config.gemini_model,
        language=config.copy_language,
    )
    publisher = Publisher(bot=bot, config=config, repository=repository)

    dispatcher = Dispatcher()
    dispatcher.include_router(
        build_router(
            config=config,
            repository=repository,
            copywriter=copywriter,
            publisher=publisher,
        )
    )

    scheduler_task = asyncio.create_task(publisher.run_scheduler())
    try:
        await bot.set_my_commands(
            [
                BotCommand(command="start", description="Главное меню"),
                BotCommand(command="queue", description="Запланированные посты"),
                BotCommand(command="template", description="Шаблон подписи"),
                BotCommand(command="settemplate", description="Изменить шаблон"),
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
