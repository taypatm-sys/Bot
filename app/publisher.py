import asyncio
import logging
from datetime import datetime, timedelta, timezone

from aiogram import Bot
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from app.config import Config
from app.formatting import contact_link, render_caption_text
from app.storage import PostRepository
from app.template_store import CaptionTemplateStore


logger = logging.getLogger(__name__)
UTC = timezone.utc


class Publisher:
    def __init__(
        self,
        *,
        bot: Bot,
        config: Config,
        repository: PostRepository,
        template_store: CaptionTemplateStore,
    ):
        self.bot = bot
        self.config = config
        self.repository = repository
        self.template_store = template_store
        self.max_attempts = 5

    def public_keyboard(self, title: str) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text=self.config.button_text,
                        url=contact_link(self.config.contact_username, title),
                    )
                ]
            ]
        )

    async def publish_one(self, post_id: int) -> bool:
        if not self.repository.claim_for_publish(post_id):
            return False
        post = self.repository.get(post_id)
        if post is None:
            return False

        try:
            caption = render_caption_text(
                self.template_store.get(),
                title=post.title,
                description=post.description,
                size=post.size,
                price=post.price,
                garment_type=post.garment_type,
                design_name=post.design_name,
                theme_hashtag=post.theme_hashtag,
            )
            message = await self.bot.send_photo(
                chat_id=self.config.channel_id,
                photo=post.photo_file_id,
                caption=caption,
                reply_markup=self.public_keyboard(post.title),
            )
            self.repository.mark_published(post_id, message.message_id)
            for admin_id in self.config.admin_ids:
                try:
                    await self.bot.send_message(
                        chat_id=admin_id,
                        text=f"Пост #{post_id} опубликован.",
                    )
                except Exception:
                    logger.warning(
                        "Не удалось уведомить администратора %s", admin_id
                    )
            return True
        except Exception as error:
            retry_minutes = min(2 ** (post.attempts + 1), 60)
            status = self.repository.mark_publish_error(
                post_id,
                error=str(error),
                next_attempt_at_utc=datetime.now(UTC)
                + timedelta(minutes=retry_minutes),
                max_attempts=self.max_attempts,
            )
            logger.exception("Не удалось опубликовать пост %s", post_id)
            if status == "failed":
                for admin_id in self.config.admin_ids:
                    try:
                        await self.bot.send_message(
                            chat_id=admin_id,
                            text=(
                                f"Пост #{post_id} не опубликован после "
                                f"{self.max_attempts} попыток. Проверьте права "
                                "бота в канале и настройки Render."
                            ),
                        )
                    except Exception:
                        logger.warning(
                            "Не удалось уведомить администратора %s", admin_id
                        )
            return False

    async def run_scheduler(self) -> None:
        while True:
            try:
                due_ids = self.repository.due_ids(datetime.now(UTC))
                for post_id in due_ids:
                    await self.publish_one(post_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Ошибка цикла публикации")
            await asyncio.sleep(10)
