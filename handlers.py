import io
import logging
from datetime import datetime, timezone

from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
)

from app.config import Config
from app.copywriter import ImageCopywriter
from app.formatting import (
    TemplateError,
    contact_link,
    normalize_price,
    normalize_size,
    render_caption,
    split_product_title,
    validate_template,
)
from app.publisher import Publisher
from app.scheduling import (
    format_local,
    from_utc_timestamp,
    parse_local_datetime,
    quick_times,
    to_utc_timestamp,
)
from app.storage import PostRepository


UTC = timezone.utc
logger = logging.getLogger(__name__)


class DraftStates(StatesGroup):
    waiting_size = State()
    waiting_custom_size = State()
    waiting_price = State()
    waiting_custom_time = State()
    waiting_text_edit = State()
    waiting_price_edit = State()
    waiting_template = State()
    preview = State()


def main_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="Создать пост")],
            [KeyboardButton(text="Запланированные"), KeyboardButton(text="Шаблон")],
        ],
        resize_keyboard=True,
    )


def schedule_keyboard(config: Config) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(text="Опубликовать сейчас", callback_data="time:now")]
    ]
    for value in quick_times(config.timezone):
        local = value.astimezone(config.timezone)
        label = local.strftime("%d.%m в %H:%M")
        rows.append(
            [
                InlineKeyboardButton(
                    text=label,
                    callback_data=f"time:{to_utc_timestamp(value)}",
                )
            ]
        )
    rows.append(
        [InlineKeyboardButton(text="Другая дата и время", callback_data="time:custom")]
    )
    rows.append([InlineKeyboardButton(text="Отмена", callback_data="draft:cancel")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def size_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="S-XXL", callback_data="size:S-XXL"),
                InlineKeyboardButton(text="XS-4XL", callback_data="size:XS-4XL"),
            ],
            [
                InlineKeyboardButton(text="XS-3XL", callback_data="size:XS-3XL"),
                InlineKeyboardButton(
                    text="Вписать свой вариант", callback_data="size:custom"
                ),
            ],
            [InlineKeyboardButton(text="Отмена", callback_data="draft:cancel")],
        ]
    )


def preview_keyboard(config: Config, title: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=config.button_text,
                    url=contact_link(config.contact_username, title),
                )
            ],
            [
                InlineKeyboardButton(text="Подтвердить", callback_data="draft:confirm"),
                InlineKeyboardButton(
                    text="Изменить название", callback_data="draft:text"
                ),
            ],
            [
                InlineKeyboardButton(text="Изменить цену", callback_data="draft:price"),
                InlineKeyboardButton(
                    text="Изменить размеры", callback_data="draft:size"
                ),
            ],
            [
                InlineKeyboardButton(text="Изменить время", callback_data="draft:time"),
            ],
            [InlineKeyboardButton(text="Отмена", callback_data="draft:cancel")],
        ]
    )


async def is_admin_message(message: Message, config: Config) -> bool:
    if message.from_user and message.from_user.id == config.admin_telegram_id:
        return True
    await message.answer("У вас нет доступа к управлению этим ботом.")
    return False


async def is_admin_callback(callback: CallbackQuery, config: Config) -> bool:
    if callback.from_user.id == config.admin_telegram_id:
        return True
    await callback.answer("Нет доступа", show_alert=True)
    return False


async def ask_for_time(message: Message, config: Config) -> None:
    await message.answer(
        "Когда опубликовать пост?", reply_markup=schedule_keyboard(config)
    )


async def send_preview(message: Message, state: FSMContext, config: Config) -> None:
    data = await state.get_data()
    try:
        caption = render_caption(
            config.caption_template_path,
            title=data["title"],
            description=data["description"],
            size=data["size"],
            price=data["price"],
            garment_type=data.get("garment_type", ""),
            design_name=data.get("design_name", ""),
            theme_hashtag=data.get("theme_hashtag", ""),
        )
    except TemplateError as error:
        await message.answer(f"Ошибка шаблона: {error}")
        return

    scheduled_at = datetime.fromisoformat(data["scheduled_at_utc"])
    await message.answer(
        "Предпросмотр. В публикации останется только верхняя кнопка.\n"
        f"Время: {format_local(scheduled_at, config.timezone)}"
    )
    await message.answer_photo(
        photo=data["photo_file_id"],
        caption=caption,
        reply_markup=preview_keyboard(config, data["title"]),
    )
    await state.set_state(DraftStates.preview)


def build_router(
    *,
    config: Config,
    repository: PostRepository,
    copywriter: ImageCopywriter,
    publisher: Publisher,
) -> Router:
    router = Router()

    @router.message(CommandStart())
    async def start(message: Message, state: FSMContext) -> None:
        if not await is_admin_message(message, config):
            return
        await state.clear()
        await message.answer(
            "Бот готов. Отправьте фотографию или нажмите «Создать пост».",
            reply_markup=main_keyboard(),
        )

    @router.message(Command("cancel"))
    async def cancel_command(message: Message, state: FSMContext) -> None:
        if not await is_admin_message(message, config):
            return
        await state.clear()
        await message.answer("Текущий пост отменен.", reply_markup=main_keyboard())

    @router.message(Command("check"))
    async def check_settings(message: Message, bot: Bot) -> None:
        if not await is_admin_message(message, config):
            return
        try:
            bot_info = await bot.get_me()
            chat = await bot.get_chat(config.channel_id)
            member = await bot.get_chat_member(config.channel_id, bot_info.id)
            await message.answer(
                "Настройки Telegram работают.\n"
                f"Бот: @{bot_info.username}\n"
                f"Канал: {chat.title or chat.id}\n"
                f"Статус бота в канале: {member.status}"
            )
        except Exception as error:
            logger.exception("Проверка настроек Telegram не пройдена")
            await message.answer(
                "Проверка не пройдена. Убедитесь, что CHANNEL_ID указан верно, "
                "а бот добавлен в администраторы канала с правом публикации.\n\n"
                f"Ошибка: {error}"
            )

    @router.message(Command("queue"))
    @router.message(F.text == "Запланированные")
    async def queue(message: Message) -> None:
        if not await is_admin_message(message, config):
            return
        posts = repository.list_pending()
        if not posts:
            await message.answer("Запланированных постов нет.")
            return
        for post in posts:
            status = "ожидает" if post.status == "scheduled" else "ошибка"
            keyboard = InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        InlineKeyboardButton(
                            text="Отменить пост",
                            callback_data=f"queue:cancel:{post.id}",
                        )
                    ]
                ]
            )
            await message.answer(
                f"#{post.id} | {post.title}\n"
                f"{format_local(post.scheduled_at_utc, config.timezone)} | {status}",
                reply_markup=keyboard,
            )

    @router.callback_query(F.data.startswith("queue:cancel:"))
    async def cancel_queued(callback: CallbackQuery) -> None:
        if not await is_admin_callback(callback, config):
            return
        post_id = int(callback.data.rsplit(":", 1)[1])
        cancelled = repository.cancel(post_id)
        await callback.answer("Пост отменен" if cancelled else "Пост уже обработан")
        if callback.message:
            await callback.message.edit_reply_markup(reply_markup=None)

    @router.message(Command("template"))
    @router.message(F.text == "Шаблон")
    async def show_template(message: Message) -> None:
        if not await is_admin_message(message, config):
            return
        template = config.caption_template_path.read_text(encoding="utf-8").strip()
        await message.answer(
            "Текущий шаблон:\n\n"
            f"{template}\n\n"
            "Чтобы изменить его, отправьте /settemplate. "
            "Можно использовать русские поля из примера команды /settemplate."
        )

    @router.message(Command("settemplate"))
    async def set_template(message: Message, state: FSMContext) -> None:
        if not await is_admin_message(message, config):
            return
        await state.set_state(DraftStates.waiting_template)
        await message.answer(
            "Отправьте новый шаблон одним сообщением. Доступные поля:\n"
            "{Тип товара}, {Название принта}, "
            "{Короткое описание, передающее настроение принта}, "
            "{Размеры}, {Цена}, {тип товара}, {тематика принта}.\n\n"
            "Старые поля {title}, {description}, {size}, {price} и {hashtags} "
            "тоже поддерживаются."
        )

    @router.message(DraftStates.waiting_template, F.text)
    async def save_template(message: Message, state: FSMContext) -> None:
        if not await is_admin_message(message, config):
            return
        try:
            validate_template(message.text)
        except TemplateError as error:
            await message.answer(f"Шаблон не сохранен: {error}")
            return
        config.caption_template_path.write_text(
            message.text.strip() + "\n", encoding="utf-8"
        )
        await state.clear()
        await message.answer("Шаблон сохранен.", reply_markup=main_keyboard())

    @router.message(F.text == "Создать пост")
    async def create_post_hint(message: Message, state: FSMContext) -> None:
        if not await is_admin_message(message, config):
            return
        await state.clear()
        await message.answer("Отправьте фотографию будущего поста.")

    @router.message(F.photo)
    async def receive_photo(message: Message, state: FSMContext, bot: Bot) -> None:
        if not await is_admin_message(message, config):
            return
        await state.clear()
        waiting = await message.answer("Анализирую изображение и пишу текст...")
        photo = message.photo[-1]
        buffer = io.BytesIO()
        try:
            await bot.download(photo, destination=buffer)
            generated = await copywriter.create_copy(buffer.getvalue(), "image/jpeg")
        except Exception:
            logger.exception("Не удалось создать текст по изображению")
            await waiting.edit_text(
                "Не удалось проанализировать изображение. Проверьте GEMINI_API_KEY "
                "и повторите отправку."
            )
            return

        await state.update_data(
            photo_file_id=photo.file_id,
            title=generated.title,
            description=generated.description,
            garment_type=generated.garment_type,
            design_name=generated.design_name,
            theme_hashtag=generated.theme_hashtag,
        )
        await state.set_state(DraftStates.waiting_size)
        await waiting.edit_text(
            f"Название: {generated.title}\n\n"
            "Выберите доступные размеры:",
            reply_markup=size_keyboard(),
        )

    @router.callback_query(F.data.startswith("size:"))
    async def choose_size(callback: CallbackQuery, state: FSMContext) -> None:
        if not await is_admin_callback(callback, config):
            return
        data = await state.get_data()
        required = {"photo_file_id", "title", "description"}
        if not required.issubset(data):
            await callback.answer(
                "Этот черновик уже закрыт. Отправьте фотографию заново.",
                show_alert=True,
            )
            return

        value = callback.data.split(":", 1)[1]
        if value == "custom":
            await state.set_state(DraftStates.waiting_custom_size)
            await callback.answer()
            if callback.message:
                await callback.message.answer(
                    "Напишите свой вариант размеров. Например: M-3XL"
                )
            return

        await state.update_data(size=normalize_size(value))
        await callback.answer(f"Размеры: {value}")
        if not callback.message:
            return
        if "scheduled_at_utc" in data:
            await send_preview(callback.message, state, config)
            return
        await state.set_state(DraftStates.waiting_price)
        await callback.message.answer("Напишите цену. Например: 250")

    @router.message(DraftStates.waiting_custom_size, F.text)
    async def receive_custom_size(message: Message, state: FSMContext) -> None:
        if not await is_admin_message(message, config):
            return
        try:
            size = normalize_size(message.text)
        except ValueError as error:
            await message.answer(str(error))
            return
        data = await state.get_data()
        await state.update_data(size=size)
        if "scheduled_at_utc" in data:
            await send_preview(message, state, config)
            return
        await state.set_state(DraftStates.waiting_price)
        await message.answer("Напишите цену. Например: 250")

    @router.message(DraftStates.waiting_price, F.text)
    async def receive_price(message: Message, state: FSMContext) -> None:
        if not await is_admin_message(message, config):
            return
        try:
            price = normalize_price(message.text)
        except ValueError as error:
            await message.answer(str(error))
            return
        await state.update_data(price=price)
        await ask_for_time(message, config)

    @router.callback_query(F.data.startswith("time:"))
    async def choose_time(callback: CallbackQuery, state: FSMContext) -> None:
        if not await is_admin_callback(callback, config):
            return
        data = await state.get_data()
        required = {"photo_file_id", "title", "description", "size", "price"}
        if not required.issubset(data):
            await callback.answer(
                "Этот черновик уже закрыт. Отправьте фотографию заново.",
                show_alert=True,
            )
            return
        value = callback.data.split(":", 1)[1]
        if value == "custom":
            await state.set_state(DraftStates.waiting_custom_time)
            await callback.answer()
            if callback.message:
                await callback.message.answer(
                    "Напишите дату и время. Например: 25.07.2026 18:30"
                )
            return
        scheduled_at = (
            datetime.now(UTC) if value == "now" else from_utc_timestamp(int(value))
        )
        await state.update_data(scheduled_at_utc=scheduled_at.isoformat())
        await callback.answer()
        if callback.message:
            await send_preview(callback.message, state, config)

    @router.message(DraftStates.waiting_custom_time, F.text)
    async def receive_custom_time(message: Message, state: FSMContext) -> None:
        if not await is_admin_message(message, config):
            return
        try:
            scheduled_at = parse_local_datetime(message.text, config.timezone)
        except ValueError as error:
            await message.answer(str(error))
            return
        await state.update_data(scheduled_at_utc=scheduled_at.isoformat())
        await send_preview(message, state, config)

    @router.callback_query(F.data == "draft:text")
    async def edit_text(callback: CallbackQuery, state: FSMContext) -> None:
        if not await is_admin_callback(callback, config):
            return
        await state.set_state(DraftStates.waiting_text_edit)
        await callback.answer()
        if callback.message:
            await callback.message.answer(
                'Отправьте новую первую строку целиком. Например: '
                'Футболка "Welcome to Turkmenistan"'
            )

    @router.message(DraftStates.waiting_text_edit, F.text)
    async def receive_text_edit(message: Message, state: FSMContext) -> None:
        if not await is_admin_message(message, config):
            return
        title = " ".join(message.text.strip().split())
        if not title:
            await message.answer("Название не может быть пустым.")
            return
        if len(title) > 100:
            await message.answer("Название слишком длинное.")
            return
        garment_type, design_name = split_product_title(title)
        await state.update_data(
            title=title,
            garment_type=garment_type,
            design_name=design_name,
        )
        await send_preview(message, state, config)

    @router.callback_query(F.data == "draft:size")
    async def edit_size(callback: CallbackQuery, state: FSMContext) -> None:
        if not await is_admin_callback(callback, config):
            return
        await state.set_state(DraftStates.waiting_size)
        await callback.answer()
        if callback.message:
            await callback.message.answer(
                "Выберите новые размеры:", reply_markup=size_keyboard()
            )

    @router.callback_query(F.data == "draft:price")
    async def edit_price(callback: CallbackQuery, state: FSMContext) -> None:
        if not await is_admin_callback(callback, config):
            return
        await state.set_state(DraftStates.waiting_price_edit)
        await callback.answer()
        if callback.message:
            await callback.message.answer("Напишите новую цену.")

    @router.message(DraftStates.waiting_price_edit, F.text)
    async def receive_price_edit(message: Message, state: FSMContext) -> None:
        if not await is_admin_message(message, config):
            return
        try:
            price = normalize_price(message.text)
        except ValueError as error:
            await message.answer(str(error))
            return
        await state.update_data(price=price)
        await send_preview(message, state, config)

    @router.callback_query(F.data == "draft:time")
    async def edit_time(callback: CallbackQuery, state: FSMContext) -> None:
        if not await is_admin_callback(callback, config):
            return
        await callback.answer()
        if callback.message:
            await ask_for_time(callback.message, config)

    @router.callback_query(F.data == "draft:cancel")
    async def cancel_draft(callback: CallbackQuery, state: FSMContext) -> None:
        if not await is_admin_callback(callback, config):
            return
        await state.clear()
        await callback.answer("Черновик удален")
        if callback.message:
            await callback.message.answer("Создание поста отменено.")

    @router.callback_query(F.data == "draft:confirm")
    async def confirm_draft(callback: CallbackQuery, state: FSMContext) -> None:
        if not await is_admin_callback(callback, config):
            return
        data = await state.get_data()
        required = {
            "photo_file_id",
            "title",
            "description",
            "size",
            "price",
            "scheduled_at_utc",
        }
        if not required.issubset(data):
            await callback.answer("Черновик устарел. Начните заново.", show_alert=True)
            await state.clear()
            return

        scheduled_at = datetime.fromisoformat(data["scheduled_at_utc"])
        post_id = repository.create(
            author_id=callback.from_user.id,
            photo_file_id=data["photo_file_id"],
            title=data["title"],
            description=data["description"],
            size=data["size"],
            price=data["price"],
            scheduled_at_utc=scheduled_at,
            garment_type=data.get("garment_type", ""),
            design_name=data.get("design_name", ""),
            theme_hashtag=data.get("theme_hashtag", ""),
        )
        await state.clear()
        await callback.answer("Готово")
        if callback.message:
            await callback.message.edit_reply_markup(reply_markup=None)

        if scheduled_at <= datetime.now(UTC):
            success = await publisher.publish_one(post_id)
            if callback.message and not success:
                await callback.message.answer(
                    f"Пост #{post_id} пока не опубликован. Бот повторит попытку автоматически."
                )
        elif callback.message:
            await callback.message.answer(
                f"Пост #{post_id} запланирован на "
                f"{format_local(scheduled_at, config.timezone)}."
            )

    @router.message()
    async def fallback(message: Message) -> None:
        if not await is_admin_message(message, config):
            return
        await message.answer(
            "Отправьте фотографию для нового поста или выберите действие в меню.",
            reply_markup=main_keyboard(),
        )

    return router
