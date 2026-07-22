import io
import logging
import secrets
from datetime import datetime, timezone

from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    BufferedInputFile,
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
    render_caption_text,
    split_product_title,
)
from app.models import ProductPreset, ScheduledPost
from app.mockup_generator import (
    MockupGenerationError,
    MockupGenerator,
    MockupSpec,
    choose_photo_directions,
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
from app.template_store import CaptionTemplateStore


UTC = timezone.utc
logger = logging.getLogger(__name__)


class DraftStates(StatesGroup):
    waiting_model_mockup = State()
    generating_model_photos = State()
    model_photos_ready = State()
    waiting_size = State()
    waiting_custom_size = State()
    waiting_price = State()
    waiting_custom_time = State()
    waiting_text_edit = State()
    waiting_description_edit = State()
    waiting_price_edit = State()
    waiting_template = State()
    waiting_preset = State()
    waiting_queue_title = State()
    waiting_queue_description = State()
    waiting_queue_size = State()
    waiting_queue_price = State()
    waiting_queue_time = State()
    preview = State()


def main_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [
                KeyboardButton(text="Создать пост"),
                KeyboardButton(text="Фото на модели"),
            ],
            [
                KeyboardButton(text="Запланированные"),
                KeyboardButton(text="Пресеты"),
            ],
            [KeyboardButton(text="Шаблон")],
        ],
        resize_keyboard=True,
    )


def model_photo_keyboard(batch_id: str, index: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Создать пост из этого",
                    callback_data=f"model:post:{batch_id}:{index}",
                )
            ]
        ]
    )


def model_batch_keyboard(batch_id: str, count: int) -> InlineKeyboardMarkup:
    count_label = "1 вариант" if count == 1 else f"{count} варианта"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=f"Еще {count_label}",
                    callback_data=f"model:more:{batch_id}",
                )
            ],
            [
                InlineKeyboardButton(
                    text="Закончить",
                    callback_data=f"model:done:{batch_id}",
                )
            ],
        ]
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


def queue_schedule_keyboard(config: Config, post_id: int) -> InlineKeyboardMarkup:
    rows = [
        [
            InlineKeyboardButton(
                text="Опубликовать сейчас",
                callback_data=f"queue:time:{post_id}:now",
            )
        ]
    ]
    for value in quick_times(config.timezone):
        local = value.astimezone(config.timezone)
        rows.append(
            [
                InlineKeyboardButton(
                    text=local.strftime("%d.%m в %H:%M"),
                    callback_data=(
                        f"queue:time:{post_id}:{to_utc_timestamp(value)}"
                    ),
                )
            ]
        )
    rows.append(
        [
            InlineKeyboardButton(
                text="Другая дата и время",
                callback_data=f"queue:time:{post_id}:custom",
            )
        ]
    )
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


def preset_choice_keyboard(presets: list[ProductPreset]) -> InlineKeyboardMarkup:
    rows = [
        [
            InlineKeyboardButton(
                text=f"{preset.name}: {preset.size}, {preset.price}",
                callback_data=f"preset:use:{preset.id}",
            )
        ]
        for preset in presets
    ]
    rows.append(
        [
            InlineKeyboardButton(
                text="Ввести размеры и цену вручную",
                callback_data="preset:manual",
            )
        ]
    )
    rows.append([InlineKeyboardButton(text="Отмена", callback_data="draft:cancel")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def preset_manager_keyboard(presets: list[ProductPreset]) -> InlineKeyboardMarkup:
    rows = [
        [
            InlineKeyboardButton(
                text=f"Удалить: {preset.name}",
                callback_data=f"preset:delete:{preset.id}",
            )
        ]
        for preset in presets
    ]
    rows.append(
        [InlineKeyboardButton(text="Добавить пресет", callback_data="preset:add")]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


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
                    text="Название", callback_data="draft:text"
                ),
            ],
            [
                InlineKeyboardButton(
                    text="Описание", callback_data="draft:description"
                ),
                InlineKeyboardButton(text="Цена", callback_data="draft:price"),
            ],
            [
                InlineKeyboardButton(text="Размеры", callback_data="draft:size"),
                InlineKeyboardButton(text="Время", callback_data="draft:time"),
            ],
            [InlineKeyboardButton(text="Отмена", callback_data="draft:cancel")],
        ]
    )


def queue_card_text(post: ScheduledPost, config: Config) -> str:
    status = "ожидает" if post.status == "scheduled" else "ошибка публикации"
    return (
        f"#{post.id} | {post.title}\n"
        f"{post.size} | {post.price}\n"
        f"{format_local(post.scheduled_at_utc, config.timezone)} | {status}"
    )


def queue_keyboard(post_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Предпросмотр",
                    callback_data=f"queue:preview:{post_id}",
                ),
                InlineKeyboardButton(
                    text="Изменить", callback_data=f"queue:edit:{post_id}"
                ),
            ],
            [
                InlineKeyboardButton(
                    text="Опубликовать сейчас",
                    callback_data=f"queue:publish:{post_id}",
                ),
            ],
            [
                InlineKeyboardButton(
                    text="Создать копию", callback_data=f"queue:copy:{post_id}"
                ),
                InlineKeyboardButton(
                    text="Отменить", callback_data=f"queue:cancel:{post_id}"
                ),
            ],
        ]
    )


def queue_edit_keyboard(post_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Название",
                    callback_data=f"queue:field:title:{post_id}",
                ),
                InlineKeyboardButton(
                    text="Описание",
                    callback_data=f"queue:field:description:{post_id}",
                ),
            ],
            [
                InlineKeyboardButton(
                    text="Размеры", callback_data=f"queue:field:size:{post_id}"
                ),
                InlineKeyboardButton(
                    text="Цена", callback_data=f"queue:field:price:{post_id}"
                ),
            ],
            [
                InlineKeyboardButton(
                    text="Дата и время",
                    callback_data=f"queue:field:time:{post_id}",
                )
            ],
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


async def send_preview(
    message: Message,
    state: FSMContext,
    config: Config,
    template_store: CaptionTemplateStore,
) -> None:
    data = await state.get_data()
    try:
        caption = render_caption_text(
            template_store.get(),
            title=data["title"],
            description=data["description"],
            size=data["size"],
            price=data["price"],
            garment_type=data.get("garment_type", ""),
            design_name=data.get("design_name", ""),
            theme_hashtag=data.get("theme_hashtag", ""),
        )
    except (KeyError, TemplateError) as error:
        await message.answer(f"Ошибка предпросмотра: {error}")
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


async def show_updated_queue_post(
    message: Message,
    repository: PostRepository,
    config: Config,
    post_id: int,
) -> None:
    post = repository.get(post_id)
    if post and post.status in {"scheduled", "failed"}:
        await message.answer(
            queue_card_text(post, config),
            reply_markup=queue_keyboard(post.id),
        )


def build_router(
    *,
    config: Config,
    repository: PostRepository,
    copywriter: ImageCopywriter,
    mockup_generator: MockupGenerator,
    publisher: Publisher,
    template_store: CaptionTemplateStore,
) -> Router:
    router = Router()

    required_draft_fields = {"photo_file_id", "title", "description"}

    async def restore_active_draft(
        state: FSMContext,
        chat_id: int,
    ) -> dict:
        data = await state.get_data()
        if required_draft_fields.issubset(data):
            return data
        saved = repository.get_active_draft(chat_id)
        if saved and required_draft_fields.issubset(saved):
            await state.update_data(**saved)
            return await state.get_data()
        return data

    async def save_active_draft(state: FSMContext, chat_id: int) -> None:
        data = await state.get_data()
        if required_draft_fields.issubset(data):
            repository.save_active_draft(chat_id, data)

    async def prepare_post_draft(
        *,
        message: Message,
        state: FSMContext,
        waiting: Message,
        photo_file_id: str,
        image_bytes: bytes,
        mime_type: str,
    ) -> bool:
        try:
            generated = await copywriter.create_copy(image_bytes, mime_type)
        except Exception:
            logger.exception("Не удалось создать текст по изображению")
            await waiting.edit_text(
                "Не удалось проанализировать изображение. Проверьте GEMINI_API_KEY "
                "и повторите отправку."
            )
            return False

        await state.clear()
        draft_data = dict(
            photo_file_id=photo_file_id,
            title=generated.title,
            description=generated.description,
            garment_type=generated.garment_type,
            design_name=generated.design_name,
            theme_hashtag=generated.theme_hashtag,
        )
        await state.update_data(**draft_data)
        repository.save_active_draft(message.chat.id, draft_data)
        presets = repository.list_presets()
        if presets:
            await waiting.edit_text(
                f"Название: {generated.title}\n\nВыберите готовый пресет:",
                reply_markup=preset_choice_keyboard(presets),
            )
        else:
            await state.set_state(DraftStates.waiting_size)
            await waiting.edit_text(
                f"Название: {generated.title}\n\nВыберите доступные размеры:",
                reply_markup=size_keyboard(),
            )
        return True

    async def generate_model_batch(
        *,
        message: Message,
        state: FSMContext,
        bot: Bot,
        status_message: Message,
    ) -> None:
        data = await state.get_data()
        source_file_id = data.get("model_source_file_id")
        source_mime_type = data.get("model_source_mime_type", "image/jpeg")
        if not source_file_id:
            await state.set_state(DraftStates.waiting_model_mockup)
            await status_message.edit_text(
                "Макет не найден. Отправьте его еще раз.",
            )
            return

        await state.set_state(DraftStates.generating_model_photos)
        source_buffer = io.BytesIO()
        try:
            await bot.download(source_file_id, destination=source_buffer)
        except Exception:
            logger.exception("Не удалось скачать исходный макет")
            await state.set_state(DraftStates.waiting_model_mockup)
            await status_message.edit_text(
                "Не удалось скачать макет из Telegram. Отправьте его еще раз."
            )
            return

        source_bytes = source_buffer.getvalue()
        stored_spec = data.get("model_mockup_spec")
        if stored_spec:
            spec = MockupSpec.model_validate(
                {"target_gender": "unisex", **stored_spec}
            )
        else:
            await status_message.edit_text(
                "Определяю пол модели, сторону, цвет, размер и положение принта..."
            )
            spec = await mockup_generator.analyze_mockup(
                source_bytes,
                source_mime_type,
            )
            await state.update_data(
                model_mockup_spec=spec.model_dump() if spec else None
            )

        batch_id = secrets.token_hex(4)
        used_labels = list(data.get("model_used_direction_labels", []))
        target_gender = spec.target_gender if spec else "unisex"
        directions = choose_photo_directions(
            config.mockup_variants,
            target_gender=target_gender,
            exclude_labels=used_labels,
        )
        generated_file_ids: list[str] = []
        generation_error: str | None = None

        for index, direction in enumerate(directions, start=1):
            wearer_label = (
                "женская модель" if direction.gender == "women" else "мужская модель"
            )
            await status_message.edit_text(
                f"Создаю вариант {index} из {len(directions)}. "
                f"Выбрана {wearer_label}. Лицо, поза и место будут новыми..."
            )
            try:
                generated_photo = await mockup_generator.generate_variant(
                    image_bytes=source_bytes,
                    mime_type=source_mime_type,
                    spec=spec,
                    direction=direction,
                    request_token=batch_id,
                )
            except MockupGenerationError as error:
                generation_error = error.user_message
                break

            sent = await message.answer_photo(
                photo=BufferedInputFile(
                    generated_photo.data,
                    filename=(
                        f"taypa_model_{batch_id}_{index}."
                        f"{generated_photo.extension}"
                    ),
                ),
                caption=(
                    f"Вариант {index} из {len(directions)}\n"
                    f"{direction.label}\n"
                    f"{wearer_label.capitalize()}\n"
                    "Формат 4:5"
                ),
                reply_markup=model_photo_keyboard(batch_id, index - 1),
            )
            generated_file_ids.append(sent.photo[-1].file_id)
            used_labels.append(direction.label)

        if not generated_file_ids:
            await state.set_state(DraftStates.waiting_model_mockup)
            await status_message.edit_text(
                generation_error
                or "Не удалось создать варианты. Отправьте макет еще раз."
            )
            return

        await state.update_data(
            model_batch_id=batch_id,
            model_generated_file_ids=generated_file_ids,
            model_used_direction_labels=used_labels,
        )
        await state.set_state(DraftStates.model_photos_ready)
        result_text = (
            f"Готово: {len(generated_file_ids)} вариантов. Выберите фотографию "
            "для поста или запросите новые."
        )
        if generation_error:
            result_text += f"\n\nСледующий вариант не создан: {generation_error}"
        await status_message.edit_text(
            result_text,
            reply_markup=model_batch_keyboard(
                batch_id,
                config.mockup_variants,
            ),
        )

    @router.message(CommandStart())
    async def start(message: Message, state: FSMContext) -> None:
        if not await is_admin_message(message, config):
            return
        await state.clear()
        repository.clear_active_draft(message.chat.id)
        storage_note = (
            "Постоянная база подключена."
            if repository.is_persistent
            else "Работаю с локальной базой SQLite."
        )
        await message.answer(
            "Бот готов. Для обычной публикации отправьте фотографию. Для создания "
            "реалистичных кадров из макета нажмите «Фото на модели».\n"
            f"{storage_note}",
            reply_markup=main_keyboard(),
        )

    @router.message(Command("cancel"))
    async def cancel_command(message: Message, state: FSMContext) -> None:
        if not await is_admin_message(message, config):
            return
        await state.clear()
        repository.clear_active_draft(message.chat.id)
        await message.answer("Текущее действие отменено.", reply_markup=main_keyboard())

    @router.message(Command("check"))
    async def check_settings(message: Message, bot: Bot) -> None:
        if not await is_admin_message(message, config):
            return
        try:
            repository.get_setting("caption_template")
            bot_info = await bot.get_me()
            chat = await bot.get_chat(config.channel_id)
            member = await bot.get_chat_member(config.channel_id, bot_info.id)
            await message.answer(
                "Настройки работают.\n"
                f"Бот: @{bot_info.username}\n"
                f"Канал: {chat.title or chat.id}\n"
                f"Статус бота в канале: {member.status}\n"
                f"База: {repository.backend_name}\n"
                f"Фото на модели: {config.gemini_image_model}, "
                f"{config.gemini_image_size}, 4:5"
            )
        except Exception as error:
            logger.exception("Проверка настроек не пройдена")
            await message.answer(
                "Проверка не пройдена. Проверьте CHANNEL_ID, права бота в канале "
                "и подключение базы.\n\n"
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
        await message.answer(f"Запланировано постов: {len(posts)}")
        for post in posts:
            await message.answer(
                queue_card_text(post, config),
                reply_markup=queue_keyboard(post.id),
            )

    @router.callback_query(F.data.startswith("queue:preview:"))
    async def preview_queued(callback: CallbackQuery) -> None:
        if not await is_admin_callback(callback, config):
            return
        post_id = int(callback.data.rsplit(":", 1)[1])
        post = repository.get(post_id)
        if not post or post.status not in {"scheduled", "failed"}:
            await callback.answer("Пост уже обработан", show_alert=True)
            return
        try:
            caption = render_caption_text(
                template_store.get(),
                title=post.title,
                description=post.description,
                size=post.size,
                price=post.price,
                garment_type=post.garment_type,
                design_name=post.design_name,
                theme_hashtag=post.theme_hashtag,
            )
        except TemplateError as error:
            await callback.answer(f"Ошибка шаблона: {error}", show_alert=True)
            return
        await callback.answer()
        if callback.message:
            await callback.message.answer_photo(
                photo=post.photo_file_id,
                caption=caption,
                reply_markup=publisher.public_keyboard(post.title),
            )

    @router.callback_query(F.data.startswith("queue:edit:"))
    async def edit_queued(callback: CallbackQuery) -> None:
        if not await is_admin_callback(callback, config):
            return
        post_id = int(callback.data.rsplit(":", 1)[1])
        post = repository.get(post_id)
        if not post or post.status not in {"scheduled", "failed"}:
            await callback.answer("Пост уже обработан", show_alert=True)
            return
        await callback.answer()
        if callback.message:
            await callback.message.answer(
                f"Что изменить в посте #{post_id}?",
                reply_markup=queue_edit_keyboard(post_id),
            )

    @router.callback_query(F.data.startswith("queue:field:"))
    async def choose_queue_field(callback: CallbackQuery, state: FSMContext) -> None:
        if not await is_admin_callback(callback, config):
            return
        _, _, field, post_id_text = callback.data.split(":", 3)
        post_id = int(post_id_text)
        post = repository.get(post_id)
        if not post or post.status not in {"scheduled", "failed"}:
            await callback.answer("Пост уже обработан", show_alert=True)
            return
        if field == "time":
            await callback.answer()
            if callback.message:
                await callback.message.answer(
                    "Выберите новое время:",
                    reply_markup=queue_schedule_keyboard(config, post_id),
                )
            return

        states = {
            "title": DraftStates.waiting_queue_title,
            "description": DraftStates.waiting_queue_description,
            "size": DraftStates.waiting_queue_size,
            "price": DraftStates.waiting_queue_price,
        }
        prompts = {
            "title": "Отправьте новое название целиком.",
            "description": "Отправьте новое короткое описание.",
            "size": "Отправьте новые размеры. Например: S-XXL",
            "price": "Отправьте новую цену. Например: 250",
        }
        if field not in states:
            await callback.answer("Неизвестное действие", show_alert=True)
            return
        await state.clear()
        await state.update_data(edit_post_id=post_id)
        await state.set_state(states[field])
        await callback.answer()
        if callback.message:
            await callback.message.answer(prompts[field])

    @router.message(DraftStates.waiting_queue_title, F.text)
    async def receive_queue_title(message: Message, state: FSMContext) -> None:
        if not await is_admin_message(message, config):
            return
        data = await state.get_data()
        title = " ".join(message.text.strip().split())
        if not title or len(title) > 100:
            await message.answer("Название должно содержать от 1 до 100 символов.")
            return
        garment_type, design_name = split_product_title(title)
        post_id = int(data["edit_post_id"])
        updated = repository.update_pending(
            post_id,
            title=title,
            garment_type=garment_type,
            design_name=design_name,
        )
        await state.clear()
        await message.answer("Название обновлено." if updated else "Пост уже обработан.")
        if updated:
            await show_updated_queue_post(message, repository, config, post_id)

    @router.message(DraftStates.waiting_queue_description, F.text)
    async def receive_queue_description(message: Message, state: FSMContext) -> None:
        if not await is_admin_message(message, config):
            return
        description = " ".join(message.text.strip().split())
        if not description or len(description) > 200:
            await message.answer("Описание должно содержать от 1 до 200 символов.")
            return
        data = await state.get_data()
        post_id = int(data["edit_post_id"])
        updated = repository.update_pending(post_id, description=description)
        await state.clear()
        await message.answer("Описание обновлено." if updated else "Пост уже обработан.")
        if updated:
            await show_updated_queue_post(message, repository, config, post_id)

    @router.message(DraftStates.waiting_queue_size, F.text)
    async def receive_queue_size(message: Message, state: FSMContext) -> None:
        if not await is_admin_message(message, config):
            return
        try:
            size = normalize_size(message.text)
        except ValueError as error:
            await message.answer(str(error))
            return
        data = await state.get_data()
        post_id = int(data["edit_post_id"])
        updated = repository.update_pending(post_id, size=size)
        await state.clear()
        await message.answer("Размеры обновлены." if updated else "Пост уже обработан.")
        if updated:
            await show_updated_queue_post(message, repository, config, post_id)

    @router.message(DraftStates.waiting_queue_price, F.text)
    async def receive_queue_price(message: Message, state: FSMContext) -> None:
        if not await is_admin_message(message, config):
            return
        try:
            price = normalize_price(message.text)
        except ValueError as error:
            await message.answer(str(error))
            return
        data = await state.get_data()
        post_id = int(data["edit_post_id"])
        updated = repository.update_pending(post_id, price=price)
        await state.clear()
        await message.answer("Цена обновлена." if updated else "Пост уже обработан.")
        if updated:
            await show_updated_queue_post(message, repository, config, post_id)

    @router.callback_query(F.data.startswith("queue:time:"))
    async def change_queue_time(callback: CallbackQuery, state: FSMContext) -> None:
        if not await is_admin_callback(callback, config):
            return
        _, _, post_id_text, value = callback.data.split(":", 3)
        post_id = int(post_id_text)
        if value == "custom":
            await state.clear()
            await state.update_data(edit_post_id=post_id)
            await state.set_state(DraftStates.waiting_queue_time)
            await callback.answer()
            if callback.message:
                await callback.message.answer(
                    "Напишите новую дату и время. Например: 25.07.2026 18:30"
                )
            return
        scheduled_at = (
            datetime.now(UTC) if value == "now" else from_utc_timestamp(int(value))
        )
        updated = repository.reschedule(post_id, scheduled_at)
        await callback.answer("Время обновлено" if updated else "Пост уже обработан")
        if updated and callback.message:
            await show_updated_queue_post(callback.message, repository, config, post_id)

    @router.message(DraftStates.waiting_queue_time, F.text)
    async def receive_queue_time(message: Message, state: FSMContext) -> None:
        if not await is_admin_message(message, config):
            return
        try:
            scheduled_at = parse_local_datetime(message.text, config.timezone)
        except ValueError as error:
            await message.answer(str(error))
            return
        data = await state.get_data()
        post_id = int(data["edit_post_id"])
        updated = repository.reschedule(post_id, scheduled_at)
        await state.clear()
        await message.answer("Время обновлено." if updated else "Пост уже обработан.")
        if updated:
            await show_updated_queue_post(message, repository, config, post_id)

    @router.callback_query(F.data.startswith("queue:publish:"))
    async def publish_queued_now(callback: CallbackQuery) -> None:
        if not await is_admin_callback(callback, config):
            return
        post_id = int(callback.data.rsplit(":", 1)[1])
        if not repository.reschedule(post_id, datetime.now(UTC)):
            await callback.answer("Пост уже обработан", show_alert=True)
            return
        await callback.answer("Публикую...")
        success = await publisher.publish_one(post_id)
        if callback.message:
            await callback.message.edit_reply_markup(reply_markup=None)
            if not success:
                await callback.message.answer(
                    f"Пост #{post_id} пока не опубликован. Бот повторит попытку."
                )

    @router.callback_query(F.data.startswith("queue:copy:"))
    async def copy_queued(callback: CallbackQuery, state: FSMContext) -> None:
        if not await is_admin_callback(callback, config):
            return
        post_id = int(callback.data.rsplit(":", 1)[1])
        post = repository.get(post_id)
        if not post or post.status not in {"scheduled", "failed"}:
            await callback.answer("Пост уже обработан", show_alert=True)
            return
        await state.clear()
        await state.update_data(
            photo_file_id=post.photo_file_id,
            title=post.title,
            description=post.description,
            garment_type=post.garment_type,
            design_name=post.design_name,
            theme_hashtag=post.theme_hashtag,
            size=post.size,
            price=post.price,
        )
        await callback.answer("Копия создана")
        if callback.message:
            await callback.message.answer(
                f"Копия поста #{post_id}. Выберите время публикации.",
                reply_markup=schedule_keyboard(config),
            )

    @router.callback_query(F.data.startswith("queue:cancel:"))
    async def cancel_queued(callback: CallbackQuery) -> None:
        if not await is_admin_callback(callback, config):
            return
        post_id = int(callback.data.rsplit(":", 1)[1])
        cancelled = repository.cancel(post_id)
        await callback.answer("Пост отменен" if cancelled else "Пост уже обработан")
        if callback.message and cancelled:
            await callback.message.edit_reply_markup(reply_markup=None)

    @router.message(Command("template"))
    @router.message(F.text == "Шаблон")
    async def show_template(message: Message) -> None:
        if not await is_admin_message(message, config):
            return
        await message.answer(
            "Текущий шаблон:\n\n"
            f"{template_store.get()}\n\n"
            "Чтобы изменить его, отправьте /settemplate."
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
            template_store.set(message.text)
        except TemplateError as error:
            await message.answer(f"Шаблон не сохранен: {error}")
            return
        await state.clear()
        await message.answer(
            "Шаблон сохранен в постоянной базе.", reply_markup=main_keyboard()
        )

    @router.message(Command("presets"))
    @router.message(F.text == "Пресеты")
    async def show_presets(message: Message) -> None:
        if not await is_admin_message(message, config):
            return
        presets = repository.list_presets()
        if presets:
            lines = [
                f"{index}. {item.name} | {item.size} | {item.price}"
                for index, item in enumerate(presets, start=1)
            ]
            text = "Готовые пресеты:\n\n" + "\n".join(lines)
        else:
            text = "Готовых пресетов пока нет."
        await message.answer(text, reply_markup=preset_manager_keyboard(presets))

    @router.callback_query(F.data == "preset:add")
    async def add_preset(callback: CallbackQuery, state: FSMContext) -> None:
        if not await is_admin_callback(callback, config):
            return
        await state.clear()
        await state.set_state(DraftStates.waiting_preset)
        await callback.answer()
        if callback.message:
            await callback.message.answer(
                "Отправьте пресет в формате:\n"
                "Название | Размеры | Цена\n\n"
                "Например: Вареная футболка | M-2XL | 290"
            )

    @router.message(DraftStates.waiting_preset, F.text)
    async def receive_preset(message: Message, state: FSMContext) -> None:
        if not await is_admin_message(message, config):
            return
        parts = [part.strip() for part in message.text.split("|")]
        if len(parts) != 3:
            await message.answer(
                "Нужны три значения через знак |. Например:\n"
                "Худи | S-2XL | 460"
            )
            return
        name, size_text, price_text = parts
        if not name or len(name) > 40:
            await message.answer("Название пресета должно содержать до 40 символов.")
            return
        if any(item.name.casefold() == name.casefold() for item in repository.list_presets()):
            await message.answer("Пресет с таким названием уже существует.")
            return
        try:
            size = normalize_size(size_text)
            price = normalize_price(price_text)
            repository.create_preset(name=name, size=size, price=price)
        except ValueError as error:
            await message.answer(str(error))
            return
        await state.clear()
        await message.answer("Пресет добавлен.", reply_markup=main_keyboard())

    @router.callback_query(F.data.startswith("preset:delete:"))
    async def delete_preset(callback: CallbackQuery) -> None:
        if not await is_admin_callback(callback, config):
            return
        preset_id = int(callback.data.rsplit(":", 1)[1])
        deleted = repository.delete_preset(preset_id)
        await callback.answer("Пресет удален" if deleted else "Пресет уже удален")
        if callback.message and deleted:
            presets = repository.list_presets()
            await callback.message.edit_reply_markup(
                reply_markup=preset_manager_keyboard(presets)
            )

    @router.message(Command("model"))
    @router.message(F.text == "Фото на модели")
    async def request_model_mockup(message: Message, state: FSMContext) -> None:
        if not await is_admin_message(message, config):
            return
        await state.clear()
        repository.clear_active_draft(message.chat.id)
        await state.set_state(DraftStates.waiting_model_mockup)
        generation_count = (
            "одно реалистичное фото"
            if config.mockup_variants == 1
            else f"{config.mockup_variants} разных реалистичных фото"
        )
        await message.answer(
            "Отправьте готовый макет одежды. Лучше отправить его как файл PNG или "
            "JPEG, тогда мелкий текст и детали принта сохранятся точнее.\n\n"
            f"Бот создаст {generation_count} 4:5."
        )

    async def accept_model_mockup(
        message: Message,
        state: FSMContext,
        bot: Bot,
        *,
        file_id: str,
        mime_type: str,
    ) -> None:
        await state.clear()
        repository.clear_active_draft(message.chat.id)
        await state.update_data(
            model_source_file_id=file_id,
            model_source_mime_type=mime_type,
            model_mockup_spec=None,
            model_used_direction_labels=[],
        )
        status_message = await message.answer(
            "Макет принят. Сначала измеряю принт, затем создаю фотографии. "
            "Это может занять несколько минут."
        )
        await generate_model_batch(
            message=message,
            state=state,
            bot=bot,
            status_message=status_message,
        )

    @router.message(DraftStates.waiting_model_mockup, F.photo)
    async def receive_model_photo(
        message: Message,
        state: FSMContext,
        bot: Bot,
    ) -> None:
        if not await is_admin_message(message, config):
            return
        await accept_model_mockup(
            message,
            state,
            bot,
            file_id=message.photo[-1].file_id,
            mime_type="image/jpeg",
        )

    @router.message(DraftStates.waiting_model_mockup, F.document)
    async def receive_model_document(
        message: Message,
        state: FSMContext,
        bot: Bot,
    ) -> None:
        if not await is_admin_message(message, config):
            return
        document = message.document
        mime_type = document.mime_type or ""
        if not mime_type.startswith("image/"):
            await message.answer("Нужен файл изображения PNG, JPEG или WEBP.")
            return
        await accept_model_mockup(
            message,
            state,
            bot,
            file_id=document.file_id,
            mime_type=mime_type,
        )

    @router.callback_query(F.data.startswith("model:more:"))
    async def more_model_photos(
        callback: CallbackQuery,
        state: FSMContext,
        bot: Bot,
    ) -> None:
        if not await is_admin_callback(callback, config):
            return
        data = await state.get_data()
        batch_id = callback.data.rsplit(":", 1)[1]
        if data.get("model_batch_id") != batch_id:
            await callback.answer("Эта серия уже закрыта", show_alert=True)
            return
        if await state.get_state() == DraftStates.generating_model_photos.state:
            await callback.answer("Фотографии уже создаются", show_alert=True)
            return
        await callback.answer("Создаю новые варианты")
        if callback.message:
            status_message = await callback.message.answer(
                "Готовлю новую серию с другими людьми и локациями..."
            )
            await generate_model_batch(
                message=callback.message,
                state=state,
                bot=bot,
                status_message=status_message,
            )

    @router.callback_query(F.data.startswith("model:done:"))
    async def finish_model_photos(
        callback: CallbackQuery,
        state: FSMContext,
    ) -> None:
        if not await is_admin_callback(callback, config):
            return
        data = await state.get_data()
        batch_id = callback.data.rsplit(":", 1)[1]
        if data.get("model_batch_id") != batch_id:
            await callback.answer("Эта серия уже закрыта", show_alert=True)
            return
        await state.clear()
        await callback.answer("Готово")
        if callback.message:
            await callback.message.answer(
                "Генерация завершена.",
                reply_markup=main_keyboard(),
            )

    @router.callback_query(F.data.startswith("model:post:"))
    async def post_from_model_photo(
        callback: CallbackQuery,
        state: FSMContext,
        bot: Bot,
    ) -> None:
        if not await is_admin_callback(callback, config):
            return
        parts = callback.data.split(":")
        if len(parts) != 4:
            await callback.answer("Кнопка устарела", show_alert=True)
            return
        batch_id = parts[2]
        try:
            index = int(parts[3])
        except ValueError:
            await callback.answer("Кнопка устарела", show_alert=True)
            return
        data = await state.get_data()
        file_ids = data.get("model_generated_file_ids", [])
        if data.get("model_batch_id") != batch_id or not 0 <= index < len(file_ids):
            await callback.answer("Эта серия уже закрыта", show_alert=True)
            return
        if not callback.message:
            await callback.answer()
            return

        selected_file_id = file_ids[index]
        await callback.answer("Фотография выбрана")
        waiting = await callback.message.answer(
            "Анализирую выбранную фотографию и пишу текст поста..."
        )
        buffer = io.BytesIO()
        try:
            await bot.download(selected_file_id, destination=buffer)
        except Exception:
            logger.exception("Не удалось скачать выбранное фото")
            await waiting.edit_text(
                "Не удалось скачать фотографию из Telegram. Выберите ее еще раз."
            )
            return
        await prepare_post_draft(
            message=callback.message,
            state=state,
            waiting=waiting,
            photo_file_id=selected_file_id,
            image_bytes=buffer.getvalue(),
            mime_type="image/jpeg",
        )

    @router.message(F.text == "Создать пост")
    async def create_post_hint(message: Message, state: FSMContext) -> None:
        if not await is_admin_message(message, config):
            return
        await state.clear()
        repository.clear_active_draft(message.chat.id)
        await message.answer("Отправьте фотографию будущего поста.")

    @router.message(F.photo)
    async def receive_photo(message: Message, state: FSMContext, bot: Bot) -> None:
        if not await is_admin_message(message, config):
            return
        await state.clear()
        repository.clear_active_draft(message.chat.id)
        waiting = await message.answer("Анализирую изображение и пишу текст...")
        photo = message.photo[-1]
        buffer = io.BytesIO()
        try:
            await bot.download(photo, destination=buffer)
        except Exception:
            logger.exception("Не удалось скачать изображение")
            await waiting.edit_text(
                "Не удалось скачать изображение из Telegram. Повторите отправку."
            )
            return
        await prepare_post_draft(
            message=message,
            state=state,
            waiting=waiting,
            photo_file_id=photo.file_id,
            image_bytes=buffer.getvalue(),
            mime_type="image/jpeg",
        )

    @router.callback_query(F.data.startswith("preset:use:"))
    async def use_preset(callback: CallbackQuery, state: FSMContext) -> None:
        if not await is_admin_callback(callback, config):
            return
        chat_id = callback.message.chat.id if callback.message else callback.from_user.id
        data = await restore_active_draft(state, chat_id)
        if not required_draft_fields.issubset(data):
            await callback.answer("Черновик уже закрыт", show_alert=True)
            return
        preset_id = int(callback.data.rsplit(":", 1)[1])
        preset = repository.get_preset(preset_id)
        if not preset:
            await callback.answer("Пресет удален", show_alert=True)
            return
        await state.update_data(size=preset.size, price=preset.price)
        await save_active_draft(state, chat_id)
        await callback.answer(f"Выбран: {preset.name}")
        if callback.message:
            await ask_for_time(callback.message, config)

    @router.callback_query(F.data == "preset:manual")
    async def manual_product_settings(
        callback: CallbackQuery, state: FSMContext
    ) -> None:
        if not await is_admin_callback(callback, config):
            return
        chat_id = callback.message.chat.id if callback.message else callback.from_user.id
        data = await restore_active_draft(state, chat_id)
        if not required_draft_fields.issubset(data):
            await callback.answer("Черновик уже закрыт", show_alert=True)
            return
        await state.set_state(DraftStates.waiting_size)
        await callback.answer()
        if callback.message:
            await callback.message.answer(
                "Выберите доступные размеры:", reply_markup=size_keyboard()
            )

    @router.callback_query(F.data.startswith("size:"))
    async def choose_size(callback: CallbackQuery, state: FSMContext) -> None:
        if not await is_admin_callback(callback, config):
            return
        chat_id = callback.message.chat.id if callback.message else callback.from_user.id
        data = await restore_active_draft(state, chat_id)
        required = required_draft_fields
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
        await save_active_draft(state, chat_id)
        await callback.answer(f"Размеры: {value}")
        if not callback.message:
            return
        if "scheduled_at_utc" in data:
            await send_preview(callback.message, state, config, template_store)
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
        await save_active_draft(state, message.chat.id)
        if "scheduled_at_utc" in data:
            await send_preview(message, state, config, template_store)
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
        await save_active_draft(state, message.chat.id)
        await ask_for_time(message, config)

    @router.callback_query(F.data.startswith("time:"))
    async def choose_time(callback: CallbackQuery, state: FSMContext) -> None:
        if not await is_admin_callback(callback, config):
            return
        chat_id = callback.message.chat.id if callback.message else callback.from_user.id
        data = await restore_active_draft(state, chat_id)
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
        await save_active_draft(state, chat_id)
        await callback.answer()
        if callback.message:
            await send_preview(callback.message, state, config, template_store)

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
        await save_active_draft(state, message.chat.id)
        await send_preview(message, state, config, template_store)

    @router.callback_query(F.data == "draft:text")
    async def edit_text(callback: CallbackQuery, state: FSMContext) -> None:
        if not await is_admin_callback(callback, config):
            return
        chat_id = callback.message.chat.id if callback.message else callback.from_user.id
        data = await restore_active_draft(state, chat_id)
        if not required_draft_fields.issubset(data):
            await callback.answer("Черновик уже закрыт", show_alert=True)
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
        if not title or len(title) > 100:
            await message.answer("Название должно содержать от 1 до 100 символов.")
            return
        garment_type, design_name = split_product_title(title)
        await state.update_data(
            title=title,
            garment_type=garment_type,
            design_name=design_name,
        )
        await save_active_draft(state, message.chat.id)
        await send_preview(message, state, config, template_store)

    @router.callback_query(F.data == "draft:description")
    async def edit_description(callback: CallbackQuery, state: FSMContext) -> None:
        if not await is_admin_callback(callback, config):
            return
        chat_id = callback.message.chat.id if callback.message else callback.from_user.id
        data = await restore_active_draft(state, chat_id)
        if not required_draft_fields.issubset(data):
            await callback.answer("Черновик уже закрыт", show_alert=True)
            return
        await state.set_state(DraftStates.waiting_description_edit)
        await callback.answer()
        if callback.message:
            await callback.message.answer("Отправьте новое короткое описание.")

    @router.message(DraftStates.waiting_description_edit, F.text)
    async def receive_description_edit(message: Message, state: FSMContext) -> None:
        if not await is_admin_message(message, config):
            return
        description = " ".join(message.text.strip().split())
        if not description or len(description) > 200:
            await message.answer("Описание должно содержать от 1 до 200 символов.")
            return
        await state.update_data(description=description)
        await save_active_draft(state, message.chat.id)
        await send_preview(message, state, config, template_store)

    @router.callback_query(F.data == "draft:size")
    async def edit_size(callback: CallbackQuery, state: FSMContext) -> None:
        if not await is_admin_callback(callback, config):
            return
        chat_id = callback.message.chat.id if callback.message else callback.from_user.id
        data = await restore_active_draft(state, chat_id)
        if not required_draft_fields.issubset(data):
            await callback.answer("Черновик уже закрыт", show_alert=True)
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
        chat_id = callback.message.chat.id if callback.message else callback.from_user.id
        data = await restore_active_draft(state, chat_id)
        if not required_draft_fields.issubset(data):
            await callback.answer("Черновик уже закрыт", show_alert=True)
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
        await save_active_draft(state, message.chat.id)
        await send_preview(message, state, config, template_store)

    @router.callback_query(F.data == "draft:time")
    async def edit_time(callback: CallbackQuery, state: FSMContext) -> None:
        if not await is_admin_callback(callback, config):
            return
        chat_id = callback.message.chat.id if callback.message else callback.from_user.id
        data = await restore_active_draft(state, chat_id)
        if not required_draft_fields.issubset(data):
            await callback.answer("Черновик уже закрыт", show_alert=True)
            return
        await callback.answer()
        if callback.message:
            await ask_for_time(callback.message, config)

    @router.callback_query(F.data == "draft:cancel")
    async def cancel_draft(callback: CallbackQuery, state: FSMContext) -> None:
        if not await is_admin_callback(callback, config):
            return
        await state.clear()
        chat_id = callback.message.chat.id if callback.message else callback.from_user.id
        repository.clear_active_draft(chat_id)
        await callback.answer("Черновик удален")
        if callback.message:
            await callback.message.answer("Создание поста отменено.")

    @router.callback_query(F.data == "draft:confirm")
    async def confirm_draft(callback: CallbackQuery, state: FSMContext) -> None:
        if not await is_admin_callback(callback, config):
            return
        chat_id = callback.message.chat.id if callback.message else callback.from_user.id
        data = await restore_active_draft(state, chat_id)
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
            repository.clear_active_draft(chat_id)
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
        repository.clear_active_draft(chat_id)
        await callback.answer("Готово")
        if callback.message:
            await callback.message.edit_reply_markup(reply_markup=None)

        if scheduled_at <= datetime.now(UTC):
            success = await publisher.publish_one(post_id)
            if callback.message and not success:
                await callback.message.answer(
                    f"Пост #{post_id} пока не опубликован. Бот повторит попытку."
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
