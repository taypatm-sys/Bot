import asyncio
import io
import logging
import os
import platform
import sys
import time
import traceback
from typing import Any, Dict, Optional

from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import (
    Message,
    CallbackQuery,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    ReplyKeyboardMarkup,
    KeyboardButton,
    BufferedInputFile,
)

from app.config import Config
from app.storage import PostRepository
from app.progress import ProgressBar
from app.image_fetcher import fetch_image_from_url, search_web_images

logger = logging.getLogger(__name__)
admin_router = Router(name="admin_control")

START_TIME = time.time()


def is_admin_user(user_id: int, config: Optional[Config] = None) -> bool:
    """Проверяет, является ли пользователь администратором бота."""
    if not config:
        return False
    allowed = {config.admin_telegram_id}
    if config.admin_telegram_ids:
        allowed.update(config.admin_telegram_ids)
    return user_id in allowed


def get_admin_keyboard() -> InlineKeyboardMarkup:
    """Генерирует главную панель управления бота."""
    kb = [
        [
            InlineKeyboardButton(text="📊 Статус системы", callback_query_data="admin:sysinfo"),
            InlineKeyboardButton(text="🔘 Управление кнопками", callback_query_data="admin:buttons_list"),
        ],
        [
            InlineKeyboardButton(text="🖼 Поиск картинки", callback_query_data="admin:search_photo_prompt"),
            InlineKeyboardButton(text="⏱ Демо прогресс-бара", callback_query_data="admin:demo_progress"),
        ],
        [
            InlineKeyboardButton(text="❓ Помощь по командам", callback_query_data="admin:help"),
        ]
    ]
    return InlineKeyboardMarkup(inline_keyboard=kb)


@admin_router.message(Command("admin"))
async def cmd_admin_panel(message: Message, config: Optional[Config] = None, **kwargs):
    if not message.from_user or not is_admin_user(message.from_user.id, config):
        await message.reply("⛔ Доступ запрещен. Эта команда только для администраторов.")
        return

    text = (
        "⚙️ <b>Панель управления администратора</b>\n\n"
        "Вы можете управлять ботом, добавлять кнопки, исполнять код и загружать фото из интернета."
    )
    await message.reply(text, reply_markup=get_admin_keyboard(), parse_mode="HTML")


@admin_router.callback_query(F.data.startswith("admin:"))
async def handle_admin_callback(
    callback: CallbackQuery,
    config: Optional[Config] = None,
    repository: Optional[PostRepository] = None,
    **kwargs,
):
    if not callback.from_user or not is_admin_user(callback.from_user.id, config):
        await callback.answer("⛔ Нет доступа", show_alert=True)
        return

    data = callback.data
    if data == "admin:sysinfo":
        info_text = await generate_sysinfo_text(repository)
        await callback.message.edit_text(info_text, reply_markup=get_admin_keyboard(), parse_mode="HTML")
        await callback.answer()

    elif data == "admin:buttons_list":
        buttons = repository.get_custom_buttons() if repository else []
        if not buttons:
            text = "🔘 <b>Список кнопок пуст.</b>\n\nЧтобы добавить кнопку, используйте:\n<code>/addbutton Название | Ответный текст или ссылка</code>"
        else:
            text = "🔘 <b>Настроенные кастомные кнопки:</b>\n\n"
            for idx, b in enumerate(buttons, 1):
                text += f"{idx}. <b>{b['name']}</b> -> <code>{b['payload']}</code>\n"
            text += "\nУдалить кнопку: <code>/delbutton Название</code>"
        
        await callback.message.edit_text(text, reply_markup=get_admin_keyboard(), parse_mode="HTML")
        await callback.answer()

    elif data == "admin:demo_progress":
        await callback.answer("Запуск индикатора прогресса...")
        msg = await callback.message.answer("⏳ Запуск тестового процесса...")
        pb = ProgressBar(total=10, min_update_interval=0.8)
        for i in range(1, 11):
            await asyncio.sleep(0.7)
            await pb.update_message(msg, current=i, prefix="⚡ <b>Тестирование процесса...</b>")
        await msg.edit_text("✅ <b>Процесс успешно завершен!</b>", parse_mode="HTML")

    elif data == "admin:search_photo_prompt":
        text = (
            "🖼 <b>Поиск и отправка фото из интернета</b>\n\n"
            "Используйте команду:\n"
            "<code>/findphoto запрос</code> — найти фото в интернете по названию\n"
            "<code>/sendphoto URL</code> — отправить фото прямо по ссылке"
        )
        await callback.message.edit_text(text, reply_markup=get_admin_keyboard(), parse_mode="HTML")
        await callback.answer()

    elif data == "admin:help":
        help_text = (
            "📖 <b>Команды управления из чата Telegram:</b>\n\n"
            "• <code>/admin</code> — Открыть панель управления\n"
            "• <code>/sysinfo</code> — Состояние сервера и базы данных\n"
            "• <code>/addbutton Кнопка | Ответ</code> — Добавить кнопку в меню\n"
            "• <code>/delbutton Кнопка</code> — Удалить кнопку\n"
            "• <code>/buttons</code> — Список всех активных кнопок\n"
            "• <code>/findphoto Запрос</code> — Найти и прислать фото из сети\n"
            "• <code>/sendphoto URL</code> — Загрузить фото по ссылке\n"
            "• <code>/exec Python_код</code> — Исполнить Python код на сервере\n"
            "• <code>/eval Выражение</code> — Вычислить Python выражение\n"
            "• <code>/cmd Консольная_команда</code> — Запустить консольную команду в ОС\n"
        )
        await callback.message.edit_text(help_text, reply_markup=get_admin_keyboard(), parse_mode="HTML")
        await callback.answer()


async def generate_sysinfo_text(repository: Optional[PostRepository] = None) -> str:
    uptime_sec = int(time.time() - START_TIME)
    mins, secs = divmod(uptime_sec, 60)
    hours, mins = divmod(mins, 60)
    days, hours = divmod(hours, 24)
    uptime_str = f"{days}д {hours}ч {mins}м {secs}с"

    db_status = "Не подключено"
    if repository:
        db_status = "PostgreSQL" if repository.database_url else "SQLite"

    return (
        "📊 <b>Системная информация бота</b>\n\n"
        f"🖥 <b>ОС:</b> {platform.system()} {platform.release()} ({platform.machine()})\n"
        f"🐍 <b>Python:</b> {platform.python_version()}\n"
        f"⏱ <b>Время работы:</b> {uptime_str}\n"
        f"🗄 <b>База данных:</b> {db_status}\n"
        f"📁 <b>Рабочая директория:</b> <code>{os.getcwd()}</code>"
    )


@admin_router.message(Command("sysinfo"))
async def cmd_sysinfo(
    message: Message,
    config: Optional[Config] = None,
    repository: Optional[PostRepository] = None,
    **kwargs,
):
    if not message.from_user or not is_admin_user(message.from_user.id, config):
        await message.reply("⛔ Доступ запрещен.")
        return
    text = await generate_sysinfo_text(repository)
    await message.reply(text, parse_mode="HTML")


@admin_router.message(Command("exec"))
async def cmd_exec_code(message: Message, config: Optional[Config] = None, **kwargs):
    if not message.from_user or not is_admin_user(message.from_user.id, config):
        await message.reply("⛔ Доступ запрещен.")
        return

    code = message.text.partition(" ")[2].strip() if message.text else ""
    if not code:
        await message.reply("⚠️ Укажите Python-код для выполнения:\n<code>/exec print(2 + 2)</code>", parse_mode="HTML")
        return

    # Очистка markdown блоков если отправлен код ```python ... ```
    if code.startswith("```"):
        lines = code.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```"):
            code = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

    status_msg = await message.reply("⚡ Выполнение кода...")

    # Перехватываем stdout и stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    redirected_output = sys.stdout = io.StringIO()
    redirected_error = sys.stderr = io.StringIO()

    stdout_val = ""
    stderr_val = ""
    result_val = None
    error_val = None

    try:
        # Создаем обертку для поддержки async/await функций
        exec_globals = {
            "asyncio": asyncio,
            "sys": sys,
            "os": os,
            "message": message,
            "config": config,
        }
        exec_locals = {}
        
        # Если в коде есть await, заворачиваем в async функцию
        if "await " in code:
            wrapped_code = f"async def __exec_func():\n" + "\n".join(f"    {line}" for line in code.splitlines())
            exec(wrapped_code, exec_globals, exec_locals)
            result_val = await exec_locals["__exec_func"]()
        else:
            exec(code, exec_globals, exec_locals)

        stdout_val = redirected_output.getvalue()
        stderr_val = redirected_error.getvalue()
    except Exception as exc:
        error_val = traceback.format_exc()
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    res_parts = ["✅ <b>Результат выполнения:</b>\n"]
    if stdout_val:
        res_parts.append(f"<b>Output:</b>\n<pre>{stdout_val[:2000]}</pre>")
    if stderr_val:
        res_parts.append(f"<b>Stderr:</b>\n<pre>{stderr_val[:1000]}</pre>")
    if result_val is not None:
        res_parts.append(f"<b>Return:</b>\n<code>{result_val}</code>")
    if error_val:
        res_parts.append(f"❌ <b>Ошибка:</b>\n<pre>{error_val[:2500]}</pre>")

    if len(res_parts) == 1:
        res_parts.append("<i>Код выполнен без вывода в консоль.</i>")

    await status_msg.edit_text("\n\n".join(res_parts), parse_mode="HTML")


@admin_router.message(Command("eval"))
async def cmd_eval_expr(message: Message, config: Optional[Config] = None, **kwargs):
    if not message.from_user or not is_admin_user(message.from_user.id, config):
        await message.reply("⛔ Доступ запрещен.")
        return

    expr = message.text.partition(" ")[2].strip() if message.text else ""
    if not expr:
        await message.reply("⚠️ Укажите выражение для оценки:\n<code>/eval 10 * 25</code>", parse_mode="HTML")
        return

    try:
        res = eval(expr)
        await message.reply(f"🧮 <b>Результат:</b>\n<code>{res}</code>", parse_mode="HTML")
    except Exception as exc:
        await message.reply(f"❌ <b>Ошибка при вычислении:</b>\n<pre>{exc}</pre>", parse_mode="HTML")


@admin_router.message(Command("cmd"))
async def cmd_shell_command(message: Message, config: Optional[Config] = None, **kwargs):
    if not message.from_user or not is_admin_user(message.from_user.id, config):
        await message.reply("⛔ Доступ запрещен.")
        return

    command_str = message.text.partition(" ")[2].strip() if message.text else ""
    if not command_str:
        await message.reply("⚠️ Укажите консольную команду:\n<code>/cmd git status</code>", parse_mode="HTML")
        return

    status_msg = await message.reply(f"💻 Запуск команды: <code>{command_str}</code>...", parse_mode="HTML")

    try:
        proc = await asyncio.create_subprocess_shell(
            command_str,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        out_str = stdout.decode("utf-8", errors="replace").strip()
        err_str = stderr.decode("utf-8", errors="replace").strip()

        res_text = f"💻 <b>Команда:</b> <code>{command_str}</code> (Код: {proc.returncode})\n\n"
        if out_str:
            res_text += f"<b>STDOUT:</b>\n<pre>{out_str[:3000]}</pre>\n"
        if err_str:
            res_text += f"<b>STDERR:</b>\n<pre>{err_str[:1500]}</pre>"

        await status_msg.edit_text(res_text, parse_mode="HTML")
    except Exception as exc:
        await status_msg.edit_text(f"❌ <b>Ошибка при запуске команды:</b>\n<pre>{exc}</pre>", parse_mode="HTML")


@admin_router.message(Command("addbutton"))
async def cmd_add_button(
    message: Message,
    config: Optional[Config] = None,
    repository: Optional[PostRepository] = None,
    **kwargs,
):
    if not message.from_user or not is_admin_user(message.from_user.id, config):
        await message.reply("⛔ Доступ запрещен.")
        return

    raw = message.text.partition(" ")[2].strip() if message.text else ""
    if "|" not in raw:
        await message.reply(
            "⚠️ <b>Формат команды:</b>\n"
            "<code>/addbutton Название кнопки | Ответный текст или ссылка</code>\n\n"
            "Пример:\n"
            "<code>/addbutton Контакты | Наш менеджер: @admin</code>\n"
            "<code>/addbutton Сайт | https://example.com</code>",
            parse_mode="HTML"
        )
        return

    name, _, payload = raw.partition("|")
    name = name.strip()
    payload = payload.strip()

    if not name or not payload:
        await message.reply("⚠️ Имя кнопки и ответный текст не должны быть пустыми.")
        return

    if repository:
        repository.add_custom_button(name, payload)
    await message.reply(f"✅ <b>Кнопка «{name}» успешно добавлена!</b>\nОтвет: <code>{payload}</code>", parse_mode="HTML")


@admin_router.message(Command("delbutton"))
async def cmd_del_button(
    message: Message,
    config: Optional[Config] = None,
    repository: Optional[PostRepository] = None,
    **kwargs,
):
    if not message.from_user or not is_admin_user(message.from_user.id, config):
        await message.reply("⛔ Доступ запрещен.")
        return

    name = message.text.partition(" ")[2].strip() if message.text else ""
    if not name:
        await message.reply("⚠️ Укажите название кнопки для удаления:\n<code>/delbutton Название</code>", parse_mode="HTML")
        return

    success = repository.delete_custom_button(name) if repository else False
    if success:
        await message.reply(f"✅ Кнопка «{name}» удалена.", parse_mode="HTML")
    else:
        await message.reply(f"⚠️ Кнопка с названием «{name}» не найдена.", parse_mode="HTML")


@admin_router.message(Command("buttons"))
async def cmd_list_buttons(
    message: Message,
    config: Optional[Config] = None,
    repository: Optional[PostRepository] = None,
    **kwargs,
):
    if not message.from_user or not is_admin_user(message.from_user.id, config):
        await message.reply("⛔ Доступ запрещен.")
        return

    buttons = repository.get_custom_buttons() if repository else []
    if not buttons:
        await message.reply("🔘 <b>Список кнопок пуст.</b>\nДобавить: <code>/addbutton Имя | Ответ</code>", parse_mode="HTML")
        return

    text = "🔘 <b>Список активных кнопок:</b>\n\n"
    for idx, b in enumerate(buttons, 1):
        text += f"{idx}. <b>{b['name']}</b> -> <code>{b['payload']}</code>\n"
    text += "\nУдалить: <code>/delbutton Название</code>"
    await message.reply(text, parse_mode="HTML")


@admin_router.message(Command("sendphoto"))
async def cmd_send_photo_url(message: Message, config: Optional[Config] = None, **kwargs):
    if not message.from_user or not is_admin_user(message.from_user.id, config):
        await message.reply("⛔ Доступ запрещен.")
        return

    parts = message.text.split(maxsplit=2) if message.text else []
    if len(parts) < 2:
        await message.reply("⚠️ Укажите URL изображения:\n<code>/sendphoto https://example.com/image.jpg [Подпись]</code>", parse_mode="HTML")
        return

    url = parts[1].strip()
    caption = parts[2].strip() if len(parts) > 2 else ""

    status_msg = await message.reply("📥 Скачивание фото из интернета...")
    pb = ProgressBar(total=100)
    await pb.update_message(status_msg, current=30, prefix="📥 <b>Загрузка картинки...</b>", force=True)

    try:
        image_bytes, filename = await fetch_image_from_url(url)
        await pb.update_message(status_msg, current=80, prefix="📤 <b>Отправка в чат...</b>", force=True)

        input_file = BufferedInputFile(image_bytes, filename=filename)
        await message.reply_photo(photo=input_file, caption=caption or f"🖼 Загружено с:\n<code>{url}</code>", parse_mode="HTML")
        await status_msg.delete()
    except Exception as exc:
        await status_msg.edit_text(f"❌ <b>Не удалось скачать или отправить фото:</b>\n<pre>{exc}</pre>", parse_mode="HTML")


@admin_router.message(Command("findphoto"))
async def cmd_find_photo(message: Message, config: Optional[Config] = None, **kwargs):
    if not message.from_user or not is_admin_user(message.from_user.id, config):
        await message.reply("⛔ Доступ запрещен.")
        return

    query = message.text.partition(" ")[2].strip() if message.text else ""
    if not query:
        await message.reply("⚠️ Укажите запрос для поиска фото:\n<code>/findphoto красная футболка</code>", parse_mode="HTML")
        return

    status_msg = await message.reply(f"🔍 Поиск фото по запросу «<b>{query}</b>» в интернете...", parse_mode="HTML")
    pb = ProgressBar(total=100)
    await pb.update_message(status_msg, current=25, prefix=f"🔍 <b>Поиск фото: {query}...</b>", force=True)

    image_urls = await search_web_images(query, limit=3)
    if not image_urls:
        await status_msg.edit_text("⚠️ Не удалось найти подходящие фото.")
        return

    await pb.update_message(status_msg, current=60, prefix="📥 <b>Загрузка найденного фото...</b>", force=True)

    sent = False
    for url in image_urls:
        try:
            image_bytes, filename = await fetch_image_from_url(url)
            await pb.update_message(status_msg, current=95, prefix="📤 <b>Отправка фото в чат...</b>", force=True)
            input_file = BufferedInputFile(image_bytes, filename=filename)
            await message.reply_photo(
                photo=input_file,
                caption=f"🖼 <b>Результат поиска по запросу:</b> «{query}»",
                parse_mode="HTML"
            )
            sent = True
            break
        except Exception as exc:
            logger.warning("Пропуск URL %s из-за ошибки: %s", url, exc)
            continue

    if sent:
        await status_msg.delete()
    else:
        await status_msg.edit_text("❌ Не удалось загрузить ни одно из найденных фото.")


async def _custom_button_filter(message: Message, repository: Optional[PostRepository] = None, **kwargs) -> bool:
    if not message.text or not repository:
        return False
    text = message.text.strip().casefold()
    buttons = repository.get_custom_buttons()
    return any(b.get("name", "").strip().casefold() == text for b in buttons)


@admin_router.message(_custom_button_filter)
async def handle_custom_button_click(
    message: Message,
    repository: Optional[PostRepository] = None,
    **kwargs,
):
    if not message.text or not repository:
        return
    text = message.text.strip().casefold()
    buttons = repository.get_custom_buttons()
    for b in buttons:
        if b.get("name", "").strip().casefold() == text:
            payload = b.get("payload", "").strip()
            if payload.startswith(("http://", "https://")):
                await message.reply(f"🔗 <a href='{payload}'>{b['name']}</a>", parse_mode="HTML")
            else:
                await message.reply(payload, parse_mode="HTML")
            return
