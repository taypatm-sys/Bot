import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from google import genai
from google.genai import types

from app.config import DEFAULT_GEMINI_MODEL, normalize_gemini_model
from app.storage import PostRepository

logger = logging.getLogger(__name__)


@dataclass
class IntentResult:
    intent: str
    parameters: dict[str, Any]
    response_text: str
    clarification_question: str = ""


SYSTEM_PROMPT = """Ты — интеллектуальный ассистент Telegram-бота Taypa, управляющего созданием постов одежды, поиском фото и администрированием бота.
Твоя задача — определить намерение (intent) и извлечь параметры из обычного человеческого языка пользователя, чтобы пользователю НЕ нужно было запоминать спец-команды.

Возможные значения intent:
- "admin_panel": пользователь просит открыть админку, панель управления ("открой админку", "покажи админ панель", "администрирование")
- "sysinfo": запрос статуса сервера, нагрузки, версии, аптайма ("как там сервер", "статус системы", "системная инфо", "аптайм")
- "find_photo": просит найти фото/картинку в интернете ("найди фото красной футболки", "найди картинку с котом", "поищи в сети фото ...") -> параметры: {"query": "запрос для поиска"}
- "send_photo_url": просит прислать/скачать фото по URL адресу -> параметры: {"url": "https://...", "caption": "подпись"}
- "add_button": просит добавить/создать новую кнопку -> параметры: {"name": "название кнопки", "payload": "ответный текст или ссылка"}
- "delete_button": просит удалить кнопку -> параметры: {"name": "название кнопки"}
- "list_buttons": просит показать список всех кнопок -> параметры: {}
- "exec_code": просит выполнить Python код -> параметры: {"code": "код"}
- "exec_cmd": просит выполнить консольную команду ОС -> параметры: {"command": "команда"}
- "switch_to_model": просит сделать макет/фото на модели ("сделай на модели", "перенеси на модель", "покажи на модели")
- "search_references": просит найти/подгрузить новые референсы одежды
- "show_queue": запрос очереди постов
- "show_settings": открыть настройки бота
- "show_status": запрос статуса каталога референсов или бота
- "ask_clarification": запрос неясен, нужно задать уточняющий вопрос
- "general_chat": общение, ответ на вопрос или приветствие

Верни строго JSON объект в формате:
{
  "intent": "название_интента",
  "parameters": {"ключ": "значение"},
  "response_text": "краткий вежливый ответ на русском языке",
  "clarification_question": ""
}
"""


class AIAssistant:
    def __init__(
        self,
        api_key: str,
        repository: PostRepository,
        model: str = DEFAULT_GEMINI_MODEL,
    ) -> None:
        self.api_key = api_key
        self.repository = repository
        self.model = normalize_gemini_model(model)
        self._client: Optional[genai.Client] = None

    def _get_client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    async def analyze_message(
        self,
        chat_id: int,
        user_text: str,
        current_state: str = "",
        active_draft: Optional[dict] = None,
    ) -> IntentResult:
        if not user_text.strip():
            return IntentResult(
                intent="general_chat",
                parameters={},
                response_text="Отправьте фотографию вещи или выберите действие в меню.",
            )

        text_lower = user_text.strip().casefold()

        # Быстрое распознавание естественных русскоязычных фраз без запроса к API
        if any(w in text_lower for w in ("админка", "админ панель", "панель управления", "открой админку")):
            return IntentResult(intent="admin_panel", parameters={}, response_text="⚙️ Открываю панель управления...")

        if any(w in text_lower for w in ("статус сервера", "системная инфо", "системная информация", "состояние сервера", "аптайм")):
            return IntentResult(intent="sysinfo", parameters={}, response_text="📊 Формирую системный отчет...")

        if any(w in text_lower for w in ("список кнопок", "мои кнопки", "покажи кнопки", "кастомные кнопки")):
            return IntentResult(intent="list_buttons", parameters={}, response_text="🔘 Загружаю список кнопок...")

        # Поиск фото: "найди фото <запрос>", "поищи фото <запрос>", "найди картинку <запрос>"
        match_photo = re.search(r"(?:найди|поищи|скачай|пришли)\s+(?:мне\s+)?(?:фото|картинку|изображение)\s+(.+)", text_lower)
        if match_photo:
            query = match_photo.group(1).strip()
            return IntentResult(
                intent="find_photo",
                parameters={"query": query},
                response_text=f"🔍 Поищу фото по запросу: «{query}»...",
            )

        # Скачивание по URL: "отправь фото https://..."
        match_url = re.search(r"(https?://\S+)", user_text)
        if match_url and any(w in text_lower for w in ("фото", "картинк", "изображени", "скачай", "отправь", "загрузи")):
            url = match_url.group(1).strip()
            return IntentResult(
                intent="send_photo_url",
                parameters={"url": url},
                response_text=f"📥 Скачиваю картинку по ссылке...",
            )

        # Добавление кнопки: "добавь кнопку <имя> | <ответ>" или "добавь кнопку <имя> с текстом <ответ>"
        match_add_btn = re.search(r"добавь\s+кнопку\s+([^|]+)(?:\||\s+с\s+(?:текстом|ссылкой)\s+)(.+)", text_lower)
        if match_add_btn:
            name = match_add_btn.group(1).strip()
            payload = match_add_btn.group(2).strip()
            return IntentResult(
                intent="add_button",
                parameters={"name": name, "payload": payload},
                response_text=f"➕ Добавляю кнопку «{name}»...",
            )

        # Удаление кнопки: "удали кнопку <имя>"
        match_del_btn = re.search(r"(?:удали|убери)\s+кнопку\s+(.+)", text_lower)
        if match_del_btn:
            name = match_del_btn.group(1).strip()
            return IntentResult(
                intent="delete_button",
                parameters={"name": name},
                response_text=f"🗑 Удаляю кнопку «{name}»...",
            )

        if any(phrase in text_lower for phrase in ("на модели", "сделай на модели", "сделать этот макет на модели", "перенести на модель", "покажи на модели", "макет на модели")):
            return IntentResult(
                intent="switch_to_model",
                parameters={},
                response_text="💡 Переключаю загруженное фото на создание макета на модели...",
            )

        # Нейросетевой разбор естественного языка через Gemini AI
        history = self.repository.get_chat_history(chat_id)
        history.append({"role": "user", "content": user_text})

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Текущее состояние диалога: {current_state or 'главное меню'}\n"
            f"История сообщений:\n"
            + "\n".join(f"{msg['role']}: {msg['content']}" for msg in history[-5:])
            + f"\n\nПоследнее сообщение пользователя: {user_text}\nJSON:"
        )

        try:
            client = self._get_client()
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                ),
            )
            raw_json = response.text or "{}"
            data = json.loads(raw_json)

            intent = data.get("intent", "general_chat")
            parameters = data.get("parameters", {})
            response_text = data.get("response_text", "Понял вас.")
            clarification = data.get("clarification_question", "")

            history.append({"role": "assistant", "content": response_text})
            self.repository.save_chat_history(chat_id, history)

            return IntentResult(
                intent=intent,
                parameters=parameters,
                response_text=response_text,
                clarification_question=clarification,
            )
        except Exception as error:
            logger.warning("Ошибка AI Assistant Gemini: %s", error)
            return IntentResult(
                intent="general_chat",
                parameters={},
                response_text="Я вас понял. Вы можете написать что вам требуется (например: 'найди фото красной футболки', 'покажи админку', 'добавь кнопку Сайт | https://...').",
            )
