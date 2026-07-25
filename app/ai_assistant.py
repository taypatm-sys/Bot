import json
import logging
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


SYSTEM_PROMPT = """Ты — интеллектуальный ассистент Telegram-бота Taypa, управляющего созданием постов одежды и подбором фото-референсов.
Твоя задача — извлечь намерение (intent) и параметры из текста пользователя и сформулировать естественный ответ.

Возможные значения intent:
- "search_references": пользователь просит найти/подгрузить новые референсы (параметры: garment_type ("t-shirt", "hoodie", "cap", "jacket", и т.д.))
- "create_post": пользователь хочет создать пост или отправить макет
- "show_queue": запрос очереди постов
- "show_settings": открыть настройки или список администраторов
- "show_status": запрос статуса каталога референсов или бота
- "change_price": просит изменить цену (параметры: price)
- "change_size": просит изменить размер (параметры: size)
- "ask_clarification": запрос слишком неясен или содержит неоднозначности, нужно задать уточняющий вопрос
- "general_chat": вежливое общение, вопрос о возможностях или ответы на приветствие

Верни строго JSON объект в формате:
{
  "intent": "название_интента",
  "parameters": {"ключ": "значение"},
  "response_text": "краткий вежливый ответ на русском языке",
  "clarification_question": "уточняющий вопрос, если intent == ask_clarification"
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
    ) -> IntentResult:
        if not user_text.strip():
            return IntentResult(
                intent="general_chat",
                parameters={},
                response_text="Отправьте фотографию вещи или выберите действие в меню.",
            )

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
                response_text="Я вас понял. Вы можете использовать меню кнопок или переслать фото товара.",
            )
