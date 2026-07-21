import os
import shutil
from dataclasses import dataclass
from datetime import timedelta, timezone as fixed_timezone, tzinfo
from pathlib import Path
from typing import Union
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv


class ConfigError(RuntimeError):
    pass


ChatId = Union[int, str]
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"
LEGACY_GEMINI_MODELS = {
    "gemini-2.5-flash",
    "models/gemini-2.5-flash",
}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUNDLED_CAPTION_TEMPLATE = PROJECT_ROOT / "caption_template.txt"


def normalize_gemini_model(value: str) -> str:
    model = value.strip()
    if not model or model in LEGACY_GEMINI_MODELS:
        return DEFAULT_GEMINI_MODEL
    return model


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(f"в .env не заполнено поле {name}")
    return value


def _parse_channel_id(value: str) -> ChatId:
    if value.lstrip("-").isdigit():
        return int(value)
    if value.startswith("@") and len(value) > 1:
        return value
    raise ConfigError("CHANNEL_ID должен быть вида @channel_name или -1001234567890")


def _load_timezone(name: str) -> tzinfo:
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError as error:
        # Turkmenistan has used UTC+5 without daylight saving time since 1991.
        # This fallback keeps the Windows version working even if its timezone
        # database is missing or was not installed correctly.
        if name == "Asia/Ashgabat":
            return fixed_timezone(timedelta(hours=5), name="Asia/Ashgabat")
        raise ConfigError(f"неизвестный часовой пояс {name}") from error


@dataclass(frozen=True)
class Config:
    telegram_bot_token: str
    gemini_api_key: str
    admin_telegram_id: int
    channel_id: ChatId
    contact_username: str
    timezone_name: str
    gemini_model: str
    button_text: str
    copy_language: str
    database_path: Path
    caption_template_path: Path

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv()
        admin_value = _required("ADMIN_TELEGRAM_ID")
        if not admin_value.isdigit():
            raise ConfigError("ADMIN_TELEGRAM_ID должен состоять только из цифр")

        timezone_name = os.getenv("TIMEZONE", "Asia/Ashgabat").strip()
        _load_timezone(timezone_name)

        contact_username = _required("CONTACT_USERNAME").lstrip("@")
        if not contact_username.replace("_", "").isalnum():
            raise ConfigError("CONTACT_USERNAME указан неверно")

        return cls(
            telegram_bot_token=_required("TELEGRAM_BOT_TOKEN"),
            gemini_api_key=_required("GEMINI_API_KEY"),
            admin_telegram_id=int(admin_value),
            channel_id=_parse_channel_id(_required("CHANNEL_ID")),
            contact_username=contact_username,
            timezone_name=timezone_name,
            gemini_model=normalize_gemini_model(
                os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
            ),
            button_text=os.getenv("BUTTON_TEXT", "Написать").strip() or "Написать",
            copy_language=os.getenv("COPY_LANGUAGE", "ru").strip() or "ru",
            database_path=Path(os.getenv("DATABASE_PATH", "data/posts.sqlite3")),
            caption_template_path=Path(
                os.getenv("CAPTION_TEMPLATE_PATH", "caption_template.txt")
            ),
        )

    @property
    def timezone(self) -> tzinfo:
        return _load_timezone(self.timezone_name)

    def ensure_runtime_paths(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.caption_template_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.caption_template_path.exists():
            if BUNDLED_CAPTION_TEMPLATE.exists():
                shutil.copyfile(
                    BUNDLED_CAPTION_TEMPLATE,
                    self.caption_template_path,
                )
            else:
                raise ConfigError(
                    f"не найден шаблон подписи {self.caption_template_path}"
                )
