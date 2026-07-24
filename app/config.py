import os
import shutil
import re
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
DEFAULT_GEMINI_IMAGE_MODEL = "gemini-3.1-flash-lite-image"
DEFAULT_GEMINI_IMAGE_SIZE = "1K"
DEFAULT_MOCKUP_VARIANTS = 1
LEGACY_GEMINI_MODELS = {
    "gemini-2.5-flash",
    "models/gemini-2.5-flash",
}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUNDLED_CAPTION_TEMPLATE = PROJECT_ROOT / "caption_template.txt"
BUNDLED_REFERENCE_SOURCES = PROJECT_ROOT / "reference_sources.txt"


def normalize_gemini_model(value: str) -> str:
    model = value.strip()
    if not model or model in LEGACY_GEMINI_MODELS:
        return DEFAULT_GEMINI_MODEL
    return model


def normalize_gemini_image_size(value: str) -> str:
    image_size = value.strip().upper() or DEFAULT_GEMINI_IMAGE_SIZE
    if image_size not in {"1K", "2K", "4K"}:
        raise ConfigError("GEMINI_IMAGE_SIZE должен быть 1K, 2K или 4K")
    return image_size


def _mockup_variants(value: str) -> int:
    try:
        variants = int(value)
    except ValueError as error:
        raise ConfigError("MOCKUP_VARIANTS должен быть целым числом") from error
    if not 1 <= variants <= 4:
        raise ConfigError("MOCKUP_VARIANTS должен быть от 1 до 4")
    return variants


def _positive_int(name: str, value: str, default: int) -> int:
    try:
        result = int(value.strip() or str(default))
    except ValueError as error:
        raise ConfigError(f"{name} должен быть целым числом") from error
    if result < 1:
        raise ConfigError(f"{name} должен быть больше нуля")
    return result


def _bool_env(name: str, value: str, default: bool = False) -> bool:
    clean = value.strip().casefold()
    if not clean:
        return default
    if clean in {"1", "true", "yes", "on", "да"}:
        return True
    if clean in {"0", "false", "no", "off", "нет"}:
        return False
    raise ConfigError(f"{name} должен быть true или false")


def _positive_float(name: str, value: str, default: float) -> float:
    try:
        result = float(value.strip() or str(default))
    except ValueError as error:
        raise ConfigError(f"{name} должен быть числом") from error
    if result <= 0:
        raise ConfigError(f"{name} должен быть больше нуля")
    return result


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(f"в .env не заполнено поле {name}")
    return value



def _parse_admin_ids(primary_value: str, extra_value: str) -> frozenset[int]:
    raw_values = [primary_value]
    raw_values.extend(re.split(r"[\s,;]+", extra_value.strip()))
    result: set[int] = set()
    for raw in raw_values:
        value = raw.strip()
        if not value:
            continue
        if not value.isdigit():
            raise ConfigError(
                "ADMIN_TELEGRAM_ID и ADMIN_TELEGRAM_IDS должны содержать только цифровые ID"
            )
        result.add(int(value))
    if not result:
        raise ConfigError("не указан ни один администратор")
    return frozenset(result)


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
    database_url: str = ""
    admin_telegram_ids: frozenset[int] = frozenset()
    gemini_image_model: str = DEFAULT_GEMINI_IMAGE_MODEL
    gemini_image_size: str = DEFAULT_GEMINI_IMAGE_SIZE
    mockup_variants: int = DEFAULT_MOCKUP_VARIANTS
    reference_sources_path: Path = BUNDLED_REFERENCE_SOURCES
    reference_import_delay_seconds: float = 5.0
    reference_idle_interval_seconds: float = 300.0
    reference_max_attempts: int = 5
    reference_min_pool_size: int = 20
    reference_analysis_timeout_seconds: float = 90.0
    mockup_analysis_timeout_seconds: float = 150.0
    reference_user_agent: str = "TaypaReferenceCatalog/5.3"
    pinterest_access_token: str = ""
    pinterest_search_enabled: bool = False
    pinterest_country_code: str = "US"
    pinterest_search_interval_seconds: float = 21600.0
    pinterest_target_pool_size: int = 160
    pinterest_queries_per_cycle: int = 2

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv()
        admin_value = _required("ADMIN_TELEGRAM_ID")
        admin_ids = _parse_admin_ids(
            admin_value, os.getenv("ADMIN_TELEGRAM_IDS", "")
        )

        timezone_name = os.getenv("TIMEZONE", "Asia/Ashgabat").strip()
        _load_timezone(timezone_name)

        contact_username = _required("CONTACT_USERNAME").lstrip("@")
        if not contact_username.replace("_", "").isalnum():
            raise ConfigError("CONTACT_USERNAME указан неверно")

        database_url = os.getenv("DATABASE_URL", "").strip()
        if database_url and not database_url.startswith(
            ("postgres://", "postgresql://")
        ):
            raise ConfigError("DATABASE_URL должен быть строкой подключения PostgreSQL")

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
            database_url=database_url,
            admin_telegram_ids=admin_ids,
            gemini_image_model=(
                os.getenv("GEMINI_IMAGE_MODEL", DEFAULT_GEMINI_IMAGE_MODEL).strip()
                or DEFAULT_GEMINI_IMAGE_MODEL
            ),
            gemini_image_size=normalize_gemini_image_size(
                os.getenv("GEMINI_IMAGE_SIZE", DEFAULT_GEMINI_IMAGE_SIZE)
            ),
            mockup_variants=_mockup_variants(
                os.getenv("MOCKUP_VARIANTS", str(DEFAULT_MOCKUP_VARIANTS))
            ),
            reference_sources_path=Path(
                os.getenv("REFERENCE_SOURCES_PATH", str(BUNDLED_REFERENCE_SOURCES))
            ),
            reference_import_delay_seconds=_positive_float(
                "REFERENCE_IMPORT_DELAY_SECONDS",
                os.getenv("REFERENCE_IMPORT_DELAY_SECONDS", "5"),
                5.0,
            ),
            reference_idle_interval_seconds=_positive_float(
                "REFERENCE_IDLE_INTERVAL_SECONDS",
                os.getenv("REFERENCE_IDLE_INTERVAL_SECONDS", "300"),
                300.0,
            ),
            reference_max_attempts=_positive_int(
                "REFERENCE_MAX_ATTEMPTS",
                os.getenv("REFERENCE_MAX_ATTEMPTS", "5"),
                5,
            ),
            reference_min_pool_size=_positive_int(
                "REFERENCE_MIN_POOL_SIZE",
                os.getenv("REFERENCE_MIN_POOL_SIZE", "20"),
                20,
            ),
            reference_analysis_timeout_seconds=_positive_float(
                "REFERENCE_ANALYSIS_TIMEOUT_SECONDS",
                os.getenv("REFERENCE_ANALYSIS_TIMEOUT_SECONDS", "90"),
                90.0,
            ),
            mockup_analysis_timeout_seconds=_positive_float(
                "MOCKUP_ANALYSIS_TIMEOUT_SECONDS",
                os.getenv("MOCKUP_ANALYSIS_TIMEOUT_SECONDS", "150"),
                150.0,
            ),
            reference_user_agent=(
                os.getenv("REFERENCE_USER_AGENT", "TaypaReferenceCatalog/5.3").strip()
                or "TaypaReferenceCatalog/5.3"
            ),
            pinterest_access_token=os.getenv("PINTEREST_ACCESS_TOKEN", "").strip(),
            pinterest_search_enabled=_bool_env(
                "PINTEREST_SEARCH_ENABLED",
                os.getenv("PINTEREST_SEARCH_ENABLED", "true"),
                default=True,
            ),
            pinterest_country_code=(
                os.getenv("PINTEREST_COUNTRY_CODE", "US").strip().upper() or "US"
            ),
            pinterest_search_interval_seconds=_positive_float(
                "PINTEREST_SEARCH_INTERVAL_SECONDS",
                os.getenv("PINTEREST_SEARCH_INTERVAL_SECONDS", "21600"),
                21600.0,
            ),
            pinterest_target_pool_size=_positive_int(
                "PINTEREST_TARGET_POOL_SIZE",
                os.getenv("PINTEREST_TARGET_POOL_SIZE", "160"),
                160,
            ),
            pinterest_queries_per_cycle=_positive_int(
                "PINTEREST_QUERIES_PER_CYCLE",
                os.getenv("PINTEREST_QUERIES_PER_CYCLE", "2"),
                2,
            ),
        )

    @property
    def admin_ids(self) -> frozenset[int]:
        return frozenset({self.admin_telegram_id}).union(self.admin_telegram_ids)

    @property
    def timezone(self) -> tzinfo:
        return _load_timezone(self.timezone_name)

    @property
    def database_source(self) -> Union[Path, str]:
        return self.database_url or self.database_path

    def ensure_runtime_paths(self) -> None:
        if not self.database_url:
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
