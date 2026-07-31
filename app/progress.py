import asyncio
import time
from typing import Optional, Union
from aiogram import Bot
from aiogram.types import Message
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter


class ProgressBar:
    """
    Индикатор прогресса в реальном времени с поддержкой графической шкалы,
    расчета оставшегося времени (ETA) и встроенным ограничением частоты (throttling)
    для предотвращения ошибок 429 Too Many Requests от Telegram API.
    """

    def __init__(
        self,
        total: int,
        length: int = 12,
        fill_char: str = "█",
        empty_char: str = "░",
        min_update_interval: float = 1.2,
    ):
        self.total = max(1, total)
        self.length = length
        self.fill_char = fill_char
        self.empty_char = empty_char
        self.min_update_interval = min_update_interval

        self.current = 0
        self.start_time = time.time()
        self.last_update_time = 0.0
        self.last_text = ""

    def render(self, current: int, prefix: str = "", suffix: str = "") -> str:
        self.current = min(max(0, current), self.total)
        percent = (self.current / self.total) * 100.0
        filled_len = int(round(self.length * self.current / float(self.total)))
        bar = self.fill_char * filled_len + self.empty_char * (self.length - filled_len)

        elapsed = time.time() - self.start_time
        if self.current > 0 and elapsed > 0:
            speed = self.current / elapsed
            remaining_items = self.total - self.current
            eta_seconds = remaining_items / speed if speed > 0 else 0
            eta_str = self._format_duration(eta_seconds)
        else:
            eta_str = "--:--"

        elapsed_str = self._format_duration(elapsed)

        lines = []
        if prefix:
            lines.append(prefix)

        lines.append(
            f"[{bar}] {percent:.1f}%\n"
            f"📊 Прогресс: {self.current}/{self.total}\n"
            f"⏱ Отработано: {elapsed_str} | Осталось: ~{eta_str}"
        )

        if suffix:
            lines.append(suffix)

        return "\n".join(lines)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        secs = int(max(0, seconds))
        mins, s = divmod(secs, 60)
        hrs, m = divmod(mins, 60)
        if hrs > 0:
            return f"{hrs:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    async def update_message(
        self,
        message: Message,
        current: int,
        prefix: str = "",
        suffix: str = "",
        force: bool = False,
    ) -> bool:
        """
        Обновляет текст сообщения с прогрессом. Возвращает True, если сообщение обновлено.
        """
        now = time.time()
        text = self.render(current, prefix=prefix, suffix=suffix)

        # Если текст не изменился или время между обновлениями слишком мало (и force=False), пропускаем
        if text == self.last_text and not force:
            return False

        if not force and (now - self.last_update_time) < self.min_update_interval:
            return False

        try:
            await message.edit_text(text, parse_mode="HTML")
            self.last_update_time = now
            self.last_text = text
            return True
        except TelegramBadRequest as err:
            # Игнорируем ошибку, если содержимое сообщения не изменилось
            if "message is not modified" in str(err).lower():
                return False
            raise
        except TelegramRetryAfter as retry:
            # При запросе задержки делаем небольшую паузу
            await asyncio.sleep(retry.retry_after)
            return False
        except Exception:
            return False
