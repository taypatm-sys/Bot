from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo


UTC = timezone.utc


def ensure_future(value: datetime, now: datetime, minimum_seconds: int = 5) -> datetime:
    if value <= now + timedelta(seconds=minimum_seconds):
        raise ValueError("укажите время в будущем")
    return value


def parse_local_datetime(
    value: str,
    timezone_info: ZoneInfo,
    *,
    now: datetime | None = None,
) -> datetime:
    local_now = (now or datetime.now(timezone_info)).astimezone(timezone_info)
    text = " ".join(value.strip().split())

    parsed: datetime | None = None
    for pattern in ("%d.%m.%Y %H:%M", "%d.%m.%y %H:%M"):
        try:
            parsed = datetime.strptime(text, pattern)
            break
        except ValueError:
            continue

    if parsed is None:
        try:
            partial = datetime.strptime(text, "%d.%m %H:%M")
            parsed = partial.replace(year=local_now.year)
            candidate = parsed.replace(tzinfo=timezone_info)
            if candidate <= local_now:
                parsed = parsed.replace(year=local_now.year + 1)
        except ValueError:
            parsed = None

    if parsed is None:
        try:
            parsed_time = datetime.strptime(text, "%H:%M").time()
            candidate = datetime.combine(
                local_now.date(), parsed_time, tzinfo=timezone_info
            )
            if candidate <= local_now:
                candidate += timedelta(days=1)
            parsed = candidate.replace(tzinfo=None)
        except ValueError as error:
            raise ValueError(
                "используйте формат ДД.ММ.ГГГГ ЧЧ:ММ, например 25.07.2026 18:30"
            ) from error

    local_value = parsed.replace(tzinfo=timezone_info)
    ensure_future(local_value, local_now)
    return local_value.astimezone(UTC)


def quick_times(timezone_info: ZoneInfo, now: datetime | None = None) -> list[datetime]:
    local_now = (now or datetime.now(timezone_info)).astimezone(timezone_info)
    candidates = []

    today_18 = datetime.combine(local_now.date(), time(18, 0), tzinfo=timezone_info)
    if today_18 > local_now + timedelta(minutes=1):
        candidates.append(today_18)

    tomorrow = local_now.date() + timedelta(days=1)
    for hour in (12, 18):
        candidates.append(
            datetime.combine(tomorrow, time(hour, 0), tzinfo=timezone_info)
        )
    return candidates[:3]


def to_utc_timestamp(value: datetime) -> int:
    return int(value.astimezone(UTC).timestamp())


def from_utc_timestamp(value: int) -> datetime:
    return datetime.fromtimestamp(value, tz=UTC)


def format_local(value_utc: datetime, timezone_info: ZoneInfo) -> str:
    return value_utc.astimezone(timezone_info).strftime("%d.%m.%Y в %H:%M")
