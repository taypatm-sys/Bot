import asyncio
import hashlib
import html
import io
import logging
import random
import re
import secrets
from collections import Counter
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Literal, Optional, Sequence
from urllib.parse import urlparse, urlunparse

import aiohttp
from google import genai
from google.genai import types
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

from app.analysis_coordinator import AnalysisCoordinator
from app.models import ReferenceAsset
from app.storage import PostRepository


UTC = timezone.utc
logger = logging.getLogger(__name__)

GarmentTag = Literal[
    "t-shirt",
    "hoodie",
    "sweatshirt",
    "long-sleeve",
    "zip-hoodie",
    "cap",
    "jacket",
]
MoodTag = Literal[
    "calm",
    "bold",
    "cozy",
    "sporty",
    "youth",
    "romantic",
    "playful",
    "minimal",
    "premium",
    "street",
]

PINTEREST_HOSTS = {"pinterest.com", "pin.it"}
PINTEREST_IMAGE_HOST = "i.pinimg.com"
URL_PATTERN = re.compile(r"https?://[^\s<>\[\]{}\"']+", re.IGNORECASE)
PIN_ID_PATTERN = re.compile(r"/pin/(\d+)", re.IGNORECASE)
MAX_HTML_BYTES = 3 * 1024 * 1024
MAX_IMAGE_DOWNLOAD_BYTES = 12 * 1024 * 1024
PINTEREST_SEARCH_ENDPOINT = "https://api.pinterest.com/v5/search/partner/pins"
PINTEREST_DISCOVERY_QUERIES = (
    "men oversized t shirt front streetwear waist up flash photography",
    "men oversized t shirt back print rear view streetwear",
    "women oversized t shirt front outfit waist up natural photo",
    "women oversized t shirt back print rear view beach outfit",
    "unisex oversized t shirt front close up street photography",
    "unisex oversized t shirt back print rear three quarter",
    "men hoodie front streetwear waist up",
    "women hoodie back print rear view outfit",
    "men sweatshirt front casual flash photography",
    "women sweatshirt back print rear view",
    "streetwear cap front close up lifestyle photo",
    "oversized long sleeve front streetwear waist up",
)


PINTEREST_GARMENT_TERMS = {
    "t-shirt": "t shirt",
    "hoodie": "hoodie",
    "sweatshirt": "sweatshirt",
    "long-sleeve": "long sleeve shirt",
    "zip-hoodie": "zip hoodie",
    "cap": "cap streetwear",
    "jacket": "streetwear jacket",
}
PINTEREST_GENDER_TERMS = {
    "women": "woman",
    "men": "man",
    "unisex": "streetwear model",
}
PINTEREST_SIDE_TERMS = {
    "front": "front view print area visible",
    "back": "rear view back print area visible",
}
PINTEREST_MOOD_TERMS = {
    "calm": "natural lifestyle",
    "bold": "bold streetwear",
    "cozy": "cozy casual",
    "sporty": "sporty streetwear",
    "youth": "youth street style",
    "romantic": "soft aesthetic",
    "playful": "playful outfit",
    "minimal": "minimal outfit",
    "premium": "premium casual",
    "street": "street photography",
}


class ReferenceTags(BaseModel):
    garment_types: list[GarmentTag] = Field(min_length=1, max_length=4)
    gender: Literal["women", "men", "unisex"]
    moods: list[MoodTag] = Field(min_length=1, max_length=4)
    pose_kind: Literal["sitting", "walking", "activity", "standing", "close-up"]
    action: str = Field(min_length=1, max_length=120)
    location_category: Literal[
        "home",
        "cafe",
        "street",
        "shop",
        "car",
        "elevator",
        "beach",
        "outdoor",
        "studio",
        "other",
    ]
    setting: str = Field(min_length=1, max_length=160)
    camera_angle: Literal[
        "front",
        "rear",
        "three-quarter",
        "side",
        "high",
        "low",
        "mirror",
    ]
    framing: Literal["detail", "close-up", "waist-up", "three-quarter", "full-body"]
    lighting: Literal["daylight", "indoor", "warm", "flash", "night", "mixed"]
    season: Literal["warm", "cold", "all-season"]
    print_side_visible: Literal["front", "back", "both", "cap-front", "unclear"]
    print_area_visibility: int = Field(ge=0, le=100)
    composition_notes: str = Field(min_length=1, max_length=240)
    usable: bool
    unusable_reason: str = Field(default="", max_length=240)


class ReferenceCompatibility(BaseModel):
    compatible: bool
    visible_side: Literal["front", "back", "both", "cap-front", "unclear"]
    camera_angle: Literal[
        "front",
        "rear",
        "three-quarter",
        "side",
        "high",
        "low",
        "mirror",
        "unclear",
    ]
    print_area_visibility: int = Field(ge=0, le=100)
    reason: str = Field(min_length=1, max_length=240)


class ReferenceImportError(RuntimeError):
    def __init__(self, message: str, *, retry_after: Optional[timedelta] = None):
        super().__init__(message)
        self.retry_after = retry_after


class _MetaImageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.image_urls: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag.casefold() != "meta":
            return
        values = {key.casefold(): value for key, value in attrs if value}
        key = (values.get("property") or values.get("name") or "").casefold()
        if key not in {
            "og:image",
            "og:image:url",
            "twitter:image",
            "twitter:image:src",
        }:
            return
        content = values.get("content")
        if content:
            self.image_urls.append(html.unescape(content.strip()))


def _is_host(host: str, root: str) -> bool:
    clean = host.casefold().split(":", 1)[0]
    return clean == root or clean.endswith(f".{root}")


def normalize_reference_urls(text: str) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for match in URL_PATTERN.finditer(text):
        raw = match.group(0).rstrip(".,;:!?)]}")
        parsed = urlparse(raw)
        host = (parsed.hostname or "").casefold()
        clean_url = ""
        if _is_host(host, "pinterest.com"):
            pin_match = PIN_ID_PATTERN.search(parsed.path)
            if pin_match:
                clean_url = f"https://www.pinterest.com/pin/{pin_match.group(1)}/"
        elif _is_host(host, "pin.it"):
            path = parsed.path.rstrip("/")
            if path:
                clean_url = f"https://pin.it{path}"
        elif host == PINTEREST_IMAGE_HOST:
            clean_url = urlunparse(("https", host, parsed.path, "", "", ""))
        if clean_url and clean_url not in seen:
            seen.add(clean_url)
            normalized.append(clean_url)
    return normalized


def _extract_pin_id(url: str) -> str:
    match = PIN_ID_PATTERN.search(urlparse(url).path)
    return match.group(1) if match else ""


def _original_image_candidate(url: str) -> str:
    parsed = urlparse(url)
    parts = parsed.path.split("/")
    if len(parts) > 2 and parts[1] in {
        "75x75_RS",
        "170x",
        "236x",
        "474x",
        "564x",
        "736x",
    }:
        parts[1] = "originals"
    return urlunparse(("https", PINTEREST_IMAGE_HOST, "/".join(parts), "", "", ""))


def _image_candidates(urls: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for url in urls:
        parsed = urlparse(url)
        if (parsed.hostname or "").casefold() != PINTEREST_IMAGE_HOST:
            continue
        for candidate in (_original_image_candidate(url), url):
            if candidate not in seen:
                seen.add(candidate)
                result.append(candidate)
    return result


def _resize_reference_image(data: bytes) -> tuple[bytes, bytes, int, int, str]:
    try:
        with Image.open(io.BytesIO(data)) as source:
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (OSError, UnidentifiedImageError) as error:
        raise ReferenceImportError("Ссылка вернула не изображение") from error

    width, height = image.size
    if min(width, height) < 320:
        raise ReferenceImportError("Разрешение референса меньше 320 пикселей")

    image.thumbnail((1600, 2000), Image.Resampling.LANCZOS)
    width, height = image.size
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=88, optimize=True, progressive=True)

    thumbnail = image.copy()
    thumbnail.thumbnail((360, 480), Image.Resampling.LANCZOS)
    thumb_output = io.BytesIO()
    thumbnail.save(thumb_output, format="JPEG", quality=80, optimize=True)
    return output.getvalue(), thumb_output.getvalue(), width, height, "image/jpeg"


class ReferenceCatalog:
    def __init__(
        self,
        *,
        repository: PostRepository,
        api_key: str,
        analysis_model: str,
        import_delay_seconds: float = 5.0,
        idle_interval_seconds: float = 300.0,
        max_attempts: int = 5,
        min_pool_size: int = 20,
        analysis_timeout_seconds: float = 90.0,
        user_agent: str = "TaypaReferenceCatalog/5.2",
        analysis_coordinator: Optional[AnalysisCoordinator] = None,
        pinterest_access_token: str = "",
        pinterest_search_enabled: bool = False,
        pinterest_country_code: str = "US",
        pinterest_search_interval_seconds: float = 21600.0,
        pinterest_target_pool_size: int = 160,
        pinterest_queries_per_cycle: int = 2,
    ) -> None:
        self.repository = repository
        self.client = genai.Client(api_key=api_key)
        self.analysis_model = analysis_model
        self.import_delay_seconds = max(2.0, import_delay_seconds)
        self.idle_interval_seconds = max(30.0, idle_interval_seconds)
        self.max_attempts = max(1, max_attempts)
        self.min_pool_size = max(1, min_pool_size)
        self.analysis_timeout_seconds = max(30.0, analysis_timeout_seconds)
        self.user_agent = user_agent
        self.analysis_coordinator = analysis_coordinator
        self.pinterest_access_token = pinterest_access_token.strip()
        self.pinterest_search_enabled = bool(pinterest_search_enabled)
        self.pinterest_country_code = (pinterest_country_code.strip().upper() or "US")[:2]
        self.pinterest_search_interval_seconds = max(
            900.0, pinterest_search_interval_seconds
        )
        self.pinterest_target_pool_size = max(20, pinterest_target_pool_size)
        self.pinterest_queries_per_cycle = max(1, min(6, pinterest_queries_per_cycle))
        self._next_discovery_at = datetime.now(UTC)
        self._wake_event = asyncio.Event()
        self._stop_event = asyncio.Event()

    def seed_file(self, path: Path) -> tuple[int, int]:
        if not path.is_file():
            return 0, 0
        text = path.read_text(encoding="utf-8-sig")
        return self.add_text(text, source_name=path.name)

    def add_text(self, text: str, *, source_name: str) -> tuple[int, int]:
        urls = normalize_reference_urls(text)
        result = self.repository.enqueue_reference_urls(urls, source_name=source_name)
        if result[0]:
            self._wake_event.set()
        return result

    def retry_failed(self) -> int:
        counts = self.resume_now()
        return sum(counts.values())

    def resume_now(self) -> dict[str, int]:
        counts = self.repository.resume_reference_imports()
        if sum(counts.values()):
            self._wake_event.set()
        return counts

    async def stop(self) -> None:
        self._stop_event.set()
        self._wake_event.set()

    async def run(self) -> None:
        while not self._stop_event.is_set():
            self._wake_event.clear()
            recovered = await asyncio.to_thread(
                self.repository.recover_stale_reference_imports
            )
            if recovered:
                logger.warning(
                    "Автоматически восстановлено зависших референсов: %s",
                    recovered,
                )

            await self._maybe_discover_from_pinterest()
            processed = await self.process_next()
            delay = (
                self.import_delay_seconds if processed else self.idle_interval_seconds
            )
            try:
                await asyncio.wait_for(self._wake_event.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass

    def build_product_search_queries(
        self,
        *,
        garment_type: GarmentTag,
        target_gender: Literal["women", "men", "unisex"],
        moods: Sequence[MoodTag],
        print_side: Literal["front", "back"],
        shirt_color: str = "",
        fit: str = "",
    ) -> list[str]:
        garment = PINTEREST_GARMENT_TERMS.get(garment_type, garment_type)
        gender = PINTEREST_GENDER_TERMS.get(target_gender, "streetwear model")
        side = PINTEREST_SIDE_TERMS[print_side]
        mood = next(
            (PINTEREST_MOOD_TERMS[item] for item in moods if item in PINTEREST_MOOD_TERMS),
            "natural lifestyle",
        )
        color = " ".join(shirt_color.strip().casefold().split())
        fit_term = "oversized" if "overs" in fit.casefold() else "relaxed fit"
        core = " ".join(part for part in (gender, color, fit_term, garment, side) if part)
        if print_side == "back":
            framing = "rear three quarter waist up no bag covering shirt"
        elif garment_type == "cap":
            framing = "close up portrait cap front visible"
        else:
            framing = "waist up torso visible natural pose"
        return list(
            dict.fromkeys(
                [
                    f"{core} {mood} {framing}",
                    f"{gender} {garment} {side} street style photography {framing}",
                    f"{gender} wearing {color} {garment} {side} casual outfit photo",
                ]
            )
        )

    async def discover_for_product(
        self,
        *,
        garment_type: GarmentTag,
        target_gender: Literal["women", "men", "unisex"],
        moods: Sequence[MoodTag],
        print_side: Literal["front", "back"],
        shirt_color: str = "",
        fit: str = "",
        import_now: int = 0,
    ) -> tuple[str, int, int]:
        """Find composition references on Pinterest and grow the catalog.

        The search is built from the current product instead of only comparing
        against the original seed catalog. Newly found pins are deduplicated by
        URL and tagged before they can be used.
        """
        if not self.pinterest_search_enabled:
            self.repository.set_setting("pinterest_discovery_status", "выключен")
            return "", 0, 0
        if not self.pinterest_access_token:
            self.repository.set_setting(
                "pinterest_discovery_status", "нужен PINTEREST_ACCESS_TOKEN"
            )
            return "", 0, 0

        queries = self.build_product_search_queries(
            garment_type=garment_type,
            target_gender=target_gender,
            moods=moods,
            print_side=print_side,
            shirt_color=shirt_color,
            fit=fit,
        )
        signature = "|".join(queries).encode("utf-8")
        source_name = f"pinterest-product-{hashlib.sha1(signature).hexdigest()[:12]}"
        added = await self._search_pinterest_terms(
            queries[: self.pinterest_queries_per_cycle],
            source_name=source_name,
            max_urls=12,
        )
        processed = 0
        for _ in range(max(0, min(import_now, 8))):
            if not await self.process_next(source_name=source_name):
                break
            processed += 1
        self.repository.set_setting(
            "pinterest_discovery_status",
            f"поиск по товару: добавлено {added}, обработано {processed}",
        )
        self.repository.set_setting(
            "pinterest_last_product_queries", " || ".join(queries)[:1500]
        )
        if added:
            self._wake_event.set()
        return source_name, added, processed

    async def _search_pinterest_terms(
        self,
        terms: Sequence[str],
        *,
        source_name: str,
        max_urls: int = 30,
    ) -> int:
        timeout = aiohttp.ClientTimeout(total=35, connect=10, sock_read=20)
        headers = {
            "Authorization": f"Bearer {self.pinterest_access_token}",
            "Accept": "application/json",
            "User-Agent": self.user_agent,
        }
        discovered_urls: list[str] = []
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            for term in terms:
                params = {
                    "term": term,
                    "country_code": self.pinterest_country_code,
                }
                async with session.get(
                    PINTEREST_SEARCH_ENDPOINT, params=params
                ) as response:
                    if response.status == 429:
                        raise ReferenceImportError(
                            "Pinterest ограничил частоту автопоиска",
                            retry_after=timedelta(hours=1),
                        )
                    if response.status in {401, 403}:
                        raise ReferenceImportError(
                            "Pinterest API не разрешил поиск. Проверьте OAuth token и доступ к search/partner/pins"
                        )
                    if response.status >= 400:
                        body = (await response.text())[:240]
                        raise ReferenceImportError(
                            f"Pinterest API вернул {response.status}: {body}"
                        )
                    payload = await response.json(content_type=None)
                discovered_urls.extend(self._pinterest_result_urls(payload))
                if len(discovered_urls) >= max_urls:
                    break
        unique_urls = list(dict.fromkeys(discovered_urls))[:max_urls]
        added, _ = self.repository.enqueue_reference_urls(
            unique_urls, source_name=source_name
        )
        return added

    async def _maybe_discover_from_pinterest(self) -> int:
        if not self.pinterest_search_enabled:
            self.repository.set_setting("pinterest_discovery_status", "выключен")
            return 0
        if not self.pinterest_access_token:
            self.repository.set_setting(
                "pinterest_discovery_status", "нужен PINTEREST_ACCESS_TOKEN"
            )
            return 0
        now = datetime.now(UTC)
        if now < self._next_discovery_at:
            return 0
        ready = self.repository.reference_stats().get("ready", 0)
        if ready >= self.pinterest_target_pool_size:
            self.repository.set_setting(
                "pinterest_discovery_status",
                f"база заполнена: {ready}/{self.pinterest_target_pool_size}",
            )
            self._next_discovery_at = now + timedelta(
                seconds=self.pinterest_search_interval_seconds
            )
            return 0

        try:
            added = await self.discover_from_pinterest()
            self.repository.set_setting(
                "pinterest_discovery_status", f"работает, новых ссылок: {added}"
            )
            self.repository.set_setting(
                "pinterest_last_discovery_at", now.isoformat()
            )
        except Exception as error:
            logger.warning("Автопоиск Pinterest не выполнен: %s", error)
            self.repository.set_setting(
                "pinterest_discovery_status", f"ошибка: {str(error)[:160]}"
            )
            added = 0
        self._next_discovery_at = now + timedelta(
            seconds=self.pinterest_search_interval_seconds
        )
        if added:
            self._wake_event.set()
        return added

    async def discover_from_pinterest(self) -> int:
        """Grow a broad fallback pool through the official Pinterest API."""
        index_raw = self.repository.get_setting("pinterest_discovery_query_index") or "0"
        try:
            start_index = int(index_raw)
        except ValueError:
            start_index = 0
        terms = [
            PINTEREST_DISCOVERY_QUERIES[
                (start_index + offset) % len(PINTEREST_DISCOVERY_QUERIES)
            ]
            for offset in range(self.pinterest_queries_per_cycle)
        ]
        added = await self._search_pinterest_terms(
            terms, source_name="pinterest-auto-search", max_urls=30
        )
        next_index = (start_index + self.pinterest_queries_per_cycle) % len(
            PINTEREST_DISCOVERY_QUERIES
        )
        self.repository.set_setting(
            "pinterest_discovery_query_index", str(next_index)
        )
        return added

    @staticmethod
    def _pinterest_result_urls(payload: object) -> list[str]:
        if not isinstance(payload, dict):
            return []
        items = payload.get("items")
        if not isinstance(items, list):
            return []
        urls: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            pin_id = str(item.get("id", "")).strip()
            if pin_id.isdigit():
                urls.append(f"https://www.pinterest.com/pin/{pin_id}/")
                continue
            link = str(item.get("link", "")).strip()
            urls.extend(normalize_reference_urls(link))
        return urls

    async def process_next(self, *, source_name: Optional[str] = None) -> bool:
        job = await asyncio.to_thread(
            self.repository.claim_reference_import, source_name=source_name
        )
        if job is None:
            return False
        try:
            image_bytes = job.image_bytes
            mime_type = job.image_mime_type or "image/jpeg"
            if not image_bytes:
                resolved_url, raw_data = await self._download_reference(job.source_url)
                (
                    image_bytes,
                    thumbnail_bytes,
                    width,
                    height,
                    mime_type,
                ) = await asyncio.to_thread(_resize_reference_image, raw_data)
                await asyncio.to_thread(
                    self.repository.store_reference_image,
                    job.id,
                    pin_id=_extract_pin_id(job.source_url),
                    resolved_image_url=resolved_url,
                    image_bytes=image_bytes,
                    image_mime_type=mime_type,
                    thumbnail_bytes=thumbnail_bytes,
                    width=width,
                    height=height,
                    image_sha256=hashlib.sha256(image_bytes).hexdigest(),
                )
            try:
                if self.analysis_coordinator is None:
                    tags = await asyncio.wait_for(
                        asyncio.to_thread(
                            self._analyze_reference_sync,
                            image_bytes,
                            mime_type,
                        ),
                        timeout=self.analysis_timeout_seconds,
                    )
                else:
                    async with self.analysis_coordinator.background():
                        tags = await asyncio.wait_for(
                            asyncio.to_thread(
                                self._analyze_reference_sync,
                                image_bytes,
                                mime_type,
                            ),
                            timeout=self.analysis_timeout_seconds,
                        )
            except asyncio.TimeoutError as error:
                raise ReferenceImportError(
                    "Gemini не ответил вовремя",
                    retry_after=timedelta(minutes=3),
                ) from error
            await asyncio.to_thread(
                self.repository.mark_reference_ready,
                job.id,
                tags=tags.model_dump(),
            )
            logger.info("Референс #%s обработан: %s", job.id, job.source_url)
        except Exception as error:
            retry_after = getattr(error, "retry_after", None)
            if not isinstance(retry_after, timedelta):
                retry_minutes = (2, 5, 15, 30, 60)[
                    min(job.attempt_count, 4)
                ]
                retry_after = timedelta(minutes=retry_minutes)
            status = await asyncio.to_thread(
                self.repository.mark_reference_import_error,
                job.id,
                error=str(error),
                retry_at_utc=datetime.now(UTC) + retry_after,
                max_attempts=self.max_attempts,
            )
            logger.warning(
                "Импорт референса #%s завершился статусом %s: %s",
                job.id,
                status,
                error,
            )
        return True

    async def _read_limited(
        self,
        response: aiohttp.ClientResponse,
        *,
        limit: int,
    ) -> bytes:
        if response.content_length and response.content_length > limit:
            raise ReferenceImportError("Файл референса слишком большой")
        chunks: list[bytes] = []
        size = 0
        async for chunk in response.content.iter_chunked(64 * 1024):
            size += len(chunk)
            if size > limit:
                raise ReferenceImportError("Файл референса слишком большой")
            chunks.append(chunk)
        return b"".join(chunks)

    async def _request_bytes(
        self,
        session: aiohttp.ClientSession,
        url: str,
        *,
        limit: int,
    ) -> tuple[bytes, str, str]:
        async with session.get(url, allow_redirects=True) as response:
            if response.status == 429:
                retry_value = response.headers.get("Retry-After", "")
                try:
                    retry_after = timedelta(seconds=max(60, int(retry_value)))
                except ValueError:
                    retry_after = timedelta(minutes=30)
                raise ReferenceImportError(
                    "Pinterest временно ограничил частоту запросов",
                    retry_after=retry_after,
                )
            if response.status in {401, 403}:
                raise ReferenceImportError(
                    f"Pinterest не разрешил загрузку, код {response.status}",
                    retry_after=timedelta(hours=2),
                )
            if response.status >= 400:
                raise ReferenceImportError(f"Ссылка недоступна, код {response.status}")
            data = await self._read_limited(response, limit=limit)
            return data, response.headers.get("Content-Type", ""), str(response.url)

    async def _download_reference(self, source_url: str) -> tuple[str, bytes]:
        timeout = aiohttp.ClientTimeout(total=35, connect=10, sock_read=20)
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,image/avif,image/webp,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.8",
        }
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            parsed = urlparse(source_url)
            if (parsed.hostname or "").casefold() == PINTEREST_IMAGE_HOST:
                candidates = [source_url]
            else:
                page, content_type, final_url = await self._request_bytes(
                    session,
                    source_url,
                    limit=MAX_HTML_BYTES,
                )
                final_host = (urlparse(final_url).hostname or "").casefold()
                if not _is_host(final_host, "pinterest.com"):
                    raise ReferenceImportError("Короткая ссылка ведет не на Pinterest")
                if "html" not in content_type.casefold():
                    raise ReferenceImportError(
                        "Pinterest вернул страницу неизвестного типа"
                    )
                parser = _MetaImageParser()
                parser.feed(page.decode("utf-8", errors="ignore"))
                candidates = _image_candidates(parser.image_urls)
                if not candidates:
                    raise ReferenceImportError(
                        "На странице Pinterest не найдена фотография"
                    )

            last_error: Optional[Exception] = None
            for candidate in candidates:
                try:
                    data, content_type, final_url = await self._request_bytes(
                        session,
                        candidate,
                        limit=MAX_IMAGE_DOWNLOAD_BYTES,
                    )
                    final_host = (urlparse(final_url).hostname or "").casefold()
                    if final_host != PINTEREST_IMAGE_HOST:
                        raise ReferenceImportError(
                            "Изображение находится не на Pinterest CDN"
                        )
                    if not content_type.casefold().startswith("image/"):
                        raise ReferenceImportError("Ссылка вернула не изображение")
                    return final_url, data
                except ReferenceImportError as error:
                    last_error = error
            if last_error:
                raise last_error
            raise ReferenceImportError("Не удалось загрузить фотографию Pinterest")

    def _analyze_reference_sync(
        self, image_bytes: bytes, mime_type: str
    ) -> ReferenceTags:
        prompt = (
            "Analyze this fashion lifestyle photo only as a composition reference for "
            "future DTF clothing mockups. Ignore logos, artwork and text printed on the "
            "clothes. Tag the real photographic situation. garment_types must list every "
            "Taypa product that could naturally replace the visible item without changing "
            "the pose: t-shirt, hoodie, sweatshirt, long-sleeve, zip-hoodie, cap or jacket. "
            "gender describes the intended wearer shown in this reference. moods must use "
            "only the supplied enum. print_area_visibility is how clearly a new front or "
            "back DTF design could be placed while keeping the pose. Mark usable false if "
            "the person appears under 18, the garment area is mostly hidden, the image is "
            "a collage, an isolated product, a drawing, a studio catalog cutout or too low "
            "quality. composition_notes must briefly preserve the useful pose, camera, "
            "crop, fabric folds and ordinary details, without identifying or copying the "
            "person. Return objective tags, not marketing language."
        )
        response = self.client.models.generate_content(
            model=self.analysis_model,
            contents=[
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ReferenceTags,
            ),
        )
        if not response.text:
            raise ReferenceImportError("Gemini не вернул теги референса")
        return ReferenceTags.model_validate_json(response.text)

    def select_reference(
        self,
        *,
        garment_type: GarmentTag,
        target_gender: Literal["women", "men", "unisex"],
        moods: Sequence[MoodTag],
        request_token: Optional[str] = None,
        season: Optional[Literal["warm", "cold", "all-season"]] = None,
        print_side: Optional[Literal["front", "back"]] = None,
        exclude_ids: Sequence[int] = (),
        preferred_source_name: str = "",
        rng: Optional[random.Random] = None,
    ) -> Optional[ReferenceAsset]:
        excluded = set(exclude_ids)
        mood_set = set(moods)
        scored: list[tuple[float, ReferenceAsset]] = []
        for asset in self.repository.list_ready_reference_assets():
            if asset.id in excluded:
                continue
            tags = asset.tags
            garments = set(tags.get("garment_types", []))
            if garment_type not in garments:
                continue
            gender = str(tags.get("gender", "unisex"))
            if target_gender != "unisex" and gender not in {target_gender, "unisex"}:
                continue
            visibility = int(tags.get("print_area_visibility", 0) or 0)
            if visibility < 55:
                continue

            visible_side = str(tags.get("print_side_visible", "unclear"))
            if garment_type == "cap":
                allowed_sides = {"cap-front", "front", "both"}
            elif print_side == "back":
                allowed_sides = {"back", "both"}
            elif print_side == "front":
                allowed_sides = {"front", "both"}
            else:
                allowed_sides = {"front", "back", "both", "cap-front", "unclear"}
            if visible_side not in allowed_sides:
                continue

            camera_angle = str(tags.get("camera_angle", ""))
            if print_side == "back" and camera_angle not in {"rear", "three-quarter"}:
                continue
            if print_side == "front" and camera_angle == "rear":
                continue

            minimum_visibility = 75 if print_side in {"front", "back"} else 55
            if visibility < minimum_visibility:
                continue

            framing = str(tags.get("framing", ""))
            if garment_type != "cap" and framing == "full-body":
                continue
            framing_score = {
                "waist-up": 45,
                "close-up": 38,
                "three-quarter": 28,
                "detail": 18,
                "full-body": -45,
            }.get(framing, 0)
            notes = " ".join(
                [
                    str(tags.get("setting", "")),
                    str(tags.get("composition_notes", "")),
                ]
            ).casefold()
            crowd_penalty = 0
            if any(word in notes for word in ("crowd", "busy", "group", "many people")):
                crowd_penalty = 60

            score = 100.0 + visibility * 0.35
            score += 35 if gender == target_gender else 10
            score += 22 * len(mood_set.intersection(tags.get("moods", [])))
            score += framing_score
            score -= crowd_penalty
            if season and tags.get("season") in {season, "all-season"}:
                score += 12
            if preferred_source_name and asset.source_name == preferred_source_name:
                score += 80

            score -= asset.use_count * 3
            scored.append((score, asset))
        if not scored:
            return None
        scored.sort(key=lambda item: (-item[0], item[1].use_count, item[1].id))
        token = request_token or secrets.token_hex(12)
        candidates = scored[: min(12, len(scored))]
        if rng is not None and len(candidates) > 1:
            top_score = candidates[0][0]
            near_equal = [item for item in candidates if top_score - item[0] <= 4]
            rng.shuffle(near_equal)
            candidates = near_equal + [item for item in candidates if item not in near_equal]
        for _, asset in candidates:
            if self.repository.reserve_reference(
                asset.id,
                request_token=token,
                garment_type=garment_type,
                target_gender=target_gender,
                moods=list(moods),
            ):
                return asset
        return None

    async def validate_reference_for_generation(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        garment_type: GarmentTag,
        print_side: Literal["front", "back"],
    ) -> ReferenceCompatibility:
        return await asyncio.to_thread(
            self._validate_reference_for_generation_sync,
            image_bytes,
            mime_type,
            garment_type,
            print_side,
        )

    def _validate_reference_for_generation_sync(
        self,
        image_bytes: bytes,
        mime_type: str,
        garment_type: GarmentTag,
        print_side: Literal["front", "back"],
    ) -> ReferenceCompatibility:
        prompt = (
            "This is a strict preflight check before a paid clothing image generation. "
            f"The target product is a {garment_type} with a {print_side} print. "
            "Judge only whether this photographic reference can safely control pose, "
            "camera and crop without hiding or contradicting the printed garment panel. "
            "For a back print, the person's back must be clearly visible from rear or "
            "rear three-quarter view. For a front print, the front panel must be clearly "
            "visible. Reject front-facing references for back prints, rear-facing "
            "references for front prints, unclear body orientation, full-body distant "
            "shots, crossed arms, hair, bags or props covering the print area, crowds, "
            "collages, drawings and low-quality images. print_area_visibility is the "
            "percentage of the required front or back torso panel that remains usable. "
            "compatible may be true only when visibility is at least 75 and the side is "
            "unambiguous. Keep reason short and objective."
        )
        response = self.client.models.generate_content(
            model=self.analysis_model,
            contents=[
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ReferenceCompatibility,
                temperature=0,
            ),
        )
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, ReferenceCompatibility):
            result = parsed
        elif parsed is not None:
            result = ReferenceCompatibility.model_validate(parsed)
        elif response.text:
            result = ReferenceCompatibility.model_validate_json(response.text)
        else:
            raise ReferenceImportError("Gemini не проверил совместимость референса")

        side_ok = (
            result.visible_side in {"back", "both"}
            if print_side == "back"
            else result.visible_side in {"front", "both", "cap-front"}
        )
        angle_ok = (
            result.camera_angle in {"rear", "three-quarter"}
            if print_side == "back"
            else result.camera_angle != "rear"
        )
        compatible = (
            result.compatible
            and side_ok
            and angle_ok
            and result.print_area_visibility >= 75
        )
        if compatible == result.compatible:
            return result
        return result.model_copy(
            update={
                "compatible": compatible,
                "reason": "Сторона, ракурс или видимость зоны принта не подходят",
            }
        )

    def status_text(self) -> str:
        stats = self.repository.reference_stats()
        queue_details = self.repository.reference_queue_details()
        assets = self.repository.list_ready_reference_assets()
        garments: Counter[str] = Counter()
        genders: Counter[str] = Counter()
        for asset in assets:
            garments.update(asset.tags.get("garment_types", []))
            genders.update([str(asset.tags.get("gender", "unisex"))])
        garment_labels = {
            "t-shirt": "футболки",
            "hoodie": "худи",
            "sweatshirt": "свитшоты",
            "long-sleeve": "лонгсливы",
            "zip-hoodie": "зип-худи",
            "cap": "кепки",
            "jacket": "куртки",
        }
        garment_line = ", ".join(
            f"{label}: {garments.get(key, 0)}" for key, label in garment_labels.items()
        )
        pending = stats.get("pending", 0)
        processing = stats.get("processing", 0)
        retry = stats.get("retry", 0)
        waiting_lines = [
            f"В очереди: {pending}",
            f"Сейчас обрабатывается: {processing}",
            f"Ждут повторной попытки: {retry}",
        ]
        next_retry = queue_details.get("next_retry_at_utc")
        if retry and isinstance(next_retry, datetime):
            seconds = max(0, int((next_retry - datetime.now(UTC)).total_seconds()))
            if seconds < 60:
                wait_label = "сейчас"
            elif seconds < 3600:
                wait_label = f"через {max(1, seconds // 60)} мин"
            else:
                hours, remainder = divmod(seconds, 3600)
                minutes = remainder // 60
                wait_label = f"через {hours} ч {minutes} мин"
            waiting_lines.append(f"Следующая попытка: {wait_label}")

        reason_counts: Counter[str] = Counter()
        for raw_reason, amount in queue_details.get("reasons", []):
            reason = raw_reason.casefold()
            if "pinterest" in reason or "код 403" in reason or "код 429" in reason:
                label = "временное ограничение Pinterest"
            elif "quota" in reason or "resource_exhausted" in reason:
                label = "временный лимит Gemini"
            elif "не ответил вовремя" in reason or "timeout" in reason:
                label = "тайм-аут ответа"
            else:
                label = "временная ошибка загрузки или анализа"
            reason_counts[label] += amount
        if reason_counts:
            reason_text = ", ".join(
                f"{label}: {amount}" for label, amount in reason_counts.items()
            )
            waiting_lines.append(f"Причина ожидания: {reason_text}")

        return (
            "Каталог референсов\n"
            f"Всего ссылок: {stats.get('total', 0)}\n"
            f"Готово: {stats.get('ready', 0)}\n"
            + "\n".join(waiting_lines)
            + "\n"
            f"Не подходят: {stats.get('disabled', 0)}\n"
            f"Ошибки: {stats.get('failed', 0)}\n\n"
            f"По одежде: {garment_line}\n"
            f"По полу: женщины {genders.get('women', 0)}, "
            f"мужчины {genders.get('men', 0)}, унисекс {genders.get('unisex', 0)}\n\n"
            f"Цель: минимум {self.min_pool_size} доступных фото для каждой "
            "используемой категории.\n"
            f"Автопоиск Pinterest: "
            f"{self.repository.get_setting('pinterest_discovery_status') or 'еще не запускался'}"
        )
