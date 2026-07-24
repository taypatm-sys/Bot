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
from urllib.parse import quote_plus, urlparse, urlunparse

import aiohttp
from google import genai
from google.genai import types
from PIL import Image, ImageOps, UnidentifiedImageError
import numpy as np
from pydantic import BaseModel, Field

from app.analysis_coordinator import AnalysisCoordinator
from app.models import ReferenceAsset
from app.local_mockup_generator import LocalCompositeNeedsGemini, LocalMockupGenerator
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
    "men oversized blank t shirt front waist up streetwear photo",
    "men oversized plain t shirt back rear view streetwear",
    "women oversized blank t shirt front waist up natural photo",
    "women oversized plain t shirt back rear view beach outfit",
    "unisex oversized blank t shirt front close up street photography",
    "unisex oversized plain t shirt back rear three quarter",
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
    garment_is_plain: bool = False
    existing_print_coverage_percent: int = Field(default=0, ge=0, le=100)
    composition_notes: str = Field(min_length=1, max_length=240)
    usable: bool
    unusable_reason: str = Field(default="", max_length=240)


class PlacementPoint(BaseModel):
    x: float = Field(ge=0, le=100)
    y: float = Field(ge=0, le=100)


class PlacementBox(BaseModel):
    x: float = Field(ge=0, le=100)
    y: float = Field(ge=0, le=100)
    width: float = Field(ge=0.01, le=100)
    height: float = Field(ge=0.01, le=100)


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
    target_print_box: Optional[PlacementBox] = None
    target_print_quad: list[PlacementPoint] = Field(default_factory=list, max_length=4)
    garment_color_match: bool = False
    existing_print_present: bool = False
    existing_print_box: Optional[PlacementBox] = None
    existing_print_quad: list[PlacementPoint] = Field(default_factory=list, max_length=4)
    existing_print_coverage_percent: int = Field(default=0, ge=0, le=100)
    existing_print_coverable: bool = False
    fabric_reconstruction_safe: bool = False
    local_composite_safe: bool = False
    reason: str = Field(min_length=1, max_length=240)


class SimpleReferencePreparation(BaseModel):
    suitable: bool
    visible_side: Literal["front", "back", "both", "unclear"]
    camera_angle: Literal[
        "front", "rear", "three-quarter", "side", "high", "low", "mirror", "unclear"
    ]
    print_area_visibility: int = Field(ge=0, le=100)
    target_print_box: Optional[PlacementBox] = None
    target_print_quad: list[PlacementPoint] = Field(default_factory=list, max_length=4)
    existing_print_present: bool = False
    existing_print_box: Optional[PlacementBox] = None
    existing_print_quad: list[PlacementPoint] = Field(default_factory=list, max_length=4)
    existing_print_coverage_percent: int = Field(default=0, ge=0, le=100)
    existing_print_coverable: bool = False
    fabric_reconstruction_safe: bool = False
    reason: str = Field(min_length=1, max_length=240)


class SimplePreparationVerification(BaseModel):
    old_print_removed: bool
    person_preserved: bool
    garment_preserved: bool
    visible_artifacts: bool
    quality_score: int = Field(ge=0, le=100)
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
        local_generator: Optional[LocalMockupGenerator] = None,
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
        self.local_generator = local_generator or LocalMockupGenerator()
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

    def prepare_all_simple(self) -> int:
        count = self.repository.reset_simple_reference_queue(include_skipped=True)
        self._wake_event.set()
        return count

    def prepare_simple_reference(self, reference_id: int) -> bool:
        queued = self.repository.reset_simple_reference(reference_id)
        if queued:
            self._wake_event.set()
        return queued

    async def stop(self) -> None:
        self._stop_event.set()
        self._wake_event.set()

    async def run(self) -> None:
        self.repository.set_setting("simple_worker_status", "запущен")
        while not self._stop_event.is_set():
            self._wake_event.clear()
            try:
                self.repository.set_setting("simple_worker_status", "работает")
                self.repository.set_setting(
                    "simple_worker_last_tick", datetime.now(UTC).isoformat()
                )
                recovered = await asyncio.to_thread(
                    self.repository.recover_stale_reference_imports
                )
                if recovered:
                    logger.warning(
                        "Автоматически восстановлено зависших референсов: %s",
                        recovered,
                    )
                recovered_simple = await asyncio.to_thread(
                    self.repository.recover_simple_reference_preparations
                )
                if recovered_simple:
                    logger.warning(
                        "Восстановлено подготовок простых референсов: %s",
                        recovered_simple,
                    )

                await self._maybe_discover_from_pinterest()
                processed_import = await self.process_next()
                processed_simple = await self.process_next_simple_reference()
                processed = processed_import or processed_simple
                self.repository.set_setting(
                    "simple_worker_status", "обработал задачу" if processed else "ожидает"
                )
                self.repository.set_setting("simple_worker_last_error", "")
                delay = (
                    self.import_delay_seconds if processed else self.idle_interval_seconds
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.exception("Фоновый обработчик референсов не остановлен после ошибки")
                self.repository.set_setting("simple_worker_status", "ошибка, перезапуск")
                self.repository.set_setting(
                    "simple_worker_last_error", str(error)[:500]
                )
                delay = 10.0
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
                    f"{core} blank plain no print {mood} {framing}",
                    f"{gender} plain blank {garment} {side} street style photo {framing}",
                    f"{gender} wearing plain {color} {garment} no logo {side} casual photo",
                    f"{gender} minimal {color} {garment} empty print area {framing}",
                ]
            )
        )

    def build_product_search_links(
        self,
        *,
        garment_type: GarmentTag,
        target_gender: Literal["women", "men", "unisex"],
        moods: Sequence[MoodTag],
        print_side: Literal["front", "back"],
        shirt_color: str = "",
        fit: str = "",
        limit: int = 4,
    ) -> list[tuple[str, str, str]]:
        """Build human-openable Pinterest searches without using the API.

        The bot does not crawl Pinterest. It only creates normal search URLs that
        an administrator can open, choose suitable Pins from, and send back to
        the bot for automatic import and preparation.
        """
        queries = self.build_product_search_queries(
            garment_type=garment_type,
            target_gender=target_gender,
            moods=moods,
            print_side=print_side,
            shirt_color=shirt_color,
            fit=fit,
        )
        side_label = "спереди" if print_side == "front" else "сзади"
        links: list[tuple[str, str, str]] = []
        for index, query in enumerate(queries[: max(1, min(limit, 6))], start=1):
            url = f"https://www.pinterest.com/search/pins/?q={quote_plus(query)}"
            label = f"Поиск {index}: {side_label}"
            links.append((label, query, url))
        self.repository.set_setting(
            "pinterest_last_product_queries", " || ".join(queries)[:1500]
        )
        self.repository.set_setting(
            "pinterest_discovery_status", "поисковые ссылки готовы, API не используется"
        )
        return links

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

    async def process_next_simple_reference(self) -> bool:
        asset = await asyncio.to_thread(
            self.repository.claim_simple_reference_preparation
        )
        if asset is None:
            return False
        await self._process_simple_reference_asset(asset)
        return True

    async def prepare_best_simple_candidates(
        self,
        *,
        garment_type: GarmentTag,
        target_gender: Literal["women", "men", "unisex"],
        moods: Sequence[MoodTag],
        print_side: Literal["front", "back"],
        shirt_color: str = "",
        fit: str = "",
        limit: int = 2,
    ) -> int:
        """Prepare the most relevant pending references for the current product.

        This avoids blocking a user request while unrelated references are processed
        in numeric ID order.
        """
        candidates: list[tuple[int, ReferenceAsset]] = []
        for asset in self.repository.list_ready_reference_assets():
            if asset.simple_status != "pending" or asset.simple_ready:
                continue
            tags = asset.tags or {}
            if garment_type not in set(tags.get("garment_types", [])):
                continue
            gender = str(tags.get("gender", "unisex"))
            if target_gender != "unisex" and gender not in {target_gender, "unisex"}:
                continue
            visible_side = str(tags.get("print_side_visible", "unclear"))
            if print_side == "front" and visible_side not in {"front", "both"}:
                continue
            if print_side == "back" and visible_side not in {"back", "both"}:
                continue
            if int(tags.get("print_area_visibility", 0) or 0) < 80:
                continue
            if str(tags.get("framing", "")) == "full-body":
                continue
            score, _ = self.score_reference(
                asset=asset,
                garment_type=garment_type,
                target_gender=target_gender,
                moods=moods,
                print_side=print_side,
                shirt_color=shirt_color,
                fit=fit,
            )
            candidates.append((score, asset))
        candidates.sort(key=lambda item: (-item[0], item[1].id))

        processed = 0
        for _, asset in candidates[: max(1, min(limit, 3))]:
            claimed = await asyncio.to_thread(
                self.repository.claim_specific_simple_reference, asset.id
            )
            if claimed is None:
                continue
            await self._process_simple_reference_asset(claimed)
            processed += 1
        return processed

    async def _process_simple_reference_asset(self, asset: ReferenceAsset) -> None:
        self.repository.set_setting("simple_worker_current_reference", str(asset.id))
        try:
            if self.analysis_coordinator is None:
                preparation = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._analyze_simple_preparation_sync,
                        asset.image_bytes,
                        asset.image_mime_type,
                        asset.tags,
                    ),
                    timeout=self.analysis_timeout_seconds,
                )
            else:
                async with self.analysis_coordinator.background():
                    preparation = await asyncio.wait_for(
                        asyncio.to_thread(
                            self._analyze_simple_preparation_sync,
                            asset.image_bytes,
                            asset.image_mime_type,
                            asset.tags,
                        ),
                        timeout=self.analysis_timeout_seconds,
                    )

            if not preparation.suitable:
                await asyncio.to_thread(
                    self.repository.store_simple_reference_variant,
                    asset.id,
                    image_bytes=None,
                    image_mime_type=None,
                    thumbnail_bytes=None,
                    ready=False,
                    reason=preparation.reason,
                    level="C",
                    quality_score=max(0, min(100, int(preparation.print_area_visibility))),
                )
                logger.info(
                    "Референс #%s не подготовлен для простого режима: %s",
                    asset.id,
                    preparation.reason,
                )
                return

            # A plain shirt is already a safe local base. Keep the original bytes so
            # the preparation stage cannot blur the person or background.
            if not preparation.existing_print_present:
                await asyncio.to_thread(
                    self.repository.store_simple_reference_variant,
                    asset.id,
                    image_bytes=asset.image_bytes,
                    image_mime_type=asset.image_mime_type,
                    thumbnail_bytes=asset.thumbnail_bytes,
                    ready=True,
                    reason="чистая зона принта, исходное фото сохранено без изменений",
                    level="A",
                    quality_score=max(0, min(100, int(preparation.print_area_visibility))),
                )
                logger.info("Референс #%s подготовлен как уровень A", asset.id)
                return

            prepared = await self.local_generator.prepare_simple_reference(
                image_bytes=asset.image_bytes,
                reference_tags=asset.tags,
                preparation=preparation.model_dump(),
            )
            normalized, thumbnail, _, _, mime_type = _resize_reference_image(
                prepared.data
            )
            if self.analysis_coordinator is None:
                verification = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._verify_simple_preparation_sync,
                        asset.image_bytes,
                        asset.image_mime_type,
                        normalized,
                        mime_type,
                    ),
                    timeout=self.analysis_timeout_seconds,
                )
            else:
                async with self.analysis_coordinator.background():
                    verification = await asyncio.wait_for(
                        asyncio.to_thread(
                            self._verify_simple_preparation_sync,
                            asset.image_bytes,
                            asset.image_mime_type,
                            normalized,
                            mime_type,
                        ),
                        timeout=self.analysis_timeout_seconds,
                    )
            verified = (
                verification.old_print_removed
                and verification.person_preserved
                and verification.garment_preserved
                and not verification.visible_artifacts
                and verification.quality_score >= 85
            )
            if not verified:
                await asyncio.to_thread(
                    self.repository.store_simple_reference_variant,
                    asset.id,
                    image_bytes=None,
                    image_mime_type=None,
                    thumbnail_bytes=None,
                    ready=False,
                    reason=f"Проверка очищенного фото не пройдена: {verification.reason}",
                    level="C",
                    quality_score=verification.quality_score,
                )
                logger.info(
                    "Референс #%s отклонен после проверки очищенного фото: %s",
                    asset.id,
                    verification.reason,
                )
                return

            await asyncio.to_thread(
                self.repository.store_simple_reference_variant,
                asset.id,
                image_bytes=normalized,
                image_mime_type=mime_type,
                thumbnail_bytes=thumbnail,
                ready=True,
                reason="старый принт удален и результат прошел повторную проверку",
                level="B",
                quality_score=verification.quality_score,
            )
            logger.info("Референс #%s подготовлен как уровень B", asset.id)
        except (asyncio.TimeoutError, LocalCompositeNeedsGemini, ReferenceImportError) as error:
            await asyncio.to_thread(
                self.repository.store_simple_reference_variant,
                asset.id,
                image_bytes=None,
                image_mime_type=None,
                thumbnail_bytes=None,
                ready=False,
                reason=str(error),
                level="C",
                quality_score=0,
            )
            logger.info(
                "Референс #%s пропущен для простого режима: %s",
                asset.id,
                error,
            )
        except Exception as error:
            await asyncio.to_thread(
                self.repository.store_simple_reference_variant,
                asset.id,
                image_bytes=None,
                image_mime_type=None,
                thumbnail_bytes=None,
                ready=False,
                reason=f"Ошибка подготовки: {str(error)[:220]}",
                level="C",
                quality_score=0,
            )
            logger.warning(
                "Ошибка подготовки референса #%s для простого режима: %s",
                asset.id,
                error,
            )
        finally:
            self.repository.set_setting("simple_worker_last_reference", str(asset.id))
            self.repository.set_setting("simple_worker_current_reference", "0")

    def _verify_simple_preparation_sync(
        self,
        original_bytes: bytes,
        original_mime_type: str,
        prepared_bytes: bytes,
        prepared_mime_type: str,
    ) -> SimplePreparationVerification:
        prompt = (
            "Compare IMAGE 1 (original reference) with IMAGE 2 (locally cleaned reference). "
            "This is a strict quality gate, not an editing request. Confirm that every old "
            "logo, word and graphic on the garment was removed in IMAGE 2, while the person, "
            "hands, face, pose, garment silhouette, seams, folds, lighting and background are "
            "preserved. visible_artifacts is true for blur patches, smeared fabric, duplicated "
            "texture, damaged hands/body, sharp rectangular borders or any remaining old print. "
            "Set quality_score below 85 if there is any doubt. Keep reason short and objective."
        )
        response = self.client.models.generate_content(
            model=self.analysis_model,
            contents=[
                prompt,
                "IMAGE 1 - ORIGINAL",
                types.Part.from_bytes(data=original_bytes, mime_type=original_mime_type),
                "IMAGE 2 - CLEANED",
                types.Part.from_bytes(data=prepared_bytes, mime_type=prepared_mime_type),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SimplePreparationVerification,
                temperature=0,
            ),
        )
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, SimplePreparationVerification):
            return parsed
        if parsed is not None:
            return SimplePreparationVerification.model_validate(parsed)
        if response.text:
            return SimplePreparationVerification.model_validate_json(response.text)
        raise ReferenceImportError("Gemini не проверил очищенный референс")

    def _analyze_simple_preparation_sync(
        self,
        image_bytes: bytes,
        mime_type: str,
        tags: dict[str, object],
    ) -> SimpleReferencePreparation:
        prompt = (
            "Prepare this real fashion photo as a reusable base for a local OpenCV DTF "
            "mockup. Do not generate or edit the image. Analyze geometry only. The visible "
            "garment must be an adult t-shirt with a clearly open front or back panel. "
            "target_print_box and target_print_quad define the largest safe central area "
            "where a new design can later be placed. Return four points in order top-left, "
            "top-right, bottom-right, bottom-left. If an existing logo, text or graphic is "
            "present, tightly bound the complete old artwork in existing_print_box and "
            "existing_print_quad. existing_print_coverable is true only when all old artwork "
            "is visible, not crossed by hands, hair, a bag, seams or deep folds, and occupies "
            "at most 10 percent of the usable garment panel. fabric_reconstruction_safe is "
            "true only for solid or mildly shaded cotton where nearby clean fabric can rebuild "
            "the old-print area. Reject acid wash, heavy mottling, gradients, complex texture, "
            "strong folds, occlusion, extreme perspective, full-body distance, or hidden print "
            "area. A plain shirt can be suitable without cleanup. suitable requires at least "
            "88 percent print-area visibility, a valid target box or quad, and safe local "
            "preparation. Keep the reason short. Existing catalog tags: "
            f"{tags}."
        )
        response = self.client.models.generate_content(
            model=self.analysis_model,
            contents=[
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SimpleReferencePreparation,
                temperature=0,
            ),
        )
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, SimpleReferencePreparation):
            result = parsed
        elif parsed is not None:
            result = SimpleReferencePreparation.model_validate(parsed)
        elif response.text:
            result = SimpleReferencePreparation.model_validate_json(response.text)
        else:
            raise ReferenceImportError(
                "Gemini не подготовил геометрию простого референса"
            )
        geometry_ok = (
            result.target_print_box is not None
            or len(result.target_print_quad) == 4
        )
        old_geometry_ok = (
            not result.existing_print_present
            or result.existing_print_box is not None
            or len(result.existing_print_quad) == 4
        )
        suitable = (
            result.suitable
            and result.print_area_visibility >= 88
            and geometry_ok
            and old_geometry_ok
            and (
                not result.existing_print_present
                or (
                    result.existing_print_coverable
                    and result.fabric_reconstruction_safe
                    and result.existing_print_coverage_percent <= 10
                )
            )
        )
        if not suitable and result.suitable:
            return result.model_copy(
                update={
                    "suitable": False,
                    "reason": "Геометрия или ткань не прошли локальную проверку",
                }
            )
        return result

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
            "back DTF design could be placed while keeping the pose. garment_is_plain is "
            "true only when the visible target panel has no logo, text or illustration. "
            "existing_print_coverage_percent estimates how much of that panel is occupied "
            "by existing artwork. Mark usable false if "
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
        simple_only: bool = False,
        shirt_color: str = "",
        fit: str = "",
        rng: Optional[random.Random] = None,
    ) -> Optional[ReferenceAsset]:
        excluded = set(exclude_ids)
        mood_set = set(moods)
        scored: list[tuple[float, ReferenceAsset, list[str]]] = []
        for asset in self.repository.list_ready_reference_assets():
            if asset.id in excluded:
                continue
            if simple_only and not asset.simple_ready:
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

            score, score_reasons = self.score_reference(
                asset=asset,
                garment_type=garment_type,
                target_gender=target_gender,
                moods=moods,
                print_side=print_side,
                shirt_color=shirt_color,
                fit=fit,
                season=season,
            )
            score -= min(15, asset.use_count * 2)
            if preferred_source_name and asset.source_name == preferred_source_name:
                score = min(100, score + 8)
            scored.append((float(score), asset, score_reasons))
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
        for score, asset, score_reasons in candidates:
            if self.repository.reserve_reference(
                asset.id,
                request_token=token,
                garment_type=garment_type,
                target_gender=target_gender,
                moods=list(moods),
            ):
                self.repository.update_reference_match(
                    asset.id,
                    score=int(round(score)),
                    reason="; ".join(score_reasons),
                )
                refreshed = self.repository.get_reference_asset(asset.id)
                return refreshed or asset
        return None

    def score_reference(
        self,
        *,
        asset: ReferenceAsset,
        garment_type: GarmentTag,
        target_gender: Literal["women", "men", "unisex"],
        moods: Sequence[MoodTag],
        print_side: Optional[Literal["front", "back"]] = None,
        shirt_color: str = "",
        fit: str = "",
        season: Optional[Literal["warm", "cold", "all-season"]] = None,
    ) -> tuple[int, list[str]]:
        tags = asset.tags or {}
        score = 0
        reasons: list[str] = []
        garments = set(tags.get("garment_types", []))
        if garment_type in garments:
            score += 20
            reasons.append("тип изделия совпадает")
        gender = str(tags.get("gender", "unisex"))
        if target_gender == "unisex" or gender == target_gender:
            score += 10
            reasons.append("категория модели совпадает")
        elif gender == "unisex":
            score += 7
            reasons.append("унисекс референс")

        visible_side = str(tags.get("print_side_visible", "unclear"))
        side_ok = (
            print_side is None
            or visible_side == "both"
            or (print_side == "front" and visible_side in {"front", "cap-front"})
            or (print_side == "back" and visible_side == "back")
        )
        if side_ok:
            score += 20
            reasons.append("сторона принта совпадает")

        visibility = max(0, min(100, int(tags.get("print_area_visibility", 0) or 0)))
        score += round(visibility * 0.20)
        reasons.append(f"видимость зоны {visibility}%")

        framing = str(tags.get("framing", ""))
        framing_points = {
            "close-up": 10,
            "waist-up": 10,
            "detail": 8,
            "three-quarter": 6,
            "full-body": 2,
        }.get(framing, 4)
        score += framing_points
        if framing_points >= 8:
            reasons.append("подходящее кадрирование")

        mood_matches = len(set(moods).intersection(tags.get("moods", [])))
        score += min(10, mood_matches * 5)
        if mood_matches:
            reasons.append("настроение совпадает")

        if asset.simple_ready:
            score += 10
            reasons.append(f"подготовлен для простого режима, уровень {asset.simple_level}")
        elif bool(tags.get("garment_is_plain", False)):
            score += 8
            reasons.append("чистая зона изделия")
        else:
            coverage = max(0, min(100, int(tags.get("existing_print_coverage_percent", 0) or 0)))
            score -= min(10, round(coverage / 4))
            if coverage:
                reasons.append(f"чужой принт занимает {coverage}%")

        if season and tags.get("season") in {season, "all-season"}:
            score += 4
        if shirt_color:
            try:
                color_match = self._fallback_color_match(
                    image_bytes=asset.simple_image_bytes or asset.image_bytes,
                    target_color=shirt_color,
                    torso=self._fallback_torso_box(
                        framing=str(tags.get("framing", "waist-up")),
                        camera_angle=str(tags.get("camera_angle", "front")),
                    ),
                )
            except Exception:
                color_match = False
            if color_match:
                score += 5
                reasons.append("цвет близок к макету")
        if fit and fit.casefold() in str(tags.get("composition_notes", "")).casefold():
            score += 3
            reasons.append("крой похож")

        learning = min(10, asset.success_count * 3) - min(10, asset.failure_count * 2)
        score += learning
        if asset.success_count:
            reasons.append(f"успешных применений: {asset.success_count}")
        if asset.failure_count:
            reasons.append(f"неудачных применений: {asset.failure_count}")
        return max(0, min(100, int(round(score)))), reasons

    def fallback_reference_compatibility(
        self,
        *,
        asset: ReferenceAsset,
        garment_type: GarmentTag,
        print_side: Literal["front", "back"],
        target_shirt_color: str = "",
        print_width_percent: int = 30,
        print_height_percent: int = 30,
        print_top_offset_percent: int = 15,
    ) -> ReferenceCompatibility:
        """Build a conservative preflight from the tags already stored for a reference.

        This is used when the optional Gemini vision preflight is unavailable. It
        never approves a clearly incompatible side or camera angle. Local compositing
        is enabled only for plain, close, high-visibility t-shirts with a matching
        estimated garment color. Other compatible references remain usable by Gemini.
        """
        tags = asset.tags or {}
        visible_side = str(tags.get("print_side_visible", "unclear"))
        if visible_side not in {"front", "back", "both", "cap-front", "unclear"}:
            visible_side = "unclear"
        camera_angle = str(tags.get("camera_angle", "unclear"))
        if camera_angle not in {
            "front", "rear", "three-quarter", "side", "high", "low", "mirror", "unclear"
        }:
            camera_angle = "unclear"
        visibility = max(0, min(100, int(tags.get("print_area_visibility", 0) or 0)))
        framing = str(tags.get("framing", ""))

        side_ok = (
            visible_side in {"back", "both"}
            if print_side == "back"
            else visible_side in {"front", "both", "cap-front"}
        )
        angle_ok = (
            camera_angle in {"rear", "three-quarter"}
            if print_side == "back"
            else camera_angle not in {"rear", "side", "unclear"}
        )
        compatible = bool(tags.get("usable", True)) and side_ok and angle_ok and visibility >= 75

        torso = self._fallback_torso_box(framing=framing, camera_angle=camera_angle)
        target_box = self._fallback_print_box(
            torso=torso,
            print_width_percent=print_width_percent,
            print_height_percent=print_height_percent,
            print_top_offset_percent=print_top_offset_percent,
        )
        garment_color_match = self._fallback_color_match(
            image_bytes=asset.image_bytes,
            target_color=target_shirt_color,
            torso=torso,
        )
        existing_coverage = max(
            0, min(100, int(tags.get("existing_print_coverage_percent", 0) or 0))
        )
        plain_value = tags.get("garment_is_plain")
        garment_is_plain = plain_value is True
        existing_present = plain_value is False or existing_coverage > 0
        # Without Gemini geometry, only a plain garment or a very small existing
        # print is safe for local cleanup. Larger prints require an exact box.
        existing_coverable = garment_is_plain or existing_coverage <= 8
        existing_box = target_box if existing_present and existing_coverage <= 8 else None
        fabric_reconstruction_safe = garment_is_plain or existing_coverage <= 8
        local_angle_ok = camera_angle in {"front", "rear"}
        local_framing_ok = framing in {"detail", "close-up", "waist-up"}
        local_safe = (
            compatible
            and garment_type == "t-shirt"
            and visibility >= 88
            and local_angle_ok
            and local_framing_ok
            and garment_color_match
            and existing_coverable
            and fabric_reconstruction_safe
            and target_box is not None
        )

        if not compatible:
            reason = "Сохраненные теги показывают неподходящую сторону, ракурс или видимость"
        elif local_safe:
            reason = "Проверено локально по сохраненным тегам и цвету изделия"
        else:
            reason = "Референс подходит для Gemini, но локальная замена требует осторожности"

        return ReferenceCompatibility(
            compatible=compatible,
            visible_side=visible_side,
            camera_angle=camera_angle,
            print_area_visibility=visibility,
            target_print_box=target_box,
            target_print_quad=[],
            garment_color_match=garment_color_match,
            existing_print_present=existing_present,
            existing_print_box=existing_box,
            existing_print_quad=[],
            existing_print_coverage_percent=existing_coverage,
            existing_print_coverable=existing_coverable,
            fabric_reconstruction_safe=fabric_reconstruction_safe,
            local_composite_safe=local_safe,
            reason=reason,
        )

    @staticmethod
    def _fallback_torso_box(*, framing: str, camera_angle: str) -> PlacementBox:
        presets = {
            "detail": (15.0, 18.0, 70.0, 66.0),
            "close-up": (17.0, 25.0, 66.0, 58.0),
            "waist-up": (20.0, 29.0, 60.0, 54.0),
            "three-quarter": (25.0, 25.0, 50.0, 47.0),
            "full-body": (34.0, 23.0, 32.0, 35.0),
        }
        x, y, width, height = presets.get(framing, presets["waist-up"])
        if camera_angle == "three-quarter":
            width *= 0.92
        return PlacementBox(x=x, y=y, width=width, height=height)

    @staticmethod
    def _fallback_print_box(
        *,
        torso: PlacementBox,
        print_width_percent: int,
        print_height_percent: int,
        print_top_offset_percent: int,
    ) -> PlacementBox:
        width = max(5.0, min(torso.width * 0.92, torso.width * print_width_percent / 100.0))
        height = max(5.0, min(torso.height * 0.78, torso.height * print_height_percent / 100.0))
        x = torso.x + (torso.width - width) / 2.0
        y = torso.y + torso.height * max(0, min(60, print_top_offset_percent)) / 100.0
        if y + height > torso.y + torso.height:
            y = torso.y + torso.height - height
        return PlacementBox(
            x=max(0.0, min(99.0, x)),
            y=max(0.0, min(99.0, y)),
            width=max(1.0, min(100.0 - x, width)),
            height=max(1.0, min(100.0 - y, height)),
        )

    @classmethod
    def _fallback_color_match(
        cls,
        *,
        image_bytes: bytes,
        target_color: str,
        torso: PlacementBox,
    ) -> bool:
        target = cls._color_family_from_name(target_color)
        if target == "unknown":
            return False
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                rgb = ImageOps.exif_transpose(image).convert("RGB")
                width, height = rgb.size
                x0 = int(width * torso.x / 100.0)
                y0 = int(height * torso.y / 100.0)
                x1 = int(width * (torso.x + torso.width) / 100.0)
                y1 = int(height * (torso.y + torso.height) / 100.0)
                crop = rgb.crop((x0, y0, max(x0 + 1, x1), max(y0 + 1, y1))).resize((48, 48))
                pixels = np.asarray(crop, dtype=np.float32).reshape(-1, 3)
        except (UnidentifiedImageError, OSError, ValueError):
            return False
        median = np.median(pixels, axis=0)
        observed = cls._color_family_from_rgb(float(median[0]), float(median[1]), float(median[2]))
        if target == observed:
            return True
        return {target, observed} <= {"black", "gray"} or {target, observed} <= {"white", "beige"}

    @staticmethod
    def _color_family_from_name(value: str) -> str:
        clean = value.casefold()
        groups = (
            ("black", ("black", "charcoal", "dark grey", "dark gray", "черн", "gara")),
            ("white", ("white", "бел", "ak ", "ak")),
            ("gray", ("grey", "gray", "сер", "silver")),
            ("beige", ("beige", "cream", "sand", "tan", "беж", "крем")),
            ("red", ("red", "burgundy", "maroon", "крас", "борд")),
            ("blue", ("blue", "navy", "cyan", "син", "голуб")),
            ("green", ("green", "khaki", "olive", "зел", "хаки")),
            ("pink", ("pink", "rose", "роз")),
            ("brown", ("brown", "chocolate", "корич")),
        )
        for family, words in groups:
            if any(word in clean for word in words):
                return family
        return "unknown"

    @staticmethod
    def _color_family_from_rgb(red: float, green: float, blue: float) -> str:
        high = max(red, green, blue)
        low = min(red, green, blue)
        spread = high - low
        light = (red + green + blue) / 3.0
        if light < 58:
            return "black"
        if light > 215 and spread < 28:
            return "white"
        if spread < 30:
            return "gray"
        if red > 175 and green > 145 and blue < 135:
            return "beige"
        if red > green * 1.18 and red > blue * 1.18:
            if blue > 120 and light > 155:
                return "pink"
            if light < 135 and green > blue:
                return "brown"
            return "red"
        if blue > red * 1.12 and blue > green * 1.05:
            return "blue"
        if green > red * 1.08 and green > blue * 1.05:
            return "green"
        if red > green > blue and light < 155:
            return "brown"
        return "unknown"

    async def validate_reference_for_generation(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        garment_type: GarmentTag,
        print_side: Literal["front", "back"],
        target_shirt_color: str = "",
        target_fit: str = "",
        print_width_percent: int = 30,
        print_height_percent: int = 30,
        print_top_offset_percent: int = 15,
    ) -> ReferenceCompatibility:
        return await asyncio.to_thread(
            self._validate_reference_for_generation_sync,
            image_bytes,
            mime_type,
            garment_type,
            print_side,
            target_shirt_color,
            target_fit,
            print_width_percent,
            print_height_percent,
            print_top_offset_percent,
        )

    def _validate_reference_for_generation_sync(
        self,
        image_bytes: bytes,
        mime_type: str,
        garment_type: GarmentTag,
        print_side: Literal["front", "back"],
        target_shirt_color: str,
        target_fit: str,
        print_width_percent: int,
        print_height_percent: int,
        print_top_offset_percent: int,
    ) -> ReferenceCompatibility:
        prompt = (
            "This is a strict preflight check for a clothing mockup. "
            f"The target product is a {target_shirt_color or 'same-color'} "
            f"{target_fit or ''} {garment_type} with a {print_side} print. "
            f"The source print uses about {print_width_percent}% of the garment width, "
            f"{print_height_percent}% of its height and begins about "
            f"{print_top_offset_percent}% below the collar. "
            "Judge whether this photo can be used in two ways: as a pose reference for "
            "Gemini, and as a direct local composite where only the garment artwork is "
            "replaced. For a back print, the back must be clearly visible from rear or "
            "rear three-quarter view. For a front print, the front panel must be clearly "
            "visible. Reject unclear orientation, distant full-body shots, crossed arms, "
            "hair, bags or props covering the print area, crowds, collages, drawings and "
            "low-quality images. print_area_visibility is the usable torso percentage. "
            "target_print_box is the exact normalized rectangle where the new artwork "
            "should sit. target_print_quad contains exactly four normalized points in "
            "this order: top-left, top-right, bottom-right, bottom-left. The quad must "
            "follow mild garment perspective. garment_color_match is true only if the "
            "visible garment color and wash are close enough to the requested product. "
            "Set existing_print_present true if the visible garment already contains any "
            "text, logo or graphic in the usable panel. If present, existing_print_box must "
            "tightly surround the complete old artwork and existing_print_quad must contain "
            "four normalized corner points in the same order as target_print_quad. Estimate "
            "existing_print_coverage_percent against the usable garment panel. "
            "existing_print_coverable is true when the complete old artwork is visible, lies "
            "on an open t-shirt panel and occupies no more than roughly 10% of that panel. "
            "fabric_reconstruction_safe is true only when the fabric behind the old print is "
            "simple enough to rebuild locally: solid or mildly shaded cotton with no acid-wash "
            "pattern, heavy texture, strong seam, deep fold or complex multicolor fabric. A "
            "plain garment is preferred but is not required. local_composite_safe may be true "
            "with an existing print only when its exact box or quad is supplied, it is not more "
            "than about 1.5 times the new target area, the garment "
            "color matches, fabric reconstruction is safe, perspective is mild, visibility is "
            "at least 88%, and there is no occlusion or severe fold. Keep reason short and objective."
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
        quad_valid = len(result.target_print_quad) in {0, 4}
        existing_quad_valid = len(result.existing_print_quad) in {0, 4}
        existing_geometry_ok = (
            not result.existing_print_present
            or result.existing_print_box is not None
            or len(result.existing_print_quad) == 4
        )
        existing_size_ok = not result.existing_print_present
        if result.existing_print_present:
            existing_size_ok = result.existing_print_coverage_percent <= 10
            if result.existing_print_box is not None and result.target_print_box is not None:
                existing_area = (
                    result.existing_print_box.width * result.existing_print_box.height
                )
                target_area = max(
                    0.01,
                    result.target_print_box.width * result.target_print_box.height,
                )
                existing_size_ok = (
                    existing_size_ok and existing_area / target_area <= 0.80
                )
        cleanup_ready = (
            not result.existing_print_present
            or (
                result.existing_print_coverable
                and result.fabric_reconstruction_safe
            )
        )
        local_safe = (
            compatible
            and garment_type == "t-shirt"
            and result.local_composite_safe
            and result.garment_color_match
            and cleanup_ready
            and result.print_area_visibility >= 88
            and result.target_print_box is not None
            and quad_valid
            and existing_quad_valid
            and existing_geometry_ok
            and existing_size_ok
        )
        updates = {
            "compatible": compatible,
            "local_composite_safe": local_safe,
        }
        if not compatible:
            updates["reason"] = "Сторона, ракурс или видимость зоны принта не подходят"
        elif result.local_composite_safe and not local_safe:
            updates["reason"] = "Референс подходит для Gemini, но не для локальной замены"
        return result.model_copy(update=updates)

    def status_text(self) -> str:
        stats = self.repository.reference_stats()
        simple_stats = self.repository.simple_reference_stats()
        queue_details = self.repository.reference_queue_details()
        assets = self.repository.list_ready_reference_assets()
        all_assets = self.repository.list_reference_assets(limit=1000)
        lifecycle: Counter[str] = Counter(asset.lifecycle_state for asset in all_assets)
        levels: Counter[str] = Counter(
            asset.simple_level
            for asset in all_assets
            if asset.simple_status in {"ready", "skipped"}
        )
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

        worker_status = self.repository.get_setting("simple_worker_status") or "не запускался"
        worker_current = self.repository.get_setting("simple_worker_current_reference") or "0"
        worker_error = self.repository.get_setting("simple_worker_last_error") or ""
        worker_line = f"Фоновая подготовка: {worker_status}"
        if worker_current not in {"", "0"}:
            worker_line += f", референс #{worker_current}"
        if worker_error:
            worker_line += f"\nПоследняя ошибка фоновой подготовки: {worker_error[:180]}"

        return (
            "Каталог референсов\n"
            f"Всего ссылок: {stats.get('total', 0)}\n"
            f"Загружено и проанализировано: {stats.get('ready', 0)}\n"
            f"Для простого режима: {simple_stats.get('ready', 0)} готово, "
            f"{simple_stats.get('pending', 0)} ожидают подготовки, "
            f"{simple_stats.get('processing', 0)} обрабатывается, "
            f"{simple_stats.get('skipped', 0)} отклонено\n"
            f"{worker_line}\n"
            f"Состояния: RAW {lifecycle.get('raw', 0)}, "
            f"PREPARED {lifecycle.get('prepared', 0)}, "
            f"MATCHED {lifecycle.get('matched', 0)}, "
            f"SUCCESSFUL {lifecycle.get('successful', 0)}\n"
            f"Уровни простого режима: A {levels.get('A', 0)}, "
            f"B {levels.get('B', 0)}, C {levels.get('C', 0)}\n"
            + "\n".join(waiting_lines)
            + "\n"
            f"Не подходят: {stats.get('disabled', 0)}\n"
            f"Ошибки: {stats.get('failed', 0)}\n\n"
            f"По одежде: {garment_line}\n"
            f"По полу: женщины {genders.get('women', 0)}, "
            f"мужчины {genders.get('men', 0)}, унисекс {genders.get('unisex', 0)}\n\n"
            f"Цель: минимум {self.min_pool_size} доступных фото для каждой "
            "используемой категории.\n"
            f"Поиск референсов: "
            f"{self.repository.get_setting('pinterest_discovery_status') or 'еще не запускался'}"
        )
