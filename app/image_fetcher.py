import logging
import re
from typing import Optional, List, Tuple
from urllib.parse import quote_plus
import aiohttp
from aiogram.types import BufferedInputFile

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


async def fetch_image_from_url(url: str, timeout: int = 15) -> Tuple[bytes, str]:
    """
    Скачивает изображение по прямому URL.
    Возвращает кортеж (байты, имя_файла).
    """
    headers = {"User-Agent": USER_AGENT}
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status != 200:
                raise ValueError(f"HTTP ошибка {resp.status} при загрузке картинки")
            
            content_type = resp.headers.get("Content-Type", "").lower()
            if "image" not in content_type and not any(ext in url.lower() for ext in [".jpg", ".jpeg", ".png", ".webp", ".gif"]):
                logger.warning("URL %s может не являться прямым изображением (%s)", url, content_type)
            
            data = await resp.read()
            if len(data) == 0:
                raise ValueError("Загруженный файл пуст")

            # Определение расширения
            ext = ".jpg"
            if "png" in content_type or url.lower().endswith(".png"):
                ext = ".png"
            elif "webp" in content_type or url.lower().endswith(".webp"):
                ext = ".webp"
            elif "gif" in content_type or url.lower().endswith(".gif"):
                ext = ".gif"

            filename = f"web_image{ext}"
            return data, filename


async def search_web_images(query: str, limit: int = 5) -> List[str]:
    """
    Выполняет поиск изображений в интернете по текстовому запросу.
    Возвращает список найденных URL изображений.
    """
    encoded_query = quote_plus(query)
    # Используем апи DuckDuckGo i.js для поиска изображений
    vqd_url = f"https://duckduckgo.com/?q={encoded_query}&iax=images&ia=images"
    headers = {"User-Agent": USER_AGENT}

    urls: List[str] = []

    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            # 1. Получаем vqd токен для DuckDuckGo
            async with session.get(vqd_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                text = await resp.text()
                vqd_match = re.search(r'vqd=([\d-]+)&', text) or re.search(r'vqd="([\d-]+)"', text)
                
            if vqd_match:
                vqd = vqd_match.group(1)
                search_url = (
                    f"https://duckduckgo.com/i.js?l=us-en&o=json&q={encoded_query}"
                    f"&vqd={vqd}&f=,,,&p=1"
                )
                async with session.get(search_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results = data.get("results", [])
                        for item in results[:limit]:
                            image_url = item.get("image")
                            if image_url:
                                urls.append(image_url)

    except Exception as exc:
        logger.error("Ошибка при поиске картинок в DuckDuckGo: %s", exc)

    # Запасной фоллбек на Unsplash Source / LoremFlickr по ключевым словам
    if not urls:
        clean_keyword = quote_plus(query.split()[0] if query else "nature")
        urls.append(f"https://loremflickr.com/800/600/{clean_keyword}")

    return urls
