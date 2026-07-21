import asyncio
import re
from typing import Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, Field


class ProductCopy(BaseModel):
    garment_type: Literal[
        "Футболка",
        "Худи",
        "Свитшот",
        "Лонгслив",
        "Кепка",
    ]
    design_name: str = Field(min_length=1, max_length=55)
    mood_description: str = Field(min_length=1, max_length=140)
    theme_hashtag: str = Field(min_length=1, max_length=40)

    @property
    def title(self) -> str:
        name = self.design_name.strip().strip('"“”«»').strip()
        return f'{self.garment_type} "{name}"'

    @property
    def hashtags(self) -> str:
        garment_hashtags = {
            "Футболка": "#футболка",
            "Худи": "#худи",
            "Свитшот": "#свитшот",
            "Лонгслив": "#лонгслив",
            "Кепка": "#кепка",
        }
        garment_hashtag = garment_hashtags[self.garment_type]
        raw_theme = self.theme_hashtag.strip().lstrip("#").lower()
        theme = re.sub(r"[^\w]+", "_", raw_theme, flags=re.UNICODE).strip("_")
        theme_hashtag = f"#{theme}" if theme else "#принт"
        if theme_hashtag == garment_hashtag:
            theme_hashtag = "#принт"
        return f"{garment_hashtag} {theme_hashtag}"

    @property
    def description(self) -> str:
        return " ".join(self.mood_description.strip().split())


LANGUAGE_NAMES = {
    "ru": "русском",
    "tk": "разговорном туркменском",
}

FALLBACK_MODELS = (
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
)


class ImageCopywriter:
    def __init__(self, *, api_key: str, model: str, language: str):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.language = LANGUAGE_NAMES.get(language, language)

    async def create_copy(self, image_bytes: bytes, mime_type: str) -> ProductCopy:
        return await asyncio.to_thread(self._create_copy_sync, image_bytes, mime_type)

    def _create_copy_sync(self, image_bytes: bytes, mime_type: str) -> ProductCopy:
        prompt = (
            "Ты создаешь название товара для Taýpa, магазина DTF печати на одежде. "
            "Внимательно изучи изображение. Определи тип вещи только из списка: "
            "Футболка, Худи, Свитшот, Лонгслив, Кепка. Если вещь не видна или есть "
            "сомнение, выбери Футболка. Затем придумай короткое запоминающееся название "
            "принта из 2-5 слов по смыслу изображения. Название пиши на "
            f"{self.language} языке, но можешь естественно смешать русский и английский, "
            "если это подходит принту. Создай одно короткое живое описание товара, "
            "которое передает настроение принта. Это должно быть естественное предложение "
            "без упоминания фотографии или искусственного интеллекта. Можно добавить один "
            "подходящий эмодзи в конце. Также создай один короткий тематический хэштег "
            "по содержанию принта, без хэштега с типом вещи. В поле theme_hashtag верни "
            "только одно слово или короткую фразу без пробелов. Не включай тип вещи в "
            "design_name. В design_name не добавляй кавычки, цену, размер, описание или "
            "эмодзи. Не "
            "упоминай фотографию, файл или искусственный интеллект."
        )
        response = None
        models = tuple(dict.fromkeys((self.model, *FALLBACK_MODELS)))
        for index, model in enumerate(models):
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=[
                        prompt,
                        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ProductCopy,
                        temperature=0.4,
                    ),
                )
                self.model = model
                break
            except Exception as error:
                is_last_model = index == len(models) - 1
                if is_last_model or not self._is_model_unavailable(error):
                    raise

        if response is None:
            raise RuntimeError("Нет доступной модели Gemini")
        if not response.text:
            raise RuntimeError("Gemini не вернул тип вещи и название")
        result = ProductCopy.model_validate_json(response.text)
        clean_name = " ".join(result.design_name.strip().split()).strip('"“”«»').strip()
        if not clean_name:
            clean_name = "Новый принт"
        return ProductCopy(
            garment_type=result.garment_type,
            design_name=clean_name[:55],
            mood_description=" ".join(result.mood_description.strip().split())[:140],
            theme_hashtag=result.theme_hashtag,
        )

    @staticmethod
    def _is_model_unavailable(error: Exception) -> bool:
        code = getattr(error, "code", None)
        message = str(error).upper()
        return code == 404 or "404 NOT_FOUND" in message
