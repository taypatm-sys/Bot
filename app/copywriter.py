import asyncio
import re
from typing import Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, Field


_GENERIC_LEADING_WORDS = {
    "стильная",
    "стильный",
    "стильное",
    "модная",
    "модный",
    "модное",
    "красивая",
    "красивый",
    "красивое",
    "яркая",
    "яркий",
    "яркое",
    "необычная",
    "необычный",
    "трендовая",
    "трендовый",
    "stylish",
    "trendy",
    "beautiful",
    "cool",
}


def shorten_design_name(value: str, *, max_words: int = 3, max_chars: int = 28) -> str:
    clean = " ".join(value.strip().strip('"“”«»').split())
    words = clean.split()
    while len(words) > 2 and words[0].casefold().strip(".,!?:;") in _GENERIC_LEADING_WORDS:
        words.pop(0)

    if len(words) > max_words:
        connector_index = next(
            (
                index
                for index, word in enumerate(words)
                if word.casefold().strip(".,!?:;") in {"с", "и", "with", "and", "&"}
                and index > 0
                and index < len(words) - 1
            ),
            None,
        )
        if connector_index is not None:
            next_index = connector_index + 1
            if (
                words[connector_index].casefold() == "with"
                and words[next_index].casefold() in {"a", "an", "the"}
                and next_index + 1 < len(words)
            ):
                next_index += 1
            words = [
                words[connector_index - 1],
                words[connector_index],
                words[next_index],
            ]
        else:
            words = words[:max_words]

    while len(" ".join(words)) > max_chars and len(words) > 1:
        words.pop()
    name = " ".join(words).strip(" .,!?:;-_")
    if not name:
        return "Новый принт"
    if len(name) > max_chars:
        name = name[:max_chars].rstrip(" .,!?:;-_")
    return name[0].upper() + name[1:]


def shorten_description(value: str, *, max_words: int = 14) -> str:
    clean = " ".join(value.strip().split())
    first_sentence = re.split(r"(?<=[.!?])\s+", clean, maxsplit=1)[0]
    words = first_sentence.split()
    if len(words) > max_words:
        first_sentence = " ".join(words[:max_words]).rstrip(" .,!?:;-") + "."
    return first_sentence[:150].strip()


class ProductCopy(BaseModel):
    garment_type: Literal[
        "Футболка",
        "Худи",
        "Свитшот",
        "Лонгслив",
        "Кепка",
        "Зип-худи",
        "Куртка",
        "Шопер",
    ]
    design_name: str = Field(min_length=1, max_length=55)
    mood_description: str = Field(min_length=1, max_length=110)
    theme_hashtag: str = Field(min_length=1, max_length=40)

    @property
    def title(self) -> str:
        name = shorten_design_name(self.design_name)
        return f'{self.garment_type} "{name}"'

    @property
    def hashtags(self) -> str:
        garment_hashtags = {
            "Футболка": "#футболка",
            "Худи": "#худи",
            "Свитшот": "#свитшот",
            "Лонгслив": "#лонгслив",
            "Кепка": "#кепка",
            "Зип-худи": "#зипхуди",
            "Куртка": "#куртка",
            "Шопер": "#шопер",
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
        return shorten_description(self.mood_description)


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
            "Внимательно изучи изображение, но анализируй только само изделие и "
            "напечатанный на нем рисунок. Полностью игнорируй реального человека, "
            "его лицо, позу, фон, локацию, мебель, машину, цветы, покупки и другие "
            "предметы, если они не являются частью самого принта. Определи тип вещи "
            "только из списка: "
            "Футболка, Худи, Свитшот, Лонгслив, Кепка, Зип-худи, Куртка, Шопер. "
            "Если вещь не видна или есть сомнение, выбери Футболка. Затем придумай "
            "очень короткое живое название принта из 1-3 слов, максимум 28 символов. "
            "Если в принте есть короткая главная надпись до трех слов, предпочти ее. "
            "Не пересказывай все изображение и не начинай название словами Стильный, "
            "Стильная, Модный, Модная, Красивый, Красивая или Яркий. Например, вместо "
            "«Стильная восточная красавица с котом» напиши «Красавица с котом» или "
            "«Восточная муза». Название пиши на "
            f"{self.language} языке, но можешь естественно смешать русский и английский, "
            "если это уже есть в принте. Создай один короткий продающий хук из 8-14 слов, "
            "который передает настроение и мягко продает эмоцию, а не перечисляет предметы "
            "на изображении. Это должна быть живая фраза для поста, которую хочется читать. "
            "Можно писать как: «Настроение, которое не объясняют словами» или «Это настроение "
            "лучше показать, чем описывать». Избегай пустых клише вроде «стильный образ» или "
            "«идеально на каждый день». Пиши как человек, без эмодзи, без восклицаний и без "
            "упоминания фотографии или искусственного интеллекта. Также создай один тематический хэштег "
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
        clean_name = shorten_design_name(result.design_name)
        return ProductCopy(
            garment_type=result.garment_type,
            design_name=clean_name,
            mood_description=shorten_description(result.mood_description),
            theme_hashtag=result.theme_hashtag,
        )

    @staticmethod
    def _is_model_unavailable(error: Exception) -> bool:
        code = getattr(error, "code", None)
        message = str(error).upper()
        return code == 404 or "404 NOT_FOUND" in message
