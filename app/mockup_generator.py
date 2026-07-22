import asyncio
import base64
import logging
import random
import secrets
from dataclasses import dataclass
from typing import Literal, Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class MockupSpec(BaseModel):
    side: Literal["front", "back"]
    garment_type: Literal["t-shirt", "hoodie", "sweatshirt", "long-sleeve"]
    shirt_color: str = Field(min_length=1, max_length=80)
    fabric_finish: str = Field(min_length=1, max_length=100)
    fit: str = Field(min_length=1, max_length=80)
    print_width_percent: int = Field(ge=5, le=100)
    print_height_percent: int = Field(ge=3, le=100)
    print_top_from_collar_percent: int = Field(ge=0, le=80)


@dataclass(frozen=True)
class PhotoDirection:
    label: str
    person: str
    setting: str
    pose: str
    camera: str
    seed: int


@dataclass(frozen=True)
class GeneratedModelPhoto:
    data: bytes
    mime_type: str

    @property
    def extension(self) -> str:
        return "png" if self.mime_type == "image/png" else "jpg"


class MockupGenerationError(RuntimeError):
    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


_PHOTO_DIRECTIONS = (
    (
        "Городской дневной свет",
        "a fictional Central Asian man in his mid twenties, short dark hair, "
        "clean-shaven, a distinctive natural face",
        "a quiet contemporary city street in soft daylight",
        "standing naturally with relaxed shoulders and both hands visible",
        "eye-level 50 mm editorial fashion photograph",
    ),
    (
        "Теплый интерьер",
        "a fictional Central Asian woman in her late twenties, shoulder-length "
        "dark hair, a distinctive natural face",
        "a warm minimal interior with wood and indirect window light",
        "a casual three-quarter stance, arms clear of the printed area",
        "eye-level 50 mm lifestyle photograph with realistic skin texture",
    ),
    (
        "Золотой час",
        "a fictional man in his early thirties, wavy dark hair, short neat beard, "
        "a distinctive natural face",
        "an open urban promenade during golden hour",
        "walking slowly and looking slightly away from the camera",
        "natural 35 mm street-style fashion photograph",
    ),
    (
        "Современный лифт",
        "a fictional woman in her mid twenties, dark bob haircut, a distinctive "
        "natural face",
        "a clean modern metal elevator with cinematic practical lighting",
        "turning naturally toward the camera without covering the shirt",
        "realistic 35 mm editorial photograph with controlled highlights",
    ),
    (
        "Архитектура",
        "a fictional man in his late twenties, dark curly hair, a distinctive "
        "natural face",
        "minimal concrete architecture under a clear blue sky",
        "a relaxed low-energy pose with the full shirt unobstructed",
        "slightly low-angle 50 mm fashion photograph without lens distortion",
    ),
    (
        "Светлая студия",
        "a fictional Central Asian woman in her early thirties, dark hair tied "
        "back, a distinctive natural face",
        "a bright neutral daylight studio with a soft textured wall",
        "standing comfortably with one hand near a pocket and the other visible",
        "clean 70 mm commercial apparel photograph",
    ),
    (
        "Кофейня",
        "a fictional man in his mid thirties, close-cropped dark hair, light "
        "stubble, a distinctive natural face",
        "outside a modern cafe with softly blurred city details",
        "leaning lightly against a wall while keeping the print fully visible",
        "candid 50 mm lifestyle photograph with shallow depth of field",
    ),
    (
        "Зеленый интерьер",
        "a fictional woman in her late twenties, long loose dark curls, a "
        "distinctive natural face",
        "a modern sunlit interior with large natural green plants",
        "a calm over-the-shoulder pose appropriate to the printed side",
        "natural 50 mm editorial photograph with soft window light",
    ),
    (
        "Пустынный пейзаж",
        "a fictional Central Asian man in his late twenties, medium-length dark "
        "hair, a distinctive natural face",
        "a quiet road near dry foothills under a dramatic but realistic sky",
        "standing at ease and turning only enough to show the correct shirt side",
        "documentary-style 50 mm fashion photograph",
    ),
    (
        "Галерея",
        "a fictional Central Asian woman in her mid thirties, straight dark hair, "
        "a distinctive natural face",
        "a spacious contemporary art gallery with neutral walls",
        "a composed editorial pose with arms away from the torso",
        "high-end 70 mm apparel campaign photograph",
    ),
)


def choose_photo_directions(
    count: int,
    rng: Optional[random.Random] = None,
) -> list[PhotoDirection]:
    if count < 1:
        raise ValueError("Количество вариантов должно быть больше нуля")
    picker = rng or secrets.SystemRandom()
    if count <= len(_PHOTO_DIRECTIONS):
        selected = picker.sample(_PHOTO_DIRECTIONS, count)
    else:
        selected = [picker.choice(_PHOTO_DIRECTIONS) for _ in range(count)]
    return [
        PhotoDirection(
            label=item[0],
            person=item[1],
            setting=item[2],
            pose=item[3],
            camera=item[4],
            seed=picker.randrange(1, 2_147_483_647),
        )
        for item in selected
    ]


def build_model_photo_prompt(
    spec: Optional[MockupSpec],
    direction: PhotoDirection,
    request_token: str,
) -> str:
    if spec is None:
        measurements = (
            "Infer the printed side, garment color, fabric finish, fit, exact print "
            "width ratio, exact print height ratio and collar-to-print distance "
            "directly from the supplied product mockup."
        )
    else:
        measurements = (
            f"The printed side is the {spec.side}. The garment is a "
            f"{spec.garment_type}, color: {spec.shirt_color}, fabric finish: "
            f"{spec.fabric_finish}, fit: {spec.fit}. The print width is about "
            f"{spec.print_width_percent}% of the wearable torso panel width. "
            f"The print height is about {spec.print_height_percent}% of the shirt "
            f"height from collar to hem. Its top begins about "
            f"{spec.print_top_from_collar_percent}% of that height below the collar."
        )

    return (
        "Create one photorealistic commercial lifestyle photograph for a real "
        "clothing shop. The supplied image is the only product reference. It is "
        "a flat garment mockup, not a composition to copy. Ignore the surrounding "
        "background, presentation graphics, shadows, watermarks and any writing "
        "outside the physical garment.\n\n"
        "PRODUCT FIDELITY IS THE HIGHEST PRIORITY:\n"
        "1. Treat the complete artwork already printed inside the garment as locked "
        "source artwork. Transfer it exactly as one unchanged visual texture.\n"
        "2. Preserve every visible letter, number, face inside the artwork, line, "
        "ornament, color, spacing, boundary and aspect ratio. Do not redraw, rewrite, "
        "translate, correct, simplify, crop, extend, recolor, duplicate or invent any "
        "part of the print. Add no new text or logos.\n"
        "3. People or faces that are part of the printed artwork must remain only "
        "inside that artwork. The real wearer must be a completely different, "
        "fictional, non-celebrity adult.\n"
        "4. Keep the print on the same front or back side shown by the source. Never "
        "move it to the opposite side.\n"
        "5. Match the shirt color, washed or clean fabric finish, neckline, cut and "
        "sleeve proportions from the source.\n"
        "6. Preserve the exact relative print scale and placement from the source. "
        "Do not enlarge the design to fill the shirt. The design may follow natural "
        "fabric folds and perspective, but it must remain complete, unobstructed and "
        "easy to inspect.\n"
        f"7. {measurements}\n\n"
        "NEW PHOTO DIRECTION:\n"
        f"- Wearer: {direction.person}.\n"
        f"- Location: {direction.setting}.\n"
        f"- Pose: {direction.pose}.\n"
        f"- Camera: {direction.camera}.\n"
        "- Style the wearer simply with neutral trousers or jeans. No jacket, bag, "
        "hair, arm, jewelry or accessory may cover the print.\n"
        "- Use believable anatomy, natural hands, realistic pores, individual hair "
        "strands, true fabric weight, seams, wrinkles, print integration, lighting "
        "and shadows. Avoid beauty-filter skin and the synthetic AI look.\n\n"
        "COMPOSITION AND SAFE AREA:\n"
        "- Vertical 4:5 image for a Telegram and social-media product post.\n"
        "- Frame approximately from the top of the head to mid-thigh so the shirt "
        "and print stay large enough to inspect. Avoid a distant full-body shot.\n"
        "- For a front print, show the wearer's natural face clearly. For a back "
        "print, show the back squarely and use only a natural partial side profile "
        "without twisting or hiding the shirt.\n"
        "- Keep the complete head, both shoulders, both sleeves, the entire printed "
        "area and the shirt hem inside the frame.\n"
        "- Keep the face and the full garment within the central 80% of the image, "
        "with at least 8% breathing room on every edge so interface previews on "
        "different devices do not cut important content.\n"
        "- The shirt and its print are the clear focus. No collage, split screen, "
        "mockup board, border, caption, watermark or extra graphic.\n"
        f"- Variation token: {request_token}-{direction.seed}. Use it only to make "
        "this wearer and photographic moment different from all other variants."
    )


class MockupGenerator:
    def __init__(
        self,
        *,
        api_key: str,
        analysis_model: str,
        image_model: str,
        image_size: str,
        aspect_ratio: str = "4:5",
    ):
        self.client = genai.Client(api_key=api_key)
        self.analysis_model = analysis_model
        self.image_model = image_model
        self.image_size = image_size
        self.aspect_ratio = aspect_ratio

    async def analyze_mockup(
        self,
        image_bytes: bytes,
        mime_type: str,
    ) -> Optional[MockupSpec]:
        try:
            return await asyncio.to_thread(
                self._analyze_mockup_sync,
                image_bytes,
                mime_type,
            )
        except Exception:
            logger.exception("Не удалось измерить макет, используется визуальный анализ")
            return None

    def _analyze_mockup_sync(
        self,
        image_bytes: bytes,
        mime_type: str,
    ) -> MockupSpec:
        prompt = (
            "Analyze this flat clothing product mockup. Inspect only the physical "
            "garment and the artwork printed inside it. Ignore all presentation "
            "background graphics and watermarks outside the garment. Return whether "
            "the printed artwork is shown on the front or back, garment type, exact "
            "visible garment color, fabric finish and fit. Estimate the artwork's "
            "bounding box: width as a percentage of the wearable torso panel between "
            "side seams, height as a percentage of shirt height from collar to hem, "
            "and the top distance from the collar as a percentage of that shirt "
            "height. Measure the full printed artwork including all of its text."
        )
        response = self.client.models.generate_content(
            model=self.analysis_model,
            contents=[
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=MockupSpec,
            ),
        )
        if not response.text:
            raise RuntimeError("Gemini не вернул параметры макета")
        return MockupSpec.model_validate_json(response.text)

    async def generate_variant(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        spec: Optional[MockupSpec],
        direction: PhotoDirection,
        request_token: str,
    ) -> GeneratedModelPhoto:
        try:
            return await asyncio.to_thread(
                self._generate_variant_sync,
                image_bytes,
                mime_type,
                spec,
                direction,
                request_token,
            )
        except MockupGenerationError:
            raise
        except Exception as error:
            logger.exception("Не удалось создать фото на модели")
            raise self._friendly_error(error) from error

    def _generate_variant_sync(
        self,
        image_bytes: bytes,
        mime_type: str,
        spec: Optional[MockupSpec],
        direction: PhotoDirection,
        request_token: str,
    ) -> GeneratedModelPhoto:
        prompt = build_model_photo_prompt(spec, direction, request_token)
        response = self.client.models.generate_content(
            model=self.image_model,
            contents=[
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio=self.aspect_ratio,
                    image_size=self.image_size,
                ),
            ),
        )
        for part in response.parts or []:
            inline_data = getattr(part, "inline_data", None)
            if inline_data is None or not inline_data.data:
                continue
            data = inline_data.data
            if isinstance(data, str):
                data = base64.b64decode(data)
            return GeneratedModelPhoto(
                data=bytes(data),
                mime_type=inline_data.mime_type or "image/jpeg",
            )
        raise MockupGenerationError(
            "Gemini не вернул изображение. Попробуйте этот макет еще раз."
        )

    @staticmethod
    def _friendly_error(error: Exception) -> MockupGenerationError:
        code = getattr(error, "code", None)
        message = str(error).upper()
        if any(
            marker in message
            for marker in (
                "BILLING",
                "PAID TIER",
                "FREE TIER",
                "FREE_TIER",
                "FAILED_PRECONDITION",
                "PERMISSION_DENIED",
                "PAYMENT",
                "LIMIT: 0",
            )
        ):
            return MockupGenerationError(
                "Для генерации фотографий Google требует включенный платный тариф "
                "Gemini API. Обычное создание и публикация постов продолжает "
                "работать бесплатно."
            )
        if code == 429 or "RESOURCE_EXHAUSTED" in message or "429" in message:
            return MockupGenerationError(
                "Лимит Gemini временно исчерпан. Подождите несколько минут и "
                "нажмите «Еще варианты»."
            )
        if code == 404 or "404 NOT_FOUND" in message:
            return MockupGenerationError(
                "Модель генерации изображений недоступна для этого ключа Gemini. "
                "Проверьте GEMINI_IMAGE_MODEL в Render."
            )
        if code in {401, 403} or any(
            marker in message
            for marker in ("UNAUTHENTICATED", "API_KEY_INVALID", "INVALID API KEY")
        ):
            return MockupGenerationError(
                "Ключ Gemini API недействителен или не имеет доступа к модели. "
                "Проверьте GEMINI_API_KEY в Render."
            )
        if code == 400 or "INVALID_ARGUMENT" in message:
            return MockupGenerationError(
                "Gemini отклонил параметры запроса, ошибка 400. Установите "
                "исправленную версию бота и попробуйте еще раз."
            )
        if "SAFETY" in message or "BLOCK" in message:
            return MockupGenerationError(
                "Gemini отклонил этот макет из-за фильтра безопасности. Попробуйте "
                "отправить изображение без лишнего фона."
            )
        return MockupGenerationError(
            "Не удалось создать фотографию. Проверьте доступ Gemini API и "
            "попробуйте снова."
        )
