import asyncio
import base64
import io
import logging
from typing import Optional

import httpx
from openai import OpenAI

from app.mockup_generator import (
    GeneratedModelPhoto,
    MockupGenerationError,
    MockupSpec,
    PhotoDirection,
    build_model_photo_prompt,
    prepare_source_print_detail,
)

logger = logging.getLogger(__name__)


class OpenAIMockupGenerator:
    """Creates model photos with OpenAI while keeping Gemini for analysis."""

    def __init__(
        self,
        *,
        api_key: str,
        image_model: str = "gpt-image-1.5",
        image_size: str = "1024x1536",
        image_quality: str = "medium",
    ) -> None:
        self.api_key = api_key.strip()
        self.image_model = image_model
        self.image_size = image_size
        self.image_quality = image_quality
        self._client: Optional[OpenAI] = None

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _get_client(self) -> OpenAI:
        if not self.api_key:
            raise MockupGenerationError(
                "OpenAI API не настроен. Добавьте OPENAI_API_KEY в Render и выполните redeploy."
            )
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, timeout=180.0, max_retries=1)
        return self._client

    @staticmethod
    def _file(data: bytes, filename: str) -> io.BytesIO:
        value = io.BytesIO(data)
        value.name = filename
        return value

    async def generate_variant(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        spec: Optional[MockupSpec],
        direction: PhotoDirection,
        request_token: str,
        print_image_bytes: Optional[bytes] = None,
        print_mime_type: Optional[str] = None,
        reference_image_bytes: Optional[bytes] = None,
        reference_mime_type: Optional[str] = None,
        reference_tags: Optional[dict[str, object]] = None,
        image_model: Optional[str] = None,
    ) -> GeneratedModelPhoto:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    self._generate_sync,
                image_bytes,
                mime_type,
                spec,
                direction,
                request_token,
                print_image_bytes,
                print_mime_type,
                reference_image_bytes,
                reference_mime_type,
                reference_tags,
                    image_model,
                ),
                timeout=240.0,
            )
        except asyncio.TimeoutError as error:
            logger.error("OpenAI generation timed out after 240 seconds")
            raise MockupGenerationError(
                "OpenAI не ответил за 4 минуты. Запрос остановлен. Проверьте логи Render, баланс и доступ к модели."
            ) from error
        except MockupGenerationError:
            raise
        except Exception as error:
            logger.exception("OpenAI не смог создать фото на модели")
            raise self._friendly_error(error) from error

    def _generate_sync(
        self,
        image_bytes: bytes,
        mime_type: str,
        spec: Optional[MockupSpec],
        direction: PhotoDirection,
        request_token: str,
        print_image_bytes: Optional[bytes],
        print_mime_type: Optional[str],
        reference_image_bytes: Optional[bytes],
        reference_mime_type: Optional[str],
        reference_tags: Optional[dict[str, object]],
        image_model: Optional[str],
    ) -> GeneratedModelPhoto:
        source_detail = None
        if not print_image_bytes:
            source_detail = prepare_source_print_detail(image_bytes, spec)

        prompt = build_model_photo_prompt(
            spec,
            direction,
            request_token,
            has_separate_print=bool(print_image_bytes),
            has_source_detail=bool(source_detail),
            has_style_reference=bool(reference_image_bytes),
            style_reference_tags=reference_tags,
        )
        prompt += (
            "\n\nOPENAI IMAGE ORDER:\n"
            "Image 1 is the product source and is the only source of garment color, fabric, cut and print.\n"
            "Image 2, when present, is the high resolution master print or source-detail crop.\n"
            "The last image, when present, is only a pose, camera, lighting and background reference.\n"
            "Never copy any text, logo, graphic or garment color from the reference image."
        )

        images = [self._file(image_bytes, "product-source.png")]
        if print_image_bytes:
            images.append(self._file(print_image_bytes, "master-print.png"))
        elif source_detail:
            images.append(self._file(source_detail, "source-print-detail.jpg"))
        if reference_image_bytes:
            images.append(self._file(reference_image_bytes, "pose-style-reference.jpg"))

        response = self._get_client().images.edit(
            model=image_model or self.image_model,
            image=images,
            prompt=prompt,
            size=self.image_size,
            quality=self.image_quality,
        )
        if not response.data:
            raise MockupGenerationError("OpenAI не вернул изображение.")

        item = response.data[0]
        b64_data = getattr(item, "b64_json", None)
        if b64_data:
            return GeneratedModelPhoto(
                data=base64.b64decode(b64_data),
                mime_type="image/png",
            )

        url = getattr(item, "url", None)
        if url:
            result = httpx.get(url, timeout=120.0)
            result.raise_for_status()
            mime = result.headers.get("content-type", "image/png").split(";")[0]
            return GeneratedModelPhoto(data=result.content, mime_type=mime)

        raise MockupGenerationError("OpenAI вернул ответ без файла изображения.")

    @staticmethod
    def _friendly_error(error: Exception) -> MockupGenerationError:
        message = str(error)
        upper = message.upper()
        status = getattr(error, "status_code", None)
        if status in {401, 403} or "INVALID_API_KEY" in upper or "INCORRECT API KEY" in upper:
            return MockupGenerationError(
                "Ключ OpenAI недействителен или не имеет доступа. Проверьте OPENAI_API_KEY."
            )
        if status == 429 or "RATE LIMIT" in upper or "INSUFFICIENT_QUOTA" in upper:
            return MockupGenerationError(
                "Лимит или баланс OpenAI исчерпан. Проверьте Billing и повторите позже."
            )
        if status == 404:
            return MockupGenerationError(
                "Модель OpenAI недоступна. Проверьте OPENAI_IMAGE_MODEL."
            )
        if status == 400:
            return MockupGenerationError(
                "OpenAI отклонил параметры изображения. Проверьте модель, размер и доступ к Image API."
            )
        return MockupGenerationError(
            "OpenAI временно не смог создать изображение. Повторите попытку позже."
        )
