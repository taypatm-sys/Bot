import asyncio
import base64
import io
import logging
from typing import Any, Optional

import httpx
from openai import OpenAI

from app.image_delivery import prepare_telegram_four_by_five

from app.mockup_generator import (
    GeneratedModelPhoto,
    MockupGenerationError,
    MockupSpec,
    PhotoDirection,
    build_model_photo_prompt,
    prepare_source_print_detail,
)

logger = logging.getLogger(__name__)

# Standard OpenAI API prices per 1M tokens. The response usage is used when
# available. Unknown models fall back to the configured per-image estimate.
_OPENAI_IMAGE_PRICES: dict[str, dict[str, float]] = {
    "gpt-image-1.5": {
        "image_input": 8.0,
        "text_input": 5.0,
        "image_output": 32.0,
        "text_output": 10.0,
    },
}





def estimate_openai_image_cost_usd(model: str, size: str, quality: str) -> float:
    """Minimum output-image estimate when token usage is absent."""
    if model != "gpt-image-1.5":
        return 0.0
    clean_quality = (quality or "medium").strip().casefold()
    clean_size = (size or "1024x1536").strip().casefold()
    square = clean_size == "1024x1024"
    prices = {
        "low": (0.009, 0.013),
        "medium": (0.034, 0.050),
        "high": (0.133, 0.200),
    }
    square_price, portrait_or_landscape_price = prices.get(
        clean_quality, prices["medium"]
    )
    return square_price if square else portrait_or_landscape_price


def _safe_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _usage_value(obj: Any, name: str) -> int:
    if obj is None:
        return 0
    if isinstance(obj, dict):
        return _safe_int(obj.get(name))
    return _safe_int(getattr(obj, name, 0))


def _calculate_openai_cost(model: str, usage: Any) -> tuple[float, dict[str, int]]:
    input_details = (
        usage.get("input_tokens_details")
        if isinstance(usage, dict)
        else getattr(usage, "input_tokens_details", None)
    )
    output_details = (
        usage.get("output_tokens_details")
        if isinstance(usage, dict)
        else getattr(usage, "output_tokens_details", None)
    )
    input_tokens = _usage_value(usage, "input_tokens")
    image_input = _usage_value(input_details, "image_tokens")
    text_input = _usage_value(input_details, "text_tokens")
    output_tokens = _usage_value(usage, "output_tokens")
    image_output = _usage_value(output_details, "image_tokens") or output_tokens
    text_output = _usage_value(output_details, "text_tokens")

    prices = _OPENAI_IMAGE_PRICES.get(model)
    if not prices or not any((input_tokens, image_input, text_input, output_tokens)):
        return 0.0, {
            "input_tokens": input_tokens,
            "image_input": image_input,
            "text_input": text_input,
            "output_tokens": output_tokens,
            "image_output": image_output,
            "text_output": text_output,
        }

    # Some SDK versions expose only aggregate input_tokens. Treat any remainder
    # as image input because edit requests are dominated by image inputs.
    if image_input == 0 and text_input == 0:
        image_input = input_tokens
    elif image_input + text_input < input_tokens:
        image_input += input_tokens - image_input - text_input

    amount = (
        image_input * prices["image_input"]
        + text_input * prices["text_input"]
        + image_output * prices["image_output"]
        + text_output * prices["text_output"]
    ) / 1_000_000
    return amount, {
        "input_tokens": input_tokens,
        "image_input": image_input,
        "text_input": text_input,
        "output_tokens": output_tokens,
        "image_output": image_output,
        "text_output": text_output,
    }


def _extension_for_mime(mime_type: Optional[str]) -> str:
    clean = (mime_type or "").casefold()
    if "png" in clean:
        return "png"
    if "webp" in clean:
        return "webp"
    return "jpg"


class OpenAIMockupGenerator:
    """Creates model photos with OpenAI while keeping Gemini for analysis."""

    def __init__(
        self,
        *,
        api_key: str,
        image_model: str = "gpt-image-1.5",
        image_size: str = "1024x1536",
        image_quality: str = "medium",
        fallback_cost_usd: float = 0.0,
    ) -> None:
        self.api_key = api_key.strip()
        self.image_model = image_model.strip() or "gpt-image-1.5"
        self.image_size = image_size.strip() or "1024x1536"
        self.image_quality = image_quality.strip() or "medium"
        self.fallback_cost_usd = max(0.0, float(fallback_cost_usd or 0.0))
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
            # Automatic SDK retries can create a second paid image request after a
            # network timeout. Disable them and let the user explicitly retry.
            self._client = OpenAI(
                api_key=self.api_key,
                timeout=360.0,
                max_retries=0,
            )
        return self._client

    @staticmethod
    def _file(data: bytes, filename: str) -> io.BytesIO:
        value = io.BytesIO(data)
        value.name = filename
        return value

    def _request_size(self, model: str) -> str:
        del model
        # The Image API accepts these documented sizes. Request the supported
        # portrait size and convert the delivered result to exact 4:5 locally.
        if self.image_size in {"1024x1024", "1024x1536", "1536x1024", "auto"}:
            return self.image_size
        return "1024x1536"

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
            # Do not wrap this thread in asyncio.wait_for. Cancelling wait_for does
            # not stop the underlying paid HTTP request, which can finish and be
            # billed after the bot has already discarded its result.
            return await asyncio.to_thread(
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
            )
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
            "Never copy any text, logo, graphic or garment color from the reference image.\n"
            "The final composition must be vertical 4:5 and must keep the entire head, torso and garment print inside the frame."
        )

        images = [
            self._file(
                image_bytes,
                f"product-source.{_extension_for_mime(mime_type)}",
            )
        ]
        if print_image_bytes:
            images.append(
                self._file(
                    print_image_bytes,
                    f"master-print.{_extension_for_mime(print_mime_type)}",
                )
            )
        elif source_detail:
            images.append(self._file(source_detail, "source-print-detail.jpg"))
        if reference_image_bytes:
            images.append(
                self._file(
                    reference_image_bytes,
                    f"pose-style-reference.{_extension_for_mime(reference_mime_type)}",
                )
            )

        model = image_model or self.image_model
        request_size = self._request_size(model)
        logger.info(
            "OpenAI image request started token=%s model=%s size=%s quality=%s inputs=%s",
            request_token,
            model,
            request_size,
            self.image_quality,
            len(images),
        )
        response = self._get_client().images.edit(
            model=model,
            image=images,
            prompt=prompt,
            size=request_size,
            quality=self.image_quality,
            output_format="jpeg",
            output_compression=90,
        )
        request_id = str(getattr(response, "_request_id", "") or "")
        usage = getattr(response, "usage", None)
        actual_cost, usage_values = _calculate_openai_cost(model, usage)
        default_estimate = estimate_openai_image_cost_usd(
            model, request_size, self.image_quality
        )
        fallback_cost = self.fallback_cost_usd or default_estimate
        cost = actual_cost if actual_cost > 0 else fallback_cost
        cost_source = (
            "OpenAI usage"
            if actual_cost > 0
            else "minimum image estimate"
        )
        logger.info(
            "OpenAI image request completed token=%s request_id=%s usage=%s estimated_cost_usd=%.6f",
            request_token,
            request_id or "unknown",
            usage_values,
            cost,
        )

        if not response.data:
            raise MockupGenerationError(
                "OpenAI завершил и тарифицировал запрос, но не вернул файл изображения. "
                f"ID запроса: {request_id or 'не передан'}.",
                provider_request_id=request_id,
                estimated_cost_usd=cost,
                cost_source=cost_source,
            )

        item = response.data[0]
        b64_data = getattr(item, "b64_json", None)
        raw_image: bytes
        if b64_data:
            try:
                raw_image = base64.b64decode(b64_data, validate=True)
            except Exception as error:
                raise MockupGenerationError(
                    "OpenAI тарифицировал запрос, но вернул поврежденные данные изображения. "
                    f"ID запроса: {request_id or 'не передан'}.",
                    provider_request_id=request_id,
                    estimated_cost_usd=cost,
                    cost_source=cost_source,
                ) from error
        else:
            url = getattr(item, "url", None)
            if not url:
                raise MockupGenerationError(
                    "OpenAI тарифицировал запрос, но вернул ответ без изображения. "
                    f"ID запроса: {request_id or 'не передан'}.",
                    provider_request_id=request_id,
                    estimated_cost_usd=cost,
                    cost_source=cost_source,
                )
            try:
                result = httpx.get(url, timeout=180.0)
                result.raise_for_status()
                raw_image = result.content
            except Exception as error:
                raise MockupGenerationError(
                    "OpenAI создал изображение, но бот не смог скачать файл по ссылке. "
                    f"ID запроса: {request_id or 'не передан'}.",
                    provider_request_id=request_id,
                    estimated_cost_usd=cost,
                    cost_source=cost_source,
                ) from error

        try:
            normalized = prepare_telegram_four_by_five(raw_image)
            normalized_mime = "image/jpeg"
        except Exception:
            # Never discard a completed paid image because local cropping failed.
            # The handler stores and sends the original result instead.
            logger.exception(
                "OpenAI result post-processing failed; returning original request_id=%s",
                request_id,
            )
            normalized = raw_image
            normalized_mime = "image/jpeg"

        return GeneratedModelPhoto(
            data=normalized,
            mime_type=normalized_mime,
            provider_request_id=request_id,
            usage_input_tokens=usage_values["input_tokens"],
            usage_input_image_tokens=usage_values["image_input"],
            usage_input_text_tokens=usage_values["text_input"],
            usage_output_tokens=usage_values["output_tokens"],
            usage_output_image_tokens=usage_values["image_output"],
            usage_output_text_tokens=usage_values["text_output"],
            estimated_cost_usd=cost,
            cost_source=cost_source,
        )

    @staticmethod
    def _friendly_error(error: Exception) -> MockupGenerationError:
        message = str(error)
        upper = message.upper()
        status = getattr(error, "status_code", None)
        request_id = str(getattr(error, "request_id", "") or "")
        suffix = f" ID запроса: {request_id}." if request_id else ""
        if status in {401, 403} or "INVALID_API_KEY" in upper or "INCORRECT API KEY" in upper:
            return MockupGenerationError(
                "Ключ OpenAI недействителен или не имеет доступа. Проверьте OPENAI_API_KEY."
                + suffix
            )
        if status == 429 or "RATE LIMIT" in upper or "INSUFFICIENT_QUOTA" in upper:
            return MockupGenerationError(
                "Лимит или баланс OpenAI исчерпан. Проверьте Billing и повторите позже."
                + suffix
            )
        if status == 404:
            return MockupGenerationError(
                "Модель OpenAI недоступна. Проверьте OPENAI_IMAGE_MODEL." + suffix
            )
        if status == 400:
            return MockupGenerationError(
                "OpenAI отклонил параметры изображения. Для gpt-image-1.5 используйте "
                "размер 1024x1536, 1024x1024 или 1536x1024."
                + suffix
            )
        if "TIMEOUT" in upper or "TIMED OUT" in upper:
            return MockupGenerationError(
                "Соединение с OpenAI прервалось по таймауту. Автоматический повтор отключен, "
                "чтобы не создавать второй платный запрос. Проверьте Usage по ID запроса."
                + suffix
            )
        return MockupGenerationError(
            "OpenAI не смог завершить передачу результата. Проверьте логи Render и Usage."
            + suffix
        )
