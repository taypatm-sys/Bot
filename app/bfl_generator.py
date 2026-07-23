import asyncio
import base64
import logging
from typing import Optional

import aiohttp

from app.mockup_generator import (
    GeneratedModelPhoto,
    MockupGenerationError,
    MockupSpec,
    PhotoDirection,
    build_model_photo_prompt,
    prepare_source_print_detail,
)


logger = logging.getLogger(__name__)


class BflMockupGenerator:
    """Low-cost FLUX.2 Klein path for routine mockups.

    Complex jobs stay on Gemini. This provider is optional and is used only when
    BFL_API_KEY is configured.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "flux-2-klein-4b",
        api_base: str = "https://api.bfl.ai/v1",
        timeout_seconds: float = 150.0,
    ) -> None:
        self.api_key = api_key.strip()
        self.model = model.strip() or "flux-2-klein-4b"
        self.api_base = api_base.rstrip("/")
        self.timeout_seconds = max(30.0, timeout_seconds)

    @property
    def available(self) -> bool:
        return bool(self.api_key)

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
    ) -> GeneratedModelPhoto:
        if not self.available:
            raise MockupGenerationError(
                "Экономная модель не подключена. Добавьте BFL_API_KEY в Render."
            )
        if not reference_image_bytes:
            raise MockupGenerationError(
                "Для экономной генерации нужен проверенный референс человека."
            )

        source_detail_bytes = None
        if not print_image_bytes:
            source_detail_bytes = prepare_source_print_detail(image_bytes, spec)

        prompt = build_model_photo_prompt(
            spec,
            direction,
            request_token,
            has_separate_print=bool(print_image_bytes),
            has_source_detail=bool(source_detail_bytes),
            has_style_reference=True,
            style_reference_tags=reference_tags,
        )
        prompt = (
            "Use the supplied images in their exact numbered order. "
            "Do not invent or rewrite any garment artwork.\n\n" + prompt
        )

        payload: dict[str, object] = {
            "prompt": prompt,
            "input_image": base64.b64encode(image_bytes).decode("ascii"),
            "output_format": "jpeg",
            "width": 896,
            "height": 1120,
            "safety_tolerance": 2,
        }
        if print_image_bytes:
            payload["input_image_2"] = base64.b64encode(print_image_bytes).decode(
                "ascii"
            )
            payload["input_image_3"] = base64.b64encode(
                reference_image_bytes
            ).decode("ascii")
        elif source_detail_bytes:
            payload["input_image_2"] = base64.b64encode(source_detail_bytes).decode(
                "ascii"
            )
            payload["input_image_3"] = base64.b64encode(
                reference_image_bytes
            ).decode("ascii")
        else:
            payload["input_image_2"] = base64.b64encode(
                reference_image_bytes
            ).decode("ascii")

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds, connect=15)
        headers = {
            "accept": "application/json",
            "x-key": self.api_key,
            "Content-Type": "application/json",
        }
        submit_url = f"{self.api_base}/{self.model}"

        try:
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                async with session.post(submit_url, json=payload) as response:
                    body = await response.json(content_type=None)
                    if response.status >= 400:
                        raise MockupGenerationError(
                            f"Экономная модель отклонила запрос, код {response.status}."
                        )
                polling_url = str(body.get("polling_url", "")).strip()
                task_id = str(body.get("id", "")).strip()
                if not polling_url and task_id:
                    polling_url = f"{self.api_base}/get_result?id={task_id}"
                if not polling_url:
                    raise MockupGenerationError(
                        "Экономная модель не вернула адрес результата."
                    )

                deadline = asyncio.get_running_loop().time() + self.timeout_seconds
                while asyncio.get_running_loop().time() < deadline:
                    await asyncio.sleep(0.8)
                    async with session.get(polling_url) as response:
                        result = await response.json(content_type=None)
                        if response.status >= 400:
                            raise MockupGenerationError(
                                f"Не удалось получить результат, код {response.status}."
                            )
                    status = str(result.get("status", ""))
                    if status == "Ready":
                        sample = str(
                            (result.get("result") or {}).get("sample", "")
                        ).strip()
                        if not sample:
                            raise MockupGenerationError(
                                "Экономная модель завершила задачу без изображения."
                            )
                        async with session.get(sample) as image_response:
                            if image_response.status >= 400:
                                raise MockupGenerationError(
                                    "Не удалось скачать готовое изображение."
                                )
                            data = await image_response.read()
                            mime = image_response.headers.get(
                                "Content-Type", "image/jpeg"
                            ).split(";", 1)[0]
                        if not data:
                            raise MockupGenerationError(
                                "Экономная модель вернула пустой файл."
                            )
                        return GeneratedModelPhoto(data=data, mime_type=mime)
                    if status in {
                        "Error",
                        "Failed",
                        "Request Moderated",
                        "Content Moderated",
                        "Task not found",
                    }:
                        raise MockupGenerationError(
                            "Экономная модель не смогла создать изображение."
                        )
        except MockupGenerationError:
            raise
        except asyncio.TimeoutError as error:
            raise MockupGenerationError(
                "Экономная модель не ответила вовремя. Платный Gemini не запускался."
            ) from error
        except aiohttp.ClientError as error:
            logger.warning("BFL request failed: %s", error)
            raise MockupGenerationError(
                "Экономная модель временно недоступна. Платный Gemini не запускался."
            ) from error

        raise MockupGenerationError(
            "Экономная модель не успела закончить генерацию. Gemini не запускался."
        )
