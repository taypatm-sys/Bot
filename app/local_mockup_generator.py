import asyncio
import io
import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.mockup_generator import (
    GeneratedModelPhoto,
    MockupGenerationError,
    MockupSpec,
    PhotoDirection,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Artwork:
    rgba: np.ndarray
    confidence: float


@dataclass(frozen=True)
class _CleanupResult:
    image: np.ndarray
    confidence: float
    removed_coverage: float
    residual_coverage: float


class LocalCompositeNeedsGemini(MockupGenerationError):
    """The local path is unsafe and the job should be escalated to Gemini."""


class LocalMockupGenerator:
    """Create simple product mockups locally without an image generation API.

    This path keeps the existing reference photo and replaces only the artwork on
    a clearly visible garment panel. It is intentionally conservative. When the
    print cannot be extracted or the target panel is not safe, it asks the caller
    to escalate the job to Gemini instead of returning a low-quality result.
    """

    @property
    def available(self) -> bool:
        return True

    async def prepare_simple_reference(
        self,
        *,
        image_bytes: bytes,
        reference_tags: dict[str, object],
        preparation: dict[str, object],
    ) -> GeneratedModelPhoto:
        return await asyncio.to_thread(
            self._prepare_simple_reference_sync,
            image_bytes=image_bytes,
            reference_tags=reference_tags,
            preparation=preparation,
        )

    def _prepare_simple_reference_sync(
        self,
        *,
        image_bytes: bytes,
        reference_tags: dict[str, object],
        preparation: dict[str, object],
    ) -> GeneratedModelPhoto:
        image = self._decode_rgb(image_bytes)
        if not bool(preparation.get("suitable", False)):
            raise LocalCompositeNeedsGemini(
                str(preparation.get("reason") or "Референс не подходит для простого режима.")
            )

        target_box = self._read_box(preparation.get("target_print_box"))
        target_quad = self._read_quad(preparation.get("target_print_quad"))
        if target_quad is None and target_box is not None:
            target_quad = self._box_to_quad(target_box)
        if target_quad is None:
            raise LocalCompositeNeedsGemini(
                "Не удалось подготовить точную свободную область футболки."
            )
        height, width = image.shape[:2]
        quad_px = np.array(
            [[x * width / 100.0, y * height / 100.0] for x, y in target_quad],
            dtype=np.float32,
        )
        if not self._quad_is_valid(quad_px, width, height):
            raise LocalCompositeNeedsGemini(
                "Область для простого режима определена ненадежно."
            )

        existing_present = bool(preparation.get("existing_print_present", False))
        if existing_present:
            preflight = {
                **preparation,
                "local_composite_safe": True,
            }
            cleanup_tags = {
                **reference_tags,
                "garment_is_plain": False,
                "existing_print_coverage_percent": int(
                    preparation.get("existing_print_coverage_percent", 0) or 0
                ),
            }
            cleanup = self._clean_existing_artwork(
                image,
                quad_px,
                reference_tags=cleanup_tags,
                preflight=preflight,
            )
            if cleanup.confidence < 0.70:
                raise LocalCompositeNeedsGemini(
                    "Не удалось заранее очистить футболку без заметных следов."
                )
            image = cleanup.image

        output = io.BytesIO()
        Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(
            output,
            format="JPEG",
            quality=92,
            subsampling=0,
            optimize=True,
        )
        return GeneratedModelPhoto(data=output.getvalue(), mime_type="image/jpeg")

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
        del mime_type, direction, request_token, print_mime_type, reference_mime_type
        return await asyncio.to_thread(
            self._generate_sync,
            image_bytes=image_bytes,
            spec=spec,
            print_image_bytes=print_image_bytes,
            reference_image_bytes=reference_image_bytes,
            reference_tags=reference_tags or {},
        )

    def _generate_sync(
        self,
        *,
        image_bytes: bytes,
        spec: Optional[MockupSpec],
        print_image_bytes: Optional[bytes],
        reference_image_bytes: Optional[bytes],
        reference_tags: dict[str, object],
    ) -> GeneratedModelPhoto:
        if spec is None:
            raise LocalCompositeNeedsGemini(
                "Локальная обработка не получила параметры изделия."
            )
        if spec.garment_type != "t-shirt":
            raise LocalCompositeNeedsGemini(
                "Локальный режим пока безопасно работает только с футболками."
            )
        if not reference_image_bytes:
            raise LocalCompositeNeedsGemini(
                "Локальному режиму нужен проверенный референс."
            )

        preflight = reference_tags.get("preflight")
        if not isinstance(preflight, dict):
            raise LocalCompositeNeedsGemini(
                "Нет координат зоны принта для локальной обработки."
            )
        if not bool(preflight.get("local_composite_safe")):
            raise LocalCompositeNeedsGemini(
                "Референс требует полноценной генерации, локальная замена небезопасна."
            )

        target_box = self._read_box(preflight.get("target_print_box"))
        target_quad = self._read_quad(preflight.get("target_print_quad"))
        if target_box is None and target_quad is None:
            raise LocalCompositeNeedsGemini(
                "Не удалось определить точную область принта на референсе."
            )

        reference = self._decode_rgb(reference_image_bytes)
        artwork = self._prepare_artwork(
            source_bytes=image_bytes,
            print_bytes=print_image_bytes,
            spec=spec,
        )
        if artwork.confidence < 0.68:
            raise LocalCompositeNeedsGemini(
                "Принт нельзя надежно отделить от ткани без отдельного PNG."
            )

        height, width = reference.shape[:2]
        if target_quad is None:
            assert target_box is not None
            target_quad = self._box_to_quad(target_box)
        quad_px = np.array(
            [[x * width / 100.0, y * height / 100.0] for x, y in target_quad],
            dtype=np.float32,
        )
        if not self._quad_is_valid(quad_px, width, height):
            raise LocalCompositeNeedsGemini(
                "Зона принта на референсе определена ненадежно."
            )

        cleanup = self._clean_existing_artwork(
            reference,
            quad_px,
            reference_tags=reference_tags,
            preflight=preflight,
        )
        if cleanup.confidence < 0.62:
            raise LocalCompositeNeedsGemini(
                "Чужой принт нельзя надежно удалить локально без следов."
            )
        logger.info(
            "Локальное удаление старого принта: coverage=%.3f residual=%.3f confidence=%.2f",
            cleanup.removed_coverage,
            cleanup.residual_coverage,
            cleanup.confidence,
        )
        composed = self._warp_and_blend(cleanup.image, artwork.rgba, quad_px)
        composed = self._crop_to_four_five(composed, quad_px)

        output = io.BytesIO()
        Image.fromarray(cv2.cvtColor(composed, cv2.COLOR_BGR2RGB)).save(
            output,
            format="JPEG",
            quality=94,
            subsampling=0,
            optimize=True,
        )
        return GeneratedModelPhoto(data=output.getvalue(), mime_type="image/jpeg")

    @staticmethod
    def _decode_rgb(data: bytes) -> np.ndarray:
        try:
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                rgb = ImageOps.exif_transpose(image).convert("RGB")
        except (UnidentifiedImageError, OSError) as error:
            raise LocalCompositeNeedsGemini(
                "Не удалось открыть изображение для локальной обработки."
            ) from error
        return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _decode_rgba(data: bytes) -> np.ndarray:
        try:
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                rgba = ImageOps.exif_transpose(image).convert("RGBA")
        except (UnidentifiedImageError, OSError) as error:
            raise LocalCompositeNeedsGemini(
                "Не удалось открыть отдельный PNG принта."
            ) from error
        return np.asarray(rgba)

    def _prepare_artwork(
        self,
        *,
        source_bytes: bytes,
        print_bytes: Optional[bytes],
        spec: MockupSpec,
    ) -> _Artwork:
        if print_bytes:
            rgba = self._trim_rgba(self._decode_rgba(print_bytes))
            alpha = rgba[:, :, 3]
            coverage = float(np.count_nonzero(alpha > 8)) / max(1, alpha.size)
            if coverage < 0.01:
                raise LocalCompositeNeedsGemini("PNG принта оказался пустым.")
            return _Artwork(rgba=rgba, confidence=0.99)

        source = self._decode_rgb(source_bytes)
        return self._extract_from_product(source, spec)

    def _extract_from_product(self, source: np.ndarray, spec: MockupSpec) -> _Artwork:
        height, width = source.shape[:2]
        if spec.print_box is not None:
            box = spec.print_box
            x0 = int(max(0, (box.x - box.width * 0.30) * width / 100.0))
            y0 = int(max(0, (box.y - box.height * 0.30) * height / 100.0))
            x1 = int(min(width, (box.x + box.width * 1.30) * width / 100.0))
            y1 = int(min(height, (box.y + box.height * 1.30) * height / 100.0))
        else:
            x0, y0, x1, y1 = (
                int(width * 0.20),
                int(height * 0.12),
                int(width * 0.80),
                int(height * 0.78),
            )
        if x1 - x0 < 40 or y1 - y0 < 40:
            raise LocalCompositeNeedsGemini(
                "Область принта слишком мала для локальной обработки."
            )

        crop = source[y0:y1, x0:x1].copy()
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        ch, cw = crop.shape[:2]
        border = max(4, int(min(ch, cw) * 0.08))
        ring_mask = np.zeros((ch, cw), dtype=np.uint8)
        ring_mask[:border, :] = 1
        ring_mask[-border:, :] = 1
        ring_mask[:, :border] = 1
        ring_mask[:, -border:] = 1
        ring_pixels = lab[ring_mask.astype(bool)]
        if ring_pixels.size == 0:
            raise LocalCompositeNeedsGemini(
                "Не удалось оценить цвет ткани вокруг принта."
            )
        background = np.median(ring_pixels, axis=0)
        distance = np.linalg.norm(lab - background, axis=2)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        border_sat = float(np.median(hsv[:, :, 1][ring_mask.astype(bool)]))
        sat_delta = hsv[:, :, 1].astype(np.float32) - border_sat

        threshold = max(14.0, float(np.percentile(distance[ring_mask.astype(bool)], 96)) + 5.0)
        raw = (distance > threshold) | (sat_delta > 28.0)
        mask = (raw.astype(np.uint8) * 255)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        components, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        cleaned = np.zeros_like(mask)
        minimum_area = max(12, int(mask.size * 0.00025))
        for index in range(1, components):
            area = int(stats[index, cv2.CC_STAT_AREA])
            if area >= minimum_area:
                cleaned[labels == index] = 255
        mask = cleaned

        visible = mask > 0
        coverage = float(np.count_nonzero(visible)) / max(1, mask.size)
        border_coverage = float(np.count_nonzero(visible & ring_mask.astype(bool))) / max(
            1, np.count_nonzero(ring_mask)
        )
        if coverage < 0.008 or coverage > 0.62 or border_coverage > 0.18:
            raise LocalCompositeNeedsGemini(
                "Автоматическое отделение принта от ткани ненадежно."
            )

        alpha = np.clip((distance - max(5.0, threshold * 0.45)) / max(8.0, threshold) * 255.0, 0, 255)
        alpha = np.maximum(alpha, np.clip(sat_delta / 45.0 * 255.0, 0, 255))
        alpha = alpha.astype(np.uint8)
        alpha[~visible] = 0
        alpha = cv2.GaussianBlur(alpha, (0, 0), 0.55)

        rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2RGBA)
        rgba[:, :, 3] = alpha
        rgba = self._trim_rgba(rgba)

        confidence = 0.90
        confidence -= min(0.22, border_coverage * 1.4)
        if coverage < 0.02:
            confidence -= 0.12
        if spec.geometry_mode == "source-guided":
            confidence -= 0.08
        return _Artwork(rgba=rgba, confidence=max(0.0, min(1.0, confidence)))

    @staticmethod
    def _trim_rgba(rgba: np.ndarray) -> np.ndarray:
        alpha = rgba[:, :, 3]
        ys, xs = np.where(alpha > 6)
        if len(xs) == 0 or len(ys) == 0:
            return rgba
        pad = max(2, int(min(rgba.shape[:2]) * 0.015))
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(rgba.shape[1], int(xs.max()) + pad + 1)
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(rgba.shape[0], int(ys.max()) + pad + 1)
        return rgba[y0:y1, x0:x1].copy()

    @staticmethod
    def _read_box(value: object) -> Optional[tuple[float, float, float, float]]:
        if not isinstance(value, dict):
            return None
        try:
            x = float(value["x"])
            y = float(value["y"])
            width = float(value["width"])
            height = float(value["height"])
        except (KeyError, TypeError, ValueError):
            return None
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            return None
        if x + width > 100.5 or y + height > 100.5:
            return None
        return x, y, width, height

    @staticmethod
    def _read_quad(value: object) -> Optional[list[tuple[float, float]]]:
        if not isinstance(value, list) or len(value) != 4:
            return None
        result: list[tuple[float, float]] = []
        for point in value:
            if not isinstance(point, dict):
                return None
            try:
                x = float(point["x"])
                y = float(point["y"])
            except (KeyError, TypeError, ValueError):
                return None
            if not 0 <= x <= 100 or not 0 <= y <= 100:
                return None
            result.append((x, y))
        return result

    @staticmethod
    def _box_to_quad(
        box: tuple[float, float, float, float]
    ) -> list[tuple[float, float]]:
        x, y, width, height = box
        return [
            (x, y),
            (x + width, y),
            (x + width, y + height),
            (x, y + height),
        ]

    @staticmethod
    def _quad_is_valid(quad: np.ndarray, width: int, height: int) -> bool:
        if quad.shape != (4, 2):
            return False
        if np.any(quad[:, 0] < 0) or np.any(quad[:, 0] >= width):
            return False
        if np.any(quad[:, 1] < 0) or np.any(quad[:, 1] >= height):
            return False
        area = abs(float(cv2.contourArea(quad)))
        return area >= width * height * 0.012

    def _clean_existing_artwork(
        self,
        image: np.ndarray,
        quad: np.ndarray,
        *,
        reference_tags: dict[str, object],
        preflight: dict[str, object],
    ) -> _CleanupResult:
        """Remove an existing print locally when its exact geometry is known.

        The vision preflight supplies a tight normalized box or quad around the old
        artwork. The local path then rebuilds only that fabric patch from nearby
        clean garment pixels. A complex wash, severe fold or missing geometry is
        escalated to Gemini before any paid image request is made.
        """
        tag_coverage = self._bounded_int(
            reference_tags.get("existing_print_coverage_percent"), 0, 100
        )
        coverage_percent = self._bounded_int(
            preflight.get("existing_print_coverage_percent", tag_coverage), 0, 100
        )
        plain_tag = reference_tags.get("garment_is_plain")
        existing_present = bool(preflight.get("existing_print_present", False))
        if plain_tag is False or coverage_percent >= 4:
            existing_present = True

        if not existing_present:
            return _CleanupResult(
                image=image.copy(),
                confidence=0.99,
                removed_coverage=0.0,
                residual_coverage=0.0,
            )
        if not bool(preflight.get("existing_print_coverable", False)):
            raise LocalCompositeNeedsGemini(
                "Чужой принт нельзя полностью удалить в локальном режиме."
            )
        if not bool(preflight.get("fabric_reconstruction_safe", False)):
            raise LocalCompositeNeedsGemini(
                "Текстура ткани слишком сложная для локального удаления чужого принта."
            )

        cleanup_quad = self._preflight_quad_pixels(
            preflight=preflight,
            box_key="existing_print_box",
            quad_key="existing_print_quad",
            image_width=image.shape[1],
            image_height=image.shape[0],
        )
        if cleanup_quad is None:
            # The local fallback may approve only a tiny old print. In that narrow
            # case the new target box is a safe approximation. Larger prints must
            # have their own exact geometry from the vision preflight.
            if coverage_percent <= 8:
                cleanup_quad = self._scale_quad(
                    quad,
                    scale_x=1.10,
                    scale_y=1.10,
                    image_width=image.shape[1],
                    image_height=image.shape[0],
                )
            else:
                raise LocalCompositeNeedsGemini(
                    "Не получены точные границы чужого принта для локального удаления."
                )

        cleanup_quad = self._scale_quad(
            cleanup_quad,
            scale_x=1.055,
            scale_y=1.055,
            image_width=image.shape[1],
            image_height=image.shape[0],
        )
        if not self._quad_is_valid(cleanup_quad, image.shape[1], image.shape[0]):
            raise LocalCompositeNeedsGemini(
                "Границы чужого принта определены ненадежно."
            )

        x, y, width, height = cv2.boundingRect(cleanup_quad.astype(np.int32))
        old_area = abs(float(cv2.contourArea(cleanup_quad)))
        target_area = max(1.0, abs(float(cv2.contourArea(quad))))
        old_to_new_ratio = old_area / target_area
        area_ratio = old_area / max(
            1.0, float(image.shape[0] * image.shape[1])
        )
        # OpenCV reconstruction is intentionally limited to small old graphics.
        # A large old print would leave a visible smooth patch outside the new art.
        if (
            area_ratio > 0.08
            or coverage_percent > 10
            or old_to_new_ratio > 0.80
        ):
            raise LocalCompositeNeedsGemini(
                "Чужой принт больше безопасной зоны локального удаления."
            )

        pad_x = max(24, int(width * 0.28))
        pad_y = max(24, int(height * 0.28))
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(image.shape[1], x + width + pad_x)
        y1 = min(image.shape[0], y + height + pad_y)
        roi = image[y0:y1, x0:x1].copy()
        if roi.size == 0:
            raise LocalCompositeNeedsGemini(
                "Зона удаления старого принта оказалась пустой."
            )

        local_quad = cleanup_quad - np.array([x0, y0], dtype=np.float32)
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, local_quad.astype(np.int32), 255)
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
        reconstructed, sample_count, seam_score = self._reconstruct_fabric_surface(
            roi=roi,
            mask=mask,
        )
        if sample_count < 350:
            raise LocalCompositeNeedsGemini(
                "Недостаточно чистой ткани вокруг чужого принта."
            )
        if seam_score > 38.0:
            raise LocalCompositeNeedsGemini(
                "Локальное восстановление ткани оставило заметную границу."
            )

        result = image.copy()
        result[y0:y1, x0:x1] = reconstructed
        removed_coverage = float(np.count_nonzero(mask)) / max(1, mask.size)
        confidence = 0.94
        confidence -= min(0.14, area_ratio * 0.45)
        confidence -= min(0.16, seam_score / 240.0)
        confidence -= 0.05 if sample_count < 1000 else 0.0
        return _CleanupResult(
            image=result,
            confidence=max(0.0, min(1.0, confidence)),
            removed_coverage=removed_coverage,
            residual_coverage=0.0,
        )

    def _preflight_quad_pixels(
        self,
        *,
        preflight: dict[str, object],
        box_key: str,
        quad_key: str,
        image_width: int,
        image_height: int,
    ) -> Optional[np.ndarray]:
        normalized_quad = self._read_quad(preflight.get(quad_key))
        if normalized_quad is None:
            normalized_box = self._read_box(preflight.get(box_key))
            if normalized_box is None:
                return None
            normalized_quad = self._box_to_quad(normalized_box)
        return np.array(
            [
                [x * image_width / 100.0, y * image_height / 100.0]
                for x, y in normalized_quad
            ],
            dtype=np.float32,
        )

    def _reconstruct_fabric_surface(
        self,
        *,
        roi: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, int, float]:
        """Fit a smooth garment-lighting surface and restore nearby texture."""
        outer = cv2.dilate(
            mask,
            np.ones((35, 35), np.uint8),
            iterations=1,
        )
        ring = (outer > 0) & (mask == 0)
        if np.count_nonzero(ring) < 200:
            outer = cv2.dilate(
                mask,
                np.ones((51, 51), np.uint8),
                iterations=1,
            )
            ring = (outer > 0) & (mask == 0)

        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
        ring_pixels = lab[ring]
        if len(ring_pixels) < 200:
            return roi.copy(), int(len(ring_pixels)), 999.0

        sample_for_cluster = ring_pixels
        if len(sample_for_cluster) > 16000:
            rng = np.random.default_rng(7125)
            indices = rng.choice(len(sample_for_cluster), 16000, replace=False)
            sample_for_cluster = sample_for_cluster[indices]
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            35,
            0.25,
        )
        cluster_count = 3 if len(sample_for_cluster) >= 600 else 2
        _, labels, centers = cv2.kmeans(
            sample_for_cluster.astype(np.float32),
            cluster_count,
            None,
            criteria,
            3,
            cv2.KMEANS_PP_CENTERS,
        )
        counts = np.bincount(labels.reshape(-1), minlength=cluster_count)
        garment_cluster = int(np.argmax(counts))
        base_lab = centers[garment_cluster]
        selected = sample_for_cluster[labels.reshape(-1) == garment_cluster]
        selected_distance = np.linalg.norm(selected - base_lab, axis=1)
        garment_threshold = max(
            24.0,
            min(62.0, self._percentile(selected_distance, 99, 18.0) + 12.0),
        )

        distance = np.linalg.norm(lab - base_lab, axis=2)
        sample_zone = cv2.dilate(
            mask,
            np.ones((65, 65), np.uint8),
            iterations=1,
        ) > 0
        sample_mask = sample_zone & (mask == 0) & (distance <= garment_threshold)
        sample_count = int(np.count_nonzero(sample_mask))
        if sample_count < 350:
            return roi.copy(), sample_count, 999.0

        surface = self._fit_polynomial_surface(roi, sample_mask)
        radius = max(4, min(10, int(round(min(roi.shape[:2]) * 0.025))))
        telea = cv2.inpaint(roi, mask, radius, cv2.INPAINT_TELEA).astype(np.float32)

        residual = roi.astype(np.float32) - surface
        residual_shifted = np.clip(residual + 128.0, 0, 255).astype(np.uint8)
        residual_fill = cv2.inpaint(
            residual_shifted,
            mask,
            max(3, radius - 2),
            cv2.INPAINT_TELEA,
        ).astype(np.float32) - 128.0
        rebuilt = np.clip(surface + residual_fill * 0.48, 0, 255)
        rebuilt = rebuilt * 0.76 + telea * 0.24

        feather = cv2.GaussianBlur(mask, (0, 0), 2.4).astype(np.float32) / 255.0
        output = (
            rebuilt * feather[:, :, None]
            + roi.astype(np.float32) * (1.0 - feather[:, :, None])
        )
        output = np.clip(output, 0, 255).astype(np.uint8)

        inner_edge = cv2.morphologyEx(
            mask, cv2.MORPH_GRADIENT, np.ones((7, 7), np.uint8)
        ) > 0
        seam_score = float(
            np.mean(
                np.abs(output.astype(np.float32) - roi.astype(np.float32))[inner_edge]
            )
        ) if np.any(inner_edge) else 0.0
        return output, sample_count, seam_score

    @staticmethod
    def _fit_polynomial_surface(
        roi: np.ndarray,
        sample_mask: np.ndarray,
    ) -> np.ndarray:
        height, width = roi.shape[:2]
        ys, xs = np.where(sample_mask)
        if len(xs) > 24000:
            rng = np.random.default_rng(1337)
            selected = rng.choice(len(xs), 24000, replace=False)
            xs = xs[selected]
            ys = ys[selected]
        x = xs.astype(np.float64) / max(1, width - 1) * 2.0 - 1.0
        y = ys.astype(np.float64) / max(1, height - 1) * 2.0 - 1.0
        design = np.column_stack(
            [
                np.ones_like(x),
                x,
                y,
                x * x,
                y * y,
                x * y,
            ]
        )
        coefficients: list[np.ndarray] = []
        for channel in range(3):
            values = roi[ys, xs, channel].astype(np.float64)
            coeff, *_ = np.linalg.lstsq(design, values, rcond=None)
            for _ in range(2):
                residual = values - design @ coeff
                median = float(np.median(residual))
                mad = float(np.median(np.abs(residual - median))) + 1e-6
                keep = np.abs(residual - median) <= 3.2 * 1.4826 * mad + 3.0
                if np.count_nonzero(keep) < 120:
                    break
                coeff, *_ = np.linalg.lstsq(
                    design[keep], values[keep], rcond=None
                )
            coefficients.append(coeff)

        grid_y, grid_x = np.mgrid[0:height, 0:width]
        gx = grid_x.astype(np.float64) / max(1, width - 1) * 2.0 - 1.0
        gy = grid_y.astype(np.float64) / max(1, height - 1) * 2.0 - 1.0
        grid_design = np.stack(
            [
                np.ones_like(gx),
                gx,
                gy,
                gx * gx,
                gy * gy,
                gx * gy,
            ],
            axis=-1,
        )
        coefficient_matrix = np.stack(coefficients, axis=1)
        surface = np.tensordot(grid_design, coefficient_matrix, axes=([2], [0]))
        return np.clip(surface, 0, 255).astype(np.float32)

    @staticmethod
    def _scale_quad(
        quad: np.ndarray,
        *,
        scale_x: float,
        scale_y: float,
        image_width: int,
        image_height: int,
    ) -> np.ndarray:
        center = np.mean(quad, axis=0)
        scaled = quad.copy().astype(np.float32)
        scaled[:, 0] = center[0] + (scaled[:, 0] - center[0]) * scale_x
        scaled[:, 1] = center[1] + (scaled[:, 1] - center[1]) * scale_y
        scaled[:, 0] = np.clip(scaled[:, 0], 1, max(1, image_width - 2))
        scaled[:, 1] = np.clip(scaled[:, 1], 1, max(1, image_height - 2))
        return scaled

    @staticmethod
    def _bounded_int(value: object, minimum: int, maximum: int) -> int:
        try:
            parsed = int(value or 0)
        except (TypeError, ValueError):
            parsed = 0
        return max(minimum, min(maximum, parsed))

    @staticmethod
    def _percentile(values: np.ndarray, percentile: float, default: float) -> float:
        return float(np.percentile(values, percentile)) if values.size else default

    def _warp_and_blend(
        self,
        base: np.ndarray,
        artwork_rgba: np.ndarray,
        target_quad: np.ndarray,
    ) -> np.ndarray:
        height, width = base.shape[:2]
        art_h, art_w = artwork_rgba.shape[:2]
        if art_h < 2 or art_w < 2:
            raise LocalCompositeNeedsGemini("Принт оказался слишком маленьким.")

        source_quad = np.array(
            [[0, 0], [art_w - 1, 0], [art_w - 1, art_h - 1], [0, art_h - 1]],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(source_quad, target_quad.astype(np.float32))
        art_bgr = cv2.cvtColor(artwork_rgba[:, :, :3], cv2.COLOR_RGB2BGR)
        alpha = artwork_rgba[:, :, 3]
        warped_art = cv2.warpPerspective(
            art_bgr,
            transform,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        warped_alpha = cv2.warpPerspective(
            alpha,
            transform,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        warped_alpha = cv2.GaussianBlur(warped_alpha, (0, 0), 0.45)

        gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY).astype(np.float32)
        low_frequency = cv2.GaussianBlur(gray, (0, 0), max(3.0, min(width, height) * 0.012))
        mask_pixels = warped_alpha > 8
        if np.any(mask_pixels):
            median_light = float(np.median(low_frequency[mask_pixels]))
        else:
            median_light = 128.0
        light_ratio = np.clip(low_frequency / max(35.0, median_light), 0.72, 1.20)
        light_ratio = np.power(light_ratio, 0.42)
        shaded_art = np.clip(
            warped_art.astype(np.float32) * light_ratio[:, :, None], 0, 255
        )

        alpha_float = (warped_alpha.astype(np.float32) / 255.0)[:, :, None]
        output = (
            shaded_art * alpha_float + base.astype(np.float32) * (1.0 - alpha_float)
        )
        return np.clip(output, 0, 255).astype(np.uint8)

    @staticmethod
    def _crop_to_four_five(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        target_ratio = 4.0 / 5.0
        current_ratio = width / height
        center_x = float(np.mean(quad[:, 0]))
        center_y = float(np.mean(quad[:, 1]))

        if abs(current_ratio - target_ratio) < 0.015:
            return image
        if current_ratio > target_ratio:
            new_width = int(round(height * target_ratio))
            x0 = int(round(center_x - new_width / 2))
            x0 = max(0, min(width - new_width, x0))
            return image[:, x0 : x0 + new_width]

        new_height = int(round(width / target_ratio))
        y0 = int(round(center_y - new_height / 2))
        y0 = max(0, min(height - new_height, y0))
        return image[y0 : y0 + new_height, :]
