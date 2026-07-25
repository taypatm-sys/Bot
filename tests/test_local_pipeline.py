import io
import unittest

import numpy as np
from PIL import Image, ImageDraw

from app.generation_router import choose_generation_model
from app.local_mockup_generator import LocalMockupGenerator
from app.mockup_generator import MockupSpec, NormalizedBox, PhotoDirection


class LocalPipelineTests(unittest.IsolatedAsyncioTestCase):
    def _spec(self) -> MockupSpec:
        return MockupSpec(
            side="front",
            garment_type="t-shirt",
            shirt_color="white",
            fabric_finish="matte",
            fit="oversized",
            print_width_percent=35,
            print_height_percent=28,
            print_top_offset_percent=18,
            print_left_offset_percent=32,
            print_center_x_percent=50,
            target_gender="unisex",
            target_age_group="adult-universal",
            moods=["minimal"],
            print_theme="red typography",
            construction_details="crew neck short sleeve",
            analysis_confidence=90,
            source_image_width_px=600,
            source_image_height_px=600,
            garment_panel_box=NormalizedBox(x=15, y=8, width=70, height=88),
            print_box=NormalizedBox(x=32, y=28, width=36, height=26),
            geometry_validated=True,
            geometry_mode="measured",
        )

    def test_router_uses_local_for_safe_tshirt(self) -> None:
        decision = choose_generation_model(
            spec=self._spec(),
            has_separate_print=True,
            local_composite_safe=True,
            gemini_lite_model="gemini-3.1-flash-lite-image",
        )
        self.assertEqual(decision.provider, "local")

    def test_router_uses_gemini_when_local_is_unsafe(self) -> None:
        decision = choose_generation_model(
            spec=self._spec(),
            has_separate_print=False,
            local_composite_safe=False,
            gemini_lite_model="gemini-3.1-flash-lite-image",
        )
        self.assertEqual(decision.provider, "gemini")
        self.assertEqual(decision.tier, "very_complex")

    async def test_local_compositor_outputs_four_by_five(self) -> None:
        reference = Image.new("RGB", (800, 1000), (180, 180, 180))
        draw = ImageDraw.Draw(reference)
        draw.rounded_rectangle((180, 170, 620, 900), radius=45, fill=(238, 238, 238))
        reference_data = io.BytesIO()
        reference.save(reference_data, format="JPEG", quality=95)

        product = Image.new("RGB", (600, 600), (180, 180, 180))
        draw = ImageDraw.Draw(product)
        draw.rectangle((90, 40, 510, 580), fill=(245, 245, 245))
        draw.rectangle((210, 190, 390, 350), fill=(220, 20, 70))
        product_data = io.BytesIO()
        product.save(product_data, format="JPEG", quality=95)

        print_asset = Image.new("RGBA", (300, 200), (0, 0, 0, 0))
        draw = ImageDraw.Draw(print_asset)
        draw.rounded_rectangle((20, 20, 280, 180), radius=30, fill=(220, 20, 70, 255))
        print_data = io.BytesIO()
        print_asset.save(print_data, format="PNG")

        direction = PhotoDirection(
            label="test",
            gender="women",
            pose_kind="standing",
            person="adult woman",
            setting="simple street",
            pose="standing naturally",
            camera="front",
            framing="waist-up",
            light="daylight",
            seed=1,
        )
        output = await LocalMockupGenerator().generate_variant(
            image_bytes=product_data.getvalue(),
            mime_type="image/jpeg",
            spec=self._spec(),
            direction=direction,
            request_token="test",
            print_image_bytes=print_data.getvalue(),
            print_mime_type="image/png",
            reference_image_bytes=reference_data.getvalue(),
            reference_mime_type="image/jpeg",
            reference_tags={
                "preflight": {
                    "local_composite_safe": True,
                    "target_print_box": {
                        "x": 24,
                        "y": 31,
                        "width": 52,
                        "height": 36,
                    },
                    "target_print_quad": [],
                }
            },
        )
        with Image.open(io.BytesIO(output.data)) as result:
            self.assertEqual(result.size[0] * 5, result.size[1] * 4)
            array = np.asarray(result.convert("RGB"))
            red_pixels = (array[:, :, 0] > 170) & (array[:, :, 1] < 90)
            self.assertGreater(int(red_pixels.sum()), 1000)

    def test_existing_print_is_removed_locally(self) -> None:
        generator = LocalMockupGenerator()
        reference = np.full((900, 720, 3), 235, dtype=np.uint8)
        # Mild vertical lighting gradient to imitate fabric shading.
        for row in range(reference.shape[0]):
            shade = int((row / reference.shape[0]) * 18)
            reference[row, :, :] = np.clip(reference[row, :, :] - shade, 0, 255)
        cv2 = __import__("cv2")
        cv2.putText(
            reference,
            "OLD PRINT",
            (245, 405),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (25, 35, 215),
            4,
            cv2.LINE_AA,
        )
        cv2.rectangle(reference, (285, 450), (435, 510), (35, 150, 240), -1)
        quad = np.array(
            [[245, 345], [475, 345], [475, 545], [245, 545]], dtype=np.float32
        )
        before = reference.copy()
        cleanup = generator._clean_existing_artwork(
            reference,
            quad,
            reference_tags={
                "garment_is_plain": False,
                "existing_print_coverage_percent": 8,
            },
            preflight={
                "existing_print_present": True,
                "existing_print_box": {
                    "x": 32.0,
                    "y": 43.0,
                    "width": 30.0,
                    "height": 15.5,
                },
                "existing_print_quad": [],
                "existing_print_coverage_percent": 8,
                "existing_print_coverable": True,
                "fabric_reconstruction_safe": True,
            },
        )
        before_color = (
            (before[:, :, 2] > 150)
            & (before[:, :, 1] < 190)
            & (before[:, :, 0] < 120)
        )
        after = cleanup.image
        after_color = (
            (after[:, :, 2] > 150)
            & (after[:, :, 1] < 190)
            & (after[:, :, 0] < 120)
        )
        self.assertGreater(int(before_color.sum()), 1800)
        self.assertLess(int(after_color.sum()), int(before_color.sum() * 0.25))
        self.assertGreater(cleanup.confidence, 0.62)

    async def test_local_compositor_replaces_existing_print(self) -> None:
        reference = Image.new("RGB", (800, 1000), (225, 225, 225))
        draw = ImageDraw.Draw(reference)
        draw.rounded_rectangle((160, 150, 640, 910), radius=55, fill=(238, 238, 238))
        draw.text((260, 390), "OLD", fill=(20, 40, 210))
        draw.rectangle((260, 450, 540, 590), fill=(230, 110, 35))
        reference_data = io.BytesIO()
        reference.save(reference_data, format="JPEG", quality=95)

        product = Image.new("RGB", (600, 600), (180, 180, 180))
        draw = ImageDraw.Draw(product)
        draw.rectangle((90, 40, 510, 580), fill=(245, 245, 245))
        draw.rectangle((210, 190, 390, 350), fill=(25, 180, 80))
        product_data = io.BytesIO()
        product.save(product_data, format="JPEG", quality=95)

        print_asset = Image.new("RGBA", (300, 200), (0, 0, 0, 0))
        draw = ImageDraw.Draw(print_asset)
        draw.rounded_rectangle((20, 20, 280, 180), radius=30, fill=(25, 180, 80, 255))
        print_data = io.BytesIO()
        print_asset.save(print_data, format="PNG")

        direction = PhotoDirection(
            label="test",
            gender="women",
            pose_kind="standing",
            person="adult woman",
            setting="simple street",
            pose="standing naturally",
            camera="front",
            framing="waist-up",
            light="daylight",
            seed=2,
        )
        output = await LocalMockupGenerator().generate_variant(
            image_bytes=product_data.getvalue(),
            mime_type="image/jpeg",
            spec=self._spec(),
            direction=direction,
            request_token="test-existing",
            print_image_bytes=print_data.getvalue(),
            print_mime_type="image/png",
            reference_image_bytes=reference_data.getvalue(),
            reference_mime_type="image/jpeg",
            reference_tags={
                "garment_is_plain": False,
                "existing_print_coverage_percent": 8,
                "preflight": {
                    "local_composite_safe": True,
                    "existing_print_present": True,
                    "existing_print_box": {
                        "x": 30,
                        "y": 37,
                        "width": 35,
                        "height": 18,
                    },
                    "existing_print_quad": [],
                    "existing_print_coverage_percent": 8,
                    "existing_print_coverable": True,
                    "fabric_reconstruction_safe": True,
                    "target_print_box": {
                        "x": 24,
                        "y": 31,
                        "width": 52,
                        "height": 36,
                    },
                    "target_print_quad": [],
                },
            },
        )
        with Image.open(io.BytesIO(output.data)) as result:
            array = np.asarray(result.convert("RGB"))
            green_pixels = (array[:, :, 1] > 130) & (array[:, :, 0] < 100)
            old_red_pixels = (array[:, :, 0] > 160) & (array[:, :, 1] < 100)
            self.assertGreater(int(green_pixels.sum()), 1000)
            self.assertLess(int(old_red_pixels.sum()), 500)

    def test_large_existing_print_is_not_removed_locally(self) -> None:
        generator = LocalMockupGenerator()
        reference = np.full((900, 720, 3), 235, dtype=np.uint8)
        quad = np.array(
            [[280, 360], [440, 360], [440, 520], [280, 520]], dtype=np.float32
        )
        from app.local_mockup_generator import LocalCompositeNeedsGemini
        with self.assertRaises(LocalCompositeNeedsGemini):
            generator._clean_existing_artwork(
                reference,
                quad,
                reference_tags={
                    "garment_is_plain": False,
                    "existing_print_coverage_percent": 40,
                },
                preflight={
                    "existing_print_present": True,
                    "existing_print_box": {
                        "x": 20.0,
                        "y": 25.0,
                        "width": 60.0,
                        "height": 45.0,
                    },
                    "existing_print_quad": [],
                    "existing_print_coverage_percent": 40,
                    "existing_print_coverable": True,
                    "fabric_reconstruction_safe": True,
                },
            )


if __name__ == "__main__":
    unittest.main()
