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
                        "x": 33,
                        "y": 38,
                        "width": 34,
                        "height": 22,
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


if __name__ == "__main__":
    unittest.main()
