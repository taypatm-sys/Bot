import io

from PIL import Image, ImageOps


def prepare_telegram_four_by_five(image_bytes: bytes) -> bytes:
    """Return a compact 1024x1280 JPEG suitable for Telegram sendPhoto."""
    with Image.open(io.BytesIO(image_bytes)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        # Keep the face and chest slightly above center when cropping 2:3 to 4:5.
        image = ImageOps.fit(
            image,
            (1024, 1280),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.42),
        )
        value = b""
        for quality in (92, 88, 84, 80):
            output = io.BytesIO()
            image.save(
                output,
                format="JPEG",
                quality=quality,
                optimize=True,
                progressive=True,
            )
            value = output.getvalue()
            if len(value) <= 9_000_000:
                return value
        return value
