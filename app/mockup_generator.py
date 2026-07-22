import asyncio
import base64
import io
import logging
import random
import secrets
from dataclasses import dataclass
from typing import Literal, Optional

from google import genai
from google.genai import types
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


GarmentType = Literal[
    "t-shirt",
    "hoodie",
    "sweatshirt",
    "long-sleeve",
    "zip-hoodie",
    "cap",
    "jacket",
]
MoodTag = Literal[
    "calm",
    "bold",
    "cozy",
    "sporty",
    "youth",
    "romantic",
    "playful",
    "minimal",
    "premium",
    "street",
]
TargetAgeGroup = Literal["18-24", "25-34", "35-44", "adult-universal"]


class NormalizedBox(BaseModel):
    x: float = Field(ge=0, le=100)
    y: float = Field(ge=0, le=100)
    width: float = Field(gt=0, le=100)
    height: float = Field(gt=0, le=100)


class _DetectedMockup(BaseModel):
    side: Literal["front", "back"]
    garment_type: GarmentType
    shirt_color: str = Field(min_length=1, max_length=80)
    fabric_finish: str = Field(min_length=1, max_length=100)
    fit: str = Field(min_length=1, max_length=80)
    target_gender: Literal["women", "men", "unisex"]
    target_age_group: TargetAgeGroup
    moods: list[MoodTag] = Field(min_length=1, max_length=4)
    print_theme: str = Field(min_length=1, max_length=140)
    construction_details: str = Field(min_length=1, max_length=160)
    garment_panel_box: NormalizedBox
    print_box: NormalizedBox
    analysis_confidence: int = Field(ge=0, le=100)


class _DetectedPrint(BaseModel):
    target_gender: Literal["women", "men", "unisex"]
    target_age_group: TargetAgeGroup
    moods: list[MoodTag] = Field(min_length=1, max_length=4)
    print_theme: str = Field(min_length=1, max_length=140)
    dominant_colors: list[str] = Field(min_length=1, max_length=6)
    analysis_confidence: int = Field(ge=0, le=100)


class MockupSpec(BaseModel):
    side: Literal["front", "back"]
    garment_type: GarmentType
    shirt_color: str = Field(min_length=1, max_length=80)
    fabric_finish: str = Field(min_length=1, max_length=100)
    fit: str = Field(min_length=1, max_length=80)
    print_width_percent: int = Field(ge=5, le=100)
    print_height_percent: int = Field(ge=3, le=100)
    print_top_offset_percent: int = Field(ge=0, le=80)
    print_left_offset_percent: int = Field(default=25, ge=0, le=95)
    print_center_x_percent: int = Field(default=50, ge=0, le=100)
    target_gender: Literal["women", "men", "unisex"]
    target_age_group: TargetAgeGroup = "adult-universal"
    moods: list[MoodTag] = Field(default_factory=lambda: ["minimal"])
    print_theme: str = "neutral graphic"
    construction_details: str = Field(min_length=1, max_length=160)
    analysis_confidence: int = Field(default=60, ge=0, le=100)
    source_image_width_px: int = Field(default=0, ge=0)
    source_image_height_px: int = Field(default=0, ge=0)
    garment_panel_box: Optional[NormalizedBox] = None
    print_box: Optional[NormalizedBox] = None


class PrintAssetSpec(BaseModel):
    width_px: int = Field(ge=1)
    height_px: int = Field(ge=1)
    content_x_px: int = Field(ge=0)
    content_y_px: int = Field(ge=0)
    content_width_px: int = Field(ge=1)
    content_height_px: int = Field(ge=1)
    has_transparency: bool
    visible_coverage_percent: float = Field(ge=0, le=100)
    target_gender: Literal["women", "men", "unisex"]
    target_age_group: TargetAgeGroup
    moods: list[MoodTag] = Field(min_length=1, max_length=4)
    print_theme: str = Field(min_length=1, max_length=140)
    dominant_colors: list[str] = Field(min_length=1, max_length=6)
    analysis_confidence: int = Field(ge=0, le=100)


def _clamp_int(value: float, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(round(value))))


def build_mockup_spec(
    detected: _DetectedMockup,
    *,
    image_width: int,
    image_height: int,
) -> MockupSpec:
    garment = detected.garment_panel_box
    artwork = detected.print_box
    relative_width = artwork.width / garment.width * 100
    relative_height = artwork.height / garment.height * 100
    relative_left = (artwork.x - garment.x) / garment.width * 100
    relative_top = (artwork.y - garment.y) / garment.height * 100
    width_percent = _clamp_int(relative_width, 5, 100)
    height_percent = _clamp_int(relative_height, 3, 100)
    left_percent = _clamp_int(relative_left, 0, max(0, 100 - width_percent))
    top_percent = _clamp_int(relative_top, 0, min(80, max(0, 100 - height_percent)))
    center_percent = _clamp_int(left_percent + width_percent / 2, 0, 100)
    return MockupSpec(
        side=detected.side,
        garment_type=detected.garment_type,
        shirt_color=detected.shirt_color,
        fabric_finish=detected.fabric_finish,
        fit=detected.fit,
        print_width_percent=width_percent,
        print_height_percent=height_percent,
        print_top_offset_percent=top_percent,
        print_left_offset_percent=left_percent,
        print_center_x_percent=center_percent,
        target_gender=detected.target_gender,
        target_age_group=detected.target_age_group,
        moods=list(dict.fromkeys(detected.moods)),
        print_theme=detected.print_theme,
        construction_details=detected.construction_details,
        analysis_confidence=detected.analysis_confidence,
        source_image_width_px=image_width,
        source_image_height_px=image_height,
        garment_panel_box=garment,
        print_box=artwork,
    )


def inspect_print_file(image_bytes: bytes) -> dict[str, int | float | bool]:
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            source.load()
            width, height = source.size
            if width < 2 or height < 2:
                raise ValueError("PNG принта слишком маленький")
            rgba = source.convert("RGBA")
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("Не удалось открыть PNG принта") from error

    alpha = rgba.getchannel("A")
    bbox = alpha.getbbox()
    if bbox is None:
        raise ValueError("PNG принта полностью прозрачный")
    x0, y0, x1, y1 = bbox
    histogram = alpha.histogram()
    transparent_pixels = histogram[0]
    visible_pixels = width * height - transparent_pixels
    has_transparency = transparent_pixels > 0 or any(histogram[1:255])
    return {
        "width_px": width,
        "height_px": height,
        "content_x_px": x0,
        "content_y_px": y0,
        "content_width_px": x1 - x0,
        "content_height_px": y1 - y0,
        "has_transparency": has_transparency,
        "visible_coverage_percent": round(visible_pixels / (width * height) * 100, 2),
    }


@dataclass(frozen=True)
class PhotoDirection:
    label: str
    gender: Literal["women", "men"]
    pose_kind: Literal["sitting", "walking", "activity", "standing", "close-up"]
    person: str
    setting: str
    pose: str
    camera: str
    framing: str
    light: str
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


@dataclass(frozen=True)
class _PhotoScenario:
    label: str
    pose_kind: Literal["sitting", "walking", "activity", "standing", "close-up"]
    setting: str
    pose: str
    camera: str
    framing: str
    light: str
    garments: frozenset[str]


_CLOTHING = frozenset(
    {
        "t-shirt",
        "hoodie",
        "sweatshirt",
        "long-sleeve",
        "zip-hoodie",
        "jacket",
    }
)
_ALL_GARMENTS = _CLOTHING | {"cap"}

_PHOTO_SCENARIOS = (
    _PhotoScenario(
        "Листает журнал на диване",
        "sitting",
        "a lived-in living room with a low table, a magazine and an ordinary sofa",
        "sitting diagonally on the sofa and reaching toward the table, caught mid-action",
        "a quick phone snapshot from slightly above and behind a shoulder",
        "a loose seated composition with a natural crop below the knees",
        "soft window light with normal indoor shadows and modest phone-camera contrast",
        _CLOTHING,
    ),
    _PhotoScenario(
        "Выбирает фрукты",
        "activity",
        "a real neighborhood fruit market with handwritten price cards and busy shelves",
        "reaching for fruit or holding a small basket without stopping to pose",
        "a friend taking a spontaneous phone photo from a slight side angle",
        "a three-quarter view with useful background context and the printed panel readable",
        "mixed shop light with slightly warm color and imperfect exposure",
        _CLOTHING,
    ),
    _PhotoScenario(
        "У открытого окна",
        "activity",
        "a simple apartment hallway ending at an open window with ordinary shutters",
        "opening a shutter or resting one hand on the wall while looking outside",
        "a handheld phone photo taken from the hallway, not perfectly level",
        "a natural vertical frame with some empty wall and a mild perspective angle",
        "bright window light with gently clipped highlights and real indoor falloff",
        _CLOTHING,
    ),
    _PhotoScenario(
        "Садится в машину",
        "sitting",
        "the real interior of a parked everyday car at night or in a dim garage",
        "turning into the seat or reaching toward the dashboard, not looking at camera",
        "a close phone snapshot from the neighboring seat with ordinary wide-angle perspective",
        "a cropped seated frame that keeps the complete print visible but not the whole body",
        "uneven cabin light or a small direct phone flash with realistic deep shadows",
        _CLOTHING,
    ),
    _PhotoScenario(
        "Идет с цветами",
        "walking",
        "an ordinary apartment or hotel corridor with repeating doors and wall lights",
        "walking away while carrying a bouquet or small shopping bag as part of the moment",
        "a companion's handheld phone snapshot from several steps behind",
        "a slightly off-center walking frame with mild motion in a hand or foot",
        "warm corridor lighting with realistic noise in darker areas",
        _CLOTHING,
    ),
    _PhotoScenario(
        "В лифте с кофе",
        "standing",
        "a normal stainless-steel elevator with small scuffs and reflections",
        "waiting for the doors while holding a takeaway cup and turning naturally",
        "an ordinary phone photo from behind or at a three-quarter angle",
        "a close vertical crop with the torso large and no forced full-body view",
        "flat mixed elevator light with unretouched colors and gentle phone grain",
        _CLOTHING,
    ),
    _PhotoScenario(
        "Переходит улицу",
        "walking",
        "a real city crossing or sidewalk with a few softly recognizable everyday details",
        "walking mid-step and glancing away, with arms swinging naturally",
        "a friend catching the moment on a phone from waist or chest height",
        "an imperfect three-quarter frame with a natural crop around mid-calf",
        "ordinary late-afternoon daylight with no cinematic grading",
        _CLOTHING,
    ),
    _PhotoScenario(
        "За столиком кафе",
        "sitting",
        "a small neighborhood cafe table near a window with cups and personal items",
        "sitting sideways, checking a phone or talking to someone outside the frame",
        "a casual phone photograph from the opposite chair",
        "a relaxed seated crop from head to lap with slight foreground clutter",
        "available window light, natural skin texture and realistic white balance",
        _CLOTHING,
    ),
    _PhotoScenario(
        "На ступенях",
        "sitting",
        "the steps of an apartment entrance or a quiet public building",
        "sitting loosely, adjusting a shoe or resting elbows on knees between movements",
        "a quick handheld phone photo from a mild high angle",
        "an asymmetrical seated composition that still keeps the print inside the safe area",
        "open shade with ordinary contrast and no beauty lighting",
        _ALL_GARMENTS,
    ),
    _PhotoScenario(
        "На парковке",
        "walking",
        "a normal parking area beside an everyday car and simple concrete walls",
        "walking toward the car while looking for keys or closing a door",
        "a spontaneous phone image taken by a companion from a few meters away",
        "a loose three-quarter frame with mild wide-angle distortion at the edges",
        "overcast daylight or realistic parking-garage light",
        _ALL_GARMENTS,
    ),
    _PhotoScenario(
        "Поправляет кепку на улице",
        "close-up",
        "a busy but ordinary outdoor gathering with people softly out of focus behind",
        "lightly touching the brim while listening to someone nearby, not presenting the cap",
        "a close phone snapshot from a natural three-quarter angle",
        "head-and-shoulders framing with the cap large and face only partly emphasized",
        "soft outdoor daylight with phone JPEG texture and no studio separation",
        frozenset({"cap"}),
    ),
    _PhotoScenario(
        "В кепке на диване",
        "sitting",
        "a warm lived-in room with a worn leather or fabric sofa",
        "sitting back comfortably and looking down toward one side",
        "a casual phone photo from slightly above eye level",
        "a seated upper-body frame where the cap and its front panel remain easy to inspect",
        "warm household light with normal shadows under the brim",
        frozenset({"cap"}),
    ),
    _PhotoScenario(
        "Кепка крупным планом",
        "close-up",
        "a simple outdoor wall, rocky seaside edge or neighborhood background",
        "wearing the cap low and turning the head slightly while one hand nears the brim",
        "an informal close smartphone shot with realistic near-field perspective",
        "a tight head-and-upper-shoulder crop with the entire crown and brim inside frame",
        "direct daylight or mild phone flash revealing fabric, seams and stitching",
        frozenset({"cap"}),
    ),
)

_PEOPLE = {
    "women": (
        "a fictional Central Asian woman in her mid twenties with an ordinary distinctive face, natural skin and dark hair",
        "a fictional Central Asian woman in her early thirties with a softly angular face, visible skin texture and dark wavy hair",
        "a fictional woman in her late twenties with a round ordinary face, subtle freckles and a dark bob haircut",
        "a fictional Central Asian woman in her mid thirties with an expressive everyday face and dark hair tied loosely",
        "a fictional woman in her early twenties with natural brows, a small facial asymmetry and long dark hair",
        "a fictional Central Asian woman around forty with a confident ordinary face, fine skin lines and dark hair",
    ),
    "men": (
        "a fictional Central Asian man in his mid twenties with an ordinary distinctive face, natural skin and short dark hair",
        "a fictional man in his early thirties with a softly angular face, dark wavy hair and light stubble",
        "a fictional Central Asian man in his late twenties with a round ordinary face and dark curly hair",
        "a fictional man in his mid thirties with visible skin texture, close-cropped dark hair and a short beard",
        "a fictional Central Asian man in his early twenties with a small facial asymmetry and medium-length dark hair",
        "a fictional Central Asian man around forty with an everyday face, fine skin lines and short salt-and-pepper hair",
    ),
}


def choose_photo_directions(
    count: int,
    rng: Optional[random.Random] = None,
    *,
    target_gender: Literal["women", "men", "unisex"] = "unisex",
    garment_type: Optional[GarmentType] = None,
    exclude_labels: Optional[list[str]] = None,
) -> list[PhotoDirection]:
    if count < 1:
        raise ValueError("Количество вариантов должно быть больше нуля")
    picker = rng or secrets.SystemRandom()
    pool = [
        item
        for item in _PHOTO_SCENARIOS
        if garment_type is None or garment_type in item.garments
    ]
    excluded = set(exclude_labels or [])
    unused = [item for item in pool if item.label not in excluded]
    if len(unused) >= count:
        selected = picker.sample(unused, count)
    elif count <= len(pool):
        selected = picker.sample(pool, count)
    else:
        selected = [picker.choice(pool) for _ in range(count)]
    directions: list[PhotoDirection] = []
    used_people: set[str] = set()
    for item in selected:
        gender = (
            picker.choice(("women", "men"))
            if target_gender == "unisex"
            else target_gender
        )
        available_people = [p for p in _PEOPLE[gender] if p not in used_people]
        person = picker.choice(available_people or list(_PEOPLE[gender]))
        used_people.add(person)
        directions.append(
            PhotoDirection(
                label=item.label,
                gender=gender,
                pose_kind=item.pose_kind,
                person=person,
                setting=item.setting,
                pose=item.pose,
                camera=item.camera,
                framing=item.framing,
                light=item.light,
                seed=picker.randrange(1, 2_147_483_647),
            )
        )
    return directions


def build_model_photo_prompt(
    spec: Optional[MockupSpec],
    direction: PhotoDirection,
    request_token: str,
    *,
    has_separate_print: bool = False,
) -> str:
    if spec is None:
        measurements = (
            "First infer whether this is a T-shirt, hoodie, sweatshirt, long-sleeve, "
            "zip hoodie, jacket or cap. Infer its printed side, construction, color, "
            "fabric finish, fit, exact print width ratio, exact print height ratio "
            "and top offset directly from the supplied product mockup."
        )
        is_cap = False
    else:
        is_cap = spec.garment_type == "cap"
        if is_cap:
            measurements = (
                f"This is a {spec.garment_type}, color: {spec.shirt_color}, material "
                f"and finish: {spec.fabric_finish}, fit and shape: {spec.fit}. The "
                f"print is on the {spec.side} and is about "
                f"{spec.print_width_percent}% of the usable front crown panel width "
                f"and {spec.print_height_percent}% of its usable height. Its top "
                f"offset is about {spec.print_top_offset_percent}% of that panel "
                f"height. Construction: {spec.construction_details}. The intended "
                f"wearer is {spec.target_gender}."
            )
        else:
            measurements = (
                f"The printed side is the {spec.side}. The garment is a "
                f"{spec.garment_type}, color: {spec.shirt_color}, fabric finish: "
                f"{spec.fabric_finish}, fit: {spec.fit}. The print width is about "
                f"{spec.print_width_percent}% of the wearable torso panel width. "
                f"The print height is about {spec.print_height_percent}% of the "
                f"garment height from neckline to hem. Its top begins about "
                f"{spec.print_top_offset_percent}% of that height below the neckline. "
                f"Its left edge begins about {spec.print_left_offset_percent}% of "
                f"the torso panel width from the left panel edge, with the print "
                f"center at {spec.print_center_x_percent}% of the panel width. "
                f"Construction: {spec.construction_details}. The intended wearer "
                f"is {spec.target_gender}, target age group {spec.target_age_group}. "
                f"The artwork mood tags are {', '.join(spec.moods)}."
            )

    source_rule = (
        "Two source images are supplied. The first image is the placement reference "
        "for garment type, color, cut, side, scale and position. The second image is "
        "the exact high-quality print source. Use the second image for every artwork "
        "pixel and use the first only to preserve placement on the product."
        if has_separate_print
        else (
            "One source image is supplied. Use the artwork visible on that product as "
            "the locked print source."
        )
    )

    if is_cap:
        product_physics = (
            "CAP-SPECIFIC DTF PHYSICS:\n"
            "- This is a real DTF heat-transfer film on a curved cap panel, not "
            "embroidery, woven thread, a patch, vinyl lettering or a floating label.\n"
            "- Keep the cap's real crown construction visible: the center vertical "
            "panel seam, rows of stitching on the brim, panel joins and eyelets. Do "
            "not erase or exaggerate them.\n"
            "- If the source print crosses the center seam, the DTF film follows the "
            "small raised seam ridge. Show a subtle vertical change in curvature and "
            "tiny local waviness through the printed area, while all letters and "
            "artwork remain readable and aligned.\n"
            "- The film has a very mild satin surface response and conforms to the "
            "rounded crown. It is never perfectly flat and never turns into raised "
            "embroidered fibers.\n"
        )
        composition = (
            "- Use the requested close or seated direction. Frame the full crown, "
            "brim and printed front panel safely inside the vertical 4:5 image. The "
            "rest of the person may be cropped naturally.\n"
            "- The cap must look worn normally on a real head, with believable brim "
            "shadow and hair interaction. Do not create a catalog cutout or product "
            "floating alone unless the supplied source itself is only a product shot.\n"
        )
    else:
        product_physics = (
            "REAL DTF ON CLOTHING:\n"
            "- The artwork is a thin opaque DTF heat-transfer layer bonded to the "
            "fabric surface. It follows body curvature and folds with tiny local "
            "wrinkles and small changes in reflection, sharpness and shadow.\n"
            "- DTF is not screen ink soaked into the weave. Keep its printed colors "
            "recognizable and mostly opaque, with only a very mild satin surface "
            "response. Scene light and white balance affect garment and print together.\n"
            "- Reduce the flat mockup's digital punch only enough to match a real "
            "phone photo. No neon glow, luminous white, uniform brightness or sticker "
            "effect. Do not blur the artwork to fake realism.\n"
        )
        composition = (
            "- Allow a natural seated, walking, active or standing composition as "
            "specified. Do not force the wearer into a straight catalog stance.\n"
            "- The complete printed artwork and the garment panel carrying it must "
            "remain inside the central safe area. Legs, hands or unused background "
            "may be cropped naturally. The person does not need to fill a fixed "
            "head-to-mid-thigh template.\n"
            "- For a back print, shoot naturally from behind or a rear three-quarter "
            "angle. A full face is unnecessary. Move long hair away from the printed "
            "area without making the hairstyle look staged.\n"
        )

    return (
        "Create one believable everyday smartphone photograph for a real clothing "
        "shop's social page. Use the visual language of an ordinary moment captured "
        "by a friend, not a fashion campaign, studio mockup, stock photo or polished "
        "AI portrait. The supplied image is the only product reference. Ignore its "
        "presentation background, mockup shadows, watermarks and writing outside the "
        "physical product. Do not copy its original pose or scene.\n\n"
        "LOCKED PRODUCT ARTWORK:\n"
        f"0. {source_rule}\n"
        "1. Transfer the complete print as locked source artwork. Preserve every "
        "visible letter, number, face within the art, line, ornament, spacing, color "
        "relationship, outer contour and aspect ratio. Do not redraw, rewrite, "
        "translate, correct, simplify, crop, extend, duplicate or invent anything.\n"
        "2. Transparent or empty space around separate artwork elements must remain "
        "the garment's own fabric. Never invent a rectangular backing, dark box, "
        "poster edge, halo or border unless that shape is clearly part of the source "
        "design itself.\n"
        "3. Keep the exact source side, relative scale and placement. Never enlarge "
        "the print to make it more dramatic. Measure against the usable garment or "
        "cap panel, not the whole image canvas.\n"
        "4. Match the product color, washed or clean finish, cut, seams and construction. "
        "People or faces printed inside the artwork remain only inside the print; the "
        "real wearer is a different fictional non-celebrity adult.\n"
        f"5. {measurements}\n\n"
        f"{product_physics}\n"
        "REFERENCE-BASED REAL PHOTO DIRECTION:\n"
        f"- Wearer: {direction.person}.\n"
        f"- Location: {direction.setting}.\n"
        f"- Action, not pose: {direction.pose}.\n"
        f"- Camera: {direction.camera}.\n"
        f"- Framing: {direction.framing}.\n"
        f"- Light: {direction.light}.\n"
        "- The person is genuinely occupied with the action. Do not make them stop, "
        "square their shoulders or present the product to camera. A hand, bag, cup or "
        "prop may overlap a small non-printed area naturally, but never hide the print.\n"
        "- Use believable anatomy, ordinary hands, pores, small skin variations, "
        "flyaway hairs and true fabric weight. Keep facial character asymmetrical and "
        "unretouched. No beauty filter, perfect teeth, waxy skin or mannequin posture.\n"
        "- The person gender must match the intended audience found in the artwork. "
        "A design dominated by a woman or feminine styling belongs on a woman unless "
        "the source clearly signals unisex or menswear.\n\n"
        "PHONE-CAMERA REALISM:\n"
        "- Use a normal 24-35 mm equivalent phone lens, modest dynamic range, mild "
        "JPEG texture, restrained sharpening and a little sensor noise in shadows. "
        "Depending on the specified light, allow slight motion softness, a small "
        "direct flash or mildly imperfect white balance.\n"
        "- Keep some ordinary background detail. Avoid fake creamy bokeh, cinematic "
        "teal-orange grading, perfect symmetry, centered advertising composition, "
        "flawless studio exposure and hyper-detailed synthetic skin.\n\n"
        "FORMAT AND SAFE AREA:\n"
        "- Vertical 4:5 image for a Telegram and social-media product post.\n"
        f"{composition}"
        "- Keep the full product and print at least 8% away from every image edge so "
        "Telegram previews on different devices do not crop important product details. "
        "The surrounding body and scene may use a more relaxed asymmetric frame.\n"
        "- The product and print remain readable without looking staged. No collage, split screen, "
        "mockup board, border, caption, watermark or extra graphic.\n"
        f"- Variation token: {request_token}-{direction.seed}. Use it only to make "
        "this wearer and photographic moment different from earlier results."
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
            logger.exception(
                "Не удалось измерить макет, используется визуальный анализ"
            )
            return None

    def _analyze_mockup_sync(
        self,
        image_bytes: bytes,
        mime_type: str,
    ) -> MockupSpec:
        try:
            with Image.open(io.BytesIO(image_bytes)) as source:
                source.load()
                image_width, image_height = source.size
        except (UnidentifiedImageError, OSError) as error:
            raise ValueError("Не удалось открыть изображение макета") from error
        prompt = (
            "Analyze this clothing product mockup for a later deterministic placement "
            "pipeline. Inspect only the physical "
            "product and the DTF artwork placed on it. Ignore presentation graphics, "
            "background, shadows and watermarks outside the product. Classify the "
            "product as exactly one of: t-shirt, hoodie, sweatshirt, long-sleeve, "
            "zip-hoodie, cap or jacket. Return whether the visible printed area is "
            "front or back, the exact product color, material or fabric finish, fit "
            "and a short description of construction details such as neckline, hood, "
            "zip, panel seams, central cap seam and brim stitching. Return two normalized "
            "boxes in percentages of the full supplied image. garment_panel_box is the "
            "usable print-bearing torso panel from neckline to hem, excluding sleeves, "
            "or the usable front crown panel for a cap. print_box is the tight bounding "
            "box around the complete artwork including all letters and separate elements. "
            "Do not use the entire canvas or garment sleeves as the panel box. The code "
            "will calculate width, height and offsets from these boxes, so do not return "
            "precomputed ratios. Also infer the intended wearer from "
            "the artwork itself: women when a female figure or clearly feminine "
            "styling dominates, men when a male figure or clearly masculine styling "
            "dominates, and unisex only for a genuinely neutral design. Infer a broad "
            "adult target age group and 1 to 4 mood tags from the allowed enum. Briefly "
            "describe the print theme. Do not infer audience from the blank product cut. "
            "analysis_confidence reflects visibility of the garment and artwork."
        )
        response = self.client.models.generate_content(
            model=self.analysis_model,
            contents=[
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_DetectedMockup,
            ),
        )
        if not response.text:
            raise RuntimeError("Gemini не вернул параметры макета")
        detected = _DetectedMockup.model_validate_json(response.text)
        return build_mockup_spec(
            detected,
            image_width=image_width,
            image_height=image_height,
        )

    async def analyze_print_asset(
        self,
        image_bytes: bytes,
        mime_type: str,
    ) -> PrintAssetSpec:
        file_info = inspect_print_file(image_bytes)
        detected = await asyncio.to_thread(
            self._analyze_print_asset_sync,
            image_bytes,
            mime_type,
        )
        return PrintAssetSpec(
            **file_info,
            **detected.model_dump(),
        )

    def _analyze_print_asset_sync(
        self,
        image_bytes: bytes,
        mime_type: str,
    ) -> _DetectedPrint:
        prompt = (
            "Analyze only this isolated DTF print artwork. Transparent pixels are empty "
            "space and are not a rectangular background. Preserve wording exactly but "
            "do not repeat the wording in the response. Infer the intended adult wearer "
            "from the artwork: women for clearly feminine art, men for clearly masculine "
            "art, and unisex only when neutral. Infer a broad adult target age group, 1 "
            "to 4 mood tags from the supplied enum, a short neutral theme description, "
            "and the dominant visible colors. analysis_confidence reflects how clear the "
            "artwork is. Do not treat a person drawn inside the print as the future model."
        )
        response = self.client.models.generate_content(
            model=self.analysis_model,
            contents=[
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_DetectedPrint,
            ),
        )
        if not response.text:
            raise RuntimeError("Gemini не вернул анализ PNG принта")
        return _DetectedPrint.model_validate_json(response.text)

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
    ) -> GeneratedModelPhoto:
        try:
            return await asyncio.to_thread(
                self._generate_variant_sync,
                image_bytes,
                mime_type,
                spec,
                direction,
                request_token,
                print_image_bytes,
                print_mime_type,
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
        print_image_bytes: Optional[bytes] = None,
        print_mime_type: Optional[str] = None,
    ) -> GeneratedModelPhoto:
        prompt = build_model_photo_prompt(
            spec,
            direction,
            request_token,
            has_separate_print=bool(print_image_bytes),
        )
        contents: list[object] = [
            prompt,
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        ]
        if print_image_bytes:
            contents.extend(
                [
                    "Exact isolated print source follows:",
                    types.Part.from_bytes(
                        data=print_image_bytes,
                        mime_type=print_mime_type or "image/png",
                    ),
                ]
            )
        response = self.client.models.generate_content(
            model=self.image_model,
            contents=contents,
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
