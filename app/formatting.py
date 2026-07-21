import string
import re
from pathlib import Path
from urllib.parse import quote


FIELD_ALIASES = {
    "title": "title",
    "description": "description",
    "hashtags": "hashtags",
    "size": "size",
    "price": "price",
    "Тип товара": "garment_type",
    "тип товара": "garment_slug",
    "Название принта": "design_name",
    "название принта": "design_name",
    "Короткое описание, передающее настроение принта": "description",
    "короткое описание, передающее настроение принта": "description",
    "Короткое описание": "description",
    "короткое описание": "description",
    "Размеры": "size",
    "размеры": "size",
    "Цена": "price",
    "цена": "price",
    "Тематика принта": "theme_slug",
    "тематика принта": "theme_slug",
}
ALLOWED_FIELDS = set(FIELD_ALIASES)
TITLE_FIELDS = {"title", "Тип товара", "тип товара", "Название принта", "название принта"}
SIZE_FIELDS = {"size", "Размеры", "размеры"}
PRICE_FIELDS = {"price", "Цена", "цена"}
HASHTAG_FIELDS = {"hashtags", "тип товара", "Тип товара", "Тематика принта", "тематика принта"}
GARMENT_TYPES = ("Футболка", "Худи", "Свитшот", "Лонгслив", "Кепка")


class TemplateError(ValueError):
    pass


def validate_template(template: str) -> None:
    fields = {
        field_name
        for _, field_name, _, _ in string.Formatter().parse(template)
        if field_name
    }
    unknown = fields - ALLOWED_FIELDS
    if unknown:
        raise TemplateError("неизвестные поля: " + ", ".join(sorted(unknown)))
    missing = []
    if not fields.intersection(TITLE_FIELDS):
        missing.append("{title} или {Тип товара} и {Название принта}")
    if not fields.intersection(SIZE_FIELDS):
        missing.append("{size} или {Размеры}")
    if not fields.intersection(PRICE_FIELDS):
        missing.append("{price} или {Цена}")
    if not fields.intersection(HASHTAG_FIELDS):
        missing.append("{hashtags} или {тип товара} и {тематика принта}")
    if missing:
        raise TemplateError("добавьте поля: " + "; ".join(missing))


def load_template(path: Path) -> str:
    template = path.read_text(encoding="utf-8").strip()
    validate_template(template)
    return template


def render_caption(
    template_path: Path,
    *,
    title: str,
    description: str,
    size: str,
    price: str,
    garment_type: str = "",
    design_name: str = "",
    theme_hashtag: str = "",
) -> str:
    template = load_template(template_path)
    return render_caption_text(
        template,
        title=title,
        description=description,
        size=size,
        price=price,
        garment_type=garment_type,
        design_name=design_name,
        theme_hashtag=theme_hashtag,
    )


def render_caption_text(
    template: str,
    *,
    title: str,
    description: str,
    size: str,
    price: str,
    garment_type: str = "",
    design_name: str = "",
    theme_hashtag: str = "",
) -> str:
    validate_template(template)
    parsed_type, parsed_name = split_product_title(title)
    garment_type = clean_garment_type(garment_type or parsed_type)
    design_name = clean_design_name(design_name or parsed_name or title)

    mood_description = description.strip()
    legacy_hashtags = []
    if mood_description.startswith("#"):
        legacy_hashtags = re.findall(r"#[\w]+", mood_description, flags=re.UNICODE)
        mood_description = ""

    garment_slug = garment_type.lower()
    theme_slug = normalize_hashtag_value(theme_hashtag)
    if not theme_slug and legacy_hashtags:
        candidates = [item.lstrip("#") for item in legacy_hashtags]
        theme_slug = next(
            (item for item in candidates if item.lower() != garment_slug),
            "принт",
        )
    if not theme_slug:
        theme_slug = "принт"

    full_title = title.strip() or f'{garment_type} "{design_name}"'
    hashtags = f"#{garment_slug} #{theme_slug}"
    values = {
        "title": full_title,
        "description": mood_description,
        "hashtags": hashtags,
        "size": size.strip(),
        "price": price.strip(),
        "garment_type": garment_type,
        "garment_slug": garment_slug,
        "design_name": design_name,
        "theme_slug": theme_slug,
    }
    caption = template.format_map(
        {field: values[target] for field, target in FIELD_ALIASES.items()}
    ).strip()
    if len(caption) > 1024:
        raise TemplateError(
            f"подпись содержит {len(caption)} символов, максимум для фото 1024"
        )
    return caption


def split_product_title(title: str) -> tuple[str, str]:
    clean = " ".join(title.strip().split())
    pattern = rf"^({'|'.join(GARMENT_TYPES)})\s*[\"“”«]?(.*?)[\"“”»]?$"
    match = re.match(pattern, clean, flags=re.IGNORECASE)
    if not match:
        return "", clean.strip('"“”«» ')
    garment = clean_garment_type(match.group(1))
    return garment, clean_design_name(match.group(2))


def clean_garment_type(value: str) -> str:
    clean = " ".join(value.strip().split())
    for garment in GARMENT_TYPES:
        if clean.casefold() == garment.casefold():
            return garment
    return clean or "Футболка"


def clean_design_name(value: str) -> str:
    return " ".join(value.strip().strip('"“”«»').split())


def normalize_hashtag_value(value: str) -> str:
    clean = value.strip().lstrip("#").lower()
    return re.sub(r"[^\w]+", "_", clean, flags=re.UNICODE).strip("_")


def contact_link(username: str, title: str) -> str:
    draft = f"Здравствуйте! Хочу заказать: {title}"
    return f"https://t.me/{username}?text={quote(draft)}"


def normalize_price(value: str) -> str:
    price = " ".join(value.strip().split())
    if not price:
        raise ValueError("цена не может быть пустой")
    if len(price) > 40:
        raise ValueError("цена слишком длинная")
    if price.replace(" ", "").isdigit():
        return f"{price} манат"
    return price


def normalize_size(value: str) -> str:
    size = " ".join(value.strip().split())
    if not size:
        raise ValueError("размеры не могут быть пустыми")
    if len(size) > 40:
        raise ValueError("вариант размеров слишком длинный")
    return size
