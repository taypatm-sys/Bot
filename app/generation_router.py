from dataclasses import dataclass
from typing import Literal, Protocol


GenerationTier = Literal["routine", "complex", "very_complex"]
GenerationProvider = Literal["bfl", "gemini", "none"]


class MockupSpecLike(Protocol):
    side: str
    garment_type: str
    print_theme: str
    geometry_mode: str
    analysis_confidence: int
    print_width_percent: int
    print_height_percent: int


@dataclass(frozen=True)
class GenerationDecision:
    tier: GenerationTier
    provider: GenerationProvider
    model: str
    score: int
    reasons: tuple[str, ...]

    @property
    def label_ru(self) -> str:
        return {
            "routine": "обычная",
            "complex": "сложная",
            "very_complex": "очень сложная",
        }[self.tier]

    @property
    def provider_label_ru(self) -> str:
        return {
            "bfl": "FLUX",
            "gemini": "Gemini",
            "none": "не подключен",
        }[self.provider]


def choose_generation_model(
    *,
    spec: MockupSpecLike,
    has_separate_print: bool,
    economy_available: bool,
    economy_model: str,
    gemini_lite_model: str,
) -> GenerationDecision:
    """Route work before any paid request starts.

    Routine and complex tasks use the cheaper FLUX path. Gemini 3.1 Flash Lite
    Image is reserved only for jobs where preserving the artwork is unusually
    difficult. If the economy provider is unavailable, normal work is blocked
    rather than silently spending Gemini credits.
    """

    score = 0
    reasons: list[str] = []

    is_back = spec.side == "back"
    if is_back:
        score += 3
        reasons.append("принт сзади")

    no_print_asset = not has_separate_print
    if no_print_asset:
        score += 3
        reasons.append("нет отдельного PNG принта")

    theme = spec.print_theme.casefold()
    text_markers = (
        "typography",
        "text",
        "letter",
        "word",
        "quote",
        "надпис",
        "типограф",
        "слоган",
    )
    has_text = any(marker in theme for marker in text_markers)
    if has_text:
        score += 4
        reasons.append("важный текст в принте")

    if spec.garment_type in {"cap", "jacket", "zip-hoodie"}:
        score += 2
        reasons.append("сложная конструкция изделия")

    if spec.geometry_mode == "source-guided":
        score += 1
        reasons.append("геометрия берется из исходного фото")

    if spec.analysis_confidence < 70:
        score += 2
        reasons.append("пониженная уверенность анализа")

    large_print = max(spec.print_width_percent, spec.print_height_percent) >= 65
    if large_print:
        score += 1
        reasons.append("крупный принт")

    critical_combination = (
        (has_text and no_print_asset)
        or (is_back and no_print_asset and large_print)
        or score >= 8
    )
    if critical_combination:
        return GenerationDecision(
            tier="very_complex",
            provider="gemini",
            model=gemini_lite_model,
            score=score,
            reasons=tuple(reasons) or ("сложное сохранение принта",),
        )

    tier: GenerationTier = "complex" if score >= 4 else "routine"
    if economy_available:
        return GenerationDecision(
            tier=tier,
            provider="bfl",
            model=economy_model,
            score=score,
            reasons=tuple(reasons) or ("стандартная задача",),
        )

    return GenerationDecision(
        tier=tier,
        provider="none",
        model="",
        score=score,
        reasons=(
            *(tuple(reasons) or ("стандартная задача",)),
            "не подключена экономная модель",
        ),
    )
