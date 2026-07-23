from dataclasses import dataclass
from typing import Literal, Protocol


GenerationTier = Literal["routine", "complex", "very_complex"]
GenerationProvider = Literal["local", "gemini"]


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
            "local": "локальный режим",
            "gemini": "Gemini",
        }[self.provider]


def choose_generation_model(
    *,
    spec: MockupSpecLike,
    has_separate_print: bool,
    local_composite_safe: bool,
    gemini_lite_model: str,
) -> GenerationDecision:
    """Choose the local compositor whenever it can preserve the product safely.

    Gemini image generation is reserved for cases where direct local replacement
    is unsafe: complex garment construction, strong perspective, occlusion, color
    mismatch, an unremovable existing print or unreliable artwork extraction.
    """

    score = 0
    reasons: list[str] = []

    if spec.side == "back":
        score += 1
        reasons.append("принт сзади")

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
    if any(marker in theme for marker in text_markers):
        score += 2
        reasons.append("важный текст в принте")

    if not has_separate_print:
        score += 1
        reasons.append("принт извлекается из исходного фото")

    if spec.garment_type != "t-shirt":
        score += 4
        reasons.append("сложная конструкция изделия")

    if spec.analysis_confidence < 65:
        score += 2
        reasons.append("пониженная уверенность анализа")

    if max(spec.print_width_percent, spec.print_height_percent) >= 68:
        score += 1
        reasons.append("очень крупный принт")

    if local_composite_safe and spec.garment_type == "t-shirt":
        tier: GenerationTier = "complex" if score >= 4 else "routine"
        return GenerationDecision(
            tier=tier,
            provider="local",
            model="OpenCV local compositor",
            score=score,
            reasons=tuple(reasons) or ("простая локальная замена принта",),
        )

    reasons.append("локальная замена небезопасна")
    return GenerationDecision(
        tier="very_complex",
        provider="gemini",
        model=gemini_lite_model,
        score=max(score, 8),
        reasons=tuple(reasons),
    )
