from pathlib import Path

from app.formatting import load_template, validate_template
from app.storage import PostRepository


CAPTION_TEMPLATE_KEY = "caption_template"


class CaptionTemplateStore:
    def __init__(self, *, repository: PostRepository, fallback_path: Path):
        self.repository = repository
        self.fallback_path = fallback_path
        self._template = ""

    def initialize(self) -> None:
        fallback = load_template(self.fallback_path)
        self.repository.seed_setting(CAPTION_TEMPLATE_KEY, fallback)
        stored = self.repository.get_setting(CAPTION_TEMPLATE_KEY) or fallback
        validate_template(stored)
        self._template = stored.strip()

    def get(self) -> str:
        if not self._template:
            self.initialize()
        return self._template

    def set(self, template: str) -> None:
        clean = template.strip()
        validate_template(clean)
        self.repository.set_setting(CAPTION_TEMPLATE_KEY, clean)
        self._template = clean
