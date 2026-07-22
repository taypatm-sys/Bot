from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional


@dataclass(frozen=True)
class ScheduledPost:
    id: int
    author_id: int
    photo_file_id: str
    title: str
    description: str
    garment_type: str
    design_name: str
    theme_hashtag: str
    size: str
    price: str
    scheduled_at_utc: datetime
    next_attempt_at_utc: datetime
    status: str
    attempts: int
    last_error: Optional[str]
    published_message_id: Optional[int]
    created_at_utc: datetime


@dataclass(frozen=True)
class ProductPreset:
    id: int
    name: str
    size: str
    price: str


@dataclass(frozen=True)
class ReferenceImportJob:
    id: int
    source_url: str
    resolved_image_url: Optional[str]
    image_bytes: Optional[bytes]
    image_mime_type: Optional[str]
    attempt_count: int


@dataclass(frozen=True)
class ReferenceAsset:
    id: int
    source_url: str
    resolved_image_url: str
    image_bytes: bytes
    image_mime_type: str
    thumbnail_bytes: bytes
    width: int
    height: int
    tags: dict[str, Any]
    use_count: int
    last_used_at_utc: Optional[datetime]
    cooldown_until_utc: Optional[datetime]
