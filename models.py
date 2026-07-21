from dataclasses import dataclass
from datetime import datetime
from typing import Optional


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
