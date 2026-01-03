"""Data models for Midas."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class NewsCategory(str, Enum):
    """Category of news."""

    LEGISLATION = "legislation"  # 法案
    REGULATION = "regulation"  # 規制
    POLICY = "policy"  # 政策
    EXECUTIVE_ORDER = "executive_order"  # 大統領令
    TRADE = "trade"  # 通商
    TECHNOLOGY = "technology"  # 技術政策
    OTHER = "other"


class NewsItem(BaseModel):
    """A news item from RSS feed."""

    id: str = Field(description="Unique identifier (hash of URL)")
    title: str = Field(description="News title")
    source: str = Field(description="Source name (e.g., 'whitehouse')")
    url: str = Field(description="Original URL")
    published: datetime = Field(description="Published datetime")
    content: str = Field(default="", description="Full content or description")
    summary: str | None = Field(default=None, description="LLM-generated summary")
    is_structural: bool = Field(
        default=False, description="Whether this relates to structural change"
    )
    category: NewsCategory | None = Field(default=None, description="News category")
    relevance_reason: str | None = Field(
        default=None, description="Why this news is relevant to structural change"
    )


class NewsCollection(BaseModel):
    """Collection of news items."""

    items: list[NewsItem] = Field(default_factory=list)
    fetched_at: datetime = Field(default_factory=datetime.now)
    source_count: dict[str, int] = Field(default_factory=dict)
