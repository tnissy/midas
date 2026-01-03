"""RSS feed fetcher tool with duplicate detection."""

import hashlib
import json
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

import feedparser
import httpx

from midas.models import NewsItem


def _generate_id(url: str) -> str:
    """Generate unique ID from URL."""
    return hashlib.md5(url.encode()).hexdigest()


def _parse_published(entry: dict) -> datetime:
    """Parse published date from RSS entry."""
    published = entry.get("published") or entry.get("updated")
    if published:
        try:
            return parsedate_to_datetime(published)
        except (ValueError, TypeError):
            pass
    return datetime.now()


def _parse_content(entry: dict) -> str:
    """Extract content from RSS entry."""
    if "content" in entry and entry["content"]:
        return entry["content"][0].get("value", "")
    if "summary" in entry:
        return entry["summary"]
    if "description" in entry:
        return entry["description"]
    return ""


def load_fetched_ids(cache_file: Path) -> set[str]:
    """Load previously fetched article IDs from cache."""
    if not cache_file.exists():
        return set()
    with open(cache_file, encoding="utf-8") as f:
        data = json.load(f)
    return set(data.get("ids", []))


def save_fetched_ids(cache_file: Path, ids: set[str]) -> None:
    """Save fetched article IDs to cache."""
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump({"ids": list(ids), "updated_at": datetime.now().isoformat()}, f, indent=2)


async def fetch_single_feed(
    feed: dict,
    fetched_ids: set[str],
) -> list[NewsItem]:
    """Fetch news from a single RSS feed.

    Args:
        feed: Feed definition {"name": ..., "url": ..., "category": ...}
        fetched_ids: Set of already fetched article IDs (will be mutated)

    Returns:
        List of new NewsItem objects
    """
    name = feed["name"]
    url = feed["url"]
    category = feed.get("category", "general")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
    except Exception as e:
        print(f"    Error fetching {name}: {e}")
        return []

    parsed = feedparser.parse(response.text)
    if parsed.bozo and not parsed.entries:
        print(f"    Error parsing {name}: {parsed.bozo_exception}")
        return []

    new_items: list[NewsItem] = []

    for entry in parsed.entries:
        article_id = _generate_id(entry.get("link", "") or entry.get("id", ""))

        # Skip duplicates
        if article_id in fetched_ids:
            continue

        item = NewsItem(
            id=article_id,
            title=entry.get("title", "No title"),
            source=name,
            url=entry.get("link", ""),
            published=_parse_published(entry),
            content=_parse_content(entry),
        )
        new_items.append(item)
        fetched_ids.add(article_id)

    return new_items


async def fetch_feeds(
    feeds: list[dict],
    cache_file: Path,
) -> list[NewsItem]:
    """Fetch news from multiple RSS feeds with duplicate detection.

    Args:
        feeds: List of feed definitions [{"name": ..., "url": ..., "category": ...}, ...]
        cache_file: Path to the cache file for duplicate detection

    Returns:
        List of new NewsItem objects (duplicates excluded)
    """
    # Ensure cache directory exists
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    # Load existing IDs
    fetched_ids = load_fetched_ids(cache_file)
    print(f"  Cache contains {len(fetched_ids)} previously fetched articles")

    all_items: list[NewsItem] = []

    for feed in feeds:
        print(f"  Fetching: {feed['name']}...")
        items = await fetch_single_feed(feed, fetched_ids)
        all_items.extend(items)
        print(f"    -> {len(items)} new articles")

    # Save updated cache
    save_fetched_ids(cache_file, fetched_ids)
    print(f"  Cache updated: {len(fetched_ids)} total article IDs")

    return all_items
