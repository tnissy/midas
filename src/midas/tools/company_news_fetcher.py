"""Company news fetcher tool using Google News RSS."""

import re
from datetime import datetime
from email.utils import parsedate_to_datetime
from urllib.parse import quote

import feedparser
import httpx

from midas.models import CompanyNews


def _parse_google_news_date(entry: dict) -> datetime:
    """Parse published date from Google News RSS entry."""
    published = entry.get("published") or entry.get("updated")
    if published:
        try:
            return parsedate_to_datetime(published)
        except (ValueError, TypeError):
            pass
    return datetime.now()


def _extract_source_from_title(title: str) -> tuple[str, str]:
    """Extract source name from Google News title format.

    Google News format: "Article Title - Source Name"
    Returns: (clean_title, source_name)
    """
    if " - " in title:
        parts = title.rsplit(" - ", 1)
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
    return title, "Unknown"


def _clean_html(text: str) -> str:
    """Remove HTML tags from text."""
    clean = re.sub(r"<[^>]+>", "", text)
    clean = re.sub(r"\s+", " ", clean)
    return clean.strip()


async def fetch_company_news(
    query: str,
    days_back: int = 180,
    max_results: int = 100,
) -> list[CompanyNews]:
    """Fetch news about a company from Google News RSS.

    Args:
        query: Search query (company name, ticker symbol, etc.)
        days_back: Number of days to look back (for filtering)
        max_results: Maximum number of results to return

    Returns:
        List of CompanyNews objects sorted by date (newest first)
    """
    # Build Google News RSS URL
    encoded_query = quote(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
    except Exception as e:
        print(f"Error fetching news for '{query}': {e}")
        return []

    parsed = feedparser.parse(response.text)
    if parsed.bozo and not parsed.entries:
        print(f"Error parsing news feed: {parsed.bozo_exception}")
        return []

    news_items: list[CompanyNews] = []
    cutoff_date = datetime.now().replace(tzinfo=None) - __import__("datetime").timedelta(
        days=days_back
    )

    for entry in parsed.entries[:max_results]:
        try:
            published = _parse_google_news_date(entry)
            # Make datetime naive for comparison
            if published.tzinfo:
                published = published.replace(tzinfo=None)

            # Skip old news
            if published < cutoff_date:
                continue

            raw_title = entry.get("title", "No title")
            title, source = _extract_source_from_title(raw_title)

            # Extract snippet from summary
            summary = entry.get("summary", "")
            snippet = _clean_html(summary)[:500]

            news_item = CompanyNews(
                title=title,
                source=source,
                url=entry.get("link", ""),
                published=published,
                snippet=snippet,
            )
            news_items.append(news_item)

        except Exception as e:
            print(f"Error parsing news entry: {e}")
            continue

    # Sort by date (newest first)
    news_items.sort(key=lambda x: x.published, reverse=True)

    return news_items


async def fetch_news_around_date(
    query: str,
    target_date: datetime,
    days_before: int = 3,
    days_after: int = 3,
) -> list[CompanyNews]:
    """Fetch news around a specific date.

    Args:
        query: Search query
        target_date: The date to search around
        days_before: Days before target date to include
        days_after: Days after target date to include

    Returns:
        List of CompanyNews filtered to the date range
    """
    # Fetch more news to have enough for filtering
    all_news = await fetch_company_news(query, days_back=365, max_results=200)

    # Filter to date range
    start_date = target_date - __import__("datetime").timedelta(days=days_before)
    end_date = target_date + __import__("datetime").timedelta(days=days_after)

    filtered = []
    for news in all_news:
        news_date = news.published
        if news_date.tzinfo:
            news_date = news_date.replace(tzinfo=None)
        if start_date <= news_date <= end_date:
            filtered.append(news)

    return filtered


async def search_topic_news(
    topic: str,
    additional_keywords: list[str] | None = None,
    max_results: int = 50,
) -> list[dict]:
    """Search for news about a topic using Google News RSS.

    Args:
        topic: The main topic to search for
        additional_keywords: Additional keywords to include in search
        max_results: Maximum number of results

    Returns:
        List of search result dictionaries with title, source, url, snippet
    """
    query_parts = [topic]
    if additional_keywords:
        query_parts.extend(additional_keywords)

    encoded_query = quote(" ".join(query_parts))
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
    except Exception as e:
        print(f"Error searching for '{topic}': {e}")
        return []

    parsed = feedparser.parse(response.text)
    if parsed.bozo and not parsed.entries:
        return []

    results: list[dict] = []
    for entry in parsed.entries[:max_results]:
        title = entry.get("title", "")
        clean_title, source = _extract_source_from_title(title)
        snippet = _clean_html(entry.get("summary", ""))[:500]

        results.append({
            "title": clean_title,
            "source": source,
            "url": entry.get("link", ""),
            "snippet": snippet,
        })

    return results
