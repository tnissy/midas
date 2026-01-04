"""Feedback loader for loading insights from model_calibration_agent.

This module provides utilities for watchers to load learned insights
and dynamically adjust their RSS feeds based on feedback.
"""

import json
from pathlib import Path

from midas.config import DATA_DIR
from midas.models import LearnedInsight, SuggestedFeed, WatcherType

# Path to insights directory
INSIGHTS_DIR = DATA_DIR / "learning" / "insights"


def load_all_insights() -> list[LearnedInsight]:
    """Load all stored insights from disk.

    Returns:
        List of all LearnedInsight objects
    """
    insights = []
    if not INSIGHTS_DIR.exists():
        return insights

    for filepath in INSIGHTS_DIR.glob("*.json"):
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
                insights.append(LearnedInsight(**data))
        except Exception:
            continue

    return insights


def get_suggested_feeds_for_watcher(watcher_type: WatcherType) -> list[SuggestedFeed]:
    """Get suggested RSS feeds for a specific watcher.

    Args:
        watcher_type: The type of watcher requesting feeds

    Returns:
        List of SuggestedFeed objects relevant to this watcher
    """
    insights = load_all_insights()
    feeds: list[SuggestedFeed] = []
    seen_urls: set[str] = set()

    for insight in insights:
        # Check if this watcher is a target
        if watcher_type not in insight.target_watchers:
            continue

        # Add suggested feeds for this watcher
        for feed in insight.suggested_feeds:
            if feed.target_watcher == watcher_type and feed.url not in seen_urls:
                feeds.append(feed)
                seen_urls.add(feed.url)

    # Sort by priority (high > medium > low)
    priority_order = {"high": 0, "medium": 1, "low": 2}
    feeds.sort(key=lambda f: priority_order.get(f.priority, 1))

    return feeds


def get_suggested_keywords_for_watcher(watcher_type: WatcherType) -> list[str]:
    """Get suggested keywords for a specific watcher.

    Args:
        watcher_type: The type of watcher requesting keywords

    Returns:
        List of keyword strings relevant to this watcher
    """
    insights = load_all_insights()
    keywords: set[str] = set()

    for insight in insights:
        # Check if this watcher is a target
        if watcher_type not in insight.target_watchers:
            continue

        # Add suggested keywords for this watcher
        for kw in insight.suggested_keywords:
            if kw.target_watcher == watcher_type:
                keywords.add(kw.keyword)

    return list(keywords)


def get_detection_patterns_for_watcher(watcher_type: WatcherType) -> list[str]:
    """Get detection patterns for a specific watcher.

    Args:
        watcher_type: The type of watcher requesting patterns

    Returns:
        List of detection pattern strings
    """
    insights = load_all_insights()
    patterns: set[str] = set()

    for insight in insights:
        if watcher_type in insight.target_watchers:
            for pattern in insight.detection_patterns:
                patterns.add(pattern)

    return list(patterns)


def build_dynamic_feeds(
    base_feeds: list[dict],
    watcher_type: WatcherType,
) -> list[dict]:
    """Build a combined feed list with base feeds and dynamic feeds from insights.

    Args:
        base_feeds: The base/hardcoded feeds for this watcher
        watcher_type: The type of watcher

    Returns:
        Combined list of feed dictionaries
    """
    # Start with base feeds
    all_feeds = list(base_feeds)
    existing_urls = {f["url"] for f in base_feeds}

    # Add dynamic feeds from insights
    suggested_feeds = get_suggested_feeds_for_watcher(watcher_type)
    for feed in suggested_feeds:
        if feed.url not in existing_urls:
            all_feeds.append({
                "name": f"[Dynamic] {feed.name}",
                "url": feed.url,
                "description": feed.reason,
                "is_dynamic": True,
            })
            existing_urls.add(feed.url)

    return all_feeds


def format_feedback_summary(watcher_type: WatcherType) -> str:
    """Format a summary of feedback for a watcher.

    Args:
        watcher_type: The type of watcher

    Returns:
        Formatted string summary
    """
    feeds = get_suggested_feeds_for_watcher(watcher_type)
    keywords = get_suggested_keywords_for_watcher(watcher_type)
    patterns = get_detection_patterns_for_watcher(watcher_type)

    lines = [f"Feedback for {watcher_type.value}:"]

    if feeds:
        lines.append(f"  Dynamic feeds: {len(feeds)}")
        for feed in feeds[:3]:
            lines.append(f"    - {feed.name} ({feed.priority})")

    if keywords:
        lines.append(f"  Keywords to watch: {', '.join(keywords[:5])}")

    if patterns:
        lines.append(f"  Detection patterns: {len(patterns)}")

    if not feeds and not keywords and not patterns:
        lines.append("  No feedback available yet")

    return "\n".join(lines)
