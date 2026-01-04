"""Base class for news watchers - eliminates code duplication."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from midas.config import extract_llm_text, DATA_DIR, GEMINI_API_KEY, LLM_MODEL
from midas.logging_config import get_agent_logger
from midas.models import NewsCategory, NewsItem, WatcherType
from midas.tools.rss_fetcher import fetch_feeds
from midas.tools.feedback_loader import build_dynamic_feeds, format_feedback_summary


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class WatcherConfig:
    """Configuration for a news watcher."""

    name: str  # e.g., "us_gov_watcher"
    watcher_type: WatcherType  # e.g., WatcherType.US_GOV
    feeds: list[dict]  # RSS feed definitions
    filter_prompt: str  # LLM prompt for filtering
    data_dir: Path  # Where to save results


# =============================================================================
# State
# =============================================================================


class WatcherState(TypedDict):
    """State for the news collection process."""

    raw_items: list[NewsItem]
    filtered_items: list[NewsItem]
    saved_path: str | None
    error: str | None


# =============================================================================
# Base NewsWatcher Class
# =============================================================================


class NewsWatcher:
    """Base class for all news watchers.

    Handles the common workflow:
    1. Fetch news from RSS feeds
    2. Filter using LLM
    3. Save results
    """

    def __init__(self, config: WatcherConfig):
        """Initialize the news watcher.

        Args:
            config: Watcher configuration
        """
        self.config = config
        self.logger = get_agent_logger(config.name)
        self.cache_file = config.data_dir / "fetched_ids.json"

    async def fetch_news(self, state: WatcherState) -> WatcherState:
        """Fetch news from RSS sources and save raw data."""
        self.logger.info(f"Fetching news for {self.config.name}...")

        # Build dynamic feed list from base feeds + insights
        all_feeds = build_dynamic_feeds(self.config.feeds, self.config.watcher_type)
        dynamic_count = sum(1 for f in all_feeds if f.get("is_dynamic", False))

        if dynamic_count > 0:
            self.logger.info(
                f"  Using {len(self.config.feeds)} base feeds + {dynamic_count} dynamic feeds from insights"
            )
            self.logger.info(format_feedback_summary(self.config.watcher_type))
        else:
            self.logger.info(f"  Using {len(self.config.feeds)} base feeds (no dynamic feeds yet)")

        try:
            items = await fetch_feeds(all_feeds, self.cache_file)
            state["raw_items"] = items
            self.logger.info(f"Total: {len(items)} new items fetched")

            # Save raw items immediately (before LLM filtering)
            if items:
                self.config.data_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                raw_filepath = self.config.data_dir / f"raw_{timestamp}.json"

                raw_data = {
                    "fetched_at": datetime.now().isoformat(),
                    "total": len(items),
                    "items": [item.model_dump(mode="json") for item in items],
                }

                with open(raw_filepath, "w", encoding="utf-8") as f:
                    json.dump(raw_data, f, ensure_ascii=False, indent=2)

                self.logger.info(f"Raw data saved to: {raw_filepath}")

        except Exception as e:
            state["error"] = f"Failed to fetch news: {e}"
            state["raw_items"] = []
        return state

    async def filter_news(self, state: WatcherState) -> WatcherState:
        """Filter news using LLM to identify structural changes."""
        if state.get("error") or not state.get("raw_items"):
            state["filtered_items"] = []
            return state

        self.logger.info("Filtering news with LLM...")

        if not GEMINI_API_KEY:
            self.logger.warning("No API key, skipping LLM filtering")
            state["filtered_items"] = state["raw_items"]
            return state

        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GEMINI_API_KEY)
        filtered: list[NewsItem] = []

        for item in state["raw_items"]:
            try:
                # Prepare prompt
                content = f"Title: {item.title}\n\nContent: {item.content[:1000]}"

                messages = [
                    SystemMessage(content=self.config.filter_prompt),
                    HumanMessage(content=content),
                ]

                response = await llm.ainvoke(messages)
                result_text = extract_llm_text(response.content)

                # Parse JSON response
                if isinstance(result_text, str):
                    start = result_text.find("{")
                    end = result_text.rfind("}") + 1
                    if start != -1 and end > start:
                        result = json.loads(result_text[start:end])

                        if result.get("is_structural"):
                            item.is_structural = True
                            item.relevance_reason = result.get("reason", "")

                            # Set category
                            cat = result.get("category", "other")
                            try:
                                item.category = NewsCategory(cat)
                            except ValueError:
                                item.category = NewsCategory.OTHER

                            filtered.append(item)
                            self.logger.info(f"  [+] {item.title[:60]}...")

            except Exception as e:
                self.logger.warning(f"Error processing item: {e}")
                continue

        state["filtered_items"] = filtered
        self.logger.info(f"Filtered: {len(filtered)} structural news items")
        return state

    def save_results(self, state: WatcherState) -> WatcherState:
        """Save filtered results to JSON file."""
        if not state.get("filtered_items"):
            state["saved_path"] = None
            return state

        self.logger.info("Saving results...")

        # Ensure directory exists
        self.config.data_dir.mkdir(parents=True, exist_ok=True)

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.config.data_dir / f"news_{timestamp}.json"

        # Convert to serializable format
        data = {
            "fetched_at": datetime.now().isoformat(),
            "total_raw": len(state.get("raw_items", [])),
            "total_filtered": len(state["filtered_items"]),
            "items": [item.model_dump(mode="json") for item in state["filtered_items"]],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        state["saved_path"] = str(filepath)
        self.logger.info(f"Saved to: {filepath}")
        return state

    async def run(self) -> WatcherState:
        """Run the complete news watching workflow.

        Returns:
            Final state with filtered news items
        """
        state: WatcherState = {
            "raw_items": [],
            "filtered_items": [],
            "saved_path": None,
            "error": None,
        }

        # Execute workflow
        state = await self.fetch_news(state)
        state = await self.filter_news(state)
        state = self.save_results(state)

        return state
