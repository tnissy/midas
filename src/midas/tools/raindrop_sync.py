"""Raindrop.io API client for syncing filtered news items.

This module provides functionality to sync filtered news items to Raindrop.io,
a bookmark management service with Web UI and mobile apps.
"""

import json
import logging
from pathlib import Path
from typing import Any

import requests

from midas.config import DATA_DIR
from midas.models import FilteredNewsItem, NewsCluster

logger = logging.getLogger(__name__)

# =============================================================================
# Collection Configuration
# =============================================================================

# Single collection name for all filtered news
DEFAULT_COLLECTION_NAME = "Midas"

# Category directories to process
CATEGORY_DIRS = ["us_gov", "tech", "general", "other_gov"]

# =============================================================================
# Sync Configuration
# =============================================================================

RAINDROP_SYNC_DIR = DATA_DIR / "raindrop_sync"
SYNCED_IDS_FILE = RAINDROP_SYNC_DIR / "synced_ids.json"


class RaindropClient:
    """Raindrop.io API Client for syncing news items."""

    BASE_URL = "https://api.raindrop.io/rest/v1"

    def __init__(self, api_token: str):
        """Initialize Raindrop client with API token.

        Args:
            api_token: Raindrop.io API token (Test Token or OAuth token)
        """
        self.api_token = api_token
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }

    def get_or_create_collection(self, name: str) -> int:
        """Get or create a collection by name.

        Args:
            name: Collection name (e.g., "Midas - US Government")

        Returns:
            Collection ID

        Raises:
            requests.HTTPError: If API request fails
        """
        # Get all collections
        response = requests.get(f"{self.BASE_URL}/collections", headers=self.headers)
        response.raise_for_status()

        # Find existing collection
        for collection in response.json()["items"]:
            if collection["title"] == name:
                logger.info(f"Found existing collection: {name} (ID: {collection['_id']})")
                return collection["_id"]

        # Create new collection
        logger.info(f"Creating new collection: {name}")
        response = requests.post(
            f"{self.BASE_URL}/collection", headers=self.headers, json={"title": name}
        )
        response.raise_for_status()
        collection_id = response.json()["item"]["_id"]
        logger.info(f"Created collection: {name} (ID: {collection_id})")
        return collection_id

    def create_raindrop(
        self, collection_id: int, news_item: FilteredNewsItem, duplicate_urls: list[str] | None = None
    ) -> dict[str, Any]:
        """Create a raindrop (bookmark) from FilteredNewsItem.

        Args:
            collection_id: Target collection ID
            news_item: Filtered news item to create bookmark from
            duplicate_urls: List of duplicate article URLs (optional)

        Returns:
            API response JSON

        Raises:
            requests.HTTPError: If API request fails
        """
        # Build title (Japanese / English if available)
        if news_item.japanese_title:
            title = f"{news_item.japanese_title} / {news_item.title}"
        else:
            title = news_item.title

        # Build excerpt (Why relevant + content snippet)
        excerpt = ""
        if news_item.relevance_reason:
            excerpt += f"**Why relevant:** {news_item.relevance_reason}\n\n"
        excerpt += news_item.content[:500]
        if len(news_item.content) > 500:
            excerpt += "..."

        # Build tags (category + source)
        tags = []
        if news_item.category:
            tags.append(news_item.category.value)
        if news_item.source:
            tags.append(news_item.source)
        if news_item.value_score is not None:
            tags.append(f"value_{news_item.value_score}")

        # Build note (duplicate URLs if any)
        note = ""
        if duplicate_urls:
            note = "**Duplicate articles:**\n"
            for url in duplicate_urls:
                note += f"- {url}\n"

        # Create raindrop data
        raindrop_data = {
            "link": news_item.url,
            "title": title,
            "excerpt": excerpt,
            "tags": tags,
            "collection": {"$id": collection_id},
            "created": news_item.published.isoformat(),
            "type": "link",
        }

        # Add note if available
        if note:
            raindrop_data["note"] = note

        # Send request
        response = requests.post(
            f"{self.BASE_URL}/raindrop", headers=self.headers, json=raindrop_data
        )
        response.raise_for_status()

        logger.info(f"Created raindrop: {news_item.title[:50]}... (ID: {news_item.id})")
        return response.json()


class SyncedIDsManager:
    """Manager for tracking synced news IDs to avoid duplicates."""

    def __init__(self, storage_path: Path):
        """Initialize synced IDs manager.

        Args:
            storage_path: Path to synced_ids.json file
        """
        self.storage_path = storage_path
        self.synced_ids: set[str] = set()
        self.load()

    def load(self) -> None:
        """Load synced IDs from storage file."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.synced_ids = set(data.get("synced_ids", []))
                logger.info(f"Loaded {len(self.synced_ids)} synced IDs from {self.storage_path}")
            except Exception as e:
                logger.error(f"Failed to load synced IDs: {e}")
                self.synced_ids = set()
        else:
            logger.info(f"No synced IDs file found at {self.storage_path}, starting fresh")
            self.synced_ids = set()

    def save(self) -> None:
        """Save synced IDs to storage file."""
        try:
            # Create parent directory if it doesn't exist
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

            # Save synced IDs
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump({"synced_ids": list(self.synced_ids)}, f, indent=2)
            logger.info(f"Saved {len(self.synced_ids)} synced IDs to {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to save synced IDs: {e}")

    def is_synced(self, news_id: str) -> bool:
        """Check if news item has been synced.

        Args:
            news_id: News item ID

        Returns:
            True if already synced, False otherwise
        """
        return news_id in self.synced_ids

    def mark_synced(self, news_id: str) -> None:
        """Mark news item as synced.

        Args:
            news_id: News item ID
        """
        self.synced_ids.add(news_id)

    def mark_multiple_synced(self, news_ids: list[str]) -> None:
        """Mark multiple news items as synced.

        Args:
            news_ids: List of news item IDs
        """
        self.synced_ids.update(news_ids)


# =============================================================================
# Main Sync Function
# =============================================================================


def sync_filtered_news_to_raindrop(api_token: str) -> dict[str, Any]:
    """Sync filtered news items to Raindrop.io.

    Args:
        api_token: Raindrop.io API token

    Returns:
        Result dictionary with sync statistics
    """
    logger.info("Starting Raindrop sync...")

    # Initialize client and synced IDs manager
    client = RaindropClient(api_token)
    synced_ids_manager = SyncedIDsManager(SYNCED_IDS_FILE)

    # Statistics
    stats = {
        "total_found": 0,
        "synced_count": 0,
        "skipped_count": 0,
        "error_count": 0,
        "errors": [],
    }

    # Get or create single collection for all news
    try:
        collection_id = client.get_or_create_collection(DEFAULT_COLLECTION_NAME)
    except Exception as e:
        logger.error(f"Failed to get/create collection '{DEFAULT_COLLECTION_NAME}': {e}")
        stats["errors"].append(f"Collection error: {DEFAULT_COLLECTION_NAME} - {e}")
        return stats

    # Process each category
    for category_dir_name in CATEGORY_DIRS:
        category_dir = DATA_DIR / category_dir_name

        # Skip if category directory doesn't exist
        if not category_dir.exists():
            logger.info(f"Category directory not found: {category_dir}, skipping")
            continue

        # Find all filtered_news_*.json files
        filtered_files = list(category_dir.glob("filtered_news_*.json"))

        if not filtered_files:
            logger.info(f"No filtered news files found in {category_dir}")
            continue

        logger.info(
            f"Processing {len(filtered_files)} filtered news files from {category_dir_name}"
        )

        # Process each filtered news file
        for filtered_file in filtered_files:
            try:
                # Load filtered news items
                with open(filtered_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Handle both single item and list of items
                if isinstance(data, dict):
                    items_data = [data]
                elif isinstance(data, list):
                    items_data = data
                else:
                    logger.warning(f"Unexpected data format in {filtered_file}, skipping")
                    continue

                # Parse FilteredNewsItem objects
                filtered_news_items = []
                for item_data in items_data:
                    try:
                        item = FilteredNewsItem(**item_data)
                        filtered_news_items.append(item)
                    except Exception as e:
                        logger.warning(f"Failed to parse news item: {e}")
                        continue

                stats["total_found"] += len(filtered_news_items)

                # Sync each filtered news item
                for news_item in filtered_news_items:
                    # Skip if already synced
                    if synced_ids_manager.is_synced(news_item.id):
                        logger.debug(f"Skipping already synced item: {news_item.id}")
                        stats["skipped_count"] += 1
                        continue

                    # Skip if filtered out (advertisement, blacklisted, low value)
                    if news_item.is_advertisement:
                        logger.debug(f"Skipping advertisement: {news_item.title[:50]}...")
                        stats["skipped_count"] += 1
                        synced_ids_manager.mark_synced(news_item.id)
                        continue

                    if news_item.is_blacklisted:
                        logger.debug(
                            f"Skipping blacklisted author: {news_item.title[:50]}..."
                        )
                        stats["skipped_count"] += 1
                        synced_ids_manager.mark_synced(news_item.id)
                        continue

                    if news_item.value_score < 4:
                        logger.debug(
                            f"Skipping low-value news (score {news_item.value_score}): {news_item.title[:50]}..."
                        )
                        stats["skipped_count"] += 1
                        synced_ids_manager.mark_synced(news_item.id)
                        continue

                    # TODO: Handle cluster duplicates (for now, sync all)
                    # In the future, we should only sync the representative news
                    # and include duplicate URLs in the note field

                    try:
                        # Create raindrop
                        client.create_raindrop(collection_id, news_item)
                        synced_ids_manager.mark_synced(news_item.id)
                        stats["synced_count"] += 1
                    except Exception as e:
                        logger.error(f"Failed to create raindrop for {news_item.id}: {e}")
                        stats["error_count"] += 1
                        stats["errors"].append(f"{news_item.id}: {e}")

            except Exception as e:
                logger.error(f"Error processing {filtered_file}: {e}")
                stats["error_count"] += 1
                stats["errors"].append(f"{filtered_file.name}: {e}")

    # Save synced IDs
    synced_ids_manager.save()

    # Log summary
    logger.info("=" * 80)
    logger.info("Raindrop sync completed!")
    logger.info(f"Total news items found: {stats['total_found']}")
    logger.info(f"Successfully synced: {stats['synced_count']}")
    logger.info(f"Skipped (already synced/filtered): {stats['skipped_count']}")
    logger.info(f"Errors: {stats['error_count']}")
    if stats["errors"]:
        logger.error("Error details:")
        for error in stats["errors"][:10]:  # Show first 10 errors
            logger.error(f"  - {error}")
    logger.info("=" * 80)

    return stats
