"""News Quality Filter - Filter and enrich collected news items.

This module applies quality filters to collected news items:
1. Advertisement detection (LLM)
2. Value assessment (LLM)
3. Author blacklist check
4. Duplicate news clustering (TF-IDF)
5. Title translation to Japanese (LLM batch)
"""

import json
import logging
from pathlib import Path
from typing import Any

from midas.config import DATA_DIR
from midas.models import NewsItem, FilteredNewsItem, NewsCluster
from midas.tools.quality_filters import (
    detect_advertisement,
    assess_news_value,
    extract_author,
    check_author_blacklist,
    cluster_duplicate_news,
    translate_titles_batch,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Category Mappings
# =============================================================================

CATEGORY_DIRS = {
    "us_gov": DATA_DIR / "us_gov",
    "tech": DATA_DIR / "tech",
    "general": DATA_DIR / "general",
    "other_gov": DATA_DIR / "other_gov",
}

# =============================================================================
# Filter Logic
# =============================================================================


def run_quality_filters(
    category: str,
    skip_ad_detection: bool = False,
    skip_value_assessment: bool = False,
    skip_translation: bool = False,
) -> dict[str, Any]:
    """
    Run quality filters on news items for a specific category.

    Args:
        category: Category name (us_gov, tech, general, other_gov)
        skip_ad_detection: Skip advertisement detection (save LLM costs)
        skip_value_assessment: Skip value assessment (save LLM costs)
        skip_translation: Skip title translation (save LLM costs)

    Returns:
        Result dictionary with statistics
    """
    logger.info(f"Running quality filters for category: {category}")

    category_dir = CATEGORY_DIRS.get(category)
    if not category_dir or not category_dir.exists():
        logger.error(f"Category directory not found: {category_dir}")
        return {"error": f"Category {category} not found"}

    # Find all news_*.json files
    news_files = list(category_dir.glob("news_*.json"))
    if not news_files:
        logger.info(f"No news files found in {category_dir}")
        return {
            "category": category,
            "total_items": 0,
            "filtered_items": 0,
            "filtered_out": 0,
            "clusters": 0,
        }

    # Load all news items
    all_news_items: list[NewsItem] = []
    for news_file in news_files:
        try:
            with open(news_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Handle both single item and list of items
            if isinstance(data, dict):
                items_data = [data]
            elif isinstance(data, list):
                items_data = data
            else:
                logger.warning(f"Unexpected data format in {news_file}, skipping")
                continue

            # Parse NewsItem objects
            for item_data in items_data:
                try:
                    item = NewsItem(**item_data)
                    all_news_items.append(item)
                except Exception as e:
                    logger.warning(f"Failed to parse news item: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error loading {news_file}: {e}")
            continue

    logger.info(f"Loaded {len(all_news_items)} news items from {len(news_files)} files")

    if not all_news_items:
        return {
            "category": category,
            "total_items": 0,
            "filtered_items": 0,
            "filtered_out": 0,
            "clusters": 0,
        }

    # Statistics
    stats = {
        "category": category,
        "total_items": len(all_news_items),
        "filtered_items": 0,
        "filtered_out": 0,
        "ad_detected": 0,
        "blacklisted": 0,
        "low_value": 0,
        "clusters": 0,
        "translated": 0,
    }

    # Step 1: Title translation (batch processing) - do this first before filtering
    if not skip_translation:
        logger.info("Step 1: Translating titles to Japanese...")
        translations = translate_titles_batch(all_news_items)
        stats["translated"] = len(translations)
    else:
        translations = {}

    # Step 2: Duplicate clustering (TF-IDF)
    logger.info("Step 2: Clustering duplicate news...")
    clusters, item_to_cluster = cluster_duplicate_news(all_news_items)
    stats["clusters"] = len(clusters)

    # Step 3: Apply filters to each news item
    filtered_news_items: list[FilteredNewsItem] = []

    for news_item in all_news_items:
        # Extract author
        author = extract_author(news_item)

        # Check blacklist
        is_blacklisted = check_author_blacklist(news_item, author)
        if is_blacklisted:
            stats["blacklisted"] += 1

        # Advertisement detection (optional, costs LLM tokens)
        if not skip_ad_detection:
            is_ad, ad_reason = detect_advertisement(news_item)
        else:
            is_ad, ad_reason = False, ""

        if is_ad:
            stats["ad_detected"] += 1

        # Value assessment (optional, costs LLM tokens)
        if not skip_value_assessment:
            value_score, value_reason = assess_news_value(news_item)
        else:
            value_score, value_reason = 5, ""  # Default mid-range

        if value_score < 4:
            stats["low_value"] += 1

        # Get Japanese title (if translated)
        japanese_title = translations.get(news_item.id)

        # Get cluster ID (if clustered)
        cluster_id = item_to_cluster.get(news_item.id)

        # Create FilteredNewsItem
        filtered_item = FilteredNewsItem(
            **news_item.model_dump(),
            is_advertisement=is_ad,
            value_score=value_score,
            author=author,
            is_blacklisted=is_blacklisted,
            cluster_id=cluster_id,
            japanese_title=japanese_title,
        )

        filtered_news_items.append(filtered_item)

    # Step 4: Save filtered news items
    stats["filtered_items"] = len(filtered_news_items)
    stats["filtered_out"] = stats["ad_detected"] + stats["blacklisted"] + stats["low_value"]

    # Save to filtered_news_{timestamp}.json
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = category_dir / f"filtered_news_{timestamp}.json"

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                [item.model_dump() for item in filtered_news_items],
                f,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        logger.info(f"Saved {len(filtered_news_items)} filtered items to {output_file}")
        stats["output_file"] = str(output_file)

    except Exception as e:
        logger.error(f"Failed to save filtered news: {e}")
        stats["error"] = str(e)

    # Save clusters separately
    if clusters:
        clusters_file = category_dir / f"clusters_{timestamp}.json"
        try:
            with open(clusters_file, "w", encoding="utf-8") as f:
                json.dump(
                    [cluster.model_dump() for cluster in clusters],
                    f,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )
            logger.info(f"Saved {len(clusters)} clusters to {clusters_file}")
            stats["clusters_file"] = str(clusters_file)

        except Exception as e:
            logger.error(f"Failed to save clusters: {e}")

    # Log summary
    logger.info("=" * 80)
    logger.info(f"Quality filter completed for category: {category}")
    logger.info(f"Total items: {stats['total_items']}")
    logger.info(f"Filtered items: {stats['filtered_items']}")
    logger.info(f"  - Advertisements: {stats['ad_detected']}")
    logger.info(f"  - Blacklisted authors: {stats['blacklisted']}")
    logger.info(f"  - Low value: {stats['low_value']}")
    logger.info(f"Clusters: {stats['clusters']}")
    logger.info(f"Translated titles: {stats['translated']}")
    logger.info("=" * 80)

    return stats


def run_all_categories(
    skip_ad_detection: bool = False,
    skip_value_assessment: bool = False,
    skip_translation: bool = False,
) -> dict[str, Any]:
    """
    Run quality filters on all categories.

    Args:
        skip_ad_detection: Skip advertisement detection (save LLM costs)
        skip_value_assessment: Skip value assessment (save LLM costs)
        skip_translation: Skip title translation (save LLM costs)

    Returns:
        Combined results for all categories
    """
    logger.info("Running quality filters for all categories...")

    results = {}
    total_stats = {
        "total_items": 0,
        "filtered_items": 0,
        "filtered_out": 0,
        "ad_detected": 0,
        "blacklisted": 0,
        "low_value": 0,
        "clusters": 0,
        "translated": 0,
    }

    for category in CATEGORY_DIRS.keys():
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Processing category: {category}")
        logger.info(f"{'=' * 80}")

        result = run_quality_filters(
            category,
            skip_ad_detection=skip_ad_detection,
            skip_value_assessment=skip_value_assessment,
            skip_translation=skip_translation,
        )

        results[category] = result

        # Accumulate stats
        for key in total_stats.keys():
            total_stats[key] += result.get(key, 0)

    logger.info("\n" + "=" * 80)
    logger.info("ALL CATEGORIES COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Total items: {total_stats['total_items']}")
    logger.info(f"Filtered items: {total_stats['filtered_items']}")
    logger.info(f"  - Advertisements: {total_stats['ad_detected']}")
    logger.info(f"  - Blacklisted authors: {total_stats['blacklisted']}")
    logger.info(f"  - Low value: {total_stats['low_value']}")
    logger.info(f"Clusters: {total_stats['clusters']}")
    logger.info(f"Translated titles: {total_stats['translated']}")
    logger.info("=" * 80)

    return {
        "categories": results,
        "total_stats": total_stats,
    }
