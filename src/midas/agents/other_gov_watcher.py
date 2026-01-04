"""Other Government Watcher (non-US) - Configuration and runner."""

from midas.agents.news_watcher_base import NewsWatcher, WatcherConfig, WatcherState
from midas.config import DATA_DIR
from midas.models import WatcherType

# =============================================================================
# Feed Definitions (Non-US Government Sources)
# =============================================================================

OTHER_GOV_FEEDS: list[dict] = [
    # EU
    {
        "name": "European Commission Press",
        "url": "https://ec.europa.eu/commission/presscorner/api/rss",
        "description": "European Commission press releases",
    },
    {
        "name": "European Central Bank",
        "url": "https://www.ecb.europa.eu/rss/press.html",
        "description": "ECB press releases and monetary policy",
    },
    # UK
    {
        "name": "UK Government News",
        "url": "https://www.gov.uk/government/announcements.atom",
        "description": "UK government announcements",
    },
    {
        "name": "Bank of England",
        "url": "https://www.bankofengland.co.uk/rss/news",
        "description": "Bank of England news and publications",
    },
    # China
    {
        "name": "China State Council",
        "url": "http://english.www.gov.cn/rss/news.xml",
        "description": "Chinese government official news",
    },
    # Japan
    {
        "name": "Bank of Japan",
        "url": "https://www.boj.or.jp/en/rss/whatsnew.xml",
        "description": "Bank of Japan announcements",
    },
    # International Organizations
    {
        "name": "IMF News",
        "url": "https://www.imf.org/en/News/rss",
        "description": "International Monetary Fund news",
    },
    {
        "name": "World Bank News",
        "url": "https://www.worldbank.org/en/news/rss.xml",
        "description": "World Bank news and updates",
    },
]

# =============================================================================
# LLM Filter Prompt
# =============================================================================

FILTER_SYSTEM_PROMPT = """You are an investment analyst assistant focused on identifying structural changes in the world.

Your task is to analyze news items from NON-US governments and international organizations to identify STRUCTURAL CHANGES that could affect long-term investment decisions.

Structural changes include:
- Major regulatory shifts in key markets (EU, China, Japan, UK)
- Trade agreements or trade barriers that affect global supply chains
- Monetary policy changes from major central banks
- Industrial policy or government investment initiatives
- Cross-border regulatory coordination or conflicts

NOT structural changes (ignore these):
- Routine diplomatic meetings without concrete outcomes
- Minor regulatory updates or clarifications
- Personnel announcements
- Ceremonial or cultural events
- Short-term economic indicators

For each news item, respond in JSON format:
{
    "is_structural": true/false,
    "category": "legislation|regulation|policy|executive_order|trade|technology|other",
    "reason": "brief explanation why this is/isn't structural"
}
"""

# =============================================================================
# Configuration
# =============================================================================

OTHER_GOV_CONFIG = WatcherConfig(
    name="other_gov_watcher",
    watcher_type=WatcherType.OTHER_GOV,
    feeds=OTHER_GOV_FEEDS,
    filter_prompt=FILTER_SYSTEM_PROMPT,
    data_dir=DATA_DIR / "other_gov",
)

# =============================================================================
# Agent Runner
# =============================================================================


async def run_agent() -> WatcherState:
    """Run the other government news watcher."""
    watcher = NewsWatcher(OTHER_GOV_CONFIG)
    return await watcher.run()
