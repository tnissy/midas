"""US Government Watcher - Configuration and runner."""

from midas.agents.news_watcher_base import NewsWatcher, WatcherConfig, WatcherState
from midas.config import DATA_DIR
from midas.models import WatcherType

# =============================================================================
# Feed Definitions (US Government Sources)
# =============================================================================

US_GOV_FEEDS: list[dict] = [
    {
        "name": "White House",
        "url": "https://www.whitehouse.gov/feed/",
        "description": "Official White House announcements and statements",
    },
    {
        "name": "Congress - Bills",
        "url": "https://www.congress.gov/rss/bill-status-all.xml",
        "description": "All bill status updates from Congress",
    },
    {
        "name": "Federal Register",
        "url": "https://www.federalregister.gov/documents/current.rss",
        "description": "Current federal regulations and notices",
    },
    {
        "name": "SEC News",
        "url": "https://www.sec.gov/news/pressreleases.rss",
        "description": "SEC press releases",
    },
    {
        "name": "US Trade Representative",
        "url": "https://ustr.gov/about-us/policy-offices/press-office/press-releases/rss",
        "description": "US trade policy announcements",
    },
]

# =============================================================================
# LLM Filter Prompt
# =============================================================================

FILTER_SYSTEM_PROMPT = """You are an investment analyst assistant focused on identifying structural changes in the world.

Your task is to analyze news items and determine if they relate to STRUCTURAL CHANGES that could affect long-term investment decisions.

Structural changes include:
- Major technology shifts (new technologies becoming practical, old ones becoming obsolete)
- Regulatory/policy changes that alter market dynamics
- Trade policy changes affecting supply chains or market access
- Government investment or industrial policy that redirects capital flows
- Changes to competitive landscapes (monopoly formation/breakup, entry barriers)

NOT structural changes (ignore these):
- Short-term market movements
- Quarterly earnings
- Personnel changes (unless CEO of major company)
- Routine regulatory filings
- Sentiment or opinion pieces

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

US_GOV_CONFIG = WatcherConfig(
    name="us_gov_watcher",
    watcher_type=WatcherType.US_GOV,
    feeds=US_GOV_FEEDS,
    filter_prompt=FILTER_SYSTEM_PROMPT,
    data_dir=DATA_DIR / "us_gov",
)

# =============================================================================
# Agent Runner
# =============================================================================


async def run_agent() -> WatcherState:
    """Run the US Government news watcher."""
    watcher = NewsWatcher(US_GOV_CONFIG)
    return await watcher.run()
