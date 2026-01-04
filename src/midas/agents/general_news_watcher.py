"""General News Watcher - Configuration and runner."""

from midas.agents.news_watcher_base import NewsWatcher, WatcherConfig, WatcherState
from midas.config import DATA_DIR
from midas.models import WatcherType

# =============================================================================
# Feed Definitions (General Business/Financial Sources)
# =============================================================================

GENERAL_FEEDS: list[dict] = [
    # Financial News
    {
        "name": "Yahoo Finance",
        "url": "https://finance.yahoo.com/news/rssindex",
        "description": "Yahoo Finance top stories",
    },
    {
        "name": "MarketWatch",
        "url": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
        "description": "MarketWatch top stories",
    },
    {
        "name": "Bloomberg Markets",
        "url": "https://feeds.bloomberg.com/markets/news.rss",
        "description": "Bloomberg market news",
    },
    {
        "name": "CNBC Top News",
        "url": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "description": "CNBC top business news",
    },
    # Business/Industry
    {
        "name": "Reuters Business",
        "url": "https://www.reuters.com/business/rss",
        "description": "Reuters business news",
    },
    {
        "name": "Financial Times",
        "url": "https://www.ft.com/rss/home",
        "description": "Financial Times headlines",
    },
    # Investment Focused
    {
        "name": "Investing.com",
        "url": "https://www.investing.com/rss/news.rss",
        "description": "Investing.com market news",
    },
    {
        "name": "Seeking Alpha",
        "url": "https://seekingalpha.com/feed.xml",
        "description": "Investment analysis and news",
    },
]

# =============================================================================
# LLM Filter Prompt
# =============================================================================

FILTER_SYSTEM_PROMPT = """You are an investment analyst assistant focused on identifying structural changes in the world.

Your task is to analyze general business/financial news and identify STRUCTURAL CHANGES that could affect long-term investment decisions.

Structural changes include:
- Industry consolidation or disruption (major M&A, new entrants reshaping markets)
- Supply chain restructuring (reshoring, new logistics models)
- Consumer behavior shifts (demographic changes, new consumption patterns)
- Energy transition milestones (renewables crossing cost thresholds)
- Infrastructure investments that enable new business models

NOT structural changes (ignore these):
- Daily market movements and stock price changes
- Quarterly earnings reports
- Analyst upgrades/downgrades
- Short-term economic indicators
- CEO interviews without concrete announcements
- Market sentiment or speculation

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

GENERAL_CONFIG = WatcherConfig(
    name="general_news_watcher",
    watcher_type=WatcherType.GENERAL_NEWS,
    feeds=GENERAL_FEEDS,
    filter_prompt=FILTER_SYSTEM_PROMPT,
    data_dir=DATA_DIR / "general",
)

# =============================================================================
# Agent Runner
# =============================================================================


async def run_agent() -> WatcherState:
    """Run the general news watcher."""
    watcher = NewsWatcher(GENERAL_CONFIG)
    return await watcher.run()
