"""Technology News Watcher - Configuration and runner."""

from midas.agents.news_watcher_base import NewsWatcher, WatcherConfig, WatcherState
from midas.config import DATA_DIR
from midas.models import WatcherType

# =============================================================================
# Feed Definitions (Technology Sources)
# =============================================================================

TECH_FEEDS: list[dict] = [
    # Major Tech News
    {
        "name": "Ars Technica",
        "url": "https://feeds.arstechnica.com/arstechnica/technology-lab",
        "description": "Technology news and analysis",
    },
    {
        "name": "TechCrunch",
        "url": "https://techcrunch.com/feed/",
        "description": "Startup and technology news",
    },
    {
        "name": "The Verge",
        "url": "https://www.theverge.com/rss/index.xml",
        "description": "Technology, science, art, and culture",
    },
    {
        "name": "Wired",
        "url": "https://www.wired.com/feed/rss",
        "description": "Technology and culture",
    },
    # AI/ML Focused
    {
        "name": "MIT Technology Review",
        "url": "https://www.technologyreview.com/feed/",
        "description": "Emerging technologies and their impact",
    },
    {
        "name": "OpenAI Blog",
        "url": "https://openai.com/blog/rss/",
        "description": "OpenAI research and announcements",
    },
    # Developer/Technical
    {
        "name": "Hacker News",
        "url": "https://hnrss.org/frontpage",
        "description": "Tech community curated news",
    },
    {
        "name": "IEEE Spectrum",
        "url": "https://spectrum.ieee.org/feeds/feed.rss",
        "description": "Engineering and technology news",
    },
]

# =============================================================================
# LLM Filter Prompt
# =============================================================================

FILTER_SYSTEM_PROMPT = """You are an investment analyst assistant focused on identifying technological structural changes.

Your task is to analyze technology news and identify STRUCTURAL CHANGES that could affect long-term investment decisions.

Structural changes include:
- New technologies becoming practical/commercial (AI breakthroughs, quantum computing milestones)
- Technology standards being adopted or abandoned
- Bottleneck shifts (e.g., GPU shortage → new chip architectures)
- Platform shifts (e.g., mobile → AR/VR, web2 → web3)
- Open source projects threatening commercial products or vice versa
- Major acquisitions that reshape competitive landscape

NOT structural changes (ignore these):
- Product updates or version releases
- Funding announcements (unless mega-rounds signaling market shift)
- Hiring news
- Conference announcements
- Opinion pieces without concrete developments
- Minor feature updates

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

TECH_CONFIG = WatcherConfig(
    name="tech_news_watcher",
    watcher_type=WatcherType.TECH_NEWS,
    feeds=TECH_FEEDS,
    filter_prompt=FILTER_SYSTEM_PROMPT,
    data_dir=DATA_DIR / "tech",
)

# =============================================================================
# Agent Runner
# =============================================================================


async def run_agent() -> WatcherState:
    """Run the technology news watcher."""
    watcher = NewsWatcher(TECH_CONFIG)
    return await watcher.run()
