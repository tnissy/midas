"""Farseer - Future outlook and social change detector.

This agent scans news sources for future predictions and structural changes,
then analyzes them to identify investment themes and opportunities.

Features:
1. Curated source list management (updated semi-annually)
2. Google News search for outlook/prediction articles
3. Integration with existing news watchers' results
4. Single LLM call for efficient analysis
5. Output: Social changes, investment themes, beneficiaries
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import TypedDict
from urllib.parse import quote

import feedparser
import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from midas.config import DATA_DIR, GEMINI_API_KEY, LLM_MODEL, extract_llm_text

# =============================================================================
# Constants
# =============================================================================

FARSEER_DATA_DIR = DATA_DIR / "farseer"
SOURCES_FILE = FARSEER_DATA_DIR / "sources.json"
ARTICLES_DIR = FARSEER_DATA_DIR / "articles"
REPORTS_DIR = FARSEER_DATA_DIR / "reports"


# =============================================================================
# Data Models
# =============================================================================


class Article(TypedDict):
    """A collected article."""

    title: str
    url: str
    source: str
    published: str
    snippet: str
    origin: str  # "google_news" or "news_watcher"


class SocialChange(TypedDict):
    """A detected social change."""

    title: str
    category: str  # technology, regulation, behavior, geopolitical, economic
    description: str
    time_horizon: str  # near (1-2y), medium (3-5y), long (5-10y)
    confidence: str  # low, medium, high
    implications: list[str]


class InvestmentTheme(TypedDict):
    """An investment theme derived from social changes."""

    title: str
    thesis: str
    related_changes: list[str]
    beneficiaries: list[dict]  # {name, symbol, reason}
    risks: list[str]
    conviction: str  # low, medium, high


class FarseerReport(TypedDict):
    """Complete Farseer analysis report."""

    generated_at: str
    year: int
    articles_analyzed: int
    executive_summary: str
    social_changes: list[SocialChange]
    investment_themes: list[InvestmentTheme]
    key_observations: list[str]
    action_items: list[str]


class FarseerState(TypedDict):
    """State for the Farseer agent."""

    year: int
    include_watchers: bool
    articles: list[Article]
    report: FarseerReport | None
    report_path: str | None
    error: str | None


# =============================================================================
# Source Management
# =============================================================================


def load_sources() -> dict:
    """Load sources configuration."""
    if SOURCES_FILE.exists():
        with open(SOURCES_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"categories": {}, "search_keywords": {}}


def save_sources(sources: dict) -> None:
    """Save sources configuration."""
    FARSEER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SOURCES_FILE, "w", encoding="utf-8") as f:
        json.dump(sources, f, ensure_ascii=False, indent=2)


# =============================================================================
# Article Collection
# =============================================================================


async def search_google_news(
    query: str,
    site: str | None = None,
    max_results: int = 20,
) -> list[Article]:
    """Search Google News RSS for articles."""
    search_query = f"site:{site} {query}" if site else query
    encoded_query = quote(search_query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

    articles: list[Article] = []

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()

        parsed = feedparser.parse(response.text)

        for entry in parsed.entries[:max_results]:
            title = entry.get("title", "")
            source = "Unknown"
            if " - " in title:
                parts = title.rsplit(" - ", 1)
                title = parts[0].strip()
                source = parts[1].strip()

            snippet = entry.get("summary", "")
            import re
            snippet = re.sub(r"<[^>]+>", "", snippet)[:500]

            articles.append(
                Article(
                    title=title,
                    url=entry.get("link", ""),
                    source=source,
                    published=entry.get("published", ""),
                    snippet=snippet,
                    origin="google_news",
                )
            )

    except Exception as e:
        print(f"  Error searching '{search_query[:50]}': {e}")

    return articles


def load_watcher_news(days_back: int = 30) -> list[Article]:
    """Load recent news from news watchers."""
    articles: list[Article] = []
    news_dir = DATA_DIR / "news"

    if not news_dir.exists():
        return articles

    cutoff = datetime.now() - timedelta(days=days_back)

    for filepath in news_dir.glob("*.json"):
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)

            for item in data.get("filtered_items", []):
                try:
                    published = item.get("published", "")
                    if isinstance(published, str) and published:
                        pub_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
                        if pub_dt.tzinfo:
                            pub_dt = pub_dt.replace(tzinfo=None)
                        if pub_dt < cutoff:
                            continue

                    articles.append(
                        Article(
                            title=item.get("title", ""),
                            url=item.get("url", ""),
                            source=item.get("source", ""),
                            published=published,
                            snippet=item.get("content", "")[:500] or item.get("relevance_reason", ""),
                            origin="news_watcher",
                        )
                    )
                except Exception:
                    continue

        except Exception:
            continue

    return articles


# =============================================================================
# LLM Analysis (Single Call)
# =============================================================================

ANALYSIS_PROMPT = """You are Farseer, an expert analyst identifying structural changes and investment opportunities from news and outlook articles.

Analyze these articles and provide a COMPREHENSIVE analysis in a SINGLE response:

1. **Social Changes**: Identify significant structural shifts (technology, regulation, behavior, geopolitical, economic)
2. **Investment Themes**: Synthesize changes into actionable investment themes
3. **Beneficiaries**: Identify companies/sectors that benefit from each theme
4. **Executive Summary**: 2-3 paragraphs in Japanese summarizing the key insights

For social changes, assess:
- Time horizon: near (1-2 years), medium (3-5 years), long (5-10 years)
- Confidence: low/medium/high

For investment themes, provide:
- Clear thesis explaining why this creates value
- Specific beneficiaries with stock symbols when possible
- Key risks

Respond in JSON:
{{
    "executive_summary": "日本語で2-3段落のサマリー",
    "social_changes": [
        {{
            "title": "Change title",
            "category": "technology|regulation|behavior|geopolitical|economic",
            "description": "Detailed description",
            "time_horizon": "near|medium|long",
            "confidence": "low|medium|high",
            "implications": ["Implication 1", "Implication 2"]
        }}
    ],
    "investment_themes": [
        {{
            "title": "Theme title",
            "thesis": "Why this creates investment value",
            "related_changes": ["Change title 1", "Change title 2"],
            "beneficiaries": [
                {{"name": "Company", "symbol": "TICKER", "reason": "Why they benefit"}}
            ],
            "risks": ["Risk 1", "Risk 2"],
            "conviction": "low|medium|high"
        }}
    ],
    "key_observations": ["Observation 1", "Observation 2"],
    "action_items": ["Action 1", "Action 2"]
}}
"""

EXPAND_SOURCES_PROMPT = """You are helping curate authoritative sources for structural change analysis.

Current sources:
{current_sources}

Suggest 5-10 additional sources focusing on:
1. Think tanks not yet included
2. Notable intellectuals/thought leaders
3. Industry-specific sources
4. Regional sources (Asia, Europe, emerging markets)
5. Academic institutions

Respond in JSON:
{{
    "suggestions": [
        {{
            "name": "Source name",
            "url": "https://...",
            "category": "consulting_research|tech_research|think_tanks|vc_investors|intellectuals|newspapers_magazines|japan|other",
            "focus": ["topic1", "topic2"],
            "language": "en|ja|other",
            "reason": "Why valuable"
        }}
    ]
}}
"""


# =============================================================================
# Agent Nodes
# =============================================================================


async def collect_articles(state: FarseerState) -> FarseerState:
    """Collect articles from Google News and optionally news watchers."""
    year = state.get("year", datetime.now().year)
    include_watchers = state.get("include_watchers", True)

    print(f"Collecting articles for {year} outlook...")
    all_articles: list[Article] = []

    # Google News search
    sources = load_sources()
    keywords = sources.get("search_keywords", {})
    outlook_queries = keywords.get("outlook", [f"{year} outlook", f"{year} predictions", f"{year} forecast"])

    for query in outlook_queries[:3]:
        print(f"  Searching: '{query}'...")
        articles = await search_google_news(query, max_results=30)
        all_articles.extend(articles)
        print(f"    Found {len(articles)} articles")

    # Site-specific searches
    key_sites = ["mckinsey.com", "bcg.com", "economist.com", "weforum.org", "gartner.com"]
    for site in key_sites:
        query = f"{year} outlook OR {year} predictions OR trends"
        print(f"  Searching site:{site}...")
        articles = await search_google_news(query, site=site, max_results=10)
        all_articles.extend(articles)
        print(f"    Found {len(articles)} articles")

    # Load news watcher results
    if include_watchers:
        print("  Loading news watcher results...")
        watcher_articles = load_watcher_news(days_back=30)
        all_articles.extend(watcher_articles)
        print(f"    Found {len(watcher_articles)} articles from watchers")

    # Deduplicate
    seen_urls: set[str] = set()
    unique_articles: list[Article] = []
    for article in all_articles:
        if article["url"] not in seen_urls:
            seen_urls.add(article["url"])
            unique_articles.append(article)

    state["articles"] = unique_articles
    print(f"Total: {len(unique_articles)} unique articles")

    # Save raw articles
    if unique_articles:
        ARTICLES_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = ARTICLES_DIR / f"articles_{year}_{timestamp}.json"

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                {"collected_at": datetime.now().isoformat(), "year": year, "total": len(unique_articles), "articles": unique_articles},
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Articles saved: {filepath}")

    return state


async def analyze_articles(state: FarseerState) -> FarseerState:
    """Analyze all articles in a single LLM call."""
    if not state.get("articles"):
        state["report"] = None
        state["error"] = "No articles to analyze"
        return state

    print("Analyzing articles (single LLM call)...")

    if not GEMINI_API_KEY:
        state["error"] = "No API key"
        return state

    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GEMINI_API_KEY)

    try:
        # Prepare context (limit to 60 articles for token efficiency)
        articles_text = []
        for i, article in enumerate(state["articles"][:60]):
            articles_text.append(
                f"[{i}] {article['title']}\n"
                f"Source: {article['source']} | Origin: {article['origin']}\n"
                f"Snippet: {article['snippet'][:250]}"
            )

        year = state.get("year", datetime.now().year)
        prompt = f"ARTICLES FOR {year} ANALYSIS:\n\n" + "\n\n".join(articles_text)

        messages = [
            SystemMessage(content=ANALYSIS_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = await llm.ainvoke(messages)
        result_text = extract_llm_text(response.content)

        if isinstance(result_text, str):
            start = result_text.find("{")
            end = result_text.rfind("}") + 1
            if start != -1 and end > start:
                result = json.loads(result_text[start:end])

                # Build report
                report: FarseerReport = {
                    "generated_at": datetime.now().isoformat(),
                    "year": year,
                    "articles_analyzed": len(state["articles"]),
                    "executive_summary": result.get("executive_summary", ""),
                    "social_changes": result.get("social_changes", []),
                    "investment_themes": result.get("investment_themes", []),
                    "key_observations": result.get("key_observations", []),
                    "action_items": result.get("action_items", []),
                }

                state["report"] = report

                # Print progress
                print(f"  Social changes: {len(report['social_changes'])}")
                print(f"  Investment themes: {len(report['investment_themes'])}")

                for change in report["social_changes"]:
                    print(f"    [{change.get('category', '?')}] {change.get('title', '')}")

    except Exception as e:
        print(f"Error: {e}")
        state["error"] = str(e)

    return state


def save_report(state: FarseerState) -> FarseerState:
    """Save the report to disk."""
    if not state.get("report"):
        return state

    print("Saving report...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    year = state.get("year", datetime.now().year)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = REPORTS_DIR / f"farseer_{year}_{timestamp}.json"

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(state["report"], f, ensure_ascii=False, indent=2)

    state["report_path"] = str(filepath)
    print(f"Report saved: {filepath}")

    return state


# =============================================================================
# Agent Graph
# =============================================================================


def create_agent() -> StateGraph:
    """Create the Farseer agent graph."""
    workflow = StateGraph(FarseerState)

    workflow.add_node("collect", collect_articles)
    workflow.add_node("analyze", analyze_articles)
    workflow.add_node("save", save_report)

    workflow.set_entry_point("collect")
    workflow.add_edge("collect", "analyze")
    workflow.add_edge("analyze", "save")
    workflow.add_edge("save", END)

    return workflow.compile()


async def run_scan(year: int | None = None, include_watchers: bool = True) -> FarseerState:
    """Run Farseer scan.

    Args:
        year: Target year (default: current year)
        include_watchers: Include news watcher results

    Returns:
        FarseerState with report
    """
    agent = create_agent()

    initial_state: FarseerState = {
        "year": year or datetime.now().year,
        "include_watchers": include_watchers,
        "articles": [],
        "report": None,
        "report_path": None,
        "error": None,
    }

    return await agent.ainvoke(initial_state)


async def expand_sources() -> list[dict]:
    """Get AI suggestions for new sources."""
    if not GEMINI_API_KEY:
        return []

    sources = load_sources()
    current_text = []
    for cat_name, category in sources.get("categories", {}).items():
        current_text.append(f"\n## {category.get('description', cat_name)}")
        for source in category.get("sources", []):
            current_text.append(f"- {source['name']}")

    prompt = EXPAND_SOURCES_PROMPT.format(current_sources="\n".join(current_text))

    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GEMINI_API_KEY)

    try:
        messages = [
            SystemMessage(content="You are an expert at curating information sources."),
            HumanMessage(content=prompt),
        ]

        response = await llm.ainvoke(messages)
        result_text = extract_llm_text(response.content)

        if isinstance(result_text, str):
            start = result_text.find("{")
            end = result_text.rfind("}") + 1
            if start != -1 and end > start:
                result = json.loads(result_text[start:end])
                return result.get("suggestions", [])

    except Exception as e:
        print(f"Error: {e}")

    return []


# =============================================================================
# Report Formatting
# =============================================================================


def format_report(state: FarseerState) -> str:
    """Format report for display."""
    if state.get("error"):
        return f"Error: {state['error']}"

    report = state.get("report")
    if not report:
        return "No report generated."

    lines = [
        "",
        "=" * 70,
        f"FARSEER REPORT - {report.get('year', 'N/A')}",
        f"Articles analyzed: {report.get('articles_analyzed', 0)}",
        "=" * 70,
        "",
        "EXECUTIVE SUMMARY",
        "-" * 40,
        report.get("executive_summary", ""),
        "",
    ]

    # Social Changes
    changes = report.get("social_changes", [])
    if changes:
        lines.append("=" * 70)
        lines.append(f"SOCIAL CHANGES ({len(changes)})")
        lines.append("=" * 70)

        category_emoji = {
            "technology": "🔬",
            "regulation": "📜",
            "behavior": "👥",
            "geopolitical": "🌍",
            "economic": "📈",
        }

        for i, change in enumerate(changes, 1):
            emoji = category_emoji.get(change.get("category", ""), "📌")
            horizon = {"near": "1-2年", "medium": "3-5年", "long": "5-10年"}.get(
                change.get("time_horizon", ""), "?"
            )
            conf = {"high": "🔥", "medium": "📊", "low": "❓"}.get(
                change.get("confidence", ""), ""
            )

            lines.append(f"\n{i}. {emoji} {conf} {change.get('title', '')}")
            lines.append(f"   Category: {change.get('category', '')} | Horizon: {horizon}")
            lines.append(f"   {change.get('description', '')[:200]}...")

            implications = change.get("implications", [])
            if implications:
                lines.append("   Implications:")
                for imp in implications[:3]:
                    lines.append(f"     • {imp}")

    # Investment Themes
    themes = report.get("investment_themes", [])
    if themes:
        lines.append("")
        lines.append("=" * 70)
        lines.append(f"INVESTMENT THEMES ({len(themes)})")
        lines.append("=" * 70)

        for i, theme in enumerate(themes, 1):
            conv = {"high": "🔥", "medium": "📊", "low": "❓"}.get(
                theme.get("conviction", ""), ""
            )

            lines.append(f"\n{i}. {conv} {theme.get('title', '')}")
            lines.append(f"   Thesis: {theme.get('thesis', '')}")

            beneficiaries = theme.get("beneficiaries", [])
            if beneficiaries:
                lines.append("   Beneficiaries:")
                for b in beneficiaries[:5]:
                    symbol = f" ({b.get('symbol', '')})" if b.get("symbol") else ""
                    lines.append(f"     • {b.get('name', '')}{symbol}: {b.get('reason', '')}")

            risks = theme.get("risks", [])
            if risks:
                lines.append(f"   Risks: {', '.join(risks[:3])}")

    # Key Observations
    observations = report.get("key_observations", [])
    if observations:
        lines.append("")
        lines.append("-" * 40)
        lines.append("KEY OBSERVATIONS:")
        for obs in observations:
            lines.append(f"  • {obs}")

    # Action Items
    actions = report.get("action_items", [])
    if actions:
        lines.append("")
        lines.append("-" * 40)
        lines.append("ACTION ITEMS:")
        for action in actions:
            lines.append(f"  → {action}")

    if state.get("report_path"):
        lines.append("")
        lines.append(f"Full report: {state['report_path']}")

    return "\n".join(lines)
