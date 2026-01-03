"""Other Government News Watcher (non-US) using LangGraph."""

import json
from datetime import datetime
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from midas.config import DATA_DIR, GEMINI_API_KEY, LLM_MODEL
from midas.models import NewsCategory, NewsItem
from midas.tools.rss_fetcher import fetch_feeds

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

# Paths for this watcher
WATCHER_DATA_DIR = DATA_DIR / "other_gov"
CACHE_FILE = WATCHER_DATA_DIR / "fetched_ids.json"

# =============================================================================
# Agent State
# =============================================================================


class AgentState(TypedDict):
    """State for the news collection agent."""

    raw_items: list[NewsItem]
    filtered_items: list[NewsItem]
    saved_path: str | None
    error: str | None


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
# Agent Nodes
# =============================================================================


async def fetch_news(state: AgentState) -> AgentState:
    """Fetch news from non-US government RSS sources and save raw data."""
    print("Fetching Other Government news...")
    try:
        items = await fetch_feeds(OTHER_GOV_FEEDS, CACHE_FILE)
        state["raw_items"] = items
        print(f"Total: {len(items)} new items fetched")

        # Save raw items immediately (before LLM filtering)
        if items:
            WATCHER_DATA_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_filepath = WATCHER_DATA_DIR / f"raw_{timestamp}.json"

            raw_data = {
                "fetched_at": datetime.now().isoformat(),
                "total": len(items),
                "items": [item.model_dump(mode="json") for item in items],
            }

            with open(raw_filepath, "w", encoding="utf-8") as f:
                json.dump(raw_data, f, ensure_ascii=False, indent=2)

            print(f"Raw data saved to: {raw_filepath}")

    except Exception as e:
        state["error"] = f"Failed to fetch news: {e}"
        state["raw_items"] = []
    return state


async def filter_news(state: AgentState) -> AgentState:
    """Filter news using LLM to identify structural changes."""
    if state.get("error") or not state.get("raw_items"):
        state["filtered_items"] = []
        return state

    print("Filtering news with LLM...")

    if not GEMINI_API_KEY:
        print("Warning: No API key, skipping LLM filtering")
        state["filtered_items"] = state["raw_items"]
        return state

    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GEMINI_API_KEY)
    filtered: list[NewsItem] = []

    for item in state["raw_items"]:
        try:
            content = f"Title: {item.title}\n\nContent: {item.content[:1000]}"

            messages = [
                SystemMessage(content=FILTER_SYSTEM_PROMPT),
                HumanMessage(content=content),
            ]

            response = await llm.ainvoke(messages)
            result_text = response.content

            if isinstance(result_text, str):
                start = result_text.find("{")
                end = result_text.rfind("}") + 1
                if start != -1 and end > start:
                    result = json.loads(result_text[start:end])

                    if result.get("is_structural"):
                        item.is_structural = True
                        item.relevance_reason = result.get("reason", "")

                        cat = result.get("category", "other")
                        try:
                            item.category = NewsCategory(cat)
                        except ValueError:
                            item.category = NewsCategory.OTHER

                        filtered.append(item)
                        print(f"  [+] {item.title[:60]}...")

        except Exception as e:
            print(f"  Error processing item: {e}")
            continue

    state["filtered_items"] = filtered
    print(f"Filtered: {len(filtered)} structural news items")
    return state


def save_results(state: AgentState) -> AgentState:
    """Save filtered results to JSON file."""
    if not state.get("filtered_items"):
        state["saved_path"] = None
        return state

    print("Saving results...")

    WATCHER_DATA_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = WATCHER_DATA_DIR / f"news_{timestamp}.json"

    data = {
        "fetched_at": datetime.now().isoformat(),
        "total_raw": len(state.get("raw_items", [])),
        "total_filtered": len(state["filtered_items"]),
        "items": [item.model_dump(mode="json") for item in state["filtered_items"]],
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    state["saved_path"] = str(filepath)
    print(f"Saved to: {filepath}")
    return state


# =============================================================================
# Agent Graph
# =============================================================================


def create_agent() -> StateGraph:
    """Create the Other Government news watcher agent graph."""
    workflow = StateGraph(AgentState)

    workflow.add_node("fetch", fetch_news)
    workflow.add_node("filter", filter_news)
    workflow.add_node("save", save_results)

    workflow.set_entry_point("fetch")
    workflow.add_edge("fetch", "filter")
    workflow.add_edge("filter", "save")
    workflow.add_edge("save", END)

    return workflow.compile()


async def run_agent() -> AgentState:
    """Run the news watcher agent."""
    agent = create_agent()

    initial_state: AgentState = {
        "raw_items": [],
        "filtered_items": [],
        "saved_path": None,
        "error": None,
    }

    result = await agent.ainvoke(initial_state)
    return result
