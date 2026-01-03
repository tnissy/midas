"""US Government News Watcher using LangGraph."""

import json
from datetime import datetime
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from midas.config import extract_llm_text, DATA_DIR, GEMINI_API_KEY, LLM_MODEL
from midas.models import NewsCategory, NewsItem
from midas.tools.rss_fetcher import fetch_feeds

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

# Paths for this watcher
WATCHER_DATA_DIR = DATA_DIR / "us_gov"
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
# Agent Nodes
# =============================================================================


async def fetch_news(state: AgentState) -> AgentState:
    """Fetch news from US Government RSS sources and save raw data."""
    print("Fetching US Government news...")
    try:
        items = await fetch_feeds(US_GOV_FEEDS, CACHE_FILE)
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
            # Prepare prompt
            content = f"Title: {item.title}\n\nContent: {item.content[:1000]}"

            messages = [
                SystemMessage(content=FILTER_SYSTEM_PROMPT),
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

    # Ensure directory exists
    WATCHER_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = WATCHER_DATA_DIR / f"news_{timestamp}.json"

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
    print(f"Saved to: {filepath}")
    return state


# =============================================================================
# Agent Graph
# =============================================================================


def create_agent() -> StateGraph:
    """Create the US Government news watcher agent graph."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("fetch", fetch_news)
    workflow.add_node("filter", filter_news)
    workflow.add_node("save", save_results)

    # Define edges
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
