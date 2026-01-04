"""Foresight Manager - Future prediction management agent.

Manages foresights (future predictions) with two modes:
1. Full mode: Uses prediction_monitor to generate initial/annual foresights
2. Incremental mode: Uses news watchers to adjust existing foresights

Usage:
    midas foresight scan              # Auto-detect mode
    midas foresight scan --force-full # Force full update
    midas foresight list              # List all foresights
    midas foresight report            # Generate Google Slides report
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from midas.agents import prediction_monitor
from midas.config import DATA_DIR, GEMINI_API_KEY, LLM_MODEL, extract_llm_text
from midas.models import Foresight, ForesightSource, NewsItem
from midas.tools.report_generator import save_report, generate_foresight_report

# =============================================================================
# Constants
# =============================================================================

FORESIGHT_DATA_DIR = DATA_DIR / "foresights"
FORESIGHTS_FILE = FORESIGHT_DATA_DIR / "foresights.json"

# Watcher directories
WATCHER_DIRS = [
    DATA_DIR / "us_gov",
    DATA_DIR / "tech",
    DATA_DIR / "general",
    DATA_DIR / "other_gov",
]


# =============================================================================
# Data Models
# =============================================================================


class ForesightStore(TypedDict):
    """Stored foresights data structure."""

    updated_at: str
    last_full_update: str | None
    foresights: list[dict]


class ForesightState(TypedDict):
    """State for the foresight manager agent."""

    mode: str  # "full" | "incremental"
    force_full: bool
    existing_foresights: list[Foresight]
    news_items: list[NewsItem]
    prediction_report: dict | None
    updated_foresights: list[Foresight]
    saved_path: str | None
    error: str | None


# =============================================================================
# Storage Functions
# =============================================================================


def load_foresights() -> list[Foresight]:
    """Load existing foresights from disk."""
    if not FORESIGHTS_FILE.exists():
        return []

    try:
        with open(FORESIGHTS_FILE, encoding="utf-8") as f:
            data: ForesightStore = json.load(f)
        return [Foresight(**fs) for fs in data.get("foresights", [])]
    except Exception as e:
        print(f"Error loading foresights: {e}")
        return []


def get_last_full_update() -> datetime | None:
    """Get the datetime of last full update."""
    if not FORESIGHTS_FILE.exists():
        return None

    try:
        with open(FORESIGHTS_FILE, encoding="utf-8") as f:
            data: ForesightStore = json.load(f)
        last_full = data.get("last_full_update")
        if last_full:
            return datetime.fromisoformat(last_full)
    except Exception:
        pass
    return None


def save_foresights(
    foresights: list[Foresight],
    is_full_update: bool = False,
) -> Path:
    """Save foresights to disk."""
    FORESIGHT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing data to preserve last_full_update if incremental
    existing_last_full = None
    if FORESIGHTS_FILE.exists() and not is_full_update:
        try:
            with open(FORESIGHTS_FILE, encoding="utf-8") as f:
                existing_data = json.load(f)
                existing_last_full = existing_data.get("last_full_update")
        except Exception:
            pass

    data: ForesightStore = {
        "updated_at": datetime.now().isoformat(),
        "last_full_update": (
            datetime.now().isoformat() if is_full_update else existing_last_full
        ),
        "foresights": [fs.model_dump(mode="json") for fs in foresights],
    }

    with open(FORESIGHTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return FORESIGHTS_FILE


# =============================================================================
# News Loading
# =============================================================================


def load_watcher_news(days_back: int = 7) -> list[NewsItem]:
    """Load recent news from all watchers."""
    news_items: list[NewsItem] = []
    cutoff = datetime.now() - timedelta(days=days_back)

    for watcher_dir in WATCHER_DIRS:
        if not watcher_dir.exists():
            continue

        # Look for news_*.json files
        for filepath in watcher_dir.glob("news_*.json"):
            try:
                with open(filepath, encoding="utf-8") as f:
                    data = json.load(f)

                # Check file date from filename
                filename = filepath.stem  # news_YYYYMMDD_HHMMSS
                try:
                    date_str = filename.split("_")[1]
                    file_date = datetime.strptime(date_str, "%Y%m%d")
                    if file_date < cutoff:
                        continue
                except (IndexError, ValueError):
                    pass

                # Load filtered items (structural news only)
                for item_data in data.get("items", []):
                    if item_data.get("is_structural", False):
                        try:
                            news_items.append(NewsItem(**item_data))
                        except Exception:
                            continue

            except Exception as e:
                print(f"  Error loading {filepath}: {e}")
                continue

    return news_items


# =============================================================================
# Mode Determination
# =============================================================================


def determine_mode(force_full: bool = False) -> str:
    """Determine which mode to use for foresight generation."""
    if force_full:
        return "full"

    foresights = load_foresights()
    if len(foresights) == 0:
        print("No existing foresights. Using full mode.")
        return "full"

    # January = annual update
    now = datetime.now()
    if now.month == 1:
        last_full = get_last_full_update()
        if last_full is None or last_full.year < now.year:
            print(f"Annual update (January {now.year}). Using full mode.")
            return "full"

    return "incremental"


# =============================================================================
# LLM Prompts
# =============================================================================

FULL_MODE_PROMPT = """You are a strategic investment analyst consolidating many future predictions into key investment foresights.

You will receive 100+ individual predictions with their index numbers. Your task is to:
1. Identify the major themes and patterns across all predictions
2. Consolidate related predictions into 5 investment-focused foresights
3. Each foresight should represent a major structural change theme

Output exactly 5 foresights that capture the most significant investment opportunities.

Respond in JSON:
{
    "foresights": [
        {
            "title": "Clear, actionable title (max 30 words)",
            "description": "Comprehensive description in Japanese (300-500 characters). Include: what structural change is happening, why it matters for investors, key beneficiaries/sectors, timeline (1-2年/3-5年/5-10年), and investment implications.",
            "source_indices": [1, 5, 12, 23, 45]
        }
    ]
}

Guidelines:
- Write descriptions in Japanese
- Each foresight should consolidate 10-30 related predictions
- source_indices: list the prediction numbers (1-based) that support this foresight
- Focus on actionable investment themes, not general trends
- Prioritize structural changes over short-term events
- Include specific sectors, technologies, or companies when relevant
"""

INCREMENTAL_MODE_PROMPT = """You are a strategic investment analyst updating existing foresights based on recent news.

Review the existing foresights and recent news. Determine what updates are needed:
1. Adjust confidence/timeline if news confirms or contradicts
2. Add relevant new information to descriptions
3. Create new foresights if news reveals new structural changes

IMPORTANT: Only make necessary updates. If no update needed, return the foresight unchanged.

Respond in JSON:
{
    "updated_foresights": [
        {
            "id": "existing_id or new_YYYYMMDD_NNN for new ones",
            "title": "Title (updated if needed)",
            "description": "Description in Japanese (updated if needed)",
            "update_reason": "Why this was updated (or 'no change' if unchanged)"
        }
    ],
    "summary": "Brief summary of changes made"
}

IMPORTANT:
- Write descriptions in Japanese
- Only update what's necessary
- Preserve existing IDs for updated foresights
- Use new IDs (format: foresight_YYYYMMDD_NNN) for new foresights
"""


# =============================================================================
# Agent Nodes
# =============================================================================


def determine_mode_node(state: ForesightState) -> ForesightState:
    """Determine processing mode."""
    mode = determine_mode(state.get("force_full", False))
    state["mode"] = mode
    state["existing_foresights"] = load_foresights()
    print(f"Mode: {mode} (existing foresights: {len(state['existing_foresights'])})")
    return state


async def run_full_mode(state: ForesightState) -> ForesightState:
    """Run full mode using prediction_monitor."""
    if state.get("mode") != "full":
        return state

    print("Running prediction_monitor for full foresight generation...")

    try:
        # Run prediction_monitor
        result = await prediction_monitor.run_scan(
            year=datetime.now().year,
            include_watchers=True,
        )

        report = result.get("report")
        if not report:
            state["error"] = "prediction_monitor returned no report"
            return state

        state["prediction_report"] = report
        predictions = report.get("social_changes", [])
        print(f"Got {len(predictions)} predictions from prediction_monitor")

        if not predictions:
            state["error"] = "No predictions found"
            return state

        # Consolidate predictions into 5 foresights using LLM (single call)
        if not GEMINI_API_KEY:
            state["error"] = "No API key"
            return state

        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GEMINI_API_KEY)

        # Prepare input - list of prediction titles and categories
        predictions_list = []
        for i, pred in enumerate(predictions, 1):
            predictions_list.append(
                f"{i}. [{pred.get('category', 'other')}] {pred.get('title', '')}"
            )
        predictions_text = "\n".join(predictions_list)

        prompt = f"""Consolidate these {len(predictions)} predictions into exactly 5 investment foresights:

{predictions_text}

Remember: Output exactly 5 foresights that capture the major investment themes."""

        messages = [
            SystemMessage(content=FULL_MODE_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = await llm.ainvoke(messages)
        result_text = extract_llm_text(response.content)

        # Parse response
        if isinstance(result_text, str):
            start = result_text.find("{")
            end = result_text.rfind("}") + 1
            if start != -1 and end > start:
                result = json.loads(result_text[start:end])

                foresights: list[Foresight] = []
                timestamp = datetime.now().strftime("%Y%m%d")

                for i, fs_data in enumerate(result.get("foresights", []), 1):
                    # Create sources from source_indices
                    sources = []
                    source_indices = fs_data.get("source_indices", [])

                    for idx in source_indices[:10]:  # Limit to 10 sources per foresight
                        pred_idx = idx - 1  # Convert to 0-based
                        if 0 <= pred_idx < len(predictions):
                            pred = predictions[pred_idx]
                            # Extract article info from implications
                            implications = pred.get("implications", [])
                            article_title = ""
                            article_url = ""
                            article_snippet = ""

                            for impl in implications:
                                if impl.startswith("article_title:"):
                                    article_title = impl[14:]
                                elif impl.startswith("article_url:"):
                                    article_url = impl[12:]
                                elif impl.startswith("article_snippet:"):
                                    article_snippet = impl[16:]

                            if article_title or article_url:
                                sources.append(
                                    ForesightSource(
                                        title=article_title[:200] if article_title else pred.get("title", "")[:200],
                                        url=article_url,
                                        published=datetime.now(),
                                        excerpt=article_snippet[:500] if article_snippet else pred.get("description", "")[:500],
                                    )
                                )

                    # Add fallback source if no valid sources found
                    if not sources:
                        sources.append(
                            ForesightSource(
                                title=f"Prediction Monitor {datetime.now().year}",
                                url="",
                                published=datetime.now(),
                                excerpt=f"Consolidated from {len(predictions)} predictions",
                            )
                        )

                    foresight = Foresight(
                        id=f"foresight_{timestamp}_{i:03d}",
                        title=fs_data.get("title", ""),
                        description=fs_data.get("description", ""),
                        sources=sources,
                        created_at=datetime.now(),
                    )
                    foresights.append(foresight)

                state["updated_foresights"] = foresights
                print(f"Consolidated into {len(foresights)} foresights")

    except Exception as e:
        print(f"Error in full mode: {e}")
        state["error"] = str(e)

    return state


async def run_incremental_mode(state: ForesightState) -> ForesightState:
    """Run incremental mode using news watchers."""
    if state.get("mode") != "incremental":
        return state

    print("Loading recent news for incremental update...")

    # Load recent news
    news_items = load_watcher_news(days_back=7)
    state["news_items"] = news_items
    print(f"Loaded {len(news_items)} structural news items")

    if not news_items:
        print("No new structural news. Keeping existing foresights.")
        state["updated_foresights"] = state["existing_foresights"]
        return state

    if not GEMINI_API_KEY:
        state["error"] = "No API key"
        return state

    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GEMINI_API_KEY)

    try:
        # Prepare existing foresights
        existing_text = json.dumps(
            [fs.model_dump(mode="json") for fs in state["existing_foresights"]],
            ensure_ascii=False,
            indent=2,
        )

        # Prepare news summary (limit for token efficiency)
        news_summary = []
        for item in news_items[:30]:  # Limit to 30 items
            news_summary.append({
                "title": item.title,
                "source": item.source,
                "published": item.published.isoformat() if item.published else "",
                "reason": item.relevance_reason or "",
            })
        news_text = json.dumps(news_summary, ensure_ascii=False, indent=2)

        prompt = f"""EXISTING FORESIGHTS:
{existing_text}

RECENT STRUCTURAL NEWS:
{news_text}

Analyze and update the foresights based on this news."""

        messages = [
            SystemMessage(content=INCREMENTAL_MODE_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = await llm.ainvoke(messages)
        result_text = extract_llm_text(response.content)

        # Parse response
        if isinstance(result_text, str):
            start = result_text.find("{")
            end = result_text.rfind("}") + 1
            if start != -1 and end > start:
                result = json.loads(result_text[start:end])

                # Build updated foresights
                updated_foresights: list[Foresight] = []
                existing_map = {fs.id: fs for fs in state["existing_foresights"]}

                for fs_data in result.get("updated_foresights", []):
                    fs_id = fs_data.get("id", "")

                    if fs_id in existing_map:
                        # Update existing
                        existing = existing_map[fs_id]
                        updated = Foresight(
                            id=fs_id,
                            title=fs_data.get("title", existing.title),
                            description=fs_data.get("description", existing.description),
                            sources=existing.sources,  # Keep existing sources
                            created_at=existing.created_at,
                        )

                        # Add new source if actually updated
                        if fs_data.get("update_reason", "").lower() != "no change":
                            new_source = ForesightSource(
                                title="Incremental Update",
                                url="",
                                published=datetime.now(),
                                excerpt=fs_data.get("update_reason", ""),
                            )
                            updated.sources.append(new_source)

                        updated_foresights.append(updated)
                    else:
                        # New foresight
                        new_source = ForesightSource(
                            title="News Analysis",
                            url="",
                            published=datetime.now(),
                            excerpt="Generated from recent structural news",
                        )
                        new_foresight = Foresight(
                            id=fs_id or f"foresight_{datetime.now().strftime('%Y%m%d')}_{len(updated_foresights)+1:03d}",
                            title=fs_data.get("title", ""),
                            description=fs_data.get("description", ""),
                            sources=[new_source],
                            created_at=datetime.now(),
                        )
                        updated_foresights.append(new_foresight)

                state["updated_foresights"] = updated_foresights

                summary = result.get("summary", "")
                if summary:
                    print(f"Update summary: {summary}")
                print(f"Updated foresights: {len(updated_foresights)}")

    except Exception as e:
        print(f"Error in incremental mode: {e}")
        state["error"] = str(e)
        # Fall back to existing foresights
        state["updated_foresights"] = state["existing_foresights"]

    return state


def save_results(state: ForesightState) -> ForesightState:
    """Save updated foresights to disk."""
    foresights = state.get("updated_foresights", [])

    if not foresights:
        print("No foresights to save.")
        return state

    is_full = state.get("mode") == "full"
    filepath = save_foresights(foresights, is_full_update=is_full)
    state["saved_path"] = str(filepath)
    print(f"Saved {len(foresights)} foresights to {filepath}")

    return state


def route_by_mode(state: ForesightState) -> str:
    """Route to appropriate mode handler."""
    return "full" if state.get("mode") == "full" else "incremental"


# =============================================================================
# Agent Graph
# =============================================================================


def create_agent() -> StateGraph:
    """Create the foresight manager agent graph."""
    workflow = StateGraph(ForesightState)

    # Add nodes
    workflow.add_node("determine_mode", determine_mode_node)
    workflow.add_node("full", run_full_mode)
    workflow.add_node("incremental", run_incremental_mode)
    workflow.add_node("save", save_results)

    # Set entry point
    workflow.set_entry_point("determine_mode")

    # Add conditional routing
    workflow.add_conditional_edges(
        "determine_mode",
        route_by_mode,
        {"full": "full", "incremental": "incremental"},
    )

    # Both modes lead to save
    workflow.add_edge("full", "save")
    workflow.add_edge("incremental", "save")
    workflow.add_edge("save", END)

    return workflow.compile()


async def run_agent(force_full: bool = False) -> ForesightState:
    """Run the foresight manager agent.

    Args:
        force_full: Force full update using prediction_monitor

    Returns:
        ForesightState with results
    """
    agent = create_agent()

    initial_state: ForesightState = {
        "mode": "",
        "force_full": force_full,
        "existing_foresights": [],
        "news_items": [],
        "prediction_report": None,
        "updated_foresights": [],
        "saved_path": None,
        "error": None,
    }

    return await agent.ainvoke(initial_state)


# =============================================================================
# Utility Functions
# =============================================================================


def list_foresights() -> list[Foresight]:
    """List all stored foresights."""
    return load_foresights()


def format_foresights_list(foresights: list[Foresight]) -> str:
    """Format foresights list for display."""
    if not foresights:
        return "No foresights stored yet. Run 'midas foresight scan' to generate."

    lines = [
        "",
        "=" * 70,
        f"FORESIGHTS ({len(foresights)} total)",
        "=" * 70,
        "",
    ]

    for i, fs in enumerate(foresights, 1):
        lines.append(f"{i}. {fs.title}")
        lines.append(f"   ID: {fs.id}")
        lines.append(f"   Created: {fs.created_at.strftime('%Y-%m-%d')}")
        lines.append(f"   {fs.description[:150]}...")
        lines.append(f"   Sources: {len(fs.sources)}")
        lines.append("")

    return "\n".join(lines)


def format_results(state: ForesightState) -> str:
    """Format agent results for display."""
    if state.get("error"):
        return f"Error: {state['error']}"

    lines = [
        "",
        "=" * 70,
        f"FORESIGHT SCAN COMPLETE",
        f"Mode: {state.get('mode', 'unknown')}",
        "=" * 70,
        "",
    ]

    foresights = state.get("updated_foresights", [])
    if foresights:
        lines.append(f"Foresights: {len(foresights)}")
        lines.append("-" * 40)

        for i, fs in enumerate(foresights, 1):
            lines.append(f"\n{i}. {fs.title}")
            lines.append(f"   {fs.description[:200]}...")

    if state.get("saved_path"):
        lines.append("")
        lines.append(f"Saved to: {state['saved_path']}")

    return "\n".join(lines)


# =============================================================================
# Report Generation
# =============================================================================


def generate_report(executive_summary: str = "") -> Path:
    """Generate a Markdown report from current foresights.

    Args:
        executive_summary: Optional executive summary text

    Returns:
        Path to the generated report
    """
    foresights = load_foresights()

    if not foresights:
        print("No foresights available. Run 'midas foresight scan' first.")
        return None

    # Convert Foresight objects to dicts for report generator
    foresight_dicts = []
    for fs in foresights:
        foresight_dicts.append({
            "title": fs.title,
            "description": fs.description,
            "sources": [
                {
                    "title": src.title,
                    "url": src.url,
                    "excerpt": src.excerpt,
                }
                for src in fs.sources
            ],
        })

    # Get last update info
    last_full = get_last_full_update()
    period = ""
    if last_full:
        period = f"Last full update: {last_full.strftime('%Y-%m-%d')}"

    # Generate report content
    content = generate_foresight_report(
        foresights=foresight_dicts,
        executive_summary=executive_summary,
        period=period,
    )

    # Save report
    report_path = save_report(content, "foresight")
    print(f"Report saved to: {report_path}")

    return report_path
