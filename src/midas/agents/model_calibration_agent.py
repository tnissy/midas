"""Model Calibration Agent - Analyzes significant price movements to extract lessons.

This agent:
1. Scans for stocks with extreme movements (3x or 1/3 in one month)
2. Investigates the root cause (structural change)
3. Extracts lessons that can improve future analysis
"""

import hashlib
import json
from datetime import datetime, timedelta
from typing import TypedDict

import yfinance as yf
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from midas.config import DATA_DIR, GEMINI_API_KEY, LLM_MODEL, extract_llm_text
from midas.models import (
    LearningCase,
    LearningReport,
    LearnedInsight,
    MovementDirection,
    StructuralChangeType,
    SuggestedFeed,
    SuggestedKeyword,
    WatcherType,
)
from midas.tools.company_news_fetcher import fetch_news_around_date
from midas.tools.stock_screener import TimeFrame, screen_movers

# =============================================================================
# Constants
# =============================================================================

LEARNING_DATA_DIR = DATA_DIR / "learning"
CASES_DIR = LEARNING_DATA_DIR / "cases"
INSIGHTS_DIR = LEARNING_DATA_DIR / "insights"

# Thresholds for significant movement (1 month)
THRESHOLD_UP = 200.0  # +200% = 3x
THRESHOLD_DOWN = -66.7  # -66.7% = 1/3


# =============================================================================
# Agent State
# =============================================================================


class LearningState(TypedDict):
    """State for the learning agent."""

    # Input
    period: str  # e.g., "month", "quarter"
    max_cases: int

    # Discovered cases
    up_movers: list[dict]
    down_movers: list[dict]

    # Analysis results
    cases: list[LearningCase]
    insights: list[LearnedInsight]
    report: LearningReport | None

    # Output
    saved_path: str | None
    error: str | None


# =============================================================================
# LLM Prompts
# =============================================================================

ANALYSIS_SYSTEM_PROMPT = """You are an expert investment analyst specializing in identifying structural changes that cause significant stock price movements.

Your task is to analyze a company that experienced extreme price movement (3x increase or 1/3 decrease in one month) and determine:

1. The TYPE of structural change that caused this movement
2. The ROOT CAUSE in detail
3. EARLY SIGNALS that could have predicted this
4. LESSONS LEARNED for future analysis

Structural Change Types:
- technology_breakthrough: Technological innovation or disruption
- regulation_change: Regulatory or policy changes
- market_structure: Fundamental changes in market dynamics
- competitive_dynamics: Shifts in competitive landscape
- business_model: Business model transformation
- management_change: Leadership changes
- macro_economic: Macroeconomic factors
- geopolitical: Geopolitical events
- fraud_scandal: Fraud, scandal, or corporate misconduct
- other: Other factors

Respond in JSON format:
{
    "structural_change_type": "one of the types above",
    "root_cause": "Detailed explanation of what caused the movement",
    "early_signals": ["Signal 1", "Signal 2", ...],
    "lessons_learned": ["Lesson 1", "Lesson 2", ...],
    "confidence": 0.0-1.0
}

Focus on STRUCTURAL changes, not temporary market noise. Look for patterns that could be detected by monitoring news and data sources.
"""

INSIGHT_SYNTHESIS_PROMPT = """You are synthesizing lessons from multiple stock price movement cases to create actionable insights.

Given the cases analyzed, identify patterns and create insights that can improve investment analysis.

For each insight:
- Provide a clear, actionable title
- Describe the pattern in detail
- List detection patterns (what to watch for in news/data)
- Specify which sectors this applies to
- Rate importance (low/medium/high/critical)
- Suggest RSS feeds that should be monitored to catch similar patterns early
- Suggest keywords to watch for in news
- Specify which watchers should act on this insight

Available watchers:
- tech_news_watcher: Technology news (Ars Technica, TechCrunch, MIT Tech Review, etc.)
- us_gov_watcher: US government news (White House, Congress, SEC, Federal Register)
- other_gov_watcher: Non-US government news (EU, UK, China, Japan, IMF, World Bank)
- general_news_watcher: General financial news (Bloomberg, Reuters, Yahoo Finance)
- prediction_monitor: Annual outlook articles (McKinsey, BCG, WEF)

Respond in JSON format:
{
    "insights": [
        {
            "title": "Brief insight title",
            "category": "structural_change_type",
            "description": "Detailed description",
            "detection_patterns": ["Pattern to watch 1", "Pattern to watch 2"],
            "applicable_sectors": ["Technology", "Healthcare", ...],
            "importance": "high",
            "target_watchers": ["tech_news_watcher", "us_gov_watcher"],
            "suggested_feeds": [
                {
                    "name": "Feed Name",
                    "url": "https://example.com/rss",
                    "target_watcher": "tech_news_watcher",
                    "reason": "Why this feed is relevant",
                    "priority": "high"
                }
            ],
            "suggested_keywords": [
                {
                    "keyword": "AI regulation",
                    "target_watcher": "us_gov_watcher",
                    "reason": "Why this keyword matters"
                }
            ]
        }
    ],
    "key_findings": ["Finding 1", "Finding 2", ...],
    "recommendations": ["Recommendation 1", "Recommendation 2", ...]
}
"""


# =============================================================================
# Helper Functions
# =============================================================================


def generate_case_id(symbol: str, period_start: datetime) -> str:
    """Generate a unique case ID."""
    key = f"{symbol}_{period_start.strftime('%Y%m%d')}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def generate_insight_id(title: str) -> str:
    """Generate a unique insight ID."""
    return hashlib.md5(title.encode()).hexdigest()[:12]


def load_existing_cases() -> list[LearningCase]:
    """Load existing learning cases from disk."""
    cases = []
    if not CASES_DIR.exists():
        return cases

    for filepath in CASES_DIR.glob("*.json"):
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
                cases.append(LearningCase(**data))
        except Exception:
            continue
    return cases


def load_existing_insights() -> list[LearnedInsight]:
    """Load existing insights from disk."""
    insights = []
    if not INSIGHTS_DIR.exists():
        return insights

    for filepath in INSIGHTS_DIR.glob("*.json"):
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
                insights.append(LearnedInsight(**data))
        except Exception:
            continue
    return insights


def save_case(case: LearningCase) -> None:
    """Save a learning case to disk."""
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    filepath = CASES_DIR / f"{case.id}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(case.model_dump(mode="json"), f, ensure_ascii=False, indent=2)


def save_insight(insight: LearnedInsight) -> None:
    """Save an insight to disk."""
    INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = INSIGHTS_DIR / f"{insight.id}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(insight.model_dump(mode="json"), f, ensure_ascii=False, indent=2)


# =============================================================================
# Agent Nodes
# =============================================================================


async def scan_extreme_movers(state: LearningState) -> LearningState:
    """Scan for stocks with extreme price movements."""
    print("Scanning for extreme price movements...")

    period = state.get("period", "month")
    max_cases = state.get("max_cases", 20)

    # Map period to TimeFrame
    timeframe_map = {
        "month": TimeFrame.MONTH,
        "quarter": TimeFrame.QUARTER,
        "half": TimeFrame.HALF_YEAR,
        "year": TimeFrame.YEAR,
    }
    timeframe = timeframe_map.get(period, TimeFrame.MONTH)

    try:
        # Screen for gainers (200%+)
        print(f"  Screening for 3x gainers ({timeframe.value})...")
        gainers = screen_movers(
            timeframe=timeframe,
            min_change=THRESHOLD_UP,
            direction="up",
            max_results=max_cases,
        )
        state["up_movers"] = [
            {
                "symbol": m.symbol,
                "name": m.name,
                "price_before": m.price_before,
                "price_now": m.price_now,
                "change_percent": m.change_percent,
            }
            for m in gainers
            if m.is_significant
        ]
        print(f"  Found {len(state['up_movers'])} 3x gainers")

        # Screen for losers (-66.7%+)
        print(f"  Screening for 1/3 losers ({timeframe.value})...")
        losers = screen_movers(
            timeframe=timeframe,
            min_change=abs(THRESHOLD_DOWN),
            direction="down",
            max_results=max_cases,
        )
        state["down_movers"] = [
            {
                "symbol": m.symbol,
                "name": m.name,
                "price_before": m.price_before,
                "price_now": m.price_now,
                "change_percent": m.change_percent,
            }
            for m in losers
            if m.is_significant
        ]
        print(f"  Found {len(state['down_movers'])} 1/3 losers")

    except Exception as e:
        state["error"] = f"Failed to scan for movers: {e}"
        state["up_movers"] = []
        state["down_movers"] = []

    return state


async def fetch_case_details(state: LearningState) -> LearningState:
    """Fetch detailed information for each case."""
    if state.get("error"):
        return state

    all_movers = state["up_movers"] + state["down_movers"]
    if not all_movers:
        print("No extreme movers found to analyze.")
        state["cases"] = []
        return state

    print(f"Fetching details for {len(all_movers)} cases...")

    # Get existing case IDs to avoid duplicates
    existing_ids = {c.id for c in load_existing_cases()}

    # Calculate period dates
    end_date = datetime.now()
    period_days = {
        "month": 30,
        "quarter": 90,
        "half": 180,
        "year": 365,
    }
    days = period_days.get(state.get("period", "month"), 30)
    start_date = end_date - timedelta(days=days)

    cases: list[LearningCase] = []

    for mover in all_movers[: state.get("max_cases", 20)]:
        symbol = mover["symbol"]
        case_id = generate_case_id(symbol, start_date)

        # Skip if already analyzed
        if case_id in existing_ids:
            print(f"  Skipping {symbol} (already analyzed)")
            continue

        print(f"  Processing {symbol}...")

        try:
            # Get company name from yfinance
            ticker = yf.Ticker(symbol)
            info = ticker.info
            company_name = info.get("shortName") or info.get("longName") or symbol

            # Determine direction
            direction = (
                MovementDirection.UP
                if mover["change_percent"] > 0
                else MovementDirection.DOWN
            )

            case = LearningCase(
                id=case_id,
                symbol=symbol,
                company_name=company_name,
                direction=direction,
                price_before=mover["price_before"],
                price_after=mover["price_now"],
                change_percent=mover["change_percent"],
                period_start=start_date,
                period_end=end_date,
            )

            # Fetch news around the period
            news_items = await fetch_news_around_date(
                query=f"{symbol} {company_name}",
                target_date=start_date + timedelta(days=days // 2),  # Middle of period
                days_before=days // 2 + 7,
                days_after=days // 2 + 7,
            )
            case.news_context = [item.title for item in news_items[:10]]

            cases.append(case)

        except Exception as e:
            print(f"    Error processing {symbol}: {e}")
            continue

    state["cases"] = cases
    print(f"Prepared {len(cases)} cases for analysis")
    return state


async def analyze_cases(state: LearningState) -> LearningState:
    """Use LLM to analyze each case and identify structural changes."""
    if state.get("error") or not state.get("cases"):
        return state

    if not GEMINI_API_KEY:
        print("Warning: No API key, skipping LLM analysis")
        return state

    print("Analyzing cases with LLM...")
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GEMINI_API_KEY)

    for case in state["cases"]:
        print(f"  Analyzing {case.symbol} ({case.change_percent:+.1f}%)...")

        try:
            direction = "increased 3x+" if case.direction == MovementDirection.UP else "decreased to 1/3"
            news_context = "\n".join(f"- {n}" for n in case.news_context) if case.news_context else "No news found"

            prompt = f"""Company: {case.company_name} ({case.symbol})
Movement: Stock {direction} ({case.change_percent:+.1f}%)
Period: {case.period_start.strftime('%Y-%m-%d')} to {case.period_end.strftime('%Y-%m-%d')}
Price: ${case.price_before:.2f} → ${case.price_after:.2f}

NEWS HEADLINES FROM THIS PERIOD:
{news_context}

Analyze what structural change caused this extreme price movement."""

            messages = [
                SystemMessage(content=ANALYSIS_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]

            response = await llm.ainvoke(messages)
            result_text = extract_llm_text(response.content)

            # Parse JSON response
            if isinstance(result_text, str):
                start = result_text.find("{")
                end = result_text.rfind("}") + 1
                if start != -1 and end > start:
                    result = json.loads(result_text[start:end])

                    # Update case with analysis
                    change_type = result.get("structural_change_type", "other")
                    try:
                        case.structural_change_type = StructuralChangeType(change_type)
                    except ValueError:
                        case.structural_change_type = StructuralChangeType.OTHER

                    case.root_cause = result.get("root_cause", "")
                    case.early_signals = result.get("early_signals", [])
                    case.lessons_learned = result.get("lessons_learned", [])
                    case.confidence = result.get("confidence", 0.5)

                    print(f"    Cause: {case.root_cause[:60]}...")

            # Save the analyzed case
            save_case(case)

        except Exception as e:
            print(f"    Error analyzing {case.symbol}: {e}")
            continue

    return state


async def synthesize_insights(state: LearningState) -> LearningState:
    """Synthesize insights from analyzed cases."""
    if state.get("error"):
        return state

    analyzed_cases = [c for c in state.get("cases", []) if c.root_cause]
    if not analyzed_cases:
        print("No analyzed cases to synthesize insights from.")
        state["insights"] = []
        return state

    if not GEMINI_API_KEY:
        print("Warning: No API key, skipping insight synthesis")
        state["insights"] = []
        return state

    print("Synthesizing insights from cases...")
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GEMINI_API_KEY)

    try:
        # Prepare case summaries
        case_summaries = []
        for case in analyzed_cases:
            summary = f"""
- {case.company_name} ({case.symbol}): {case.direction.value} {case.change_percent:+.1f}%
  Type: {case.structural_change_type.value if case.structural_change_type else 'unknown'}
  Cause: {case.root_cause}
  Signals: {', '.join(case.early_signals[:3]) if case.early_signals else 'None'}
  Lessons: {', '.join(case.lessons_learned[:3]) if case.lessons_learned else 'None'}
"""
            case_summaries.append(summary)

        prompt = f"""Analyze these {len(analyzed_cases)} cases of extreme stock movements and synthesize actionable insights:

{''.join(case_summaries)}

Create insights that can help detect similar situations in the future."""

        messages = [
            SystemMessage(content=INSIGHT_SYNTHESIS_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = await llm.ainvoke(messages)
        result_text = extract_llm_text(response.content)

        insights: list[LearnedInsight] = []
        key_findings: list[str] = []
        recommendations: list[str] = []

        if isinstance(result_text, str):
            start = result_text.find("{")
            end = result_text.rfind("}") + 1
            if start != -1 and end > start:
                result = json.loads(result_text[start:end])

                key_findings = result.get("key_findings", [])
                recommendations = result.get("recommendations", [])

                for insight_data in result.get("insights", []):
                    insight_id = generate_insight_id(insight_data.get("title", ""))

                    try:
                        category = StructuralChangeType(insight_data.get("category", "other"))
                    except ValueError:
                        category = StructuralChangeType.OTHER

                    # Parse suggested feeds
                    suggested_feeds = []
                    for feed_data in insight_data.get("suggested_feeds", []):
                        try:
                            target_watcher = WatcherType(feed_data.get("target_watcher", "tech_news_watcher"))
                            suggested_feeds.append(SuggestedFeed(
                                name=feed_data.get("name", ""),
                                url=feed_data.get("url", ""),
                                target_watcher=target_watcher,
                                reason=feed_data.get("reason", ""),
                                priority=feed_data.get("priority", "medium"),
                            ))
                        except (ValueError, KeyError):
                            continue

                    # Parse suggested keywords
                    suggested_keywords = []
                    for kw_data in insight_data.get("suggested_keywords", []):
                        try:
                            target_watcher = WatcherType(kw_data.get("target_watcher", "tech_news_watcher"))
                            suggested_keywords.append(SuggestedKeyword(
                                keyword=kw_data.get("keyword", ""),
                                target_watcher=target_watcher,
                                reason=kw_data.get("reason", ""),
                            ))
                        except (ValueError, KeyError):
                            continue

                    # Parse target watchers
                    target_watchers = []
                    for tw in insight_data.get("target_watchers", []):
                        try:
                            target_watchers.append(WatcherType(tw))
                        except ValueError:
                            continue

                    insight = LearnedInsight(
                        id=insight_id,
                        title=insight_data.get("title", ""),
                        category=category,
                        description=insight_data.get("description", ""),
                        detection_patterns=insight_data.get("detection_patterns", []),
                        applicable_sectors=insight_data.get("applicable_sectors", []),
                        source_cases=[c.id for c in analyzed_cases],
                        importance=insight_data.get("importance", "medium"),
                        suggested_feeds=suggested_feeds,
                        suggested_keywords=suggested_keywords,
                        target_watchers=target_watchers,
                    )
                    insights.append(insight)
                    save_insight(insight)

        state["insights"] = insights

        # Create report
        state["report"] = LearningReport(
            period_analyzed=state.get("period", "month"),
            total_cases_analyzed=len(analyzed_cases),
            new_insights=insights,
            key_findings=key_findings,
            recommendations=recommendations,
        )

        print(f"Created {len(insights)} insights")

    except Exception as e:
        print(f"Error synthesizing insights: {e}")
        state["insights"] = []

    return state


def save_report(state: LearningState) -> LearningState:
    """Save the learning report."""
    if not state.get("report"):
        state["saved_path"] = None
        return state

    print("Saving learning report...")

    LEARNING_DATA_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = LEARNING_DATA_DIR / f"learning_report_{timestamp}.json"

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(state["report"].model_dump(mode="json"), f, ensure_ascii=False, indent=2)

    state["saved_path"] = str(filepath)
    print(f"Saved to: {filepath}")
    return state


# =============================================================================
# Agent Graph
# =============================================================================


def create_agent() -> StateGraph:
    """Create the learning agent graph."""
    workflow = StateGraph(LearningState)

    # Add nodes
    workflow.add_node("scan", scan_extreme_movers)
    workflow.add_node("fetch_details", fetch_case_details)
    workflow.add_node("analyze", analyze_cases)
    workflow.add_node("synthesize", synthesize_insights)
    workflow.add_node("save", save_report)

    # Define edges
    workflow.set_entry_point("scan")
    workflow.add_edge("scan", "fetch_details")
    workflow.add_edge("fetch_details", "analyze")
    workflow.add_edge("analyze", "synthesize")
    workflow.add_edge("synthesize", "save")
    workflow.add_edge("save", END)

    return workflow.compile()


async def run_agent(period: str = "month", max_cases: int = 20) -> LearningState:
    """Run the learning agent.

    Args:
        period: Time period to scan (month, quarter, half, year)
        max_cases: Maximum number of cases to analyze

    Returns:
        Learning results including cases and insights
    """
    agent = create_agent()

    initial_state: LearningState = {
        "period": period,
        "max_cases": max_cases,
        "up_movers": [],
        "down_movers": [],
        "cases": [],
        "insights": [],
        "report": None,
        "saved_path": None,
        "error": None,
    }

    result = await agent.ainvoke(initial_state)
    return result


def format_report(state: LearningState) -> str:
    """Format learning results for display."""
    if state.get("error"):
        return f"Error: {state['error']}"

    lines = [
        f"\n{'=' * 70}",
        "Learning Report: Stock Price Movement Analysis",
        f"{'=' * 70}",
    ]

    # Summary
    cases = state.get("cases", [])
    up_count = len([c for c in cases if c.direction == MovementDirection.UP])
    down_count = len([c for c in cases if c.direction == MovementDirection.DOWN])

    lines.append(f"\nPeriod: {state.get('period', 'month')}")
    lines.append(f"Total cases analyzed: {len(cases)}")
    lines.append(f"  - 3x gainers: {up_count}")
    lines.append(f"  - 1/3 losers: {down_count}")

    # Cases
    if cases:
        lines.append(f"\n{'=' * 70}")
        lines.append("Analyzed Cases")
        lines.append(f"{'=' * 70}")

        for case in cases:
            direction = "UP" if case.direction == MovementDirection.UP else "DOWN"
            lines.append(f"\n{case.symbol} ({case.company_name}) - {direction} {case.change_percent:+.1f}%")
            lines.append(f"  Price: ${case.price_before:.2f} → ${case.price_after:.2f}")

            if case.structural_change_type:
                lines.append(f"  Change Type: {case.structural_change_type.value}")

            if case.root_cause:
                lines.append(f"  Root Cause: {case.root_cause[:100]}...")

            if case.early_signals:
                lines.append("  Early Signals:")
                for signal in case.early_signals[:3]:
                    lines.append(f"    - {signal}")

            if case.lessons_learned:
                lines.append("  Lessons:")
                for lesson in case.lessons_learned[:3]:
                    lines.append(f"    - {lesson}")

    # Insights
    insights = state.get("insights", [])
    if insights:
        lines.append(f"\n{'=' * 70}")
        lines.append("Synthesized Insights")
        lines.append(f"{'=' * 70}")

        for i, insight in enumerate(insights, 1):
            lines.append(f"\n{i}. [{insight.importance.upper()}] {insight.title}")
            lines.append(f"   Category: {insight.category.value}")
            lines.append(f"   {insight.description}")

            if insight.detection_patterns:
                lines.append("   Detection Patterns:")
                for pattern in insight.detection_patterns[:3]:
                    lines.append(f"     - {pattern}")

    # Report summary
    report = state.get("report")
    if report:
        if report.key_findings:
            lines.append(f"\n{'=' * 70}")
            lines.append("Key Findings")
            lines.append(f"{'=' * 70}")
            for finding in report.key_findings:
                lines.append(f"  - {finding}")

        if report.recommendations:
            lines.append(f"\n{'=' * 70}")
            lines.append("Recommendations")
            lines.append(f"{'=' * 70}")
            for rec in report.recommendations:
                lines.append(f"  - {rec}")

    if state.get("saved_path"):
        lines.append(f"\nFull report saved to: {state['saved_path']}")

    return "\n".join(lines)


def list_insights() -> list[LearnedInsight]:
    """List all stored insights."""
    return load_existing_insights()


def list_cases() -> list[LearningCase]:
    """List all stored cases."""
    return load_existing_cases()


def format_insights_list(insights: list[LearnedInsight]) -> str:
    """Format insights list for display."""
    if not insights:
        return "No insights stored yet. Run 'midas learn scan' to analyze cases."

    lines = [
        f"\n{'=' * 70}",
        f"Stored Insights ({len(insights)} total)",
        f"{'=' * 70}",
    ]

    # Group by category
    by_category: dict[str, list[LearnedInsight]] = {}
    for insight in insights:
        cat = insight.category.value
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(insight)

    for category, cat_insights in sorted(by_category.items()):
        lines.append(f"\n[{category.upper()}]")
        for insight in cat_insights:
            lines.append(f"  [{insight.importance}] {insight.title}")
            if insight.detection_patterns:
                lines.append(f"    Patterns: {', '.join(insight.detection_patterns[:2])}")

    return "\n".join(lines)
