"""Price Event Analyzer - Analyzes causes of significant stock price movements."""

import json
from datetime import datetime, timedelta
from typing import TypedDict

import yfinance as yf
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from midas.config import extract_llm_text, DATA_DIR, GEMINI_API_KEY, LLM_MODEL
from midas.logging_config import get_agent_logger
from midas.models import CompanyNews, PriceEvent, PriceEventAnalysis
from midas.tools.company_news_fetcher import fetch_news_around_date

# Logger setup
logger = get_agent_logger("price_event_analyzer")

# =============================================================================
# Constants
# =============================================================================

WATCHER_DATA_DIR = DATA_DIR / "company_analysis"

# Threshold for significant daily movement
SIGNIFICANT_MOVE_THRESHOLD = 5.0  # 5% daily change

# =============================================================================
# Agent State
# =============================================================================


class AnalyzerState(TypedDict):
    """State for the price event analyzer agent."""

    symbol: str
    company_name: str
    price_events: list[PriceEvent]
    news_by_event: dict[str, list[CompanyNews]]  # event date -> news list
    analyses: list[PriceEventAnalysis]
    saved_path: str | None
    error: str | None


# =============================================================================
# LLM Analysis Prompt
# =============================================================================

ANALYSIS_SYSTEM_PROMPT = """You are an expert financial analyst specializing in understanding stock price movements.

Your task is to analyze a significant price movement and determine its likely cause based on news articles from around that time.

Consider:
1. Earnings reports (beats/misses)
2. Product launches or failures
3. Management changes
4. Regulatory actions
5. Legal issues (lawsuits, investigations)
6. Mergers, acquisitions, or divestitures
7. Analyst upgrades/downgrades
8. Macro events affecting the sector
9. Competitor news
10. Guidance changes

Respond in JSON format:
{
    "likely_cause": "Brief description of the most likely cause",
    "confidence": 0.0-1.0,
    "first_reporter": "Name of the source that likely broke the news first (if identifiable)",
    "news_rankings": [
        {
            "title": "Article title",
            "relevance_score": 0.0-1.0,
            "sentiment": "positive/negative/neutral",
            "is_first_report": true/false
        }
    ]
}
"""


# =============================================================================
# Agent Nodes
# =============================================================================


async def fetch_price_data(state: AnalyzerState) -> AnalyzerState:
    """Fetch stock price data and identify significant movements."""
    symbol = state["symbol"]
    logger.info(f"Fetching price data for {symbol}...")

    try:
        ticker = yf.Ticker(symbol)

        # Get company name
        info = ticker.info
        state["company_name"] = info.get("shortName") or info.get("longName") or symbol

        # Get 1 year of historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        hist = ticker.history(start=start_date, end=end_date)

        if hist.empty:
            state["error"] = f"No price data available for {symbol}"
            return state

        # Calculate daily returns
        hist["Return"] = hist["Close"].pct_change() * 100
        hist["Volume_Avg"] = hist["Volume"].rolling(window=20).mean()
        hist["Volume_Ratio"] = hist["Volume"] / hist["Volume_Avg"]

        # Find significant movements
        events: list[PriceEvent] = []

        for i in range(1, len(hist)):
            daily_return = hist["Return"].iloc[i]
            if abs(daily_return) >= SIGNIFICANT_MOVE_THRESHOLD:
                event_date = hist.index[i].to_pydatetime()
                if event_date.tzinfo:
                    event_date = event_date.replace(tzinfo=None)

                event = PriceEvent(
                    date=event_date,
                    price_before=round(hist["Close"].iloc[i - 1], 2),
                    price_after=round(hist["Close"].iloc[i], 2),
                    change_percent=round(daily_return, 2),
                    volume_ratio=round(hist["Volume_Ratio"].iloc[i], 2)
                    if not hist["Volume_Ratio"].isna().iloc[i]
                    else 1.0,
                )
                events.append(event)

        # Sort by absolute change (most significant first)
        events.sort(key=lambda x: abs(x.change_percent), reverse=True)

        # Keep top 10 events
        state["price_events"] = events[:10]
        logger.info(f"Found {len(state['price_events'])} significant price events")

    except Exception as e:
        state["error"] = f"Failed to fetch price data: {e}"
        state["price_events"] = []

    return state


async def fetch_related_news(state: AnalyzerState) -> AnalyzerState:
    """Fetch news around each significant price event."""
    if state.get("error") or not state.get("price_events"):
        state["news_by_event"] = {}
        return state

    symbol = state["symbol"]
    company_name = state["company_name"]
    logger.info(f"Fetching news for {len(state['price_events'])} price events...")

    news_by_event: dict[str, list[CompanyNews]] = {}

    for event in state["price_events"]:
        event_key = event.date.strftime("%Y-%m-%d")
        logger.info(f"  Fetching news around {event_key}...")

        # Search with both symbol and company name
        news_items: list[CompanyNews] = []

        for query in [symbol, company_name]:
            items = await fetch_news_around_date(
                query=query,
                target_date=event.date,
                days_before=3,
                days_after=1,
            )
            news_items.extend(items)

        # Deduplicate by URL
        seen_urls: set[str] = set()
        unique_news: list[CompanyNews] = []
        for item in news_items:
            if item.url not in seen_urls:
                seen_urls.add(item.url)
                unique_news.append(item)

        news_by_event[event_key] = unique_news
        logger.info(f"    Found {len(unique_news)} news items")

    state["news_by_event"] = news_by_event
    return state


async def analyze_events(state: AnalyzerState) -> AnalyzerState:
    """Use LLM to analyze each price event and its related news."""
    if state.get("error") or not state.get("price_events"):
        state["analyses"] = []
        return state

    logger.info("Analyzing price events with LLM...")

    if not GEMINI_API_KEY:
        logger.warning("No API key, skipping LLM analysis")
        # Return basic analysis without LLM
        state["analyses"] = [
            PriceEventAnalysis(
                event=event,
                likely_cause="Analysis unavailable (no API key)",
                related_news=state["news_by_event"].get(
                    event.date.strftime("%Y-%m-%d"), []
                ),
            )
            for event in state["price_events"]
        ]
        return state

    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GEMINI_API_KEY)
    analyses: list[PriceEventAnalysis] = []

    for event in state["price_events"]:
        event_key = event.date.strftime("%Y-%m-%d")
        news_items = state["news_by_event"].get(event_key, [])

        logger.info(f"  Analyzing event on {event_key} ({event.change_percent:+.1f}%)...")

        try:
            # Prepare context for LLM
            direction = "UP" if event.change_percent > 0 else "DOWN"
            news_context = "\n\n".join(
                [
                    f"[{n.source}] {n.title}\n{n.snippet[:300]}"
                    for n in news_items[:15]  # Limit to 15 articles
                ]
            )

            prompt = f"""Company: {state['company_name']} ({state['symbol']})
Event Date: {event_key}
Price Movement: {direction} {abs(event.change_percent):.1f}%
Volume Ratio: {event.volume_ratio:.1f}x average

NEWS ARTICLES FROM AROUND THIS DATE:
{news_context if news_context else "No news articles found."}

Analyze what likely caused this price movement."""

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

                    # Update news items with LLM analysis
                    updated_news: list[CompanyNews] = []
                    rankings = result.get("news_rankings", [])

                    for news in news_items:
                        # Find matching ranking
                        for rank in rankings:
                            if rank.get("title", "").lower() in news.title.lower():
                                news.relevance_score = rank.get("relevance_score", 0.0)
                                news.sentiment = rank.get("sentiment")
                                news.is_first_report = rank.get("is_first_report", False)
                                break
                        updated_news.append(news)

                    # Sort by relevance
                    updated_news.sort(key=lambda x: x.relevance_score, reverse=True)

                    analysis = PriceEventAnalysis(
                        event=event,
                        likely_cause=result.get("likely_cause", "Unknown"),
                        related_news=updated_news,
                        first_reporter=result.get("first_reporter"),
                        confidence=result.get("confidence", 0.0),
                    )
                    analyses.append(analysis)
                    logger.info(f"    Cause: {analysis.likely_cause[:60]}...")

        except Exception as e:
            logger.error(f"Error analyzing event: {e}")
            # Add basic analysis
            analyses.append(
                PriceEventAnalysis(
                    event=event,
                    likely_cause=f"Analysis failed: {e}",
                    related_news=news_items,
                )
            )

    state["analyses"] = analyses
    return state


def save_results(state: AnalyzerState) -> AnalyzerState:
    """Save analysis results to JSON file."""
    if not state.get("analyses"):
        state["saved_path"] = None
        return state

    logger.info("Saving analysis results...")

    # Ensure directory exists
    WATCHER_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Create filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = WATCHER_DATA_DIR / f"price_analysis_{state['symbol']}_{timestamp}.json"

    # Convert to serializable format
    data = {
        "symbol": state["symbol"],
        "company_name": state["company_name"],
        "analyzed_at": datetime.now().isoformat(),
        "total_events": len(state["analyses"]),
        "analyses": [
            {
                "event": analysis.event.model_dump(mode="json"),
                "likely_cause": analysis.likely_cause,
                "first_reporter": analysis.first_reporter,
                "confidence": analysis.confidence,
                "related_news_count": len(analysis.related_news),
                "top_news": [
                    news.model_dump(mode="json") for news in analysis.related_news[:5]
                ],
            }
            for analysis in state["analyses"]
        ],
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    state["saved_path"] = str(filepath)
    logger.info(f"Saved to: {filepath}")
    return state


# =============================================================================
# Agent Runner
# =============================================================================


async def run_agent(symbol: str) -> AnalyzerState:
    """Run the price event analyzer agent.

    Args:
        symbol: Stock ticker symbol to analyze

    Returns:
        Analysis results
    """
    state: AnalyzerState = {
        "symbol": symbol.upper(),
        "company_name": "",
        "price_events": [],
        "news_by_event": {},
        "analyses": [],
        "saved_path": None,
        "error": None,
    }

    # Execute pipeline steps sequentially
    state = await fetch_price_data(state)
    state = await fetch_related_news(state)
    state = await analyze_events(state)
    state = await save_results(state)

    return state


def format_analysis(state: AnalyzerState) -> str:
    """Format analysis results for display."""
    if state.get("error"):
        return f"Error: {state['error']}"

    if not state.get("analyses"):
        return f"No significant price events found for {state['symbol']}"

    lines = [
        f"\n{'=' * 70}",
        f"Price Event Analysis: {state['company_name']} ({state['symbol']})",
        f"{'=' * 70}",
    ]

    for i, analysis in enumerate(state["analyses"], 1):
        event = analysis.event
        direction = "📈" if event.change_percent > 0 else "📉"

        lines.append(f"\n{i}. {direction} {event.date.strftime('%Y-%m-%d')}: {event.change_percent:+.1f}%")
        lines.append(f"   Price: ${event.price_before:.2f} → ${event.price_after:.2f}")
        lines.append(f"   Volume: {event.volume_ratio:.1f}x average")
        lines.append(f"   Likely Cause: {analysis.likely_cause}")
        lines.append(f"   Confidence: {analysis.confidence:.0%}")

        if analysis.first_reporter:
            lines.append(f"   First Reporter: {analysis.first_reporter}")

        if analysis.related_news:
            lines.append("   Top Related News:")
            for news in analysis.related_news[:3]:
                lines.append(f"      - [{news.source}] {news.title[:50]}...")

    if state.get("saved_path"):
        lines.append(f"\nFull results saved to: {state['saved_path']}")

    return "\n".join(lines)
