"""Company Watcher - Monitors companies for negative news and risks."""

import json
import logging
from datetime import datetime
from typing import TypedDict

import yfinance as yf
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from midas.config import DATA_DIR, GEMINI_API_KEY, LLM_MODEL
from midas.models import CompanyNews, NegativeInfo
from midas.tools.company_news_fetcher import fetch_company_news

# =============================================================================
# Logging Setup
# =============================================================================

LOG_DIR = DATA_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("company_watcher")
logger.setLevel(logging.DEBUG)

# =============================================================================
# Constants
# =============================================================================

WATCHER_DATA_DIR = DATA_DIR / "company_analysis"

# Negative keywords for initial filtering
NEGATIVE_KEYWORDS = [
    # Legal/Regulatory
    "lawsuit", "sued", "litigation", "investigation", "probe", "subpoena",
    "fraud", "violation", "penalty", "fine", "settlement", "indictment",
    "regulatory", "sec", "ftc", "doj", "antitrust",
    # Business Issues
    "recall", "defect", "safety", "warning", "layoff", "restructuring",
    "bankruptcy", "default", "downgrade", "miss", "shortfall", "decline",
    "loss", "writedown", "impairment",
    # Reputation
    "scandal", "controversy", "allegation", "misconduct", "harassment",
    "breach", "hack", "leak", "privacy",
    # Analyst Actions
    "downgrade", "sell rating", "underperform", "reduce", "cut",
]

# =============================================================================
# Agent State
# =============================================================================


class LLMAnalysisResult(TypedDict):
    """Result of LLM analysis for a single news item."""

    title: str
    url: str
    source: str
    published: str
    matched_keywords: list[str]
    is_negative: bool
    category: str | None
    severity: str | None
    summary: str | None
    error: str | None


class WatcherState(TypedDict, total=False):
    """State for the negative info watcher agent."""

    symbol: str
    company_name: str
    all_news: list[CompanyNews]
    filtered_news: list[CompanyNews]
    keyword_matches: dict[str, list[str]]  # url -> matched keywords
    llm_analysis_results: list[LLMAnalysisResult]  # All LLM analysis results
    negative_info: list[NegativeInfo]
    risk_summary: str | None
    saved_path: str | None
    log_path: str | None
    error: str | None


# =============================================================================
# LLM Prompts
# =============================================================================

FILTER_SYSTEM_PROMPT = """You are a risk analyst specializing in identifying negative information about companies.

Analyze MULTIPLE news articles and determine if each contains NEGATIVE information that could:
1. Impact the company's stock price
2. Indicate legal or regulatory risk
3. Suggest business problems or failures
4. Damage the company's reputation

Respond with a JSON array, one object per article in the same order:
[
    {
        "id": 1,
        "is_negative": true/false,
        "category": "lawsuit|recall|investigation|earnings_miss|downgrade|scandal|layoff|regulatory|other|null",
        "severity": "low|medium|high|critical|null",
        "summary": "One sentence summary (required even if not negative)"
    },
    ...
]

Categories (use null if not negative):
- lawsuit: Legal actions, litigation
- recall: Product recalls, safety issues
- investigation: Regulatory or criminal investigations
- earnings_miss: Missed earnings, lowered guidance
- downgrade: Analyst downgrades, sell ratings
- scandal: Executive misconduct, fraud allegations
- layoff: Layoffs, restructuring
- regulatory: Regulatory fines, violations
- other: Other negative news

Severity levels (use null if not negative):
- low: Minor issue, limited impact expected
- medium: Significant issue, may affect stock
- high: Major issue, likely to significantly impact stock
- critical: Severe issue, potential existential threat

IMPORTANT: Return ONLY a valid JSON array. No additional text.
"""

# Batch size for LLM analysis (to reduce API calls)
LLM_BATCH_SIZE = 10

SUMMARY_SYSTEM_PROMPT = """You are a risk analyst providing executive summaries.

Given a list of negative information about a company, provide a concise risk assessment summary.

Consider:
1. Overall risk level (low/medium/high/critical)
2. Most concerning issues
3. Potential financial impact
4. Recommended monitoring actions

Keep the summary to 2-3 sentences. Be direct and factual.
"""


# =============================================================================
# Agent Nodes
# =============================================================================


async def fetch_company_info(state: WatcherState) -> WatcherState:
    """Fetch company information and news."""
    symbol = state["symbol"]
    logger.info(f"Fetching company info for {symbol}...")
    print(f"Fetching company info for {symbol}...")

    try:
        # Get company name from yfinance
        ticker = yf.Ticker(symbol)
        info = ticker.info
        state["company_name"] = info.get("shortName") or info.get("longName") or symbol

        logger.info(f"Company: {state['company_name']}")
        print(f"Company: {state['company_name']}")

        # Fetch recent news (last 90 days)
        logger.info("Fetching recent news...")
        print("Fetching recent news...")

        # Search with both symbol and company name
        news_items: list[CompanyNews] = []

        for query in [symbol, state["company_name"]]:
            items = await fetch_company_news(
                query=query,
                days_back=90,
                max_results=100,
            )
            news_items.extend(items)
            logger.debug(f"Fetched {len(items)} items for query: {query}")

        # Deduplicate by URL
        seen_urls: set[str] = set()
        unique_news: list[CompanyNews] = []
        for item in news_items:
            if item.url not in seen_urls:
                seen_urls.add(item.url)
                unique_news.append(item)

        state["all_news"] = unique_news
        logger.info(f"Total: {len(unique_news)} news items found (deduplicated)")
        print(f"Total: {len(unique_news)} news items found")

    except Exception as e:
        logger.error(f"Failed to fetch company info: {e}")
        state["error"] = f"Failed to fetch company info: {e}"
        state["all_news"] = []

    return state


async def pre_filter_news(state: WatcherState) -> WatcherState:
    """Pre-filter news using keyword matching."""
    if state.get("error") or not state.get("all_news"):
        state["filtered_news"] = []
        return state

    logger.info("Pre-filtering news for negative keywords...")
    print("Pre-filtering news for negative keywords...")

    filtered: list[CompanyNews] = []
    keyword_matches: dict[str, list[str]] = {}  # url -> matched keywords

    for news in state["all_news"]:
        text = f"{news.title} {news.snippet}".lower()
        matched = []

        # Check for negative keywords
        for keyword in NEGATIVE_KEYWORDS:
            if keyword.lower() in text:
                matched.append(keyword)

        if matched:
            filtered.append(news)
            keyword_matches[news.url] = matched
            logger.debug(f"Pre-filtered: {news.title[:60]}... | Keywords: {matched}")

    state["filtered_news"] = filtered
    state["keyword_matches"] = keyword_matches
    logger.info(f"Pre-filtered: {len(filtered)} potentially negative items")
    print(f"Pre-filtered: {len(filtered)} potentially negative items")
    return state


def _extract_text_from_response(content: str | list) -> str:
    """Extract text from LLM response (handles gemini-3-flash-preview list format)."""
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                text_parts.append(item["text"])
            elif isinstance(item, str):
                text_parts.append(item)
        return "".join(text_parts)
    return content if isinstance(content, str) else ""


async def _analyze_batch(
    llm: ChatGoogleGenerativeAI,
    news_batch: list[CompanyNews],
    keyword_matches: dict[str, list[str]],
) -> tuple[list[LLMAnalysisResult], list[NegativeInfo]]:
    """Analyze a batch of news items with a single LLM call."""
    llm_results: list[LLMAnalysisResult] = []
    negative_items: list[NegativeInfo] = []

    # Build batch prompt
    articles_text = "\n\n".join(
        f"[Article {i + 1}]\nTitle: {news.title}\nContent: {news.snippet}"
        for i, news in enumerate(news_batch)
    )

    messages = [
        SystemMessage(content=FILTER_SYSTEM_PROMPT),
        HumanMessage(content=f"Analyze these {len(news_batch)} articles:\n\n{articles_text}"),
    ]

    try:
        response = await llm.ainvoke(messages)
        result_text = _extract_text_from_response(response.content)

        # Parse JSON array response
        start = result_text.find("[")
        end = result_text.rfind("]") + 1
        if start != -1 and end > start:
            results = json.loads(result_text[start:end])

            for i, news in enumerate(news_batch):
                matched_kw = keyword_matches.get(news.url, [])

                # Find matching result (by id or index)
                result = None
                if i < len(results):
                    result = results[i]

                if result:
                    is_negative = result.get("is_negative", False)
                    category = result.get("category")
                    severity = result.get("severity")
                    summary = result.get("summary")

                    llm_results.append(
                        LLMAnalysisResult(
                            title=news.title,
                            url=news.url,
                            source=news.source,
                            published=news.published.isoformat(),
                            matched_keywords=matched_kw,
                            is_negative=is_negative,
                            category=category,
                            severity=severity,
                            summary=summary,
                            error=None,
                        )
                    )

                    if is_negative:
                        negative_info = NegativeInfo(
                            category=category or "other",
                            severity=severity or "medium",
                            title=news.title,
                            description=summary or news.snippet,
                            source=news.source,
                            url=news.url,
                            published=news.published,
                            potential_impact=summary or "",
                        )
                        negative_items.append(negative_info)

                        severity_emoji = {
                            "low": "🟡",
                            "medium": "🟠",
                            "high": "🔴",
                            "critical": "⚫",
                        }.get(negative_info.severity, "⚪")

                        logger.info(f"NEGATIVE: [{severity}] [{category}] {news.title}")
                        print(f"  {severity_emoji} [{category}] {news.title[:50]}...")
                    else:
                        logger.debug(f"NOT NEGATIVE: {news.title[:60]}... | Reason: {summary}")
                else:
                    # No result for this item
                    llm_results.append(
                        LLMAnalysisResult(
                            title=news.title,
                            url=news.url,
                            source=news.source,
                            published=news.published.isoformat(),
                            matched_keywords=matched_kw,
                            is_negative=False,
                            category=None,
                            severity=None,
                            summary=None,
                            error="No result in batch response",
                        )
                    )

    except Exception as e:
        logger.error(f"Error analyzing batch: {e}")
        print(f"  Error analyzing batch: {e}")
        # Record error for all items in batch
        for news in news_batch:
            matched_kw = keyword_matches.get(news.url, [])
            llm_results.append(
                LLMAnalysisResult(
                    title=news.title,
                    url=news.url,
                    source=news.source,
                    published=news.published.isoformat(),
                    matched_keywords=matched_kw,
                    is_negative=False,
                    category=None,
                    severity=None,
                    summary=None,
                    error=str(e),
                )
            )

    return llm_results, negative_items


async def analyze_negative_news(state: WatcherState) -> WatcherState:
    """Use LLM to analyze and categorize negative news (batch processing)."""
    if state.get("error"):
        state["negative_info"] = []
        state["llm_analysis_results"] = []
        return state

    # If no filtered news, nothing to analyze
    if not state.get("filtered_news"):
        logger.info("No potentially negative news to analyze")
        print("No potentially negative news to analyze")
        state["negative_info"] = []
        state["llm_analysis_results"] = []
        return state

    filtered_news = state["filtered_news"]
    keyword_matches = state.get("keyword_matches", {})
    llm_results: list[LLMAnalysisResult] = []
    negative_items: list[NegativeInfo] = []

    # Calculate number of API calls needed
    num_batches = (len(filtered_news) + LLM_BATCH_SIZE - 1) // LLM_BATCH_SIZE
    logger.info(f"Analyzing {len(filtered_news)} items in {num_batches} batch(es)...")
    print(f"Analyzing negative news with LLM ({num_batches} API call(s))...")

    if not GEMINI_API_KEY:
        logger.warning("No API key, using keyword-based analysis only")
        print("Warning: No API key, using keyword-based analysis only")
        for news in filtered_news[:10]:
            llm_results.append(
                LLMAnalysisResult(
                    title=news.title,
                    url=news.url,
                    source=news.source,
                    published=news.published.isoformat(),
                    matched_keywords=keyword_matches.get(news.url, []),
                    is_negative=True,
                    category="other",
                    severity="medium",
                    summary="Analysis unavailable (no API key)",
                    error=None,
                )
            )
            negative_items.append(
                NegativeInfo(
                    category="other",
                    severity="medium",
                    title=news.title,
                    description=news.snippet,
                    source=news.source,
                    url=news.url,
                    published=news.published,
                    potential_impact="Analysis unavailable (no API key)",
                )
            )
        state["llm_analysis_results"] = llm_results
        state["negative_info"] = negative_items
        return state

    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GEMINI_API_KEY)

    # Process in batches
    for batch_idx in range(0, len(filtered_news), LLM_BATCH_SIZE):
        batch = filtered_news[batch_idx : batch_idx + LLM_BATCH_SIZE]
        batch_num = batch_idx // LLM_BATCH_SIZE + 1
        logger.info(f"Processing batch {batch_num}/{num_batches} ({len(batch)} items)")

        batch_results, batch_negative = await _analyze_batch(llm, batch, keyword_matches)
        llm_results.extend(batch_results)
        negative_items.extend(batch_negative)

    # Sort by severity (critical first)
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    negative_items.sort(key=lambda x: severity_order.get(x.severity, 4))

    state["llm_analysis_results"] = llm_results
    state["negative_info"] = negative_items
    logger.info(f"Analysis complete: {len(negative_items)} negative / {len(llm_results)} total")
    print(f"Found {len(negative_items)} negative information items")
    return state


async def generate_risk_summary(state: WatcherState) -> WatcherState:
    """Generate overall risk summary."""
    if state.get("error") or not state.get("negative_info"):
        state["risk_summary"] = None
        return state

    print("Generating risk summary...")

    if not GEMINI_API_KEY:
        # Generate basic summary
        critical_count = sum(
            1 for n in state["negative_info"] if n.severity == "critical"
        )
        high_count = sum(1 for n in state["negative_info"] if n.severity == "high")

        if critical_count > 0:
            level = "CRITICAL"
        elif high_count > 0:
            level = "HIGH"
        else:
            level = "MODERATE"

        state["risk_summary"] = (
            f"Risk Level: {level}. "
            f"Found {len(state['negative_info'])} negative items "
            f"({critical_count} critical, {high_count} high severity)."
        )
        return state

    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GEMINI_API_KEY)

    try:
        # Prepare context
        issues = "\n".join(
            [
                f"- [{n.severity.upper()}] [{n.category}] {n.title}"
                for n in state["negative_info"][:15]
            ]
        )

        prompt = f"""Company: {state['company_name']} ({state['symbol']})

NEGATIVE INFORMATION FOUND:
{issues}

Provide a brief risk assessment summary."""

        messages = [
            SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = await llm.ainvoke(messages)
        state["risk_summary"] = _extract_text_from_response(response.content)

    except Exception as e:
        print(f"Error generating summary: {e}")
        state["risk_summary"] = f"Summary generation failed: {e}"

    return state


def save_results(state: WatcherState) -> WatcherState:
    """Save results to JSON file."""
    logger.info("Saving results...")
    print("Saving results...")

    # Ensure directory exists
    WATCHER_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Create filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = WATCHER_DATA_DIR / f"negative_info_{state['symbol']}_{timestamp}.json"

    # Convert pre-filtered news to serializable format
    pre_filtered_news = [
        {
            "title": news.title,
            "url": news.url,
            "source": news.source,
            "published": news.published.isoformat(),
            "snippet": news.snippet,
            "matched_keywords": state.get("keyword_matches", {}).get(news.url, []),
        }
        for news in state.get("filtered_news", [])
    ]

    # Convert to serializable format
    data = {
        "symbol": state["symbol"],
        "company_name": state["company_name"],
        "analyzed_at": datetime.now().isoformat(),
        "risk_summary": state.get("risk_summary"),
        "statistics": {
            "total_news_scanned": len(state.get("all_news", [])),
            "pre_filtered_count": len(state.get("filtered_news", [])),
            "llm_analyzed_count": len(state.get("llm_analysis_results", [])),
            "negative_found_count": len(state.get("negative_info", [])),
        },
        "pre_filtered_news": pre_filtered_news,
        "llm_analysis_results": state.get("llm_analysis_results", []),
        "negative_info": [
            info.model_dump(mode="json") for info in state.get("negative_info", [])
        ],
        "log_path": state.get("log_path"),
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    state["saved_path"] = str(filepath)
    logger.info(f"Saved to: {filepath}")
    print(f"Saved to: {filepath}")
    return state


# =============================================================================
# Agent Graph
# =============================================================================


def create_agent() -> StateGraph:
    """Create the negative info watcher agent graph."""
    workflow = StateGraph(WatcherState)

    # Add nodes
    workflow.add_node("fetch_info", fetch_company_info)
    workflow.add_node("pre_filter", pre_filter_news)
    workflow.add_node("analyze", analyze_negative_news)
    workflow.add_node("summarize", generate_risk_summary)
    workflow.add_node("save", save_results)

    # Define edges
    workflow.set_entry_point("fetch_info")
    workflow.add_edge("fetch_info", "pre_filter")
    workflow.add_edge("pre_filter", "analyze")
    workflow.add_edge("analyze", "summarize")
    workflow.add_edge("summarize", "save")
    workflow.add_edge("save", END)

    return workflow.compile()


def _setup_file_logging(symbol: str) -> str:
    """Set up file logging for this run.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Path to log file
    """
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create log file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"negative_info_{symbol}_{timestamp}.log"

    # File handler for detailed logging
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler for info level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info(f"=== Negative Info Watcher: {symbol} ===")
    logger.info(f"Log file: {log_path}")

    return str(log_path)


async def run_agent(symbol: str) -> WatcherState:
    """Run the negative info watcher agent.

    Args:
        symbol: Stock ticker symbol to monitor

    Returns:
        Watcher results
    """
    symbol = symbol.upper()

    # Set up file logging
    log_path = _setup_file_logging(symbol)

    agent = create_agent()

    initial_state: WatcherState = {
        "symbol": symbol,
        "company_name": "",
        "all_news": [],
        "filtered_news": [],
        "keyword_matches": {},
        "llm_analysis_results": [],
        "negative_info": [],
        "risk_summary": None,
        "saved_path": None,
        "log_path": log_path,
        "error": None,
    }

    result = await agent.ainvoke(initial_state)

    logger.info("=== Analysis Complete ===")
    return result


def format_results(state: WatcherState) -> str:
    """Format results for display."""
    if state.get("error"):
        return f"Error: {state['error']}"

    lines = [
        f"\n{'=' * 70}",
        f"Negative Information Report: {state['company_name']} ({state['symbol']})",
        f"{'=' * 70}",
    ]

    # Statistics
    total_news = len(state.get("all_news", []))
    pre_filtered = len(state.get("filtered_news", []))
    llm_analyzed = len(state.get("llm_analysis_results", []))
    negative_found = len(state.get("negative_info", []))

    lines.append(f"\n📈 STATISTICS:")
    lines.append(f"   Total news scanned: {total_news}")
    lines.append(f"   Pre-filtered (keyword match): {pre_filtered}")
    lines.append(f"   LLM analyzed: {llm_analyzed}")
    lines.append(f"   Negative found: {negative_found}")

    if state.get("risk_summary"):
        lines.append(f"\n📊 RISK SUMMARY:")
        lines.append(f"   {state['risk_summary']}")

    if not state.get("negative_info"):
        lines.append("\n✅ No significant negative information found.")
    else:
        lines.append(f"\n⚠️  NEGATIVE INFORMATION ({len(state['negative_info'])} items):")

        # Group by severity
        by_severity: dict[str, list[NegativeInfo]] = {}
        for info in state["negative_info"]:
            by_severity.setdefault(info.severity, []).append(info)

        for severity in ["critical", "high", "medium", "low"]:
            if severity in by_severity:
                emoji = {
                    "critical": "⚫",
                    "high": "🔴",
                    "medium": "🟠",
                    "low": "🟡",
                }.get(severity, "⚪")

                lines.append(f"\n{emoji} {severity.upper()}:")
                for info in by_severity[severity]:
                    lines.append(f"   [{info.category}] {info.title}")
                    lines.append(f"      Source: {info.source}")
                    lines.append(f"      Date: {info.published.strftime('%Y-%m-%d')}")

    lines.append(f"\n📁 OUTPUT FILES:")
    if state.get("saved_path"):
        lines.append(f"   Report: {state['saved_path']}")
    if state.get("log_path"):
        lines.append(f"   Log: {state['log_path']}")

    return "\n".join(lines)
