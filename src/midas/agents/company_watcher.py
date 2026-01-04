"""Company Watcher - Monitors companies for risk news and issues."""

import json
from datetime import datetime
from typing import TypedDict

import yfinance as yf
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from midas.config import DATA_DIR, GEMINI_API_KEY, LLM_MODEL, extract_llm_text
from midas.logging_config import get_agent_logger, AGENT_LOG_DIR
from midas.models import CompanyNews, RiskInfo, Foresight
from midas.tools.company_news_fetcher import fetch_company_news
from midas.agents.foresight_manager import load_foresights

# =============================================================================
# Logging Setup
# =============================================================================

# Logger will be initialized per-run with symbol suffix
logger = None

# =============================================================================
# Constants
# =============================================================================

WATCHER_DATA_DIR = DATA_DIR / "company_analysis"

# Risk keywords for initial filtering
RISK_KEYWORDS = [
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
    """State for the risk info watcher agent."""

    symbol: str
    company_name: str
    all_news: list[CompanyNews]
    filtered_news: list[CompanyNews]
    keyword_matches: dict[str, list[str]]  # url -> matched keywords
    llm_analysis_results: list[LLMAnalysisResult]  # All LLM analysis results
    risk_info: list[RiskInfo]
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

    logger.info("Pre-filtering news for risk keywords...")
    print("Pre-filtering news for risk keywords...")

    filtered: list[CompanyNews] = []
    keyword_matches: dict[str, list[str]] = {}  # url -> matched keywords

    for news in state["all_news"]:
        text = f"{news.title} {news.snippet}".lower()
        matched = []

        # Check for risk keywords
        for keyword in RISK_KEYWORDS:
            if keyword.lower() in text:
                matched.append(keyword)

        if matched:
            filtered.append(news)
            keyword_matches[news.url] = matched
            logger.debug(f"Pre-filtered: {news.title[:60]}... | Keywords: {matched}")

    state["filtered_news"] = filtered
    state["keyword_matches"] = keyword_matches
    logger.info(f"Pre-filtered: {len(filtered)} potentially risky items")
    print(f"Pre-filtered: {len(filtered)} potentially risky items")
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
) -> tuple[list[LLMAnalysisResult], list[RiskInfo]]:
    """Analyze a batch of news items with a single LLM call."""
    llm_results: list[LLMAnalysisResult] = []
    risk_items: list[RiskInfo] = []

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
                        risk_info = RiskInfo(
                            category=category or "other",
                            severity=severity or "medium",
                            title=news.title,
                            description=summary or news.snippet,
                            source=news.source,
                            url=news.url,
                            published=news.published,
                            potential_impact=summary or "",
                        )
                        risk_items.append(risk_info)

                        severity_emoji = {
                            "low": "🟡",
                            "medium": "🟠",
                            "high": "🔴",
                            "critical": "⚫",
                        }.get(risk_info.severity, "⚪")

                        logger.info(f"RISK: [{severity}] [{category}] {news.title}")
                        print(f"  {severity_emoji} [{category}] {news.title[:50]}...")
                    else:
                        logger.debug(f"NOT RISKY: {news.title[:60]}... | Reason: {summary}")
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

    return llm_results, risk_items


async def analyze_risk_news(state: WatcherState) -> WatcherState:
    """Use LLM to analyze and categorize risk news (batch processing)."""
    if state.get("error"):
        state["risk_info"] = []
        state["llm_analysis_results"] = []
        return state

    # If no filtered news, nothing to analyze
    if not state.get("filtered_news"):
        logger.info("No potentially risky news to analyze")
        print("No potentially risky news to analyze")
        state["risk_info"] = []
        state["llm_analysis_results"] = []
        return state

    filtered_news = state["filtered_news"]
    keyword_matches = state.get("keyword_matches", {})
    llm_results: list[LLMAnalysisResult] = []
    risk_items: list[RiskInfo] = []

    # Calculate number of API calls needed
    num_batches = (len(filtered_news) + LLM_BATCH_SIZE - 1) // LLM_BATCH_SIZE
    logger.info(f"Analyzing {len(filtered_news)} items in {num_batches} batch(es)...")
    print(f"Analyzing risky news with LLM ({num_batches} API call(s))...")

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
            risk_items.append(
                RiskInfo(
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
        state["risk_info"] = risk_items
        return state

    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GEMINI_API_KEY)

    # Process in batches
    for batch_idx in range(0, len(filtered_news), LLM_BATCH_SIZE):
        batch = filtered_news[batch_idx : batch_idx + LLM_BATCH_SIZE]
        batch_num = batch_idx // LLM_BATCH_SIZE + 1
        logger.info(f"Processing batch {batch_num}/{num_batches} ({len(batch)} items)")

        batch_results, batch_risks = await _analyze_batch(llm, batch, keyword_matches)
        llm_results.extend(batch_results)
        risk_items.extend(batch_risks)

    # Sort by severity (critical first)
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    risk_items.sort(key=lambda x: severity_order.get(x.severity, 4))

    state["llm_analysis_results"] = llm_results
    state["risk_info"] = risk_items
    logger.info(f"Analysis complete: {len(risk_items)} risks / {len(llm_results)} total")
    print(f"Found {len(risk_items)} risk information items")
    return state


async def generate_risk_summary(state: WatcherState) -> WatcherState:
    """Generate overall risk summary."""
    if state.get("error") or not state.get("risk_info"):
        state["risk_summary"] = None
        return state

    print("Generating risk summary...")

    if not GEMINI_API_KEY:
        # Generate basic summary
        critical_count = sum(
            1 for n in state["risk_info"] if n.severity == "critical"
        )
        high_count = sum(1 for n in state["risk_info"] if n.severity == "high")

        if critical_count > 0:
            level = "CRITICAL"
        elif high_count > 0:
            level = "HIGH"
        else:
            level = "MODERATE"

        state["risk_summary"] = (
            f"Risk Level: {level}. "
            f"Found {len(state['risk_info'])} risk items "
            f"({critical_count} critical, {high_count} high severity)."
        )
        return state

    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GEMINI_API_KEY)

    try:
        # Prepare context
        issues = "\n".join(
            [
                f"- [{n.severity.upper()}] [{n.category}] {n.title}"
                for n in state["risk_info"][:15]
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
    filepath = WATCHER_DATA_DIR / f"risk_info_{state['symbol']}_{timestamp}.json"

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
            "risk_found_count": len(state.get("risk_info", [])),
        },
        "pre_filtered_news": pre_filtered_news,
        "llm_analysis_results": state.get("llm_analysis_results", []),
        "risk_info": [
            info.model_dump(mode="json") for info in state.get("risk_info", [])
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
# Agent Runner
# =============================================================================


def _setup_logging(symbol: str) -> str:
    """Set up logging for this run.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Path to log file
    """
    global logger
    logger = get_agent_logger("company_watcher", suffix=symbol)

    # Get the log file path from the handler
    log_path = None
    for handler in logger.handlers:
        if hasattr(handler, 'baseFilename'):
            log_path = handler.baseFilename
            break

    return log_path or str(AGENT_LOG_DIR / f"company_watcher_{symbol}.log")


async def run_agent(symbol: str) -> WatcherState:
    """Run the risk info watcher agent.

    Args:
        symbol: Stock ticker symbol to monitor

    Returns:
        Watcher results
    """
    symbol = symbol.upper()

    # Set up logging
    log_path = _setup_logging(symbol)

    state: WatcherState = {
        "symbol": symbol,
        "company_name": "",
        "all_news": [],
        "filtered_news": [],
        "keyword_matches": {},
        "llm_analysis_results": [],
        "risk_info": [],
        "risk_summary": None,
        "saved_path": None,
        "log_path": log_path,
        "error": None,
    }

    # Execute pipeline steps sequentially
    state = await fetch_company_info(state)
    state = await pre_filter_news(state)
    state = await analyze_risk_news(state)
    state = await generate_risk_summary(state)
    state = await save_results(state)

    logger.info("=== Analysis Complete ===")
    return state


def format_results(state: WatcherState) -> str:
    """Format results for display."""
    if state.get("error"):
        return f"Error: {state['error']}"

    lines = [
        f"\n{'=' * 70}",
        f"Risk Information Report: {state['company_name']} ({state['symbol']})",
        f"{'=' * 70}",
    ]

    # Statistics
    total_news = len(state.get("all_news", []))
    pre_filtered = len(state.get("filtered_news", []))
    llm_analyzed = len(state.get("llm_analysis_results", []))
    risk_found = len(state.get("risk_info", []))

    lines.append(f"\n📈 STATISTICS:")
    lines.append(f"   Total news scanned: {total_news}")
    lines.append(f"   Pre-filtered (keyword match): {pre_filtered}")
    lines.append(f"   LLM analyzed: {llm_analyzed}")
    lines.append(f"   Risk found: {risk_found}")

    if state.get("risk_summary"):
        lines.append(f"\n📊 RISK SUMMARY:")
        lines.append(f"   {state['risk_summary']}")

    if not state.get("risk_info"):
        lines.append("\n✅ No significant risk information found.")
    else:
        lines.append(f"\n⚠️  RISK INFORMATION ({len(state['risk_info'])} items):")

        # Group by severity
        by_severity: dict[str, list[RiskInfo]] = {}
        for info in state["risk_info"]:
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


# =============================================================================
# Buy/Hold Analysis
# =============================================================================

BUY_ANALYSIS_PROMPT = """You are an investment analyst evaluating whether to BUY a stock.

Given the company information and current foresights (future predictions), analyze:

1. **Foresight Relevance** (0-100%): What percentage of this company's business is related to the foresights?
2. **Competitive Advantage**: Does this company have a sustainable competitive advantage (moat)?
3. **Management Quality**: Is the management team capable and trustworthy?
4. **Valuation**: Is the stock price reasonable based on PER and growth potential?

Respond in JSON:
{
    "foresight_relevance_pct": 0-100,
    "foresight_relevance_reason": "Which foresights relate to this company and how",
    "competitive_advantage": "strong|moderate|weak|none",
    "competitive_advantage_reason": "Explanation of moat",
    "management_quality": "excellent|good|average|poor|unknown",
    "management_reason": "Explanation",
    "valuation": "undervalued|fair|overvalued|unknown",
    "valuation_reason": "Based on PER and growth",
    "buy_recommendation": "strong_buy|buy|hold|avoid",
    "summary": "Overall assessment in Japanese (2-3 sentences)"
}

Respond in Japanese for summary field.
"""

HOLD_ANALYSIS_PROMPT = """You are an investment analyst evaluating whether to HOLD or SELL a stock you already own.

Given the company information and recent news, analyze:

1. **Thesis Intact**: Is the original investment thesis still valid?
2. **Competitive Position**: Has a competitor gained significant advantage?
3. **Management Changes**: Any concerning leadership changes?
4. **Legal/Regulatory Risk**: Any lawsuits, investigations, or regulatory issues?
5. **Valuation**: Has the stock become significantly overvalued?

Respond in JSON:
{
    "thesis_status": "intact|weakening|broken",
    "thesis_reason": "Explanation",
    "competitive_threat": "none|minor|moderate|severe",
    "competitive_threat_reason": "Explanation",
    "management_concern": true/false,
    "management_concern_reason": "Explanation if true",
    "legal_regulatory_risk": "none|low|medium|high|critical",
    "legal_regulatory_reason": "Explanation",
    "valuation_concern": true/false,
    "valuation_reason": "Is it too expensive now?",
    "hold_recommendation": "strong_hold|hold|reduce|sell",
    "summary": "Overall assessment in Japanese (2-3 sentences)"
}

Respond in Japanese for summary field.
"""


async def analyze_for_buy(symbol: str) -> dict:
    """Analyze a company for potential purchase.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Analysis result dictionary
    """
    symbol = symbol.upper()
    print(f"Analyzing {symbol} for BUY decision...")

    result = {
        "symbol": symbol,
        "analysis_type": "buy",
        "analyzed_at": datetime.now().isoformat(),
        "company_info": {},
        "foresights_used": [],
        "analysis": None,
        "error": None,
    }

    try:
        # Get company info from yfinance
        ticker = yf.Ticker(symbol)
        info = ticker.info

        company_info = {
            "name": info.get("shortName") or info.get("longName") or symbol,
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "business_summary": info.get("longBusinessSummary", "")[:500],
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "peg_ratio": info.get("pegRatio"),
            "price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
        }
        result["company_info"] = company_info
        print(f"  Company: {company_info['name']}")
        print(f"  Sector: {company_info['sector']}")
        print(f"  P/E: {company_info['pe_ratio']}")

        # Load foresights
        foresights = load_foresights()
        foresight_text = ""
        if foresights:
            result["foresights_used"] = [f.title for f in foresights]
            foresight_text = "\n".join([
                f"- {f.title}: {f.description[:200]}..."
                for f in foresights[:5]
            ])
            print(f"  Using {len(foresights)} foresights for analysis")
        else:
            print("  Warning: No foresights available")

        # Run LLM analysis
        if not GEMINI_API_KEY:
            result["error"] = "No API key available"
            return result

        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GEMINI_API_KEY)

        prompt = f"""Company: {company_info['name']} ({symbol})
Sector: {company_info['sector']}
Industry: {company_info['industry']}
Business: {company_info['business_summary']}

Financial Metrics:
- Market Cap: ${company_info['market_cap']:,} if company_info['market_cap'] else 'N/A'
- P/E Ratio: {company_info['pe_ratio']}
- Forward P/E: {company_info['forward_pe']}
- PEG Ratio: {company_info['peg_ratio']}
- Current Price: ${company_info['price']}
- 52-Week Range: ${company_info['52w_low']} - ${company_info['52w_high']}

Current Foresights (Future Predictions):
{foresight_text if foresight_text else 'No foresights available'}

Analyze this company for a potential BUY decision."""

        messages = [
            SystemMessage(content=BUY_ANALYSIS_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = await llm.ainvoke(messages)
        response_text = extract_llm_text(response.content)

        # Parse JSON response
        if isinstance(response_text, str):
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start != -1 and end > start:
                result["analysis"] = json.loads(response_text[start:end])

        print(f"  Recommendation: {result['analysis'].get('buy_recommendation', 'N/A')}")

    except Exception as e:
        print(f"  Error: {e}")
        result["error"] = str(e)

    # Save result
    WATCHER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = WATCHER_DATA_DIR / f"buy_analysis_{symbol}_{timestamp}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  Saved to: {filepath}")

    return result


async def analyze_for_hold(symbol: str) -> dict:
    """Analyze a held company for hold/sell decision.

    This extends the risk info analysis with additional hold-specific checks.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Analysis result dictionary
    """
    symbol = symbol.upper()
    print(f"Analyzing {symbol} for HOLD decision...")

    result = {
        "symbol": symbol,
        "analysis_type": "hold",
        "analyzed_at": datetime.now().isoformat(),
        "company_info": {},
        "risk_info_summary": None,
        "analysis": None,
        "error": None,
    }

    try:
        # First run risk info analysis
        risk_state = await run_agent(symbol)
        result["risk_info_summary"] = {
            "total_news": len(risk_state.get("all_news", [])),
            "risk_count": len(risk_state.get("risk_info", [])),
            "risk_summary": risk_state.get("risk_summary"),
            "critical_issues": [
                n.title for n in risk_state.get("risk_info", [])
                if n.severity in ("critical", "high")
            ][:5],
        }

        # Get company info
        ticker = yf.Ticker(symbol)
        info = ticker.info

        company_info = {
            "name": info.get("shortName") or info.get("longName") or symbol,
            "sector": info.get("sector", "Unknown"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "price_to_52w_high_pct": None,
        }

        # Calculate how close to 52w high
        if company_info["price"] and company_info["52w_high"]:
            company_info["price_to_52w_high_pct"] = (
                company_info["price"] / company_info["52w_high"] * 100
            )

        result["company_info"] = company_info

        # Run LLM analysis
        if not GEMINI_API_KEY:
            result["error"] = "No API key available"
            return result

        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GEMINI_API_KEY)

        # Prepare risk info text
        risk_text = "None found" if not result["risk_info_summary"]["critical_issues"] else "\n".join([
            f"- {issue}" for issue in result["risk_info_summary"]["critical_issues"]
        ])

        prompt = f"""Company: {company_info['name']} ({symbol})
Sector: {company_info['sector']}

Financial Metrics:
- P/E Ratio: {company_info['pe_ratio']}
- Forward P/E: {company_info['forward_pe']}
- Current Price: ${company_info['price']}
- 52-Week High: ${company_info['52w_high']}
- Price vs 52W High: {company_info['price_to_52w_high_pct']:.1f}% if company_info['price_to_52w_high_pct'] else 'N/A'

Recent Risk Information:
{risk_text}

Risk Summary: {result['risk_info_summary']['risk_summary'] or 'No significant risks found'}

Analyze whether to HOLD or SELL this position."""

        messages = [
            SystemMessage(content=HOLD_ANALYSIS_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = await llm.ainvoke(messages)
        response_text = extract_llm_text(response.content)

        # Parse JSON response
        if isinstance(response_text, str):
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start != -1 and end > start:
                result["analysis"] = json.loads(response_text[start:end])

        print(f"  Recommendation: {result['analysis'].get('hold_recommendation', 'N/A')}")

    except Exception as e:
        print(f"  Error: {e}")
        result["error"] = str(e)

    # Save result
    WATCHER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = WATCHER_DATA_DIR / f"hold_analysis_{symbol}_{timestamp}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  Saved to: {filepath}")

    return result
