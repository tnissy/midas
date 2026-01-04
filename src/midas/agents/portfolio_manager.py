"""Portfolio Manager Agent using LangGraph.

This agent analyzes the user's portfolio:
- Updates current prices
- Checks for related news
- Provides risk analysis and recommendations
"""

import json
from datetime import datetime
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from midas.config import extract_llm_text, DATA_DIR, GEMINI_API_KEY, LLM_MODEL
from midas.logging_config import (
    get_agent_logger,
    log_agent_start,
    log_agent_end,
    log_node_start,
    log_node_end,
    log_transition,
)
from midas.models import Portfolio
from midas.tools.portfolio_manager import (
    generate_portfolio_report,
    generate_portfolio_summary,
    load_portfolio,
    save_portfolio,
    update_portfolio_prices,
)
from midas.tools.report_generator import (
    save_report,
    generate_portfolio_report as generate_portfolio_md_report,
)

# Logger setup
logger = get_agent_logger("portfolio_manager")

# Path for foresight analysis results
PREDICTION_ANALYSIS_DIR = DATA_DIR / "prediction_analysis"

# =============================================================================
# Paths
# =============================================================================

ANALYSIS_DIR = DATA_DIR / "portfolio"

# =============================================================================
# Agent State
# =============================================================================


class AgentState(TypedDict):
    """State for the portfolio analyzer agent."""

    portfolio: Portfolio | None
    price_updated: bool
    analysis: str | None
    recommendations: list[str]
    report_path: str | None
    error: str | None


# =============================================================================
# LLM Analysis Prompt
# =============================================================================

ANALYSIS_SYSTEM_PROMPT = """You are a professional investment analyst helping individual investors manage their portfolio.

Your task is to analyze the portfolio and provide:
1. Overall portfolio health assessment
2. Concentration risk (if any single stock is >20% of portfolio)
3. Sector diversification analysis
4. Specific recommendations for each holding
5. Action items (buy more, hold, or consider selling)

Be objective and factual. Focus on:
- Unrealized gains/losses and their significance
- Risk factors for individual holdings
- Portfolio balance and diversification

Respond in JSON format:
{
    "overall_assessment": "Brief overall assessment of portfolio health",
    "concentration_risk": "Analysis of concentration risk",
    "diversification": "Analysis of sector/industry diversification",
    "holding_analysis": [
        {
            "symbol": "stock symbol",
            "assessment": "brief assessment",
            "recommendation": "buy_more/hold/reduce/sell",
            "reason": "reason for recommendation"
        }
    ],
    "action_items": ["list of specific action items"],
    "risk_level": "low/medium/high"
}

Respond in Japanese.
"""

# =============================================================================
# Agent Nodes
# =============================================================================


async def load_portfolio_node(state: AgentState) -> AgentState:
    """Load portfolio from file."""
    log_node_start(logger, "load")
    logger.info("Loading portfolio...")
    try:
        portfolio = load_portfolio()
        if not portfolio.holdings:
            state["error"] = "No holdings found in portfolio"
            state["portfolio"] = None
        else:
            state["portfolio"] = portfolio
            logger.info(f"Loaded {len(portfolio.holdings)} holdings")
    except Exception as e:
        state["error"] = f"Failed to load portfolio: {e}"
        state["portfolio"] = None
    log_node_end(logger, "load")
    log_transition(logger, "load", "update_prices")
    return state


async def update_prices_node(state: AgentState) -> AgentState:
    """Update current prices for all holdings."""
    log_node_start(logger, "update_prices")
    if state.get("error") or not state.get("portfolio"):
        state["price_updated"] = False
        log_node_end(logger, "update_prices")
        log_transition(logger, "update_prices", "analyze")
        return state

    logger.info("Updating prices...")
    try:
        portfolio = update_portfolio_prices(state["portfolio"])
        state["portfolio"] = portfolio
        state["price_updated"] = True
        # Save updated portfolio
        save_portfolio(portfolio)
        logger.info("Prices updated and saved")
    except Exception as e:
        logger.warning(f"Failed to update some prices: {e}")
        state["price_updated"] = False
    log_node_end(logger, "update_prices")
    log_transition(logger, "update_prices", "analyze")
    return state


async def analyze_portfolio_node(state: AgentState) -> AgentState:
    """Analyze portfolio using LLM."""
    log_node_start(logger, "analyze")
    if state.get("error") or not state.get("portfolio"):
        state["analysis"] = None
        state["recommendations"] = []
        log_node_end(logger, "analyze")
        log_transition(logger, "analyze", "save")
        return state

    logger.info("Analyzing portfolio with LLM...")

    if not GEMINI_API_KEY:
        # Generate basic report without LLM
        report = generate_portfolio_report(state["portfolio"])
        state["analysis"] = report
        state["recommendations"] = []
        logger.warning("No API key, skipping LLM analysis")
        return state

    try:
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GEMINI_API_KEY)

        # Prepare portfolio summary for LLM
        summary = generate_portfolio_summary(state["portfolio"])
        content = f"Portfolio Summary:\n{json.dumps(summary, ensure_ascii=False, indent=2)}"

        messages = [
            SystemMessage(content=ANALYSIS_SYSTEM_PROMPT),
            HumanMessage(content=content),
        ]

        response = await llm.ainvoke(messages)
        result_text = extract_llm_text(response.content)

        # Parse JSON response
        if isinstance(result_text, str):
            start = result_text.find("{")
            end = result_text.rfind("}") + 1
            if start != -1 and end > start:
                analysis = json.loads(result_text[start:end])
                state["analysis"] = json.dumps(analysis, ensure_ascii=False, indent=2)
                state["recommendations"] = analysis.get("action_items", [])
            else:
                state["analysis"] = result_text
                state["recommendations"] = []
        else:
            state["analysis"] = str(result_text)
            state["recommendations"] = []

        logger.info("Analysis completed")

    except Exception as e:
        logger.error(f"Error analyzing portfolio: {e}")
        # Fallback to basic report
        report = generate_portfolio_report(state["portfolio"])
        state["analysis"] = report
        state["recommendations"] = []

    log_node_end(logger, "analyze")
    log_transition(logger, "analyze", "save")
    return state


def save_analysis_node(state: AgentState) -> AgentState:
    """Save analysis report to file."""
    log_node_start(logger, "save")
    if not state.get("portfolio"):
        state["report_path"] = None
        log_node_end(logger, "save")
        log_transition(logger, "save", "END")
        return state

    logger.info("Saving analysis report...")

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = ANALYSIS_DIR / f"analysis_{timestamp}.json"

    report_data = {
        "analyzed_at": datetime.now().isoformat(),
        "portfolio_summary": generate_portfolio_summary(state["portfolio"]),
        "text_report": generate_portfolio_report(state["portfolio"]),
        "llm_analysis": state.get("analysis"),
        "recommendations": state.get("recommendations", []),
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    state["report_path"] = str(filepath)
    logger.info(f"Report saved to: {filepath}")

    log_node_end(logger, "save")
    log_transition(logger, "save", "END")
    return state


# =============================================================================
# Simplified Agent Runner (no StateGraph)
# =============================================================================


async def run_agent() -> AgentState:
    """Run the portfolio analyzer agent."""
    log_agent_start(logger, "portfolio_manager")

    state: AgentState = {
        "portfolio": None,
        "price_updated": False,
        "analysis": None,
        "recommendations": [],
        "report_path": None,
        "error": None,
    }

    try:
        # Step 1: Load portfolio
        state = await load_portfolio_node(state)

        # Step 2: Update prices
        state = await update_prices_node(state)

        # Step 3: Analyze
        state = await analyze_portfolio_node(state)

        # Step 4: Save
        state = save_analysis_node(state)

        # Print summary
        if state.get("error"):
            logger.info(f"\nError: {state['error']}")
        else:
            logger.info("\n" + "=" * 60)
            logger.info("Portfolio Analysis Complete")
            logger.info("=" * 60)
            if state.get("portfolio"):
                logger.info(generate_portfolio_report(state["portfolio"]))

            if state.get("recommendations"):
                logger.info("\nRecommendations:")
                for i, rec in enumerate(state["recommendations"], 1):
                    logger.info(f"  {i}. {rec}")

            if state.get("report_path"):
                logger.info(f"\nFull report saved to: {state['report_path']}")

        error = state.get("error")
        log_agent_end(logger, "portfolio_manager", state, error)
        return state

    except Exception as e:
        logger.exception("Unhandled exception in portfolio_manager")
        log_agent_end(logger, "portfolio_manager", None, str(e))
        raise


# =============================================================================
# Report Generation
# =============================================================================


def generate_markdown_report() -> str | None:
    """Generate a Markdown report from current portfolio.

    Returns:
        Path to the generated report, or None if no portfolio
    """
    try:
        portfolio = load_portfolio()
    except Exception as e:
        logger.error(f"Error loading portfolio: {e}")
        return None

    if not portfolio.holdings:
        logger.info("No holdings in portfolio. Add holdings first.")
        return None

    # Update prices
    try:
        portfolio = update_portfolio_prices(portfolio)
        save_portfolio(portfolio)
    except Exception as e:
        logger.warning(f"Could not update prices: {e}")

    # Convert holdings to dicts
    holdings_dicts = []
    for h in portfolio.holdings:
        holdings_dicts.append({
            "symbol": h.symbol,
            "name": h.name,
            "shares": h.shares,
            "avg_cost": h.avg_cost,
            "current_price": h.current_price,
        })

    # Try to load latest analysis
    analysis = None
    recommendations = []
    try:
        analysis_files = sorted(ANALYSIS_DIR.glob("analysis_*.json"), reverse=True)
        if analysis_files:
            with open(analysis_files[0], encoding="utf-8") as f:
                data = json.load(f)
                analysis_json = data.get("llm_analysis", "")
                if analysis_json:
                    analysis = json.loads(analysis_json)
                recommendations = data.get("recommendations", [])
    except Exception:
        pass

    # Generate report content
    content = generate_portfolio_md_report(
        holdings=holdings_dicts,
        analysis=analysis,
        recommendations=recommendations,
    )

    # Save report
    report_path = save_report(content, "portfolio")
    logger.info(f"Report saved to: {report_path}")

    return str(report_path)


# =============================================================================
# Buy Candidates from Foresight Analysis
# =============================================================================


def load_buy_candidates() -> list[dict]:
    """Load buy candidates from foresight_to_company_translator results.

    Returns:
        List of company dictionaries with buy candidate information
    """
    candidates = []

    if not PREDICTION_ANALYSIS_DIR.exists():
        logger.info("No prediction analysis results found.")
        return candidates

    # Get most recent analysis files
    analysis_files = sorted(PREDICTION_ANALYSIS_DIR.glob("prediction_*.json"), reverse=True)

    if not analysis_files:
        logger.info("No prediction analysis files found.")
        return candidates

    seen_symbols = set()

    # Load candidates from recent analyses
    for filepath in analysis_files[:5]:  # Last 5 analyses
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)

            prediction = data.get("prediction", "")

            for company in data.get("critical_companies", []):
                symbol = company.get("symbol")
                if not symbol or symbol in seen_symbols:
                    continue

                seen_symbols.add(symbol)
                candidates.append({
                    "symbol": symbol,
                    "name": company.get("name", ""),
                    "exchange": company.get("exchange"),
                    "country": company.get("country", ""),
                    "role": company.get("role", ""),
                    "competitive_advantage": company.get("competitive_advantage", ""),
                    "market_position": company.get("market_position", ""),
                    "confidence": company.get("confidence", 0.5),
                    "layer_name": company.get("layer_name", ""),
                    "source_prediction": prediction[:100],
                })
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            continue

    # Sort by confidence
    candidates.sort(key=lambda x: x.get("confidence", 0), reverse=True)

    return candidates


def get_buy_candidates_not_in_portfolio() -> list[dict]:
    """Get buy candidates that are not already in the portfolio.

    Returns:
        List of company dictionaries for potential purchases
    """
    # Load portfolio
    try:
        portfolio = load_portfolio()
        held_symbols = {h.symbol for h in portfolio.holdings}
    except Exception:
        held_symbols = set()

    # Load all candidates
    all_candidates = load_buy_candidates()

    # Filter out already held
    new_candidates = [c for c in all_candidates if c["symbol"] not in held_symbols]

    logger.info(f"Found {len(new_candidates)} buy candidates not in portfolio")
    for c in new_candidates[:5]:
        logger.info(f"  - {c['symbol']}: {c['name']} ({c['market_position']}, conf: {c['confidence']:.2f})")

    return new_candidates


async def generate_buy_recommendations() -> list[dict]:
    """Generate buy recommendations by analyzing candidates with company_watcher.

    Returns:
        List of recommendation dictionaries
    """
    from midas.agents.company_watcher import analyze_for_buy

    candidates = get_buy_candidates_not_in_portfolio()

    if not candidates:
        logger.info("No buy candidates to analyze.")
        return []

    recommendations = []

    # Analyze top candidates
    for candidate in candidates[:5]:  # Top 5
        symbol = candidate["symbol"]
        logger.info(f"\nAnalyzing {symbol} for buy recommendation...")

        try:
            result = await analyze_for_buy(symbol)

            if result.get("analysis"):
                analysis = result["analysis"]
                rec = analysis.get("buy_recommendation", "hold")

                if rec in ("strong_buy", "buy"):
                    recommendations.append({
                        "symbol": symbol,
                        "name": candidate["name"],
                        "recommendation": rec,
                        "foresight_relevance": analysis.get("foresight_relevance_pct", 0),
                        "competitive_advantage": analysis.get("competitive_advantage", "unknown"),
                        "valuation": analysis.get("valuation", "unknown"),
                        "summary": analysis.get("summary", ""),
                        "source": candidate.get("source_prediction", ""),
                    })
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            continue

    # Sort by foresight relevance
    recommendations.sort(key=lambda x: x.get("foresight_relevance", 0), reverse=True)

    logger.info(f"\n=== Buy Recommendations ({len(recommendations)}) ===")
    for r in recommendations:
        logger.info(f"  {r['recommendation'].upper()}: {r['symbol']} - {r['name']}")
        logger.info(f"    Foresight Relevance: {r['foresight_relevance']}%")
        logger.info(f"    Competitive Advantage: {r['competitive_advantage']}")

    # Save recommendations
    if recommendations:
        ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = ANALYSIS_DIR / f"buy_recommendations_{timestamp}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "generated_at": datetime.now().isoformat(),
                "recommendations": recommendations,
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved to: {filepath}")

    return recommendations
