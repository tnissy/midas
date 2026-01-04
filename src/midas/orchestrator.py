"""Midas Orchestrator - Coordinates all agents in a unified workflow.

This orchestrator manages the execution flow of all Midas agents:
1. News Watchers (parallel) → collect structural news
2. Foresight Manager → generate/update future predictions
3. Foresight-to-Company Translator → identify affected companies
4. Company Watcher → collect company-specific news
5. Price Event Analyzer → analyze price movements
6. Portfolio Manager → analyze portfolio and provide recommendations
7. Prediction Monitor → generate annual predictions (conditional)
8. Model Calibration Agent → evaluate prediction accuracy (conditional)
"""

import asyncio
from datetime import datetime
from typing import TypedDict

from langgraph.graph import END, StateGraph

from midas.config import DATA_DIR
from midas.logging_config import (
    get_agent_logger,
    log_agent_start,
    log_agent_end,
    log_node_start,
    log_node_end,
    log_transition,
)
from midas.models import NewsItem, Foresight, Portfolio

# Import agent modules
from midas.agents import us_gov_watcher, tech_news_watcher, general_news_watcher, other_gov_watcher
from midas.agents import foresight_manager, foresight_to_company_translator
from midas.agents import company_watcher, price_event_analyzer, portfolio_manager

logger = get_agent_logger("orchestrator")

# =============================================================================
# Global State
# =============================================================================


class MidasState(TypedDict):
    """Global state shared across all agents."""

    # News watching
    news_items: list[NewsItem]

    # Foresight generation
    foresights: list[Foresight]

    # Company analysis
    companies: list[str]
    company_news: dict  # company_symbol -> news items

    # Portfolio management
    portfolio: Portfolio | None
    portfolio_analysis: dict | None

    # Metadata
    run_id: str
    started_at: str
    completed_at: str | None
    error: str | None


# =============================================================================
# Orchestrator Nodes
# =============================================================================


async def run_news_watchers(state: MidasState) -> MidasState:
    """Run all news watchers in parallel."""
    log_node_start(logger, "news_watchers")

    logger.info("Running all news watchers in parallel...")

    # Run all watchers concurrently
    results = await asyncio.gather(
        us_gov_watcher.run_agent(),
        tech_news_watcher.run_agent(),
        general_news_watcher.run_agent(),
        other_gov_watcher.run_agent(),
        return_exceptions=True,
    )

    # Collect all filtered news items
    all_news: list[NewsItem] = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Watcher {i} failed: {result}")
            continue
        all_news.extend(result.get("filtered_items", []))

    state["news_items"] = all_news
    logger.info(f"Total structural news collected: {len(all_news)} items")

    log_node_end(logger, "news_watchers")
    log_transition(logger, "news_watchers", "foresight_manager")
    return state


async def run_foresight_manager(state: MidasState) -> MidasState:
    """Generate or update foresights based on news."""
    log_node_start(logger, "foresight_manager")

    logger.info("Running foresight manager...")

    try:
        result = await foresight_manager.run_agent(force_full=False)
        state["foresights"] = result.get("updated_foresights", [])
        logger.info(f"Foresights updated: {len(state['foresights'])} foresights")

    except Exception as e:
        logger.error(f"Foresight manager failed: {e}")
        state["foresights"] = []

    log_node_end(logger, "foresight_manager")
    log_transition(logger, "foresight_manager", "translator")
    return state


async def run_translator(state: MidasState) -> MidasState:
    """Translate foresights to critical companies."""
    log_node_start(logger, "translator")

    logger.info("Running foresight-to-company translator...")

    if not state.get("foresights"):
        logger.info("No foresights to translate")
        state["companies"] = []
        log_node_end(logger, "translator")
        log_transition(logger, "translator", "portfolio_manager")
        return state

    companies_set = set()

    try:
        # Process each foresight (limit to top 5 to avoid API rate limits)
        for foresight in state["foresights"][:5]:
            logger.info(f"Analyzing foresight: {foresight.title[:60]}...")

            try:
                result = await foresight_to_company_translator.run_agent(
                    foresight.description
                )

                # Extract company symbols from the result
                for company in result.get("companies", []):
                    if company.ticker_symbol:
                        companies_set.add(company.ticker_symbol)

            except Exception as e:
                logger.warning(f"Failed to translate foresight: {e}")
                continue

        state["companies"] = list(companies_set)
        logger.info(f"Identified {len(state['companies'])} critical companies")

    except Exception as e:
        logger.error(f"Translator failed: {e}")
        state["companies"] = []

    log_node_end(logger, "translator")
    log_transition(logger, "translator", "company_watcher")
    return state


async def run_company_watcher(state: MidasState) -> MidasState:
    """Watch company-specific risk information."""
    log_node_start(logger, "company_watcher")

    logger.info("Running company watcher...")

    if not state.get("companies"):
        logger.info("No companies to watch")
        state["company_news"] = {}
        log_node_end(logger, "company_watcher")
        log_transition(logger, "company_watcher", "price_event_analyzer")
        return state

    company_news = {}

    try:
        # Watch each company (limit to top 3 to avoid rate limits)
        for symbol in state["companies"][:3]:
            logger.info(f"Watching company: {symbol}")

            try:
                result = await company_watcher.run_agent(symbol)
                # Store news items for this company
                company_news[symbol] = result.get("filtered_items", [])

            except Exception as e:
                logger.warning(f"Failed to watch {symbol}: {e}")
                company_news[symbol] = []
                continue

        state["company_news"] = company_news
        total_news = sum(len(items) for items in company_news.values())
        logger.info(f"Collected company news: {total_news} items across {len(company_news)} companies")

    except Exception as e:
        logger.error(f"Company watcher failed: {e}")
        state["company_news"] = {}

    log_node_end(logger, "company_watcher")
    log_transition(logger, "company_watcher", "price_event_analyzer")
    return state


async def run_price_analyzer(state: MidasState) -> MidasState:
    """Analyze price events for companies."""
    log_node_start(logger, "price_event_analyzer")

    logger.info("Running price event analyzer...")

    if not state.get("companies"):
        logger.info("No companies to analyze")
        log_node_end(logger, "price_event_analyzer")
        log_transition(logger, "price_event_analyzer", "portfolio_manager")
        return state

    try:
        # Analyze each company (limit to top 3)
        for symbol in state["companies"][:3]:
            logger.info(f"Analyzing price events for: {symbol}")

            try:
                result = await price_event_analyzer.run_agent(symbol)
                logger.info(f"  Found {len(result.get('price_events', []))} price events")

            except Exception as e:
                logger.warning(f"Failed to analyze {symbol}: {e}")
                continue

        logger.info("Price analysis completed")

    except Exception as e:
        logger.error(f"Price analyzer failed: {e}")

    log_node_end(logger, "price_event_analyzer")
    log_transition(logger, "price_event_analyzer", "portfolio_manager")
    return state


async def run_portfolio_manager(state: MidasState) -> MidasState:
    """Analyze portfolio and provide recommendations."""
    log_node_start(logger, "portfolio_manager")

    logger.info("Running portfolio manager...")

    try:
        result = await portfolio_manager.run_agent()
        state["portfolio"] = result.get("portfolio")
        state["portfolio_analysis"] = {
            "analysis": result.get("analysis"),
            "recommendations": result.get("recommendations", []),
            "report_path": result.get("report_path"),
        }
        logger.info("Portfolio analysis completed")

    except Exception as e:
        logger.error(f"Portfolio manager failed: {e}")
        state["portfolio_analysis"] = None

    log_node_end(logger, "portfolio_manager")
    log_transition(logger, "portfolio_manager", "END")
    return state


# =============================================================================
# Orchestrator Graph
# =============================================================================


def create_orchestrator() -> StateGraph:
    """Create the main Midas orchestrator graph."""
    workflow = StateGraph(MidasState)

    # Add nodes
    workflow.add_node("news_watchers", run_news_watchers)
    workflow.add_node("foresight_manager", run_foresight_manager)
    workflow.add_node("translator", run_translator)
    workflow.add_node("company_watcher", run_company_watcher)
    workflow.add_node("price_analyzer", run_price_analyzer)
    workflow.add_node("portfolio_manager", run_portfolio_manager)

    # Define flow
    workflow.set_entry_point("news_watchers")
    workflow.add_edge("news_watchers", "foresight_manager")
    workflow.add_edge("foresight_manager", "translator")
    workflow.add_edge("translator", "company_watcher")
    workflow.add_edge("company_watcher", "price_analyzer")
    workflow.add_edge("price_analyzer", "portfolio_manager")
    workflow.add_edge("portfolio_manager", END)

    return workflow.compile()


# =============================================================================
# Main Runner
# =============================================================================


async def run_midas() -> MidasState:
    """Run the complete Midas workflow.

    Returns:
        Final state with all results
    """
    log_agent_start(logger, "midas_orchestrator")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    initial_state: MidasState = {
        "news_items": [],
        "foresights": [],
        "companies": [],
        "company_news": {},
        "portfolio": None,
        "portfolio_analysis": None,
        "run_id": run_id,
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "error": None,
    }

    try:
        orchestrator = create_orchestrator()
        final_state = await orchestrator.ainvoke(initial_state)

        final_state["completed_at"] = datetime.now().isoformat()

        # Summary
        logger.info("=" * 80)
        logger.info("MIDAS ORCHESTRATOR COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Run ID: {final_state['run_id']}")
        logger.info(f"News Items: {len(final_state['news_items'])}")
        logger.info(f"Foresights: {len(final_state['foresights'])}")
        if final_state["portfolio_analysis"]:
            recs = final_state["portfolio_analysis"].get("recommendations", [])
            logger.info(f"Portfolio Recommendations: {len(recs)}")
        logger.info("=" * 80)

        log_agent_end(logger, "midas_orchestrator", final_state, None)
        return final_state

    except Exception as e:
        logger.exception("Midas orchestrator failed")
        initial_state["error"] = str(e)
        initial_state["completed_at"] = datetime.now().isoformat()
        log_agent_end(logger, "midas_orchestrator", None, str(e))
        raise
