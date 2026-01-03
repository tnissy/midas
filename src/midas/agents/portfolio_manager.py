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
from langgraph.graph import END, StateGraph

from midas.config import extract_llm_text, DATA_DIR, GEMINI_API_KEY, LLM_MODEL
from midas.models import Portfolio
from midas.tools.portfolio_manager import (
    generate_portfolio_report,
    generate_portfolio_summary,
    load_portfolio,
    save_portfolio,
    update_portfolio_prices,
)

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
    print("Loading portfolio...")
    try:
        portfolio = load_portfolio()
        if not portfolio.holdings:
            state["error"] = "No holdings found in portfolio"
            state["portfolio"] = None
        else:
            state["portfolio"] = portfolio
            print(f"Loaded {len(portfolio.holdings)} holdings")
    except Exception as e:
        state["error"] = f"Failed to load portfolio: {e}"
        state["portfolio"] = None
    return state


async def update_prices_node(state: AgentState) -> AgentState:
    """Update current prices for all holdings."""
    if state.get("error") or not state.get("portfolio"):
        state["price_updated"] = False
        return state

    print("Updating prices...")
    try:
        portfolio = update_portfolio_prices(state["portfolio"])
        state["portfolio"] = portfolio
        state["price_updated"] = True
        # Save updated portfolio
        save_portfolio(portfolio)
        print("Prices updated and saved")
    except Exception as e:
        print(f"Warning: Failed to update some prices: {e}")
        state["price_updated"] = False
    return state


async def analyze_portfolio_node(state: AgentState) -> AgentState:
    """Analyze portfolio using LLM."""
    if state.get("error") or not state.get("portfolio"):
        state["analysis"] = None
        state["recommendations"] = []
        return state

    print("Analyzing portfolio with LLM...")

    if not GEMINI_API_KEY:
        # Generate basic report without LLM
        report = generate_portfolio_report(state["portfolio"])
        state["analysis"] = report
        state["recommendations"] = []
        print("Warning: No API key, skipping LLM analysis")
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

        print("Analysis completed")

    except Exception as e:
        print(f"Error analyzing portfolio: {e}")
        # Fallback to basic report
        report = generate_portfolio_report(state["portfolio"])
        state["analysis"] = report
        state["recommendations"] = []

    return state


def save_analysis_node(state: AgentState) -> AgentState:
    """Save analysis report to file."""
    if not state.get("portfolio"):
        state["report_path"] = None
        return state

    print("Saving analysis report...")

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
    print(f"Report saved to: {filepath}")

    return state


# =============================================================================
# Agent Graph
# =============================================================================


def create_agent() -> StateGraph:
    """Create the portfolio analyzer agent graph."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("load", load_portfolio_node)
    workflow.add_node("update_prices", update_prices_node)
    workflow.add_node("analyze", analyze_portfolio_node)
    workflow.add_node("save", save_analysis_node)

    # Define edges
    workflow.set_entry_point("load")
    workflow.add_edge("load", "update_prices")
    workflow.add_edge("update_prices", "analyze")
    workflow.add_edge("analyze", "save")
    workflow.add_edge("save", END)

    return workflow.compile()


async def run_agent() -> AgentState:
    """Run the portfolio analyzer agent."""
    agent = create_agent()

    initial_state: AgentState = {
        "portfolio": None,
        "price_updated": False,
        "analysis": None,
        "recommendations": [],
        "report_path": None,
        "error": None,
    }

    result = await agent.ainvoke(initial_state)

    # Print summary
    if result.get("error"):
        print(f"\nError: {result['error']}")
    else:
        print("\n" + "=" * 60)
        print("Portfolio Analysis Complete")
        print("=" * 60)
        if result.get("portfolio"):
            print(generate_portfolio_report(result["portfolio"]))

        if result.get("recommendations"):
            print("\nRecommendations:")
            for i, rec in enumerate(result["recommendations"], 1):
                print(f"  {i}. {rec}")

        if result.get("report_path"):
            print(f"\nFull report saved to: {result['report_path']}")

    return result
