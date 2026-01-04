"""Foresight to Company Translator - Identifies companies critical to realizing future predictions.

Uses a multi-step value chain analysis approach:
1. Decompose prediction into value chain layers
2. Find companies for each layer
3. Analyze bottlenecks and investment implications
"""

import json
from datetime import datetime
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from midas.config import DATA_DIR, GEMINI_API_KEY, LLM_MODEL, extract_llm_text
from midas.logging_config import get_agent_logger
from midas.models import (
    CriticalCompany,
    FuturePredictionAnalysis,
    TimeHorizon,
    ValueChainLayer,
    ValueChainLayerType,
)

# Logger setup
logger = get_agent_logger("foresight_to_company_translator")

# =============================================================================
# Constants
# =============================================================================

ANALYSIS_DATA_DIR = DATA_DIR / "prediction_analysis"

# =============================================================================
# Agent State
# =============================================================================


class FinderState(TypedDict):
    """State for the critical company finder agent."""

    prediction: str
    time_horizon: str
    value_chain_layers: list[ValueChainLayer]
    companies: list[CriticalCompany]
    analysis: FuturePredictionAnalysis | None
    saved_path: str | None
    error: str | None


# =============================================================================
# LLM Prompts
# =============================================================================

VALUE_CHAIN_PROMPT = """You are an expert analyst specializing in value chain analysis and industry structure.

Given a future prediction about a technological or social change, decompose it into VALUE CHAIN LAYERS.

Think about what is needed to realize this prediction:
- END_PRODUCT: Final products/services that embody this change
- CORE_COMPONENT: Key components that go into the end products
- MATERIAL: Raw materials, chemicals, rare elements needed
- EQUIPMENT: Manufacturing equipment, tools ("picks and shovels")
- INFRASTRUCTURE: Physical/digital infrastructure required
- SOFTWARE_SERVICE: Software, platforms, services that enable this

For each layer, assess the BOTTLENECK LEVEL:
- critical: Major constraint, limited suppliers, hard to scale
- high: Significant challenges, concentration of expertise
- medium: Some challenges but manageable
- low: Commoditized, many suppliers

Respond in JSON format:
{{
    "time_horizon": "near|medium|long|uncertain",
    "layers": [
        {{
            "name": "Layer name (e.g., 'Lithium-ion battery cells')",
            "layer_type": "end_product|core_component|material|equipment|infrastructure|software_service",
            "description": "What this layer provides and why it matters",
            "key_technologies": ["tech1", "tech2"],
            "bottleneck_level": "low|medium|high|critical",
            "bottleneck_reason": "Why this is/isn't a bottleneck"
        }}
    ]
}}

Identify 5-8 layers covering the full value chain from raw materials to end products.
"""

COMPANY_FINDER_PROMPT = """You are an expert analyst identifying key companies in a specific value chain layer.

VALUE CHAIN LAYER:
- Name: {layer_name}
- Type: {layer_type}
- Description: {layer_description}
- Key Technologies: {key_technologies}
- Bottleneck Level: {bottleneck_level}

For this layer, identify companies that are:
1. Market leaders with proven track records
2. Innovative challengers with promising technology
3. Key suppliers or enablers (especially for high bottleneck layers)
4. Companies with significant competitive advantages

Focus on publicly traded companies when possible, but include important private companies too.

Respond in JSON format:
{{
    "companies": [
        {{
            "name": "Company name",
            "symbol": "Stock ticker (e.g., AAPL, 6758.T) or null if private",
            "exchange": "Exchange (NASDAQ, NYSE, TSE, etc.) or null",
            "country": "Country of headquarters",
            "role": "Specific role in this layer",
            "competitive_advantage": "Why this company has an edge",
            "market_position": "leader|challenger|niche|emerging",
            "confidence": 0.0-1.0
        }}
    ]
}}

Guidelines:
- Identify 3-6 companies for this layer
- More companies for critical/high bottleneck layers
- Be specific about WHY each company matters
- Include both Western and Asian companies where relevant
- Do NOT include generic large tech companies unless specifically relevant
"""

SYNTHESIS_PROMPT = """You are an expert investment analyst synthesizing a value chain analysis.

PREDICTION: {prediction}
TIME HORIZON: {time_horizon}

VALUE CHAIN LAYERS:
{layers_summary}

IDENTIFIED COMPANIES:
{companies_summary}

Based on this analysis, provide:
1. A summary of the investment opportunity
2. Key investment implications (actionable insights)
3. Major risks to consider

Respond in JSON format:
{{
    "analysis_summary": "2-3 paragraph summary focusing on where the real value creation happens",
    "investment_implications": [
        "Specific, actionable implication 1",
        "Specific, actionable implication 2"
    ],
    "risks": [
        "Specific risk 1",
        "Specific risk 2"
    ]
}}

Focus on:
- Which layers offer the best risk/reward
- Where competitive advantages are most durable
- Potential disruptions or substitution risks
"""


# =============================================================================
# Agent Nodes
# =============================================================================


async def decompose_value_chain(state: FinderState) -> FinderState:
    """Step 1: Decompose the prediction into value chain layers."""
    prediction = state["prediction"]
    logger.info(f"Step 1: Decomposing value chain for: {prediction[:60]}...")

    if not GEMINI_API_KEY:
        state["error"] = "No API key configured. Set GEMINI_API_KEY environment variable."
        return state

    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GEMINI_API_KEY)

    try:
        messages = [
            SystemMessage(content=VALUE_CHAIN_PROMPT),
            HumanMessage(content=f"Future Prediction:\n{prediction}"),
        ]

        response = await llm.ainvoke(messages)
        result_text = extract_llm_text(response.content)

        start = result_text.find("{")
        end = result_text.rfind("}") + 1
        if start != -1 and end > start:
            result = json.loads(result_text[start:end])

            layers = []
            for layer_data in result.get("layers", []):
                try:
                    layer_type = ValueChainLayerType(layer_data.get("layer_type", "core_component"))
                except ValueError:
                    layer_type = ValueChainLayerType.CORE_COMPONENT

                layer = ValueChainLayer(
                    name=layer_data.get("name", "Unknown"),
                    layer_type=layer_type,
                    description=layer_data.get("description", ""),
                    key_technologies=layer_data.get("key_technologies", []),
                    bottleneck_level=layer_data.get("bottleneck_level", "medium"),
                    bottleneck_reason=layer_data.get("bottleneck_reason", ""),
                )
                layers.append(layer)

            state["value_chain_layers"] = layers
            state["time_horizon"] = result.get("time_horizon", "medium")
            logger.info(f"  Identified {len(layers)} value chain layers")

    except Exception as e:
        state["error"] = f"Failed to decompose value chain: {e}"

    return state


async def find_companies_per_layer(state: FinderState) -> FinderState:
    """Step 2: Find companies for each value chain layer."""
    if state.get("error") or not state.get("value_chain_layers"):
        return state

    logger.info(f"Step 2: Finding companies for {len(state['value_chain_layers'])} layers...")

    if not GEMINI_API_KEY:
        state["error"] = "No API key configured."
        return state

    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GEMINI_API_KEY)
    all_companies: list[CriticalCompany] = []

    for layer in state["value_chain_layers"]:
        logger.info(f"  Searching: {layer.name} ({layer.layer_type.value})...")

        prompt = COMPANY_FINDER_PROMPT.format(
            layer_name=layer.name,
            layer_type=layer.layer_type.value,
            layer_description=layer.description,
            key_technologies=", ".join(layer.key_technologies),
            bottleneck_level=layer.bottleneck_level,
        )

        try:
            messages = [HumanMessage(content=prompt)]
            response = await llm.ainvoke(messages)
            result_text = extract_llm_text(response.content)

            start = result_text.find("{")
            end = result_text.rfind("}") + 1
            if start != -1 and end > start:
                result = json.loads(result_text[start:end])

                for comp_data in result.get("companies", []):
                    company = CriticalCompany(
                        name=comp_data.get("name", "Unknown"),
                        symbol=comp_data.get("symbol"),
                        exchange=comp_data.get("exchange"),
                        country=comp_data.get("country", ""),
                        role=comp_data.get("role", ""),
                        layer=layer.name,
                        competitive_advantage=comp_data.get("competitive_advantage", ""),
                        market_position=comp_data.get("market_position", "challenger"),
                        confidence=comp_data.get("confidence", 0.5),
                    )
                    all_companies.append(company)

                logger.info(f"    Found {len(result.get('companies', []))} companies")

        except Exception as e:
            logger.error(f"Error: {e}")

    state["companies"] = all_companies
    logger.info(f"  Total: {len(all_companies)} companies across all layers")

    return state


async def synthesize_analysis(state: FinderState) -> FinderState:
    """Step 3: Synthesize the analysis with bottleneck evaluation."""
    if state.get("error"):
        return state

    logger.info("Step 3: Synthesizing analysis...")

    if not GEMINI_API_KEY or not state.get("companies"):
        # Create basic analysis without synthesis
        state["analysis"] = FuturePredictionAnalysis(
            prediction=state["prediction"],
            time_horizon=TimeHorizon(state.get("time_horizon", "medium")),
            value_chain_layers=state.get("value_chain_layers", []),
            critical_companies=state.get("companies", []),
        )
        return state

    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GEMINI_API_KEY)

    # Prepare summaries
    layers_summary = "\n".join([
        f"- {layer.name} [{layer.layer_type.value}] (Bottleneck: {layer.bottleneck_level}): {layer.description}"
        for layer in state["value_chain_layers"]
    ])

    companies_by_layer: dict[str, list[str]] = {}
    for company in state["companies"]:
        if company.layer not in companies_by_layer:
            companies_by_layer[company.layer] = []
        ticker = f" ({company.symbol})" if company.symbol else ""
        companies_by_layer[company.layer].append(
            f"{company.name}{ticker} - {company.market_position}: {company.role}"
        )

    companies_summary = "\n".join([
        f"[{layer}]\n" + "\n".join(f"  - {c}" for c in companies)
        for layer, companies in companies_by_layer.items()
    ])

    prompt = SYNTHESIS_PROMPT.format(
        prediction=state["prediction"],
        time_horizon=state.get("time_horizon", "medium"),
        layers_summary=layers_summary,
        companies_summary=companies_summary,
    )

    try:
        messages = [HumanMessage(content=prompt)]
        response = await llm.ainvoke(messages)
        result_text = extract_llm_text(response.content)

        summary = ""
        implications: list[str] = []
        risks: list[str] = []

        start = result_text.find("{")
        end = result_text.rfind("}") + 1
        if start != -1 and end > start:
            result = json.loads(result_text[start:end])
            summary = result.get("analysis_summary", "")
            implications = result.get("investment_implications", [])
            risks = result.get("risks", [])

        state["analysis"] = FuturePredictionAnalysis(
            prediction=state["prediction"],
            time_horizon=TimeHorizon(state.get("time_horizon", "medium")),
            value_chain_layers=state.get("value_chain_layers", []),
            critical_companies=state.get("companies", []),
            analysis_summary=summary,
            investment_implications=implications,
            risks=risks,
        )
        logger.info("  Analysis complete")

    except Exception as e:
        logger.error(f"Error in synthesis: {e}")
        state["analysis"] = FuturePredictionAnalysis(
            prediction=state["prediction"],
            time_horizon=TimeHorizon(state.get("time_horizon", "medium")),
            value_chain_layers=state.get("value_chain_layers", []),
            critical_companies=state.get("companies", []),
        )

    return state


def save_results(state: FinderState) -> FinderState:
    """Save analysis results to JSON file."""
    if not state.get("analysis"):
        state["saved_path"] = None
        return state

    logger.info("Saving results...")

    ANALYSIS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() else "_" for c in state["prediction"][:30])
    filepath = ANALYSIS_DATA_DIR / f"prediction_{safe_name}_{timestamp}.json"

    analysis = state["analysis"]
    data = {
        "prediction": analysis.prediction,
        "analyzed_at": analysis.analyzed_at.isoformat(),
        "time_horizon": analysis.time_horizon.value,
        "value_chain_layers": [
            {
                "name": layer.name,
                "layer_type": layer.layer_type.value,
                "description": layer.description,
                "key_technologies": layer.key_technologies,
                "bottleneck_level": layer.bottleneck_level,
                "bottleneck_reason": layer.bottleneck_reason,
            }
            for layer in analysis.value_chain_layers
        ],
        "critical_companies": [comp.model_dump() for comp in analysis.critical_companies],
        "analysis_summary": analysis.analysis_summary,
        "investment_implications": analysis.investment_implications,
        "risks": analysis.risks,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    state["saved_path"] = str(filepath)
    logger.info(f"Saved to: {filepath}")
    return state


# =============================================================================
# Agent Graph
# =============================================================================


def create_agent() -> StateGraph:
    """Create the critical company finder agent graph."""
    workflow = StateGraph(FinderState)

    # Add nodes (3-step value chain analysis)
    workflow.add_node("decompose", decompose_value_chain)
    workflow.add_node("find_companies", find_companies_per_layer)
    workflow.add_node("synthesize", synthesize_analysis)
    workflow.add_node("save", save_results)

    # Define edges
    workflow.set_entry_point("decompose")
    workflow.add_edge("decompose", "find_companies")
    workflow.add_edge("find_companies", "synthesize")
    workflow.add_edge("synthesize", "save")
    workflow.add_edge("save", END)

    return workflow.compile()


async def run_agent(prediction: str) -> FinderState:
    """Run the critical company finder agent.

    Args:
        prediction: A future prediction to analyze

    Returns:
        Analysis results including critical companies
    """
    agent = create_agent()

    initial_state: FinderState = {
        "prediction": prediction,
        "time_horizon": "medium",
        "value_chain_layers": [],
        "companies": [],
        "analysis": None,
        "saved_path": None,
        "error": None,
    }

    result = await agent.ainvoke(initial_state)
    return result


def format_analysis(state: FinderState) -> str:
    """Format analysis results for display."""
    if state.get("error"):
        return f"Error: {state['error']}"

    analysis = state.get("analysis")
    if not analysis:
        return "No analysis results available"

    lines = [
        f"\n{'=' * 70}",
        "CRITICAL COMPANY ANALYSIS (Value Chain Approach)",
        f"{'=' * 70}",
        f"\nPrediction: {analysis.prediction}",
        f"Time Horizon: {analysis.time_horizon.value}",
    ]

    if analysis.value_chain_layers:
        lines.append(f"\n{'─' * 40}")
        lines.append("VALUE CHAIN LAYERS")
        lines.append(f"{'─' * 40}")

        # Sort by bottleneck level
        level_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        sorted_layers = sorted(
            analysis.value_chain_layers,
            key=lambda x: level_order.get(x.bottleneck_level, 4)
        )

        for layer in sorted_layers:
            icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(
                layer.bottleneck_level, "⚪"
            )
            type_label = {
                "end_product": "完成品",
                "core_component": "コア部品",
                "material": "素材",
                "equipment": "製造装置",
                "infrastructure": "インフラ",
                "software_service": "ソフト/サービス",
            }.get(layer.layer_type.value, layer.layer_type.value)

            lines.append(f"\n{icon} {layer.name} [{type_label}]")
            lines.append(f"   {layer.description}")
            if layer.bottleneck_reason:
                lines.append(f"   Bottleneck: {layer.bottleneck_reason}")

    if analysis.critical_companies:
        lines.append(f"\n{'─' * 40}")
        lines.append("CRITICAL COMPANIES BY LAYER")
        lines.append(f"{'─' * 40}")

        # Group by layer
        by_layer: dict[str, list[CriticalCompany]] = {}
        for company in analysis.critical_companies:
            if company.layer not in by_layer:
                by_layer[company.layer] = []
            by_layer[company.layer].append(company)

        for layer_name, companies in by_layer.items():
            lines.append(f"\n[{layer_name}]")
            # Sort by confidence
            companies.sort(key=lambda x: x.confidence, reverse=True)
            for company in companies:
                pos_icon = {"leader": "👑", "challenger": "⚔️", "niche": "🎯", "emerging": "🌱"}.get(
                    company.market_position, "•"
                )
                ticker = f" ({company.symbol})" if company.symbol else ""
                lines.append(f"  {pos_icon} {company.name}{ticker} [{company.country}]")
                lines.append(f"     {company.role}")
                if company.competitive_advantage:
                    lines.append(f"     Edge: {company.competitive_advantage}")

    if analysis.analysis_summary:
        lines.append(f"\n{'─' * 40}")
        lines.append("ANALYSIS SUMMARY")
        lines.append(f"{'─' * 40}")
        lines.append(analysis.analysis_summary)

    if analysis.investment_implications:
        lines.append(f"\n{'─' * 40}")
        lines.append("INVESTMENT IMPLICATIONS")
        lines.append(f"{'─' * 40}")
        for impl in analysis.investment_implications:
            lines.append(f"  • {impl}")

    if analysis.risks:
        lines.append(f"\n{'─' * 40}")
        lines.append("RISKS")
        lines.append(f"{'─' * 40}")
        for risk in analysis.risks:
            lines.append(f"  ⚠️ {risk}")

    if state.get("saved_path"):
        lines.append(f"\nFull results saved to: {state['saved_path']}")

    return "\n".join(lines)
