"""Tests for company_watcher agent."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from midas.agents.company_watcher import (
    WatcherState,
    pre_filter_news,
    analyze_risk_news,
    save_results,
    create_agent,
    RISK_KEYWORDS,
    _extract_text_from_response,
    _analyze_batch,
)
from midas.models import CompanyNews, RiskInfo


# =============================================================================
# Utility Function Tests
# =============================================================================


def test_extract_text_from_response_with_string():
    """Test extracting text from string response."""
    content = "This is a simple string"
    result = _extract_text_from_response(content)
    assert result == "This is a simple string"


def test_extract_text_from_response_with_list():
    """Test extracting text from list response (Gemini format)."""
    content = [
        {"text": "First part "},
        {"text": "second part"},
    ]
    result = _extract_text_from_response(content)
    assert result == "First part second part"


def test_extract_text_from_response_with_mixed_list():
    """Test extracting text from mixed list."""
    content = [
        "Plain string",
        {"text": " and dict"},
    ]
    result = _extract_text_from_response(content)
    assert result == "Plain string and dict"


# =============================================================================
# Node Function Tests
# =============================================================================


@pytest.mark.asyncio
async def test_pre_filter_news_finds_risk_keywords(sample_risk_news, sample_neutral_news):
    """Test that pre_filter correctly identifies news with risk keywords."""
    state: WatcherState = {
        "symbol": "TEST",
        "company_name": "Test Corp",
        "all_news": [sample_risk_news, sample_neutral_news],
        "filtered_news": [],
        "keyword_matches": {},
        "llm_analysis_results": [],
        "risk_info": [],
        "risk_summary": None,
        "saved_path": None,
        "log_path": None,
        "error": None,
    }

    result = await pre_filter_news(state)

    # Only the risk news should be filtered
    assert len(result["filtered_news"]) == 1
    assert result["filtered_news"][0].title == sample_risk_news.title

    # Check keyword matches
    assert sample_risk_news.url in result["keyword_matches"]
    matched_keywords = result["keyword_matches"][sample_risk_news.url]
    assert "sued" in matched_keywords
    assert "fraud" in matched_keywords


@pytest.mark.asyncio
async def test_pre_filter_news_with_multiple_risk_items(
    sample_risk_news, sample_downgrade_news, sample_recall_news, sample_neutral_news
):
    """Test pre-filtering with multiple risk items."""
    state: WatcherState = {
        "symbol": "TEST",
        "company_name": "Test Corp",
        "all_news": [
            sample_risk_news,
            sample_neutral_news,
            sample_downgrade_news,
            sample_recall_news,
        ],
        "filtered_news": [],
        "keyword_matches": {},
        "llm_analysis_results": [],
        "risk_info": [],
        "risk_summary": None,
        "saved_path": None,
        "log_path": None,
        "error": None,
    }

    result = await pre_filter_news(state)

    # Should find 3 risk items (not the neutral one)
    assert len(result["filtered_news"]) == 3

    # All should have keyword matches
    assert len(result["keyword_matches"]) == 3
    assert "downgrade" in result["keyword_matches"][sample_downgrade_news.url]
    assert "recall" in result["keyword_matches"][sample_recall_news.url]


@pytest.mark.asyncio
async def test_pre_filter_news_with_empty_input():
    """Test pre-filtering with no news."""
    state: WatcherState = {
        "symbol": "TEST",
        "company_name": "Test Corp",
        "all_news": [],
        "filtered_news": [],
        "keyword_matches": {},
        "llm_analysis_results": [],
        "risk_info": [],
        "risk_summary": None,
        "saved_path": None,
        "log_path": None,
        "error": None,
    }

    result = await pre_filter_news(state)

    assert len(result["filtered_news"]) == 0
    assert len(result["keyword_matches"]) == 0


@pytest.mark.asyncio
async def test_pre_filter_news_with_error():
    """Test pre-filtering when there's an error in state."""
    state: WatcherState = {
        "symbol": "TEST",
        "company_name": "Test Corp",
        "all_news": [],
        "filtered_news": [],
        "keyword_matches": {},
        "llm_analysis_results": [],
        "risk_info": [],
        "risk_summary": None,
        "saved_path": None,
        "log_path": None,
        "error": "Previous error occurred",
    }

    result = await pre_filter_news(state)

    # Should return empty list when there's an error
    assert len(result["filtered_news"]) == 0


# =============================================================================
# LLM Analysis Tests (with mocking)
# =============================================================================


@pytest.mark.asyncio
async def test_analyze_batch_with_negative_news(
    sample_risk_news, mock_llm_response_negative
):
    """Test batch analysis with negative news."""
    # Create mock LLM
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = mock_llm_response_negative
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    keyword_matches = {sample_risk_news.url: ["sued", "fraud"]}

    llm_results, risk_items = await _analyze_batch(
        mock_llm, [sample_risk_news], keyword_matches
    )

    # Should identify as negative
    assert len(llm_results) == 1
    assert llm_results[0]["is_negative"] is True
    assert llm_results[0]["category"] == "lawsuit"
    assert llm_results[0]["severity"] == "high"

    # Should create a RiskInfo
    assert len(risk_items) == 1
    assert risk_items[0].category == "lawsuit"
    assert risk_items[0].severity == "high"


@pytest.mark.asyncio
async def test_analyze_batch_with_neutral_news(
    sample_neutral_news, mock_llm_response_neutral
):
    """Test batch analysis with neutral news."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = mock_llm_response_neutral
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    keyword_matches = {}

    llm_results, risk_items = await _analyze_batch(
        mock_llm, [sample_neutral_news], keyword_matches
    )

    # Should identify as neutral
    assert len(llm_results) == 1
    assert llm_results[0]["is_negative"] is False

    # Should not create any RiskInfo
    assert len(risk_items) == 0


@pytest.mark.asyncio
async def test_analyze_batch_with_multiple_items(
    sample_risk_news,
    sample_neutral_news,
    sample_downgrade_news,
    mock_llm_response_batch,
):
    """Test batch analysis with multiple news items."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = mock_llm_response_batch
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    news_batch = [sample_risk_news, sample_neutral_news, sample_downgrade_news]
    keyword_matches = {
        sample_risk_news.url: ["sued", "fraud"],
        sample_downgrade_news.url: ["downgrade"],
    }

    llm_results, risk_items = await _analyze_batch(mock_llm, news_batch, keyword_matches)

    # Should process all 3 items
    assert len(llm_results) == 3

    # Should find 2 negative items
    assert len(risk_items) == 2
    assert risk_items[0].category == "lawsuit"
    assert risk_items[1].category == "downgrade"


@pytest.mark.asyncio
async def test_analyze_batch_with_llm_error(sample_risk_news):
    """Test batch analysis when LLM throws an error."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("API Error"))

    keyword_matches = {sample_risk_news.url: ["sued"]}

    llm_results, risk_items = await _analyze_batch(
        mock_llm, [sample_risk_news], keyword_matches
    )

    # Should return results with error
    assert len(llm_results) == 1
    assert llm_results[0]["error"] == "API Error"
    assert llm_results[0]["is_negative"] is False

    # Should not create any RiskInfo on error
    assert len(risk_items) == 0


@pytest.mark.asyncio
async def test_analyze_risk_news_without_api_key(sample_risk_news):
    """Test analyze_risk_news when no API key is available."""
    with patch("midas.agents.company_watcher.GEMINI_API_KEY", None):
        state: WatcherState = {
            "symbol": "TEST",
            "company_name": "Test Corp",
            "all_news": [sample_risk_news],
            "filtered_news": [sample_risk_news],
            "keyword_matches": {sample_risk_news.url: ["sued", "fraud"]},
            "llm_analysis_results": [],
            "risk_info": [],
            "risk_summary": None,
            "saved_path": None,
            "log_path": None,
            "error": None,
        }

        result = await analyze_risk_news(state)

        # Should fall back to keyword-based analysis
        assert len(result["llm_analysis_results"]) == 1
        assert result["llm_analysis_results"][0]["category"] == "other"
        assert result["llm_analysis_results"][0]["summary"] == "Analysis unavailable (no API key)"


@pytest.mark.asyncio
async def test_analyze_risk_news_with_no_filtered_news():
    """Test analyze_risk_news when there's no filtered news."""
    state: WatcherState = {
        "symbol": "TEST",
        "company_name": "Test Corp",
        "all_news": [],
        "filtered_news": [],
        "keyword_matches": {},
        "llm_analysis_results": [],
        "risk_info": [],
        "risk_summary": None,
        "saved_path": None,
        "log_path": None,
        "error": None,
    }

    result = await analyze_risk_news(state)

    # Should return empty results
    assert len(result["llm_analysis_results"]) == 0
    assert len(result["risk_info"]) == 0


# =============================================================================
# Save Results Tests
# =============================================================================


def test_save_results(tmp_path, sample_risk_info):
    """Test saving results to file."""
    # Patch the WATCHER_DATA_DIR to use tmp_path
    with patch("midas.agents.company_watcher.WATCHER_DATA_DIR", tmp_path):
        state: WatcherState = {
            "symbol": "TEST",
            "company_name": "Test Corp",
            "all_news": [],
            "filtered_news": [],
            "keyword_matches": {},
            "llm_analysis_results": [],
            "risk_info": [sample_risk_info],
            "risk_summary": "High risk level detected",
            "saved_path": None,
            "log_path": "/tmp/test.log",
            "error": None,
        }

        result = save_results(state)

        # Check that file was saved
        assert result["saved_path"] is not None
        assert "risk_info_TEST_" in result["saved_path"]

        # Verify file exists
        import json
        from pathlib import Path

        saved_file = Path(result["saved_path"])
        assert saved_file.exists()

        # Verify file contents
        with open(saved_file, encoding="utf-8") as f:
            data = json.load(f)

        assert data["symbol"] == "TEST"
        assert data["company_name"] == "Test Corp"
        assert data["risk_summary"] == "High risk level detected"
        assert len(data["risk_info"]) == 1
        assert data["risk_info"][0]["category"] == "lawsuit"


# =============================================================================
# Agent Graph Tests
# =============================================================================


def test_create_agent_graph_structure():
    """Test that the agent graph is created with correct structure."""
    agent = create_agent()

    # Graph should be compiled
    assert agent is not None

    # Test that we can get the graph structure
    # (LangGraph compiled graphs have a graph attribute)
    assert hasattr(agent, "nodes")


@pytest.mark.asyncio
async def test_agent_handles_error_state():
    """Test that agent properly handles error states."""
    agent = create_agent()

    initial_state: WatcherState = {
        "symbol": "TEST",
        "company_name": "Test Corp",
        "all_news": [],
        "filtered_news": [],
        "keyword_matches": {},
        "llm_analysis_results": [],
        "risk_info": [],
        "risk_summary": None,
        "saved_path": None,
        "log_path": None,
        "error": "Simulated error",
    }

    # Even with error, the graph should complete without exceptions
    result = await agent.ainvoke(initial_state)

    # Error should propagate through
    assert result.get("error") == "Simulated error"
    # Should not generate risk info
    assert len(result.get("risk_info", [])) == 0
