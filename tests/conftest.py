"""Shared test fixtures for Midas tests."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock
from midas.models import CompanyNews, RiskInfo


@pytest.fixture(autouse=True)
def mock_logger(monkeypatch):
    """Mock the logger for all tests to avoid NoneType errors."""
    mock_log = MagicMock()
    # Patch the logger in company_watcher module
    import midas.agents.company_watcher as cw
    monkeypatch.setattr(cw, "logger", mock_log)
    return mock_log


@pytest.fixture
def sample_risk_news():
    """Sample news with risk keywords."""
    return CompanyNews(
        title="Company sued for fraud allegations",
        source="Reuters",
        url="https://example.com/news/lawsuit",
        published=datetime.now() - timedelta(days=1),
        snippet="The company faces a major lawsuit over fraud allegations...",
    )


@pytest.fixture
def sample_neutral_news():
    """Sample neutral news without risk keywords."""
    return CompanyNews(
        title="Company releases quarterly earnings report",
        source="TechCrunch",
        url="https://example.com/news/earnings",
        published=datetime.now() - timedelta(days=2),
        snippet="The company reported strong quarterly results...",
    )


@pytest.fixture
def sample_downgrade_news():
    """Sample news about analyst downgrade."""
    return CompanyNews(
        title="Analyst downgrades stock to sell rating",
        source="Bloomberg",
        url="https://example.com/news/downgrade",
        published=datetime.now() - timedelta(hours=12),
        snippet="Major analyst firm downgrades the company citing valuation concerns...",
    )


@pytest.fixture
def sample_recall_news():
    """Sample news about product recall."""
    return CompanyNews(
        title="Company announces major product recall due to safety issues",
        source="WSJ",
        url="https://example.com/news/recall",
        published=datetime.now() - timedelta(hours=6),
        snippet="Safety defects discovered in flagship product...",
    )


@pytest.fixture
def sample_risk_info():
    """Sample RiskInfo object."""
    return RiskInfo(
        category="lawsuit",
        severity="high",
        title="Major fraud lawsuit filed",
        description="The company is facing a class action lawsuit for alleged fraud",
        source="Reuters",
        url="https://example.com/news/lawsuit",
        published=datetime.now() - timedelta(days=1),
        potential_impact="Could result in significant financial penalties and reputation damage",
    )


@pytest.fixture
def mock_llm_response_negative():
    """Mock LLM response for negative news analysis."""
    return '''[
        {
            "id": 1,
            "is_negative": true,
            "category": "lawsuit",
            "severity": "high",
            "summary": "Company faces serious fraud allegations with potential major financial impact"
        }
    ]'''


@pytest.fixture
def mock_llm_response_neutral():
    """Mock LLM response for neutral news analysis."""
    return '''[
        {
            "id": 1,
            "is_negative": false,
            "category": null,
            "severity": null,
            "summary": "Routine earnings report with positive results"
        }
    ]'''


@pytest.fixture
def mock_llm_response_batch():
    """Mock LLM response for batch analysis (multiple items)."""
    return '''[
        {
            "id": 1,
            "is_negative": true,
            "category": "lawsuit",
            "severity": "high",
            "summary": "Fraud lawsuit with major implications"
        },
        {
            "id": 2,
            "is_negative": false,
            "category": null,
            "severity": null,
            "summary": "Normal business announcement"
        },
        {
            "id": 3,
            "is_negative": true,
            "category": "downgrade",
            "severity": "medium",
            "summary": "Analyst downgrade due to valuation concerns"
        }
    ]'''
