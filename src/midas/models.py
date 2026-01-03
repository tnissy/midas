"""Data models for Midas."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class NewsCategory(str, Enum):
    """Category of news."""

    LEGISLATION = "legislation"  # 法案
    REGULATION = "regulation"  # 規制
    POLICY = "policy"  # 政策
    EXECUTIVE_ORDER = "executive_order"  # 大統領令
    TRADE = "trade"  # 通商
    TECHNOLOGY = "technology"  # 技術政策
    OTHER = "other"


class NewsItem(BaseModel):
    """A news item from RSS feed."""

    id: str = Field(description="Unique identifier (hash of URL)")
    title: str = Field(description="News title")
    source: str = Field(description="Source name (e.g., 'whitehouse')")
    url: str = Field(description="Original URL")
    published: datetime = Field(description="Published datetime")
    content: str = Field(default="", description="Full content or description")
    summary: str | None = Field(default=None, description="LLM-generated summary")
    is_structural: bool = Field(
        default=False, description="Whether this relates to structural change"
    )
    category: NewsCategory | None = Field(default=None, description="News category")
    relevance_reason: str | None = Field(
        default=None, description="Why this news is relevant to structural change"
    )


class NewsCollection(BaseModel):
    """Collection of news items."""

    items: list[NewsItem] = Field(default_factory=list)
    fetched_at: datetime = Field(default_factory=datetime.now)
    source_count: dict[str, int] = Field(default_factory=dict)


class StockMovement(BaseModel):
    """Stock price movement over a period."""

    symbol: str = Field(description="Stock ticker symbol")
    name: str = Field(description="Company name")
    price_before: float = Field(description="Price at start of period")
    price_now: float = Field(description="Current price")
    change_percent: float = Field(description="Percentage change")
    is_significant: bool = Field(
        default=False, description="Whether this is a significant movement (3x or 1/3)"
    )


# =============================================================================
# Company Analysis Models
# =============================================================================


class PriceEvent(BaseModel):
    """A significant price movement event."""

    date: datetime = Field(description="Date of the price event")
    price_before: float = Field(description="Price before the event")
    price_after: float = Field(description="Price after the event")
    change_percent: float = Field(description="Percentage change")
    volume_ratio: float = Field(default=1.0, description="Volume compared to average")


class CompanyNews(BaseModel):
    """News article related to a company."""

    title: str = Field(description="News title")
    source: str = Field(description="News source/publisher")
    url: str = Field(description="Article URL")
    published: datetime = Field(description="Published datetime")
    snippet: str = Field(default="", description="Article snippet/summary")
    sentiment: str | None = Field(
        default=None, description="Sentiment: positive/negative/neutral"
    )
    relevance_score: float = Field(
        default=0.0, description="Relevance to price movement (0-1)"
    )
    is_first_report: bool = Field(
        default=False, description="Whether this was the first to report"
    )


class PriceEventAnalysis(BaseModel):
    """Analysis of why a price event occurred."""

    event: PriceEvent = Field(description="The price event being analyzed")
    likely_cause: str = Field(description="Most likely cause of the price movement")
    related_news: list[CompanyNews] = Field(
        default_factory=list, description="News articles related to this event"
    )
    first_reporter: str | None = Field(
        default=None, description="Source that first reported the news"
    )
    confidence: float = Field(
        default=0.0, description="Confidence in the analysis (0-1)"
    )


class NegativeInfo(BaseModel):
    """Negative information about a company."""

    category: str = Field(
        description="Category: lawsuit/recall/investigation/earnings_miss/downgrade/scandal/other"
    )
    severity: str = Field(description="Severity: low/medium/high/critical")
    title: str = Field(description="Brief title of the issue")
    description: str = Field(description="Detailed description")
    source: str = Field(description="Information source")
    url: str = Field(description="Source URL")
    published: datetime = Field(description="When it was published")
    potential_impact: str = Field(description="Potential impact on stock/business")


class CompanyAnalysis(BaseModel):
    """Complete analysis of a company."""

    symbol: str = Field(description="Stock ticker symbol")
    name: str = Field(description="Company name")
    analyzed_at: datetime = Field(default_factory=datetime.now)
    price_events: list[PriceEventAnalysis] = Field(
        default_factory=list, description="Significant price events and their causes"
    )
    negative_info: list[NegativeInfo] = Field(
        default_factory=list, description="Current negative information"
    )
    risk_summary: str | None = Field(
        default=None, description="Overall risk assessment summary"
    )


# =============================================================================
# Portfolio Management Models
# =============================================================================


class AccountType(str, Enum):
    """Type of securities account."""

    GENERAL = "general"  # 一般口座
    SPECIFIC = "specific"  # 特定口座
    NISA_GROWTH = "nisa_growth"  # NISA成長投資枠
    NISA_RESERVE = "nisa_reserve"  # NISAつみたて投資枠
    IDECO = "ideco"  # iDeCo


class StockHolding(BaseModel):
    """A stock holding in the portfolio."""

    symbol: str = Field(description="Stock ticker symbol (e.g., '7203' for Toyota)")
    name: str = Field(description="Company name")
    shares: int = Field(description="Number of shares held")
    avg_cost: float = Field(description="Average acquisition cost per share")
    current_price: float | None = Field(
        default=None, description="Current market price per share"
    )
    account_type: AccountType = Field(
        default=AccountType.SPECIFIC, description="Type of account"
    )
    broker: str = Field(default="nomura", description="Broker name")
    acquired_at: datetime | None = Field(
        default=None, description="Date of first acquisition"
    )
    notes: str | None = Field(default=None, description="Additional notes")

    @property
    def total_cost(self) -> float:
        """Total acquisition cost."""
        return self.shares * self.avg_cost

    @property
    def current_value(self) -> float | None:
        """Current market value."""
        if self.current_price is None:
            return None
        return self.shares * self.current_price

    @property
    def unrealized_gain(self) -> float | None:
        """Unrealized gain/loss."""
        if self.current_price is None:
            return None
        return self.current_value - self.total_cost

    @property
    def unrealized_gain_percent(self) -> float | None:
        """Unrealized gain/loss percentage."""
        if self.current_price is None or self.total_cost == 0:
            return None
        return (self.unrealized_gain / self.total_cost) * 100


class Transaction(BaseModel):
    """A stock transaction record."""

    symbol: str = Field(description="Stock ticker symbol")
    name: str = Field(description="Company name")
    transaction_type: str = Field(description="Type: buy/sell")
    shares: int = Field(description="Number of shares")
    price: float = Field(description="Price per share")
    fees: float = Field(default=0.0, description="Transaction fees")
    account_type: AccountType = Field(
        default=AccountType.SPECIFIC, description="Type of account"
    )
    broker: str = Field(default="nomura", description="Broker name")
    executed_at: datetime = Field(description="Execution datetime")
    notes: str | None = Field(default=None, description="Additional notes")

    @property
    def total_amount(self) -> float:
        """Total transaction amount including fees."""
        base = self.shares * self.price
        if self.transaction_type == "buy":
            return base + self.fees
        return base - self.fees


class Portfolio(BaseModel):
    """A portfolio containing multiple stock holdings."""

    name: str = Field(default="Main Portfolio", description="Portfolio name")
    holdings: list[StockHolding] = Field(
        default_factory=list, description="Stock holdings"
    )
    transactions: list[Transaction] = Field(
        default_factory=list, description="Transaction history"
    )
    updated_at: datetime = Field(default_factory=datetime.now)
    cash_balance: float = Field(default=0.0, description="Cash balance in account")

    @property
    def total_cost(self) -> float:
        """Total acquisition cost of all holdings."""
        return sum(h.total_cost for h in self.holdings)

    @property
    def total_value(self) -> float | None:
        """Total current market value."""
        values = [h.current_value for h in self.holdings]
        if None in values:
            return None
        return sum(values)

    @property
    def total_unrealized_gain(self) -> float | None:
        """Total unrealized gain/loss."""
        if self.total_value is None:
            return None
        return self.total_value - self.total_cost

    def get_holding(self, symbol: str) -> StockHolding | None:
        """Get a holding by symbol."""
        for h in self.holdings:
            if h.symbol == symbol:
                return h
        return None


# =============================================================================
# Future Insight Models
# =============================================================================


class SignalCategory(str, Enum):
    """Category of future signal."""

    TECHNOLOGY_SHIFT = "technology_shift"  # 技術転換
    REGULATION_CHANGE = "regulation_change"  # 制度・規制変更
    BEHAVIOR_CHANGE = "behavior_change"  # 行動様式の変化
    GEOPOLITICAL = "geopolitical"  # 地政学的変化
    DEMOGRAPHIC = "demographic"  # 人口動態
    INFRASTRUCTURE = "infrastructure"  # インフラ変化
    PLATFORM_SHIFT = "platform_shift"  # プラットフォーム移行
    OTHER = "other"


class TimeHorizon(str, Enum):
    """Expected time horizon for the change."""

    NEAR = "near"  # 1-2年
    MEDIUM = "medium"  # 3-5年
    LONG = "long"  # 5-10年
    UNCERTAIN = "uncertain"  # 不明


class FutureSignal(BaseModel):
    """A signal indicating potential future change."""

    title: str = Field(description="Brief title of the signal")
    category: SignalCategory = Field(description="Category of the signal")
    description: str = Field(description="Detailed description of the signal")
    source_news: list[str] = Field(
        default_factory=list, description="URLs of source news articles"
    )
    time_horizon: TimeHorizon = Field(
        default=TimeHorizon.UNCERTAIN, description="Expected time horizon"
    )
    confidence: float = Field(
        default=0.5, description="Confidence level (0-1)"
    )
    potential_impact: str = Field(
        default="", description="Potential impact on markets/society"
    )
    detected_at: datetime = Field(default_factory=datetime.now)


class Beneficiary(BaseModel):
    """Entity that may benefit from a structural change."""

    entity_type: str = Field(description="Type: company/sector/technology/region")
    name: str = Field(description="Name of the beneficiary")
    symbol: str | None = Field(default=None, description="Stock symbol if applicable")
    reason: str = Field(description="Why this entity benefits")
    confidence: float = Field(default=0.5, description="Confidence level (0-1)")


class InvestmentTheme(BaseModel):
    """An investment theme derived from structural changes."""

    title: str = Field(description="Theme title")
    thesis: str = Field(description="Investment thesis - why this matters")
    signals: list[FutureSignal] = Field(
        default_factory=list, description="Signals supporting this theme"
    )
    beneficiaries: list[Beneficiary] = Field(
        default_factory=list, description="Potential beneficiaries"
    )
    risks: list[str] = Field(
        default_factory=list, description="Key risks to this thesis"
    )
    time_horizon: TimeHorizon = Field(
        default=TimeHorizon.MEDIUM, description="Investment time horizon"
    )
    conviction: str = Field(
        default="medium", description="Conviction level: low/medium/high"
    )


class FutureInsightReport(BaseModel):
    """A comprehensive future insight report."""

    generated_at: datetime = Field(default_factory=datetime.now)
    period_start: datetime = Field(description="Analysis period start")
    period_end: datetime = Field(description="Analysis period end")
    executive_summary: str = Field(description="Executive summary of insights")
    signals: list[FutureSignal] = Field(
        default_factory=list, description="Detected future signals"
    )
    themes: list[InvestmentTheme] = Field(
        default_factory=list, description="Identified investment themes"
    )
    key_observations: list[str] = Field(
        default_factory=list, description="Key observations and takeaways"
    )
    action_items: list[str] = Field(
        default_factory=list, description="Recommended action items"
    )


# =============================================================================
# Future Prediction Analysis Models
# =============================================================================


class ValueChainLayerType(str, Enum):
    """Type of value chain layer."""

    END_PRODUCT = "end_product"  # 完成品・最終製品
    CORE_COMPONENT = "core_component"  # コア部品
    MATERIAL = "material"  # 素材・原料
    EQUIPMENT = "equipment"  # 製造装置（ツルハシ売り）
    INFRASTRUCTURE = "infrastructure"  # インフラ
    SOFTWARE_SERVICE = "software_service"  # ソフトウェア・サービス


class ValueChainLayer(BaseModel):
    """A layer in the value chain for realizing a future prediction."""

    name: str = Field(description="Layer name (e.g., 'Battery cells')")
    layer_type: ValueChainLayerType = Field(description="Type of layer in value chain")
    description: str = Field(description="What this layer provides and why it matters")
    key_technologies: list[str] = Field(
        default_factory=list, description="Key technologies in this layer"
    )
    bottleneck_level: str = Field(
        default="medium", description="Bottleneck severity: low/medium/high/critical"
    )
    bottleneck_reason: str = Field(
        default="", description="Why this is a bottleneck (supply, tech, regulation, etc.)"
    )


class CriticalComponent(BaseModel):
    """A critical component needed to realize a future prediction."""

    name: str = Field(description="Component name (e.g., 'Solid-state batteries')")
    category: str = Field(
        description="Category: technology/infrastructure/service/material/regulation"
    )
    description: str = Field(description="Why this component is critical")
    current_status: str = Field(
        default="", description="Current development status"
    )
    bottleneck_level: str = Field(
        default="medium", description="Bottleneck severity: low/medium/high/critical"
    )


class CriticalCompany(BaseModel):
    """A company playing a critical role in realizing a future prediction."""

    name: str = Field(description="Company name")
    symbol: str | None = Field(default=None, description="Stock ticker symbol")
    exchange: str | None = Field(default=None, description="Stock exchange")
    country: str = Field(default="", description="Country of headquarters")
    role: str = Field(description="Company's role in the prediction")
    layer: str = Field(description="Which value chain layer this company belongs to")
    competitive_advantage: str = Field(
        default="", description="Why this company has an edge"
    )
    market_position: str = Field(
        default="", description="Market position: leader/challenger/niche/emerging"
    )
    confidence: float = Field(
        default=0.5, description="Confidence in this company's importance (0-1)"
    )
    sources: list[str] = Field(
        default_factory=list, description="Information sources"
    )


class FuturePredictionAnalysis(BaseModel):
    """Complete analysis of companies critical to realizing a future prediction."""

    prediction: str = Field(description="The original future prediction")
    analyzed_at: datetime = Field(default_factory=datetime.now)
    time_horizon: TimeHorizon = Field(
        default=TimeHorizon.MEDIUM, description="Expected time horizon"
    )
    value_chain_layers: list[ValueChainLayer] = Field(
        default_factory=list, description="Value chain layers identified"
    )
    critical_companies: list[CriticalCompany] = Field(
        default_factory=list, description="Companies playing critical roles"
    )
    analysis_summary: str = Field(
        default="", description="Summary of the analysis"
    )
    investment_implications: list[str] = Field(
        default_factory=list, description="Investment implications"
    )
    risks: list[str] = Field(
        default_factory=list, description="Key risks to consider"
    )


# =============================================================================
# Learning Models (株価急変からの学習)
# =============================================================================


class MovementDirection(str, Enum):
    """Direction of price movement."""

    UP = "up"  # 急騰 (3倍以上)
    DOWN = "down"  # 急落 (1/3以下)


class StructuralChangeType(str, Enum):
    """Type of structural change that caused the price movement."""

    TECHNOLOGY_BREAKTHROUGH = "technology_breakthrough"  # 技術的ブレークスルー
    REGULATION_CHANGE = "regulation_change"  # 規制変更
    MARKET_STRUCTURE = "market_structure"  # 市場構造の変化
    COMPETITIVE_DYNAMICS = "competitive_dynamics"  # 競争環境の変化
    BUSINESS_MODEL = "business_model"  # ビジネスモデルの変革
    MANAGEMENT_CHANGE = "management_change"  # 経営陣の変化
    MACRO_ECONOMIC = "macro_economic"  # マクロ経済要因
    GEOPOLITICAL = "geopolitical"  # 地政学的変化
    FRAUD_SCANDAL = "fraud_scandal"  # 不正・スキャンダル
    OTHER = "other"


class LearningCase(BaseModel):
    """A case study of significant price movement for learning."""

    id: str = Field(description="Unique case identifier")
    symbol: str = Field(description="Stock ticker symbol")
    company_name: str = Field(description="Company name")
    direction: MovementDirection = Field(description="Direction of movement")
    price_before: float = Field(description="Price at start of period")
    price_after: float = Field(description="Price at end of period")
    change_percent: float = Field(description="Percentage change")
    period_start: datetime = Field(description="Start of the movement period")
    period_end: datetime = Field(description="End of the movement period")
    analyzed_at: datetime = Field(default_factory=datetime.now)

    # Analysis results
    structural_change_type: StructuralChangeType | None = Field(
        default=None, description="Type of structural change identified"
    )
    root_cause: str = Field(
        default="", description="Root cause of the price movement"
    )
    news_context: list[str] = Field(
        default_factory=list, description="Key news headlines around the event"
    )
    early_signals: list[str] = Field(
        default_factory=list, description="Early warning signals that could have been detected"
    )
    lessons_learned: list[str] = Field(
        default_factory=list, description="Lessons that can be applied to future analysis"
    )
    confidence: float = Field(
        default=0.0, description="Confidence in the analysis (0-1)"
    )


class LearnedInsight(BaseModel):
    """An insight learned from analyzing price movements."""

    id: str = Field(description="Unique insight identifier")
    title: str = Field(description="Brief title of the insight")
    category: StructuralChangeType = Field(description="Category of insight")
    description: str = Field(description="Detailed description of the insight")
    detection_patterns: list[str] = Field(
        default_factory=list,
        description="Patterns to watch for in news/data to detect similar situations"
    )
    applicable_sectors: list[str] = Field(
        default_factory=list, description="Sectors where this insight applies"
    )
    source_cases: list[str] = Field(
        default_factory=list, description="Case IDs that contributed to this insight"
    )
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    importance: str = Field(
        default="medium", description="Importance level: low/medium/high/critical"
    )


class LearningReport(BaseModel):
    """A report summarizing learning from price movements."""

    generated_at: datetime = Field(default_factory=datetime.now)
    period_analyzed: str = Field(description="Period analyzed (e.g., '2024 Q4')")
    total_cases_analyzed: int = Field(default=0, description="Number of cases analyzed")
    new_insights: list[LearnedInsight] = Field(
        default_factory=list, description="New insights discovered"
    )
    updated_insights: list[str] = Field(
        default_factory=list, description="IDs of insights that were updated"
    )
    key_findings: list[str] = Field(
        default_factory=list, description="Key findings from this analysis"
    )
    recommendations: list[str] = Field(
        default_factory=list, description="Recommendations for improving analysis"
    )
