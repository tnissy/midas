"""Portfolio management tool for stock holdings."""

import csv
import json
import re
from datetime import datetime
from pathlib import Path

import yfinance as yf

from midas.config import DATA_DIR
from midas.models import AccountType, Portfolio, StockHolding, Transaction

# =============================================================================
# Paths
# =============================================================================

PORTFOLIO_DIR = DATA_DIR / "portfolio"
PORTFOLIO_FILE = PORTFOLIO_DIR / "portfolio.json"


def ensure_portfolio_dir() -> None:
    """Ensure portfolio directory exists."""
    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Portfolio Persistence
# =============================================================================


def load_portfolio() -> Portfolio:
    """Load portfolio from JSON file."""
    ensure_portfolio_dir()
    if PORTFOLIO_FILE.exists():
        with open(PORTFOLIO_FILE, encoding="utf-8") as f:
            data = json.load(f)
            return Portfolio.model_validate(data)
    return Portfolio()


def save_portfolio(portfolio: Portfolio) -> Path:
    """Save portfolio to JSON file."""
    ensure_portfolio_dir()
    portfolio.updated_at = datetime.now()
    with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
        json.dump(portfolio.model_dump(mode="json"), f, ensure_ascii=False, indent=2)
    return PORTFOLIO_FILE


# =============================================================================
# Stock Price Fetching
# =============================================================================


def normalize_symbol(symbol: str) -> str:
    """Normalize stock symbol for yfinance.

    Japanese stocks need '.T' suffix for Tokyo Stock Exchange.
    """
    symbol = symbol.strip()
    # If already has suffix, return as-is
    if "." in symbol:
        return symbol
    # If it's a numeric code (Japanese stock), add .T suffix
    if re.match(r"^\d+$", symbol):
        return f"{symbol}.T"
    return symbol


def fetch_current_price(symbol: str) -> float | None:
    """Fetch current price for a stock symbol.

    Args:
        symbol: Stock ticker (e.g., '7203' for Toyota, 'AAPL' for Apple)

    Returns:
        Current price or None if not found
    """
    try:
        normalized = normalize_symbol(symbol)
        ticker = yf.Ticker(normalized)
        info = ticker.info
        # Try different price fields
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if price:
            return float(price)
        # Fallback to history
        hist = ticker.history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
        return None
    except Exception as e:
        print(f"Error fetching price for {symbol}: {e}")
        return None


def update_portfolio_prices(portfolio: Portfolio) -> Portfolio:
    """Update current prices for all holdings.

    Args:
        portfolio: Portfolio to update

    Returns:
        Updated portfolio
    """
    print("Updating stock prices...")
    for holding in portfolio.holdings:
        print(f"  Fetching {holding.symbol} ({holding.name})...")
        price = fetch_current_price(holding.symbol)
        if price:
            holding.current_price = price
            print(f"    -> ¥{price:,.0f}")
        else:
            print(f"    -> Price not found")
    portfolio.updated_at = datetime.now()
    return portfolio


# =============================================================================
# CSV Import (Nomura-compatible)
# =============================================================================


# Column name mappings for different brokers/formats
COLUMN_MAPPINGS = {
    # Japanese column names
    "銘柄コード": "symbol",
    "銘柄名": "name",
    "保有数量": "shares",
    "数量": "shares",
    "取得単価": "avg_cost",
    "平均取得単価": "avg_cost",
    "口座区分": "account_type",
    "預り区分": "account_type",
    # English column names
    "Symbol": "symbol",
    "Ticker": "symbol",
    "Code": "symbol",
    "Name": "name",
    "Company": "name",
    "Shares": "shares",
    "Quantity": "shares",
    "Cost": "avg_cost",
    "Average Cost": "avg_cost",
    "Account": "account_type",
}

# Account type mappings
ACCOUNT_TYPE_MAPPINGS = {
    "特定": AccountType.SPECIFIC,
    "特定口座": AccountType.SPECIFIC,
    "一般": AccountType.GENERAL,
    "一般口座": AccountType.GENERAL,
    "NISA成長": AccountType.NISA_GROWTH,
    "NISA（成長投資枠）": AccountType.NISA_GROWTH,
    "NISAつみたて": AccountType.NISA_RESERVE,
    "NISA（つみたて投資枠）": AccountType.NISA_RESERVE,
    "iDeCo": AccountType.IDECO,
}


def parse_number(value: str) -> float:
    """Parse a number string with Japanese/English formatting."""
    if not value:
        return 0.0
    # Remove commas, yen sign, etc.
    cleaned = re.sub(r"[¥￥,$、円株]", "", str(value).strip())
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def map_account_type(value: str) -> AccountType:
    """Map account type string to AccountType enum."""
    value = value.strip()
    return ACCOUNT_TYPE_MAPPINGS.get(value, AccountType.SPECIFIC)


def import_from_csv(
    csv_path: str | Path,
    broker: str = "nomura",
    encoding: str = "utf-8",
) -> list[StockHolding]:
    """Import stock holdings from CSV file.

    Args:
        csv_path: Path to CSV file
        broker: Broker name (e.g., 'nomura', 'sbi', 'rakuten')
        encoding: CSV file encoding (try 'shift_jis' for Japanese files)

    Returns:
        List of StockHolding objects
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    holdings: list[StockHolding] = []

    # Try different encodings
    encodings_to_try = [encoding, "utf-8", "shift_jis", "cp932"]
    content = None

    for enc in encodings_to_try:
        try:
            with open(path, encoding=enc) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue

    if content is None:
        raise ValueError(f"Could not decode CSV file with any encoding")

    # Parse CSV
    lines = content.strip().split("\n")
    if not lines:
        return holdings

    # Detect delimiter
    first_line = lines[0]
    delimiter = "\t" if "\t" in first_line else ","

    reader = csv.DictReader(lines, delimiter=delimiter)

    # Map columns
    fieldnames = reader.fieldnames or []
    column_map = {}
    for field in fieldnames:
        clean_field = field.strip()
        if clean_field in COLUMN_MAPPINGS:
            column_map[field] = COLUMN_MAPPINGS[clean_field]
        else:
            # Try lowercase matching
            for key, value in COLUMN_MAPPINGS.items():
                if key.lower() == clean_field.lower():
                    column_map[field] = value
                    break

    print(f"Detected columns: {column_map}")

    for row in reader:
        try:
            # Extract values using column mapping
            symbol = ""
            name = ""
            shares = 0
            avg_cost = 0.0
            account_type = AccountType.SPECIFIC

            for csv_col, mapped_field in column_map.items():
                value = row.get(csv_col, "").strip()
                if not value:
                    continue

                if mapped_field == "symbol":
                    symbol = value
                elif mapped_field == "name":
                    name = value
                elif mapped_field == "shares":
                    shares = int(parse_number(value))
                elif mapped_field == "avg_cost":
                    avg_cost = parse_number(value)
                elif mapped_field == "account_type":
                    account_type = map_account_type(value)

            # Skip empty rows
            if not symbol or shares <= 0:
                continue

            holding = StockHolding(
                symbol=symbol,
                name=name or symbol,
                shares=shares,
                avg_cost=avg_cost,
                account_type=account_type,
                broker=broker,
            )
            holdings.append(holding)
            print(f"  Imported: {symbol} - {name} x {shares}株 @{avg_cost:,.0f}")

        except Exception as e:
            print(f"  Error parsing row: {e}")
            continue

    return holdings


def add_holdings_to_portfolio(
    portfolio: Portfolio, holdings: list[StockHolding], merge: bool = True
) -> Portfolio:
    """Add holdings to portfolio.

    Args:
        portfolio: Existing portfolio
        holdings: Holdings to add
        merge: If True, merge with existing holdings by symbol

    Returns:
        Updated portfolio
    """
    for new_holding in holdings:
        existing = portfolio.get_holding(new_holding.symbol)
        if existing and merge:
            # Merge: calculate weighted average cost
            total_shares = existing.shares + new_holding.shares
            total_cost = (existing.shares * existing.avg_cost) + (
                new_holding.shares * new_holding.avg_cost
            )
            existing.shares = total_shares
            existing.avg_cost = total_cost / total_shares if total_shares > 0 else 0
            print(f"  Merged: {new_holding.symbol} -> {total_shares}株")
        else:
            portfolio.holdings.append(new_holding)
            print(f"  Added: {new_holding.symbol}")

    return portfolio


# =============================================================================
# Portfolio Reports
# =============================================================================


def generate_portfolio_report(portfolio: Portfolio) -> str:
    """Generate a text report of the portfolio.

    Args:
        portfolio: Portfolio to report

    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"Portfolio Report: {portfolio.name}")
    lines.append(f"Updated: {portfolio.updated_at.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)
    lines.append("")

    # Holdings table
    lines.append("Holdings:")
    lines.append("-" * 60)
    lines.append(f"{'Symbol':<8} {'Name':<20} {'Shares':>8} {'Avg Cost':>10} {'Current':>10} {'P/L %':>8}")
    lines.append("-" * 60)

    for h in sorted(portfolio.holdings, key=lambda x: x.symbol):
        current_str = f"¥{h.current_price:,.0f}" if h.current_price else "N/A"
        pl_str = f"{h.unrealized_gain_percent:+.1f}%" if h.unrealized_gain_percent else "N/A"
        lines.append(
            f"{h.symbol:<8} {h.name[:20]:<20} {h.shares:>8,} ¥{h.avg_cost:>9,.0f} {current_str:>10} {pl_str:>8}"
        )

    lines.append("-" * 60)

    # Summary
    lines.append("")
    lines.append("Summary:")
    lines.append(f"  Total Holdings: {len(portfolio.holdings)}")
    lines.append(f"  Total Cost: ¥{portfolio.total_cost:,.0f}")
    if portfolio.total_value:
        lines.append(f"  Current Value: ¥{portfolio.total_value:,.0f}")
        gain = portfolio.total_unrealized_gain
        gain_pct = (gain / portfolio.total_cost * 100) if portfolio.total_cost > 0 else 0
        lines.append(f"  Unrealized P/L: ¥{gain:+,.0f} ({gain_pct:+.1f}%)")
    if portfolio.cash_balance > 0:
        lines.append(f"  Cash Balance: ¥{portfolio.cash_balance:,.0f}")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def generate_portfolio_summary(portfolio: Portfolio) -> dict:
    """Generate a summary dict of the portfolio.

    Args:
        portfolio: Portfolio to summarize

    Returns:
        Summary dict with key metrics
    """
    return {
        "name": portfolio.name,
        "updated_at": portfolio.updated_at.isoformat(),
        "total_holdings": len(portfolio.holdings),
        "total_cost": portfolio.total_cost,
        "total_value": portfolio.total_value,
        "unrealized_gain": portfolio.total_unrealized_gain,
        "unrealized_gain_percent": (
            (portfolio.total_unrealized_gain / portfolio.total_cost * 100)
            if portfolio.total_cost > 0 and portfolio.total_unrealized_gain
            else None
        ),
        "cash_balance": portfolio.cash_balance,
        "holdings": [
            {
                "symbol": h.symbol,
                "name": h.name,
                "shares": h.shares,
                "avg_cost": h.avg_cost,
                "current_price": h.current_price,
                "unrealized_gain_percent": h.unrealized_gain_percent,
            }
            for h in portfolio.holdings
        ],
    }
