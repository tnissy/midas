"""Stock screener tool using FINVIZ for bulk screening."""

from enum import Enum

from finvizfinance.screener.performance import Performance

from midas.models import StockMovement


class TimeFrame(str, Enum):
    """Time frame for performance screening."""

    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    HALF_YEAR = "half"
    YEAR = "year"
    YTD = "ytd"


def _get_performance_filter_options(timeframe: TimeFrame) -> dict[str, tuple[list[float], list[float]]]:
    """Get available performance filter thresholds for each timeframe.

    Returns dict mapping timeframe prefix to (up_thresholds, down_thresholds).
    Based on FINVIZ's actual filter options.
    """
    # FINVIZ filter options vary by timeframe
    return {
        TimeFrame.DAY: {
            "prefix": "Today",
            "up": [5, 10, 15],
            "down": [5, 10, 15],
        },
        TimeFrame.WEEK: {
            "prefix": "Week",
            "up": [10, 20, 30],
            "down": [10, 20, 30],
        },
        TimeFrame.MONTH: {
            "prefix": "Month",
            "up": [10, 20, 30, 50],
            "down": [10, 20, 30, 50],
        },
        TimeFrame.QUARTER: {
            "prefix": "Quarter",
            "up": [10, 20, 30, 50],
            "down": [10, 20, 30, 50],
        },
        TimeFrame.HALF_YEAR: {
            "prefix": "Half",
            "up": [10, 20, 30, 50, 100],
            "down": [10, 20, 30, 50, 75],
        },
        TimeFrame.YEAR: {
            "prefix": "Year",
            "up": [10, 20, 30, 50, 100, 200, 300, 500],
            "down": [10, 20, 30, 50, 75],
        },
        TimeFrame.YTD: {
            "prefix": "YTD",
            "up": [5, 10, 20, 30, 50, 100],
            "down": [5, 10, 20, 30, 50, 75],
        },
    }


# Threshold for significant movement (3x = +200%, 1/3 = -66.7%)
THRESHOLD_UP = 200.0  # +200% means 3x
THRESHOLD_DOWN = -66.7  # -66.7% means 1/3


def _get_performance_column(timeframe: TimeFrame) -> str:
    """Get the DataFrame column name for a timeframe."""
    mapping = {
        TimeFrame.DAY: "Change",  # Today's change
        TimeFrame.WEEK: "Perf Week",
        TimeFrame.MONTH: "Perf Month",
        TimeFrame.QUARTER: "Perf Quart",
        TimeFrame.HALF_YEAR: "Perf Half",
        TimeFrame.YEAR: "Perf Year",
        TimeFrame.YTD: "Perf YTD",
    }
    return mapping.get(timeframe, "Perf Week")



def screen_movers(
    timeframe: TimeFrame = TimeFrame.WEEK,
    min_change: float = 20.0,
    direction: str = "both",
    max_results: int = 100,
    market_cap: str | None = None,
    sector: str | None = None,
) -> list[StockMovement]:
    """Screen stocks for significant price movements using FINVIZ.

    Args:
        timeframe: Time period to check (day, week, month, quarter, half, year, ytd)
        min_change: Minimum absolute percentage change to include
        direction: "up", "down", or "both"
        max_results: Maximum number of results to return
        market_cap: Filter by market cap (e.g., "Small ($300mln to $2bln)")
        sector: Filter by sector (e.g., "Technology", "Healthcare")

    Returns:
        List of StockMovement objects sorted by change_percent
    """
    print(f"Screening stocks with FINVIZ ({timeframe.value} performance)...")

    movements: list[StockMovement] = []

    # Screen for gainers if needed
    if direction in ("up", "both"):
        gainers = _screen_direction(
            timeframe=timeframe,
            min_change=min_change,
            is_up=True,
            market_cap=market_cap,
            sector=sector,
        )
        movements.extend(gainers)
        print(f"  Found {len(gainers)} gainers (>{min_change}%)")

    # Screen for losers if needed
    if direction in ("down", "both"):
        losers = _screen_direction(
            timeframe=timeframe,
            min_change=min_change,
            is_up=False,
            market_cap=market_cap,
            sector=sector,
        )
        movements.extend(losers)
        print(f"  Found {len(losers)} losers (<-{min_change}%)")

    # Sort by absolute change (biggest movers first)
    movements.sort(key=lambda x: abs(x.change_percent), reverse=True)

    # Limit results
    if len(movements) > max_results:
        movements = movements[:max_results]

    print(f"  Total: {len(movements)} stocks with significant movements")

    return movements


def _screen_direction(
    timeframe: TimeFrame,
    min_change: float,
    is_up: bool,
    market_cap: str | None = None,
    sector: str | None = None,
) -> list[StockMovement]:
    """Screen for stocks moving in a specific direction."""
    try:
        # Use Performance view to get performance data
        fperf = Performance()

        # Build filters
        filters_dict = {}

        # Add performance filter (FINVIZ uses "Performance" key for all timeframes)
        perf_filter = _get_best_performance_filter(timeframe, min_change, is_up)
        if perf_filter:
            filters_dict["Performance"] = perf_filter

        # Add optional filters
        if market_cap:
            filters_dict["Market Cap."] = market_cap
        if sector:
            filters_dict["Sector"] = sector

        # Apply filters
        if filters_dict:
            fperf.set_filter(filters_dict=filters_dict)

        # Get the data
        df = fperf.screener_view()

        if df is None or df.empty:
            return []

        # Get the correct performance column
        perf_col = _get_performance_column(timeframe)

        movements = []
        for _, row in df.iterrows():
            try:
                symbol = row.get("Ticker", "")
                # Performance view doesn't have Company name, use Ticker
                name = symbol

                # Parse change percent (FINVIZ returns as decimal, e.g., 0.4635 = 46.35%)
                perf_value = row.get(perf_col, 0)
                if isinstance(perf_value, str):
                    # Handle string format (with % sign)
                    change_percent = float(perf_value.replace("%", "").strip()) * 100
                else:
                    # Already a float (decimal format)
                    change_percent = float(perf_value) * 100

                # Filter by direction and minimum change
                if is_up and change_percent < min_change:
                    continue
                if not is_up and change_percent > -min_change:
                    continue

                # Get price
                price_now = float(row.get("Price", 0))

                # Calculate estimated price before
                if change_percent != 0:
                    price_before = price_now / (1 + change_percent / 100)
                else:
                    price_before = price_now

                # Check if significant (3x or 1/3)
                is_significant = (
                    change_percent >= THRESHOLD_UP or change_percent <= THRESHOLD_DOWN
                )

                movements.append(
                    StockMovement(
                        symbol=symbol,
                        name=name,
                        price_before=round(price_before, 2),
                        price_now=round(price_now, 2),
                        change_percent=round(change_percent, 2),
                        is_significant=is_significant,
                    )
                )
            except (ValueError, TypeError):
                # Skip rows with parsing errors
                continue

        return movements

    except Exception as e:
        print(f"  Error screening: {e}")
        return []


def _get_best_performance_filter(
    timeframe: TimeFrame, min_change: float, is_up: bool
) -> str | None:
    """Get the best matching FINVIZ performance filter for the given timeframe.

    Args:
        timeframe: The time period for the filter
        min_change: Minimum change percentage to filter for
        is_up: True for gains, False for losses

    Returns:
        Filter string like "Week +20%" or "Month -30%", or None
    """
    options = _get_performance_filter_options(timeframe)
    config = options.get(timeframe)

    if not config:
        return None

    prefix = config["prefix"]
    thresholds = sorted(config["up" if is_up else "down"], reverse=True)

    # Find the best matching threshold
    for threshold in thresholds:
        if min_change >= threshold:
            sign = "+" if is_up else "-"
            return f"{prefix} {sign}{int(threshold)}%"

    # If no threshold matches, use the smallest one
    if thresholds:
        smallest = min(thresholds)
        sign = "+" if is_up else "-"
        return f"{prefix} {sign}{int(smallest)}%"

    return None


def screen_all_performance(
    timeframe: TimeFrame = TimeFrame.WEEK,
    max_results: int = 200,
    sector: str | None = None,
) -> list[StockMovement]:
    """Get all stocks sorted by performance without filtering by change amount.

    This is useful for getting a broad view of the market.

    Args:
        timeframe: Time period to check
        max_results: Maximum number of results
        sector: Optional sector filter

    Returns:
        List of StockMovement objects sorted by change_percent (descending)
    """
    print(f"Getting all stocks by {timeframe.value} performance from FINVIZ...")

    try:
        fperf = Performance()

        filters_dict = {}
        if sector:
            filters_dict["Sector"] = sector

        if filters_dict:
            fperf.set_filter(filters_dict=filters_dict)

        df = fperf.screener_view()

        if df is None or df.empty:
            print("  No data returned from FINVIZ")
            return []

        perf_col = _get_performance_column(timeframe)

        movements = []
        for _, row in df.iterrows():
            try:
                symbol = row.get("Ticker", "")
                name = symbol  # Performance view doesn't have Company name

                # Parse change percent (FINVIZ returns as decimal)
                perf_value = row.get(perf_col, 0)
                if isinstance(perf_value, str):
                    change_percent = float(perf_value.replace("%", "").strip()) * 100
                else:
                    change_percent = float(perf_value) * 100

                price_now = float(row.get("Price", 0))
                if change_percent != 0:
                    price_before = price_now / (1 + change_percent / 100)
                else:
                    price_before = price_now

                is_significant = (
                    change_percent >= THRESHOLD_UP or change_percent <= THRESHOLD_DOWN
                )

                movements.append(
                    StockMovement(
                        symbol=symbol,
                        name=name,
                        price_before=round(price_before, 2),
                        price_now=round(price_now, 2),
                        change_percent=round(change_percent, 2),
                        is_significant=is_significant,
                    )
                )
            except (ValueError, TypeError):
                continue

        # Sort by change percent (biggest gains first)
        movements.sort(key=lambda x: x.change_percent, reverse=True)

        if len(movements) > max_results:
            movements = movements[:max_results]

        print(f"  Retrieved {len(movements)} stocks")
        return movements

    except Exception as e:
        print(f"  Error: {e}")
        return []


def format_movements(movements: list[StockMovement]) -> str:
    """Format movements for display."""
    if not movements:
        return "No stock movements found."

    lines = []
    lines.append(f"\n{'Symbol':<8} {'Name':<30} {'Before':>10} {'Now':>10} {'Change':>10}")
    lines.append("-" * 72)

    for m in movements:
        name = m.name[:28] + ".." if len(m.name) > 30 else m.name
        marker = "*" if m.is_significant else " "
        lines.append(
            f"{marker}{m.symbol:<7} {name:<30} ${m.price_before:>8.2f} ${m.price_now:>8.2f} {m.change_percent:>+9.1f}%"
        )

    significant_count = sum(1 for m in movements if m.is_significant)
    if significant_count > 0:
        lines.append(f"\n* = Significant movement ({significant_count} found)")

    return "\n".join(lines)
