"""Midas CLI entry point."""

import argparse
import asyncio
import sys

from midas.agents import (
    critical_company_finder,
    farseer,
    general_news_watcher,
    learning_agent,
    negative_info_watcher,
    other_gov_news_watcher,
    portfolio_analyzer,
    price_event_analyzer,
    tech_news_watcher,
    us_gov_news_watcher,
)
from midas.models import AccountType, StockHolding
from midas.tools.portfolio_manager import (
    add_holdings_to_portfolio,
    generate_portfolio_report,
    import_from_csv,
    load_portfolio,
    save_portfolio,
    update_portfolio_prices,
)
from midas.tools.stock_screener import TimeFrame, format_movements, screen_movers

# Watcher definitions
WATCHERS = {
    "us-gov": {
        "name": "US Government News",
        "module": us_gov_news_watcher,
    },
    "tech": {
        "name": "Technology News",
        "module": tech_news_watcher,
    },
    "other-gov": {
        "name": "Other Government News",
        "module": other_gov_news_watcher,
    },
    "general": {
        "name": "General News",
        "module": general_news_watcher,
    },
}


def print_banner():
    """Print Midas banner."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║  Midas - Investment Decision Support Agent                   ║
║  "Identifying structural changes for long-term investors"    ║
╚═══════════════════════════════════════════════════════════════╝
""")


def print_result(result: dict, source_name: str) -> int:
    """Print collection result and return exit code."""
    print("-" * 50)

    if result.get("error"):
        print(f"Error: {result['error']}")
        return 1

    filtered = result.get("filtered_items", [])
    if filtered:
        print(f"\n=== [{source_name}] Found {len(filtered)} structural news items ===\n")
        for item in filtered[:10]:  # Show first 10
            print(f"[{item.category.value if item.category else 'unknown'}] {item.title}")
            if item.relevance_reason:
                print(f"    Reason: {item.relevance_reason}")
            print(f"    URL: {item.url}")
            print()

        if len(filtered) > 10:
            print(f"... and {len(filtered) - 10} more items")

        print(f"Results saved to: {result.get('saved_path')}")
    else:
        print(f"[{source_name}] No structural news items found.")

    return 0


async def collect_news(source: str) -> int:
    """Collect news from specified source."""
    watcher = WATCHERS[source]
    print(f"Starting {watcher['name']} Collection...")
    print("-" * 50)

    result = await watcher["module"].run_agent()
    return print_result(result, watcher["name"])


async def collect_all() -> int:
    """Collect news from all sources."""
    print("Starting collection from ALL sources...")
    print("=" * 50)

    exit_code = 0
    for source, watcher in WATCHERS.items():
        print(f"\n>>> {watcher['name']} <<<")
        print("-" * 50)

        result = await watcher["module"].run_agent()
        code = print_result(result, watcher["name"])
        if code != 0:
            exit_code = code

    print("\n" + "=" * 50)
    print("All collections completed.")
    return exit_code


def run_screen(timeframe: str, min_change: float, direction: str, sector: str | None) -> int:
    """Run stock screening."""
    tf = TimeFrame(timeframe)
    print(f"Screening stocks for {tf.value} performance...")
    print("-" * 50)

    movements = screen_movers(
        timeframe=tf,
        min_change=min_change,
        direction=direction,
        sector=sector,
    )

    print(format_movements(movements))

    significant = [m for m in movements if m.is_significant]
    if significant:
        print(f"\n=== {len(significant)} significant movements detected ===")

    return 0


async def run_analyze(symbol: str, mode: str) -> int:
    """Run company analysis."""
    print(f"Analyzing {symbol}...")
    print("-" * 50)

    if mode == "price":
        result = await price_event_analyzer.run_agent(symbol)
        print(price_event_analyzer.format_analysis(result))
    elif mode == "risk":
        result = await negative_info_watcher.run_agent(symbol)
        print(negative_info_watcher.format_results(result))
    else:  # full
        print("\n>>> Price Event Analysis <<<")
        price_result = await price_event_analyzer.run_agent(symbol)
        print(price_event_analyzer.format_analysis(price_result))

        print("\n>>> Negative Information Scan <<<")
        risk_result = await negative_info_watcher.run_agent(symbol)
        print(negative_info_watcher.format_results(risk_result))

    return 0


# =============================================================================
# Portfolio Commands
# =============================================================================


def run_portfolio_show() -> int:
    """Show current portfolio."""
    portfolio = load_portfolio()
    if not portfolio.holdings:
        print("No holdings in portfolio.")
        print("Use 'midas portfolio import --file <csv_path>' to import from CSV.")
        print("Or use 'midas portfolio add <symbol> <shares> <cost>' to add manually.")
        return 0

    print(generate_portfolio_report(portfolio))
    return 0


def run_portfolio_import(csv_path: str, broker: str) -> int:
    """Import holdings from CSV file."""
    print(f"Importing from: {csv_path}")
    print("-" * 50)

    try:
        holdings = import_from_csv(csv_path, broker=broker)
        if not holdings:
            print("No holdings found in CSV file.")
            return 1

        portfolio = load_portfolio()
        portfolio = add_holdings_to_portfolio(portfolio, holdings)
        save_portfolio(portfolio)

        print("-" * 50)
        print(f"Successfully imported {len(holdings)} holdings.")
        print(generate_portfolio_report(portfolio))
        return 0

    except Exception as e:
        print(f"Error importing CSV: {e}")
        return 1


def run_portfolio_add(symbol: str, shares: int, cost: float, name: str | None, account: str) -> int:
    """Add a holding manually."""
    print(f"Adding {symbol}...")

    try:
        account_type = AccountType(account)
    except ValueError:
        account_type = AccountType.SPECIFIC

    holding = StockHolding(
        symbol=symbol,
        name=name or symbol,
        shares=shares,
        avg_cost=cost,
        account_type=account_type,
        broker="manual",
    )

    portfolio = load_portfolio()
    portfolio = add_holdings_to_portfolio(portfolio, [holding])
    save_portfolio(portfolio)

    print(f"Added: {symbol} x {shares}株 @ ¥{cost:,.0f}")
    return 0


def run_portfolio_update() -> int:
    """Update current prices for all holdings."""
    print("Updating prices...")
    print("-" * 50)

    portfolio = load_portfolio()
    if not portfolio.holdings:
        print("No holdings in portfolio.")
        return 1

    portfolio = update_portfolio_prices(portfolio)
    save_portfolio(portfolio)

    print("-" * 50)
    print(generate_portfolio_report(portfolio))
    return 0


async def run_portfolio_analyze() -> int:
    """Analyze portfolio."""
    print("Analyzing portfolio...")
    print("-" * 50)

    await portfolio_analyzer.run_agent()
    return 0


# =============================================================================
# Find Companies Commands
# =============================================================================


async def run_find_companies(prediction: str) -> int:
    """Find critical companies for a future prediction."""
    print("Finding critical companies for the prediction...")
    print("-" * 50)

    result = await critical_company_finder.run_agent(prediction)
    print(critical_company_finder.format_analysis(result))

    return 0


# =============================================================================
# Farseer Commands
# =============================================================================


async def run_farseer_scan(year: int) -> int:
    """Scan for outlook articles and analyze social changes."""
    print(f"Farseer: Scanning for {year} outlook...")
    print("-" * 50)

    result = await farseer.run_scan(year=year)
    print(farseer.format_report(result))

    return 0


async def run_farseer_expand() -> int:
    """Get suggestions for expanding the source list."""
    print("Farseer: Getting source expansion suggestions...")
    print("-" * 50)

    suggestions = await farseer.expand_sources()

    if not suggestions:
        print("No suggestions available.")
        return 0

    print(f"\nSuggested sources to add ({len(suggestions)}):\n")
    for i, sug in enumerate(suggestions, 1):
        print(f"{i}. {sug.get('name', 'Unknown')}")
        print(f"   URL: {sug.get('url', 'N/A')}")
        print(f"   Category: {sug.get('category', 'N/A')}")
        print(f"   Focus: {', '.join(sug.get('focus', []))}")
        print(f"   Reason: {sug.get('reason', 'N/A')}")
        print()

    return 0


def run_farseer_sources() -> int:
    """Show current source list."""
    sources = farseer.load_sources()

    print("Farseer Source List")
    print(f"Version: {sources.get('version', 'N/A')}")
    print(f"Last updated: {sources.get('updated_at', 'N/A')}")
    print(f"Next review: {sources.get('next_review', 'N/A')}")
    print("=" * 50)

    for cat_name, category in sources.get("categories", {}).items():
        print(f"\n{category.get('description', cat_name)}")
        print("-" * 40)
        for source in category.get("sources", []):
            focus = ", ".join(source.get("focus", []))
            print(f"  - {source['name']} [{source.get('language', 'en')}]")
            print(f"    Focus: {focus}")

    print(f"\nTotal categories: {len(sources.get('categories', {}))}")
    total_sources = sum(
        len(cat.get("sources", []))
        for cat in sources.get("categories", {}).values()
    )
    print(f"Total sources: {total_sources}")

    return 0


# =============================================================================
# Learning Commands
# =============================================================================


async def run_learn_scan(period: str, max_cases: int) -> int:
    """Scan for extreme price movements and analyze them."""
    print(f"Learning: Scanning for extreme movements ({period})...")
    print("-" * 50)

    result = await learning_agent.run_agent(period=period, max_cases=max_cases)
    print(learning_agent.format_report(result))

    return 0


def run_learn_insights() -> int:
    """Show all learned insights."""
    insights = learning_agent.list_insights()
    print(learning_agent.format_insights_list(insights))
    return 0


def run_learn_cases() -> int:
    """Show all analyzed cases."""
    cases = learning_agent.list_cases()

    if not cases:
        print("No cases stored yet. Run 'midas learn scan' to analyze cases.")
        return 0

    print(f"\n{'=' * 70}")
    print(f"Stored Cases ({len(cases)} total)")
    print(f"{'=' * 70}")

    # Group by direction
    up_cases = [c for c in cases if c.direction.value == "up"]
    down_cases = [c for c in cases if c.direction.value == "down"]

    if up_cases:
        print(f"\n[3x GAINERS] ({len(up_cases)} cases)")
        for case in sorted(up_cases, key=lambda c: c.change_percent, reverse=True)[:10]:
            print(f"  {case.symbol}: {case.change_percent:+.1f}% - {case.root_cause[:50] if case.root_cause else 'Not analyzed'}...")

    if down_cases:
        print(f"\n[1/3 LOSERS] ({len(down_cases)} cases)")
        for case in sorted(down_cases, key=lambda c: c.change_percent)[:10]:
            print(f"  {case.symbol}: {case.change_percent:+.1f}% - {case.root_cause[:50] if case.root_cause else 'Not analyzed'}...")

    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Midas - Investment Decision Support Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # collect command
    collect_parser = subparsers.add_parser("collect", help="Collect news")
    collect_parser.add_argument(
        "--source",
        choices=list(WATCHERS.keys()) + ["all"],
        default="all",
        help="News source to collect from (default: all)",
    )

    # screen command
    screen_parser = subparsers.add_parser("screen", help="Screen stocks for price movements")
    screen_parser.add_argument(
        "--timeframe",
        "-t",
        choices=["day", "week", "month", "quarter", "half", "year", "ytd"],
        default="week",
        help="Time period for screening (default: week)",
    )
    screen_parser.add_argument(
        "--min-change",
        "-m",
        type=float,
        default=20.0,
        help="Minimum percentage change to include (default: 20)",
    )
    screen_parser.add_argument(
        "--direction",
        "-d",
        choices=["up", "down", "both"],
        default="both",
        help="Filter direction: up, down, or both (default: both)",
    )
    screen_parser.add_argument(
        "--sector",
        "-s",
        type=str,
        help="Filter by sector (e.g., 'Technology', 'Healthcare')",
    )

    # analyze command
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze a company for price events and risks"
    )
    analyze_parser.add_argument(
        "symbol",
        type=str,
        help="Stock ticker symbol (e.g., AAPL, TSLA)",
    )
    analyze_parser.add_argument(
        "--mode",
        choices=["price", "risk", "full"],
        default="full",
        help="Analysis mode: price (price events), risk (negative info), full (both)",
    )

    # portfolio command
    portfolio_parser = subparsers.add_parser("portfolio", help="Manage stock portfolio")
    portfolio_subparsers = portfolio_parser.add_subparsers(dest="portfolio_command")

    # portfolio show
    portfolio_subparsers.add_parser("show", help="Show current portfolio")

    # portfolio import
    import_parser = portfolio_subparsers.add_parser("import", help="Import holdings from CSV")
    import_parser.add_argument(
        "--file", "-f",
        type=str,
        required=True,
        help="Path to CSV file",
    )
    import_parser.add_argument(
        "--broker", "-b",
        type=str,
        default="nomura",
        help="Broker name (default: nomura)",
    )

    # portfolio add
    add_parser = portfolio_subparsers.add_parser("add", help="Add a holding manually")
    add_parser.add_argument("symbol", type=str, help="Stock ticker symbol")
    add_parser.add_argument("shares", type=int, help="Number of shares")
    add_parser.add_argument("cost", type=float, help="Average acquisition cost per share")
    add_parser.add_argument("--name", "-n", type=str, help="Company name")
    add_parser.add_argument(
        "--account", "-a",
        type=str,
        default="specific",
        choices=["general", "specific", "nisa_growth", "nisa_reserve", "ideco"],
        help="Account type (default: specific)",
    )

    # portfolio update
    portfolio_subparsers.add_parser("update", help="Update current prices")

    # portfolio analyze
    portfolio_subparsers.add_parser("analyze", help="Analyze portfolio with LLM")

    # find-companies command
    find_companies_parser = subparsers.add_parser(
        "find-companies", help="Find companies critical to realizing a future prediction"
    )
    find_companies_parser.add_argument(
        "prediction",
        type=str,
        help="A future prediction to analyze (e.g., 'Electric vehicles will dominate by 2030')",
    )

    # farseer command
    farseer_parser = subparsers.add_parser(
        "farseer", help="Farseer - Scan for future outlook and social changes"
    )
    farseer_subparsers = farseer_parser.add_subparsers(dest="farseer_command")

    # farseer scan
    farseer_scan_parser = farseer_subparsers.add_parser(
        "scan", help="Scan for outlook articles and analyze social changes"
    )
    farseer_scan_parser.add_argument(
        "--year",
        "-y",
        type=int,
        default=None,
        help="Target year for outlook (default: current year)",
    )

    # farseer expand
    farseer_subparsers.add_parser(
        "expand", help="Get AI suggestions for new sources to add"
    )

    # farseer sources
    farseer_subparsers.add_parser(
        "sources", help="Show current source list"
    )

    # learn command
    learn_parser = subparsers.add_parser(
        "learn", help="Learn from extreme price movements"
    )
    learn_subparsers = learn_parser.add_subparsers(dest="learn_command")

    # learn scan
    learn_scan_parser = learn_subparsers.add_parser(
        "scan", help="Scan for and analyze extreme price movements"
    )
    learn_scan_parser.add_argument(
        "--period",
        "-p",
        choices=["month", "quarter", "half", "year"],
        default="month",
        help="Time period to scan (default: month)",
    )
    learn_scan_parser.add_argument(
        "--max-cases",
        "-m",
        type=int,
        default=20,
        help="Maximum number of cases to analyze (default: 20)",
    )

    # learn insights
    learn_subparsers.add_parser(
        "insights", help="Show all learned insights"
    )

    # learn cases
    learn_subparsers.add_parser(
        "cases", help="Show all analyzed cases"
    )

    args = parser.parse_args()

    print_banner()

    if args.command == "collect":
        if args.source == "all":
            return asyncio.run(collect_all())
        else:
            return asyncio.run(collect_news(args.source))
    elif args.command == "screen":
        return run_screen(args.timeframe, args.min_change, args.direction, args.sector)
    elif args.command == "analyze":
        return asyncio.run(run_analyze(args.symbol, args.mode))
    elif args.command == "portfolio":
        if args.portfolio_command == "show":
            return run_portfolio_show()
        elif args.portfolio_command == "import":
            return run_portfolio_import(args.file, args.broker)
        elif args.portfolio_command == "add":
            return run_portfolio_add(args.symbol, args.shares, args.cost, args.name, args.account)
        elif args.portfolio_command == "update":
            return run_portfolio_update()
        elif args.portfolio_command == "analyze":
            return asyncio.run(run_portfolio_analyze())
        else:
            portfolio_parser.print_help()
            return 0
    elif args.command == "find-companies":
        return asyncio.run(run_find_companies(args.prediction))
    elif args.command == "farseer":
        if args.farseer_command == "scan":
            from datetime import datetime
            year = args.year or datetime.now().year
            return asyncio.run(run_farseer_scan(year))
        elif args.farseer_command == "expand":
            return asyncio.run(run_farseer_expand())
        elif args.farseer_command == "sources":
            return run_farseer_sources()
        else:
            farseer_parser.print_help()
            return 0
    elif args.command == "learn":
        if args.learn_command == "scan":
            return asyncio.run(run_learn_scan(args.period, args.max_cases))
        elif args.learn_command == "insights":
            return run_learn_insights()
        elif args.learn_command == "cases":
            return run_learn_cases()
        else:
            learn_parser.print_help()
            return 0
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
