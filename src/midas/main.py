"""Midas CLI entry point."""

import argparse
import asyncio
import sys

from midas.agents.us_gov_news import run_agent


def print_banner():
    """Print Midas banner."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║  Midas - Investment Decision Support Agent                   ║
║  "Identifying structural changes for long-term investors"    ║
╚═══════════════════════════════════════════════════════════════╝
""")


async def collect_us_news():
    """Collect US government news."""
    print("Starting US Government News Collection...")
    print("-" * 50)

    result = await run_agent()

    print("-" * 50)

    if result.get("error"):
        print(f"Error: {result['error']}")
        return 1

    filtered = result.get("filtered_items", [])
    if filtered:
        print(f"\n=== Found {len(filtered)} structural news items ===\n")
        for item in filtered[:10]:  # Show first 10
            print(f"[{item.category.value if item.category else 'unknown'}] {item.title}")
            if item.relevance_reason:
                print(f"    Reason: {item.relevance_reason}")
            print(f"    URL: {item.url}")
            print()

        if len(filtered) > 10:
            print(f"... and {len(filtered) - 10} more items")

        print(f"\nResults saved to: {result.get('saved_path')}")
    else:
        print("No structural news items found.")

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
        choices=["us-gov", "all"],
        default="us-gov",
        help="News source to collect from",
    )

    args = parser.parse_args()

    print_banner()

    if args.command == "collect":
        if args.source == "us-gov":
            return asyncio.run(collect_us_news())
        else:
            print("Only 'us-gov' source is implemented for now")
            return 1
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
