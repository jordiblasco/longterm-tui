"""Command line entry point:  python -m ltui [watchlist.yaml] [--demo] [--print]"""
from __future__ import annotations

import argparse
import sys

from .data import fetch_all, load_watchlist


def print_table(wl, demo: bool) -> None:
    """Non-interactive mode: fetch everything once and print the dashboard."""
    from rich.console import Console
    from rich.table import Table

    from .render import COLUMNS, row_cells

    console = Console()
    with console.status("Fetching data…"):
        reports = fetch_all(wl.entries, wl.thresholds, wl.period, demo)

    table = Table(title="Long-Term Investor Dashboard" + (" (DEMO DATA)" if demo else ""),
                  header_style="bold cyan", show_lines=False)
    for label, justify in COLUMNS:
        table.add_column(label, justify=justify, no_wrap=True)
    ordered = sorted(reports.values(), key=lambda r: (bool(r.error), -r.score))
    for rep in ordered:
        table.add_row(*row_cells(rep))
    console.print(table)
    console.print("[green]Green[/] = buy signal · [red]Red[/] = sell signal · plain = neutral · "
                  "Not financial advice.", highlight=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ltui", description=__doc__)
    ap.add_argument("watchlist", nargs="?", default="watchlist.yaml",
                    help="YAML file with the stocks to review (default: watchlist.yaml)")
    ap.add_argument("--demo", action="store_true", help="use synthetic offline data (no Yahoo access)")
    ap.add_argument("--print", dest="print_only", action="store_true",
                    help="print the table once and exit instead of opening the TUI")
    args = ap.parse_args(argv)

    try:
        wl = load_watchlist(args.watchlist)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if args.print_only:
        print_table(wl, args.demo)
        return 0

    from .app import LongTermApp
    LongTermApp(wl.entries, wl.thresholds, wl.period, args.demo).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
