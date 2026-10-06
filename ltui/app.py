"""Textual user interface: one dashboard table + a reasoning panel."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from textual.app import App, ComposeResult
from textual.widgets import DataTable, Footer, Header, Static

from .analysis import StockReport, Thresholds
from .data import Entry, fetch_all
from .render import COLUMNS, detail_text, loading_cells, row_cells

SORTS = ["score ▼ (best first)", "score ▲ (worst first)", "ticker A→Z", "watchlist order"]


class LongTermApp(App):
    TITLE = "Long-Term Investor Dashboard"
    CSS = """
    #table  { height: 1fr; }
    #detail { height: auto; max-height: 12; padding: 0 1; border-top: solid $primary; }
    """
    BINDINGS = [
        ("r", "refresh", "Refresh"),
        ("s", "cycle_sort", "Sort"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, entries: list, thresholds: Thresholds, period: str = "2y",
                 demo: bool = False) -> None:
        super().__init__()
        self.entries: list = entries
        self.thresholds = thresholds
        self.period = period
        self.demo = demo
        self.reports: dict = {}
        self.sort_mode = 0
        self.loading = False
        self.selected: Optional[str] = None
        self.order: list = []

    # ---- layout -----------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="table", cursor_type="row", zebra_stripes=True)
        yield Static(detail_text(None), id="detail")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#table", DataTable)
        for label, _ in COLUMNS:
            table.add_column(label)
        self.render_table()
        self.action_refresh()

    # ---- data loading (background thread) -----------------------------------
    def action_refresh(self) -> None:
        if self.loading:
            return
        self.loading = True
        self.reports = {}
        self.sub_title = f"Loading 0/{len(self.entries)}…"
        self.render_table()
        self.run_worker(self._load_all, thread=True, exclusive=True, name="load")

    def _load_all(self) -> None:
        total = len(self.entries)

        def on_report(rep: StockReport, done: int) -> None:
            self.call_from_thread(self._on_report, rep, done, total)

        fetch_all(self.entries, self.thresholds, self.period, self.demo, on_report)
        self.call_from_thread(self._on_done)

    def _on_report(self, rep: StockReport, done: int, total: int) -> None:
        self.reports[rep.ticker] = rep
        self.sub_title = f"Loading {done}/{total}…"
        self.render_table()

    def _on_done(self) -> None:
        self.loading = False
        self._update_subtitle()

    # ---- table rendering ------------------------------------------------------
    def _sorted_entries(self) -> list:
        entries = list(self.entries)
        mode = self.sort_mode
        if mode == 3:
            return entries
        if mode == 2:
            return sorted(entries, key=lambda e: e.ticker)
        loaded = [e for e in entries if e.ticker in self.reports and not self.reports[e.ticker].error]
        rest = [e for e in entries if e not in loaded]
        loaded.sort(key=lambda e: self.reports[e.ticker].score, reverse=(mode == 0))
        return loaded + rest

    def render_table(self) -> None:
        table = self.query_one("#table", DataTable)
        ordered = self._sorted_entries()
        self.order = [e.ticker for e in ordered]
        target = self.selected if self.selected in self.order else None

        table.clear()
        for e in ordered:
            rep = self.reports.get(e.ticker)
            cells = row_cells(rep) if rep else loading_cells(e.ticker, e.name or "")
            table.add_row(*cells, key=e.ticker)
        if target is not None:
            table.move_cursor(row=self.order.index(target))
        self._show_detail(target)

    def _show_detail(self, ticker: Optional[str]) -> None:
        rep = self.reports.get(ticker) if ticker else None
        self.query_one("#detail", Static).update(detail_text(rep))

    def _update_subtitle(self) -> None:
        ok = sum(1 for r in self.reports.values() if not r.error)
        bad = len(self.reports) - ok
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        note = f" · {bad} failed" if bad else ""
        mode = " · DEMO DATA" if self.demo else ""
        self.sub_title = f"{ok}/{len(self.entries)} stocks · updated {stamp} · sort: {SORTS[self.sort_mode]}{note}{mode}"

    # ---- events ---------------------------------------------------------------
    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        key = event.row_key.value if event.row_key is not None else None
        if key:
            self.selected = key
            self._show_detail(key)

    def action_cycle_sort(self) -> None:
        self.sort_mode = (self.sort_mode + 1) % len(SORTS)
        self.render_table()
        if not self.loading:
            self._update_subtitle()
