"""Turns StockReports into coloured Rich cells (shared by the TUI and --print mode)."""
from __future__ import annotations

from typing import Optional

from rich.text import Text

from .analysis import BUY, SELL, Signal, StockReport

GREEN, RED, NEUTRAL, DIM = "bold green", "bold red", "", "dim"

RATING_STYLE = {
    "STRONG BUY": "bold white on green",
    "BUY": "bold green",
    "HOLD": "yellow",
    "SELL": "bold red",
    "STRONG SELL": "bold white on red",
    "N/A": "dim",
    "ERROR": "bold red",
}

# (header, justify) – the order here is the order of the cells below.
COLUMNS = [
    ("Ticker", "left"), ("Name", "left"), ("Price", "right"), ("Day %", "right"),
    ("Day range", "right"), ("vs 52w high", "right"),
    ("RSI(14)", "right"), ("MACD hist", "right"), ("SMA50", "right"), ("SMA200", "right"),
    ("PEG", "right"), ("ROE", "right"), ("Curr.ratio", "right"),
    ("Score", "right"), ("Rating", "center"),
]


def verdict_style(sig: Optional[Signal]) -> str:
    if sig is None or not sig.available:
        return DIM
    return GREEN if sig.verdict == BUY else RED if sig.verdict == SELL else NEUTRAL


def _fmt(v: Optional[float], nd: int = 2, suffix: str = "", signed: bool = False) -> str:
    if v is None:
        return "—"
    return f"{v:+,.{nd}f}{suffix}" if signed else f"{v:,.{nd}f}{suffix}"


def loading_cells(ticker: str, name: str = "") -> list:
    cells = [Text(ticker, style="bold"), Text(name[:22], style=DIM)]
    cells += [Text("…", style=DIM) for _ in COLUMNS[2:]]
    return cells


def row_cells(r: StockReport) -> list:
    if r.error:
        cells = [Text(r.ticker, style="bold"), Text((r.name or "")[:22], style=DIM),
                 Text("no data", style="red")]
        cells += [Text("") for _ in COLUMNS[3:-1]]
        cells.append(Text("ERROR", style=RATING_STYLE["ERROR"]))
        return cells

    s = r.signals
    price = f"{r.price:,.2f} {r.currency}".strip() if r.price is not None else "—"
    day = f"{_fmt(r.day_low)} – {_fmt(r.day_high)}" if r.day_low is not None and r.day_high is not None else "—"
    trend_style = verdict_style(s.get("trend"))

    return [
        Text(r.ticker, style="bold"),
        Text((r.name or "")[:22]),
        Text(price),
        Text(_fmt(r.change_pct, 2, "%", signed=True)),
        Text(day),
        Text(_fmt(r.off_52w_high_pct, 1, "%", signed=True)),
        Text(_fmt(r.rsi, 1), style=verdict_style(s.get("rsi"))),
        Text(_fmt(r.macd_hist, 2, signed=True), style=verdict_style(s.get("macd"))),
        Text(_fmt(r.sma50), style=trend_style),
        Text(_fmt(r.sma200), style=trend_style),
        Text(_fmt(r.peg), style=verdict_style(s.get("peg"))),
        Text(_fmt(r.roe_pct, 1, "%"), style=verdict_style(s.get("roe"))),
        Text(_fmt(r.current_ratio), style=verdict_style(s.get("cr"))),
        Text(f"{r.score:+d}", style=GREEN if r.score > 0 else RED if r.score < 0 else NEUTRAL),
        Text(f" {r.rating} ", style=RATING_STYLE.get(r.rating, "")),
    ]


def detail_text(r: Optional[StockReport]) -> Text:
    """Explains *why* a stock got its rating (shown under the table)."""
    t = Text()
    if r is None:
        t.append("Green", style=GREEN)
        t.append(" = buy signal   ")
        t.append("Red", style=RED)
        t.append(" = sell signal   plain = neutral   ")
        t.append("—", style=DIM)
        t.append(" = no data.   Move with ↑/↓ to see the reasoning for each stock.")
        return t
    if r.error:
        t.append(f"{r.ticker}: ", style="bold")
        t.append(r.error, style="red")
        return t

    t.append(f"{r.ticker} — {r.name}   ", style="bold")
    t.append(f"score {r.score:+d} → ")
    t.append(r.rating, style=RATING_STYLE.get(r.rating, ""))
    extras = []
    if r.pe is not None:
        extras.append(f"P/E {r.pe:,.1f}")
    if r.wk52_low is not None and r.wk52_high is not None:
        extras.append(f"52w range {r.wk52_low:,.2f}–{r.wk52_high:,.2f}")
    if extras:
        t.append("    " + "   ".join(extras), style=DIM)
    t.append("\n")
    for sig in r.signals.values():
        word = "BUY " if sig.verdict == BUY and sig.available else \
               "SELL" if sig.verdict == SELL and sig.available else \
               "HOLD" if sig.available else "n/a "
        t.append(f"  {sig.label:<14}", style="bold")
        t.append(f"{word}  ", style=verdict_style(sig))
        t.append(sig.note, style=DIM)
        t.append("\n")
    return t
