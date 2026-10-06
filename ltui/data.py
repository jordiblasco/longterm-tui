"""Watchlist loading and data fetching (Yahoo Finance via yfinance, or demo data)."""
from __future__ import annotations

import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import yaml

from .analysis import StockReport, Thresholds, build_report


# --------------------------------------------------------------------------
# Watchlist
# --------------------------------------------------------------------------
@dataclass
class Entry:
    ticker: str
    name: Optional[str] = None


@dataclass
class Watchlist:
    entries: list = field(default_factory=list)
    thresholds: Thresholds = field(default_factory=Thresholds)
    period: str = "2y"


def load_watchlist(path: str | Path) -> Watchlist:
    """Read the YAML file. Accepted shapes:

        stocks: [AAPL, MSFT]                       # simple
        stocks:
          - ticker: NESN.SW
            name: Nestlé                           # optional display name
        settings:
          period: 2y
          thresholds: {rsi_buy: 25}
    or just a top-level list of tickers.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Watchlist file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if isinstance(raw, list):
        raw = {"stocks": raw}
    if not isinstance(raw, dict) or not raw.get("stocks"):
        raise ValueError(f"{path}: expected a 'stocks:' list with at least one ticker")

    entries, seen = [], set()
    for item in raw["stocks"]:
        if isinstance(item, str):
            ticker, name = item, None
        elif isinstance(item, dict) and item.get("ticker"):
            ticker, name = str(item["ticker"]), item.get("name")
        else:
            raise ValueError(f"{path}: invalid stock entry: {item!r}")
        ticker = ticker.strip().upper()
        if ticker and ticker not in seen:
            seen.add(ticker)
            entries.append(Entry(ticker, name))

    settings = raw.get("settings") or {}
    return Watchlist(
        entries=entries,
        thresholds=Thresholds.from_dict(settings.get("thresholds")),
        period=str(settings.get("period", "2y")),
    )


# --------------------------------------------------------------------------
# Yahoo Finance
# --------------------------------------------------------------------------
def fetch_report(entry: Entry, th: Thresholds, period: str = "2y") -> StockReport:
    """Download one ticker and turn it into a StockReport (never raises)."""
    try:
        import yfinance as yf  # imported lazily so --demo works without it

        tk = yf.Ticker(entry.ticker)
        hist = tk.history(period=period, interval="1d", auto_adjust=False)
        if hist is None or hist.empty:
            return StockReport(entry.ticker, name=entry.name or entry.ticker,
                               error="No price history returned (wrong ticker symbol?)")
        try:
            info = tk.info or {}
        except Exception:  # fundamentals are optional; technicals still work
            info = {}
        return build_report(entry.ticker, entry.name, hist, info, th)
    except Exception as exc:  # network, parsing, rate limit...
        return StockReport(entry.ticker, name=entry.name or entry.ticker,
                           error=f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
# Demo data (offline): random walks with plausible fundamentals
# --------------------------------------------------------------------------
def fetch_demo(entry: Entry, th: Thresholds, period: str = "2y") -> StockReport:
    rng = np.random.default_rng(zlib.crc32(entry.ticker.encode()))
    n = 520
    drift = rng.uniform(-0.0006, 0.0014)
    rets = rng.normal(drift, 0.015, n)
    close = rng.uniform(20, 400) * np.exp(np.cumsum(rets))
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    spread = np.abs(rng.normal(0.01, 0.004, n))
    hist = pd.DataFrame({"Close": close, "High": close * (1 + spread),
                         "Low": close * (1 - spread)}, index=idx)
    info = {
        "currency": "USD",
        "longName": f"{entry.ticker} Demo Corp.",
        "trailingPE": float(rng.uniform(8, 40)),
        "pegRatio": float(rng.uniform(0.4, 3.2)),
        "returnOnEquity": float(rng.uniform(0.02, 0.35)),
        "currentRatio": float(rng.uniform(0.6, 2.8)),
    }
    return build_report(entry.ticker, entry.name, hist, info, th)


# --------------------------------------------------------------------------
# Parallel fetch
# --------------------------------------------------------------------------
def fetch_all(entries: list, th: Thresholds, period: str = "2y", demo: bool = False,
              on_report: Optional[Callable[[StockReport, int], None]] = None,
              max_workers: int = 8) -> dict:
    """Fetch every entry concurrently. `on_report(report, done_count)` fires as each finishes."""
    fetch = fetch_demo if demo else fetch_report
    results: dict = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch, e, th, period): e for e in entries}
        for done, fut in enumerate(as_completed(futures), start=1):
            rep = fut.result()
            results[rep.ticker] = rep
            if on_report:
                on_report(rep, done)
    return results
