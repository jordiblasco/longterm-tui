"""Indicators, signal rules and scoring.

Pure pandas/numpy logic: no network, no UI. Everything that decides
"buy / hold / sell" lives here so the rules are easy to read and tweak.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Any, Optional

import pandas as pd

BUY, HOLD, SELL = 1, 0, -1


# --------------------------------------------------------------------------
# Thresholds (all overridable from the YAML file)
# --------------------------------------------------------------------------
@dataclass
class Thresholds:
    rsi_buy: float = 30.0            # RSI below  -> oversold   -> buy
    rsi_sell: float = 70.0           # RSI above  -> overbought -> sell
    peg_buy: float = 1.0             # PEG below  -> cheap for its growth
    peg_sell: float = 2.0            # PEG above  -> expensive for its growth
    roe_buy: float = 15.0            # ROE %, above -> efficient business
    roe_sell: float = 8.0            # ROE %, below -> weak returns on equity
    current_ratio_buy: float = 1.5   # liquidity comfortably above 1
    current_ratio_sell: float = 1.0  # below 1 -> short-term liabilities > assets
    score_buy: int = 2               # total score >= this -> BUY
    score_strong: int = 4            # total score >= this -> STRONG BUY
    min_indicators: int = 3          # fewer available indicators -> "N/A"

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "Thresholds":
        data = data or {}
        valid = {f.name for f in fields(cls)}
        unknown = set(data) - valid
        if unknown:
            raise ValueError(
                f"Unknown threshold(s): {', '.join(sorted(unknown))}. "
                f"Valid: {', '.join(sorted(valid))}"
            )
        return cls(**data)


# --------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -1 * delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Returns (macd_line, signal_line, histogram)."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window, min_periods=window).mean()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def num(x: Any) -> Optional[float]:
    """Coerce to a finite float or None (Yahoo returns None, 'Infinity', etc.)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def last(series: Optional[pd.Series]) -> Optional[float]:
    if series is None:
        return None
    s = series.dropna()
    return float(s.iloc[-1]) if len(s) else None


def first_num(*values: Any) -> Optional[float]:
    for v in values:
        n = num(v)
        if n is not None:
            return n
    return None


# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------
@dataclass
class Signal:
    label: str
    value: Optional[float]
    verdict: int            # +1 buy, 0 neutral, -1 sell
    note: str
    available: bool = True


def _na(label: str, why: str) -> Signal:
    return Signal(label, None, HOLD, why, available=False)


def rsi_signal(value: Optional[float], th: Thresholds) -> Signal:
    if value is None:
        return _na("RSI(14)", "not enough price history")
    if value < th.rsi_buy:
        return Signal("RSI(14)", value, BUY, f"oversold (< {th.rsi_buy:g})")
    if value > th.rsi_sell:
        return Signal("RSI(14)", value, SELL, f"overbought (> {th.rsi_sell:g})")
    return Signal("RSI(14)", value, HOLD, f"neutral ({th.rsi_buy:g}-{th.rsi_sell:g})")


def macd_signal(line: Optional[float], sig: Optional[float], hist: Optional[float]) -> Signal:
    if line is None or sig is None or hist is None:
        return _na("MACD", "not enough price history")
    if hist > 0 and line > 0:
        return Signal("MACD", hist, BUY, "above signal line and above zero (bullish momentum)")
    if hist < 0 and line < 0:
        return Signal("MACD", hist, SELL, "below signal line and below zero (bearish momentum)")
    if hist > 0:
        return Signal("MACD", hist, HOLD, "above signal line but still below zero (recovering)")
    return Signal("MACD", hist, HOLD, "below signal line but still above zero (cooling off)")


def trend_signal(price, sma50, sma200) -> Signal:
    if price is None or sma50 is None or sma200 is None:
        return _na("SMA 50/200", "needs 200 trading days of history")
    if price > sma200 and sma50 > sma200:
        return Signal("SMA 50/200", sma200, BUY, "price > SMA200 and SMA50 > SMA200 (uptrend / golden cross)")
    if price < sma200 and sma50 < sma200:
        return Signal("SMA 50/200", sma200, SELL, "price < SMA200 and SMA50 < SMA200 (downtrend / death cross)")
    return Signal("SMA 50/200", sma200, HOLD, "mixed: price and SMA50 disagree about SMA200")


def peg_signal(peg: Optional[float], th: Thresholds) -> Signal:
    if peg is None:
        return _na("PEG", "no data from Yahoo")
    if peg <= 0:
        return _na("PEG", "negative: no earnings growth or losses")
    if peg < th.peg_buy:
        return Signal("PEG", peg, BUY, f"cheap relative to growth (< {th.peg_buy:g})")
    if peg > th.peg_sell:
        return Signal("PEG", peg, SELL, f"expensive relative to growth (> {th.peg_sell:g})")
    return Signal("PEG", peg, HOLD, f"fairly priced ({th.peg_buy:g}-{th.peg_sell:g})")


def roe_signal(roe_pct: Optional[float], th: Thresholds) -> Signal:
    if roe_pct is None:
        return _na("ROE", "no data from Yahoo")
    if roe_pct > th.roe_buy:
        return Signal("ROE", roe_pct, BUY, f"high return on equity (> {th.roe_buy:g}%)")
    if roe_pct < th.roe_sell:
        return Signal("ROE", roe_pct, SELL, f"weak return on equity (< {th.roe_sell:g}%)")
    return Signal("ROE", roe_pct, HOLD, f"acceptable ({th.roe_sell:g}-{th.roe_buy:g}%)")


def current_ratio_signal(cr: Optional[float], th: Thresholds) -> Signal:
    if cr is None:
        return _na("Current ratio", "no data (common for banks/insurers)")
    if cr > th.current_ratio_buy:
        return Signal("Current ratio", cr, BUY, f"strong liquidity (> {th.current_ratio_buy:g})")
    if cr < th.current_ratio_sell:
        return Signal("Current ratio", cr, SELL, f"liquidity risk (< {th.current_ratio_sell:g})")
    return Signal("Current ratio", cr, HOLD, f"adequate ({th.current_ratio_sell:g}-{th.current_ratio_buy:g})")


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
@dataclass
class StockReport:
    ticker: str
    name: str = ""
    currency: str = ""
    price: Optional[float] = None
    prev_close: Optional[float] = None
    day_low: Optional[float] = None
    day_high: Optional[float] = None
    wk52_low: Optional[float] = None
    wk52_high: Optional[float] = None
    rsi: Optional[float] = None
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist: Optional[float] = None
    sma50: Optional[float] = None
    sma200: Optional[float] = None
    pe: Optional[float] = None
    peg: Optional[float] = None
    roe_pct: Optional[float] = None
    current_ratio: Optional[float] = None
    signals: dict = field(default_factory=dict)   # key -> Signal
    score: int = 0
    rating: str = "N/A"
    error: Optional[str] = None

    @property
    def change_pct(self) -> Optional[float]:
        if self.price is None or not self.prev_close:
            return None
        return (self.price / self.prev_close - 1) * 100

    @property
    def off_52w_high_pct(self) -> Optional[float]:
        if self.price is None or not self.wk52_high:
            return None
        return (self.price / self.wk52_high - 1) * 100


def rating_for(score: int, available: int, th: Thresholds) -> str:
    if available < th.min_indicators:
        return "N/A"
    if score >= th.score_strong:
        return "STRONG BUY"
    if score >= th.score_buy:
        return "BUY"
    if score <= -th.score_strong:
        return "STRONG SELL"
    if score <= -th.score_buy:
        return "SELL"
    return "HOLD"


def build_report(ticker: str, name: Optional[str], hist: pd.DataFrame,
                 info: dict, th: Thresholds) -> StockReport:
    """Combine price history (OHLC DataFrame) and Yahoo `info` into a report."""
    info = info or {}
    close = hist["Close"].dropna()

    rep = StockReport(ticker=ticker)
    rep.name = name or info.get("longName") or info.get("shortName") or ticker
    rep.currency = info.get("currency") or info.get("financialCurrency") or ""

    # --- quote data (info first, fall back to the last candle) -------------
    last_bar = hist.iloc[-1]
    rep.price = first_num(info.get("currentPrice"), info.get("regularMarketPrice"), last(close))
    rep.prev_close = first_num(
        info.get("regularMarketPreviousClose"), info.get("previousClose"),
        float(close.iloc[-2]) if len(close) > 1 else None,
    )
    rep.day_low = first_num(info.get("dayLow"), info.get("regularMarketDayLow"), last_bar.get("Low"))
    rep.day_high = first_num(info.get("dayHigh"), info.get("regularMarketDayHigh"), last_bar.get("High"))
    year = hist.tail(252)
    rep.wk52_low = first_num(info.get("fiftyTwoWeekLow"), year["Low"].min() if "Low" in year else None)
    rep.wk52_high = first_num(info.get("fiftyTwoWeekHigh"), year["High"].max() if "High" in year else None)

    # --- technicals -----------------------------------------------------------
    rep.rsi = last(rsi(close))
    line, sig, hist_ = macd(close)
    rep.macd_line, rep.macd_signal, rep.macd_hist = last(line), last(sig), last(hist_)
    rep.sma50, rep.sma200 = last(sma(close, 50)), last(sma(close, 200))

    # --- fundamentals -----------------------------------------------------------
    rep.pe = first_num(info.get("trailingPE"))
    rep.peg = first_num(info.get("pegRatio"), info.get("trailingPegRatio"))
    if rep.peg is None:  # derive: P/E divided by earnings growth in %
        growth = num(info.get("earningsGrowth"))
        if rep.pe and growth and growth > 0:
            rep.peg = rep.pe / (growth * 100)
    roe = num(info.get("returnOnEquity"))
    rep.roe_pct = roe * 100 if roe is not None else None
    rep.current_ratio = num(info.get("currentRatio"))

    # --- signals & score ----------------------------------------------------------
    price_for_trend = last(close) if rep.price is None else rep.price
    rep.signals = {
        "rsi": rsi_signal(rep.rsi, th),
        "macd": macd_signal(rep.macd_line, rep.macd_signal, rep.macd_hist),
        "trend": trend_signal(price_for_trend, rep.sma50, rep.sma200),
        "peg": peg_signal(rep.peg, th),
        "roe": roe_signal(rep.roe_pct, th),
        "cr": current_ratio_signal(rep.current_ratio, th),
    }
    rep.score = sum(s.verdict for s in rep.signals.values())
    available = sum(1 for s in rep.signals.values() if s.available)
    rep.rating = rating_for(rep.score, available, th)
    return rep
