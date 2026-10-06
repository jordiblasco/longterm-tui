# ltui – Long-Term Investor Dashboard (terminal)

A single-table terminal dashboard that scores every stock in a YAML watchlist
on long-term fundamentals and trend, using Yahoo Finance data (`yfinance`).
Inspired by stocksTUI, but built for buy/hold/sell decisions instead of intraday updates.

## Install & run

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

    python -m ltui watchlist.yaml          # interactive TUI
    python -m ltui watchlist.yaml --print  # print the table once and exit
    python -m ltui --demo                  # offline synthetic data, to preview the UI

Keys: `↑/↓` select a stock (reasoning shown below the table) · `r` refresh · `s` cycle sort · `q` quit.

## Watchlist (YAML)

    stocks:
      - AAPL
      - ticker: NESN.SW
        name: Nestlé          # optional display name
    settings:                 # optional
      period: 2y
      thresholds: { rsi_buy: 25, roe_buy: 12 }

Use Yahoo symbols: `VOD.L` (London), `FPH.NZ` (NZX), `BHP.AX` (ASX), `NESN.SW` (Swiss), `BRK-B`.

## How the signals work

Each indicator votes +1 (green / buy), 0 (plain / neutral) or -1 (red / sell).
Missing data votes 0 and shows `—`. The votes are summed into a score from -6 to +6.

| Indicator | Buy (+1) | Sell (-1) |
|---|---|---|
| RSI(14), Wilder | < 30 (oversold) | > 70 (overbought) |
| MACD (12/26/9) | histogram > 0 **and** MACD line > 0 | histogram < 0 **and** MACD line < 0 |
| SMA 50 / 200 | price > SMA200 **and** SMA50 > SMA200 | price < SMA200 **and** SMA50 < SMA200 |
| PEG | < 1 | > 2 (negative PEG = n/a) |
| ROE | > 15 % | < 8 % |
| Current ratio | > 1.5 | < 1.0 |

Rating: score ≥ +4 STRONG BUY · ≥ +2 BUY · ≤ -2 SELL · ≤ -4 STRONG SELL · otherwise HOLD
(`N/A` if fewer than 3 indicators have data). All cut-offs are editable under `settings.thresholds`.

## Notes & limits

* Yahoo's fundamentals are unofficial and sometimes missing (banks/insurers often have no
  current ratio; PEG is derived from P/E ÷ earnings growth when Yahoo omits it).
* Prices may be delayed ~15 min. London prices are quoted in pence (`GBp`).
* The score is a screening aid built on generic rules of thumb, not financial advice;
  sector context (e.g. high-debt utilities, cyclicals) matters.

## Layout

    ltui/analysis.py   indicators, signal rules, scoring (pure pandas)
    ltui/data.py       YAML loading, Yahoo Finance fetching (threaded), demo data
    ltui/render.py     colour/format of table cells and reasoning panel
    ltui/app.py        Textual UI
