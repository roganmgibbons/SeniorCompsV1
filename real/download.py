# downloads monthly prices for different portfolios ("universes") from yfinance
#
# usage:  python -m real.download main      -> data/real_prices.csv (your 25)
#         python -m real.download picks     -> data/real_prices_picks.csv
#         python -m real.download pool      -> data/real_prices_pool.csv
#         python -m real.download spy       -> data/real_prices_spy.csv
#         python -m real.download all       -> all of the above

import sys
from pathlib import Path

import yfinance as yf

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

UNIVERSES = {
    # the main test: sector-balanced, all trading since ~1990
    "main": ["AAPL", "MSFT", "INTC", "CSCO", "ORCL", "IBM", "QCOM",
             "XOM", "CVX", "VLO", "JPM", "BAC", "AXP", "JNJ", "PFE", "MRK",
             "PG", "KO", "WMT", "HD", "MCD", "DIS", "CAT", "BA", "T"],

    # "an investor's picks": popular, tech/energy heavy. hindsight built in,
    # this is a stress test, not the main result. younger stocks = shorter test
    "picks": ["NVDA", "GOOGL", "META", "AMD", "AAPL", "MSFT", "AMZN", "AVGO",
              "ORCL", "CRM", "ADBE", "NFLX", "TSLA", "QCOM", "INTC",
              "CVX", "XOM", "VLO", "TPL", "COP",
              "JPM", "UNH", "LLY", "COST", "V"],

    # a bigger pool of large, long-lived companies across sectors.
    # random 25-stock portfolios get drawn from this to test robustness
    "pool": ["AAPL", "MSFT", "INTC", "CSCO", "ORCL", "IBM", "QCOM", "TXN", "ADI", "AMAT", "HPQ",
             "XOM", "CVX", "VLO", "COP", "OXY", "SLB", "HAL",
             "JPM", "BAC", "AXP", "WFC", "C", "USB", "PNC",
             "JNJ", "PFE", "MRK", "ABT", "BMY", "LLY", "AMGN", "MDT",
             "PG", "KO", "WMT", "PEP", "CL", "MO", "KMB",
             "HD", "MCD", "DIS", "NKE", "LOW", "TGT",
             "CAT", "BA", "MMM", "HON", "GE", "UNP", "DE",
             "T", "VZ", "SO", "DUK", "NEE", "AEP", "ED",
             "APD", "SHW"],

    # the S&P 500 index fund, as a real-world reference line
    "spy": ["SPY"],
}

FILES = {"main": "real_prices.csv", "picks": "real_prices_picks.csv",
         "pool": "real_prices_pool.csv", "spy": "real_prices_spy.csv"}


def download(name):
    tickers = UNIVERSES[name]
    prices = yf.download(tickers, start="1990-01-01", interval="1mo",
                         auto_adjust=True, progress=False)["Close"]
    if len(tickers) == 1:                       # a single ticker comes back as a Series
        prices = prices.to_frame(tickers[0]) if prices.ndim == 1 else prices

    print(f"\n{name}: {len(tickers)} tickers")
    print(f"{'ticker':>7} {'first month':>12} {'months':>7}")
    for t in tickers:
        s = prices[t].dropna() if t in prices else []
        if len(s) == 0:
            print(f"{t:>7} {'NO DATA':>12}")
        else:
            print(f"{t:>7} {str(s.index[0].date()):>12} {len(s):>7}")

    path = DATA_DIR / FILES[name]
    prices.to_csv(path)
    print(f"saved to {path}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name in (UNIVERSES if which == "all" else [which]):
        download(name)
