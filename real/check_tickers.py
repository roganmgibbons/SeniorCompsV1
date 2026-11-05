# file to check if tickers are availible via yFinance

import yfinance as yf

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# sample portfolio meant to balance realism as well as not be dictated by swings of one field
    # ie -> 6 tech, 3 finance, 3 energy, etc.

TICKERS = ["AAPL", "MSFT", "INTC", "CSCO", "ORCL", "IBM", "QCOM",
           "XOM", "CVX", "VLO",
           "JPM", "BAC", "AXP",
           "JNJ", "PFE", "MRK",
           "PG", "KO", "WMT",
           "HD", "MCD", "DIS",
           "CAT", "BA",
           "T"]

# monthly prices, adjusted for splits and dividends
prices = yf.download(TICKERS, start="1990-01-01", interval="1mo",
                     auto_adjust=True)["Close"]

print(f"{'ticker':>7} {'first month':>12} {'months':>7}")
for t in TICKERS:
    s = prices[t].dropna()
    print(f"{t:>7} {str(s.index[0].date()):>12} {len(s):>7}")

prices.to_csv(DATA_DIR / "real_prices.csv")
print("\nsaved to data/real_prices.csv")