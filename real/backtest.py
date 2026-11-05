# tests the trained model on real stocks
#
# the model never trains on real data. it was trained on fake markets,
# and here it sees real history for the first time:
#
#   every month: take the past 10 years of real monthly returns,
#   compute the same 351 numbers as in training, let each strategy
#   pick 25 weights, hold them for one month, record what actually happened.
#   then slide forward a month and repeat.

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from ml.dataset import make_features
from src.estimate import sample_mean
from src.optimize import markowitz, mean_variance_weights, GAMMA

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"

# settings
MODEL_FILE = DATA_DIR / "model_100k.joblib"
LOOKBACK = 120          # months of history each decision sees (10 years)
N_ASSETS = 25


# ---------------------------------------------------------------- prices -> returns
prices = pd.read_csv(DATA_DIR / "real_prices.csv", index_col=0, parse_dates=True)

# the current month is not over yet, so its "monthly" price is partial
this_month = pd.Timestamp.today().to_period("M").to_timestamp()
prices = prices[prices.index < this_month]

# monthly return = this month's price / last month's price - 1
# dropna removes the early months where a stock had not started trading yet
returns = prices.pct_change().dropna()
assert returns.shape[1] == N_ASSETS, "expected 25 stocks"

print(f"{len(returns)} months of returns, {returns.index[0]:%b %Y} to {returns.index[-1]:%b %Y}")
print(f"first portfolio: {returns.index[LOOKBACK]:%b %Y} (after {LOOKBACK} months of history)")


# ---------------------------------------------------------------- strategies
saved = joblib.load(MODEL_FILE)
model, scaler = saved["model"], saved["scaler"]


def model_weights(R):
    x = scaler.transform([make_features(R)])        # same 351 numbers, same rescaling
    w = model.predict(x)[0]
    return w + (1.0 - w.sum()) / N_ASSETS           # sum to 100%, same fix as train.py


def even_split(R):
    return np.full(N_ASSETS, 1.0 / N_ASSETS)


def ledoit_wolf(R):
    S_lw = LedoitWolf().fit(R).covariance_          # shrunk covariance, the standard fix
    return mean_variance_weights(sample_mean(R), S_lw, GAMMA)


STRATEGIES = {
    "model": model_weights,
    "even split": even_split,
    "markowitz": markowitz,
    "ledoit-wolf": ledoit_wolf,
}


# ---------------------------------------------------------------- rolling backtest
dates = returns.index[LOOKBACK:]
realized = {name: [] for name in STRATEGIES}       # each month's actual return
weights = {name: [] for name in STRATEGIES}        # each month's chosen weights
scaled_features = []

R_all = returns.values
for t in range(LOOKBACK, len(R_all)):
    R = R_all[t - LOOKBACK:t]        # the past 10 years, nothing from the future
    r_next = R_all[t]                # what actually happened the following month

    scaled_features.append(scaler.transform([make_features(R)])[0])

    for name, pick in STRATEGIES.items():
        w = pick(R)
        weights[name].append(w)
        realized[name].append(w @ r_next)

realized = {k: np.array(v) for k, v in realized.items()}
weights = {k: np.array(v) for k, v in weights.items()}


# ---------------------------------------------------------------- does real data look like training data?
# after rescaling, training features sit mostly between -3 and +3.
# lots of real values outside that range = the model is seeing something unfamiliar
z = np.abs(np.array(scaled_features))
print(f"\nreal features outside the training range (|z| > 3): {(z > 3).mean():.1%}")


# ---------------------------------------------------------------- scorecard
def max_drawdown(r):
    wealth = np.cumprod(1 + r)
    if np.any(wealth <= 0):
        return -1.0                                  # lost everything at some point
    peak = np.maximum.accumulate(wealth)
    return (wealth / peak - 1).min()


print(f"\n{len(dates)} months tested, {dates[0]:%b %Y} to {dates[-1]:%b %Y}\n")
header = f"{'strategy':>12} {'return/yr':>10} {'vol/yr':>8} {'sharpe':>7} {'CE/yr':>8} {'worst drop':>11} {'turnover':>9} {'leverage':>9}"
print(header)
print("-" * len(header))

for name, r in realized.items():
    ann_ret = r.mean() * 12
    ann_vol = r.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol
    # same score as the simulation: return minus a risk penalty (gamma = 3)
    ce = (r.mean() - GAMMA / 2 * r.var()) * 12
    turnover = np.abs(np.diff(weights[name], axis=0)).sum(axis=1).mean()
    leverage = np.abs(weights[name]).sum(axis=1).mean()
    print(f"{name:>12} {ann_ret:>10.2%} {ann_vol:>8.2%} {sharpe:>7.2f} {ce:>8.2%} "
          f"{max_drawdown(r):>11.1%} {turnover:>9.2f} {leverage:>9.2f}")

print("\nsharpe here is return / volatility (no risk-free rate subtracted)")
print("turnover = average share of the portfolio traded each month (1.00 = 100%)")
print("leverage = average sum of absolute weights (1.00 = no shorting or borrowing)")


# ---------------------------------------------------------------- the model's latest portfolio
latest = pd.Series(weights["model"][-1], index=returns.columns).sort_values(ascending=False)
print(f"\nmodel's portfolio for {dates[-1]:%b %Y}:")
for ticker, w in latest.items():
    print(f"  {ticker:>5} {w:>7.1%}")


# ---------------------------------------------------------------- chart: growth of $1
COLORS = {"model": "#4a3aa7", "even split": "#2a78d6", "markowitz": "#eb6834", "ledoit-wolf": "#2f8f5b"}
MUTED, GRID, INK = "#52514e", "#dcdcd8", "#0b0b0b"

fig, ax = plt.subplots(figsize=(11, 6.2), dpi=150)
for name, r in realized.items():
    wealth = np.cumprod(1 + r)
    if np.any(wealth <= 0):                          # a strategy that went bust stops there
        wealth[np.argmax(wealth <= 0):] = np.nan
    ax.plot(dates, wealth, color=COLORS[name], lw=2.2, label=name)

ax.set_yscale("log")
ax.set_ylabel("value of $1 invested (log scale)", fontsize=12, color=MUTED)
ax.set_title("real stocks: growth of $1, each strategy re-chosen monthly",
             fontsize=14.5, color=INK, fontweight="bold", loc="left", pad=16)
ax.grid(color=GRID, lw=0.8, which="both"); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
ax.tick_params(colors=MUTED, labelsize=11)
ax.legend(frameon=False, fontsize=11)
fig.tight_layout()
fig.savefig(FIG_DIR / "real_backtest.png", facecolor="white")
print(f"\nsaved chart to {FIG_DIR / 'real_backtest.png'}")
plt.show()
