# micro-experiments on real stocks with the trained model
#
#   1. long-only versions of every strategy (no shorting, no borrowing)
#   2. minimum-variance (ignores average returns entirely)
#   3. different lookback windows (5, 10, 20 years of history)
#   4. different time periods (calm years vs 2008 vs 2020)
#
# usage:  python -m real.experiments
#         python -m real.experiments --prices real_prices_picks.csv   (another portfolio)

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from ml.dataset import make_features
from src.estimate import sample_mean, sample_covariance
from src.optimize import mean_variance_weights, GAMMA

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

parser = argparse.ArgumentParser()
parser.add_argument("--prices", default="real_prices.csv")
parser.add_argument("--model", default="model_100k.joblib")
parser.add_argument("--lookback", type=int, default=120, help="months of history per decision")
args = parser.parse_args()


# ---------------------------------------------------------------- data
prices = pd.read_csv(DATA_DIR / args.prices, index_col=0, parse_dates=True)
this_month = pd.Timestamp.today().to_period("M").to_timestamp()
prices = prices[prices.index < this_month]          # drop the unfinished month
returns = prices.pct_change().dropna()
N = returns.shape[1]
print(f"{args.prices}: {N} stocks, {len(returns)} months, "
      f"{returns.index[0]:%b %Y} to {returns.index[-1]:%b %Y}")

saved = joblib.load(DATA_DIR / args.model)
model, scaler = saved["model"], saved["scaler"]
if N != 25:
    raise SystemExit("the model was trained on 25 stocks, so the portfolio must have exactly 25")


# ---------------------------------------------------------------- helpers
def long_only_mv(mu, S, gamma=GAMMA):
    """best return-minus-risk portfolio with no shorting: weights >= 0, sum to 1"""
    obj = lambda w: -(w @ mu - gamma / 2 * w @ S @ w)
    grad = lambda w: -(mu - gamma * S @ w)
    w0 = np.full(len(mu), 1 / len(mu))
    res = minimize(obj, w0, jac=grad, method="SLSQP",
                   bounds=[(0, 1)] * len(mu),
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    return res.x


def long_only_minvar(S):
    """lowest-risk portfolio with no shorting"""
    return long_only_mv(np.zeros(len(S)), S, gamma=2.0)


def clip_long(w):
    """drop the shorts, rescale the rest to 100%"""
    w = np.clip(w, 0, None)
    return w / w.sum()


def model_w(R):
    w = model.predict(scaler.transform([make_features(R)]))[0]
    return w + (1 - w.sum()) / N


# every strategy gets the same history R and returns 25 weights
def all_strategies(R):
    mu = sample_mean(R)
    S = sample_covariance(R)
    S_lw = LedoitWolf().fit(R).covariance_
    ones = np.ones(N)
    w_model = model_w(R)
    minvar = np.linalg.solve(S_lw, ones); minvar /= minvar.sum()
    return {
        # unconstrained (can short and borrow)
        "model": w_model,
        "markowitz": mean_variance_weights(mu, S),
        "ledoit-wolf": mean_variance_weights(mu, S_lw),
        "min-variance": minvar,
        # long-only (no shorting, no borrowing)
        "model LO": clip_long(w_model),
        "markowitz LO": long_only_mv(mu, S),
        "ledoit-wolf LO": long_only_mv(mu, S_lw),
        "min-variance LO": long_only_minvar(S_lw),
        "even split": np.full(N, 1 / N),
    }


def run(lookback):
    """rolling backtest: returns a DataFrame of realized monthly returns per strategy"""
    R_all = returns.values
    rows, weights = [], []
    for t in range(lookback, len(R_all)):
        ws = all_strategies(R_all[t - lookback:t])
        rows.append({k: w @ R_all[t] for k, w in ws.items()})
        weights.append(ws)
    realized = pd.DataFrame(rows, index=returns.index[lookback:])
    return realized, weights


def scorecard(realized, weights=None):
    out = {}
    for name in realized.columns:
        r = realized[name].values
        wealth = np.cumprod(1 + r)
        bust = np.any(wealth <= 0)
        peak = np.maximum.accumulate(np.where(bust, np.nan, wealth))
        row = {
            "return/yr": r.mean() * 12,
            "vol/yr": r.std() * np.sqrt(12),
            "sharpe": r.mean() / r.std() * np.sqrt(12),
            "CE/yr": (r.mean() - GAMMA / 2 * r.var()) * 12,
            "worst drop": -1.0 if bust else (wealth / peak - 1).min(),
            "$1 grows to": 0.0 if bust else wealth[-1],
        }
        if weights is not None:
            W = np.array([w[name] for w in weights])
            row["turnover"] = np.abs(np.diff(W, axis=0)).sum(axis=1).mean()
            row["leverage"] = np.abs(W).sum(axis=1).mean()
        out[name] = row
    return pd.DataFrame(out).T


def show(df, title):
    fmt = {"return/yr": "{:.2%}", "vol/yr": "{:.2%}", "sharpe": "{:.2f}", "CE/yr": "{:.2%}",
           "worst drop": "{:.1%}", "$1 grows to": "{:.2f}", "turnover": "{:.2f}", "leverage": "{:.2f}"}
    print(f"\n=== {title} ===")
    print(df.apply(lambda col: col.map(fmt.get(col.name, "{}").format)).to_string())


# ---------------------------------------------------------------- experiment 1 + 2
realized, weights = run(lookback=args.lookback)
show(scorecard(realized, weights),
     f"EXP 1-2: all strategies, {args.lookback // 12}-year lookback, "
     f"{realized.index[0]:%b %Y} - {realized.index[-1]:%b %Y}")


# ---------------------------------------------------------------- experiment 3: lookback
KEEP = ["model", "model LO", "ledoit-wolf LO", "min-variance LO", "even split"]
for lb in [60, 240]:
    if lb == args.lookback or len(returns) <= lb + 24:      # skip if too little data
        continue
    r_lb, w_lb = run(lookback=lb)
    show(scorecard(r_lb[KEEP], [{k: w[k] for k in KEEP} for w in w_lb]),
         f"EXP 3: {lb // 12}-year lookback, {r_lb.index[0]:%b %Y} - {r_lb.index[-1]:%b %Y}")


# ---------------------------------------------------------------- experiment 4: time periods
PERIODS = {
    "2002-2007 (pre-crisis)": ("2002-01", "2007-12"),
    "2008-2009 (financial crisis)": ("2008-01", "2009-12"),
    "2010-2019 (long bull market)": ("2010-01", "2019-12"),
    "2020-2026 (COVID and after)": ("2020-01", "2026-12"),
}
for label, (a, b) in PERIODS.items():
    part = realized.loc[a:b, KEEP]
    if len(part) < 12:                                      # period not covered by this data
        continue
    show(scorecard(part), f"EXP 4: {label}")
