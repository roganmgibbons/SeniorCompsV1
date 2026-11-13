# how much variance is in each parameter the generator draws?
# compares every draw (the rules) with the markets that survived the leverage filter
# (what the model actually trained on)

from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
d = np.load(DATA_DIR / "train.npz")          # loads only the arrays we ask for

kept_flag = d["draw_kept"]
print(f"{len(kept_flag):,} draws, {kept_flag.sum():,} kept "
      f"({kept_flag.mean():.1%}), {(~kept_flag).sum():,} rejected by the leverage filter\n")

# name on screen, key in train.npz, how to print it
PARAMS = [
    ("market volatility (per market)",      "market_vol", "pct"),
    ("typical stock-specific vol (per mkt)", "idio_level", "pct"),
    ("beta (per stock)",                     "beta",       "num"),
    ("stock-specific volatility (per stock)", "idio_vol",  "pct"),
    ("alpha, hidden bonus (per stock)",      "alpha",      "pct"),
]

def fmt(x, kind):
    return f"{x:.1%}" if kind == "pct" else f"{x:.2f}"

cols = ["mean", "sd", "variance", "p10", "p50", "p90"]
print(f"{'parameter':40} {'':8}" + "".join(f"{c:>10}" for c in cols))
print("-" * 108)

for name, key, kind in PARAMS:
    for label, x in [("all", d[f"draw_{key}"]), ("kept", d[f"param_{key}"])]:
        x = np.asarray(x).ravel()           # per-stock params: every stock of every market
        stats = [x.mean(), x.std(), x.var()] + list(np.percentile(x, [10, 50, 90]))
        # variance is in squared units, so always print it as a plain number
        shown = [fmt(s, kind) for s in stats[:2]] + [f"{stats[2]:.4f}"] + [fmt(s, kind) for s in stats[3:]]
        print(f"{name if label == 'all' else '':40} {label:8}" + "".join(f"{v:>10}" for v in shown))
    print()

# a stock's total volatility combines all three ingredients:
#   sqrt(beta^2 * market_vol^2 + stock_specific_vol^2)
for label, pre in [("all", "draw_"), ("kept", "param_")]:
    beta, idio = d[pre + "beta"], d[pre + "idio_vol"]
    mvol = d[pre + "market_vol"][:, None]
    total = np.sqrt(beta**2 * mvol**2 + idio**2).ravel()
    print(f"total stock volatility ({label:4}):  mean {total.mean():.1%}  sd {total.std():.1%}  "
          f"p10 {np.percentile(total, 10):.1%}  p50 {np.percentile(total, 50):.1%}  "
          f"p90 {np.percentile(total, 90):.1%}")