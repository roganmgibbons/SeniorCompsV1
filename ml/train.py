# trains the model and compares it to the formula on markets it never saw

import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from src.metrics import ce_loss

N_ASSETS = 25
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"

# settings, change these between runs
ALPHA = 1e-1               # L2 penalty against memorizing (not the market's alpha)
HIDDEN = (128, 64)         # network size: two layers of 128 and 64 neurons


# ---------------------------------------------------------------- load
d = np.load(DATA_DIR / "train.npz")
X, y = d["X"], d["y"]
market_id = d["market_id"]
market_mu, market_Sigma = d["market_mu"], d["market_Sigma"]
w_markowitz = d["w_markowitz"]

n_markets = market_id.max() + 1
print(f"{X.shape[0]:,} examples from {n_markets:,} markets")

# the saved model is named after the data size, e.g. model_100k.joblib,
# so a run can never be saved under the wrong name
MODEL_FILE = DATA_DIR / f"model_{n_markets // 1000}k.joblib"


# ---------------------------------------------------------------- split
# split by market, not by row
# if samples from one market landed in both halves the model would have
# already seen that market's answer and the score would mean nothing
rng = np.random.default_rng(0)
shuffled = rng.permutation(n_markets)
train_markets = shuffled[: int(0.8 * n_markets)]

is_train = np.isin(market_id, train_markets)
print(f"train: {is_train.sum():,} rows   test: {(~is_train).sum():,} rows")


# ---------------------------------------------------------------- scale
# fit only on training data, otherwise the test set leaks into the scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X[is_train])
X_test = scaler.transform(X[~is_train])
y_train, y_test = y[is_train], y[~is_train]


# ---------------------------------------------------------------- fit
model = MLPRegressor(
    hidden_layer_sizes=HIDDEN,
    alpha=ALPHA,
    max_iter=500,
    early_stopping=True,
    random_state=0,
    verbose=True,
)

start = time.perf_counter()
model.fit(X_train, y_train)
train_seconds = time.perf_counter() - start


# ---------------------------------------------------------------- train vs test
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

r2_train = model.score(X_train, y_train)
r2_test = model.score(X_test, y_test)
mse_train = np.mean((pred_train - y_train) ** 2)
mse_test = np.mean((pred_test - y_test) ** 2)

print(f"\nstopped after {model.n_iter_} iterations, {train_seconds:.0f} seconds")
print(f"\n{'':>6} {'r2':>8} {'mse':>10}")
print(f"{'train':>6} {r2_train:>8.4f} {mse_train:>10.6f}")
print(f"{'test':>6} {r2_test:>8.4f} {mse_test:>10.6f}")


# ---------------------------------------------------------------- save
# save the model AND the scaler: real data must be rescaled
# exactly the way the training data was
joblib.dump({"model": model, "scaler": scaler}, MODEL_FILE)
print(f"\nsaved model to {MODEL_FILE}")


# ---------------------------------------------------------------- weights
pred = pred_test

# check the raw sums before touching anything
# every target sums to 1 so these should cluster near 1.0
sums = pred.sum(axis=1)
print(f"\nraw prediction sums:")
print(f"  min {sums.min():.4f}  max {sums.max():.4f}")
print(f"  median {np.median(sums):.4f}")
print(f"  how many below 0.5: {(np.abs(sums) < 0.5).sum()}")

# force the weights to sum to 1 without dividing, so a near-zero sum
# cannot blow the weights up
pred = pred + (1.0 - pred.sum(axis=1, keepdims=True)) / N_ASSETS


# ---------------------------------------------------------------- ce loss
# the real test, how much return each strategy gives up
test_rows = np.where(~is_train)[0]
eq = np.full(N_ASSETS, 1.0 / N_ASSETS)
losses = {"model": [], "markowitz": [], "even split": []}

for row, w_pred in zip(test_rows, pred):
    m = market_id[row]
    mu_t = market_mu[m]
    Sigma_t = market_Sigma[m]
    w_star = y[row]

    losses["model"].append(ce_loss(w_pred, w_star, mu_t, Sigma_t))
    losses["markowitz"].append(ce_loss(w_markowitz[row], w_star, mu_t, Sigma_t))
    losses["even split"].append(ce_loss(eq, w_star, mu_t, Sigma_t))

losses = {k: np.array(v) for k, v in losses.items()}

# median matters here, if a few rows are catastrophic the mean hides it
print(f"\n{'strategy':>12} {'mean':>12} {'median':>12}  per year")
print("-" * 52)
for name, vals in losses.items():
    print(f"{name:>12} {vals.mean():>12.5f} {np.median(vals):>12.5f}  {vals.mean() * 12:>7.2%}")


# ---------------------------------------------------------------- by leverage
# how extreme is the oracle portfolio in each test market
gross = np.abs(y[test_rows]).sum(axis=1)

print(f"\noracle gross leverage across test rows:")
for q in [50, 75, 90, 95, 99, 100]:
    print(f"  {q}th percentile: {np.percentile(gross, q):>8.2f}")

# split the test set by how extreme the market is, score each bucket
edges = [0, 2, 4, 8, np.inf]
for lo, hi in zip(edges[:-1], edges[1:]):
    sel = (gross >= lo) & (gross < hi)
    if sel.sum() == 0:
        continue
    print(f"\noracle leverage {lo} to {hi}   ({sel.sum():,} rows)")
    for name, vals in losses.items():
        print(f"  {name:>12} {vals[sel].mean():>10.5f}")


# ---------------------------------------------------------------- summary
# one line to copy into the results table in PROGRESS.md
print(f"\nSUMMARY  markets={n_markets:,}  alpha={ALPHA}  hidden={HIDDEN}  "
      f"iters={model.n_iter_}  secs={train_seconds:.0f}  "
      f"r2 train/test={r2_train:.3f}/{r2_test:.3f}  "
      f"mse train/test={mse_train:.5f}/{mse_test:.5f}  "
      f"model={losses['model'].mean() * 12:.2%}  "
      f"even={losses['even split'].mean() * 12:.2%}  "
      f"markowitz={losses['markowitz'].mean() * 12:.2%}")


# ---------------------------------------------------------------- plot
# bin by how extreme the perfect portfolio is, average the loss in each bin
bins = np.array([1, 2, 3, 4, 6, 8, 12, 20, 1000])
centers, curves = [], {k: [] for k in losses}

for lo, hi in zip(bins[:-1], bins[1:]):
    sel = (gross >= lo) & (gross < hi)
    if sel.sum() < 20:
        continue
    centers.append(gross[sel].mean())
    for k in losses:
        curves[k].append(losses[k][sel].mean())

BLUE, ORANGE, VIOLET = "#2a78d6", "#eb6834", "#4a3aa7"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdcd8"

fig, ax = plt.subplots(figsize=(11, 6.2), dpi=150)
fig.patch.set_facecolor("white"); ax.set_facecolor("white")

for k, c in [("model", VIOLET), ("markowitz", ORANGE), ("even split", BLUE)]:
    ax.plot(centers, curves[k], color=c, lw=2.4, marker="o", ms=7,
            mec="white", mew=1.5, label=k)

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("how extreme the perfect portfolio is (sum of absolute weights)",
              fontsize=12, color=MUTED, labelpad=9)
ax.set_ylabel("return given up, per month", fontsize=12, color=MUTED, labelpad=9)
ax.set_title("where the model struggles: loss by how extreme the market is",
             fontsize=14.5, color=INK, fontweight="bold", loc="left", pad=16)
ax.grid(color=GRID, lw=0.8, which="both"); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
ax.tick_params(colors=MUTED, labelsize=11)
ax.legend(frameon=False, fontsize=11)

fig.tight_layout()
fig.savefig(FIG_DIR / f"loss_by_leverage_{n_markets // 1000}k.png", facecolor="white")
plt.show()   # last line: the script pauses here until you close the window
