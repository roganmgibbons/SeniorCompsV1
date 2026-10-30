# trains the model and compares it to the formula on markets it never saw

from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from src.metrics import ce_loss

N_ASSETS = 25
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# load
d = np.load(DATA_DIR / "train.npz")
X, y = d["X"], d["y"]
market_id = d["market_id"]
market_mu, market_Sigma = d["market_mu"], d["market_Sigma"]
w_markowitz = d["w_markowitz"]

print(f"{X.shape[0]:,} examples from {market_id.max() + 1:,} markets")

# split by market, not by row
# if samples from one market landed in both halves the model would have
# already seen that market's answer and the score would mean nothing
n_markets = market_id.max() + 1
rng = np.random.default_rng(0)
shuffled = rng.permutation(n_markets)
train_markets = set(shuffled[: int(0.8 * n_markets)].tolist())

is_train = np.array([m in train_markets for m in market_id])
print(f"train: {is_train.sum():,} rows   test: {(~is_train).sum():,} rows")

# scale
# fit only on training data, otherwise the test set leaks into the scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X[is_train])
X_test = scaler.transform(X[~is_train])

# fit
model = MLPRegressor(
    hidden_layer_sizes=(128, 64),
    alpha=1e-4,
    max_iter=500,
    early_stopping=True,
    random_state=0,
    verbose=True,
)
model.fit(X_train, y[is_train])
print(f"\ntrain r2: {model.score(X_train, y[is_train]):.4f}")
print(f"test  r2: {model.score(X_test, y[~is_train]):.4f}")

# predict
pred = model.predict(X_test)

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

# how far the predicted weights are from the oracle weights
mse = np.mean((pred - y[~is_train]) ** 2)
print(f"\nmse on held-out markets: {mse:.6f}")

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

# median matters here, if a few rows are catastrophic the mean hides it
print(f"\n{'strategy':>12} {'mean':>12} {'median':>12}  per year")
print("-" * 52)
for name, vals in losses.items():
    print(f"{name:>12} {np.mean(vals):>12.5f} {np.median(vals):>12.5f}  {np.mean(vals) * 12:>7.2%}")

# how extreme is the oracle portfolio in each test market
gross = np.array([np.abs(y[r]).sum() for r in test_rows])

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
        print(f"  {name:>12} {np.array(vals)[sel].mean():>10.5f}")

import matplotlib.pyplot as plt

# how extreme is the perfect portfolio in each test market
gross = np.array([np.abs(y[r]).sum() for r in test_rows])

# bin by how extreme, average the loss in each bin
bins = np.array([1, 2, 3, 4, 6, 8, 12, 20, 1000])
centers, curves = [], {k: [] for k in losses}

for lo, hi in zip(bins[:-1], bins[1:]):
    sel = (gross >= lo) & (gross < hi)
    if sel.sum() < 20:
        continue
    centers.append(gross[sel].mean())
    for k in losses:
        curves[k].append(np.array(losses[k])[sel].mean())

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
ax.set_title("where the model fails", fontsize=14.5, color=INK,
             fontweight="bold", loc="left", pad=16)
ax.grid(color=GRID, lw=0.8, which="both"); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
ax.tick_params(colors=MUTED, labelsize=11)
ax.legend(frameon=False, fontsize=11)

fig.tight_layout()
plt.show()