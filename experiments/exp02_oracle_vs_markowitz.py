"""the best possible portfolio vs what the formula picks
when it only has ten years of data."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.artificial_market import factor_model_known
from src.sample import draw_returns
from src.optimize import markowitz, oracle_weights

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(exist_ok=True)
N_ASSETS, T, SEED = 25, 120, 7

mu, Sigma = factor_model_known(n_assets=N_ASSETS, seed=42)
rng = np.random.default_rng(SEED)
R = draw_returns(mu, Sigma, T, rng)

w_star = oracle_weights(mu, Sigma)
w_hat = markowitz(R)

print(f"{'asset':>6} {'BEST POSSIBLE':>15} {'WHAT IT PICKS':>15} {'off by':>10}")
print("-" * 50)
for i in range(N_ASSETS):
    print(f"{i:>6} {w_star[i]:>15.2%} {w_hat[i]:>15.2%} {w_hat[i]-w_star[i]:>+10.2%}")
print("-" * 50)
print(f"{'total':>6} {w_star.sum():>15.2%} {w_hat.sum():>15.2%}")
print(f"{'size':>6} {np.abs(w_star).sum():>15.2f} {np.abs(w_hat).sum():>15.2f}")

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#dcdcd8"
x = np.arange(N_ASSETS)
w = 0.4

fig, ax = plt.subplots(figsize=(13, 6), dpi=150)
fig.patch.set_facecolor("white"); ax.set_facecolor("white")

ax.bar(x - w/2, w_star * 100, w, color=BLUE, label="Best possible (knows the answer)")
ax.bar(x + w/2, w_hat * 100, w, color=ORANGE, label="Markowitz (10 years of data)")
ax.axhline(0, color=MUTED, lw=1.2)

ax.set_xlabel("Stock", fontsize=12, color=MUTED)
ax.set_ylabel("Share of the portfolio (%)", fontsize=12, color=MUTED)
ax.set_title("Same market, same formula — only the inputs were guessed",
             fontsize=14, color=INK, fontweight="bold", loc="left", pad=14)
ax.set_xticks(x); ax.tick_params(colors=MUTED, labelsize=10)
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
ax.legend(frameon=False, fontsize=11)

fig.tight_layout()
fig.savefig(FIG_DIR / "oracle_vs_markowitz.png", facecolor="white")