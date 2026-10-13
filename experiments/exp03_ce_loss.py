"""How much return does guessing cost you?"""

import numpy as np

from src.artificial_market import factor_model_known
from src.sample import draw_returns
from src.optimize import markowitz, oracle_weights, equal_weight
from src.metrics import ce_loss, certainty_equivalent
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, FuncFormatter


N_ASSETS, N_TRIALS = 25, 300
MONTHS = [60, 120, 240, 600]

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(exist_ok=True)

mu, Sigma = factor_model_known(n_assets=N_ASSETS, seed=42)
w_star = oracle_weights(mu, Sigma)
best_score = certainty_equivalent(w_star, mu, Sigma)

print(f"best possible score: {best_score:.5f}/month ({best_score*12:.2%}/year)\n")
print(f"{'months':>7} {'years':>6} {'Markowitz gives up':>20} {'even split gives up':>21} {'per year':>10}")
print("-" * 68)

markowitz_losses = {}
equal_losses = {}
for T in MONTHS:
    mk, eq = [], []
    for trial in range(N_TRIALS):
        rng = np.random.default_rng(1000 * trial + T)
        R = draw_returns(mu, Sigma, T, rng)
        mk.append(ce_loss(markowitz(R), w_star, mu, Sigma))
        eq.append(ce_loss(equal_weight(R), w_star, mu, Sigma))
    mk, eq = np.array(mk), np.array(eq)
    markowitz_losses[T] = mk
    equal_losses[T] = eq
    print(f"{T:>7} {T/12:>6.0f} {mk.mean():>20.5f} {eq.mean():>21.5f} {mk.mean()*12:>9.2%}")


all_losses = np.concatenate(list(markowitz_losses.values()))
print(f"\nnegative losses (must be 0): {(all_losses < -1e-12).sum()}")
# ---------- chart ----------
years = [T / 12 for T in MONTHS]
mk_annual = [markowitz_losses[T].mean() * 12 * 100 for T in MONTHS]
eq_annual = [equal_losses[T].mean() * 12 * 100 for T in MONTHS]

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#dcdcd8"

fig, ax = plt.subplots(figsize=(11, 6.2), dpi=150)
fig.patch.set_facecolor("white"); ax.set_facecolor("white")

ax.plot(years, mk_annual, color=ORANGE, lw=2.4, marker="o", ms=8, mec="white", mew=1.5)
ax.plot(years, eq_annual, color=BLUE, lw=2.4, marker="o", ms=8, mec="white", mew=1.5)

ax.text(years[-1] * 1.06, mk_annual[-1], " Markowitz", color=ORANGE,
        fontsize=12, va="center", fontweight="bold")
ax.text(years[-1] * 1.06, eq_annual[-1], " Even split", color=BLUE,
        fontsize=12, va="center", fontweight="bold")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks(years); ax.set_xticklabels([f"{y:.0f}" for y in years])
ax.set_xlabel("Years of history the strategy is given", fontsize=12, color=MUTED, labelpad=9)
ax.set_ylabel("Return given up, per year (%)  —  log scale", fontsize=12, color=MUTED, labelpad=9)
ax.set_title("Visualizations of losses",
             fontsize=14.5, color=INK, fontweight="bold", loc="left", pad=16)

ax.grid(color=GRID, lw=0.8, which="both"); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
ax.tick_params(colors=MUTED, labelsize=11)
ax.set_xlim(years[0] * 0.85, years[-1] * 1.5)


ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
fig.tight_layout()
fig.savefig(FIG_DIR / "ce_loss_vs_data.png", facecolor="white")
plt.show()