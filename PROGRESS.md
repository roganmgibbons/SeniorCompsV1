# SeniorComps progress log

Last updated: Thu Oct 29, 2026

## Thesis (keep this in mind for every change)

Markowitz portfolios fail because of estimation error, not because the formula is wrong.
In a simulated market where the true answer is known, I measure how much return that
error costs, and test whether a model trained across many markets can recover part of it,
compared against the best possible (Bayes floor) and the standard fix (Ledoit-Wolf).

Every task should support one of:
- A. Estimation error costs a measurable amount
- B. A model can recover some of it
- C. It holds up outside training conditions (crisis markets, new market types, real data)

## Pipeline (what goes in, what comes out)

1. `artificial_market.py`: build a market (25 stocks, random betas, vols, alphas) -> true mu, Sigma
2. `optimize.py` `oracle_weights`: Markowitz with TRUE mu, Sigma -> w* (answer key, 25 weights)
3. `sample.py` `draw_returns`: 3 independent histories per market, 5-50 years of monthly RETURNS (no prices)
4. `dataset.py` `make_features`: each history -> 351 numbers (25 means, 25 vols, 300 correlations, # months)
5. `train.py`: MLP takes the 351 numbers ONLY -> 25 weights (sum to 100%)
6. Training target = w*. Grading = CE loss vs w* using true mu, Sigma (`metrics.py`)

Diagram: `figures/flow_inputs_outputs.png`

## Changes made

- Seed fix in `dataset.py`: seeds drawn from the function's own rng (crisis set no longer shares
  random numbers with training markets); RuntimeError if a market can't meet the leverage cap
- Regularization in `train.py`: `alpha=1e-1` (best of 1e-4, 1e-2, 1e-1, 1.0)
- Data scaled up: `n_markets=60_000` (180,000 rows), `samples_per_market=3`

## Results so far (held-out markets, return given up per year)

| Run | Test R2 | Model | Even split | Markowitz |
|---|---|---|---|---|
| 20k markets, alpha 1e-4 | 0.046 | 4.51% | 5.06% | 41.76% |
| 20k markets, alpha 1e-2 | 0.118 | 3.92% | 5.06% | 41.76% |
| 20k markets, alpha 1e-1 | 0.198 | 3.54% | 5.06% | 41.76% |
| 20k markets, alpha 1.0 | 0.090 | 3.98% | 5.06% | 41.76% |
| **60k markets, alpha 1e-1** | **0.228** | **3.39%** | 5.05% | 42.13% |

- Bayes floor (best possible without knowing the future): 2.66%/yr on the 20k test set
- Gap closed (even split -> floor): 63% at 20k markets; roughly 69% at 60k (estimate, floor not yet recomputed on the new test set)
- Diagnosis: ~85% of the variation in w* comes from alpha, which can't be detected from history (Merton 1980). With alpha switched off, the same pipeline reaches test R2 0.72.
- Crisis vs normal markets: average stock correlation 0.72 vs 0.23; average stock vol about the same (~40%/yr)

## Known issues / caveats

- Regularization was chosen on the test set, so reported numbers are slightly optimistic -> need a separate validation set
- `early_stopping=True` validates on random ROWS, not markets (leaky, ~3x too optimistic) -> hand-roll a market-based validation split
- Bayes floor code lives outside the repo -> add `experiments/exp04_bayes_floor.py`; store betas/alphas in `dataset.py` so the floor is exact
- All markets are one-factor (professor's feedback) -> may not transfer to real data
- Crisis set doesn't raise total volatility like 2008/2020 did
- Only one training run per setting -> no error bars yet

## Next steps

1. Commit current state: `git add ml/ && git commit -m "60k markets, alpha=1e-1"`
2. Try 100,000 markets (does performance keep improving with data?)
3. Try `alpha=1e-2` at 60k markets
4. Add formula weights as features (Markowitz, minimum-variance, Ledoit-Wolf) in `make_features`
5. Market-based validation split
6. Ledoit-Wolf baseline (moves Goal 2 to Good)
7. Crisis stress test on `stress.npz`
8. Multi-factor markets + "unseen market type" test
9. Real data (yfinance): prices -> monthly returns -> same 351 features, rolling window
10. Poster (early December)

## Rubric status

- Goal 1 (fake market): Good
- Goal 2 (classical baselines): Passing
- Goal 3a (model beats formulas): Passing -> Good
- Goal 3b (understanding): Good
- Goal 4 (real data): Not started

## Notes for the final paper

Methods / data
- Inputs: no prices. Each history (5-50 years of monthly returns) is summarized into 351 numbers
  (25 means, 25 vols, 300 correlations, # months). The model sees only these; true mu, Sigma are
  used only to build the answer key (w*) and to grade. The model gets the same information as
  Markowitz and Ledoit-Wolf, so the comparison is fair.
- In the simulation, returns are drawn the same way every month (i.i.d. normal), so the sample
  mean and covariance are sufficient statistics: raw sequences would add no information. Real data
  has trends and volatility regimes, so sequence models (e.g. Zhang et al.'s LSTM) are a possible extension.
- 3 independent histories per market: same answer key, different luck. Train/test split is by
  market so no market appears in both. Untested choice: compare 1 / 3 / 10 histories per market.
- Datasets are nested: with seed=0, the first 60,000 markets of the 100k dataset are identical to
  the 60k dataset, so the scaling comparison only adds data.

Data-scaling experiment (20k -> 60k -> 100k markets, alpha=1e-1, same network)
- Report train/test R2, train/test MSE, model loss/yr, even split loss/yr, iterations, training time.
- Caveat: each run's test set is 20% of its own markets; even split staying ~5.05% shows they are
  comparable. Stricter version: one fixed test set for every size.
- Memory: ~2-2.5 GB peak at 100k markets (300k rows); scales linearly (~5 GB at 200k, ~12 GB at 500k).
  Fixes if needed: float32, drop unscaled copy, chunked generation.

Real data
- 25 stocks, sector-balanced, all trading since ~1990 (QCOM from Dec 1991, CSCO from Feb 1990):
  AAPL MSFT INTC CSCO ORCL IBM QCOM | XOM CVX VLO | JPM BAC AXP | JNJ PFE MRK | PG KO WMT |
  HD MCD DIS | CAT BA | T. Monthly, adjusted for splits/dividends (yfinance, auto_adjust=True).
- Chosen by size and sector, not later performance, to avoid hindsight bias. Still survivorship
  bias (only companies that exist today). Swapping in QCOM dropped the only utility.
- Future work: point-in-time universe (largest 25 firms at each date, including delisted ones)
  needs CRSP/WRDS; yfinance has no delisted stocks or historical market caps.
- Secondary "investor's picks" stress test (tech/energy heavy: NVDA, GOOGL, META, AMD, CVX, TPL, VLO, XOM...):
  hindsight built in, label as a stress test, not the main result.
- No oracle on real data: score with annual return, volatility, Sharpe, max drawdown, turnover;
  compare model vs even split, Markowitz, Ledoit-Wolf, S&P 500.

Limitations to state
- Regularization chosen on the test set (slightly optimistic numbers) -> separate validation set.
- early_stopping validates on random rows, not markets (leaky).
- One-factor markets only (professor's feedback); crisis set raises correlation (0.72 vs 0.23) but
  not total volatility.
- One training run per setting, no error bars yet.

## Session log

### Thu Nov 5, 2026
Changes
- Real data: `real/check_tickers.py` downloads monthly adjusted prices for the 25 stocks
  (Jan 1990 - Nov 2026, 443 months) to `data/real_prices.csv`. Columns come out alphabetical.
  The Nov 2026 row is a partial month: drop it when computing returns.
- `train.py` rewritten (same calculations, same split): settings (ALPHA, HIDDEN) at the top;
  prints train vs test R2 and MSE, iterations, training time; saves model + scaler as
  `data/model_<N>k.joblib` (name taken from the data size); prints a one-line SUMMARY;
  saves the leverage plot to `figures/loss_by_leverage_<N>k.png` before showing it.
- `dataset.py`: n_markets 60,000 -> 100,000 (300,000 rows, ~1.2 GB file, ~5 min to generate).

Problems hit (and fixes)
- Relative path `data/...` failed when running from `real/`: PyCharm runs scripts from their own
  folder. Fix: build paths from `Path(__file__).resolve().parents[1]`.
- Changed n_markets but trained before regenerating: train.py just loads whatever train.npz is on
  disk. Always run dataset first; check the first line ("N examples from M markets").
- A placeholder `...` pasted from an example became real code (`mse_train = ...`) -> crash
  ("unsupported format string passed to ellipsis"). Fixed in the rewrite.

Results: 100k markets, alpha=1e-1 (first run, crashed before the strategy table)
- train R2 0.248 / test R2 0.233; train MSE 0.01253 / test MSE 0.01267; stopped at 38 iterations.
- Overfitting essentially gone (train ~ test). Test R2 gains shrinking: 0.198 (20k) -> 0.228 (60k)
  -> 0.233 (100k). The bottleneck has moved from data to model capacity / penalty strength.
  Next: try alpha=1e-2 and/or a bigger network at 100k.

Results: 100k markets, alpha=1e-1, full run (model saved as data/model_100k.joblib)
- SUMMARY markets=100,000 alpha=0.1 hidden=(128, 64) iters=38 secs=21 r2 train/test=0.248/0.233
  mse train/test=0.01253/0.01266 model=3.39% even=5.08% markowitz=42.24%
- Same loss as 60k (3.39%): plateau. Data is no longer the bottleneck; model capacity / penalty is.
  Gap closed ~70% (even 5.08% -> floor ~2.66%, floor not recomputed on this test set).
- Leverage buckets: 0-2: model 0.00204 vs even 0.00287; 2-4: model 0.00348 vs even 0.00537.

Data-scaling table so far (alpha=1e-1, hidden=(128,64))
| Markets | Train R2 | Test R2 | Train MSE | Test MSE | Model | Even |
|---|---|---|---|---|---|---|
| 20k  | 0.341 | 0.198 | -       | -       | 3.54% | 5.06% |
| 60k  | -     | 0.228 | -       | -       | 3.39% | 5.05% |
| 100k | 0.248 | 0.233 | 0.01253 | 0.01266 | 3.39% | 5.08% |
(Missing cells: rerun 20k and 60k with the new train.py to fill in.)

Regularization sweep at 100k markets (hidden=(128,64))
| ALPHA | Train R2 | Test R2 | Train MSE | Test MSE | Model | iters |
|---|---|---|---|---|---|---|
| 1e-1 | 0.248 | 0.233 | 0.01253 | 0.01266 | 3.39% | 38 |
| 1e-2 | 0.274 | 0.236 | 0.01209 | 0.01261 | 3.44% | 54 |
- R2 and CE loss disagree slightly (1e-2 better R2, worse CE) -> trust CE (money measure).
- Lighter penalty brought back a little memorizing (train/test R2 gap 0.015 -> 0.038).
- Differences are tiny (0.05%/yr, single run) and probably within run-to-run noise -> need error bars.
- Conclusion: plateau ~3.4% is not caused by data size (60k = 100k) or penalty (1e-1 ~ 1e-2).
  Remaining levers: model capacity (bigger network) and inputs (formula weights as features).
- Caveat: settings chosen by looking at test results (slightly optimistic); proper version uses a
  market-based validation set and touches the test set once.
- NOTE: rerun alpha=1e-1 last so data/model_100k.joblib is the best model before the real-data test.
| 1e-3 | 0.232 | 0.184 | 0.01279 | 0.01348 | 3.71% | 85 |
- 1e-3 is worse on BOTH train and test (train R2 lower than at 1e-2): not just overfitting; training
  wandered (validation stuck ~0.006 for the first 12 iterations) and early stopping ended at a worse
  spot. Tentative (single run): too little regularization -> overfits AND optimizes less reliably.
- Winner at 100k: alpha=1e-1 (same as at 20k). Loss curve: 3.39% / 3.44% / 3.71% for 1e-1 / 1e-2 / 1e-3.
