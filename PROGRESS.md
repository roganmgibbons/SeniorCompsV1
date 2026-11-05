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
