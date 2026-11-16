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

Final 1e-1 rerun at 100k: identical to first run (model=3.39%), training is repeatable (random_state=0).
data/model_100k.joblib = alpha 1e-1, hidden (128,64), 100k markets.

### First real-data results (real/backtest.py, model_100k)
Setup: 25 stocks, monthly returns Jan 1992 - Oct 2026 (418 months; Nov 2026 dropped as partial).
Rolling: 120-month lookback, re-chosen monthly, held one month. 298 decisions, Jan 2002 - Oct 2026.
Familiarity check: 0.6% of real scaled features outside |z| > 3 -> real inputs look like training data.

| Strategy | Return/yr | Vol/yr | Sharpe | CE/yr | Worst drop | Turnover | Leverage |
|---|---|---|---|---|---|---|---|
| model       |  9.17% | 15.16% |  0.61 |   5.73% |  -47.2% | 0.17 |  1.55 |
| even split  | 13.73% | 15.22% |  0.90 |  10.26% |  -44.6% | 0.00 |  1.00 |
| markowitz   | -3.54% | 75.48% | -0.05 | -88.99% | -100.0% | 2.32 | 14.47 |
| ledoit-wolf |  1.66% | 54.99% |  0.03 | -43.69% |  -98.1% | 1.49 |  9.68 |
(Sharpe = return/vol, no risk-free rate. CE uses gamma=3, same score as the simulation.)
Chart: figures/real_backtest.png ($1 -> ~$7 model, ~$22 even split; Markowitz bust 2009).

Findings
1. Model survives where the formulas crash: Markowitz bust (-100%, 14.5x leverage), Ledoit-Wolf
   -98%. Model -47% worst drop, 1.55x leverage, low turnover, same volatility as even split.
   Ledoit-Wolf fixes the covariance but still uses noisy sample means -> blows up (Merton again).
2. Model loses to even split (Sharpe 0.61 vs 0.90, CE 5.7% vs 10.3%) at the same volatility.
   Matches DeMiguel, Garlappi & Uppal (2009), "Optimal Versus Naive Diversification": no optimizer
   reliably beats 1/N out of sample. Cite it.
Likely reasons even split wins
- Survivorship bias favors 1/N: all 25 stocks survived to 2026; shorting known survivors loses money
  (model currently shorts DIS, CVX, T, PG, PFE, IBM, BA).
- Real expected returns across large caps may be more similar than the simulation's beta*6% + alpha.
- Model learned from an oracle that uses ~2x leverage, so tilting/shorting looks normal to it.
- Inputs are familiar (0.6% outside range), so the gap is in how the real world rewards the weights.
Rubric: Goal 4 -> Passing (runs on real data, compared to even split). Good needs beating 1/N on Sharpe.
Known simplification: even split turnover shown as 0 (ignores drift between rebalances).

Next experiments suggested by these results
1. Long-only versions of every strategy, including the model (Tier 3 item 8): removes shorting of
   survivors and the blow-ups.
2. Minimum-variance baseline (ignores means; often strong on Sharpe in the literature).
3. Crisis windows (2008, 2020) as a separate real-data stress test.
4. S&P 500 (SPY) as a real-world reference line.

### Real-data micro-experiments (real/experiments.py, model_100k)
Full output: real/results/experiments_results_2026-11-05.txt. Run: `python -m real.experiments`
New strategies: min-variance (Ledoit-Wolf covariance, ignores means) and long-only (LO) versions:
model LO = model weights with shorts dropped and rescaled; markowitz LO / ledoit-wolf LO / min-variance LO
solved with no shorting (scipy SLSQP, weights >= 0, sum to 1).

EXP 1-2: 10-year lookback, Jan 2002 - Oct 2026
| Strategy | Return | Vol | Sharpe | CE | Worst drop | $1 -> | Leverage |
|---|---|---|---|---|---|---|---|
| model           |  9.17% | 15.16% |  0.61 |   5.73% |  -47.2% |  7.27 |  1.55 |
| markowitz       | -3.54% | 75.48% | -0.05 | -88.99% | -100.0% |  0.00 | 14.47 |
| ledoit-wolf     |  1.66% | 54.99% |  0.03 | -43.69% |  -98.1% |  0.02 |  9.68 |
| min-variance    | 10.02% | 12.87% |  0.78 |   7.54% |  -33.0% |  9.71 |  1.41 |
| model LO        | 11.08% | 14.14% |  0.78 |   8.08% |  -47.0% | 12.08 |  1.00 |
| markowitz LO    | 15.72% | 20.75% |  0.76 |   9.27% |  -51.0% | 28.42 |  1.00 |
| ledoit-wolf LO  | 16.53% | 21.33% |  0.77 |   9.70% |  -51.8% | 33.56 |  1.00 |
| min-variance LO | 10.30% | 12.27% |  0.84 |   8.04% |  -34.1% | 10.60 |  1.00 |
| even split      | 13.73% | 15.22% |  0.90 |  10.26% |  -44.6% | 22.33 |  1.00 |

EXP 3: lookback (each covers a different period, so compare within a table)
- 5-yr (1997-2026): model LO Sharpe 0.87 vs even 0.96; model (unconstrained) 0.61.
- 20-yr (2012-2026): model LO Sharpe 1.06 / CE 11.5% vs even 1.14 / 13.1%; model 0.96 / 10.4%.

EXP 4: periods (10-yr lookback), Sharpe for model LO vs even split
- 2002-07: 0.72 vs 0.97 (min-variance LO 1.00 best)
- 2008-09: -0.43 vs 0.01 (min-variance LO best worst drop -32%; even split best CE)
- 2010-19: **1.26 vs 1.21** (model LO best Sharpe, lower vol 11.6% vs 12.5%, smaller drop -10.7% vs -13.6%)
- 2020-26: 0.88 vs 0.96 (ledoit-wolf LO best: 1.03)

Findings
1. Banning shorts rescues the formulas: Markowitz/LW go from bust to Sharpe ~0.77 and the highest
   raw growth ($1 -> $28-34 vs $22 for even split), but with ~21% volatility.
2. Banning shorts helps the model too: Sharpe 0.61 -> 0.78, CE 5.7% -> 8.1%. Shorting survivors was
   a big part of the gap (survivorship bias).
3. Even split still wins overall on Sharpe and CE (DeMiguel et al. 2009), but not in every period:
   model LO beat it on Sharpe in 2010-2019.
4. More history helps the model most (20-yr lookback: model Sharpe 0.96 vs 0.61 at 10-yr).
5. Min-variance LO = lowest volatility and smallest crashes throughout (good crisis behavior).
Caveats: many comparisons -> risk of cherry-picking a winner; sub-periods are short (2008-09 is 24
months) so their Sharpe ratios are noisy; turnover ignores drift; no trading costs; survivorship bias.

### Fri Nov 6, 2026: meeting with professor (last weekly meeting)
- Whiteboard exercise: explained every step of the pipeline in detail. Main feedback:
  1. Quantify every step of the project (numbers, not descriptions).
  2. Refine how the whole pipeline is explained.
  3. "Explain the variance in the generated markets": any measurable randomness should be quantified.
- Whiteboard flow chart corrections (for the poster version):
  - Split "market simulation" into 100,000 markets -> 3 histories each = 300,000 histories.
  - Features = 351 numbers (25 avg returns, 25 vols, 300 correlations, # months), not just returns/time.
  - Oracle info has two arrows: -> answer key (MLP training target) and -> grading (CE loss).
  - Add the train/test split by market: 80k/20k markets = 240k/60k rows.
  - "Improved Markowitz" = Ledoit-Wolf (+ long-only, min-variance).
  - Add final scoring step (CE in simulation; Sharpe/drawdown on real data) and the real-data branch
    (real prices -> monthly returns -> same 351 features -> trained MLP).

### Plan: Sun Nov 8 - Tue Nov 10 presentation
Three layers of randomness to quantify:
1. Between markets: distributions of generated betas, vols, alphas, market vol; resulting avg
   correlation, oracle leverage, oracle CE. Compare against the 25 real stocks (real betas, vols,
   correlations) -> are the fake markets realistic?
2. Within a market: how far sample means / vols / correlations land from the truth at 5 vs 50 years
   (quantifies estimation error directly).
3. In results: retrain with 5 seeds -> error bars on 3.39% and on the real-data Sharpe ratios.
Then: slides for Tuesday (pipeline, quantified; experiments; results), poster planning.
- Professor's note (verbatim): "Test randomness and variance of portfolios -> Variance of MLP portfolio performance."
  Interpretation: report the SPREAD of the model's performance, not just the average:
  a) across test markets: p10/median/p90 of CE loss + % of markets where model beats even split
     (train.py already computes per-market losses)
  b) across training runs: 5 seeds -> 3.39% +/- X
  c) across real portfolios and time: random 25-stock portfolios from the pool; confidence
     intervals on real-data Sharpe ratios

### Mon Nov 9, 2026: presentation prep + variance of model performance
- Added to train.py (after the strategy table): spread of the model's CE loss across the 60,000
  test histories + win rate vs even split.
- Result (100k markets, alpha 1e-1):
  - p10 1.79% / median 3.12% / p90 5.34% per year (mean 3.39%: a minority of hard cases pulls it up)
  - Model beats even split in 86.7% of test histories -> consistent, not driven by a few wins.
  - Spread is mostly market difficulty (more extreme oracle portfolio -> more loss for every strategy).
- Still to do for "variance": exp04_market_variance output (spread of fake markets), 5-seed error
  bars, random real portfolios.
- New figure: figures/pipeline_detailed.png (clean redraw of the whiteboard flow chart, for slide 3
  and the poster).
- Presentation 5 outline: 1 title, 2 question/foundation/goals, 3 detailed pipeline diagram,
  4 what I did (model + real stocks + quantifying randomness), 5+ results, rubric, next steps.

Variance BETWEEN fake markets (experiments/exp04.py, 100,000 markets, after the leverage filter)
| Measure | p0 | p10 | p50 | p90 | p100 | mean |
|---|---|---|---|---|---|---|
| stock volatility (annual)   |   8.3% | 29.2% | 41.6% | 54.4% | 86.8% | 41.7% |
| stock expected return       | -10.4% |  1.4% |  6.0% | 10.7% | 24.4% |  6.0% |
| avg correlation in a market |   0.03 |  0.10 |  0.23 |  0.40 |  0.69 |  0.24 |
| perfect portfolio leverage  |   1.01 |  1.45 |  2.09 |  3.40 |  4.00 |  2.26 |
| best possible CE (annual)   |  -7.2% | -0.0% |  3.8% |  7.6% | 16.5% |  3.8% |
- This answers the professor's question ("how different are the generated markets"): wide range of
  conditions (correlation 0.10-0.40, leverage 1.45-3.40 for the middle 80%), not one world.
- ~10% of markets: even the perfect portfolio has CE <= 0 (risk outweighs reward; weights must sum
  to 1 so it can't sit in cash).
- Realism flag: fake stocks are more volatile (median 42%/yr) than real large caps (~15-30%).
  Correlation (median 0.23) looks realistic. Possible reason for the real-data gap.
  Next: compute the same table for the 25 real stocks and compare side by side.

Real vs fake markets (real = 25 stocks, monthly, Yahoo Finance via yfinance, Jan 1992 - Oct 2026)
| | Real 25 stocks | Fake markets |
|---|---|---|
| Stock volatility, median | 27.6%/yr (2002-26 only: 24.9%) | 41.6% |
| Stock volatility, range  | 18.3% (JNJ) - 49.4% (QCOM)     | 8.3% - 86.8% |
| Avg pairwise correlation | 0.25 (2002-26: 0.29)           | 0.24 (median 0.23) |
- Fake stocks ~1.5x more volatile than real large caps; correlation realistic.
- Likely cause of the high fake volatility: the leverage filter rejects markets with similar
  (low idiosyncratic vol) stocks, so kept markets skew toward high volatility.
- Implied scores in simulation (approx; best CE avg over all markets, losses over test markets):
  perfect ~3.8%/yr, model ~0.4%, even split ~-1.3%, Markowitz ~-38%.

## Citations (verify volume/pages on Google Scholar before the paper)
1. Markowitz, H. (1952). Portfolio Selection. Journal of Finance, 7(1), 77-91.
2. Merton, R. C. (1980). On Estimating the Expected Return on the Market. Journal of Financial Economics, 8(4), 323-361.
3. Jorion, P. (1986). Bayes-Stein Estimation for Portfolio Analysis. Journal of Financial and Quantitative Analysis, 21(3), 279-292.
4. Campbell, J. Y., Lettau, M., Malkiel, B. G., & Xu, Y. (2001). Have Individual Stocks Become More Volatile? An Empirical Exploration of Idiosyncratic Risk. Journal of Finance, 56(1), 1-43. (verified)
5. Ledoit, O., & Wolf, M. (2004). A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices. Journal of Multivariate Analysis, 88(2), 365-411. (method used by sklearn LedoitWolf; alt: "Honey, I Shrunk the Sample Covariance Matrix", J. Portfolio Management 2004)
6. DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal Versus Naive Diversification: How Inefficient Is the 1/N Portfolio Strategy? Review of Financial Studies, 22(5), 1915-1953.
7. Zhang, Z., Zohren, S., & Roberts, S. (2020). Deep Learning for Portfolio Optimization. Journal of Financial Data Science, 2(4), 8-20.
8. NYU Stern Volatility Lab, S&P 500 volatility (GARCH): https://vlab.stern.nyu.edu/volatility/VOL.SPX%3AIND-R.GARCH
9. Yahoo Finance monthly adjusted prices via the yfinance Python library (1990-2026).

### Parameter variance (professor's request), experiments/exp05.py
Code changes: factor_model_known(..., return_params=True) returns beta, idio_vol, alpha (annual);
dataset.py saves param_* (kept markets) and draw_* (every draw, kept or rejected) + draw_kept into
train.npz. Random draw order unchanged -> data reproduced exactly (74,894 rejected draws, same as
before), so model_100k.joblib and all results stay valid.
Problems hit: old report_and_save left below the new one (Python uses the LAST definition) ->
"too many values to unpack"; exp05 run before train.npz finished saving -> "File is not a zip file".

174,894 draws, 100,000 kept (57.2%), 74,894 rejected by the leverage filter.
| Parameter | | mean | sd | variance | p10 | p50 | p90 |
|---|---|---|---|---|---|---|---|
| market volatility (per market)        | all  | 20.0% | 5.8%  | 0.0033 | 12.0% | 20.0% | 28.0% |
|                                       | kept | 19.9% | 5.8%  | 0.0033 | 12.0% | 19.9% | 27.9% |
| typical stock-specific vol (per mkt)  | all  | 28.5% | 9.5%  | 0.0091 | 15.3% | 28.5% | 41.7% |
|                                       | kept | 35.3% | 5.8%  | 0.0034 | 27.4% | 35.6% | 43.1% |
| beta (per stock)                      | all  | 1.00  | 0.35  | 0.1200 | 0.55  | 1.00  | 1.45  |
|                                       | kept | 1.00  | 0.35  | 0.1200 | 0.55  | 1.00  | 1.45  |
| stock-specific volatility (per stock) | all  | 28.7% | 12.0% | 0.0144 | 12.2% | 28.5% | 44.8% |
|                                       | kept | 35.5% | 9.7%  | 0.0093 | 22.9% | 35.5% | 48.0% |
| alpha (per stock)                     | all  | 0.0%  | 3.0%  | 0.0009 | -3.8% | 0.0%  | 3.8%  |
|                                       | kept | 0.0%  | 3.0%  | 0.0009 | -3.8% | 0.0%  | 3.8%  |
Total stock volatility: all mean 36.4% (p10 21.6 / p50 36.3 / p90 51.3); kept mean 41.7% (29.2 / 41.6 / 54.4).

Findings
- Generator matches its design exactly (uniform 10-30 -> mean 20 sd 5.8; uniform 12-45 -> mean 28.5
  sd 9.5; beta 1.00/0.35; alpha 0/3%).
- The leverage filter only reshapes stock-specific volatility: it removes nearly all low-idio markets
  (p10 15.3% -> 27.4%), because then stocks are near-copies of the market and the oracle needs huge
  long/short bets (>4x). Market vol, beta, alpha unaffected.
- Realism gap explained: total vol median 36% from the rules alone, 42% after the filter, vs 28% for
  the 25 real stocks.
- Possible fixes (later): lower idio range; long-only oracle (no leverage filter needed); then retest
  real data.
Next (piece 4): estimate real betas / idio vol / alphas by regressing the 25 real stocks on SPY
(python -m real.download spy) and compare side by side.

### Tue Nov 10, 2026: practice-run feedback (roommates) -> new experiments
1. History depth: 3 -> 10 histories per market.
   - Version A (do this): same total rows. 30,000 markets x 10 histories = 300,000 rows vs current
     100,000 x 3. Isolates "many markets vs same markets many times". ~5 min, same file size.
     Prediction: 3 histories wins (more distinct answer keys).
   - Version B (later, overnight): 100,000 x 10 = 1M rows; ~8 GB peak memory -> use float32.
2. Portfolio size: 25 vs 30 vs 35 stocks.
   - Input size = 2N + N(N-1)/2 + 1 -> 351 / 496 / 666 features; each N needs its own dataset + model.
   - N_ASSETS hard-coded in dataset.py, train.py, real scripts -> make it a setting.
   - Real data: draw 30/35 stocks from the pool universe (python -m real.download pool).
   - Theory: estimation error grows with N for fixed history -> Markowitz should degrade; does the
     model degrade less?
Order after presentation: real-parameter comparison (piece 4) -> history depth A -> portfolio sizes
-> 5-seed error bars.

Deck fixes noted for Presentation 5: add rubric slide (required Q3); slide 4 MSE wording is
train vs test (0.0125 vs 0.0127), not a change over time; use fixed real_model_vs_even.png.

### Thu Nov 12, 2026: metrics decision + glossary (post-presentation feedback)
Headline metrics (one per thesis question):
1. CE loss (simulation): exact Markowitz objective vs a known answer -> cost of estimation error.
2. Sharpe ratio (real data): standard risk-adjusted score, comparable to DeMiguel et al.
3. Max drawdown (real data): what a personal investor feels; tests the crisis goal.
Supporting: win rate vs even split (87%), gap closed toward Bayes floor (~70%, approximate floor),
$1 grows to (communication), volatility / turnover / leverage (explain the headline numbers).
Diagnostics only: R2, MSE (weight accuracy, not money; disagreed with CE once).
Paper sentence: "I evaluate strategies by certainty-equivalent loss in simulation, where the optimal
portfolio is known, and by Sharpe ratio and maximum drawdown on real data, where it is not."
Glossary saved as GLOSSARY.md (finance + ML; starred terms for the poster box).
Not yet committed: Nov 9 changes (param saving, exp04, exp05, win rate).

### Sun Nov 15, 2026: real vs fake parameters (real/real_params.py), finishes professor's request
Setup: SPY downloaded (python -m real.download spy; data/real_prices_spy.csv, from Jan 1993).
Each real stock regressed on SPY monthly returns, Feb 1993 - Oct 2026 (405 months):
stock return = alpha + beta * market return + noise (np.polyfit). Fake = kept markets (param_*).
Problem hit: ran real_params.py before downloading SPY -> FileNotFoundError.

| Parameter | | mean | sd | p10 | p50 | p90 |
|---|---|---|---|---|---|---|
| beta                      | fake | 1.00  | 0.35 | 0.55  | 1.00  | 1.45  |
|                           | real | 0.99  | 0.35 | 0.54  | 1.10  | 1.41  |
| alpha (annual)            | fake | 0.0%  | 3.0% | -3.8% | 0.0%  | 3.8%  |
|                           | real | 5.1%  | 3.9% | 1.3%  | 5.0%  | 9.4%  |
| stock-specific volatility | fake | 35.5% | 9.7% | 22.9% | 35.5% | 48.0% |
|                           | real | 24.8% | 7.2% | 17.6% | 22.4% | 35.9% |
| total stock volatility    | fake | 41.7% | 9.7% | 29.2% | 41.6% | 54.4% |
|                           | real | 28.9% | 8.3% | 19.5% | 27.3% | 40.3% |
Market volatility: fake mean 19.9% (range 10-30%) vs real S&P 500 14.7%.
Avg correlation: real 0.26 vs fake median 0.23.
Per-stock real betas: PG 0.41 (lowest) ... INTC 1.47 (highest). Negative real alpha: DIS -3%, BAC -3%
(23 of 25 positive). Highest alpha: AAPL +14%, QCOM +12%, VLO +10%.

Findings
1. Structure realistic: beta (1.00 vs 0.99, same spread and percentiles) and correlation (0.23 vs 0.26).
2. Too volatile at every level: market +35% (19.9 vs 14.7), stock-specific +43% (35.5 vs 24.8),
   total +44% (41.7 vs 28.9). Causes: design ranges (market vol 10-30%) + leverage filter (exp05).
3. Real alphas average +5.1%: survivorship bias, now measured (chose companies that became giants).
   Explains part of the real-data gap: the model shorted stocks with positive real alpha
   (PG +7%, PFE +4%, CVX +5%, T +3%) because positive alpha doesn't exist on average in training.
Caveat: each real alpha has standard error ~ 25%/sqrt(34 yrs) ~ +/-4%/yr, so the spread of real
alphas (sd 3.9%) is about what noise alone gives (Merton 1980 on real data). The +5% average across
25 stocks is too consistent to be noise.
Note: regression uses raw returns (no risk-free rate), so low-beta stocks get a small upward alpha
bias of about (1 - beta) x risk-free rate (~1%/yr at most). Minor vs +5%.

Next experiment (poster candidate): recalibrate the fake market (market vol ~15%, lower
stock-specific vol), retrain, rerun the real backtest; does the gap to even split shrink?
Tradeoff: lower stock-specific vol -> leverage filter rejects more markets -> likely switch to a
long-only oracle (never needs extreme leverage).

### Plan: three robustness experiments (started Sun Nov 15)
Rules: change one setting at a time (alpha=0.1, hidden=(128,64), seed fixed); every run saves to
its own files (train_<tag>.npz, model_<tag>.joblib) so the 100k baseline is never overwritten.
Code change: dataset.py and train.py take command-line settings (--markets, --histories, --assets,
--market-vol, --idio-vol, --out, --no-crisis / --data, --tag, --alpha). Defaults reproduce the baseline.

1. Volatility recalibration (most important)
   baseline: market vol U(10%,30%), stock-specific U(12%,45%)
   realistic: market vol U(10%,20%) (mean ~15% vs S&P 14.7%), stock-specific U(10%,30%)
   python -m ml.dataset --market-vol 0.10 0.20 --idio-vol 0.10 0.30 --out train_lowvol.npz --no-crisis
   python -m ml.train --data train_lowvol.npz --tag lowvol
   Judge by: train vs test loss; model vs even split WITHIN each world (CE not comparable across
   calibrations: calmer markets are easier for everyone); real-data backtest (same test for both).
   Watch: rejected draws (filter rejects more calm markets) -> log it; may need long-only oracle.
2. Histories per market: 3 vs 5 vs 10 at the SAME total rows (100k x 3, 60k x 5, 30k x 10 = 300k).
3. Portfolio size: 25 vs 30 vs 35 stocks (351 / 496 / 666 features). Real data: add 1-2 stocks per
   sector from the pool (e.g. 30: + TXN, PEP, WFC, ABT, SO; 35: + NEE, LOW, COP, AMGN, UNP) and make
   backtest/experiments use the number of stocks from the data instead of 25.

### Experiment 1 (volatility): data generation, Sun Nov 15
Command: python -m ml.dataset --market-vol 0.10 0.20 --idio-vol 0.10 0.30 --out train_lowvol.npz --no-crisis
Fixes before running: train.py rewritten with argparse (--data/--tag/--alpha), ROOT/DATA_DIR/FIG_DIR
restored, model saved as model_<tag>.joblib (old MODEL_FILE line would have overwritten
model_100k.joblib); dataset.py generate() got the n_assets parameter. Note: PyCharm overwrote the new
train.py with its open buffer once -> reload from disk when PyCharm asks.
| | Baseline | Low volatility |
|---|---|---|
| draws rejected by leverage filter | 74,894 / 174,894 (43%) | 374,026 / 474,026 (79%) |
| perfect-portfolio leverage, median | 2.09 | 3.18 |
- Calm markets -> stocks look alike -> oracle makes bigger long/short bets -> most break the 4x cap.
  Survivors sit near the cap (median 3.18x): more extreme targets, harder to learn, less like real
  investing. The filter likely keeps the upper end of the new vol ranges -> check with exp05 on this file.
- Strengthens the case for a long-only oracle (no leverage, no filter).

### Experiment 1 (volatility): training results, Sun Nov 15
SUMMARY data=train_lowvol.npz tag=lowvol markets=100,000 stocks=25 alpha=0.1 hidden=(128,64) iters=39
secs=22 r2 train/test=0.257/0.243 mse train/test=0.02401/0.02454 model=4.34% even=5.42%
markowitz=41.24% win rate=82.2%  (loss spread p10/p50/p90: 2.72% / 4.19% / 6.12%)
| | Baseline | Low vol |
|---|---|---|
| R2 train / test | 0.248 / 0.233 | 0.257 / 0.243 |
| MSE train / test | 0.0125 / 0.0127 | 0.0240 / 0.0245 |
| model loss/yr | 3.39% | 4.34% |
| even split loss/yr | 5.08% | 5.42% |
| model edge over even | 1.69 pts (33% less) | 1.08 pts (20% less) |
| win rate vs even | 86.7% | 82.2% |
| oracle leverage median | 2.09 | 3.19 |
| leverage 0-2 bucket (model vs even) | 0.00204 vs 0.00287 | 0.00212 vs 0.00207 (921 rows: even wins) |
Findings
- No overfitting in either world.
- Model's edge over even split shrinks in the calmer world; in the simplest markets even split wins.
- MSE doubled only because targets are more extreme (bigger weights -> bigger squared errors):
  MSE is not comparable across worlds (diagnostic, not headline). CE also not comparable across worlds.
- The "calm" world is not calm in practice: the leverage filter (79% rejected) keeps the most extreme
  oracles. The filter, not volatility, now dominates -> next: long-only oracle (no filter needed).
Decisive test: python -m real.experiments --model model_lowvol.joblib (compare real Sharpe to baseline:
model 0.61, model LO 0.78, even split 0.90).

### Experiment 1 (volatility): real-data results, model_lowvol vs model_100k
Command: python -m real.experiments --model model_lowvol.joblib (formula strategies identical to
before -> sanity check passes; only the model changed).
10-year lookback, Jan 2002 - Oct 2026:
| | Original model | Low-vol model |
|---|---|---|
| model: Sharpe / CE / worst drop / $1 -> / leverage | 0.61 / 5.73% / -47.2% / 7.27 / 1.55 | 0.58 / 5.32% / -39.6% / 6.42 / 1.44 |
| model LO: Sharpe / CE / worst drop / $1 -> | 0.78 / 8.08% / -47.0% / 12.08 | 0.74 / 7.46% / -41.0% / 10.33 |
| even split | 0.90 / 10.26% / -44.6% / 22.33 | same |
Lookback (model LO Sharpe; even split): 5-yr 0.87 -> 0.98 (even 0.96); 10-yr 0.78 -> 0.74 (0.90);
20-yr 1.06 -> 0.96 (1.14).
5-yr lookback, low-vol model LO vs even split: Sharpe 0.98 vs 0.96, CE 11.87% vs 11.54%,
worst drop -40.1% vs -44.6%, $1 -> 73.56 vs 65.39  (first time a model version beats 1/N on all
headline metrics; one comparison of many, one seed, small margin -> needs error bars).
Periods (10-yr lookback, model LO Sharpe original -> low-vol): 2002-07 0.72 -> 0.73; 2008-09 -0.43 ->
-0.37; 2010-19 1.26 -> 1.08; 2020-26 0.88 -> 0.85.
Findings
- Consistent gain: smaller crashes (worst drop -47% -> -40/-41%, better than even split's -45%).
- Mixed Sharpe: slightly worse at 10/20-yr lookbacks, better at 5-yr.
- Verdict: realistic volatility improved crash protection but did not clearly close the gap to 1/N;
  the leverage filter (79% rejected, oracle leverage 3.2x) likely offset the calibration.
Paper sentence: "Calibrating the simulator's volatility to real stocks reduced the model's worst
drawdown on real data from -47% to -40%, but did not consistently improve its Sharpe ratio, because
the leverage filter offset the calibration by retaining markets with more extreme optimal portfolios."
Next: long-only oracle (removes the leverage filter), then error bars (5 seeds), histories, portfolio size.
