# Glossary

★ = essential for the poster box (aim for ~10-12 there; the full list goes in the paper appendix).
Note: "alpha" means two things in this project: a stock's hidden bonus return (finance) and the
regularization strength in the MLP (machine learning).

## Finance

| Term | Plain-English meaning |
|---|---|
| ★ Portfolio / portfolio weights | How your money is split across stocks; the weights are the percentages, summing to 100% |
| ★ Return | How much an investment gained or lost over a period (e.g. +2% this month) |
| Expected return | The average return you'd expect in the long run; unknown in reality |
| ★ Volatility | How much a stock's returns swing up and down; the standard measure of risk |
| Variance | Volatility squared; how spread out returns are |
| Covariance | How two stocks' returns move together, in raw units |
| ★ Correlation | Covariance rescaled to -1..+1; +1 = always move together, 0 = unrelated |
| ★ Diversification | Spreading money across stocks that don't move together, so losses partly cancel out |
| ★ Markowitz / mean-variance optimization | The 1952 formula that picks weights to balance expected return against risk |
| Risk aversion (gamma = 3) | How much an investor dislikes risk; higher means safer portfolios |
| Beta | How strongly a stock follows the overall market (1 = moves with it) |
| ★ Alpha | A stock's extra return beyond what its market risk explains; small and hidden by noise |
| Market risk premium | Extra return investors earn for taking market risk (6%/yr in my fake markets) |
| Stock-specific (idiosyncratic) risk | The part of a stock's swings unrelated to the market; can be diversified away |
| Factor model | Stocks move because of shared drivers (here, one: the market) plus their own noise |
| ★ Estimation error | The gap between what history suggests and the truth; the core problem of the project |
| Shrinkage | Pulling noisy estimates toward a simpler, more stable target |
| Ledoit-Wolf | The standard shrinkage method for the covariance; the professional fix for Markowitz |
| ★ Even split (1/N) | Putting the same share in every stock; simple, and famously hard to beat |
| Minimum-variance portfolio | The lowest-risk portfolio; ignores expected returns entirely |
| Short selling | Betting a stock will fall (a negative weight) |
| Long-only | No shorting allowed; every weight is 0% or more |
| Leverage | Borrowing or shorting to hold more than 100% of your money in positions |
| ★ Certainty equivalent (CE) | A portfolio's return minus a penalty for its risk: "as good as a guaranteed X%" |
| ★ CE loss | How much CE a strategy gives up vs the perfect portfolio; the main simulation score |
| ★ Sharpe ratio | Return divided by volatility: return per unit of risk |
| Max drawdown | The worst peak-to-bottom drop in value |
| Turnover | How much of the portfolio is traded each month; trading costs money |
| Backtest / rolling window | Testing on past data, re-deciding each month using only data available then |
| Survivorship bias | Testing only on companies that still exist today, which flatters results |
| Hindsight bias | Choosing what to test using knowledge of what later happened |
| Oracle / perfect portfolio | The best portfolio given the true parameters; the answer key |
| Bayes floor | The best possible score without knowing the future; the realistic target |

## Machine learning

| Term | Plain-English meaning |
|---|---|
| ★ Neural network / MLP | A model that learns a mapping from inputs to outputs from examples (2 layers: 128 and 64 neurons) |
| ★ Features | The model's inputs: 351 summary numbers per history |
| Target | What the model is trained to output: the 25 perfect weights |
| Training / test split | Learn on some markets (80%), score on others never seen (20%) |
| ★ Overfitting | Memorizing the training examples instead of learning general rules |
| Regularization | A penalty that discourages overfitting (alpha = 0.1) |
| MSE | Mean squared error: average squared distance between predicted and perfect weights |
| R2 | Share of the variation in the perfect weights the model explains (0 = none, 1 = all) |
