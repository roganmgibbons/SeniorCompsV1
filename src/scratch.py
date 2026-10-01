import numpy as np
from src.known import make_truth, ASSET_NAMES

monthly, Sigma = make_truth()

print("True monthly expected returns:")
for name, m in zip(ASSET_NAMES, monthly):
    print(f"  {name:<8} {m:>8.4f}   ({m * 12:>6.1%} per year)")

print("\nTrue covariance matrix:")
print(np.round(Sigma, 6))

print("\nVolatility")
for name, v in zip(ASSET_NAMES, np.sqrt(np.diag(Sigma))):
    print(f"  {name:<8} {v * np.sqrt(12):>6.1%} per year")

