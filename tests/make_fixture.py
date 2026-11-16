"""Create a SYNTHETIC file with the exact UCI Taiwan column layout, for code testing only.
Numbers are random - never report results from this file."""
import numpy as np, pandas as pd, sys
rng = np.random.default_rng(0); n = 30000
risk = rng.normal(0, 1, n)
lim = np.round(np.exp(rng.normal(11.8 - 0.3 * risk, 0.7)), -4).clip(10000, 1000000)
d = {"ID": np.arange(1, n + 1), "LIMIT_BAL": lim, "SEX": rng.choice([1, 2], n),
     "EDUCATION": rng.choice([0, 1, 2, 3, 4, 5, 6], n, p=[.001, .35, .47, .16, .005, .01, .004]),
     "MARRIAGE": rng.choice([0, 1, 2, 3], n, p=[.002, .45, .53, .018]), "AGE": rng.integers(21, 75, n)}
for i, c in enumerate(["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]):
    base = np.clip(np.round(rng.normal(-0.3 + 0.9 * risk - 0.1 * i, 1.0)), -2, 8)
    d[c] = base.astype(int)
for i in range(1, 7):
    d[f"BILL_AMT{i}"] = np.round(lim * np.clip(rng.beta(2, 3, n) + 0.15 * risk, -0.1, 1.2))
for i in range(1, 7):
    d[f"PAY_AMT{i}"] = np.round(np.abs(d[f"BILL_AMT{i}"]) * np.clip(rng.beta(2, 5, n) - 0.08 * risk, 0, 1))
z = -1.6 + 0.9 * risk + 0.35 * np.clip(d["PAY_0"], 0, None) + rng.normal(0, 0.8, n)
d["default payment next month"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
pd.DataFrame(d).to_csv(sys.argv[1], index=False)
print("fixture written", sys.argv[1])
