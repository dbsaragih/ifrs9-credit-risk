"""Weight of Evidence (WoE) binning with monotonic merging, Information Value (IV).

Convention: WoE = ln(%good / %bad). Positive WoE = lower risk than average.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 0.5  # Laplace-style smoothing for empty cells


def _stats(bins: pd.Series, y: pd.Series) -> pd.DataFrame:
    t = pd.DataFrame({"bin": bins, "y": y}).groupby("bin", observed=True)["y"].agg(n="size", bad="sum")
    t["good"] = t.n - t.bad
    tot_g, tot_b = t.good.sum(), t.bad.sum()
    t["pct_good"] = (t.good + EPS) / (tot_g + EPS * len(t))
    t["pct_bad"] = (t.bad + EPS) / (tot_b + EPS * len(t))
    t["bad_rate"] = t.bad / t.n
    t["woe"] = np.log(t.pct_good / t.pct_bad)
    t["iv"] = (t.pct_good - t.pct_bad) * t.woe
    t["pct_n"] = t.n / t.n.sum()
    return t


def fit_numeric(x: pd.Series, y: pd.Series, max_bins: int = 6, min_pct: float = 0.05,
                init_bins: int = 20) -> np.ndarray:
    """Return bin edges whose bad rate is monotonic and every bin has >= min_pct of rows."""
    vals = np.sort(x.dropna().unique())
    if len(vals) <= init_bins:
        cuts = (vals[:-1] + vals[1:]) / 2                       # one bin per distinct value
    else:
        cuts = np.unique(np.quantile(x.dropna(), np.linspace(0, 1, init_bins + 1)[1:-1]))
    edges = np.concatenate([[-np.inf], cuts, [np.inf]])
    rho = np.corrcoef(x.rank(), y)[0, 1]
    direction = 1 if rho >= 0 else -1                           # +1: bad rate rises with x

    def table(e):
        return _stats(pd.cut(x, e, right=False), y)

    while len(edges) > 2:
        t = table(edges)
        br, pn = t.bad_rate.values, t.pct_n.values
        k = len(br)
        diffs = np.diff(br) * direction                          # should all be > 0
        viol = np.where(diffs <= 0)[0]
        small = np.where(pn < min_pct)[0]
        if len(viol):
            i = viol[np.argmin(np.abs(np.diff(br))[viol])]       # merge least-different violating pair
        elif len(small):
            j = small[0]
            if j == 0:
                i = 0
            elif j == k - 1:
                i = k - 2
            else:
                i = j - 1 if abs(br[j] - br[j - 1]) <= abs(br[j] - br[j + 1]) else j
        elif k > max_bins:
            i = int(np.argmin(np.abs(np.diff(br))))
        else:
            break
        edges = np.delete(edges, i + 1)                          # remove the cut between bin i and i+1
    return edges


def fit_categorical(x: pd.Series, y: pd.Series, min_pct: float = 0.05) -> dict:
    """Map each category to a group label; small categories join the closest bad-rate neighbour."""
    t = _stats(x.astype(str), y).sort_values("bad_rate")
    groups = {c: c for c in t.index}
    small = t[t.pct_n < min_pct].index.tolist()
    order = t.index.tolist()
    for c in small:
        i = order.index(c)
        nb = order[i - 1] if i > 0 else order[i + 1] if i + 1 < len(order) else c
        for k, v in groups.items():
            if v == c:
                groups[k] = groups[nb]
    return groups


class WoEBinner:
    """Fit on train, transform any sample. Stores a bin table per variable."""

    def __init__(self, max_bins: int = 6, min_pct: float = 0.05):
        self.max_bins, self.min_pct = max_bins, min_pct
        self.spec: dict[str, dict] = {}
        self.tables: dict[str, pd.DataFrame] = {}

    def _bin(self, col: str, x: pd.Series) -> pd.Series:
        s = self.spec[col]
        if s["type"] == "num":
            return pd.cut(x, s["edges"], right=False).astype(str)
        return x.astype(str).map(s["groups"]).fillna("__unseen__")

    def fit(self, X: pd.DataFrame, y: pd.Series, categorical: list[str] = ()):
        for col in X.columns:
            if col in categorical:
                self.spec[col] = {"type": "cat", "groups": fit_categorical(X[col], y, self.min_pct)}
            else:
                self.spec[col] = {"type": "num", "edges": fit_numeric(X[col], y, self.max_bins, self.min_pct)}
            t = _stats(self._bin(col, X[col]), y)
            if self.spec[col]["type"] == "num":            # keep numeric bins in ascending order
                order = pd.cut(pd.Series([0.0]), self.spec[col]["edges"], right=False).cat.categories.astype(str)
                t = t.reindex([o for o in order if o in t.index])      # same labels as _bin()
            assert len(t) == self._bin(col, X[col]).nunique(), f"bin table incomplete for {col}"
            self.spec[col]["woe"] = t.woe.to_dict()
            self.tables[col] = t.reset_index().assign(variable=col)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col, s in self.spec.items():
            out[col] = self._bin(col, X[col]).map(s["woe"]).fillna(0.0)   # unseen bin -> neutral WoE
        return pd.DataFrame(out, index=X.index)

    def iv(self) -> pd.Series:
        return pd.Series({c: t.iv.sum() for c, t in self.tables.items()}).sort_values(ascending=False)

    def bin_table(self) -> pd.DataFrame:
        cols = ["variable", "bin", "n", "pct_n", "good", "bad", "bad_rate", "woe", "iv"]
        return pd.concat(self.tables.values(), ignore_index=True)[cols]
