"""Model validation metrics: Gini, KS, gains table, PSI/CSI, calibration, bootstrap CI."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


def gini(y, pd_hat) -> float:
    return 2 * roc_auc_score(y, pd_hat) - 1


def ks(y, score) -> float:
    """KS on a score where higher = safer (or any monotone risk measure)."""
    y, score = np.asarray(y), np.asarray(score)
    return float(stats.ks_2samp(score[y == 0], score[y == 1]).statistic)


def gini_bootstrap(y, p1, p2=None, n_boot: int = 500, seed: int = 0) -> dict:
    """95% CI for Gini of p1, and for Gini(p1) - Gini(p2) when p2 is given (paired bootstrap)."""
    rng = np.random.default_rng(seed)
    y, p1 = np.asarray(y), np.asarray(p1)
    p2 = None if p2 is None else np.asarray(p2)
    g1, d = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        a = gini(y[i], p1[i]); g1.append(a)
        if p2 is not None:
            d.append(a - gini(y[i], p2[i]))
    out = {"gini_lo": np.percentile(g1, 2.5), "gini_hi": np.percentile(g1, 97.5)}
    if p2 is not None:
        out.update(diff_lo=np.percentile(d, 2.5), diff_hi=np.percentile(d, 97.5))
    return out


def score_edges(train_score, n: int = 10) -> np.ndarray:
    e = np.unique(np.quantile(train_score, np.linspace(0, 1, n + 1)))
    e[0], e[-1] = -np.inf, np.inf
    return e


def gains_table(y, score, pd_hat, edges) -> pd.DataFrame:
    """Score bands, riskiest (lowest score) first."""
    d = pd.DataFrame({"y": np.asarray(y), "score": np.asarray(score), "pd": np.asarray(pd_hat)})
    d["band"] = pd.cut(d.score, edges, right=False)
    t = d.groupby("band", observed=False).agg(n=("y", "size"), bads=("y", "sum"), avg_pd=("pd", "mean"),
                                              min_score=("score", "min"), max_score=("score", "max"))
    t["goods"] = t.n - t.bads
    t["bad_rate"] = t.bads / t.n
    t["cum_bad_pct"] = t.bads.cumsum() / t.bads.sum()
    t["cum_good_pct"] = t.goods.cumsum() / t.goods.sum()
    t["ks"] = t.cum_bad_pct - t.cum_good_pct
    t["lift"] = t.bad_rate / d.y.mean()
    t = t.reset_index()
    t["band"] = t.band.astype(str)
    return t


def psi(expected, actual, edges) -> tuple[float, pd.DataFrame]:
    e = pd.cut(pd.Series(expected), edges, right=False).value_counts(normalize=True, sort=False)
    a = pd.cut(pd.Series(actual), edges, right=False).value_counts(normalize=True, sort=False)
    e, a = e.clip(lower=1e-4), a.clip(lower=1e-4)
    t = pd.DataFrame({"expected_pct": e, "actual_pct": a})
    t["psi"] = (t.actual_pct - t.expected_pct) * np.log(t.actual_pct / t.expected_pct)
    t = t.reset_index(names="band")
    t["band"] = t.band.astype(str)
    return float(t.psi.sum()), t


def csi_binned(expected_bins: pd.Series, actual_bins: pd.Series) -> float:
    """Characteristic stability index on already-binned values (e.g. WoE bins)."""
    e = expected_bins.value_counts(normalize=True)
    a = actual_bins.value_counts(normalize=True)
    idx = e.index.union(a.index)
    e, a = e.reindex(idx).fillna(1e-4).clip(lower=1e-4), a.reindex(idx).fillna(1e-4).clip(lower=1e-4)
    return float(((a - e) * np.log(a / e)).sum())


def rag(value: float, green: float, amber: float, higher_is_better: bool = True) -> str:
    if higher_is_better:
        return "GREEN" if value >= green else "AMBER" if value >= amber else "RED"
    return "GREEN" if value < green else "AMBER" if value < amber else "RED"


def hosmer_lemeshow(y, pd_hat, groups: int = 10) -> tuple[float, float]:
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(pd_hat)})
    d["g"] = pd.qcut(d.p.rank(method="first"), groups, labels=False)
    t = d.groupby("g").agg(n=("y", "size"), obs=("y", "sum"), exp=("p", "sum"))
    chi2 = (((t.obs - t.exp) ** 2) / (t.exp * (1 - t.exp / t.n))).sum()
    return float(chi2), float(1 - stats.chi2.cdf(chi2, groups - 2))


def calibration_by_band(y, pd_hat, edges, score) -> pd.DataFrame:
    """Observed vs predicted per score band with Jeffreys-interval test (one-sided, under-prediction)."""
    t = gains_table(y, score, pd_hat, edges)
    t["jeffreys_p"] = [stats.beta.cdf(p, b + 0.5, n - b + 0.5) if n > 0 else np.nan
                       for p, b, n in zip(t.avg_pd, t.bads, t.n)]
    t["obs_vs_exp"] = t.bad_rate / t.avg_pd
    return t[["band", "n", "bads", "bad_rate", "avg_pd", "obs_vs_exp", "jeffreys_p"]]
