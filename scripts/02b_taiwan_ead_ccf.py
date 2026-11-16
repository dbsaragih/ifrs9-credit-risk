"""
Module 2b - EAD and Credit Conversion Factor (CCF) for revolving credit, UCI Taiwan data (CC BY 4.0)
====================================================================================================
Fixed-horizon approach:
  reference date  = April 2005 (bill_m6, oldest month)       -> drawn balance, undrawn limit
  default         = October 2005 (default_flag)
  EAD proxy       = September 2005 statement balance (bill_m1), the last balance before default

  CCF = (EAD - drawn_ref) / (limit - drawn_ref)          defined where undrawn_ref > 0
  EAD = drawn + CCF x undrawn                             (IFRS 9: unbiased, no regulatory floor)

Compared estimators: CCF, limit factor (EAD / limit), balance factor (EAD / drawn).
Run from the repository root:  python scripts/02b_taiwan_ead_ccf.py
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_loader import load_taiwan                     # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
OUT = ROOT / "outputs" / "module2b"
SEED = 42
UTIL_BANDS = [-np.inf, 0.10, 0.30, 0.50, 0.70, 0.90, np.inf]
UTIL_LABELS = ["<10%", "10-30%", "30-50%", "50-70%", "70-90%", "≥90%"]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame({"limit": df.limit_bal, "drawn_ref": df.bill_m6, "ead": df.bill_m1,
                      "dlq_ref": df.dlq_m6, "default_flag": df.default_flag})
    d["undrawn_ref"] = d.limit - d.drawn_ref
    d["util_ref"] = d.drawn_ref / d.limit
    d["ccf"] = np.where(d.undrawn_ref > 0, (d.ead - d.drawn_ref) / d.undrawn_ref.where(d.undrawn_ref > 0), np.nan)
    d["util_band"] = pd.cut(d.util_ref, UTIL_BANDS, labels=UTIL_LABELS, right=False)
    d["delinquent_ref"] = np.where(d.dlq_ref > 0, "delinquent", "current")
    d["limit_band"] = pd.qcut(d.limit, 3, labels=["low limit", "mid limit", "high limit"])
    return d


def fit_ccf_table(train: pd.DataFrame, min_n: int = 100) -> pd.DataFrame:
    """Per utilisation band x delinquency cell:
    ccf_mean  : count-weighted mean of account CCFs (naive; winsorised at -1..2)
    ccf_wtd   : exposure-weighted CCF = sum(EAD - drawn) / sum(undrawn)  -> unbiased for total EAD
    bal_factor: sum(EAD) / sum(drawn), used by the hybrid method for near-limit accounts
    Thin cells fall back to the utilisation-band value."""
    t = train.dropna(subset=["ccf"]).copy()
    t["ccf_c"] = t.ccf.clip(-1, 2)
    t["inc"] = t.ead - t.drawn_ref
    def agg(g):
        pos = g[g.drawn_ref > 0]
        return pd.Series({"n": len(g), "ccf_mean": g.ccf_c.mean(), "ccf_wtd": g.inc.sum() / g.undrawn_ref.sum(),
                          "bal_factor": pos.ead.sum() / pos.drawn_ref.sum() if len(pos) else np.nan})
    band = t.groupby("util_band", observed=True).apply(agg)
    cell = t.groupby(["util_band", "delinquent_ref"], observed=True).apply(agg).reset_index()
    for c in ["ccf_mean", "ccf_wtd", "bal_factor"]:
        cell[c] = np.where(cell.n >= min_n, cell[c], cell.util_band.map(band[c]).astype(float))
    return cell


def predict(test: pd.DataFrame, ccf_tbl: pd.DataFrame, lf: float, bf: float) -> pd.DataFrame:
    p = test.merge(ccf_tbl, on=["util_band", "delinquent_ref"], how="left", suffixes=("", "_tbl"))
    und = p.undrawn_ref.clip(lower=0)
    p["ead_ccf_naive"] = p.drawn_ref + p.ccf_mean * und
    p["ead_ccf"] = p.drawn_ref + p.ccf_wtd * und
    near_limit = p.util_band.astype(str).eq("≥90%")
    p["ead_hybrid"] = np.where(near_limit, p.bal_factor * p.drawn_ref, p.ead_ccf)   # CCF unstable when undrawn ~ 0
    p["ead_lf"] = lf * p.limit
    p["ead_bf"] = bf * p.drawn_ref
    return p


def accuracy(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, name in [("ead_ccf_naive", "CCF, count-weighted (naive)"), ("ead_ccf", "CCF, exposure-weighted"),
                      ("ead_hybrid", "Hybrid: weighted CCF + balance factor ≥90% util"),
                      ("ead_lf", "Limit factor"), ("ead_bf", "Balance factor")]:
        err = p[col] - p.ead
        rows.append(dict(method=name, total_predicted=p[col].sum(), total_actual=p.ead.sum(),
                         bias_ratio=p[col].sum() / p.ead.sum(), mae_pct_of_limit=(err.abs() / p.limit).mean(),
                         r2=1 - (err ** 2).sum() / ((p.ead - p.ead.mean()) ** 2).sum()))
    return pd.DataFrame(rows)


def main(data_path):
    OUT.mkdir(parents=True, exist_ok=True)
    d = prepare(load_taiwan(data_path))
    dft = d[d.default_flag == 1]
    print(f"Defaulted accounts: {len(dft):,} of {len(d):,}")
    print(f"  with undrawn limit > 0 at reference (CCF defined): {dft.ccf.notna().sum():,}")
    print(f"  at or over limit at reference: {(dft.undrawn_ref <= 0).sum():,}")
    print(f"  negative/zero balance at reference: {(dft.drawn_ref <= 0).sum():,}")

    # --- Realised CCF profile: defaulters vs non-defaulters (the core EAD insight)
    prof = (d.dropna(subset=["ccf"]).assign(ccf_c=lambda x: x.ccf.clip(-1, 2))
            .groupby(["util_band", "default_flag"], observed=True).ccf_c.agg(["mean", "median", "size"]).unstack())
    prof.columns = [f"{a}_{'default' if b else 'non_default'}" for a, b in prof.columns]
    print("\n=== Realised CCF by utilisation at reference ===\n", prof.round(3).to_string())
    dist = dft.ccf.dropna()
    dist_tbl = pd.Series({"n": len(dist), "mean (winsorised -1..2)": dist.clip(-1, 2).mean(), "median": dist.median(),
                          "share < 0 (paid down)": (dist < 0).mean(), "share 0-1": dist.between(0, 1).mean(),
                          "share > 1 (over limit)": (dist > 1).mean()})
    print("\n=== Realised CCF distribution, defaulters ===\n", dist_tbl.round(4).to_string())

    # --- Estimation on 70% of defaulters, validation on 30%
    tr, te = train_test_split(dft, test_size=0.3, random_state=SEED, stratify=dft.util_band.astype(str))
    ccf_tbl = fit_ccf_table(tr)
    lf = tr.ead.sum() / tr.limit.sum()                              # portfolio-weighted factors
    bf = tr.loc[tr.drawn_ref > 0, "ead"].sum() / tr.loc[tr.drawn_ref > 0, "drawn_ref"].sum()
    print("\n=== CCF lookup table (train) ===\n", ccf_tbl.round(3).to_string(index=False))
    print(f"Limit factor {lf:.3f} | balance factor {bf:.3f}")

    p = predict(te, ccf_tbl, lf, bf)
    acc = accuracy(p)
    print("\n=== EAD validation on held-out defaulters ===\n", acc.round(4).to_string(index=False))
    by_band = p.groupby("util_band", observed=True).agg(n=("ead", "size"), actual=("ead", "sum"), pred_naive=("ead_ccf_naive", "sum"),
                                                        pred_ccf=("ead_ccf", "sum"), pred_hybrid=("ead_hybrid", "sum"),
                                                        pred_lf=("ead_lf", "sum"), pred_bf=("ead_bf", "sum"))
    for c in ["pred_naive", "pred_ccf", "pred_hybrid", "pred_lf", "pred_bf"]:
        by_band[c.replace("pred", "bias")] = by_band[c] / by_band.actual
    print("\n=== Bias ratio (predicted / actual) by utilisation band ===\n",
          by_band[["n", "bias_naive", "bias_ccf", "bias_hybrid", "bias_lf", "bias_bf"]].round(3).to_string())

    # --- Charts (light background, dark text)
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "text.color": "#222", "axes.labelcolor": "#222",
                         "xtick.color": "#333", "ytick.color": "#333", "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8, 5)); x = np.arange(len(prof))
    ax.bar(x - .2, prof["mean_non_default"], .4, color="#1f77b4", label="Non-defaulters")
    ax.bar(x + .2, prof["mean_default"], .4, color="#d62728", label="Defaulters")
    ax.axhline(0, color="#555", lw=.8); ax.set_xticks(x, prof.index.astype(str))
    ax.set(xlabel="Utilisation at reference (Apr-2005)", ylabel="Mean CCF", title="Drawdown of undrawn limit: defaulters vs non-defaulters")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "1_ccf_by_util.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(dist.clip(-1, 2), bins=60, color="#d62728", alpha=.8)
    ax.set(xlabel="Realised CCF (winsorised at -1 and 2)", ylabel="Defaulted accounts", title="Realised CCF distribution, defaulters")
    fig.tight_layout(); fig.savefig(OUT / "2_ccf_distribution.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(8, 5)); x = np.arange(len(by_band))
    for i, (c, lab, colr) in enumerate([("bias_naive", "CCF count-weighted", "#bbbbbb"), ("bias_hybrid", "Hybrid (recommended)", "#2ca02c"),
                                         ("bias_lf", "Limit factor", "#ff7f0e"), ("bias_bf", "Balance factor", "#9467bd")]):
        ax.bar(x + (i - 1.5) * .2, by_band[c], .2, color=colr, label=lab)
    ax.set_ylim(0, 3)
    ax.axhline(1, color="#222", ls="--", lw=1); ax.set_xticks(x, by_band.index.astype(str))
    ax.set(xlabel="Utilisation at reference", ylabel="Predicted / actual EAD", title="EAD bias by segment, held-out defaulters")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "3_ead_bias_by_band.png", dpi=150); plt.close()

    # --- Excel
    from openpyxl.styles import Font, PatternFill
    xl = OUT / "module2b_ead_ccf.xlsx"
    with pd.ExcelWriter(xl, engine="openpyxl") as w:
        acc.to_excel(w, sheet_name="1_EAD_accuracy", index=False)
        ccf_tbl.to_excel(w, sheet_name="2_CCF_table", index=False)
        by_band.reset_index().astype({"util_band": str}).to_excel(w, sheet_name="3_Bias_by_band", index=False)
        prof.reset_index().astype({"util_band": str}).to_excel(w, sheet_name="4_CCF_profile", index=False)
        dist_tbl.rename_axis("statistic").reset_index(name="value").to_excel(w, sheet_name="5_CCF_distribution", index=False)
        pd.DataFrame({"factor": ["limit_factor", "balance_factor"], "value": [lf, bf]}).to_excel(w, sheet_name="6_Factors", index=False)
        for ws in w.book.worksheets:
            for row in ws.iter_rows():
                for cell in row:
                    cell.font = Font(name="Aptos Narrow", bold=cell.row == 1, color="000000")
            for cell in ws[1]:
                cell.fill = PatternFill("solid", fgColor="BDD7EE")
            for cc in ws.columns:
                ws.column_dimensions[cc[0].column_letter].width = min(max(len(str(x.value or "")) for x in cc) + 2, 45)
    print(f"\nSaved {xl.relative_to(ROOT)} and charts in {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default=None)
    main(ap.parse_args().data)
