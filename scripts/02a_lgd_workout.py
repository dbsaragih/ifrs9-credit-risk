"""
Module 2a - Workout LGD on a synthetic Indonesian-style defaulted portfolio (MIT, simulated data)
================================================================================================
Segments: KTA (unsecured), KKB (vehicle), SME (property-secured). Amounts in IDR million.

  Workout LGD   = 1 - PV_EIR(recoveries - costs) / EAD, discounted to the default date
  Methods compared (all EAD-weighted, per segment):
    1. Naive        : average LGD of closed workouts only (ignores open cases)
    2. Recovery curve (chain-ladder): marginal recovery rate per month since default, using every
                      default for the months it has been observed, cumulated over the workout horizon
    3. Cure-adjusted: CR x LGD_cure + (1 - CR) x LGD_non-cure (non-cure LGD by recovery curve)
  Because the data is simulated, every estimate is checked against the true final LGD (answer key).
  Also: time-in-default LGD for Stage 3, LGD by LTV band and by default year.

Run from the repository root:  python scripts/02a_lgd_workout.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.synthetic import REPORTING_DATE, generate_lgd_data    # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
OUT = ROOT / "outputs" / "module2a"
PROC = ROOT / "data" / "processed"
HORIZON = 60          # months of workout considered
SEGS = ["KTA", "KKB", "SME"]


def months_observed(loans: pd.DataFrame) -> pd.Series:
    return ((REPORTING_DATE.year - loans.default_date.dt.year) * 12 + REPORTING_DATE.month - loans.default_date.dt.month)


def recovery_curve(loans: pd.DataFrame, flows: pd.DataFrame, start: int = 0) -> pd.Series:
    """Chain-ladder recovery curve. Returns cumulative discounted net recovery rate (EAD-weighted)
    by month since default. With start=t, recoveries are discounted to month t and only months > t count."""
    if loans.empty:
        return pd.Series(dtype=float)
    obs = loans.set_index("loan_id").months_obs
    f = flows[flows.loan_id.isin(loans.loan_id) & (flows.month_since_default > start)].copy()
    f["pv"] = f.amount * (1 + f.eir) ** (-(f.month_since_default - start) / 12)
    num = f.groupby("month_since_default").pv.sum().reindex(range(start + 1, HORIZON + 1), fill_value=0)
    ead = loans.set_index("loan_id").ead
    den = pd.Series({m: ead[obs >= m].sum() for m in num.index})
    marginal = (num / den.replace(0, np.nan)).fillna(0)
    return marginal.cumsum()


def lgd_from_curve(curve: pd.Series) -> float:
    return float(1 - curve.iloc[-1]) if len(curve) else np.nan


def main():
    OUT.mkdir(parents=True, exist_ok=True); PROC.mkdir(parents=True, exist_ok=True)
    loans, flows, key = generate_lgd_data()
    loans.to_csv(PROC / "synthetic_lgd_loans.csv", index=False)
    flows.to_csv(PROC / "synthetic_lgd_cashflows.csv", index=False)
    key.to_csv(PROC / "synthetic_lgd_answer_key.csv", index=False)
    loans["months_obs"] = months_observed(loans)
    flows = flows.merge(loans[["loan_id", "eir", "ead"]], on="loan_id")
    loans = loans.merge(key, on="loan_id")
    print(f"Defaults: {len(loans):,} | closed {(loans.status == 'closed').mean():.1%} | open {(loans.status == 'open').mean():.1%} "
          f"| reporting date {REPORTING_DATE.date()}")

    # Realised LGD for closed cases
    pv = flows.assign(pv=flows.amount * (1 + flows.eir) ** (-flows.month_since_default / 12)).groupby("loan_id").pv.sum()
    loans["realised_lgd"] = np.where(loans.status == "closed", 1 - loans.loan_id.map(pv).fillna(0) / loans.ead, np.nan)
    loans["close_m"] = ((loans.close_date.dt.year - loans.default_date.dt.year) * 12 + loans.close_date.dt.month - loans.default_date.dt.month)

    wavg = lambda d, c: (d[c] * d.ead).sum() / d.ead.sum()
    rows, curves = [], {}
    for s in SEGS:
        L = loans[loans.segment == s]
        truth = wavg(L, "true_lgd")
        naive = wavg(L[L.status == "closed"], "realised_lgd")
        curves[s] = recovery_curve(L, flows)
        chain = lgd_from_curve(curves[s])
        # cure-adjusted: cure status is known for defaults observed >= 10 months (cures happen within 9)
        M = L[L.months_obs >= 10]
        cr = (M.cured * M.ead).sum() / M.ead.sum()
        lgd_cure = wavg(M[M.cured == 1], "realised_lgd")
        lgd_nc = lgd_from_curve(recovery_curve(M[M.cured == 0], flows))
        cure_adj = cr * lgd_cure + (1 - cr) * lgd_nc
        rows.append(dict(segment=s, n=len(L), open_pct=(L.status == "open").mean(), true_lgd=truth,
                         naive_closed_only=naive, recovery_curve=chain, cure_rate=cr, lgd_cure=lgd_cure,
                         lgd_non_cure=lgd_nc, cure_adjusted=cure_adj))
    comp = pd.DataFrame(rows).set_index("segment")
    for c in ["naive_closed_only", "recovery_curve", "cure_adjusted"]:
        comp[f"err_{c}_pp"] = (comp[c] - comp.true_lgd) * 100
    print("\n=== LGD estimates vs truth (EAD-weighted) ===\n", comp.round(4).to_string())

    # Time-in-default LGD (Stage 3): loans still in workout (not cured, not closed) after t months
    tid_rows = []
    for s in SEGS:
        L = loans[(loans.segment == s) & (loans.cured == 0) & (loans.true_outcome != "cured")]
        for t in [0, 3, 6, 12, 18, 24, 36]:
            alive = L[(L.months_obs >= t) & (L.close_m.isna() | (L.close_m > t))]
            if len(alive) < 50:
                continue
            curve = recovery_curve(alive, flows, start=t)
            tid_rows.append(dict(segment=s, months_in_default=t, n=len(alive), lgd=lgd_from_curve(curve)))
    tid = pd.DataFrame(tid_rows)
    print("\n=== Time-in-default LGD (non-cured, unresolved at t) ===\n",
          tid.pivot(index="months_in_default", columns="segment", values="lgd").round(3).to_string())

    # LGD by LTV band (secured) and by default year - recovery-curve method, checked against truth
    def by_group(df, col):
        out = []
        for g, G in df.groupby(col, observed=True):
            if len(G) < 100:
                continue
            mature = G.months_obs.min() >= {"KTA": 36, "KKB": 24, "SME": 48}.get(G.segment.iloc[0], 48)
            out.append(dict(group=str(g), n=len(G), recovery_curve=lgd_from_curve(recovery_curve(G, flows)), true_lgd=wavg(G, "true_lgd"),
                            min_months_observed=int(G.months_obs.min()),
                            maturity="mature" if mature else "IMMATURE - use pooled segment curve"))
        return pd.DataFrame(out)
    sec = loans[loans.segment.isin(["KKB", "SME"])].copy()
    sec["ltv_band"] = pd.cut(sec.ltv_at_default, [0, .6, .8, 1.0, 9], labels=["≤60%", "60-80%", "80-100%", ">100%"])
    ltv_tbl = pd.concat([by_group(sec[sec.segment == s], "ltv_band").assign(segment=s) for s in ["KKB", "SME"]])
    loans["default_year"] = loans.default_date.dt.year
    year_tbl = pd.concat([by_group(loans[loans.segment == s], "default_year").assign(segment=s) for s in SEGS])
    print("\n=== LGD by LTV band (secured) ===\n", ltv_tbl.round(3).to_string(index=False))
    print("\n=== LGD by default year: recovery-curve estimate vs truth ===\n",
          year_tbl.pivot(index="group", columns="segment", values=["recovery_curve", "true_lgd"]).round(3).to_string())

    # ---------------- charts (light background, dark text)
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "text.color": "#222", "axes.labelcolor": "#222",
                         "xtick.color": "#333", "ytick.color": "#333", "axes.spines.top": False, "axes.spines.right": False})
    col = {"KTA": "#d62728", "KKB": "#1f77b4", "SME": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in SEGS:
        ax.plot(curves[s].index, curves[s].values * 100, color=col[s], label=f"{s}: LGD {lgd_from_curve(curves[s]):.1%}")
    ax.set(xlabel="Months since default", ylabel="Cumulative discounted net recovery, % of EAD", title="Recovery curves (chain-ladder)")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "1_recovery_curves.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(8, 5)); x = np.arange(len(SEGS))
    for i, (c, lab, colr) in enumerate([("true_lgd", "True LGD (answer key)", "#222222"), ("naive_closed_only", "Closed cases only", "#bbbbbb"),
                                         ("recovery_curve", "Recovery curve", "#1f77b4"), ("cure_adjusted", "Cure-adjusted", "#2ca02c")]):
        ax.bar(x + (i - 1.5) * .2, comp[c] * 100, .2, color=colr, label=lab)
    ax.set_xticks(x, SEGS); ax.set(ylabel="LGD %", title="LGD estimation methods vs truth"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "2_methods_vs_truth.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    for s in SEGS:
        t = tid[tid.segment == s]; ax.plot(t.months_in_default, t.lgd * 100, marker="o", color=col[s], label=s)
    ax.set(xlabel="Months already in default (unresolved)", ylabel="LGD %", title="Time-in-default LGD (Stage 3)")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "3_time_in_default_lgd.png", dpi=150); plt.close()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    for ax, s in zip(axs, SEGS):
        v = loans[(loans.segment == s) & (loans.status == "closed")].realised_lgd.clip(-.1, 1.1)
        ax.hist(v, bins=40, color=col[s]); ax.set(title=f"{s}: realised LGD, closed cases", xlabel="LGD")
    fig.tight_layout(); fig.savefig(OUT / "4_lgd_distribution.png", dpi=150); plt.close()

    # ---------------- excel
    from openpyxl.styles import Font, PatternFill
    xl = OUT / "module2a_lgd.xlsx"
    with pd.ExcelWriter(xl, engine="openpyxl") as w:
        comp.reset_index().to_excel(w, sheet_name="1_Methods_vs_truth", index=False)
        pd.DataFrame({s: curves[s] for s in SEGS}).rename_axis("month_since_default").reset_index().to_excel(w, sheet_name="2_Recovery_curves", index=False)
        tid.to_excel(w, sheet_name="3_Time_in_default", index=False)
        ltv_tbl.to_excel(w, sheet_name="4_LGD_by_LTV", index=False)
        year_tbl.to_excel(w, sheet_name="5_LGD_by_year", index=False)
        for ws in w.book.worksheets:
            for row in ws.iter_rows():
                for cell in row:
                    cell.font = Font(name="Aptos Narrow", bold=cell.row == 1, color="000000")
            for cell in ws[1]:
                cell.fill = PatternFill("solid", fgColor="BDD7EE")
            for cc in ws.columns:
                ws.column_dimensions[cc[0].column_letter].width = min(max(len(str(x.value or "")) for x in cc) + 2, 40)
    print(f"\nSaved {xl.relative_to(ROOT)}, charts in {OUT.relative_to(ROOT)}/, data in {PROC.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
