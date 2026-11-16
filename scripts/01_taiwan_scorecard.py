"""
Module 1 - PD scorecard on the UCI Taiwan credit card dataset (CC BY 4.0)
==========================================================================
Champion : WoE logistic regression scorecard (points-based, explainable)
Challenger: XGBoost with monotonic constraints
Validation: Gini (with bootstrap CI), KS, gains/lift, PSI, CSI, calibration (Jeffreys, Hosmer-Lemeshow)

Run from the repository root:
    python scripts/01_taiwan_scorecard.py
    python scripts/01_taiwan_scorecard.py --data path/to/file.zip
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
import statsmodels.api as sm
import xgboost as xgb
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_loader import load_taiwan                      # noqa: E402
from src.validation import (gains_table, gini, gini_bootstrap, hosmer_lemeshow, ks, psi,  # noqa: E402
                            rag, score_edges, calibration_by_band, csi_binned)
from src.woe import WoEBinner                                 # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
SEED = 42
OUT = ROOT / "outputs" / "module1"
PDO, BASE_SCORE, BASE_ODDS = 20, 600, 50        # 600 points at good:bad odds of 50:1; 20 points doubles the odds
FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)


# ------------------------------------------------------------------ 1. FEATURES
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Behavioural features from 6 months of history. m1 = Sep-2005 (latest), m6 = Apr-2005."""
    f = pd.DataFrame(index=df.index)
    f["limit_bal"] = df.limit_bal
    f["age"] = df.age
    f["education"] = df.education
    f["marriage"] = df.marriage
    dlq = df[[f"dlq_m{i}" for i in range(1, 7)]]
    for i in range(1, 4):
        f[f"dlq_m{i}"] = df[f"dlq_m{i}"]                       # -2 no use, -1 paid in full, 0 revolving, 1..8 months late
    f["max_dlq_6m"] = dlq.clip(lower=0).max(axis=1)
    f["n_months_dlq_6m"] = (dlq > 0).sum(axis=1)
    f["n_months_paid_full_6m"] = (dlq == -1).sum(axis=1)
    lim = df.limit_bal.replace(0, np.nan)
    util = pd.concat([df[f"bill_m{i}"] / lim for i in range(1, 7)], axis=1)
    f["util_m1"] = util.iloc[:, 0].clip(-0.5, 2)
    f["avg_util_6m"] = util.mean(axis=1).clip(-0.5, 2)
    f["util_change_6m"] = (util.iloc[:, 0] - util.iloc[:, 5]).clip(-2, 2)
    # Payment in month i settles the bill of month i+1 (previous statement)
    ratios = []
    for i in range(1, 6):
        bill_prev = df[f"bill_m{i + 1}"]
        r = np.where(bill_prev > 0, df[f"paid_m{i}"] / bill_prev.where(bill_prev > 0), 1.0)
        ratios.append(pd.Series(r, index=df.index).clip(0, 2))
    pr = pd.concat(ratios, axis=1)
    f["pay_ratio_m1"] = pr.iloc[:, 0]
    f["avg_pay_ratio_3m"] = pr.iloc[:, :3].mean(axis=1)
    f["n_months_no_bill_6m"] = (df[[f"bill_m{i}" for i in range(1, 7)]] <= 0).sum(axis=1)
    return f


CATEGORICAL = ["education", "marriage"]


# ------------------------------------------------------------------ 2. LOGISTIC SCORECARD
def select_and_fit(woe_tr: pd.DataFrame, y_tr: pd.Series, iv: pd.Series, log: list) -> tuple:
    cand = iv[iv >= 0.02].index.tolist()
    log.append(f"IV >= 0.02: kept {len(cand)} of {len(iv)} features")
    # Correlation filter: among pairs with |r| > 0.7 keep the higher-IV feature
    corr = woe_tr[cand].corr().abs()
    keep = []
    for c in sorted(cand, key=lambda c: -iv[c]):
        if all(corr.loc[c, k] <= 0.7 for k in keep):
            keep.append(c)
        else:
            log.append(f"  drop {c}: |corr| > 0.7 with a higher-IV feature")
    # Backward elimination: drop wrong sign (WoE coef must be negative) or p > 0.05
    while True:
        m = sm.Logit(y_tr, sm.add_constant(woe_tr[keep])).fit(disp=0)
        coef, pv = m.params.drop("const"), m.pvalues.drop("const")
        bad_sign = coef[coef >= 0]
        if len(bad_sign):
            worst = bad_sign.index[0]; log.append(f"  drop {worst}: positive WoE coefficient (sign reversal)")
        elif pv.max() > 0.05:
            worst = pv.idxmax(); log.append(f"  drop {worst}: p-value {pv.max():.3f}")
        else:
            break
        keep.remove(worst)
    log.append(f"Final logistic model: {len(keep)} features")
    return m, keep


def scorecard_points(model, features, binner: WoEBinner) -> pd.DataFrame:
    """Points per bin: -(beta*WoE + alpha/n) * factor + offset/n. Summing gives the total score."""
    a, n = model.params["const"], len(features)
    rows = []
    for v in features:
        b = model.params[v]
        for _, r in binner.tables[v].iterrows():
            pts = -(b * r.woe + a / n) * FACTOR + OFFSET / n
            rows.append(dict(variable=v, bin=r.bin, n=r.n, bad_rate=r.bad_rate, woe=r.woe, coef=b, points=round(pts)))
    return pd.DataFrame(rows)


def to_score(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return OFFSET + FACTOR * np.log((1 - p) / p)


# ------------------------------------------------------------------ 3. MAIN
def main(data_path: str | None):
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_taiwan(data_path)
    y = df.default_flag.astype(int)
    X = build_features(df)
    print(f"Rows {len(df):,} | default rate {y.mean():.2%} | features {X.shape[1]} (sex excluded)")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    samples = pd.DataFrame({"sample": ["TRAIN", "TEST"], "n": [len(y_tr), len(y_te)],
                            "defaults": [y_tr.sum(), y_te.sum()], "default_rate": [y_tr.mean(), y_te.mean()]})

    # --- Champion: WoE logistic
    log = []
    binner = WoEBinner(max_bins=6, min_pct=0.05).fit(X_tr, y_tr, categorical=CATEGORICAL)
    iv = binner.iv()
    woe_tr, woe_te = binner.transform(X_tr), binner.transform(X_te)
    lr, feats = select_and_fit(woe_tr, y_tr, iv, log)
    print("\n".join(log))
    pd_lr_tr = np.asarray(lr.predict(sm.add_constant(woe_tr[feats])))
    pd_lr_te = np.asarray(lr.predict(sm.add_constant(woe_te[feats])))
    card = scorecard_points(lr, feats, binner)
    # Reconciliation: summed scorecard points must reproduce the model score (difference = rounding only)
    pts_te = sum(binner._bin(v, X_te[v]).map(card[card.variable == v].set_index("bin").points) for v in feats)
    recon = np.abs(pts_te.values - to_score(pd_lr_te)).max()
    print(f"Scorecard reconciliation: max |points total - model score| = {recon:.2f} (rounding tolerance {0.5 * len(feats):.1f})")
    assert recon <= 0.5 * len(feats) + 1e-6, "Scorecard points do not reconcile to the model score"
    vif = pd.Series([variance_inflation_factor(woe_tr[feats].values, i) for i in range(len(feats))], index=feats)
    coef_tbl = pd.DataFrame({"coef": lr.params, "std_err": lr.bse, "p_value": lr.pvalues}).join(
        iv.rename("iv")).join(vif.rename("vif"))

    # --- Challenger: XGBoost with monotone constraints
    Xd_tr = pd.get_dummies(X_tr, columns=CATEGORICAL, dtype=int)
    Xd_te = pd.get_dummies(X_te, columns=CATEGORICAL, dtype=int).reindex(columns=Xd_tr.columns, fill_value=0)
    mono = {"dlq_m1": 1, "dlq_m2": 1, "dlq_m3": 1, "max_dlq_6m": 1, "n_months_dlq_6m": 1,
            "limit_bal": -1, "avg_pay_ratio_3m": -1, "pay_ratio_m1": -1}
    Xf, Xes, yf, yes = train_test_split(Xd_tr, y_tr, test_size=0.2, random_state=SEED, stratify=y_tr)
    xg = xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc", n_estimators=2000, learning_rate=0.03,
                           max_depth=4, min_child_weight=30, subsample=0.8, colsample_bytree=0.8, reg_lambda=5,
                           gamma=1, monotone_constraints=tuple(mono.get(c, 0) for c in Xd_tr.columns),
                           early_stopping_rounds=100, random_state=SEED, n_jobs=-1)
    xg.fit(Xf, yf, eval_set=[(Xes, yes)], verbose=False)
    pd_xg_tr, pd_xg_te = xg.predict_proba(Xd_tr)[:, 1], xg.predict_proba(Xd_te)[:, 1]
    shap = xg.get_booster().predict(xgb.DMatrix(Xd_te), pred_contribs=True)[:, :-1]
    xg_imp = pd.DataFrame({"feature": Xd_tr.columns, "mean_abs_shap": np.abs(shap).mean(0)}).sort_values(
        "mean_abs_shap", ascending=False)

    # --- Validation
    res = {}
    for name, ptr, pte in [("Logistic (champion)", pd_lr_tr, pd_lr_te), ("XGBoost (challenger)", pd_xg_tr, pd_xg_te)]:
        s_tr, s_te = to_score(ptr), to_score(pte)
        edges = score_edges(s_tr)
        ci = gini_bootstrap(y_te, pte)
        hl, hl_p = hosmer_lemeshow(y_te, pte)
        p_val, p_tbl = psi(s_tr, s_te, edges)
        res[name] = dict(
            metrics=dict(gini_train=gini(y_tr, ptr), gini_test=gini(y_te, pte), gini_test_ci=f"{ci['gini_lo']:.3f}–{ci['gini_hi']:.3f}",
                         ks_train=ks(y_tr, s_tr), ks_test=ks(y_te, s_te),
                         gini_drop_pct=(1 - gini(y_te, pte) / gini(y_tr, ptr)) * 100,
                         avg_pd_test=float(np.mean(pte)), odr_test=float(y_te.mean()),
                         psi_train_test=p_val, hosmer_lemeshow_p=hl_p),
            gains=gains_table(y_te, s_te, pte, edges), psi=p_tbl,
            calib=calibration_by_band(y_te, pte, edges, s_te), s_te=s_te, p_te=pte)
    perf = pd.DataFrame({k: v["metrics"] for k, v in res.items()}).T
    diff = gini_bootstrap(y_te, pd_xg_te, pd_lr_te)
    print("\n=== PERFORMANCE ===\n", perf.round(4).to_string())
    print(f"XGBoost minus logistic Gini (test), 95% CI: {diff['diff_lo']:+.3f} to {diff['diff_hi']:+.3f}")

    # CSI on WoE bins for the champion's features
    csi = pd.DataFrame({"feature": feats, "csi": [csi_binned(binner._bin(f, X_tr[f]), binner._bin(f, X_te[f])) for f in feats]})

    champ = res["Logistic (champion)"]
    g = champ["gains"]
    monotonic = bool((g.bad_rate.diff().dropna() <= 0.005).all())
    m = perf.loc["Logistic (champion)"]
    verdict = pd.DataFrame([
        ["Gini (test)", f"{m.gini_test:.3f}  [{m.gini_test_ci}]", "≥0.40 G / 0.30–0.40 A", rag(m.gini_test, .40, .30)],
        ["KS (test)", f"{m.ks_test:.3f}", "≥0.30 G / 0.20–0.30 A", rag(m.ks_test, .30, .20)],
        ["Gini drop train→test", f"{m.gini_drop_pct:.1f}%", "≤10% G / ≤20% A", rag(m.gini_drop_pct, 10, 20, False)],
        ["Score PSI train→test", f"{m.psi_train_test:.4f}", "<0.10 G / <0.25 A", rag(m.psi_train_test, .10, .25, False)],
        ["Max feature CSI", f"{csi.csi.max():.4f}", "<0.10 G / <0.25 A", rag(csi.csi.max(), .10, .25, False)],
        ["Avg PD vs observed (test)", f"{m.avg_pd_test:.2%} vs {m.odr_test:.2%}", "ratio 0.9–1.1 G",
         "GREEN" if 0.9 <= m.odr_test / m.avg_pd_test <= 1.1 else "AMBER"],
        ["Rank ordering by decile", str(monotonic), "Bad rate falls as score rises", "GREEN" if monotonic else "AMBER"],
        ["Max VIF", f"{vif.max():.2f}", "<5 G / <10 A", rag(vif.max(), 5, 10, False)],
        ["Challenger Gini uplift", f"{diff['diff_lo']:+.3f} to {diff['diff_hi']:+.3f}", "CI excludes 0 = significant",
         "INFO"],
    ], columns=["test", "value", "threshold", "status"])
    print("\n=== CHAMPION VALIDATION VERDICT ===\n", verdict.to_string(index=False))
    print("\n=== GAINS TABLE, champion, test (riskiest band first) ===\n",
          g[["band", "n", "bads", "bad_rate", "avg_pd", "cum_bad_pct", "ks", "lift"]].round(4).to_string(index=False))

    # ------------------------------------------------------------------ 4. CHARTS (light background, dark text)
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "text.color": "#222",
                         "axes.labelcolor": "#222", "xtick.color": "#333", "ytick.color": "#333",
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    col = {"Logistic (champion)": "#1f77b4", "XGBoost (challenger)": "#ff7f0e"}
    fig, ax = plt.subplots(figsize=(6, 5))
    for k, v in res.items():
        fpr, tpr, _ = roc_curve(y_te, v["p_te"]); ax.plot(fpr, tpr, color=col[k], label=f"{k}: Gini {v['metrics']['gini_test']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#999"); ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC, test sample")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "1_roc.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    s, yy = champ["s_te"], y_te.values; o = np.argsort(s)
    cb, cg = np.cumsum(yy[o] == 1) / yy.sum(), np.cumsum(yy[o] == 0) / (yy == 0).sum(); i = np.argmax(cb - cg)
    ax.plot(s[o], cb, color="#d62728", label="Cumulative defaults"); ax.plot(s[o], cg, color="#1f77b4", label="Cumulative non-defaults")
    ax.vlines(s[o][i], cg[i], cb[i], color="#222", ls=":", label=f"KS {cb[i] - cg[i]:.3f} at score {s[o][i]:.0f}")
    ax.set(xlabel="Score", ylabel="Cumulative share", title="KS chart, champion, test"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "2_ks.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(7, 5)); t = iv.sort_values()
    ax.barh(t.index, t.values, color=["#2ca02c" if v in feats else "#bbb" for v in t.index])
    ax.axvline(0.02, color="#d62728", ls="--", lw=1); ax.set(xlabel="Information value", title="IV by feature (green = in final model)")
    fig.tight_layout(); fig.savefig(OUT / "3_iv.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    for k, v in res.items():
        ax.plot(range(1, len(v["gains"]) + 1), v["gains"].bad_rate * 100, marker="o", color=col[k], label=k)
    ax.set(xlabel="Score decile (1 = lowest score)", ylabel="Default rate %", title="Rank ordering, test"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "4_rank_ordering.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(8, 5)); c = champ["calib"]; x = np.arange(len(c))
    ax.bar(x - .2, c.avg_pd * 100, .4, color="#1f77b4", label="Predicted PD"); ax.bar(x + .2, c.bad_rate * 100, .4, color="#d62728", label="Observed default rate")
    ax.set_xticks(x, range(1, len(c) + 1)); ax.set(xlabel="Score decile (1 = lowest score)", ylabel="%", title="Calibration, champion, test")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "5_calibration.png", dpi=150); plt.close()

    # ------------------------------------------------------------------ 5. ACCOUNT-LEVEL SCORES
    # Every application: value, bin and points per scorecard characteristic, then total score and PD.
    woe_all = binner.transform(X)
    pd_all = np.asarray(lr.predict(sm.add_constant(woe_all[feats], has_constant="add")))
    acc = pd.DataFrame({"app_id": df.app_id if "app_id" in df else df.index + 1,
                        "sample": np.where(X.index.isin(X_tr.index), "TRAIN", "TEST"),
                        "actual_default": y.values})
    for v in feats:
        pts_map = card[card.variable == v].set_index("bin").points
        b = binner._bin(v, X[v])
        acc[f"{v} | value"] = X[v].round(4).values
        acc[f"{v} | bin"] = b.values
        acc[f"{v} | points"] = b.map(pts_map).astype(int).values
    pts_cols = [c for c in acc.columns if c.endswith("| points")]
    acc["total_score"] = acc[pts_cols].sum(axis=1)
    acc["model_score"] = to_score(pd_all).round(1)
    acc["pd"] = pd_all.round(5)
    lr_edges = score_edges(to_score(pd_lr_tr))
    acc["score_decile"] = pd.cut(acc.model_score, lr_edges, labels=range(1, len(lr_edges)), right=False).astype(int)
    acc = acc.sort_values("app_id").reset_index(drop=True)
    print(f"\nAccount-level scores: {len(acc):,} applications | score range {acc.total_score.min()}–{acc.total_score.max()} "
          f"| max |total - model score| {np.abs(acc.total_score - acc.model_score).max():.2f}")
    print(acc.head(5).to_string(index=False))
    card_view = card[["variable", "bin", "n", "bad_rate", "woe", "points"]].copy()
    card_view.insert(0, "characteristic_no", card_view.variable.map({v: i + 1 for i, v in enumerate(feats)}))
    acc.to_csv(OUT / "module1_scored_accounts.csv", index=False)

    # ------------------------------------------------------------------ 6. EXCEL
    from openpyxl.styles import Font, PatternFill
    xl = OUT / "module1_validation.xlsx"
    with pd.ExcelWriter(xl, engine="openpyxl") as w:
        verdict.to_excel(w, sheet_name="0_Verdict", index=False)
        samples.to_excel(w, sheet_name="1_Samples", index=False)
        perf.reset_index(names="model").to_excel(w, sheet_name="2_Performance", index=False)
        iv.rename_axis("feature").reset_index(name="iv").to_excel(w, sheet_name="3_IV", index=False)
        binner.bin_table().to_excel(w, sheet_name="4_WoE_bins", index=False)
        coef_tbl.reset_index(names="term").to_excel(w, sheet_name="5_Logit_coef", index=False)
        card.to_excel(w, sheet_name="6_Scorecard_points", index=False)
        for num, part, label in [(7, "gains", "Gains"), (8, "calib", "Calibration"), (9, "psi", "PSI")]:
            for k, v in res.items():
                tag = "LR" if "Logistic" in k else "XGB"
                v[part].to_excel(w, sheet_name=f"{num}_{label}_{tag}", index=False)
        csi.to_excel(w, sheet_name="10_CSI", index=False)
        xg_imp.to_excel(w, sheet_name="11_XGB_SHAP", index=False)
        pd.DataFrame({"step": log}).to_excel(w, sheet_name="12_Selection_log", index=False)
        for ws in w.book.worksheets:
            for row in ws.iter_rows():
                for cell in row:
                    cell.font = Font(name="Aptos Narrow", bold=cell.row == 1, color="000000")
            for cell in ws[1]:
                cell.fill = PatternFill("solid", fgColor="BDD7EE")
            for cc in ws.columns:
                ws.column_dimensions[cc[0].column_letter].width = min(max(len(str(x.value or "")) for x in cc) + 2, 50)
    xl2 = OUT / "module1_scored_accounts.xlsx"
    with pd.ExcelWriter(xl2, engine="openpyxl") as w:
        card_view.to_excel(w, sheet_name="Scorecard_points", index=False)
        acc.to_excel(w, sheet_name="Scored_accounts", index=False)
        notes = pd.DataFrame({"note": [
            f"Score scaling: {BASE_SCORE} points at good:bad odds {BASE_ODDS}:1, PDO {PDO}. Higher score = lower risk.",
            "total_score = sum of the points columns; model_score = exact score from the logistic PD (differs by rounding only).",
            "Intercept is spread evenly across characteristics, so there is no separate base-points line.",
            "score_decile: 1 = riskiest tenth of the training population, 10 = safest.",
            "Data: UCI Default of Credit Card Clients (CC BY 4.0), Yeh (2009), https://doi.org/10.24432/C55S3H"]})
        notes.to_excel(w, sheet_name="Notes", index=False)
        for ws in w.book.worksheets:
            for row in ws.iter_rows():
                for cell in row:
                    cell.font = Font(name="Aptos Narrow", bold=cell.row == 1, color="000000")
            for cell in ws[1]:
                cell.fill = PatternFill("solid", fgColor="BDD7EE")
            for cc in ws.columns:
                ws.column_dimensions[cc[0].column_letter].width = min(max(len(str(x.value or "")) for x in cc[:200]) + 2, 40)
        sa = w.book["Scored_accounts"]; sa.freeze_panes = "B2"; sa.auto_filter.ref = sa.dimensions
        hl = PatternFill("solid", fgColor="E2EFDA")                   # light green: points and total columns
        for cc in sa.iter_cols(min_row=1, max_row=1):
            if str(cc[0].value).endswith("points") or cc[0].value == "total_score":
                cc[0].fill = hl
    print(f"\nSaved {xl.relative_to(ROOT)}, {xl2.relative_to(ROOT)} and charts in {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="Path to the UCI zip/xls/csv (default: data/raw/default+of+credit+card+clients.zip)")
    main(ap.parse_args().data)
