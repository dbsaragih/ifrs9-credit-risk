"""Synthetic datasets (MIT licence). Simulated data - not real borrowers.

generate_lgd_data(): defaulted loans and their post-default cash flows, Indonesian-style portfolio
  segments: KTA (unsecured personal loan), KKB (vehicle loan), SME (property-secured small business)
  amounts in IDR million; reporting date 2025-12-31; cash flows after that date are hidden
  (open workouts) but kept in an answer-key file so estimators can be checked against the truth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REPORTING_DATE = pd.Timestamp("2025-12-31")

SEG = {  # share, EAD lognormal (median IDR mn, sigma), EIR range, base cure probability
    "KTA": dict(share=.50, ead=(40, .6), eir=(.18, .26), cure=.25),
    "KKB": dict(share=.35, ead=(150, .45), eir=(.10, .15), cure=.30),
    "SME": dict(share=.15, ead=(800, .7), eir=(.11, .15), cure=.20),
}


def _macro(default_month: pd.Timestamp) -> float:
    """Recovery environment factor: 1 = normal; COVID period depresses recoveries and collateral prices."""
    if pd.Timestamp("2020-03-01") <= default_month <= pd.Timestamp("2021-06-30"):
        return 0.85
    return 1.0


def generate_lgd_data(n: int = 12000, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (loans, observed_cashflows, answer_key)."""
    rng = np.random.default_rng(seed)
    months = pd.date_range("2018-01-01", "2025-06-01", freq="MS")
    seg = rng.choice(list(SEG), n, p=[v["share"] for v in SEG.values()])
    loans, flows = [], []
    for i in range(n):
        s, cfg = seg[i], SEG[seg[i]]
        d0 = months[rng.integers(len(months))]
        mf = _macro(d0)
        ead = round(float(np.exp(rng.normal(np.log(cfg["ead"][0]), cfg["ead"][1]))), 2)
        eir = round(float(rng.uniform(*cfg["eir"])), 4)
        dpd_at_default = int(rng.choice([90, 120, 150, 180], p=[.6, .2, .1, .1]))
        coll_value, ltv = np.nan, np.nan
        if s == "KKB":
            ltv = float(np.clip(rng.normal(0.85, 0.18), 0.4, 1.4)); coll_value = ead / ltv
        elif s == "SME":
            ltv = float(np.clip(rng.normal(0.65, 0.18), 0.25, 1.2)); coll_value = ead / ltv
        p_cure = cfg["cure"] * (1.25 if dpd_at_default == 90 else 0.6) * (0.75 if mf < 1 else 1)
        cf = []                                                     # (month_since_default, amount, type)
        if rng.random() < p_cure:                                   # ---- cure: back to performing
            m_cure = int(rng.integers(1, 10))
            cf.append((m_cure, ead, "cure_repayment"))              # NPV of resumed schedule ~ EAD at EIR
            cf.append((m_cure, -ead * rng.uniform(.005, .02), "cost"))
            outcome = "cured"
        else:
            outcome = "workout"
            if s == "KTA":
                rr = rng.beta(2, 4.5) * mf                          # mean ~31% nominal
                horizon = int(rng.integers(18, 49))
                w = np.exp(-np.arange(1, horizon + 1) / rng.uniform(6, 14)); w /= w.sum()
                for m, a in enumerate(rr * ead * w, start=1):
                    if a > ead * 0.002:
                        cf.append((m, a, "collection")); cf.append((m, -a * .12, "cost"))   # agency fee 12%
                if rng.random() < .25:                              # debt sale closes the case
                    ms = int(rng.integers(18, 31))
                    cf = [c for c in cf if c[0] < ms]
                    cf.append((ms, ead * rng.uniform(.04, .10), "debt_sale"))
            elif s == "KKB":
                if rng.random() < .72 * mf:                         # vehicle repossessed and auctioned
                    ms = int(rng.integers(2, 10))
                    age_dep = 0.985 ** ms
                    price = coll_value * age_dep * mf * rng.uniform(.62, .82)              # forced-sale haircut
                    cf.append((ms, min(price, ead * 1.02), "collateral_sale"))
                    cf.append((ms, -coll_value * rng.uniform(.03, .07), "cost"))           # repo, storage, auction
                    if price < ead and rng.random() < .3:
                        for m in range(ms + 1, ms + 13):
                            cf.append((m, (ead - price) * .01, "collection"))
                else:                                               # vehicle not recovered: unsecured collection
                    rr = rng.beta(1.5, 6) * mf
                    for m in range(1, 25):
                        cf.append((m, rr * ead / 24, "collection")); cf.append((m, -rr * ead / 24 * .12, "cost"))
            else:                                                   # SME: Hak Tanggungan auction, slow
                ms = int(rng.integers(12, 43))
                for m in range(1, min(ms, 12)):
                    cf.append((m, ead * rng.uniform(0, .006), "collection"))
                if rng.random() < .85:
                    price = coll_value * mf * rng.uniform(.55, .80)
                    cf.append((ms, min(price, ead * 1.05), "collateral_sale"))
                    cf.append((ms, -min(price, ead) * rng.uniform(.05, .10), "cost"))  # legal, auction fees
                else:
                    cf.append((ms, ead * rng.uniform(.0, .1), "collection"))
        close_m = max(c[0] for c in cf) if cf else 1
        disc = lambda m: (1 + eir) ** (-m / 12)
        true_lgd = 1 - sum(a * disc(m) for m, a, _ in cf) / ead
        close_date = d0 + pd.DateOffset(months=close_m)
        loans.append(dict(loan_id=f"L{i:05d}", segment=s, default_date=d0, ead=ead, eir=eir, dpd_at_default=dpd_at_default,
                          collateral_value=round(coll_value, 2) if coll_value == coll_value else np.nan,
                          ltv_at_default=round(ltv, 3) if ltv == ltv else np.nan,
                          status="closed" if close_date <= REPORTING_DATE else "open",
                          cured=int(outcome == "cured" and close_date <= REPORTING_DATE),   # cure observed by reporting date
                          close_date=close_date if close_date <= REPORTING_DATE else pd.NaT,
                          _true_lgd=true_lgd, _true_outcome=outcome))
        for m, a, t in cf:
            flows.append(dict(loan_id=f"L{i:05d}", month_since_default=m, date=d0 + pd.DateOffset(months=m), amount=round(a, 4), type=t))
    loans = pd.DataFrame(loans); flows = pd.DataFrame(flows)
    key = loans[["loan_id", "_true_lgd", "_true_outcome"]].rename(columns=lambda c: c.lstrip("_"))
    loans = loans.drop(columns=["_true_lgd", "_true_outcome"])
    observed = flows[flows.date <= REPORTING_DATE].reset_index(drop=True)
    return loans, observed, key
