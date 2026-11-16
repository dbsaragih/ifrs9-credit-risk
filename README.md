# IFRS 9 Credit Risk Modelling: Scorecard, PD, LGD, EAD and ECL

An end-to-end, reproducible credit risk project built on openly licensed data. It follows the
workflow used in banks for IFRS 9 (PSAK 109 in Indonesia): develop a scorecard, estimate the risk
parameters, validate them, and combine them into expected credit loss.

| Module | Topic | Data | Status |
|---|---|---|---|
| 1 | PD scorecard: WoE logistic regression (champion) vs XGBoost (challenger), full validation | UCI Taiwan credit card (CC BY 4.0) | Done |
| 2a | Workout LGD: recovery curves, cure-adjusted LGD, time-in-default LGD, LTV and vintage effects | Synthetic defaults and cash flows (MIT) | Done |
| 2b | EAD and CCF for revolving credit: count- vs exposure-weighted CCF, hybrid method | UCI Taiwan credit card (CC BY 4.0) | Done |
| 3 | Lifetime PD: migration matrix, forward-looking macro adjustment, staging | Synthetic panel (MIT) | Planned |
| 4 | ECL: 12-month and lifetime, scenario-weighted | Outputs of modules 1–3 | Planned |
| 5 | Small-SME PD and charge-off loss | SBA 7(a) loans (US public domain) | Planned |

## Quick start

```bash
git clone <your-repo-url> && cd ifrs9-credit-risk
python -m venv .venv && .venv\Scripts\activate      # Windows  (source .venv/bin/activate on Mac/Linux)
pip install -r requirements.txt
python scripts/00_download_data.py                  # downloads Taiwan + SBA 7(a) data into data/raw/
python scripts/01_taiwan_scorecard.py               # module 1
python scripts/02a_lgd_workout.py                   # module 2a (generates its own synthetic data)
python scripts/02b_taiwan_ead_ccf.py                # module 2b
```

Each module writes a workbook and charts to `outputs/<module>/`.

## Module 1: PD scorecard

**Data preparation**
- 30,000 credit card accounts; target = default in the month after the observation window
- Undocumented category codes regrouped; `SEX` excluded from modelling (protected characteristic)
- 16 behavioural features from 6 months of history: delinquency status, months delinquent,
  utilisation level and trend, payment-to-bill ratios, months with no balance

**Champion: WoE logistic scorecard**
- Monotonic WoE binning (each bin ≥ 5% of accounts, bad rate monotonic across bins)
- Selection: IV ≥ 0.02 → correlation filter (|r| > 0.7) → backward elimination on sign and p-value
- Scaling: 600 points at good:bad odds of 50:1, PDO = 20; points table with automatic
  reconciliation to the model score

**Challenger: XGBoost** with monotonic constraints and early stopping; SHAP importance

**Validation**

| Area | Tests |
|---|---|
| Discrimination | Gini with bootstrap 95% CI, KS, gains and lift; paired bootstrap of challenger uplift |
| Calibration | Average PD vs observed default rate, per-decile Jeffreys test, Hosmer-Lemeshow |
| Stability | Score PSI, feature CSI |
| Model hygiene | Coefficient signs, p-values, VIF, rank ordering |

**Limitation:** the Taiwan data covers a single observation date, so the sample split is random,
not out-of-time. PSI therefore checks sampling stability, not population drift over time.
Module 3 covers time-based validation with a panel dataset.

## Module 2a: Workout LGD (synthetic data)

Openly licensed loan-level recovery data hardly exists, so this module simulates an Indonesian-style
defaulted portfolio (KTA unsecured, KKB vehicle, SME property-secured) with monthly post-default cash
flows: cures, collections, collateral auctions, debt sales and recovery costs. Cash flows after the
reporting date are hidden, as they would be in a real bank, but kept in an answer key so each
estimator can be checked against the true LGD.

- **Workout LGD** = 1 − PV(recoveries − costs) / EAD, discounted at the EIR to the default date
- **Methods compared:** closed cases only (biased), chain-ladder recovery curve, cure-adjusted LGD
- **Stage 3:** time-in-default LGD; LGD rises for unsecured and vehicle loans the longer a case stays unresolved
- **Drivers:** LGD by LTV band and by default year (COVID cohort), with cohort-maturity flags

## Module 2b: EAD and CCF (Taiwan data)

Fixed-horizon approach: drawn balance and undrawn limit in April 2005, exposure in September 2005
for accounts that default in October 2005.

- CCF = (EAD − drawn) / (limit − drawn); compared with limit-factor and balance-factor methods
- **Exposure-weighted** CCF per utilisation × delinquency cell removes the bias of averaging ratios
- **Hybrid** method uses a balance factor for accounts at ≥ 90% utilisation, where the CCF is unstable
- Validated on held-out defaulters: total-EAD bias and bias by utilisation band

## Repository layout

```
├── data/raw/            downloaded data (git-ignored)
├── src/
│   ├── data_loader.py   loading and cleaning
│   ├── woe.py           WoE binning and IV
│   ├── validation.py    Gini, KS, PSI/CSI, calibration tests
│   └── synthetic.py     synthetic data generators (MIT)
├── scripts/
│   ├── 00_download_data.py
│   ├── 01_taiwan_scorecard.py
│   ├── 02a_lgd_workout.py
│   └── 02b_taiwan_ead_ccf.py
├── tests/make_fixture.py   synthetic file in the UCI layout, for code testing only
├── outputs/
├── DATA_SOURCES.md      licences and citations
└── LICENSE              MIT (code only)
```

## Data and licences

No data is stored in this repository. All datasets are downloaded from their original source
and used under their licences; see [DATA_SOURCES.md](DATA_SOURCES.md). If you use the Taiwan
data, cite:

> Yeh, I. (2009). Default of Credit Card Clients [Dataset]. UCI Machine Learning Repository.
> https://doi.org/10.24432/C55S3H

## Author

Daniel B. Saragih: credit risk practitioner (20+ years in Indonesian banking and fintech).
