"""Load and clean source datasets. Raw files live in data/raw/ (git-ignored)."""
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

TAIWAN_ZIP = "default+of+credit+card+clients.zip"
TAIWAN_XLS = "default of credit card clients.xls"

# UCI column names -> clearer names. PAY_0 is September 2005 (the latest month).
TAIWAN_RENAME = {
    "LIMIT_BAL": "limit_bal", "SEX": "sex", "EDUCATION": "education", "MARRIAGE": "marriage", "AGE": "age",
    "PAY_0": "dlq_m1", "PAY_2": "dlq_m2", "PAY_3": "dlq_m3", "PAY_4": "dlq_m4", "PAY_5": "dlq_m5", "PAY_6": "dlq_m6",
    **{f"BILL_AMT{i}": f"bill_m{i}" for i in range(1, 7)},
    **{f"PAY_AMT{i}": f"paid_m{i}" for i in range(1, 7)},
    "default payment next month": "default_flag",
}


def _read_taiwan_raw(path: Path) -> pd.DataFrame:
    """Read the UCI zip (xls inside), a bare .xls, or a .csv with the UCI header."""
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.lower().endswith((".xls", ".xlsx", ".csv")))
            with z.open(name) as f:
                if name.lower().endswith(".csv"):
                    return pd.read_csv(f)
                return pd.read_excel(f, header=1)      # row 0 is X1..X23, row 1 holds the names
    if path.suffix in (".xls", ".xlsx"):
        return pd.read_excel(path, header=1)
    return pd.read_csv(path)


def load_taiwan(path: str | Path | None = None) -> pd.DataFrame:
    """Return the cleaned Taiwan credit card dataset (30,000 rows)."""
    path = Path(path) if path else RAW / TAIWAN_ZIP
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download it from "
            "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients "
            "and save it in data/raw/ (see DATA_SOURCES.md).")
    df = _read_taiwan_raw(path)
    df = df.rename(columns={"default.payment.next.month": "default payment next month",
                            "PAY_1": "PAY_0"})       # Kaggle-style CSV variants
    df = df.rename(columns=TAIWAN_RENAME)
    df = df.drop(columns=[c for c in ("Unnamed: 0",) if c in df.columns])
    df = df.rename(columns={"ID": "app_id"})                     # keep the UCI client ID for account-level output

    # Undocumented codes -> documented "other" groups
    df["education"] = df["education"].where(df["education"].isin([1, 2, 3, 4]), 4)
    df["marriage"] = df["marriage"].where(df["marriage"].isin([1, 2, 3]), 3)
    df["education"] = df["education"].map({1: "graduate", 2: "university", 3: "high_school", 4: "other"})
    df["marriage"] = df["marriage"].map({1: "married", 2: "single", 3: "other"})

    expected = set(TAIWAN_RENAME.values())
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected file layout, missing columns: {sorted(missing)}")
    if len(df) != 30000:
        print(f"WARNING: expected 30,000 rows, found {len(df):,}")
    return df.reset_index(drop=True)
