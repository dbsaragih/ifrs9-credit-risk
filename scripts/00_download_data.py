"""
Download / check the public source datasets into data/raw/ (git-ignored). See DATA_SOURCES.md.

    python scripts/00_download_data.py            # Taiwan + SBA 7(a) FY2010-FY2019
    python scripts/00_download_data.py --sba-all  # all SBA 7(a) files (large, ~1 GB)

Private-track datasets (LendingClub, Freddie Mac, Home Credit) need accounts and are downloaded
manually into private/data/ - instructions are printed at the end.
"""
from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
TAIWAN_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
TAIWAN_FILE = RAW / "default+of+credit+card+clients.zip"
SBA_API = "https://data.sba.gov/api/3/action/package_show?id=7-a-504-foia"   # CKAN catalogue API


def get_taiwan() -> None:
    if TAIWAN_FILE.exists():
        print(f"[ok] Taiwan file present: {TAIWAN_FILE.name}")
    else:
        print("Downloading Taiwan dataset from UCI ...")
        r = requests.get(TAIWAN_URL, timeout=120); r.raise_for_status()
        TAIWAN_FILE.write_bytes(r.content)
        print(f"[ok] Saved {TAIWAN_FILE.name} ({len(r.content) / 1e6:.1f} MB)")
    with zipfile.ZipFile(TAIWAN_FILE) as z:
        print(f"     contents: {z.namelist()}")
    sys.path.insert(0, str(ROOT))
    from src.data_loader import load_taiwan
    df = load_taiwan(TAIWAN_FILE)
    print(f"     {len(df):,} rows, default rate {df.default_flag.mean():.2%} (UCI: 30,000 rows, 22.12%)")


def get_sba(all_files: bool) -> None:
    """Find the current 7(a) CSV files through the CKAN API (file names change with each refresh)."""
    try:
        meta = requests.get(SBA_API, timeout=60).json()["result"]["resources"]
    except Exception as e:                                         # noqa: BLE001
        print(f"[action] Could not reach the SBA catalogue ({e}).")
        print("         Download the 7(a) CSV files manually from https://data.sba.gov/dataset/7-a-504-foia")
        print(f"         and save them in {RAW} as sba_7a_<period>.csv")
        return
    files = [r for r in meta if r.get("format", "").upper() == "CSV" and re.search(r"7\s*\(a\)", r.get("name", ""))]
    if not all_files:
        files = [r for r in files if "2010" in r["name"]]           # FY2010-FY2019: a full decade incl. post-crisis
    if not files:
        print("[action] No 7(a) CSV resources found; download manually from https://data.sba.gov/dataset/7-a-504-foia")
        return
    for r in files:
        period = re.search(r"FY\s*(\d{4})\s*-\s*(FY\s*\d{4}|Present)", r["name"].replace("(", " ").replace(")", " "))
        tag = f"{period.group(1)}_{period.group(2).replace('FY', '').strip().lower()}" if period else re.sub(r"\W+", "_", r["name"])[:40]
        out = RAW / f"sba_7a_{tag}.csv"
        if out.exists():
            print(f"[ok] {out.name} present"); continue
        print(f"Downloading {r['name']} ...")
        with requests.get(r["url"], stream=True, timeout=600) as resp:
            resp.raise_for_status()
            with open(out, "wb") as f:
                for chunk in resp.iter_content(1 << 20):
                    f.write(chunk)
        print(f"[ok] Saved {out.name} ({out.stat().st_size / 1e6:.0f} MB)")


def private_instructions() -> None:
    print("\nPrivate track (manual download into private/data/, never committed):")
    print("  - LendingClub 2007-2018: Kaggle dataset 'wordsforthewise/lending-club' (accepted_2007_to_2018Q4.csv.gz)")
    print("  - Freddie Mac SF Loan-Level: register at https://freddiemac.embs.com/FLoan/ and download the sample files")
    print("  - Home Credit (optional): Kaggle competition 'home-credit-default-risk' (accept the rules first)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sba-all", action="store_true", help="download every SBA 7(a) file, not just FY2010-FY2019")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    get_taiwan()
    get_sba(args.sba_all)
    private_instructions()
    print("\nCite every dataset you use - see DATA_SOURCES.md.")
