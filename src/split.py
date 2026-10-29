"""Rebuild the train/val/test CSV files from ``fer2013.csv``.

Two modes:

``legacy``
    What the 2025 coursework did: a random 90/10 split, then a second 90/10
    split of the remainder, both with ``random_state=42`` and no stratification.
    This yields 29,069 / 3,231 / 3,590 rows. It is kept so the published
    numbers can be reproduced, but it is **not** a fair held-out evaluation:
    FER2013 rows are not i.i.d. and the official test set is unused.

``official``
    Use the ``Usage`` column Fer2013 ships with: Training / PublicTest /
    PrivateTest. Only ``Training`` may be used to fit anything; a validation
    slice is carved out of it (the official test rows stay untouched).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "fer2013"


def legacy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(df, test_size=0.1, random_state=42)
    train, val = train_test_split(train, test_size=0.1, random_state=42)
    return train, val, test


def official(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_pool = df[df["Usage"] == "Training"]
    test = df[df["Usage"].isin(["PublicTest", "PrivateTest"])]
    train, val = train_test_split(train_pool, test_size=0.1, random_state=42)
    return train, val, test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["legacy", "official"], default="legacy")
    parser.add_argument("--out", type=Path, default=DATA)
    args = parser.parse_args()

    df = pd.read_csv(DATA / "fer2013.csv")
    print(f"fer2013.csv: {len(df)} rows, usage={df['Usage'].value_counts().to_dict()}")

    train, val, test = (legacy if args.mode == "legacy" else official)(df)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("val", val), ("test", test)):
        part.to_csv(args.out / f"{name}.csv", index=False)
        counts = part["emotion"].value_counts().sort_index().tolist()
        print(f"{name:5s} {len(part):6d}  per-class {counts}")


if __name__ == "__main__":
    main()
