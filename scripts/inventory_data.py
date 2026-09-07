"""Phase 0.1 - load every raw data file, print its shape/dtypes/head, and dump
an auto-generated inventory to notes/data_inventory.md.

Usage:
    .venv/bin/python scripts/inventory_data.py
    .venv/bin/python scripts/inventory_data.py --data-dir data/raw --rows 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TABLE_SUFFIXES = {".csv", ".parquet"}
MEDIA_SUFFIXES = {".mp4", ".jpg", ".jpeg", ".png", ".avi", ".mov"}
# Columns with at most this many distinct values get their full value list printed.
CATEGORICAL_MAX_UNIQUE = 25


def human_bytes(n: int) -> str:
    step = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if step < 1024 or unit == "TB":
            return f"{step:,.1f} {unit}"
        step /= 1024
    return f"{step:,.1f} TB"


def load(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: dtype, nulls, cardinality, and a value summary."""
    rows = []
    for col in df.columns:
        s = df[col]
        n_unique = s.nunique(dropna=True)
        if pd.api.types.is_numeric_dtype(s) and n_unique > CATEGORICAL_MAX_UNIQUE:
            summary = f"min={s.min():g}, max={s.max():g}, mean={s.mean():g}"
        elif n_unique <= CATEGORICAL_MAX_UNIQUE:
            summary = ", ".join(repr(v) for v in sorted(s.dropna().unique().tolist(), key=str))
        else:
            summary = ", ".join(repr(v) for v in s.dropna().unique()[:3].tolist()) + ", ..."
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "non_null": int(s.notna().sum()),
                "null_pct": round(100 * s.isna().mean(), 2),
                "n_unique": int(n_unique),
                "values": summary[:300],
            }
        )
    return pd.DataFrame(rows)


def inventory_media(data_dir: Path) -> pd.DataFrame:
    """Group non-tabular files (video frames, clips) by folder and extension."""
    buckets: dict[tuple[str, str], list[int]] = {}
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in MEDIA_SUFFIXES:
            key = (str(path.parent.relative_to(data_dir)), path.suffix.lower())
            buckets.setdefault(key, []).append(path.stat().st_size)
    rows = [
        {
            "folder": folder,
            "ext": ext,
            "n_files": len(sizes),
            "total_size": human_bytes(sum(sizes)),
            "mean_size": human_bytes(sum(sizes) // len(sizes)),
        }
        for (folder, ext), sizes in sorted(buckets.items())
    ]
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("notes/data_inventory.md"))
    ap.add_argument("--rows", type=int, default=5, help="head() rows to show")
    args = ap.parse_args()

    if not args.data_dir.exists():
        raise SystemExit(f"No such directory: {args.data_dir} - download the data first.")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 100)

    tables = sorted(
        p for p in args.data_dir.rglob("*") if p.suffix.lower() in TABLE_SUFFIXES
    )
    if not tables:
        raise SystemExit(f"No .csv/.parquet files under {args.data_dir}")

    md: list[str] = [
        "# Data inventory (auto-generated)",
        "",
        f"Source: `{args.data_dir}` - regenerate with `.venv/bin/python scripts/inventory_data.py`.",
        "",
    ]

    for path in tables:
        rel = path.relative_to(args.data_dir)
        df = load(path)
        mem = human_bytes(int(df.memory_usage(deep=True).sum()))
        header = f"{rel}  |  {df.shape[0]:,} rows x {df.shape[1]} cols  |  {human_bytes(path.stat().st_size)} on disk, {mem} in memory"

        print("\n" + "=" * len(header))
        print(header)
        print("=" * len(header))
        print("\n--- dtypes ---")
        print(df.dtypes.to_string())
        print(f"\n--- head({args.rows}) ---")
        print(df.head(args.rows).to_string())

        cols = profile_columns(df)
        print("\n--- columns ---")
        print(cols.to_string(index=False))

        md += [
            f"## `{rel}`",
            "",
            f"- **Shape:** {df.shape[0]:,} rows x {df.shape[1]} columns",
            f"- **Size:** {human_bytes(path.stat().st_size)} on disk, {mem} in memory",
            "",
            cols.to_markdown(index=False),
            "",
            f"<details><summary>head({args.rows})</summary>",
            "",
            "```",
            df.head(args.rows).to_string(),
            "```",
            "",
            "</details>",
            "",
        ]

    media = inventory_media(args.data_dir)
    if not media.empty:
        print("\n--- media files ---")
        print(media.to_string(index=False))
        md += ["## Media files", "", media.to_markdown(index=False), ""]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
