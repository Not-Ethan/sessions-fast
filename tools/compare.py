#!/usr/bin/env python3
"""
Exact, order-insensitive comparison of two result folders.

Each folder holds either TABLE.csv (fixture/reference format) or TABLE/*.parquet
(pipeline output), for TABLE in user_daily, campaign_daily. Parquet schemas are
checked the way the starter's checker does (exact column set, integer columns,
string columns, DATE day).

Usage: compare.py A B            -> exit 0 on MATCH
"""
import sys
from pathlib import Path

import polars as pl

SCHEMAS = {
    "user_daily": {"user_id": pl.String, "day": pl.Date, "sessions": pl.Int128, "events": pl.Int128,
                   "revenue_cents": pl.Int128, "top_category": pl.String},
    "campaign_daily": {"campaign_id": pl.String, "channel": pl.String, "day": pl.Date, "sessions": pl.Int128,
                       "purchasing_sessions": pl.Int128, "revenue_cents": pl.Int128, "assisted_sessions": pl.Int128},
}
KEYS = {"user_daily": ["user_id", "day"], "campaign_daily": ["campaign_id", "day"]}


def load_csv(path, schema):
    df = pl.read_csv(path, schema_overrides={k: (pl.String if v == pl.Date else v) for k, v in schema.items()})
    if df.height == 0:
        return pl.DataFrame(schema=schema)
    return df.with_columns(pl.col("day").str.to_date())


def load_parquet_dir(folder, schema):
    folder = Path(folder)
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError(f"expected an output directory: {folder}")
    frames = []
    for path in sorted(folder.glob("*.parquet")):
        if path.name.startswith((".", "_")) or path.is_symlink() or not path.is_file():
            continue
        frame = pl.read_parquet(path)
        if set(frame.columns) != set(schema) or len(frame.columns) != len(schema):
            raise ValueError(f"{path}: expected {list(schema)}, got {frame.columns}")
        for column, expected in schema.items():
            actual = frame.schema[column]
            if expected == pl.Int128 and actual.is_integer():
                continue
            if expected == pl.String and actual.base_type() in (pl.String, pl.Categorical, pl.Enum):
                continue
            if actual != expected:
                raise ValueError(f"{path}: {column} has type {actual}, expected {expected}")
        frames.append(frame.select(schema.keys()).cast(schema))
    if not frames:
        raise ValueError(f"no parquet in {folder}")
    return pl.concat(frames)


def load(folder, table):
    folder = Path(folder)
    csv = folder / f"{table}.csv"
    if csv.is_file():
        return load_csv(csv, SCHEMAS[table])
    return load_parquet_dir(folder / table, SCHEMAS[table])


def rows(df):
    return sorted(df.rows(), key=lambda r: tuple("" if v is None else str(v) for v in r))


def compare(a_dir, b_dir, verbose=True):
    matched = True
    for table, schema in SCHEMAS.items():
        try:
            a = load(a_dir, table)
            b = load(b_dir, table)
        except Exception as error:  # noqa: BLE001
            print(f"{table}: ERROR {error}")
            matched = False
            continue
        ra, rb = rows(a), rows(b)
        keys = KEYS[table]
        dup_a = a.height - a.n_unique(subset=keys)
        dup_b = b.height - b.n_unique(subset=keys)
        if ra == rb and dup_a == 0 and dup_b == 0:
            if verbose:
                print(f"{table}: {a.height} rows identical")
            continue
        matched = False
        sa, sb = set(ra), set(rb)
        only_a = sorted(sa - sb, key=str)
        only_b = sorted(sb - sa, key=str)
        print(f"{table}: rows {a.height} vs {b.height}; only in A {len(only_a)}; only in B {len(only_b)}; "
              f"duplicate keys {dup_a} vs {dup_b}")
        for r in only_a[:8]:
            print("  A:", r)
        for r in only_b[:8]:
            print("  B:", r)
    return matched


if __name__ == "__main__":
    ok = compare(sys.argv[1], sys.argv[2])
    print("MATCH" if ok else "DIFFER")
    sys.exit(0 if ok else 1)
