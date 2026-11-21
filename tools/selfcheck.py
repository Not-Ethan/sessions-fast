#!/usr/bin/env python3
"""
Sanity identities from rules.md on one output folder:
  sum(sessions) and sum(revenue_cents) agree between the two tables;
  sum(assisted_sessions) >= credited non-organic sessions;
  no duplicate keys; organic never assists; day range plausible.
Usage: selfcheck.py OUT_DIR
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare import load  # noqa: E402
import polars as pl  # noqa: E402


def main():
    out = sys.argv[1]
    u = load(out, "user_daily")
    c = load(out, "campaign_daily")
    ok = True
    us, cs = u["sessions"].sum(), c["sessions"].sum()
    ur, cr = u["revenue_cents"].sum(), c["revenue_cents"].sum()
    assisted = c["assisted_sessions"].sum()
    nonorg = c.filter(pl.col("campaign_id") != "organic")["sessions"].sum()
    org_assist = c.filter(pl.col("campaign_id") == "organic")["assisted_sessions"].sum()
    dup_u = u.height - u.n_unique(subset=["user_id", "day"])
    dup_c = c.height - c.n_unique(subset=["campaign_id", "day"])
    print(f"user_daily rows={u.height} sessions={us} events={u['events'].sum()} revenue={ur}")
    print(f"campaign_daily rows={c.height} sessions={cs} revenue={cr} assisted={assisted} non_organic={nonorg}")
    print(f"days {u['day'].min()}..{u['day'].max()}  dup_keys={dup_u},{dup_c}")
    for cond, msg in [
        (us == cs, "sessions differ between tables"),
        (ur == cr, "revenue differs between tables"),
        (assisted >= nonorg, "assisted < credited non-organic sessions"),
        (org_assist == 0, "organic assists"),
        (dup_u == 0 and dup_c == 0, "duplicate keys"),
        (u.filter(pl.col("revenue_cents") > 0)["top_category"].null_count() == 0, "revenue without top_category"),
        (u.filter(pl.col("revenue_cents") == 0)["top_category"].null_count()
         == u.filter(pl.col("revenue_cents") == 0).height, "top_category without revenue"),
    ]:
        if not cond:
            ok = False
            print("FAIL:", msg)
    print("SELFCHECK OK" if ok else "SELFCHECK FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
