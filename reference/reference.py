#!/usr/bin/env python3
"""
Slow, literal reference implementation of rules.md, used only to cross-check the
fast native pipeline. Written line by line from the rules, favouring clarity.

Usage: reference.py DATA_DIR OUT_DIR
Writes OUT_DIR/user_daily.csv and OUT_DIR/campaign_daily.csv in the fixtures'
expected-CSV format (day as YYYY-MM-DD, empty top_category for null).
"""
import glob
import json
import os
import sys

DAY = 86_400_000
GAP = 1_800_000
WEEK = 7 * DAY
LATE = 86_400_000


def days_from_civil(y, m, d):
    y = y - 1 if m <= 2 else y
    era = y // 400
    yoe = y - era * 400
    doy = (153 * ((m + 9) % 12) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def civil_from_days(z):
    z += 719468
    era = z // 146097
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    d = doy - (153 * mp + 2) // 5 + 1
    m = mp + 3 if mp < 10 else mp - 9
    return (y + (1 if m <= 2 else 0), m, d)


def is_leap(y):
    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)


DIM = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

DIGITS = set(b"0123456789")


def all_digits(b):
    return len(b) > 0 and all(c in DIGITS for c in b)


def parse_civil(s):
    """s: 19 bytes YYYY-MM-DDTHH:MM:SS -> ms or None"""
    if len(s) != 19:
        return None
    if s[4:5] != b"-" or s[7:8] != b"-" or s[10:11] != b"T" or s[13:14] != b":" or s[16:17] != b":":
        return None
    parts = [s[0:4], s[5:7], s[8:10], s[11:13], s[14:16], s[17:19]]
    if not all(all_digits(p) for p in parts):
        return None
    y, mo, d, h, mi, se = (int(p) for p in parts)
    if not (1 <= mo <= 12):
        return None
    dim = 29 if (mo == 2 and is_leap(y)) else DIM[mo - 1]
    if not (1 <= d <= dim):
        return None
    if h > 23 or mi > 59 or se > 59:
        return None
    return ((days_from_civil(y, mo, d) * 24 + h) * 60 + mi) * 60_000 + se * 1000


def parse_ts(s):
    """bytes or None -> integer ms since epoch (unbounded int for F3) or None"""
    if s is None or len(s) == 0:
        return None
    if all_digits(s):
        return int(s)  # F3; out-of-window values are filtered by the caller
    if len(s) == 20 and s[19:20] == b"Z":
        return parse_civil(s[:19])
    if len(s) == 25 and s[19:20] in (b"+", b"-") and s[22:23] == b":":
        base = parse_civil(s[:19])
        if base is None:
            return None
        hh, mm = s[20:22], s[23:25]
        if not (all_digits(hh) and all_digits(mm)):
            return None
        hh, mm = int(hh), int(mm)
        if hh > 23 or mm > 59:
            return None
        off = (hh * 60 + mm) * 60_000
        return base - off if s[19:20] == b"+" else base + off
    return None


def parse_fields(line, n):
    parts = line.split(b",")
    if len(parts) != n:
        return None
    out = []
    for p in parts:
        if b'"' in p:
            if len(p) >= 2 and p[:1] == b'"' and p[-1:] == b'"' and b'"' not in p[1:-1]:
                v = p[1:-1]
            else:
                return None
        else:
            v = p
        out.append(v if len(v) > 0 else None)
    return out


def data_lines(path):
    with open(path, "rb") as f:
        data = f.read()
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()  # the text after the final \n is not a line
    return lines[1:]  # the first line is the header


def csv_files(data, folder):
    return sorted(glob.glob(os.path.join(data, folder, "*.csv")))


def main():
    data, out = sys.argv[1], sys.argv[2]
    with open(os.path.join(data, "manifest.json")) as f:
        manifest = json.load(f)
    ws = parse_ts(manifest["window_start"].encode())
    we = parse_ts(manifest["window_end"].encode())
    assert ws is not None and we is not None

    products = {}
    for line in data_lines(os.path.join(data, "products.csv")):
        f = parse_fields(line, 2)
        assert f is not None, line
        products[f[0]] = f[1]
    campaigns = {}
    for line in data_lines(os.path.join(data, "campaigns.csv")):
        f = parse_fields(line, 2)
        assert f is not None, line
        campaigns[f[0]] = f[1]

    # ---- links: union-find over IDs
    parent = {}

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a, b):
        for x in (a, b):
            if x not in parent:
                parent[x] = x
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for path in csv_files(data, "links"):
        for line in data_lines(path):
            f = parse_fields(line, 2)
            if f is None or f[0] is None or f[1] is None:
                continue
            union(f[0], f[1])
    name_of_root = {}
    for x in parent:
        r = find(x)
        if r not in name_of_root or x < name_of_root[r]:
            name_of_root[r] = x

    def person(uid):
        if uid in parent:
            return name_of_root[find(uid)]
        return uid

    # ---- events: parse, dedup on event_id (smallest ingest_ts), validate, drop late
    kept = {}  # event_id -> (ingest_ms, fields)
    bad_ingest = 0
    for path in csv_files(data, "events"):
        for line in data_lines(path):
            f = parse_fields(line, 7)
            if f is None or f[0] is None:
                continue
            ingest = parse_ts(f[3])
            if ingest is None:
                bad_ingest += 1
                continue
            prev = kept.get(f[0])
            if prev is None or ingest < prev[0]:
                kept[f[0]] = (ingest, f)
            elif ingest == prev[0] and f != prev[1]:
                print("WARNING: equal-ingest copies differ", f, prev[1], file=sys.stderr)
    if bad_ingest:
        print(f"WARNING: {bad_ingest} lines with unparseable ingest_ts", file=sys.stderr)

    by_person = {}  # person -> list of (ts, price_or_0, category)
    for eid, (ingest, f) in kept.items():
        _, uid, ts_s, _, etype, pid, price_s = f
        if uid is None:
            continue
        ts = parse_ts(ts_s)
        if ts is None or not (ws <= ts < we):
            continue
        if etype not in (b"view", b"cart", b"purchase"):
            continue
        if pid is None:
            continue
        price = 0
        if etype == b"purchase":
            if price_s is None or not all_digits(price_s):
                continue
            price = int(price_s)
            if not (1 <= price <= 10_000_000):
                continue
        if ingest - ts > LATE:
            continue
        cat = products.get(pid, b"unknown")
        by_person.setdefault(person(uid), []).append((ts, etype == b"purchase", price, cat))

    # ---- clicks: parse, dedup, validate
    clicks_seen = {}
    differing_click_copies = 0
    for path in csv_files(data, "clicks"):
        for line in data_lines(path):
            f = parse_fields(line, 4)
            if f is None or f[0] is None:
                continue
            if f[0] in clicks_seen:
                if clicks_seen[f[0]] != f:
                    differing_click_copies += 1
                continue
            clicks_seen[f[0]] = f
    if differing_click_copies:
        print(f"WARNING: {differing_click_copies} click copies differ", file=sys.stderr)
    clicks_by_person = {}
    for cid, f in clicks_seen.items():
        _, uid, camp, ts_s = f
        if uid is None:
            continue
        ts = parse_ts(ts_s)
        if ts is None or not (ws - WEEK <= ts < we):
            continue
        if camp not in campaigns:
            continue
        clicks_by_person.setdefault(person(uid), []).append((ts, cid, camp))

    # ---- sessions, user_daily, attribution, campaign_daily
    user_daily = {}  # (person, day) -> [sessions, events, revenue, {cat: sum}]
    camp_daily = {}  # (campaign, day) -> [sessions, purchasing, revenue, assisted]
    for p, evs in by_person.items():
        evs.sort(key=lambda e: e[0])
        sessions = []
        prev_ts = None
        for e in evs:
            if prev_ts is None or e[0] - prev_ts > GAP:
                sessions.append([e[0], []])
            sessions[-1][1].append(e)
            prev_ts = e[0]
        pclicks = clicks_by_person.get(p, [])
        for start, sev in sessions:
            day = start // DAY
            ud = user_daily.setdefault((p, day), [0, 0, 0, {}])
            ud[0] += 1
            ud[1] += len(sev)
            purchases = [e for e in sev if e[1]]
            rev = sum(e[2] for e in purchases)
            ud[2] += rev
            for e in purchases:
                ud[3][e[3]] = ud[3].get(e[3], 0) + e[2]
            # attribution
            q = [c for c in pclicks if c[0] <= start and start - c[0] <= WEEK]
            if q:
                best_ts = max(c[0] for c in q)
                best = min(c[1] for c in q if c[0] == best_ts)
                credit = [c for c in q if c[0] == best_ts and c[1] == best][0][2]
            else:
                credit = b"organic"
            cd = camp_daily.setdefault((credit, day), [0, 0, 0, 0])
            cd[0] += 1
            cd[1] += 1 if purchases else 0
            cd[2] += rev
            for camp in set(c[2] for c in q):
                cd2 = camp_daily.setdefault((camp, day), [0, 0, 0, 0])
                cd2[3] += 1

    os.makedirs(out, exist_ok=True)

    def day_str(d):
        y, m, dd = civil_from_days(d)
        return f"{y:04d}-{m:02d}-{dd:02d}".encode()

    with open(os.path.join(out, "user_daily.csv"), "wb") as f:
        f.write(b"user_id,day,sessions,events,revenue_cents,top_category\n")
        for (p, day), (s, n, rev, cats) in sorted(user_daily.items()):
            top = b""
            if cats:
                best = max(cats.values())
                top = min(c for c, v in cats.items() if v == best)
            f.write(b",".join([p, day_str(day), b"%d" % s, b"%d" % n, b"%d" % rev, top]) + b"\n")
    with open(os.path.join(out, "campaign_daily.csv"), "wb") as f:
        f.write(b"campaign_id,channel,day,sessions,purchasing_sessions,revenue_cents,assisted_sessions\n")
        for (c, day), (s, ps, rev, a) in sorted(camp_daily.items()):
            ch = b"organic" if c == b"organic" else campaigns[c]
            f.write(b",".join([c, ch, day_str(day), b"%d" % s, b"%d" % ps, b"%d" % rev, b"%d" % a]) + b"\n")


if __name__ == "__main__":
    main()
