#!/usr/bin/env python3
"""
Shared helpers for the differential / adversarial harness (v5_adv.py): CSV
text editing of generated datasets (split / join with random quoting), legal
mutations of IDs, duplicates, links, line endings and files, two datasets
concatenated, and the output comparisons (tools/compare.py, reference.py).
IDs are treated as opaque strings throughout.
"""
import argparse
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import zlib
from collections import defaultdict
from contextlib import redirect_stdout
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
HERE = Path(__file__).resolve().parent
SS = HERE.parent.parent  # shopping-sessions
from compare import KEYS, SCHEMAS, compare, load  # noqa: E402
REFERENCE = SS / "reference/reference.py"
FIXTURES = SS / "fixtures"
B36 = b"0123456789abcdefghijklmnopqrstuvwxyz"


def enc36(v):
    out = bytearray()
    while True:
        out.append(B36[v % 36])
        v //= 36
        if v == 0:
            break
    return bytes(reversed(out))


def split(line):
    """(values, quoted flags) of a well-formed line, None for a malformed one.
    Quotes wrap whole fields only (rules: no value holds a comma or quote)."""
    if not line:
        return None
    parts = line.split(b",")
    vals, q = [], []
    for p in parts:
        if b'"' in p:
            if len(p) >= 2 and p[0] == 34 and p[-1] == 34 and b'"' not in p[1:-1]:
                vals.append(p[1:-1])
                q.append(True)
            else:
                return None
        else:
            vals.append(p)
            q.append(False)
    return vals, q


def join(vals, q):
    return b",".join([b'"' + v + b'"' if qq else v for v, qq in zip(vals, q)])


def rand_q(rng, n, p=0.2):
    return [rng.random() < p for _ in range(n)]


class CsvFile:
    def __init__(self, path):
        self.path = Path(path)
        data = self.path.read_bytes()
        self.trail = data.endswith(b"\n")
        lines = data.split(b"\n")
        if self.trail:
            lines.pop()
        self.header = lines[0] if lines else b""
        self.rows = lines[1:]

    def save(self, path=None):
        path = Path(path) if path else self.path
        out = b"\n".join([self.header] + self.rows)
        if self.trail:
            out += b"\n"
        tmp = path.with_name(".tmp-" + path.name)  # never write through a hard link
        tmp.write_bytes(out)
        os.replace(tmp, path)
NCOLS = {"events": 7, "clicks": 4, "links": 2}
ID_COLS = {"events": [(0, "e"), (1, "u")], "clicks": [(0, "k"), (1, "u")], "links": [(0, "u"), (1, "u")]}


class DS:
    def __init__(self, root):
        self.root = Path(root)

    def paths(self, kind):
        return sorted(p for p in (self.root / kind).glob("*.csv") if not p.name.startswith("."))

    def load(self, kind):
        return [CsvFile(p) for p in self.paths(kind)]

    def window(self):
        m = json.loads((self.root / "manifest.json").read_text())
        return parse_ts(m["window_start"].encode()), parse_ts(m["window_end"].encode())

    def event_bytes(self):
        return sum(p.stat().st_size for p in self.paths("events"))

    def rows(self, kind):
        """(file index, row index, vals, quoted) of every well-formed row."""
        n = NCOLS[kind]
        for fi, f in enumerate(self.load(kind)):
            for ri, line in enumerate(f.rows):
                sp = split(line)
                if sp and len(sp[0]) == n:
                    yield fi, ri, sp[0], sp[1]
F1 = re.compile(rb"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})Z$")
F2 = re.compile(rb"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})([+-])(\d{2}):(\d{2})$")
F3 = re.compile(rb"^\d+$")
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
DAY = 86_400_000


def parse_ts(b):
    """Epoch ms of a valid F1/F2/F3 timestamp, else None."""
    try:
        m = F1.match(b)
        if m:
            y, mo, d, h, mi, s = map(int, m.groups())
            return int((datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc) - EPOCH).total_seconds()) * 1000
        m = F2.match(b)
        if m:
            y, mo, d, h, mi, s = map(int, m.groups()[:6])
            oh, om = int(m.group(8)), int(m.group(9))
            if oh > 23 or om > 59:
                return None
            sign = 1 if m.group(7) == b"+" else -1
            t = int((datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc) - EPOCH).total_seconds()) * 1000
            return t - sign * (oh * 3600 + om * 60) * 1000
        if F3.match(b):
            return int(b)
    except (ValueError, OverflowError):
        return None
    return None


def fmt_f1(ms):
    dt = EPOCH + timedelta(milliseconds=(ms // 1000) * 1000)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ").encode()


def fmt_f3(ms):
    return str(ms).encode()
DATE_PREFIX = re.compile(rb"^(\d{4})-(\d{2})-(\d{2})(.*)$", re.S)


def shift_raw(b, days):
    """Shifts a raw timestamp by whole days, keeping its form; bad ones stay as they are."""
    m = DATE_PREFIX.match(b)
    if m:
        try:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3))) + timedelta(days=days)
        except (ValueError, OverflowError):
            return b
        return b"%04d-%02d-%02d" % (d.year, d.month, d.day) + m.group(4)
    if F3.match(b):
        v = int(b) + days * DAY
        if v < 0:
            return b
        s = str(v).encode()
        return s.rjust(len(b), b"0") if len(s) < len(b) else s
    return b
BAD_TS = [b"garbage", b"", b"2027-02-30T10:00:00Z", b"1791071890000.0", b"2026-10-03 10:00:00", b"-5",
          b"2026-10-03T10:00:00z", b"2026-10-03T10:00:00+0200"]


def f_upper(v, r):
    u = v.upper()
    return u if u != v else None


def f_pad(v, r):
    return b" " + v if r.random() < 0.5 else v + b" "


def f_lead0(v, r):
    return v[:1] + b"0" + v[1:] if len(v) >= 2 else None


def f_long(v, r):
    return v + bytes(r.choice(B36) for _ in range(r.randint(1, 4)))


def identity(kind, vals):
    """What makes two rows copies that must stay identical: (event_id, ingest_ts)
    for events (equal-ingest copies are identical), click_id for clicks; links
    have no copies."""
    if kind == "events":
        return b"E" + vals[0] + b"|" + vals[3]
    if kind == "clicks":
        return b"C" + vals[0]
    return None


def mutate_ids(ds, rng, fn, rate, crate, kinds="euk", salt=b"", skip_first_event_file=False):
    """Applies fn to `rate` of ID fields (per occurrence) and to every occurrence
    of `crate` of the distinct IDs (consistently, links included). A per-occurrence
    change of a non-key field (the user_id of an event or click) is decided and
    computed once per copy identity, so copies that must be identical stay so;
    changing an event_id / click_id of one copy just makes it another event."""
    cut = int(crate * (1 << 20))
    count = 0
    cache = {}
    for kind, cols in ID_COLS.items():
        n = NCOLS[kind]
        for fi, f in enumerate(ds.load(kind)):
            if skip_first_event_file and kind == "events" and fi == 0:
                continue
            changed = False
            rows = f.rows
            for i, line in enumerate(rows):
                sp = split(line)
                if not sp or len(sp[0]) != n:
                    continue
                vals, q = sp
                ident = identity(kind, vals)
                hit = False
                for c, k in cols:
                    v = vals[c]
                    if not v or k not in kinds:
                        continue
                    nv = None
                    if cut and (zlib.crc32(salt + v) & 0xFFFFF) < cut:
                        nv = fn(v, random.Random(zlib.crc32(b"x" + salt + v)))
                    elif rate and ident is not None and c != 0:
                        key = (ident, c)
                        if key not in cache:
                            r = random.Random(zlib.crc32(salt + ident + bytes([c])))
                            cache[key] = fn(v, r) if r.random() < rate else None
                        nv = cache[key]
                    elif rate and rng.random() < rate:
                        nv = fn(v, rng)
                    if nv is not None and nv != v:
                        vals[c] = nv
                        hit = True
                if hit:
                    rows[i] = join(vals, q)
                    changed = True
                    count += 1
            if changed:
                f.save()
    return count


def user_pool(ds, limit=400_000):
    pool = set()
    for _, _, v, _ in ds.rows("events"):
        if v[1]:
            pool.add(v[1])
        if len(pool) >= limit:
            break
    return sorted(pool)


def retype(v, rng):
    r = rng.random()
    if r < 0.25:
        v[4] = b"view"
    elif r < 0.4:
        v[4] = b"cart"
    elif r < 0.6:
        v[4] = b"purchase"
        v[6] = rng.choice([str(rng.randint(1, 50_000)).encode(), b"0", b"12.5", b"10000001", b"0250", b""])
        v[5] = b"p%d" % rng.randint(1, 50_000)


def add_dups(ds, rng, rate=0.02):
    """Extra copies of `rate` of the events: later copies under other users,
    earlier (winning) copies under other users, an earliest copy with a bad ts
    (or bad type) before valid ones, identical copies quoted differently.
    Copies land in random files at random positions."""
    files = ds.load("events")
    pool = user_pool(ds, 50_000)
    extra = [[] for _ in files]
    stats = defaultdict(int)
    chosen = []
    for f in files:
        for line in f.rows:
            if rng.random() >= rate:
                continue
            sp = split(line)
            if sp and len(sp[0]) == 7 and sp[0][0] and parse_ts(sp[0][3]) is not None:
                chosen.append(sp[0])
    # every ingest_ts already used by a chosen event: a new copy never gets one
    # (a copy with an equal ingest must be identical)
    ids = {v[0] for v in chosen}
    taken = defaultdict(set)
    for f in files:
        for line in f.rows:
            if line.split(b",", 1)[0].strip(b'"') in ids:
                sp = split(line)
                if sp and len(sp[0]) == 7 and sp[0][0] in ids:
                    t = parse_ts(sp[0][3])
                    if t is not None:
                        taken[sp[0][0]].add(t)

    def new_ing(eid, base, lo, hi, sign):
        for _ in range(100):
            t = base + sign * rng.randint(lo, hi) * 1000
            if t not in taken[eid]:
                taken[eid].add(t)
                return fmt_f1(t)
        raise RuntimeError("no free ingest_ts")

    def emit(v):
        extra[rng.randrange(len(files))].append(join(v, rand_q(rng, 7)))

    for vals in chosen:
        eid, ing = vals[0], parse_ts(vals[3])
        r = rng.random()
        if r < 0.35:
            stats["later-other-user"] += 1
            for _ in range(rng.randint(1, 3)):
                v = list(vals)
                v[3] = new_ing(eid, ing, 1, 3600, 1)
                v[1] = rng.choice(pool)
                retype(v, rng)
                emit(v)
        elif r < 0.6:
            stats["earlier-other-user-wins"] += 1
            v = list(vals)
            v[3] = new_ing(eid, ing, 1, 3600, -1)
            v[1] = rng.choice(pool)
            retype(v, rng)
            emit(v)
        elif r < 0.75:
            stats["earliest-bad-ts"] += 1
            v = list(vals)
            v[2] = rng.choice(BAD_TS)
            v[3] = new_ing(eid, ing, 1, 600, -1)
            emit(v)
            v2 = list(vals)
            v2[3] = new_ing(eid, ing, 1, 600, 1)
            v2[1] = rng.choice(pool)
            emit(v2)
        elif r < 0.9:
            stats["identical"] += 1
            for _ in range(rng.randint(1, 2)):
                emit(list(vals))
        else:
            stats["earliest-bad-type"] += 1
            v = list(vals)
            v[3] = new_ing(eid, ing, 1, 600, -1)
            v[4] = rng.choice([b"Purchase", b"cart ", b"VIEW", b""])
            emit(v)
    for f, ex in zip(files, extra):
        if ex:
            f.rows.extend(ex)
            rng.shuffle(f.rows)
            f.save()
    return dict(stats)


def link_mutations(ds, rng, mega=2000):
    users = user_pool(ds)
    links = ds.load("links")
    removed = 0
    for f in links:
        before = len(f.rows)
        f.rows = [l for l in f.rows if rng.random() >= 0.01]
        removed += before - len(f.rows)
    new = []
    for _ in range(max(1, len(users) // 200)):  # random pairs
        a, b = rng.sample(users, 2)
        new.append((a, b))
    for _ in range(10):  # chains merging 50 people
        ch = rng.sample(users, min(50, len(users)))
        for x, y in zip(ch, ch[1:]):
            new.append((x, y) if rng.random() < 0.5 else (y, x))
    ch = rng.sample(users, min(mega, len(users)))  # one mega person, shuffled chain
    for x, y in zip(ch, ch[1:]):
        new.append((x, y) if rng.random() < 0.5 else (y, x))
    smalls = [b"u0", b"u1", b"u2", b"uz"]
    for _ in range(300):  # link-only IDs smaller than every other ID of the person
        u = rng.choice(users)
        k = rng.random()
        if k < 0.25:
            s = b"A" + bytes(rng.choice(B36) for _ in range(6)).upper()
        elif k < 0.45:
            s = b"0" + bytes(rng.choice(B36) for _ in range(6))
        elif k < 0.6:
            s = b"-" + u
        elif k < 0.75:
            s = b" " + u
        elif k < 0.9:
            s = b"U" + u[1:]
        else:
            s = rng.choice(smalls)  # shared: merges several people
        new.append((u, s) if rng.random() < 0.5 else (s, u))
    rng.shuffle(new)
    extra = CsvFile.__new__(CsvFile)
    extra.path = ds.root / "links" / "zz-extra.csv"
    extra.trail, extra.header, extra.rows = True, b"user_id_a,user_id_b", []
    for i, (a, b) in enumerate(new):
        line = join([a, b], rand_q(rng, 2))
        if i % 2:
            extra.rows.append(line)
        else:
            f = rng.choice(links)
            f.rows.insert(rng.randrange(len(f.rows) + 1), line)
    for f in links:
        f.save()
    extra.save()
    return {"removed": removed, "added": len(new)}


def ingest_clamp(ds, rng, n=300):
    """Copies whose ingest_ts are > 68 years before the window (1700/1800/1900/1901):
    the earliest still must win; an encoding that clamps would tie them."""
    files = ds.load("events")
    ws, we = ds.window()
    pool = user_pool(ds, 20_000)
    cands = []
    for fi, f in enumerate(files):
        for line in f.rows:
            sp = split(line)
            if sp and len(sp[0]) == 7 and sp[0][0] and sp[0][1] and sp[0][4] in (b"view", b"cart"):
                t = parse_ts(sp[0][2])
                if t is not None and ws <= t < we:
                    cands.append(sp[0])
    first, last = files[0], files[-1]
    pairs = [(b"1900-01-01T00:00:00Z", b"1901-06-01T00:00:00Z"), (b"1700-03-01T12:00:00Z", b"1800-01-01T00:00:00Z"),
             (b"1950-01-01T00:00:00Z", b"1950-01-01T00:00:01Z")]
    for k, vals in enumerate(rng.sample(cands, min(n, len(cands)))):
        early, late = rng.choice(pairs)
        a, b = list(vals), list(vals)
        a[3], a[1] = early, rng.choice(pool)
        b[3], b[1] = late, rng.choice(pool)
        # the later-ingested copy is read first half the time
        (first if k % 2 else last).rows.insert(0, join(b, rand_q(rng, 7)))
        (last if k % 2 else first).rows.append(join(a, rand_q(rng, 7)))
    first.save()
    if last is not first:
        last.save()
    return {"events": min(n, len(cands))}


def crlf_and_files(ds, rng):
    """3% of lines end in \\r; the largest events file split into 150 files; extra
    header-only files; some files without a trailing newline."""
    def cr(kind, line):
        sp = split(line)
        ident = identity(kind, sp[0]) if sp and len(sp[0]) == NCOLS[kind] else None
        if ident is None:
            return rng.random() < 0.03
        return zlib.crc32(b"crlf" + ident) % 1000 < 30  # the same for every copy

    for kind in ("events", "clicks", "links"):
        files = ds.load(kind)
        for f in files:
            f.rows = [l + b"\r" if l and cr(kind, l) else l for l in f.rows]
            if rng.random() < 0.3:
                f.trail = False
            f.save()
        for j in range(3):
            (ds.root / kind / f"zz-empty-{j}.csv").write_bytes(files[0].header + b"\n")
    files = ds.load("events")
    big = max(files, key=lambda f: len(f.rows))
    rows = big.rows
    k = 150
    step = len(rows) // k + 1
    for j in range(k):
        g = CsvFile.__new__(CsvFile)
        g.path = big.path.with_name(big.path.stem + f"-s{j:03d}.csv")
        g.header, g.rows, g.trail = big.header, rows[j * step:(j + 1) * step], j % 7 != 0
        g.save()
    big.path.unlink()
    return {"split_rows": len(rows)}


def quote_all(ds, rng):
    """Every field quoted in half of the files, each field with p=0.5 in the rest."""
    for kind in ("events", "clicks", "links"):
        for fi, f in enumerate(ds.load(kind)):
            out = []
            for line in f.rows:
                sp = split(line)
                if not sp:
                    out.append(line)
                    continue
                q = [True] * len(sp[0]) if fi % 2 == 0 else rand_q(rng, len(sp[0]), 0.5)
                out.append(join(sp[0], q))
            f.rows = out
            f.save()
    return {}


def legality(ds):
    """The input preconditions a mutation could break (the reference's checks):
    equal-ingest event copies must be identical, click copies must be identical.
    Returns a short problem string, '' when legal."""
    kept = {}
    eq_diff = 0
    for _, _, v, _ in ds.rows("events"):
        if not v[0]:
            continue
        h = hash(tuple(v))
        prev = kept.get(v[0])
        if prev is None or v[3] < prev[0]:  # raw F1 order is time order
            kept[v[0]] = (v[3], h)
        elif v[3] == prev[0] and h != prev[1]:
            eq_diff += 1
    kept = {}
    cl_diff = 0
    for _, _, v, _ in ds.rows("clicks"):
        if not v[0]:
            continue
        h = hash(tuple(v))
        prev = kept.setdefault(v[0], h)
        cl_diff += prev != h
    probs = []
    if eq_diff:
        probs.append(f"{eq_diff} equal-ingest event copies differ")
    if cl_diff:
        probs.append(f"{cl_diff} click copies differ")
    return "; ".join(probs)


def long_window(ds, days_total=24, shift=10, frac=0.4):
    """window_end moved to window_start + days_total (the binary's limit is < 24.86
    days); the events and clicks of `frac` of the user IDs move `shift` days later."""
    m = json.loads((ds.root / "manifest.json").read_text())
    ws, _ = ds.window()
    m["window_end"] = fmt_f1(ws + days_total * DAY).decode()
    (ds.root / "manifest.json").write_text(json.dumps(m, indent=2) + "\n")
    cut = int(frac * (1 << 20))
    moved = 0
    for kind, cols in (("events", (2, 3)), ("clicks", (3,))):
        for f in ds.load(kind):
            out = []
            for line in f.rows:
                sp = split(line)
                if sp and len(sp[0]) == NCOLS[kind] and (zlib.crc32(b"lw" + sp[0][1]) & 0xFFFFF) < cut:
                    vals, q = sp
                    for c in cols:
                        vals[c] = shift_raw(vals[c], shift)
                    line = join(vals, q)
                    moved += 1
                out.append(line)
            f.rows = out
            f.save()
    return {"window_end": m["window_end"], "rows_moved": moved}


def concat_two(dsa, dsb, variant):
    """Appends set B (timestamps shifted by whole days into A's window) to set A's
    events/clicks/links dirs; A's manifest, products, campaigns stay."""
    wa, _ = dsa.window()
    wb, _ = dsb.window()
    days = (wa - wb) // DAY
    rng = random.Random(days)
    for kind, cols in (("events", (2, 3)), ("clicks", (3,)), ("links", ())):
        afiles = dsa.load(kind) if variant == "shuffled" else []
        for f in dsb.load(kind):
            out = []
            for line in f.rows:
                sp = split(line)
                if not sp or len(sp[0]) != NCOLS[kind]:
                    out.append(line)
                    continue
                vals, q = sp
                for c in cols:
                    vals[c] = shift_raw(vals[c], days)
                out.append(join(vals, q))
            f.rows = out
            if variant == "shuffled":  # B's rows mixed into A's files: two keys in every sample
                for line in out:
                    rng.choice(afiles).rows.append(line)
                continue
            name = f.path.name
            if variant == "b-last":
                name = "zz-" + name
            elif variant == "b-first":
                name = "00-" + name
            else:  # interleaved with A's part files in sorted order
                name = f.path.stem + "b.csv"
            f.save(dsa.root / kind / name)
        for g in afiles:
            rng.shuffle(g.rows)
            g.save()
    return {"shift_days": days}


class Harness:
    """Output comparisons (the harness classes live in the scripts)."""

    @staticmethod
    def ref_cmp(ref_dir, out_dir):
        """reference CSV vs pipeline Parquet, byte-exact: polars' CSV reader drops a
        '\\r' at the end of an ID (IDs from CRLF link lines), so the reference CSV is
        parsed by hand here (it never quotes; values never hold commas)."""
        import polars as pl
        ok, buf = True, []
        for table, schema in SCHEMAS.items():
            lines = (Path(ref_dir) / f"{table}.csv").read_bytes().split(b"\n")
            names = lines[0].decode().split(",")
            ra = []
            for l in lines[1:]:
                if not l:
                    continue
                row = dict(zip(names, l.split(b",")))
                vals = []
                for col, typ in schema.items():
                    v = row[col]
                    if typ == pl.Date:
                        vals.append(date.fromisoformat(v.decode()))
                    elif typ == pl.String:
                        vals.append(v.decode("utf-8", "surrogateescape") if v else None)
                    else:
                        vals.append(int(v))
                ra.append(tuple(vals))
            rb = [tuple(r) for r in load(out_dir, table).rows()]
            sa, sb = sorted(ra, key=repr), sorted(rb, key=repr)
            keys = [list(schema).index(k) for k in KEYS[table]]
            dup = len(ra) - len({tuple(r[i] for i in keys) for r in ra})
            if sa == sb and dup == 0:
                continue
            ok = False
            setb, seta = set(rb), set(ra)
            oa = sorted(seta - setb, key=repr)
            ob = sorted(setb - seta, key=repr)
            buf.append(f"{table}: rows {len(ra)} vs {len(rb)}; only in reference {len(oa)}; only in general "
                       f"{len(ob)}; duplicate keys in reference {dup}")
            buf += [f"  ref: {r}" for r in oa[:8]] + [f"  gen: {r}" for r in ob[:8]]
        return ok, "\n".join(buf)

    @staticmethod
    def cmp(a, b):
        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                ok = compare(str(a), str(b), verbose=False)
            except Exception as e:  # noqa: BLE001
                print("compare error:", e)
                ok = False
        return ok, buf.getvalue()
