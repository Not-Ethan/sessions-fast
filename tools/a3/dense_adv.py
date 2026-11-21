#!/usr/bin/env python3
"""
Differential / adversarial harness for the dense pipeline (SS_DENSE).

For every case: build a dataset (generator output, then CSV text mutated in
legal ways), run the binary as the general pipeline (no SS_DENSE) and in the
dense modes below (NCPU=2 unless noted, SS_VERBOSE=1), compare each dense
output with the general one (tools/compare.py), and on small cases (<= 300K
events) also run reference/reference.py as a third opinion. Prints one line
per case; exits 1 on any mismatch or crash and keeps the mismatching dataset
(its path is printed).

Dense modes (columns):
  d1      SS_DENSE=1          ([dense] = it really took the dense path, [general] = it fell back)
  force   SS_DENSE=force
  capdiv  SS_DENSE=force SS_CAPDIV=64   (tiny Parts capacities: overflow paths)
  nc1     SS_DENSE=force NCPU=1

Usage (run with the venv python that has polars):
  dense_adv.py --smoke                quick: plain 50K, upper-cased IDs, the 36 fixtures
  dense_adv.py                        every case
  dense_adv.py --cases 2,3,13         chosen cases (see --list)
  options: --no-ref  --modes d1,force  --keep  --scratch DIR  --bin PATH  --list
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
sys.path.insert(0, str(SS / "tools"))
from compare import KEYS, SCHEMAS, compare, load  # noqa: E402

BIN = SS / "target/release/sessions"
REFERENCE = SS / "reference/reference.py"
FIXTURES = SS / "fixtures"
GEN = Path("/Users/Ethan_1/buildersbuildathon/_public/work-ss/gen-target/release/generate")
SCRATCH = Path("/private/tmp/claude-501/-Users-Ethan-1-buildersbuildathon/258aa4f2-bbdf-41a5-8073-44d5c18781f7/scratchpad/a3adv")

ALL_MODES = {
    "d1": {"SS_DENSE": "1"},
    "force": {"SS_DENSE": "force"},
    "capdiv": {"SS_DENSE": "force", "SS_CAPDIV": "64"},
    "nc1": {"SS_DENSE": "force", "NCPU": "1"},
}
REF_MAX_EVENTS = 300_000

# ------------------------------------------------------------- dense ID math (generator's scramble)
MASK = (1 << 40) - 1
M1 = 0x9E3779B97F4B
M2 = 0xD6E8FEB86659
M1I = pow(M1, -1, 1 << 40)
M2I = pow(M2, -1, 1 << 40)
B36 = b"0123456789abcdefghijklmnopqrstuvwxyz"


def scramble(v, key):
    m = ((v & MASK) * M1 ^ key) & MASK
    m ^= m >> 17
    return (m * M2) & MASK


def half(v):
    t = (v * M2I) & MASK
    return t ^ (t >> 17) ^ (t >> 34)


def unscramble(v, key):
    return ((half(v) ^ key) * M1I) & MASK


def enc36(v):
    out = bytearray()
    while True:
        out.append(B36[v % 36])
        v //= 36
        if v == 0:
            break
    return bytes(reversed(out))


def decode36(s, prefix):
    """The 40-bit number of a canonical ID (prefix + 1-8 lowercase base-36 digits,
    no leading zero unless the single digit 0, value < 2^40), else None."""
    if not (2 <= len(s) <= 9) or s[0] != prefix or (len(s) > 2 and s[1] == 0x30):
        return None
    if any(c not in B36 for c in s[1:]):
        return None
    v = int(s[1:], 36)
    return v if v <= MASK else None


def recover_key(samples, bound, step, need=0.9, tries=12):
    """A key under which >= need of the samples unscramble below bound, trying
    sample a's number = 0, step, 2 step, ... (brute force, like dense.rs)."""
    hs = [half(v) for v in samples]
    if len(hs) < 8:
        return None
    for a in range(min(tries, len(hs))):
        ha = hs[a]
        others = hs[:a] + hs[a + 1:]
        h1, h2 = others[0], others[1]
        mstep = (M1 * step) & MASK
        m = 0
        for _ in range(0, bound, step):
            k = ha ^ m
            m = (m + mstep) & MASK
            if (((h1 ^ k) * M1I) & MASK) < bound or (((h2 ^ k) * M1I) & MASK) < bound:
                ok = sum(1 for h in others if (((h ^ k) * M1I) & MASK) < bound)
                if ok >= need * len(others):
                    return k
    return None


# ------------------------------------------------------------------ CSV text handling
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

    def bounds(self):
        """bound / pmax exactly as densep::decide computes them."""
        est = self.event_bytes() // 70 + 1
        return min(max(2 * est, 1 << 16), 1 << 32), min(est // 8 + 4096, 1 << 27)

    def rows(self, kind):
        """(file index, row index, vals, quoted) of every well-formed row."""
        n = NCOLS[kind]
        for fi, f in enumerate(self.load(kind)):
            for ri, line in enumerate(f.rows):
                sp = split(line)
                if sp and len(sp[0]) == n:
                    yield fi, ri, sp[0], sp[1]


# ------------------------------------------------------------------ timestamps
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


# ------------------------------------------------------------------ mutators
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


def far_event_ids(ds, rng, rate, crate, skip_first_file=False):
    """Replaces event IDs with random IDs not otherwise present: canonical 8-digit
    (mostly), 8 digits >= 2^40 (non-canonical), short canonical."""
    existing = {v[0] for _, _, v, _ in ds.rows("events")}

    def newid(r):
        while True:
            x = r.random()
            if x < 0.7:
                v = r.randrange(36 ** 7, 1 << 40)
            elif x < 0.85:
                v = r.randrange(1 << 40, 36 ** 8)
            else:
                v = r.randrange(0, 36 ** r.randint(1, 7))
            s = b"e" + enc36(v)
            if s not in existing:
                return s

    return mutate_ids(ds, rng, lambda v, r: newid(r), rate, crate, kinds="e", salt=b"far",
                      skip_first_event_file=skip_first_file)


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


def event_keys(ds, E):
    """(event key, user key) of a generated dataset, recovered from its first rows."""
    es, us = [], []
    for _, _, v, _ in ds.rows("events"):
        a = decode36(v[0], 0x65)
        if a is not None and len(es) < 64:
            es.append(a)
        b = decode36(v[1], 0x75)
        if b is not None and len(us) < 64:
            us.append(b)
        if len(es) >= 64 and len(us) >= 64:
            break
    ke = recover_key(es, int(E * 1.1) + 1000, 1)
    ku = recover_key(us, 512 * (E // 8 + 4096), 512, need=0.8)
    if ke is None or ku is None:
        raise RuntimeError("could not recover dataset keys")
    return ke, ku


def backwards_times(ds, rng, ke, frac=0.01):
    """For `frac` of users: reassign their events' times so that in event-number
    order the times run backwards (all copies of an event shift together, ingest
    with them, so dedup winners and late status stay)."""
    files = ds.load("events")
    by_user = defaultdict(list)
    for fi, f in enumerate(files):
        for ri, line in enumerate(f.rows):
            sp = split(line)
            if not sp or len(sp[0]) != 7:
                continue
            v = sp[0]
            a = decode36(v[0], 0x65)
            i = parse_ts(v[3])
            if a is None or i is None or not v[1]:
                continue
            by_user[v[1]].append((fi, ri, unscramble(a, ke), parse_ts(v[2]), i))
    users = sorted(u for u, l in by_user.items() if len({x[2] for x in l if x[3] is not None}) >= 3)
    chosen = rng.sample(users, max(1, int(len(users) * frac)))
    touched = 0
    for u in chosen:
        evs = by_user[u]
        prim = {}
        for e in evs:
            if e[3] is not None and (e[2] not in prim or e[4] < prim[e[2]][4]):
                prim[e[2]] = e
        ns = sorted(prim)
        tss = sorted(prim[n][3] for n in ns)
        newts = dict(zip(ns, reversed(tss)))
        if len(ns) >= 3 and rng.random() < 0.3:
            newts[ns[1]] = newts[ns[0]]  # a tie
        shift = {n: round((newts[n] - prim[n][3]) / 1000) * 1000 for n in ns}
        for fi, ri, n, t, i in evs:
            if n not in shift:
                continue
            vals, q = split(files[fi].rows[ri])
            s = shift[n]
            if t is not None:
                nt = t + s
                vals[2] = fmt_f1(nt) if (n % 2 and nt % 1000 == 0) else fmt_f3(nt)
            vals[3] = fmt_f1(i + s)
            files[fi].rows[ri] = join(vals, q)
            touched += 1
    for f in files:
        f.save()
    return {"users": len(chosen), "rows": touched}


def canon_edges(ds, rng, ke, ku, E):
    """Canonical IDs on the dense pipeline's edges: event numbers in [E, bound),
    around bound and in the last bucket; users with alias >= 8, unused aliases
    < 8, person index >= pmax or near it, numbers >= 2^39; link-only canonical
    IDs (dense or not) that may become a person's name."""
    bound, pmax = ds.bounds()
    ev_ids = {v[0] for _, _, v, _ in ds.rows("events")}
    users = set()
    for kind in ("events", "clicks", "links"):
        for _, _, v, _ in ds.rows(kind):
            users.add(v[1])
            if kind == "links":
                users.add(v[0])
    users.discard(b"")

    def eenc(n):
        return b"e" + enc36(scramble(n, ke))

    def uenc(u):
        return b"u" + enc36(scramble(u, ku))

    def person_of(s):
        a = decode36(s, 0x75)
        if a is None:
            return None
        u = unscramble(a, ku)
        return u >> 9 if u < (1 << 39) else None

    lastb = ((bound - 1) >> 16) << 16

    def new_event_id(r):
        while True:
            x = r.random()
            if x < 0.4:
                n = r.randrange(E, bound) if bound > E else r.randrange(bound)
            elif x < 0.6:
                n = r.randrange(max(0, bound - 64), bound + 64)
            elif x < 0.8:
                n = r.randrange(lastb, bound)
            else:
                n = r.randrange(bound, 4 * bound)
            s = eenc(n)
            if s not in ev_ids:
                return s

    def new_user(s, r):
        p = person_of(s)
        for _ in range(100):
            x = r.random()
            if p is not None and x < 0.3:
                u = p * 512 + r.randint(8, 511)
            elif p is not None and x < 0.55:
                u = p * 512 + r.randint(0, 7)
            elif x < 0.7:
                u = (pmax + r.randint(0, 1000)) * 512 + r.randint(0, 7)
            elif x < 0.85:
                u = (pmax - 1 - r.randint(0, 3)) * 512 + r.randint(0, 7)
            else:
                u = r.randrange(1 << 39, 1 << 40)
            t = uenc(u)
            if t not in users:
                users.add(t)
                return t
        return None

    new_links = []

    def nu(s, r):
        t = new_user(s, r)
        if t is not None and r.random() < 0.5:
            new_links.append((s, t))
        return t

    ne = mutate_ids(ds, rng, lambda v, r: new_event_id(r), 0.003, 0.003, kinds="e", salt=b"ce")
    nuc = mutate_ids(ds, rng, nu, 0.005, 0.0, kinds="u", salt=b"cu")
    # link-only canonical IDs for 2% of persons
    for s in rng.sample(sorted(users), max(1, len(users) // 50)):
        p = person_of(s)
        if p is None:
            continue
        for _ in range(20):
            u = p * 512 + (rng.randint(1, 7) if rng.random() < 0.6 else rng.randint(8, 511))
            t = uenc(u)
            if t not in users:
                users.add(t)
                new_links.append((s, t) if rng.random() < 0.5 else (t, s))
                break
    links = ds.load("links")
    for a, b in new_links:
        f = rng.choice(links)
        f.rows.insert(rng.randrange(len(f.rows) + 1), join([a, b], rand_q(rng, 2)))
    for f in links:
        f.save()
    return {"event_rows": ne, "user_rows": nuc, "links": len(new_links), "bound": bound, "pmax": pmax}


def sort_by_number(ds, ke, variant):
    """All event rows sorted by event number: 'asc' in one file, 'desc3' reversed
    across three files (60/30/10%); the remaining files keep only their header."""
    files = ds.load("events")
    rows = []
    for f in files:
        for line in f.rows:
            sp = split(line)
            a = decode36(sp[0][0], 0x65) if sp and len(sp[0]) == 7 else None
            rows.append((unscramble(a, ke) if a is not None else 1 << 41, line))
    rows.sort(key=lambda x: x[0], reverse=(variant == "desc3"))
    lines = [l for _, l in rows]
    for f in files:
        f.rows = []
    if variant == "asc":
        files[0].rows = lines
    else:
        a, b = int(len(lines) * 0.6), int(len(lines) * 0.9)
        files[0].rows, files[1].rows, files[2].rows = lines[:a], lines[a:b], lines[b:]
    for f in files:
        f.save()
    return {"rows": len(lines)}


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


# ------------------------------------------------------------------ harness
class Harness:
    def __init__(self, args):
        self.args = args
        self.scratch = Path(args.scratch)
        self.bin = Path(args.bin)
        self.modes = [(m, ALL_MODES[m]) for m in args.modes.split(",")]
        self.results = []
        self.failed = []
        (self.scratch / "cases").mkdir(parents=True, exist_ok=True)
        (self.scratch / "out").mkdir(parents=True, exist_ok=True)

    def lock(self):
        """--flock FILE: every generator / binary / reference run under flock."""
        return ["flock", self.args.flock] if self.args.flock else []

    def gen(self, name, seed, events, clicks):
        d = self.scratch / "cases" / name
        shutil.rmtree(d, ignore_errors=True)
        subprocess.run(self.lock() + [str(self.args.gen), "--seed", str(seed), "--events", str(events), "--clicks", str(int(clicks)),
                        "--output", str(d), "--threads", "2"], check=True, capture_output=True)
        return d

    def run_bin(self, data, out, extra):
        env = {k: v for k, v in os.environ.items() if not k.startswith("SS_")}
        env.update({"NCPU": str(self.args.ncpu), "SS_VERBOSE": "1"})
        # --env: applied to every run (general and dense), e.g. SS_NOAVX2=1
        for kv in self.args.env:
            k, _, v = kv.partition("=")
            env[k] = v
        env.update(extra)
        shutil.rmtree(out, ignore_errors=True)
        t0 = time.time()
        try:
            p = subprocess.run(self.lock() + [str(self.bin), str(data), str(out)], env=env, capture_output=True, text=True,
                               timeout=self.args.timeout)
            rc, err = p.returncode, p.stderr
        except subprocess.TimeoutExpired:
            rc, err = -999, "timeout"
        return rc, err, time.time() - t0

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

    def evaluate(self, name, data, events, prep, dense_env=None, note="", mutated=True):
        out = self.scratch / "out" / name
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        cols, times, fails, details = [], [], [], []
        if mutated:
            t0 = time.time()
            prob = legality(DS(data))
            prep += time.time() - t0
            if prob:
                fails.append(f"ILLEGAL INPUT (harness bug): {prob}")
        g_rc, g_err, g_t = self.run_bin(data, out / "general", {"SS_DENSE": "0"})
        times.append(f"gen {g_t:.2f}")
        if g_rc != 0:
            fails.append(f"general exit {g_rc}")
            details.append(g_err[-1500:])
        for label, env in self.modes:
            env = dict(env)
            env.update(dense_env or {})
            rc, err, t = self.run_bin(data, out / label, env)
            times.append(f"{label} {t:.2f}")
            taken = "pipeline=dense" in err
            tag = "[dense]" if taken else "[general]"
            if rc != 0:
                st = f"CRASH({rc})"
                details.append(f"--- {label} stderr:\n{err[-1500:]}")
            elif g_rc != 0:
                st = "n/a"
            else:
                ok, txt = self.cmp(out / "general", out / label)
                st = "MATCH" if ok else "MISMATCH"
                if not ok:
                    details.append(f"--- general (A) vs {label} (B):\n{txt}")
            if st != "MATCH":
                fails.append(f"{label} {st}")
            cols.append(f"{label}={st}{tag if label == 'd1' or not taken else ''}")
        if self.args.ref and events <= REF_MAX_EVENTS and g_rc == 0:
            t0 = time.time()
            p = subprocess.run(self.lock() + [sys.executable, str(REFERENCE), str(data), str(out / "ref")], capture_output=True,
                               text=True)
            times.append(f"ref {time.time() - t0:.1f}")
            warns = [l for l in p.stderr.splitlines() if "WARNING" in l]
            if warns:
                fails.append("reference warns (illegal input?): " + " | ".join(w[:200] for w in warns[:3]))
            if p.returncode != 0:
                cols.append("ref=CRASH")
                fails.append("ref crash")
                details.append(p.stderr[-1500:])
            else:
                ok, txt = self.ref_cmp(out / "ref", out / "general")
                cols.append("ref=" + ("MATCH" if ok else "MISMATCH"))
                if not ok:
                    fails.append("ref MISMATCH (general vs reference)")
                    details.append(f"--- reference (A) vs general (B):\n{txt}")
        else:
            cols.append("ref=skip")
        line = f"{name:<26} ev={events:<9} {' '.join(cols)}  | prep {prep:.1f}s, run " + ", ".join(times) + "s"
        if note:
            line += f"  ({note})"
        print(line, flush=True)
        self.results.append(line)
        if fails:
            self.failed.append(name)
            print(f"    FAIL: {', '.join(fails)}\n    dataset kept: {data}\n    outputs kept: {out}")
            for d in details:
                print("    " + d.replace("\n", "\n    "))
            sys.stdout.flush()
        elif not self.args.keep:
            shutil.rmtree(out, ignore_errors=True)
            if str(data).startswith(str(self.scratch)):
                shutil.rmtree(data, ignore_errors=True)
        return not fails

    # -------------------------------------------------------------- cases
    def c01_plain(self):
        sets = [(105, 50_000, 0.25)] if self.args.smoke else [
            (101, 1_000, 0.25), (102, 1_000, 0.025), (103, 1_000, 1.0), (104, 1_000, 3.0),
            (105, 50_000, 0.25), (106, 50_000, 0.025), (107, 50_000, 1.0), (108, 50_000, 3.0),
            (109, 300_000, 1.0), (110, 300_000, 0.025), (111, 1_000_000, 3.0), (112, 3_000_000, 0.25),
            (113, 5_000_000, 0.25)]
        for seed, e, ratio in sets:
            seed += self.args.seed_offset
            name = f"01-plain-{e // 1000}K-c{int(ratio * 1000)}-s{seed}"
            t0 = time.time()
            d = self.gen(name, seed, e, e * ratio)
            self.evaluate(name, d, e, time.time() - t0, note=f"seed {seed}, clicks {ratio:.1%}", mutated=False)

    def _ids(self, num, label, fn, rate, crate, seed, e=300_000):
        name = f"{num:02d}-{label}"
        t0 = time.time()
        d = self.gen(name, seed, e, e // 4)
        n = mutate_ids(DS(d), random.Random(seed), fn, rate, crate, salt=label.encode())
        self.evaluate(name, d, e, time.time() - t0, note=f"{n} rows changed")

    def c02_upper(self):
        self._ids(2, "upper", f_upper, 0.01, 0.005, 202)

    def c03_pad(self):
        self._ids(3, "space-pad", f_pad, 0.005, 0.0025, 203)

    def c04_lead0(self):
        self._ids(4, "leading-zero", f_lead0, 0.005, 0.0025, 204)

    def c05_long(self):
        self._ids(5, "long-ids", f_long, 0.005, 0.0025, 205)

    def c06_dups(self):
        name, e, t0 = "06-dup-copies", 300_000, time.time()
        d = self.gen(name, 206, e, e // 4)
        st = add_dups(DS(d), random.Random(206), 0.02)
        self.evaluate(name, d, e, time.time() - t0, note=", ".join(f"{k} {v}" for k, v in st.items()))

    def c07_far(self):
        for name, seed, rate, crate, skip in (("07-far-event-ids", 207, 0.005, 0.0025, False),
                                              ("07b-far-ids-heavy", 217, 0.3, 0.0, True)):
            e, t0 = 300_000, time.time()
            d = self.gen(name, seed, e, e // 4)
            n = far_event_ids(DS(d), random.Random(seed), rate, crate, skip_first_file=skip)
            self.evaluate(name, d, e, time.time() - t0, note=f"{n} rows changed")

    def c08_links(self):
        name, e, t0 = "08-link-mutations", 300_000, time.time()
        d = self.gen(name, 208, e, e // 4)
        st = link_mutations(DS(d), random.Random(208))
        self.evaluate(name, d, e, time.time() - t0, note=f"{st}")

    def c09_backwards(self):
        name, e, t0 = "09-backwards-times", 300_000, time.time()
        d = self.gen(name, 209, e, e // 4)
        ke, _ = event_keys(DS(d), e)
        st = backwards_times(DS(d), random.Random(209), ke, 0.01)
        self.evaluate(name, d, e, time.time() - t0, note=f"{st}")

    def c10_two_sets(self):
        for variant in ("b-last", "b-first", "interleaved", "shuffled"):
            name, e, t0 = f"10-two-sets-{variant}", 150_000, time.time()
            a = self.gen(name, 210, e, e // 4)
            b = self.gen(name + "-B", 211, e, e // 4)
            st = concat_two(DS(a), DS(b), variant)
            shutil.rmtree(b, ignore_errors=True)
            self.evaluate(name, a, 2 * e, time.time() - t0, note=f"{st}")

    def c11_wrong_keys(self):
        # 100K events are under SS_DENSE=1's 8 MB floor (d1 falls back); 130K are over it,
        # so there SS_DENSE=1 also runs the dense path with the wrong keys
        for name, e in (("11-wrong-keys", 100_000), ("11b-wrong-keys-130K", 130_000)):
            t0 = time.time()
            d = self.gen(name, 212, e, e // 4)
            self.evaluate(name, d, e, time.time() - t0, dense_env={"SS_DENSE_KEYS": "123456789a,abcdef0123"},
                          note="SS_DENSE_KEYS=123456789a,abcdef0123 on every dense mode", mutated=False)

    def c12_fixtures(self):
        t0 = time.time()
        cases = sorted(p.name for p in FIXTURES.iterdir() if (p / "input").is_dir())
        bad = []
        n_vs_gen = n_vs_exp = 0
        for c in cases:
            data = FIXTURES / c / "input"
            out = self.scratch / "out" / f"fx-{c}"
            g = self.run_bin(data, out / "general", {"SS_DENSE": "0"})
            ok_all = g[0] == 0
            for label, env in self.modes:
                rc, err, _ = self.run_bin(data, out / label, env)
                if rc != 0:
                    bad.append(f"{c}:{label} exit {rc}")
                    ok_all = False
                    continue
                ok1, txt1 = self.cmp(out / "general", out / label)
                ok2, txt2 = self.cmp(FIXTURES / c / "expected", out / label)
                if label == "force":
                    n_vs_gen += ok1
                    n_vs_exp += ok2
                if not ok1:
                    bad.append(f"{c}:{label} vs general\n{txt1}")
                if not ok2:
                    bad.append(f"{c}:{label} vs expected\n{txt2}")
                ok_all = ok_all and ok1 and ok2
            if ok_all:
                shutil.rmtree(out, ignore_errors=True)
        line = (f"{'12-fixtures':<26} {len(cases)} cases: force=MATCH vs general {n_vs_gen}/{len(cases)}, "
                f"vs expected {n_vs_exp}/{len(cases)}; modes {','.join(m for m, _ in self.modes)}: "
                f"{'all MATCH' if not bad else str(len(bad)) + ' failures'}  | {time.time() - t0:.1f}s")
        print(line, flush=True)
        self.results.append(line)
        if bad:
            self.failed.append("12-fixtures")
            for b in bad:
                print("    " + b.replace("\n", "\n    "))

    def c13_mixed(self):
        name, e, t0 = "13-mixed-2M", 2_000_000, time.time()
        d = self.gen(name, 213, e, e // 4)
        rng = random.Random(213)
        ds = DS(d)
        notes = [f"dups {sum(add_dups(ds, rng, 0.02).values())}", f"links {link_mutations(ds, rng)}"]
        for label, fn, rate, crate in (("upper", f_upper, 0.01, 0.005), ("pad", f_pad, 0.005, 0.0025),
                                       ("lead0", f_lead0, 0.005, 0.0025)):
            notes.append(f"{label} {mutate_ids(ds, rng, fn, rate, crate, salt=label.encode())}")
        self.evaluate(name, d, e, time.time() - t0, note="; ".join(notes))

    def c14_sorted(self):
        for variant in ("asc", "desc3"):
            name, e, t0 = f"14-sorted-by-number-{variant}", 300_000, time.time()
            d = self.gen(name, 214, e, e // 4)
            ke, _ = event_keys(DS(d), e)
            st = sort_by_number(DS(d), ke, variant)
            self.evaluate(name, d, e, time.time() - t0, note=f"{st}")

    def c15_canon_edges(self):
        name, e, t0 = "15-canonical-edges", 300_000, time.time()
        d = self.gen(name, 215, e, e // 4)
        ke, ku = event_keys(DS(d), e)
        st = canon_edges(DS(d), random.Random(215), ke, ku, e)
        self.evaluate(name, d, e, time.time() - t0, note=f"{st}")

    def c16_ingest_clamp(self):
        name, e, t0 = "16-ingest-far-past", 50_000, time.time()
        d = self.gen(name, 216, e, e // 4)
        st = ingest_clamp(DS(d), random.Random(216))
        self.evaluate(name, d, e, time.time() - t0,
                      note=f"{st}; fails while main.rs ingest_rel clamps ingest_ts > 2^31 s before window_start "
                           "to one value: such copies tie and the first one read wins, in both pipelines")

    def c17_crlf_files(self):
        name, e, t0 = "17-crlf-many-files", 300_000, time.time()
        d = self.gen(name, 218, e, e // 4)
        st = crlf_and_files(DS(d), random.Random(218))
        self.evaluate(name, d, e, time.time() - t0, note=f"{st}")

    def c18_quotes(self):
        name, e, t0 = "18-quote-everything", 300_000, time.time()
        d = self.gen(name, 219, e, e // 4)
        quote_all(DS(d), random.Random(219))
        self.evaluate(name, d, e, time.time() - t0)

    def c19_long_window(self):
        name, e, t0 = "19-window-24-days", 300_000, time.time()
        d = self.gen(name, 220, e, e // 4)
        st = long_window(DS(d))
        self.evaluate(name, d, e, time.time() - t0, note=f"{st}")


CASES = {
    1: ("plain generated sets", Harness.c01_plain),
    2: ("upper-cased IDs", Harness.c02_upper),
    3: ("space-padded IDs", Harness.c03_pad),
    4: ("leading zeros", Harness.c04_lead0),
    5: ("long IDs", Harness.c05_long),
    6: ("duplicate copies", Harness.c06_dups),
    7: ("event IDs outside the dense range (+ heavy variant)", Harness.c07_far),
    8: ("link mutations", Harness.c08_links),
    9: ("times backwards in event-number order", Harness.c09_backwards),
    10: ("two generated sets, two keys", Harness.c10_two_sets),
    11: ("wrong keys (SS_DENSE_KEYS)", Harness.c11_wrong_keys),
    12: ("fixtures", Harness.c12_fixtures),
    13: ("mixed 2+3+4+6+8 on 2M events", Harness.c13_mixed),
    14: ("rows sorted by event number", Harness.c14_sorted),
    15: ("canonical IDs on slot / bound edges", Harness.c15_canon_edges),
    16: ("ingest_ts far in the past (clamping)", Harness.c16_ingest_clamp),
    17: ("CRLF lines, many files, no trailing newline", Harness.c17_crlf_files),
    18: ("every field quoted", Harness.c18_quotes),
    19: ("24-day window (ts offsets near 2^31 ms)", Harness.c19_long_window),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cases", default="")
    ap.add_argument("--modes", default="d1,force,capdiv,nc1")
    ap.add_argument("--no-ref", dest="ref", action="store_false")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--scratch", default=str(SCRATCH))
    ap.add_argument("--bin", default=str(BIN))
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--gen", default=str(GEN))
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE for every run (repeatable)")
    ap.add_argument("--ncpu", type=int, default=2)
    ap.add_argument("--flock", default="", help="lock file for flock around each heavy run (shared VM)")
    ap.add_argument("--seed-offset", type=int, default=0, help="added to case 1's seeds (new sets)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for k, (desc, _) in CASES.items():
            print(f"{k:3d}  {desc}")
        return
    if args.cases:
        ids = [int(x) for x in args.cases.split(",")]
    elif args.smoke:
        ids = [1, 2, 12]
    else:
        ids = list(CASES)
    h = Harness(args)
    t0 = time.time()
    print(f"binary {h.bin} | scratch {h.scratch} | modes {args.modes} | ref {'on' if args.ref else 'off'} "
          f"| env {args.env} | ncpu {args.ncpu}", flush=True)
    for i in ids:
        try:
            CASES[i][1](h)
        except Exception as e:  # noqa: BLE001  (a harness error must not hide later cases)
            import traceback
            traceback.print_exc()
            print(f"case {i}: HARNESS ERROR {e!r}", flush=True)
            h.failed.append(f"case {i} (harness error)")
    print(f"\n{len(h.results)} lines, {len(h.failed)} failing: {', '.join(h.failed) or 'none'} "
          f"| total {time.time() - t0:.0f}s")
    sys.exit(1 if h.failed else 0)


if __name__ == "__main__":
    main()
