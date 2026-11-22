#!/usr/bin/env python3
"""
Differential / adversarial harness for scorer version 5 (long IDs).

v5 data: 1 in 8 distinct user_id / event_id / click_id values is prefix + 32
lowercase hex digits (33 bytes), the rest prefix + 1..8 base-36 digits. The
binary keys exactly-long-shaped IDs (one byte of -0-9A-Z_a-z + exactly 32 of
[0-9a-f]) by a seeded 60-bit hash, verifies all (key, value) pairs and reruns
on a collision (finally with long IDs in the exact interner); every other shape
goes to the interner. This harness builds datasets that aim at that: near-twin
long IDs, the same hex under other prefixes, almost-long shapes, quoting, links
that make long IDs names, long-ID duplicates, click ties, long / odd-length
lines, and runs the binary in every switch mode below against its default run
(tools/compare.py), plus reference/reference.py as a third opinion on sets with
<= 300K events. One line per case; exits 1 on any mismatch or crash and keeps
the mismatching dataset (its path is printed).

Modes (columns; each compared with the default run "base", NCPU=--ncpu):
  off     SS_LONGOFF=1                    long IDs through the interner (old path)
  b6      SS_LONGBITS=6                   forced collisions: rerun path ([rN] = N reruns)
  b12     SS_LONGBITS=12                  collisions on all but tiny sets
  b16     SS_LONGBITS=16                  few bits, no collision on the 1K-3K sets (no [rN])
  nowide  SS_WIDECUT=0                    no 128-byte splitters
  sse     SS_NOAVX2=1
  nc1     NCPU=1
  mixA    SS_NOAVX2=1 SS_WIDECUT=0 SS_LONGBITS=6 NCPU=3
  mixB    SS_LONGBITS=12 SS_WIDECUT=0 NCPU=1
  mixC    SS_LONGOFF=1 SS_NOAVX2=1 NCPU=1
  mixD    SS_LONGBITS=6 SS_NOAVX2=1 NCPU=1
  quiet   SS_VERBOSE unset (every other run has SS_VERBOSE=1)

Usage (venv python with polars; dense_adv.py next to this file or in the tree):
  v5_adv.py --tree SS_DIR --bin BIN --gen GEN --scratch DIR [--flock /tmp/bench.lock]
            [--cases 1,2,3] [--sizes 1000,3000,200000] [--modes off,b6] [--no-ref] [--keep] [--list]
"""
import argparse
import contextlib
import fcntl
import os
import random
import re
import shutil
import subprocess
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _tree():
    for i, a in enumerate(sys.argv):
        if a == "--tree" and i + 1 < len(sys.argv):
            return Path(sys.argv[i + 1]).resolve()
        if a.startswith("--tree="):
            return Path(a.split("=", 1)[1]).resolve()
    return HERE.parent.parent


SS = _tree()
sys.path.insert(0, str(SS / "tools"))
import compare as _compare  # noqa: E402,F401  (dense_adv imports from it)

sys.path.insert(0, str(SS / "tools" / "a3"))
sys.path.insert(0, str(HERE))
import dense_adv as D  # noqa: E402

DS, split, join, rand_q, CsvFile = D.DS, D.split, D.join, D.rand_q, D.CsvFile
parse_ts, fmt_f1, NCOLS, ID_COLS = D.parse_ts, D.fmt_f1, D.NCOLS, D.ID_COLS

ALL_MODES = {
    "off": {"SS_LONGOFF": "1"},
    "b6": {"SS_LONGBITS": "6"},
    "b12": {"SS_LONGBITS": "12"},
    "b16": {"SS_LONGBITS": "16"},
    "nowide": {"SS_WIDECUT": "0"},
    "sse": {"SS_NOAVX2": "1"},
    "nc1": {"NCPU": "1"},
    "mixA": {"SS_NOAVX2": "1", "SS_WIDECUT": "0", "SS_LONGBITS": "6", "NCPU": "3"},
    "mixB": {"SS_LONGBITS": "12", "SS_WIDECUT": "0", "NCPU": "1"},
    "mixC": {"SS_LONGOFF": "1", "SS_NOAVX2": "1", "NCPU": "1"},
    "mixD": {"SS_LONGBITS": "6", "SS_NOAVX2": "1", "NCPU": "1"},
    "quiet": {"SS_VERBOSE": None},
}
REF_MAX_EVENTS = 300_000

HEXD = b"0123456789abcdef"
LONG_RE = re.compile(rb"\A[-0-9A-Z_a-z][0-9a-f]{32}\Z")


def is_long(v):
    return LONG_RE.match(v) is not None


def rhex(r, n=32):
    return bytes(r.choice(HEXD) for _ in range(n))


def crc_r(*parts):
    return random.Random(zlib.crc32(b"|".join(parts)))


def flip(h, pos, r):
    return h[:pos] + bytes([r.choice([d for d in HEXD if d != h[pos]])]) + h[pos + 1:]


POS = {"last": [31], "first": [0], "b15": [15], "b16": [16], "first16": list(range(16)), "last16": list(range(16, 32)),
       "any": list(range(32))}


def twin(v, r, style="any"):
    """v (long) with one hex digit changed at a position of `style`; first16 / last16
    change 1..16 digits inside that half."""
    p, h = v[:1], v[1:]
    if style in ("first16", "last16"):
        hb = bytearray(h)
        for pos in r.sample(POS[style], r.randint(1, 16)):
            hb[pos] = r.choice([d for d in HEXD if d != hb[pos]])
        return p + bytes(hb)
    return p + flip(h, r.choice(POS[style]), r)


# ------------------------------------------------------------------ dataset text helpers
def rewrite(ds, kind, fn):
    """fn(vals, q) -> None (unchanged) or a list of (vals, q) to write in its place."""
    n, count = NCOLS[kind], 0
    for f in ds.load(kind):
        out, ch = [], False
        for line in f.rows:
            sp = split(line)
            if not sp or len(sp[0]) != n:
                out.append(line)
                continue
            r = fn(sp[0], sp[1])
            if r is None:
                out.append(line)
                continue
            ch = True
            count += 1
            out.extend(join(v, q) for v, q in r)
        if ch:
            f.rows = out
            f.save()
    return count


def scatter(ds, kind, lines, rng):
    """Inserts lines at random positions of random files of `kind`."""
    if not lines:
        return 0
    files = ds.load(kind)
    per = defaultdict(list)
    for l in lines:
        per[rng.randrange(len(files))].append(l)
    for i, ls in per.items():
        f = files[i]
        rows = f.rows
        pos = sorted(rng.randrange(len(rows) + 1) for _ in ls)
        out, prev = [], 0
        for p, l in zip(pos, ls):
            out.extend(rows[prev:p])
            out.append(l)
            prev = p
        out.extend(rows[prev:])
        f.rows = out
        f.save()
    return len(lines)


def add_links(ds, pairs, rng, p=0.3):
    return scatter(ds, "links", [join([a, b], rand_q(rng, 2, p)) for a, b in pairs], rng)


class Index:
    """Every ID per column (events' event_id, clicks' click_id, user IDs anywhere)
    and the valid campaigns. fresh_*: registers a new ID, None if it exists."""

    def __init__(self, ds):
        self.ev, self.cl, self.us = set(), set(), set()
        for _, _, v, _ in ds.rows("events"):
            self.ev.add(v[0])
            self.us.add(v[1])
        for _, _, v, _ in ds.rows("clicks"):
            self.cl.add(v[0])
            self.us.add(v[1])
        for _, _, v, _ in ds.rows("links"):
            self.us.add(v[0])
            self.us.add(v[1])
        for s in (self.ev, self.cl, self.us):
            s.discard(b"")
        lines = (ds.root / "campaigns.csv").read_bytes().split(b"\n")[1:]
        self.camps = sorted(l.split(b",")[0].strip(b'"') for l in lines if l)

    @staticmethod
    def _fresh(s, v):
        if v is None or v in s or not v:
            return None
        s.add(v)
        return v

    def fu(self, v):
        return self._fresh(self.us, v)

    def fe(self, v):
        return self._fresh(self.ev, v)

    def fk(self, v):
        return self._fresh(self.cl, v)

    def longs(self, which):
        return sorted(x for x in which if is_long(x))


def rename(ds, umap=None, emap=None, kmap=None):
    """Consistent renames: umap on every user field, emap on event_id, kmap on click_id."""
    umap, emap, kmap = umap or {}, emap or {}, kmap or {}
    total = 0
    for kind, first in (("events", emap), ("clicks", kmap), ("links", umap)):
        def fn(vals, q, kind=kind, first=first):
            hit = False
            for c in range(len(vals) if kind == "links" else 2):
                mp = umap if (kind == "links" or c == 1) else first
                nv = mp.get(vals[c])
                if nv is not None:
                    vals[c] = nv
                    hit = True
            return [(vals, q)] if hit else None
        total += rewrite(ds, kind, fn)
    return total


def reassign(ds, tw, salt, share=0.5, kinds=("events", "clicks")):
    """Rows whose user is in tw get user tw[user] for `share` of the event / click IDs
    (decided per (key, user): copies that must be identical stay identical)."""
    cut = int(share * 1000)

    def fn(vals, q):
        t = tw.get(vals[1])
        if t is None or zlib.crc32(salt + vals[0] + b"|" + vals[1]) % 1000 >= cut:
            return None
        if isinstance(t, list):
            t = t[zlib.crc32(b"w" + vals[0]) % len(t)]
        vals[1] = t
        return [(vals, q)]
    return sum(rewrite(ds, k, fn) for k in kinds)


def first_rows(ds, kind, pred=None):
    """The first well-formed row of every distinct key (event_id / click_id)."""
    seen = set()
    for _, _, v, q in ds.rows(kind):
        if v[0] and v[0] not in seen:
            seen.add(v[0])
            if pred is None or pred(v):
                yield v, q


def same_instant(raw, r):
    t = parse_ts(raw)
    if t is None:
        return raw
    x = r.random()
    if x < 0.4:
        return str(t).encode()
    if x < 0.6 and t % 1000 == 0:
        return fmt_f1(t)
    if x < 0.7:
        return str(t).encode().rjust(16, b"0")
    return raw


def interleave_ids(s, r):
    """Long IDs around a short one ('u' + base-36 digits): s is a prefix of the
    first (so s sorts before it), the second sorts just below s."""
    p, d = s[:1], s[1:]
    if not d or len(d) > 31 or any(c not in HEXD for c in d):
        return []
    out = [p + d + rhex(r, 32 - len(d)), p + d + b"0" * (32 - len(d))]
    nz = [i for i, c in enumerate(d) if c != 0x30]
    if nz:
        i = nz[-1]
        c = d[i]
        dc = c - 1 if c != 0x61 else 0x39
        out.append(p + d[:i] + bytes([dc]) + b"f" * (32 - i - 1))
    return out


# ------------------------------------------------------------------ mutations (cases)
def cluster_rename(ds, idx, rng, frac=0.6):
    """Case 2a: `frac` of the long user / event / click IDs renamed (everywhere) into
    clusters of 2..12 IDs that differ only in the last digit, only digit 0, 15 or
    16, only in the first 16 or only in the last 16 digits."""
    maps, stats = {}, defaultdict(int)
    for kind, pool in (("u", idx.us), ("e", idx.ev), ("k", idx.cl)):
        longs = idx.longs(pool)
        rng.shuffle(longs)
        longs = longs[:int(len(longs) * frac)]
        mp, i = {}, 0
        while i < len(longs):
            size = rng.randint(2, 12)
            grp = longs[i:i + size]
            i += size
            style = rng.choice(["last", "first16", "last16", "b15", "b16", "first"])
            base = rhex(rng)
            for old in grp:
                for _ in range(64):
                    if style == "last":
                        h = base[:31] + bytes([rng.choice(HEXD)])
                    elif style == "first16":
                        h = rhex(rng, 16) + base[16:]
                    elif style == "last16":
                        h = base[:16] + rhex(rng, 16)
                    else:
                        pos = {"b15": 15, "b16": 16, "first": 0}[style]
                        h = base[:pos] + bytes([rng.choice(HEXD)]) + base[pos + 1:]
                    new = old[:1] + h
                    if new not in pool:
                        pool.add(new)
                        mp[old] = new
                        stats[style] += 1
                        break
        maps[kind] = mp
    n = rename(ds, maps["u"], maps["e"], maps["k"])
    return {"rows": n, **{k: len(v) for k, v in maps.items()}, **stats}


def user_twins(ds, idx, rng, frac=0.06, salt=b"ut"):
    """Case 2b: near-twin user IDs as separate IDs: half of the rows of a long user
    move to a twin (one digit changed: last / first / 15 / 16 / any / several in a
    half), linked to the original or not; link-only twins (some smaller)."""
    longs = idx.longs(idx.us)
    tw, links, linked = {}, [], 0
    for u in rng.sample(longs, int(len(longs) * frac)):
        t = idx.fu(twin(u, rng, rng.choice(list(POS))))
        if t is None:
            continue
        tw[u] = t
        if rng.random() < 0.5:
            links.append((u, t) if rng.random() < 0.5 else (t, u))
            linked += 1
    lo = 0
    for u in rng.sample(longs, int(len(longs) * frac / 2)):
        t = twin(u, rng, rng.choice(["last", "first", "b15", "b16"]))
        if rng.random() < 0.5 and t > u and u[-1] != 0x30:  # make it the smaller one (last digit - 1)
            t = u[:-1] + bytes([0x39 if u[-1] == 0x61 else u[-1] - 1])
        t = idx.fu(t)
        if t is not None:
            links.append((u, t))
            lo += t < u
    n = reassign(ds, tw, salt)
    add_links(ds, links, rng)
    return {"twins": len(tw), "linked": linked, "link_only": len(links) - linked, "link_only_smaller": lo, "rows": n}


def event_twins(ds, idx, rng, frac=0.03, styles=("last", "any", "b15", "b16"), bad=0.0):
    """New events whose event_id is a near twin of a long one (same other fields;
    `bad` of them with a bad ts or type and an earlier ingest)."""
    lines = []
    for v, q in first_rows(ds, "events", lambda v: is_long(v[0])):
        if rng.random() >= frac:
            continue
        t = idx.fe(twin(v[0], rng, rng.choice(styles)))
        if t is None:
            continue
        nv = list(v)
        nv[0] = t
        if rng.random() < bad:
            if rng.random() < 0.5:
                nv[2] = rng.choice(D.BAD_TS)
            else:
                nv[4] = rng.choice([b"Purchase", b"cart ", b"VIEW"])
        lines.append(join(nv, rand_q(rng, 7)))
    return scatter(ds, "events", lines, rng)


def click_twins(ds, idx, rng, frac=0.05, styles=("last", "any", "b15", "b16")):
    """New clicks whose click_id is a near twin of a long one: same user and ts
    (a tie), another campaign (so the tie-break decides the credit)."""
    lines = []
    for v, q in first_rows(ds, "clicks", lambda v: is_long(v[0]) and v[1]):
        if rng.random() >= frac:
            continue
        t = idx.fk(twin(v[0], rng, rng.choice(styles)))
        if t is None:
            continue
        nv = list(v)
        nv[0], nv[2], nv[3] = t, rng.choice(idx.camps), same_instant(v[3], rng)
        lines.append(join(nv, rand_q(rng, 4)))
    return scatter(ds, "clicks", lines, rng)


PREFIXES = [b"e", b"k", b"U", b"-", b"_", b"0", b"9", b"A", b"z", b"E", b"K", b"a", b"f"]


def prefix_variants(ds, idx, rng):
    """Case 3: the same 32 hex digits under other prefixes, as user, event and click
    IDs, including strings equal to IDs of another column."""
    st = {}
    longs_u = idx.longs(idx.us)
    tw, links = {}, []
    for u in rng.sample(longs_u, len(longs_u) // 16):
        vs = []
        for p in rng.sample(PREFIXES, rng.randint(1, 3)):
            t = idx.fu(p + u[1:])
            if t:
                vs.append(t)
                if rng.random() < 0.5:
                    links.append((u, t) if rng.random() < 0.5 else (t, u))
        if vs:
            tw[u] = vs
    for u in rng.sample(longs_u, len(longs_u) // 32):  # link-only, mostly smaller
        t = idx.fu(rng.choice([b"-", b"0", b"A", b"U", b"e", b"z"]) + u[1:])
        if t:
            links.append((u, t))
    # user IDs equal to existing long event / click ID strings
    longs_e, longs_k = idx.longs(idx.ev), idx.longs(idx.cl)
    for u in rng.sample(longs_u, len(longs_u) // 32):
        t = idx.fu(rng.choice(longs_e + longs_k))
        if t:
            tw.setdefault(u, []).append(t)
            if rng.random() < 0.3:
                links.append((u, t))
    st["users"] = reassign(ds, tw, b"pv", 0.4)
    st["links"] = add_links(ds, links, rng)
    # events: same hex under other prefixes, and user-ID strings as event IDs
    lines = []
    for v, q in first_rows(ds, "events", lambda v: is_long(v[0])):
        if rng.random() >= 0.03:
            continue
        cands = [p + v[0][1:] for p in (b"u", b"k", b"-", b"E", b"0", b"_")]
        if longs_u:
            cands.append(rng.choice(longs_u))
        t = idx.fe(rng.choice(cands))
        if t:
            nv = list(v)
            nv[0] = t
            lines.append(join(nv, rand_q(rng, 7)))
    st["events"] = scatter(ds, "events", lines, rng)
    # clicks: ties between k<H> and u<H> / e<H> / -<H> and user / event ID strings
    lines = []
    for v, q in first_rows(ds, "clicks", lambda v: is_long(v[0]) and v[1]):
        if rng.random() >= 0.05:
            continue
        cands = [p + v[0][1:] for p in (b"u", b"e", b"K", b"-", b"0", b"z")]
        if longs_e:
            cands.append(rng.choice(longs_e))
        for c in rng.sample(cands, 2):
            t = idx.fk(c)
            if t:
                nv = list(v)
                nv[0], nv[2], nv[3] = t, rng.choice(idx.camps), same_instant(v[3], rng)
                lines.append(join(nv, rand_q(rng, 4)))
    st["clicks"] = scatter(ds, "clicks", lines, rng)
    return st


def shape_variant(v, r):
    """An almost-long shape of the long ID v (all must take the exact interner path),
    plus 33 hex digits with no prefix (a hex digit as the prefix: long shape)."""
    p, h = v[:1], v[1:]
    k = r.randrange(17)
    if k == 0:
        u = p + h.upper()
        return u if u != v else p + h[:31]
    if k == 1:
        letters = [i for i, c in enumerate(h) if c >= 0x61]
        if not letters:
            return p + h[:31]
        i = r.choice(letters)
        return p + h[:i] + h[i:i + 1].upper() + h[i + 1:]
    if k == 2:
        return p + h[:31]
    if k == 3:
        return p + h + bytes([r.choice(HEXD)])
    if k == 4:
        return p + h + rhex(r, 2)
    if k == 5:
        return p + h + rhex(r, 32)
    if k == 6:
        i = r.randrange(32)
        return p + h[:i] + r.choice([b"g", b"G", b"x", b"z", b":", b"-", b"_", b"/", b"F"]) + h[i + 1:]
    if k == 7:
        return b" " + v
    if k == 8:
        return v + b" "
    if k == 9:
        return b" " + h
    if k == 10:
        return "é".encode() + h
    if k == 11:
        return r.choice([b".", b"~", b"+", b"!", b"#", b"*", b"@", b"\t", b"\xc3\xa9"[:1] + b"\xa9"]) + h
    if k == 12:
        return h
    if k == 13:
        return h + bytes([r.choice(HEXD)])
    if k == 14:
        return p + p + h
    if k == 15:
        return p + h[:16] + b" " + h[16:]
    return p + h[:31] + b" "


def shapes(ds, idx, rng, crate=0.15, frac=0.06):
    """Case 4: (a) `crate` of the distinct long IDs of every column renamed to an
    almost-long shape everywhere; (b) half the rows of `frac` of the long users
    moved to such a variant of their ID, linked or not."""
    maps = {}
    for kind, pool in (("u", idx.us), ("e", idx.ev), ("k", idx.cl)):
        mp = {}
        for x in idx.longs(pool):
            if rng.random() < crate:
                t = idx._fresh(pool, shape_variant(x, rng))
                if t:
                    mp[x] = t
        maps[kind] = mp
    n = rename(ds, maps["u"], maps["e"], maps["k"])
    longs = idx.longs(idx.us)
    tw, links = {}, []
    for u in rng.sample(longs, int(len(longs) * frac)):
        t = idx.fu(shape_variant(u, rng))
        if t:
            tw[u] = t
            if rng.random() < 0.5:
                links.append((u, t) if rng.random() < 0.5 else (t, u))
    m = reassign(ds, tw, b"sh")
    add_links(ds, links, rng)
    return {"renamed": {k: len(v) for k, v in maps.items()}, "rename_rows": n, "twins": len(tw), "twin_rows": m,
            "links": len(links)}


def requote(ds, idx, rng, p=0.5, allq=0.15):
    """Case 5: every field holding a long ID quoted with p = 0.5 per occurrence
    (copies of one ID quoted and unquoted), `allq` of those lines fully quoted;
    extra copies of long-ID clicks and events that differ only in quoting."""
    def fn_for(kind):
        n = NCOLS[kind]

        def fn(vals, q):
            cols = [c for c, _ in ID_COLS[kind] if is_long(vals[c])]
            if not cols:
                return None
            if rng.random() < allq:
                return [(vals, [True] * n)]
            q = list(q)
            for c in cols:
                q[c] = rng.random() < p
            return [(vals, q)]
        return fn
    n = sum(rewrite(ds, k, fn_for(k)) for k in ("events", "clicks", "links"))
    ce = [join(v, [not x for x in q]) for v, q in first_rows(ds, "clicks", lambda v: is_long(v[0]))
          if rng.random() < 0.08]
    ee = [join(v, [not x for x in q]) for v, q in first_rows(ds, "events", lambda v: is_long(v[0]) or is_long(v[1]))
          if rng.random() < 0.05]
    return {"rows": n, "click_copies": scatter(ds, "clicks", ce, rng), "event_copies": scatter(ds, "events", ee, rng)}


SPECIALS = [b"u" + b"0" * 32, b"u" + b"0" * 31 + b"1", b"u00", b"u0", b"u1", b"u" + b"f" * 32, b"-" + b"0" * 32,
            b"_" + b"f" * 32, b"u0" + b"f" * 31, b"u1" + b"0" * 31, b"0" * 33, b"-", b"u", b"U" + b"0" * 32,
            b"e" + b"0" * 32]


def long_links(ds, idx, rng):
    """Case 6: long IDs in links that merge and split people and become names."""
    users = sorted(idx.us)
    longs = idx.longs(idx.us)
    shorts = [u for u in users if not is_long(u) and len(u) >= 2]
    new, st = [], defaultdict(int)
    for u in rng.sample(users, max(1, len(users) // 50)):  # link-only long IDs smaller than everything
        t = idx.fu(rng.choice([b"-", b"0", b"A", b"U", b"_", b"9"]) + rhex(rng))
        if t:
            new.append((u, t) if rng.random() < 0.5 else (t, u))
            st["smaller"] += 1
    cands = [s for s in shorts if s[:1] == b"u" and all(c in HEXD for c in s[1:])]
    for s in rng.sample(cands, min(len(cands), max(1, len(cands) // 10))):  # long vs short in byte order
        for t in interleave_ids(s, rng):
            if rng.random() < 0.6 and idx.fu(t):
                other = s if rng.random() < 0.6 else rng.choice(users)
                new.append((other, t) if rng.random() < 0.5 else (t, other))
                st["interleave"] += 1
    for sp in SPECIALS:  # shared link-only IDs: each merges 1-3 people
        for u in rng.sample(users, rng.randint(1, 3)):
            new.append((u, sp) if rng.random() < 0.5 else (sp, u))
            st["specials"] += 1
    for _ in range(6):  # chains of long IDs merging people
        ch = rng.sample(longs, min(25, len(longs)))
        new += list(zip(ch, ch[1:]))
        st["chain"] += len(ch) - 1
    for _ in range(max(1, len(longs) // 100)):
        a, b = rng.sample(longs, 2)
        new.append((a, b))
        st["pairs"] += 1
    for u in rng.sample(longs, max(1, len(longs) // 100)):  # self links
        new.append((u, u))
    # existing long links repeated reversed, and 3% of long links removed (splits)
    rem = 0
    rev = []
    for f in ds.load("links"):
        out = []
        for line in f.rows:
            sp = split(line)
            if sp and len(sp[0]) == 2 and (is_long(sp[0][0]) or is_long(sp[0][1])):
                if rng.random() < 0.03:
                    rem += 1
                    continue
                if rng.random() < 0.03:
                    rev.append((sp[0][1], sp[0][0]))
            out.append(line)
        if len(out) != len(f.rows):
            f.rows = out
            f.save()
    new += rev
    rng.shuffle(new)
    add_links(ds, new, rng)
    st["removed"], st["reversed"], st["added"] = rem, len(rev), len(new)
    return dict(st)


def add_dups_long(ds, idx, rng, rate=0.15):
    """Case 7: extra copies of `rate` of the long-ID events (dense_adv.add_dups
    restricted to long event IDs, other users mostly long)."""
    files = ds.load("events")
    longs = idx.longs(idx.us)
    pool = sorted(idx.us)
    extra = [[] for _ in files]
    stats = defaultdict(int)
    chosen = []
    for f in files:
        for line in f.rows:
            sp = split(line)
            if sp and len(sp[0]) == 7 and is_long(sp[0][0]) and parse_ts(sp[0][3]) is not None and rng.random() < rate:
                chosen.append(sp[0])
    ids = {v[0] for v in chosen}
    taken = defaultdict(set)
    for _, _, v, _ in ds.rows("events"):
        if v[0] in ids:
            t = parse_ts(v[3])
            if t is not None:
                taken[v[0]].add(t)
    first = {}
    for v in chosen:
        first.setdefault(v[0], v)
    chosen = list(first.values())

    def other_user():
        return rng.choice(longs) if (longs and rng.random() < 0.7) else rng.choice(pool)

    def new_ing(eid, base, lo, hi, sign):
        for _ in range(100):
            t = base + sign * rng.randint(lo, hi) * 1000
            if t not in taken[eid]:
                taken[eid].add(t)
                return fmt_f1(t)
        raise RuntimeError("no free ingest_ts")

    def emit(v):
        extra[rng.randrange(len(files))].append(join(v, rand_q(rng, 7, 0.4)))

    for vals in chosen:
        eid, ing = vals[0], parse_ts(vals[3])
        r = rng.random()
        if r < 0.3:
            stats["later-other-user"] += 1
            for _ in range(rng.randint(1, 3)):
                v = list(vals)
                v[3] = new_ing(eid, ing, 1, 3600, 1)
                v[1] = other_user()
                D.retype(v, rng)
                emit(v)
        elif r < 0.55:
            stats["earlier-other-user-wins"] += 1
            v = list(vals)
            v[3] = new_ing(eid, ing, 1, 3600, -1)
            v[1] = other_user()
            D.retype(v, rng)
            emit(v)
        elif r < 0.7:
            stats["earliest-bad-ts"] += 1
            v = list(vals)
            v[2] = rng.choice(D.BAD_TS)
            v[3] = new_ing(eid, ing, 1, 600, -1)
            emit(v)
            v2 = list(vals)
            v2[3] = new_ing(eid, ing, 1, 600, 1)
            v2[1] = other_user()
            emit(v2)
        elif r < 0.85:
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


def click_ties(ds, idx, rng, frac=0.05, sfrac=0.01):
    """Case 8: ties at one ts between long and short click IDs, long IDs differing
    in the last digit, long vs short in byte order; ties exactly at session starts."""
    lines, st = [], defaultdict(int)
    for v, q in first_rows(ds, "clicks", lambda v: bool(v[1])):
        if rng.random() >= frac:
            continue
        k = v[0]
        ids = [b"k" + rhex(rng), b"k" + D.enc36(rng.randrange(36 ** 6, 36 ** 8))]
        if is_long(k):
            ids += [twin(k, rng, "last"), twin(k, rng, "last"), twin(k, rng, "any"), k[:1] + k[1:9]]
        else:
            ids += interleave_ids(k, rng)
        for c in rng.sample(ids, rng.randint(1, min(4, len(ids)))):
            t = idx.fk(c)
            if t:
                nv = list(v)
                nv[0], nv[2], nv[3] = t, rng.choice(idx.camps), same_instant(v[3], rng)
                lines.append(join(nv, rand_q(rng, 4)))
                st["tie"] += 1
    for v, q in first_rows(ds, "events", lambda v: bool(v[1]) and v[4] in (b"view", b"cart", b"purchase")):
        if rng.random() >= sfrac:
            continue
        t = parse_ts(v[2])
        if t is None:
            continue
        base = b"k" + rhex(rng)
        ids = [base, twin(base, rng, "last"), twin(base, rng, "last"), b"k" + base[1:6], b"k" + base[1:4]]
        for c in rng.sample(ids, rng.randint(2, 4)):
            if idx.fk(c):
                ts = str(t).encode() if (t % 1000 or rng.random() < 0.5) else fmt_f1(t)
                lines.append(join([c, v[1], rng.choice(idx.camps), ts], rand_q(rng, 4)))
                st["at-start"] += 1
    scatter(ds, "clicks", lines, rng)
    return dict(st)


def longify(ds, idx, rng, frac=0.3):
    """`frac` of the short user / event / click IDs renamed to fresh long IDs (same prefix)."""
    maps = {}
    for kind, pool in (("u", idx.us), ("e", idx.ev), ("k", idx.cl)):
        mp = {}
        for x in sorted(pool):
            if not is_long(x) and x[:1] in (b"u", b"e", b"k") and rng.random() < frac:
                t = idx._fresh(pool, x[:1] + rhex(rng))
                if t:
                    mp[x] = t
        maps[kind] = mp
    return {"rows": rename(ds, maps["u"], maps["e"], maps["k"]), **{k: len(v) for k, v in maps.items()}}


TARGETS = [62, 63, 64, 65, 66, 95, 96, 97, 127, 128, 129, 130, 159, 160, 161, 191, 192, 193, 255, 256, 257, 300,
           383, 384, 385, 511, 512, 513, 1023, 1024, 1025, 4096, 4097]


def pad_lines(ds, rng, frac=0.1, pfrac=0.01):
    """Case 9: `frac` of the events / clicks get a ts rewritten as F3 with leading
    zeros so the line is exactly a TARGETS length (bytes before the newline);
    `pfrac` of the events get a 100..400-byte unknown product_id. Values are
    decided per copy identity, so copies that must be identical stay identical."""
    cache, st = {}, defaultdict(int)

    def fn_for(kind, tcol):
        def fn(vals, q):
            ident = D.identity(kind, vals)
            if ident not in cache:
                r = crc_r(b"pad", ident)
                nts = None
                t = parse_ts(vals[tcol])
                if r.random() < frac and t is not None and t >= 0:
                    vv = list(vals)
                    vv[tcol] = b""
                    prod = None
                    if kind == "events" and r.random() < pfrac / frac:
                        prod = vv[5] = b"p" + bytes(r.choice(b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
                                                    for _ in range(r.randint(100, 400)))
                    base = len(join(vv, q))
                    s = str(t).encode()
                    ok = [x for x in TARGETS if x - base >= len(s)]
                    if ok:
                        tg = r.choice(ok[:12])
                        nts = (prod, s.rjust(tg - base, b"0"))
                cache[ident] = nts
            nts = cache[ident]
            if nts is None:
                return None
            if nts[0] is not None:
                vals[5] = nts[0]
                st["long-product"] += 1
            vals[tcol] = nts[1]
            st[kind] += 1
            return [(vals, q)]
        return fn
    rewrite(ds, "events", fn_for("events", 2))
    rewrite(ds, "clicks", fn_for("clicks", 3))
    return dict(st)


def long_link_lines(ds, idx, rng, n=None):
    """Link lines over 128 bytes: 64-hex and space-padded long IDs, quoted."""
    longs = idx.longs(idx.us)
    n = n or max(10, len(longs) // 20)
    pairs = []
    for u in rng.sample(longs, min(n, len(longs))):
        t = idx.fu(rng.choice([u + rhex(rng, 32), u + rhex(rng, 96), b"  " + u + b"  ", u + b" " + rhex(rng, 60)]))
        if t:
            pairs.append((u, t) if rng.random() < 0.5 else (t, u))
    return add_links(ds, pairs, rng, p=0.7)


def backwards_times(ds, rng, frac=0.02):
    """For `frac` of users: their events' ts permuted so that in read order (file,
    row) the times run backwards; all copies of an event shift together (ingest
    with them, so dedup winners and late status stay)."""
    by_eid = defaultdict(list)
    files = ds.load("events")
    for fi, f in enumerate(files):
        for ri, line in enumerate(f.rows):
            sp = split(line)
            if sp and len(sp[0]) == 7 and sp[0][0] and parse_ts(sp[0][3]) is not None:
                v = sp[0]
                by_eid[v[0]].append((fi, ri, v[1], parse_ts(v[2]), parse_ts(v[3])))
    by_user = defaultdict(list)
    for eid, cs in by_eid.items():
        p = min(cs, key=lambda c: c[4])
        if p[3] is not None and p[2]:
            by_user[p[2]].append((p[0], p[1], eid, p[3]))
    users = sorted(u for u, l in by_user.items() if len(l) >= 3)
    touched = 0
    for u in rng.sample(users, max(1, int(len(users) * frac))):
        evs = sorted(by_user[u])
        tss = sorted((e[3] for e in evs), reverse=True)
        for e, nt in zip(evs, tss):
            s = round((nt - e[3]) / 1000) * 1000
            if not s:
                continue
            for fi, ri, _, t, i in by_eid[e[2]]:
                vals, q = split(files[fi].rows[ri])
                if t is not None:
                    vals[2] = fmt_f1(t + s) if (t + s) % 1000 == 0 and rng.random() < 0.5 else str(t + s).encode()
                vals[3] = fmt_f1(i + s)
                files[fi].rows[ri] = join(vals, q)
                touched += 1
    for f in files:
        f.save()
    return {"rows": touched}


# ------------------------------------------------------------------ harness
class Harness:
    def __init__(self, args):
        self.args = args
        self.scratch = Path(args.scratch)
        self.bin = Path(args.bin)
        self.modes = [(m, ALL_MODES[m]) for m in args.modes.split(",") if m]
        self.sizes = [int(x) for x in args.sizes.split(",")]
        self.results, self.failed = [], []
        (self.scratch / "cases").mkdir(parents=True, exist_ok=True)
        (self.scratch / "out").mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def locked(self):
        """--flock FILE: flock(2) on FILE (the lock flock(1) takes) held around each
        batch of generator / binary / reference runs (one batch = the generator, the
        base run plus every mode of one case, or one reference run)."""
        if not self.args.flock:
            yield
            return
        fd = os.open(self.args.flock, os.O_RDONLY | os.O_CREAT, 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def gen(self, name, seed, events, clicks):
        d = self.scratch / "cases" / name
        shutil.rmtree(d, ignore_errors=True)
        with self.locked():
            self._gen(d, seed, events, clicks)
        return d

    def _gen(self, d, seed, events, clicks):
        subprocess.run([str(self.args.gen), "--seed", str(seed), "--events", str(events), "--clicks",
                                      str(int(clicks)), "--output", str(d), "--threads", str(self.args.gthreads)],
                       check=True, capture_output=True)

    def run_bin(self, data, out, extra):
        env = {k: v for k, v in os.environ.items() if not k.startswith("SS_")}
        env.update({"NCPU": str(self.args.ncpu), "SS_VERBOSE": "1"})
        for kv in self.args.env:
            k, _, v = kv.partition("=")
            env[k] = v
        env.update(extra)
        env = {k: v for k, v in env.items() if v is not None}
        shutil.rmtree(out, ignore_errors=True)
        t0 = time.time()
        try:
            p = subprocess.run([str(self.bin), str(data), str(out)], env=env, capture_output=True,
                               text=True, timeout=self.args.timeout)
            rc, err = p.returncode, p.stderr
        except subprocess.TimeoutExpired:
            rc, err = -999, "timeout"
        return rc, err, time.time() - t0

    def evaluate(self, name, data, events, prep, note="", mutated=True):
        out = self.scratch / "out" / name
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        cols, times, fails, details = [], [], [], []
        nlong = None
        if mutated:
            t0 = time.time()
            prob = D.legality(DS(data))
            prep += time.time() - t0
            if prob:
                fails.append(f"ILLEGAL INPUT (harness bug): {prob}")
        with self.locked():
            runs = [self.run_bin(data, out / "base", {})] + [self.run_bin(data, out / l, env) for l, env in self.modes]
        b_rc, b_err, b_t = runs[0]
        times.append(f"base {b_t:.2f}")
        rr = re.search(r"long_key_rerun=(\d+)", b_err)
        cols.append("base" + (f"[r{rr.group(1)}]" if rr else "") + ("=CRASH" if b_rc else ""))
        if b_rc != 0:
            fails.append(f"base exit {b_rc}")
            details.append(b_err[-1500:])
        for (label, env), (rc, err, t) in zip(self.modes, runs[1:]):
            times.append(f"{label} {t:.2f}")
            rr = re.search(r"long_key_rerun=(\d+)", err)
            tag = f"[r{rr.group(1)}]" if rr else ""
            if rc != 0:
                st = f"CRASH({rc})"
                details.append(f"--- {label} stderr:\n{err[-1500:]}")
            elif b_rc != 0:
                st = "n/a"
            else:
                ok, txt = D.Harness.cmp(out / "base", out / label)
                st = "MATCH" if ok else "MISMATCH"
                if not ok:
                    details.append(f"--- base (A) vs {label} (B) env {env}:\n{txt}")
            if st != "MATCH":
                fails.append(f"{label} {st}")
            cols.append(f"{label}={st}{tag}")
        if self.args.ref and events <= REF_MAX_EVENTS and b_rc == 0:
            t0 = time.time()
            with self.locked():
                p = subprocess.run([sys.executable, str(SS / "reference/reference.py"), str(data), str(out / "ref")],
                                   capture_output=True, text=True)
            times.append(f"ref {time.time() - t0:.1f}")
            warns = [l for l in p.stderr.splitlines() if "WARNING" in l]
            if warns:
                fails.append("reference warns (illegal input?): " + " | ".join(w[:200] for w in warns[:3]))
            if p.returncode != 0:
                cols.append("ref=CRASH")
                fails.append("ref crash")
                details.append(p.stderr[-1500:])
            else:
                ok, txt = D.Harness.ref_cmp(out / "ref", out / "base")
                cols.append("ref=" + ("MATCH" if ok else "MISMATCH"))
                if not ok:
                    fails.append("ref MISMATCH (reference vs base)")
                    details.append(f"--- reference (A) vs base (B; 'gen:' lines are the binary's):\n{txt}")
        else:
            cols.append("ref=skip")
        line = f"{name:<30} ev={events:<8} {' '.join(cols)}  | prep {prep:.1f}s, run " + ", ".join(times) + "s"
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

    def per_size(self, num, label, mutate, cratio=0.5, sizes=None):
        """Generates one set per size, applies mutate(ds, idx, rng) -> note, evaluates."""
        for si, e in enumerate(sizes or self.sizes):
            seed = 5000 + num * 100 + si + self.args.seed_offset
            name = f"{num:02d}-{label}-{e // 1000 if e >= 1000 else e}{'K' if e >= 1000 else ''}"
            t0 = time.time()
            d = self.gen(name, seed, e, e * cratio)
            ds = DS(d)
            rng = random.Random(seed)
            note = mutate(ds, Index(ds), rng)
            self.evaluate(name, d, e, time.time() - t0, note=f"seed {seed}; {note}")

    # -------------------------------------------------------------- cases
    def c01_plain(self):
        sets = [(1, 1_000, 0.025), (2, 1_000, 3.0), (3, 5_000, 0.25), (4, 20_000, 1.0), (5, 50_000, 0.025),
                (6, 50_000, 3.0), (7, 100_000, 0.5), (8, 300_000, 0.25), (9, 300_000, 3.0), (10, 1_000_000, 0.025),
                (11, 1_000_000, 3.0), (12, 3_000_000, 0.25), (13, 2_000, 1.0), (14, 200_000, 1.5)]
        if self.args.smoke:
            sets = [(3, 5_000, 0.25), (8, 300_000, 0.25)]
        for k, e, ratio in sets:
            seed = 900 + k + self.args.seed_offset
            name = f"01-plain-{e // 1000}K-c{int(ratio * 1000)}-s{seed}"
            t0 = time.time()
            d = self.gen(name, seed, e, e * ratio)
            self.evaluate(name, d, e, time.time() - t0, note=f"seed {seed}, clicks {ratio:.1%}", mutated=False)

    def c02_twins(self):
        def m(ds, idx, rng):
            a = cluster_rename(ds, idx, rng)
            b = user_twins(ds, idx, rng)
            e = event_twins(ds, idx, rng)
            k = click_twins(ds, idx, rng)
            return f"clusters {a}; user twins {b}; event twins {e}; click twins {k}"
        self.per_size(2, "near-twins", m)

    def c03_prefixes(self):
        self.per_size(3, "same-hex-other-prefix", lambda ds, idx, rng: f"{prefix_variants(ds, idx, rng)}")

    def c04_shapes(self):
        self.per_size(4, "almost-long-shapes", lambda ds, idx, rng: f"{shapes(ds, idx, rng)}")

    def c05_quotes(self):
        self.per_size(5, "quoted-unquoted-long", lambda ds, idx, rng: f"{requote(ds, idx, rng)}")

    def c06_links(self):
        def m(ds, idx, rng):
            a = long_links(ds, idx, rng)
            b = D.link_mutations(ds, rng, mega=min(2000, len(idx.us) // 4))
            return f"long links {a}; generic {b}"
        self.per_size(6, "long-links-names", m)

    def c07_dups(self):
        def m(ds, idx, rng):
            a = add_dups_long(ds, idx, rng)
            b = event_twins(ds, idx, rng, frac=0.05, styles=("last", "b15", "b16"), bad=0.5)
            return f"dups {a}; twin events {b}"
        self.per_size(7, "long-event-dups", m)

    def c08_ties(self):
        def m(ds, idx, rng):
            a = click_ties(ds, idx, rng)
            b = click_twins(ds, idx, rng, frac=0.1, styles=("last",))
            return f"ties {a}; last-digit twins {b}"
        self.per_size(8, "click-ties", m)

    def c09_lines(self):
        def m_len(ds, idx, rng):
            a = longify(ds, idx, rng)
            b = pad_lines(ds, rng)
            c = long_link_lines(ds, idx, rng)
            d = requote(ds, idx, rng, p=0.5, allq=0.3)
            return f"longify {a}; pad {b}; long link lines {c}; quoting {d}"

        def m_crlf(ds, idx, rng):
            a = longify(ds, idx, rng)
            b = pad_lines(ds, rng, frac=0.05)
            c = long_link_lines(ds, idx, rng)
            D.crlf_and_files(ds, rng)
            return f"longify {a}; pad {b}; long link lines {c}; CRLF 3%, 150 small files, no trailing newlines"

        def m_quote(ds, idx, rng):
            a = longify(ds, idx, rng, 0.5)
            b = pad_lines(ds, rng, frac=0.05)
            D.quote_all(ds, rng)
            return f"longify {a}; pad {b}; every field quoted (half the files)"
        self.per_size(9, "exact-lengths", m_len)
        self.per_size(9, "crlf-small-files", m_crlf)
        self.per_size(9, "quote-all", m_quote)

    def c10_all_long(self):
        def m(ds, idx, rng):
            a = longify(ds, idx, rng, 1.0)
            b = cluster_rename(ds, Index(ds), rng, 0.5)
            c = pad_lines(ds, rng, frac=0.05)
            return f"every u/e/k ID long {a}; clusters {b}; pad {c}"
        self.per_size(10, "all-ids-long", m)

    def c11_generic(self):
        for label, fn, rate, crate in (("upper", D.f_upper, 0.01, 0.005), ("space-pad", D.f_pad, 0.005, 0.0025),
                                       ("leading-zero", D.f_lead0, 0.005, 0.0025),
                                       ("over-long", D.f_long, 0.005, 0.0025)):
            self.per_size(11, label, lambda ds, idx, rng, fn=fn, rate=rate, crate=crate, label=label:
                          f"{D.mutate_ids(ds, rng, fn, rate, crate, salt=label.encode())} rows changed",
                          sizes=self.sizes[-1:])

    def c12_link_mut(self):
        self.per_size(12, "link-mutations", lambda ds, idx, rng: f"{D.link_mutations(ds, rng)}", sizes=self.sizes[-1:])

    def c13_backwards(self):
        self.per_size(13, "backwards-times", lambda ds, idx, rng: f"{backwards_times(ds, rng)}", sizes=self.sizes[-1:])

    def c14_two_sets(self):
        for variant in ("b-last", "shuffled"):
            name, e, t0 = f"14-two-sets-{variant}", 150_000, time.time()
            a = self.gen(name, 6401 + self.args.seed_offset, e, e // 2)
            b = self.gen(name + "-B", 6402 + self.args.seed_offset, e, e // 2)
            st = D.concat_two(DS(a), DS(b), variant)
            shutil.rmtree(b, ignore_errors=True)
            self.evaluate(name, a, 2 * e, time.time() - t0, note=f"{st}")

    def c15_mixed(self):
        def m(ds, idx, rng):
            notes = [f"clusters {cluster_rename(ds, idx, rng, 0.3)}", f"twins {user_twins(ds, idx, rng, 0.03)}",
                     f"prefixes {prefix_variants(ds, idx, rng)}", f"shapes {shapes(ds, idx, rng, 0.05, 0.03)}",
                     f"dups {add_dups_long(ds, idx, rng, 0.05)}", f"links {long_links(ds, idx, rng)}",
                     f"ties {click_ties(ds, idx, rng, 0.02, 0.005)}", f"quotes {requote(ds, idx, rng, 0.5, 0.1)}"]
            return "; ".join(notes)
        self.per_size(15, "mixed", m, sizes=[self.args.mixed_events])


CASES = {
    1: ("plain v5 generated sets (14 seeds, 1K..3M, clicks 2.5%..300%)", Harness.c01_plain),
    2: ("near-twin long IDs (renamed clusters, twin users / events / clicks)", Harness.c02_twins),
    3: ("same 32 hex under other prefixes, cross-column equal strings", Harness.c03_prefixes),
    4: ("almost-long shapes (interner path) and 33-hex no-prefix", Harness.c04_shapes),
    5: ("quoted vs unquoted copies of long IDs", Harness.c05_quotes),
    6: ("links with long IDs: merge, split, names, long/short byte order", Harness.c06_links),
    7: ("duplicate events with long event IDs", Harness.c07_dups),
    8: ("click ties: long vs short, last-digit twins, at session starts", Harness.c08_ties),
    9: ("long / exact-length lines, CRLF, small files, quote-all", Harness.c09_lines),
    10: ("every ID long (+ near-twin clusters, padded lines)", Harness.c10_all_long),
    11: ("generic dense_adv ID mutations (upper, pad, lead0, over-long)", Harness.c11_generic),
    12: ("generic link mutations", Harness.c12_link_mut),
    13: ("times backwards in read order", Harness.c13_backwards),
    14: ("two generated sets concatenated (two ID keys)", Harness.c14_two_sets),
    15: ("mixed 2..8 on a large set (no reference)", Harness.c15_mixed),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cases", default="")
    ap.add_argument("--modes", default=",".join(ALL_MODES))
    ap.add_argument("--sizes", default="1000,3000,200000", help="event counts of cases 2..9 (11..13 use the last)")
    ap.add_argument("--mixed-events", type=int, default=2_000_000)
    ap.add_argument("--no-ref", dest="ref", action="store_false")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--tree", default=str(SS), help="shopping-sessions tree (reference/, tools/compare.py)")
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--gthreads", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE for every run (repeatable)")
    ap.add_argument("--ncpu", type=int, default=8, help="NCPU of the base run and of modes without NCPU")
    ap.add_argument("--flock", default="")
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for k, (desc, _) in CASES.items():
            print(f"{k:3d}  {desc}")
        return
    ids = [int(x) for x in args.cases.split(",")] if args.cases else ([1, 2, 9] if args.smoke else list(CASES))
    h = Harness(args)
    t0 = time.time()
    print(f"binary {h.bin} | tree {SS} | scratch {h.scratch} | modes {args.modes} | sizes {args.sizes} "
          f"| ref {'on' if args.ref else 'off'} | env {args.env} | ncpu {args.ncpu}", flush=True)
    for i in ids:
        try:
            CASES[i][1](h)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"case {i}: HARNESS ERROR {e!r}", flush=True)
            h.failed.append(f"case {i} (harness error)")
    print(f"\n{len(h.results)} lines, {len(h.failed)} failing: {', '.join(h.failed) or 'none'} "
          f"| total {time.time() - t0:.0f}s")
    sys.exit(1 if h.failed else 0)


if __name__ == "__main__":
    main()
