#!/usr/bin/env python3
"""Instructions per event line by stage from a cachegrind output (SS_BENCH=line).
Usage: cgstage.py CG_OUT LINES PASSES [STAGEMAP]
STAGEMAP: file:lo-hi=stage,... (first match wins); default map below.
Prints Ir/line per stage, per file, and the top source lines."""
import sys, collections, os, re

CG = sys.argv[1]
N = int(sys.argv[2]) * int(sys.argv[3])
DEFAULT = [
    # k4.rs
    ("k4.rs", 193, 202, "ts"), ("k4.rs", 203, 255, "ts"), ("k4.rs", 256, 311, "ids"),
    ("k4.rs", 312, 481, "split"), ("k4.rs", 494, 511, "ids"), ("k4.rs", 512, 524, "validity"),
    ("k4.rs", 548, 556, "ids"), ("k4.rs", 556, 579, "ts"), ("k4.rs", 579, 589, "validity"),
    ("k4.rs", 589, 598, "ids"), ("k4.rs", 598, 604, "validity"), ("k4.rs", 604, 614, "validity"),
    ("k4.rs", 614, 622, "ids"), ("k4.rs", 622, 624, "sink"), ("k4.rs", 640, 760, "loop"),
    # fastev.rs
    ("fastev.rs", 1, 388, "split"), ("fastev.rs", 389, 475, "loop"),
    # main.rs
    ("main.rs", 593, 622, "validity"), ("main.rs", 623, 664, "validity"), ("main.rs", 3690, 3702, "sink"),
    ("main.rs", 3703, 3760, "loop"),
    ("ts.rs", 1, 99999, "ts"), ("keys.rs", 1, 99999, "ids"), ("scan.rs", 1, 99999, "scanner"),
]
smap = DEFAULT
if len(sys.argv) > 4:
    smap = []
    for it in sys.argv[4].split(','):
        f, rest = it.split(':'); rng, st = rest.split('='); lo, hi = rng.split('-')
        smap.append((f, int(lo), int(hi), st))

def stage(fl, ln):
    b = os.path.basename(fl)
    for f, lo, hi, st in smap:
        if b == f and lo <= ln < hi:
            return st
    return "other:" + b

ev = None; fl = None; fn = None
bystage = collections.Counter(); byfile = collections.Counter(); byline = collections.Counter()
bys_b = collections.Counter()
instr_mode = False
skip = False
for l in open(CG, errors='replace'):
    if skip:
        skip = False; continue
    if l.startswith('calls='):
        skip = True; continue
    if l.startswith(('fi=', 'fe=')): fl = l[3:].strip(); continue
    if l.startswith('positions:'):
        instr_mode = 'instr' in l
        continue
    if l.startswith('events:'):
        ev = l.split()[1:]; iI = ev.index('Ir'); iB = ev.index('Bcm') if 'Bcm' in ev else None; continue
    if l.startswith('fl='): fl = l[3:].strip(); continue
    if l.startswith('fn='): fn = l[3:].strip(); continue
    if not (l[0].isdigit()):
        continue
    if l.startswith('0x') and not instr_mode:
        continue
    p = l.split()
    if instr_mode:
        ln = int(p[1]); vals = p[2:]
    else:
        ln = int(p[0]); vals = p[1:]
    v = [int(x) for x in vals] + [0] * 13
    st = stage(fl, ln)
    bystage[st] += v[iI]
    if iB is not None:
        bys_b[st] += v[iB]
    byfile[os.path.basename(fl)] += v[iI]
    byline[(os.path.basename(fl), ln)] += v[iI]

T = sum(bystage.values())
print(f"TOTAL Ir {T:,}; per line (N={N:,}): {T/N:.1f}")
print("--- by stage: Ir/line  mispredicts/line")
for s, c in bystage.most_common():
    if c / N >= 0.05:
        print(f"{c/N:8.1f} {bys_b[s]/N:6.3f}  {s}")
print("--- by file")
for f, c in byfile.most_common(10):
    print(f"{c/N:8.1f}  {f}")
print("--- top lines")
for (f, ln), c in byline.most_common(40):
    print(f"{c/N:7.1f}  {f}:{ln}  [{stage(f, ln)}]")
