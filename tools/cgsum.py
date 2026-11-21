#!/usr/bin/env python3
"""Digest of a cachegrind output: per-file and per-line instructions, branch
mispredicts and D1 misses per event line. Usage: cgsum.py [CG_OUT] [EVENT_LINES]"""
import sys, collections, os
ev = None; fl = None
byfile = collections.Counter(); byfile_b = collections.Counter(); byfile_d = collections.Counter()
byline = collections.Counter(); byline_b = collections.Counter(); byline_d = collections.Counter()
CG = sys.argv[1] if len(sys.argv) > 1 else '/tmp/cg.out'
for l in open(CG, errors='replace'):
    if l.startswith('events:'): ev = l.split()[1:]; iI = ev.index('Ir'); iB = ev.index('Bcm'); iD = ev.index('D1mr'); iW = ev.index('D1mw'); continue
    if l.startswith('fl='): fl = l[3:].strip(); continue
    if l.startswith(('fn=', 'desc:', 'cmd:', 'summary:', 'version', 'creator')) or not l[0].isdigit(): continue
    p = l.split(); ln = int(p[0]); v = [int(x) for x in p[1:]] + [0] * 13
    byfile[fl] += v[iI]; byfile_b[fl] += v[iB]; byfile_d[fl] += v[iD] + v[iW]
    byline[(fl, ln)] += v[iI]; byline_b[(fl, ln)] += v[iB]; byline_d[(fl, ln)] += v[iD] + v[iW]
T = sum(byfile.values()); TB = sum(byfile_b.values()); TD = sum(byfile_d.values()); N = int(sys.argv[2]) if len(sys.argv) > 2 else 1072116  # event lines of the dev set
print(f"TOTAL Ir {T:,} = {T/N:.0f} per event line | branch mispredicts {TB:,} = {TB/N:.2f}/line | D1 misses {TD:,} = {TD/N:.2f}/line")
print("--- per file: Ir/line, mispredicts/line, D1 misses/line")
for f, c in byfile.most_common(14):
    print(f"{c/N:8.1f} {byfile_b[f]/N:6.2f} {byfile_d[f]/N:6.2f}  {f[-60:]}")
src = {}
def text(f, ln):
    if f not in src:
        try: src[f] = open(f, errors='replace').read().split('\n')
        except Exception: src[f] = []
    s = src[f]; return s[ln - 1].strip()[:110] if 0 < ln <= len(s) else ''
print("--- top lines by instructions (Ir/line  mispred/line  D1miss/line)")
for (f, ln), c in byline.most_common(48):
    print(f"{c/N:7.1f} {byline_b[(f,ln)]/N:5.2f} {byline_d[(f,ln)]/N:5.2f}  {os.path.basename(f)}:{ln}  {text(f, ln)}")
print("--- top lines by branch mispredicts")
for (f, ln), c in byline_b.most_common(26):
    print(f"{c/N:5.2f}  {os.path.basename(f)}:{ln}  {text(f, ln)}")
print("--- top lines by D1 misses")
for (f, ln), c in byline_d.most_common(16):
    print(f"{c/N:5.2f}  {os.path.basename(f)}:{ln}  {text(f, ln)}")
