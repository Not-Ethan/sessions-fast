#!/usr/bin/env python3
"""The hot path of the event loop as straight-line assembly for llvm-mca.
From a cachegrind --dump-instr=yes output, take every instruction executed at
least FRAC * N times (N = lines * passes), in address order, and print its
AT&T disassembly from OBJDUMP (objdump -d --no-show-raw-insn output), with
branch targets replaced by a local label. Also prints a summary to stderr.
Usage: hottrace.py CG_OUT OBJDUMP N FRAC > hot.s"""
import sys, re, collections

CG, OD, N, FRAC = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
cnt = collections.Counter()
ev = None; instr_mode = False
skip = False
for l in open(CG, errors='replace'):
    if skip:
        skip = False; continue
    if l.startswith('calls='):
        skip = True; continue
    if l.startswith('positions:'):
        instr_mode = 'instr' in l; continue
    if l.startswith('events:'):
        ev = l.split()[1:]; iI = ev.index('Ir'); continue
    if not instr_mode or not l.startswith('0x'):
        continue
    p = l.split()
    a = int(p[0], 16)
    cnt[a] += int(p[2 + iI]) if len(p) > 2 + iI else 0
hot = {a: c for a, c in cnt.items() if c >= FRAC * N}
dis = {}
for l in open(OD, errors='replace'):
    m = re.match(r'\s*([0-9a-f]+):\s+(.*)$', l)
    if not m:
        continue
    a = int(m.group(1), 16)
    txt = m.group(2).split('#')[0].strip()
    dis[a] = txt
out = []
tot = 0
for a in sorted(hot):
    t = dis.get(a)
    if t is None:
        continue
    t = re.sub(r'<[^>]*>', '', t).strip()
    op = t.split()[0] if t else ''
    if op.startswith('j') or op.startswith('call'):
        if op.startswith('call'):
            t = '# ' + t
        else:
            t = op + ' L0'
    t = t.replace('nopw', 'nop').replace('cs nop', 'nop')
    if t.startswith('nop') or t.startswith('data16') or t.startswith('xchg   %ax,%ax'):
        continue
    out.append((a, t, hot[a]))
    tot += hot[a]
print('L0:')
for a, t, c in out:
    print(f'\t{t}')
sys.stderr.write(f'hot instructions: {len(out)}; their executions per line: {tot / N:.1f}; '
                 f'range 0x{min(hot):x}-0x{max(hot):x}\n')
