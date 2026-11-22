#!/usr/bin/env python3
"""Per-stage instructions per event line from a callgrind --dump-instr=yes
output, classified by the DWARF inline chain of every instruction (addr2line
-i), plus op-class counts per line (port-1-only scalar ops, shuffles,
divisions, calls) from the disassembly, and the hot trace for llvm-mca.
Usage: cgaddr.py CG_OUT BINARY LINES PASSES [HOT_S_OUT [FRAC]]"""
import sys, re, collections, subprocess, os

CG, BIN, N = sys.argv[1], sys.argv[2], int(sys.argv[3]) * int(sys.argv[4])
HOT = sys.argv[5] if len(sys.argv) > 5 else None
FRAC = float(sys.argv[6]) if len(sys.argv) > 6 else 0.5

cnt = collections.Counter()
instr_mode = False; skip = False; iI = 0
for l in open(CG, errors='replace'):
    if skip:
        skip = False; continue
    if l.startswith('calls='):
        skip = True; continue
    if l.startswith('positions:'):
        instr_mode = 'instr' in l; continue
    if l.startswith('events:'):
        iI = l.split()[1:].index('Ir'); continue
    if not l.startswith('0x'):
        continue
    p = l.split()
    cnt[int(p[0], 16)] += int(p[2 + iI])

addrs = sorted(a for a, c in cnt.items() if c >= 0.002 * N)
# inline chains
inp = '\n'.join(hex(a) for a in addrs) + '\n'
out = subprocess.run(['addr2line', '-i', '-f', '-C', '-a', '-e', BIN], input=inp, capture_output=True, text=True).stdout.split('\n')
chains = {}
cur = None
i = 0
while i < len(out):
    l = out[i]
    if l.startswith('0x'):
        cur = int(l, 16); chains[cur] = []; i += 1; continue
    if cur is not None and i + 1 < len(out):
        chains[cur].append((l, out[i + 1])); i += 2; continue
    i += 1

RULES = [
    # (file, function regex, stage); first frame (innermost first) that matches wins
    ('k4.rs', r'k4::date|k4::tvec|k4::f3_13', 'ts'),
    ('k4.rs', r'k4::pack2|k4::key_ev|k4::key_user', 'ids'),
    ('k4.rs', r'k4::split|k4::window|k4::masks64|k4::shift128|k4::upto|k4::blsr|k4::tz', 'split'),
    ('fastev.rs', r'fastev::cut|fastev::window|fastev::masks|fastev::upto|fastev::blsr|fastev::tz|fastev::fields', 'split'),
    ('ts.rs', r'.', 'ts'),
    ('keys.rs', r'keys::mix', 'ids'),
    ('keys.rs', r'.', 'ids'),
    ('main.rs', r'event_pay', 'validity'),
    ('main.rs', r'NullSink|EvSink|BSink|copy', 'sink'),
]
def k4_decode_stage(line):
    # decode_body line ranges (k4.rs as of the lean kernel; update when it moves)
    if 628 <= line < 651: return 'ts'
    if 661 <= line < 670 or 686 <= line < 694: return 'ids'
    if line in (673, 681, 694): return 'sink'
    return 'validity'

def classify(a):
    ch = chains.get(a, [])
    for fn, loc in ch:
        f = os.path.basename(loc.split(':')[0]); m = re.search(r':(\d+)', loc); ln = int(m.group(1)) if m else 0
        for rf, rfn, st in RULES:
            if f == rf and re.search(rfn, fn):
                return st
        if f == 'k4.rs' and 'decode' in fn:
            return k4_decode_stage(ln)
        if f == 'main.rs' and 'event_line' in fn:
            return 'validity'
        if f == 'k4.rs' or f == 'fastev.rs':
            return 'loop'
        if f == 'main.rs':
            return 'loop'
        if f == 'scan.rs':
            return 'scanner'
    return 'other'

# disassembly of the hot addresses
dis = {}
od = subprocess.run(['objdump', '-d', '--no-show-raw-insn', BIN], capture_output=True, text=True).stdout
for l in od.split('\n'):
    m = re.match(r'\s*([0-9a-f]+):\s+(.*)$', l)
    if m:
        a = int(m.group(1), 16)
        if a in cnt:
            dis[a] = re.sub(r'<[^>]*>', '', m.group(2).split('#')[0]).strip()

P1 = re.compile(r'^(tzcnt|lzcnt|popcnt|pext|pdep|imul|mul|bsf|bsr|crc32)')
SHUF = re.compile(r'^(vpshufb|pshufb|vpermq|vperm2i128|vinserti128|vextracti128|vpunpck|punpck|vpalignr|palignr|vpshufd|pshufd|vpbroadcast|vpackus|packus|vpacks|vpslldq|pslldq|vpsrldq|psrldq|vphadd|phadd|vpinsr|pinsr|vpextr|pextr|vmovq.*xmm|movq.*xmm|vmovd|movd|vpblendvb|pblendvb)')
DIV = re.compile(r'^(div|idiv)')
st_ir = collections.Counter(); st_ops = collections.defaultdict(collections.Counter)
for a, c in cnt.items():
    if a not in chains:
        continue
    st = classify(a)
    st_ir[st] += c
    t = dis.get(a, '')
    op = t.split()[0] if t else '?'
    if P1.match(op): st_ops[st]['p1'] += c
    if SHUF.match(op): st_ops[st]['p5'] += c
    if DIV.match(op): st_ops[st]['div'] += c
    if op.startswith('call'): st_ops[st]['call'] += c
    if re.match(r'^(mov|vmov)', op) and '(%rsp' in t: st_ops[st]['stack'] += c
    if op.startswith('j'): st_ops[st]['br'] += c
tot = sum(cnt.values())
print(f'all instructions {tot/N:.1f}/line; classified (>=0.2% of lines) {sum(st_ir.values())/N:.1f}/line')
print(f'{"stage":10s} {"Ir/line":>8s} {"p1-only":>8s} {"shuf-p5":>8s} {"branch":>7s} {"stackmov":>8s} {"call":>6s} {"div":>5s}')
for st, c in st_ir.most_common():
    o = st_ops[st]
    print(f'{st:10s} {c/N:8.1f} {o["p1"]/N:8.1f} {o["p5"]/N:8.1f} {o["br"]/N:7.1f} {o["stack"]/N:8.1f} {o["call"]/N:6.2f} {o["div"]/N:5.2f}')
# calls in the hot part
calls = collections.Counter()
for a, c in cnt.items():
    t = dis.get(a, '')
    if t.startswith('call') and c >= 0.01 * N:
        calls[t] += c
for t, c in calls.most_common(12):
    print(f'call {c/N:6.3f}/line  {t}')
if HOT:
    with open(HOT, 'w') as f:
        f.write('L0:\n')
        n = 0; ex = 0
        for a in sorted(cnt):
            if cnt[a] < FRAC * N or a not in dis:
                continue
            t = dis[a]; op = t.split()[0] if t else ''
            if not t or op.startswith('nop') or op.startswith('data16') or op.startswith('cs'):
                continue
            if op.startswith('call'):
                f.write('\t# ' + t + '\n'); continue
            if op.startswith('j'):
                t = op + ' L0'
            f.write('\t' + t + '\n'); n += 1; ex += cnt[a]
        sys.stderr.write(f'hot trace: {n} instructions (>= {FRAC} per line), {ex/N:.1f} executions/line\n')
# optional: dump the instructions of one stage (env STAGE=split), with counts per line and source
if os.environ.get('STAGE'):
    want = os.environ['STAGE']
    for a in sorted(cnt):
        if a in chains and classify(a) == want and cnt[a] >= 0.05 * N:
            ch = chains[a]
            loc = ' < '.join(f"{fn.split('::')[-1][:18]}:{os.path.basename(l)}" for fn, l in ch[:3])
            print(f'{cnt[a]/N:5.2f} {a:x} {dis.get(a, "?")[:48]:48s} {loc[:110]}')
