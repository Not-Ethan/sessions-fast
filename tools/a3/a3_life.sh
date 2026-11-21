#!/bin/bash
# interleaved container lifetimes: a3_life.sh REF_IMAGE IMAGE "DATASETS" ROUNDS
REF=$1; IMG=$2; DS=$3; N=${4:-5}
OUT=/tmp/a3_life_$$.txt
for d in $DS; do
  for i in $(seq $N); do
    flock /tmp/bench.lock bash ~/tools/life.sh $REF $d 1
    flock /tmp/bench.lock bash ~/tools/life.sh $IMG $d 1
  done
done 2>&1 | tee $OUT
python3 - $OUT $REF $IMG <<'PY'
import sys, collections, statistics
f, ref, img = sys.argv[1:4]
v = collections.defaultdict(list)
for l in open(f):
    p = l.split()
    if len(p) > 7 and p[2] == 'lifetime':
        v[(p[0], p[1].rstrip(':'))].append((float(p[3]), float(p[7])))
for d in sorted(set(d for _, d in v)):
    r = v.get((ref, d), []); m = v.get((img, d), [])
    if not r or not m:
        continue
    lr = statistics.median(a for a, _ in r); lm = statistics.median(a for a, _ in m)
    pr = statistics.median(b for _, b in r); pm = statistics.median(b for _, b in m)
    print(f"{d}: {ref} lifetime median {lr:.1f} ms (program {pr:.1f}), {img} {lm:.1f} ms (program {pm:.1f}); ratio {lm/lr:.3f} (n={len(r)},{len(m)}); "
          f"all {ref} {sorted(round(a) for a, _ in r)} {img} {sorted(round(a) for a, _ in m)}")
PY
