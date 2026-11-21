#!/bin/bash
# Interleaved container-lifetime A/B with the scorer's flags (as ~/tools/life.sh), env per variant.
# usage: a3_ab.sh DATASET ROUNDS "LABEL|IMAGE|ENV" "LABEL|IMAGE|ENV" ...   (ENV: VAR=VAL or -)
D=$1; N=$2; shift 2
OUT=/tmp/a3_ab_$$.txt
: > $OUT
for i in $(seq $N); do
  for spec in "$@"; do
    IFS='|' read -r label img envs <<< "$spec"
    O=$(mktemp -d); S=$(mktemp -d); chmod 777 $O $S
    E=""; [ "$envs" != "-" ] && E="-e $envs"
    C=$(sudo docker create --network none --read-only --tmpfs /tmp:rw,size=64m,mode=1777 --cap-drop ALL --security-opt no-new-privileges --pids-limit 4096 --cpus 8 --memory 24g --memory-swap 24g --user 1000:1000 -v /mnt/data/$D:/data:ro -v $O:/output -v $S:/scratch -e DATA_DIR=/data -e OUTPUT_DIR=/output -e SCRATCH_DIR=/scratch -e HOME=/scratch -e NCPU=8 -e MEM_GB=24 -e SS_VERBOSE=1 $E $img)
    T=$(flock /tmp/bench.lock sudo docker start -a $C 2>&1 | grep -o "total=[0-9.]*ms" | head -1)
    python3 - "$(sudo docker inspect -f '{{.State.StartedAt}} {{.State.FinishedAt}}' $C)" "$T" "$label" "$D" <<'PY' | tee -a $OUT
import sys, datetime
a, b = sys.argv[1].split()
def p(s):
    s = s.rstrip('Z'); d, f = (s.split('.') + ['0'])[:2]
    return datetime.datetime.strptime(d, '%Y-%m-%dT%H:%M:%S').timestamp() + float('0.' + f)
life = (p(b) - p(a)) * 1000
tot = float(sys.argv[2].split('=')[1].rstrip('ms')) if sys.argv[2] else float('nan')
print(f"{sys.argv[3]} {sys.argv[4]} {life:.1f} {tot:.1f}")
PY
    sudo docker rm $C >/dev/null; sudo rm -rf $O $S
  done
done
python3 - $OUT <<'PY'
import sys, collections, statistics
v = collections.defaultdict(list); order = []
for l in open(sys.argv[1]):
    lab, d, life, tot = l.split()
    if lab not in order: order.append(lab)
    v[lab].append((float(life), float(tot)))
base = None
for lab in order:
    L = [a for a, _ in v[lab]]; P = [b for _, b in v[lab]]
    ml = statistics.median(L); mp = statistics.median(P)
    if base is None: base = (ml, mp)
    print(f"{lab:14} lifetime median {ml:7.1f} (min {min(L):7.1f}) program median {mp:7.1f}  d_life {ml-base[0]:+6.1f} d_prog {mp-base[1]:+6.1f}  all {sorted(round(a) for a in L)}")
PY
