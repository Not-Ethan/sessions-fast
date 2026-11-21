#!/bin/bash
# CI benchmark (GitHub Actions x86-64 runner). Expects images ss-base, ss-new,
# ss-noop; datasets $D20 and $DEV; extracted binaries $W/base-bin, $W/new-bin,
# and (optionally) $W/prof-bin with symbols. Sections are selected by
# $SECTIONS (space-separated), default all.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
DR="$HERE/drun.sh"
W=${W:-/mnt/ss}
D20=${D20:-$W/d20}
DEV=${DEV:-$W/dev}
O=$W/o
REPS=${REPS:-5}
SECTIONS=${SECTIONS:-"info noop warm cold knobs threads native perf compare"}
has() { case " $SECTIONS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
drop() { sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null; }
warmup() { cat "$1"/events/* "$1"/clicks/* "$1"/links/* > /dev/null; }

if has info; then
  echo "=== info"
  uname -a; nproc; lscpu | grep -E "Model name|Socket|Thread|Core|MHz|L2|L3|NUMA|Flags" | cut -c1-200
  free -g; cat /sys/kernel/mm/transparent_hugepage/enabled /sys/kernel/mm/transparent_hugepage/defrag
  cat /sys/kernel/mm/transparent_hugepage/shmem_enabled 2>/dev/null
  docker info 2>/dev/null | grep -iE "cgroup|storage driver|kernel|runtime"
  du -sh "$D20"/* ; ls "$D20"/events | wc -l
fi

if has noop; then
  echo "=== noop container (start/exit overhead)"
  for i in $(seq $REPS); do bash "$DR" noop ss-noop "$DEV" $O/noop; done
fi

warmup "$D20"
if has warm; then
  echo "=== warm page cache, scorer flags, CPUS=$(nproc)"
  for img in base new; do
    for i in $(seq $REPS); do bash "$DR" "warm-$img" ss-$img "$D20" $O/$img; done
  done
  echo "--- NCPU=4 and 8 on $(nproc) CPUs (oversubscribed)"
  for i in 1 2; do NCPU=4 bash "$DR" "warm-new-ncpu4" ss-new "$D20" $O/new; done
  for i in 1 2; do NCPU=8 bash "$DR" "warm-new-ncpu8" ss-new "$D20" $O/new; done
fi

if has cold; then
  echo "=== cold page cache (drop_caches before each run)"
  for img in base new; do
    for i in 1 2 3; do drop; bash "$DR" "cold-$img" ss-$img "$D20" $O/$img; done
  done
  echo "--- cold, SS_READ=1"
  for i in 1 2 3; do drop; bash "$DR" "cold-new-read" ss-new "$D20" $O/new SS_READ=1; done
  warmup "$D20"
fi

if has knobs; then
  echo "=== knobs (warm)"
  for k in SS_READ=1 SS_FREE=1 SS_NOHUGE=1 SS_NOAVX2=1 "SS_READ=1 SS_FREE=1"; do
    for i in 1 2 3; do bash "$DR" "knob-${k// /+}" ss-new "$D20" $O/new $k; done
  done
fi

if has threads; then
  echo "=== thread scaling (container, warm)"
  for n in 1 2 4; do
    for i in 1 2; do NCPU=$n bash "$DR" "threads-$n" ss-new "$D20" $O/new; done
  done
fi

if has native; then
  echo "=== native (no container), warm"
  for b in base new; do
    for i in 1 2 3; do
      sudo rm -rf $O/nat; T0=$(date +%s%N)
      SS_VERBOSE=1 NCPU=4 "$W/$b-bin" "$D20" $O/nat 2> /tmp/nat_err.txt
      T1=$(date +%s%N)
      echo "NATIVE $b wall=$(( (T1 - T0) / 1000000 ))ms $(grep -o 'total=[0-9.]*ms' /tmp/nat_err.txt) $(grep -o 'unix_end_ns=[0-9]*' /tmp/nat_err.txt | sed "s/unix_end_ns=//;s/^/end->exit=/" | awk -v t1=$T1 -F= '{printf "end->exit=%.1fms", (t1-$2)/1e6}')"
      tr '|' '\n' < /tmp/nat_err.txt | sed 's/^/    /'
    done
  done
fi

if has perf; then
  echo "=== perf"
  PERF=$(command -v perf || ls /usr/lib/linux-tools/*/perf 2>/dev/null | head -1)
  if [ -n "$PERF" ] && [ -x "$W/prof-bin" ]; then
    for n in 1 4; do
      sudo rm -rf $O/nat
      sudo env NCPU=$n SS_VERBOSE=1 "$PERF" stat -e task-clock,context-switches,cpu-migrations,page-faults,cycles,instructions,cache-misses,branch-misses \
        "$W/prof-bin" "$D20" $O/nat 2>&1 | tail -25
    done
    sudo rm -rf $O/nat
    sudo env NCPU=4 "$PERF" record -F 2999 -g -o /tmp/perf.data "$W/prof-bin" "$D20" $O/nat 2>&1 | tail -3
    sudo "$PERF" report -i /tmp/perf.data --stdio --no-children --sort symbol 2>/dev/null | grep -v "^$" | grep -E "^ +[0-9]" | head -60
    echo "--- by dso"
    sudo "$PERF" report -i /tmp/perf.data --stdio --no-children --sort dso 2>/dev/null | grep -E "^ +[0-9]" | head -10
    echo "--- kernel symbols"
    sudo "$PERF" report -i /tmp/perf.data --stdio --no-children --sort symbol --dsos '[kernel.kallsyms]' 2>/dev/null | grep -E "^ +[0-9]" | head -30
  else
    echo "perf not available ($PERF)"
  fi
fi

if has compare; then
  echo "=== correctness: new vs base on 20M, new on dev vs dev-answer"
  bash "$DR" cmp-base ss-base "$D20" $O/cmp_base > /dev/null
  bash "$DR" cmp-new ss-new "$D20" $O/cmp_new > /dev/null
  python3 "$HERE/../compare.py" $O/cmp_base $O/cmp_new && echo "20M new == base: OK" || echo "20M new != base: MISMATCH"
  python3 "$HERE/../selfcheck.py" $O/cmp_new || true
  for k in "" SS_NOAVX2=1 SS_READ=1; do
    bash "$DR" "dev $k" ss-new "$DEV" $O/dev_new $k > /dev/null
    python3 "$HERE/../compare.py" $O/dev_new "$DEVANS" && echo "dev [$k] == dev-answer: OK" || echo "dev [$k]: MISMATCH"
  done
fi
