#!/bin/bash
# CI benchmark (GitHub Actions x86-64 runner). Expects images ss-base, ss-new,
# ss-noop; datasets under $W; extracted binaries $W/base-bin, $W/new-bin,
# and $W/prof-bin (symbols). Sections are selected by $SECTIONS.
# The scorer's /data is tmpfs, so the timed dataset is copied to /dev/shm.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
DR="$HERE/drun.sh"
W=${W:-/mnt/ss}
DEV=${DEV:-$W/dev}
O=$W/o
SECTIONS=${SECTIONS:-"info native docker compare fixtures"}
SEEDS=${SEEDS:-"7"}
has() { case " $SECTIONS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
G=${GEN:-generate}
PHASES='total=[0-9.]*ms|parse=[0-9.]*ms\(cpu [0-9]*|dedup=[0-9.]*ms\(cpu [0-9]*|sessions=[0-9.]*ms\(cpu [0-9]*'

if has info; then
  echo "=== info"
  uname -r; nproc; lscpu | grep -E "Model name|Thread|L2|L3" | cut -c1-120
  free -g | head -2
fi

nat() { # nat LABEL BIN DATA NCPU [ENV=VAL ...]
  local l=$1 b=$2 d=$3 n=$4; shift 4
  sudo rm -rf $O/nat
  T0=$(date +%s%N)
  env "$@" SS_VERBOSE=1 NCPU=$n "$b" "$d" $O/nat 2> /tmp/nat_err.txt
  T1=$(date +%s%N)
  echo "NATIVE $l ncpu=$n wall=$(( (T1 - T0) / 1000000 ))ms $(grep -oE "$PHASES" /tmp/nat_err.txt | tr '\n' ' ')"
}

for seed in $SEEDS; do
  D=/dev/shm/d$seed
  if [ ! -d $D ]; then
    rm -rf /dev/shm/d*
    $G --seed $seed --events 20000000 --clicks 5000000 --output $D --threads 2 > /dev/null
  fi
  echo "=== seed $seed: $(du -sh $D | cut -f1), $(ls $D/events | wc -l) event files"
  if has native; then
    for n in 2 1; do
      reps=3; [ $n = 1 ] && reps=1
      for i in $(seq $reps); do
        nat base "$W/base-bin" $D $n
        nat new "$W/new-bin" $D $n
      done
    done
  fi
  if has docker; then
    for i in 1 2 3; do bash "$DR" "docker-new-s$seed" ss-new "$D" $O/new; done
  fi
  if has compare; then
    nat cmp-base "$W/base-bin" $D 2 > /dev/null; sudo rm -rf $O/cb; cp -r $O/nat $O/cb
    nat cmp-new "$W/new-bin" $D 2 > /dev/null
    python3 "$HERE/../compare.py" $O/cb $O/nat && echo "seed $seed 20M new == base: OK" || echo "seed $seed 20M new != base: MISMATCH"
    nat cmp-new-noavx2 "$W/new-bin" $D 2 SS_NOAVX2=1 > /dev/null
    python3 "$HERE/../compare.py" $O/cb $O/nat && echo "seed $seed 20M new(noavx2) == base: OK" || echo "seed $seed 20M new(noavx2) != base: MISMATCH"
  fi
done

if has compare; then
  echo "=== dev set vs dev-answer"
  for k in "" SS_NOAVX2=1; do
    bash "$DR" "dev $k" ss-new "$DEV" $O/dev_new $k > /dev/null
    python3 "$HERE/../compare.py" $O/dev_new "$DEVANS" && echo "dev docker [$k] == dev-answer: OK" || echo "dev docker [$k]: MISMATCH"
  done
fi

if has fixtures; then
  echo "=== fixtures (native new binary, AVX2 and SSE paths)"
  python3 "$HERE/../run_fixtures.py" "$HERE/../../fixtures" -- "$W/new-bin" 2>&1 | tail -2
  SS_NOAVX2=1 python3 "$HERE/../run_fixtures.py" "$HERE/../../fixtures" -- "$W/new-bin" 2>&1 | tail -2
fi

if has perf; then
  echo "=== perf (new, 1 thread)"
  PERF=$(command -v perf || ls /usr/lib/linux-tools/*/perf 2>/dev/null | head -1)
  D=/dev/shm/d$(echo $SEEDS | cut -d' ' -f1)
  if [ -n "$PERF" ] && [ -x "$W/prof-bin" ]; then
    sudo rm -rf $O/nat
    sudo env NCPU=1 "$PERF" record -F 4999 -o /tmp/perf.data "$W/prof-bin" "$D" $O/nat 2>&1 | tail -1
    sudo "$PERF" report -i /tmp/perf.data --stdio --no-children --sort symbol 2>/dev/null | grep -E "^ +[0-9]" | head -40 | cut -c1-150
  fi
fi
