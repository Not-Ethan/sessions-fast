#!/bin/bash
# cachegrind of decomp stages (1 thread, dev set: 6 passes per stage) for ~/w/a3-TAG.
# usage: a3cg.sh TAG STAGE [ENV=VAL ...]   -> /tmp/a3cg-TAG-STAGE.out, digest per event line
set -e
export PATH=$HOME/.cargo/bin:$PATH
TAG=$1; ST=$2; shift 2
cd ~/w/a3-$TAG/shopping-sessions
[ -x target/profiling/sessions ] || flock /tmp/bench.lock env RUSTFLAGS="-C target-cpu=x86-64-v2" cargo build --profile profiling -q 2>&1 | tail -3
B=target/profiling/sessions
mkdir -p /tmp/a3cgo
OUT=/tmp/a3cg-$TAG-$ST$(echo "$@" | tr -c 'A-Za-z0-9' '_').out
flock /tmp/bench.lock env "$@" SS_DECOMP=1 SS_DECOMP_STAGES=$ST NCPU=1 valgrind --tool=cachegrind --branch-sim=yes --cache-sim=no --cachegrind-out-file=$OUT $B /mnt/data/dev /tmp/a3cgo 2>&1 | grep -E "decomp" | head -3
# 6 passes over 1072086 event lines
python3 ~/cgsum.py $OUT $((6 * 1072086)) 2>/dev/null | head -${LINES_OUT:-40}
