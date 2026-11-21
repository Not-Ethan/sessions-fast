#!/bin/bash
# Instruction-level profile of the whole pipeline with cachegrind (single thread, dev set).
# Usage: cg.sh [SRC_TGZ] [TAG] [DATA]
#   SRC_TGZ  tarball holding shopping-sessions/ (default /tmp/ssopt.tgz)
#   TAG      names the work dir and outputs: ~/cg-TAG, /tmp/cg-TAG.out (default: none -> ~/cg, /tmp/cg.out)
#   DATA     dataset (default /mnt/data/dev)
# Then: python3 tools/cgsum.py /tmp/cg-TAG.out
set -e
export PATH=$HOME/.cargo/bin:$PATH
TGZ=${1:-/tmp/ssopt.tgz}
TAG=${2:+-$2}
DATA=${3:-/mnt/data/dev}
WD=~/cg$TAG
OUT=/tmp/cg$TAG.out
rm -rf $WD && mkdir -p $WD && tar -xzf $TGZ -C $WD && cd $WD/shopping-sessions
command -v valgrind >/dev/null || (sudo apt-get update -qq >/dev/null 2>&1; sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq valgrind >/dev/null 2>&1)
valgrind --version
flock /tmp/bench.lock env RUSTFLAGS="-C target-cpu=x86-64-v2" cargo build --profile profiling -q 2>&1 | tail -3
B=target/profiling/sessions; ls -la $B | awk '{print $5}'
rm -rf /tmp/cgout$TAG && mkdir -p /tmp/cgout$TAG
# native reference timing, 1 thread
flock /tmp/bench.lock env DATA_DIR=$DATA OUTPUT_DIR=/tmp/cgout$TAG SCRATCH_DIR=/tmp NCPU=1 SS_VERBOSE=1 $B 2>&1 | cut -c1-600
rm -rf /tmp/cgout$TAG && mkdir -p /tmp/cgout$TAG
flock /tmp/bench.lock env DATA_DIR=$DATA OUTPUT_DIR=/tmp/cgout$TAG SCRATCH_DIR=/tmp NCPU=1 valgrind --tool=cachegrind --branch-sim=yes --cache-sim=yes --cachegrind-out-file=$OUT $B 2>&1 | tail -25
rm -rf /tmp/cgout$TAG
echo "CG-DONE $OUT"
