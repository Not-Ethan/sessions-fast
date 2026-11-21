#!/bin/bash
# Benchmark helpers for the Cascade Lake twin VM (see README "Performance").
#   vm.sh gen SEED EVENTS CLICKS NAME   generate a dataset into /mnt/data/NAME (tmpfs, like the scorer's /data)
#   vm.sh build NAME SRC_DIR            native binary ~/bin/NAME (Dockerfile flags) + image ss-NAME
#   vm.sh time NAME DATA REPS [ENV=VAL ...]   timed docker runs with the scorer's flags
#   vm.sh nat NAME DATA REPS [ENV=VAL ...]    native runs (phase marks)
set -u
cmd=$1; shift
export PATH=$HOME/.cargo/bin:$PATH
GEN=$HOME/gen/target/release/generate
HERE=$(cd "$(dirname "$0")" && pwd)
case $cmd in
gen)
  seed=$1; ev=$2; cl=$3; name=$4
  [ -d /mnt/data/$name ] || $GEN --seed $seed --events $ev --clicks $cl --output /mnt/data/$name --threads 8 > /dev/null
  echo "$name: $(du -sh /mnt/data/$name | cut -f1) $(ls /mnt/data/$name/events | wc -l) event files"
  ;;
build)
  name=$1; src=$2
  (cd $src && RUSTFLAGS="-C target-cpu=x86-64-v2 -C target-feature=+crt-static" cargo build --release -q --target x86_64-unknown-linux-gnu) || exit 1
  mkdir -p $HOME/bin && cp $src/target/x86_64-unknown-linux-gnu/release/sessions $HOME/bin/$name
  docker build -q -t ss-$name $src > /dev/null && echo "built $name"
  ;;
time)
  name=$1; data=$2; reps=$3; shift 3
  for i in $(seq $reps); do
    CPUS=8 MEM=24g NCPU=8 bash $HERE/../ci/drun.sh "$name $(basename $data) $*" ss-$name $data $HOME/out_$name "$@" | head -1
  done
  ;;
nat)
  name=$1; data=$2; reps=$3; shift 3
  for i in $(seq $reps); do
    rm -rf $HOME/onat; T0=$(date +%s%N)
    env "$@" SS_VERBOSE=1 NCPU=${NCPU:-8} $HOME/bin/$name $data $HOME/onat 2> /tmp/nat.txt
    T1=$(date +%s%N)
    echo "NAT $name $(basename $data) $* wall=$(( (T1 - T0) / 1000000 ))ms $(grep -oE 'total=[0-9.]*ms|links=[0-9.]*ms|parse=[0-9.]*ms\(cpu [0-9]*|dedup=[0-9.]*ms\(cpu [0-9]*|sessions=[0-9.]*ms\(cpu [0-9]*|write=[0-9.]*ms|peak_rss=[0-9]*MB' /tmp/nat.txt | tr '\n' ' ')"
  done
  ;;
esac
