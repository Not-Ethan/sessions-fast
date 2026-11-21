#!/bin/bash
# a3 helpers on the twin. usage:
#   a3.sh build TAG            unpack ~/a3-TAG.tgz into ~/w/a3-TAG, native build -> ~/bin/a3-TAG
#   a3.sh image TAG            docker image ss-a3-TAG from ~/w/a3-TAG
#   a3.sh check TAG DATA [ENV=VAL..]   SS_EVCHECK on DATA
#   a3.sh decomp TAG DATA STAGES [ENV=VAL..]
#   a3.sh test TAG             cargo test --release (x86, AVX2 tests run here)
set -u
export PATH=$HOME/.cargo/bin:$PATH
cmd=$1; tag=$2; shift 2
W=$HOME/w/a3-$tag
case $cmd in
build)
  rm -rf $W && mkdir -p $W && tar xzf $HOME/a3-$tag.tgz -C $W
  (cd $W/shopping-sessions && RUSTFLAGS="-C target-cpu=x86-64-v2 -C target-feature=+crt-static" flock /tmp/bench.lock cargo build --release -q --target x86_64-unknown-linux-gnu) || exit 1
  mkdir -p $HOME/bin && cp $W/shopping-sessions/target/x86_64-unknown-linux-gnu/release/sessions $HOME/bin/a3-$tag && echo "built a3-$tag"
  ;;
image)
  (cd $W/shopping-sessions && flock /tmp/bench.lock docker build -q -t ss-a3-$tag . > /dev/null) && echo "image ss-a3-$tag"
  ;;
test)
  (cd $W/shopping-sessions && RUSTFLAGS="-C target-cpu=x86-64-v2" flock /tmp/bench.lock cargo test --release -q 2>&1 | tail -15)
  ;;
check)
  data=$1; shift
  mkdir -p /tmp/a3o && flock /tmp/bench.lock env "$@" SS_EVCHECK=1 $HOME/bin/a3-$tag /mnt/data/$data /tmp/a3o
  ;;
decomp)
  data=$1; st=$2; shift 2
  mkdir -p /tmp/a3o && flock /tmp/bench.lock env "$@" SS_DECOMP=1 SS_DECOMP_STAGES=$st NCPU=8 $HOME/bin/a3-$tag /mnt/data/$data /tmp/a3o 2>&1 | grep -E "decomp"
  ;;
esac
