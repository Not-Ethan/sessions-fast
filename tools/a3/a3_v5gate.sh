#!/bin/bash
# Gate on scorer-version-5 data (ss-bench): native ~/bin/a3-TAG and image ss-a3-TAG from ~/w/a3-TAG.
# Data: /mnt/data/{dev5,e7,e11,e23,e60} (~/regen5.sh), references ~/ref5_* (milestone 10 image, accepted by the
# real scorer on v5 data), dev answer ~/devans5/dev-answer.   usage: a3_v5gate.sh TAG [quick]
TAG=$1; N=a3-$TAG
PY=~/venv/bin/python
L="flock /tmp/bench.lock"
export PATH=$HOME/.cargo/bin:$PATH
cd ~/w/$N/shopping-sessions || exit 1
T=$PWD/tools
DEVANS=~/devans5/dev-answer
drun() {  # drun DATA OUT [VAR=VAL ...]: the image with the scorer's container flags
  local data=$1 out=$2; shift 2
  local ev=() s
  for s in "$@"; do ev+=(-e "$s"); done
  sudo rm -rf "$out"; mkdir -p "$out"; chmod 777 "$out"
  local sc; sc=$(mktemp -d); chmod 777 "$sc"
  $L sudo docker run --rm --network none --read-only --tmpfs /tmp:rw,size=64m,mode=1777 \
    --cap-drop ALL --security-opt no-new-privileges --pids-limit 4096 --cpus 8 --memory 24g --memory-swap 24g \
    --user 1000:1000 -v "$data":/data:ro -v "$out":/output -v "$sc":/scratch \
    -e DATA_DIR=/data -e OUTPUT_DIR=/output -e SCRATCH_DIR=/scratch -e HOME=/scratch -e NCPU=8 -e MEM_GB=24 \
    -e SS_VERBOSE=1 "${ev[@]}" ss-$N 2>&1 | tail -1 | grep -o "pipeline=[a-zA-Z-]*\|long_key_rerun=[0-9]*\|long_distinct=[0-9]*" | tr '\n' ' '
  rm -rf "$sc"
}
nrun() {  # nrun DATA OUT [VAR=VAL ...]: the native binary
  local data=$1 out=$2; shift 2
  sudo rm -rf "$out"
  local p; p=$(env SS_VERBOSE=1 NCPU=8 "$@" $L ~/bin/$N "$data" "$out" 2>&1; echo "exit=$?")
  echo "$p" | grep -o "pipeline=[a-zA-Z-]*\|long_key_rerun=[0-9]*\|exit=[0-9]*" | tr '\n' ' '
}
echo "image: $($L sudo docker build -q -t ss-$N . 2>&1 | tail -1)"
echo "unit tests: $($L env RUSTFLAGS='-C target-cpu=x86-64-v2' cargo test --release -q 2>&1 | grep -E 'test result|FAILED|panicked' | tr '\n' ' ')"
for e in "-" "SS_NOAVX2=1" "SS_LONGOFF=1" "SS_LONGBITS=4" "SS_WIDECUT=0" "SS_DENSE=force" "SS_DENSE=force SS_NOAVX2=1"; do
  set -- ; [ "$e" != "-" ] && set -- $e
  echo "fixtures native [$e]: $(env "$@" $PY $T/run_fixtures.py fixtures -- ~/bin/$N 2>&1 | tail -1)"
done
for e in "-" "SS_NOAVX2=1" "SS_LONGBITS=4"; do
  x=""; [ "$e" != "-" ] && x=$e
  echo "fixtures image [$e]: $($L sudo IMAGE=ss-$N EXTRA_ENV=$x $PY $T/run_fixtures.py fixtures -- bash $T/docker_run.sh 2>&1 | tail -1)"
done
for e in "-" "SS_NOAVX2=1" "SS_LONGOFF=1" "SS_LONGBITS=6" "SS_LONGBITS=20" "SS_WIDECUT=0" "SS_FASTEV=0 SS_CLICKFAST=0" "SS_WIDECUT=0 SS_NOAVX2=1" "SS_DENSE=force" "SS_DENSE=1"; do
  set -- ; [ "$e" != "-" ] && set -- $e
  p=$(drun /mnt/data/dev5 ~/v5g_dev "$@")
  echo "dev5 image [$e] $p: $($PY $T/compare.py ~/v5g_dev $DEVANS | tail -1)"
done
for e in "-" "SS_NOAVX2=1" "SS_LONGOFF=1" "SS_LONGBITS=6" "NCPU=1"; do
  set -- ; [ "$e" != "-" ] && set -- $e
  p=$(nrun /mnt/data/dev5 ~/v5g_ndev "$@")
  echo "dev5 native [$e] $p: $($PY $T/compare.py ~/v5g_ndev $DEVANS | tail -1)"
done
[ "$2" = quick ] && exit 0
for d in e7 e11 e23 e60; do
  p=$(drun /mnt/data/$d ~/v5g_c)
  echo "$d image $p vs ref5: $($L $PY $T/compare.py ~/ref5_$d ~/v5g_c | tail -1)"
done
for d in e7 e23; do
  p=$(nrun /mnt/data/$d ~/v5g_c)
  echo "$d native $p vs ref5: $($L $PY $T/compare.py ~/ref5_$d ~/v5g_c | tail -1)"
done
for e in "SS_NOAVX2=1" "SS_LONGOFF=1" "SS_LONGBITS=6" "SS_WIDECUT=0"; do
  p=$(drun /mnt/data/e23 ~/v5g_c $e)
  echo "e23 image [$e] $p vs ref5: $($L $PY $T/compare.py ~/ref5_e23 ~/v5g_c | tail -1)"
done
sudo rm -rf ~/v5g_c ~/v5g_dev ~/v5g_ndev
echo V5GATE-DONE
