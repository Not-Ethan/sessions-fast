#!/bin/bash
# Gate for the dense milestone on ~/w/a3-TAG (native binary ~/bin/a3-TAG must
# exist; builds image ss-a3-TAG). Dense is the default; SS_DENSE=0 = general.
# usage: a3_dgate.sh TAG [quick]
TAG=$1; N=a3-$TAG
PY=~/venv/bin/python
L="flock /tmp/bench.lock"
export PATH=$HOME/.cargo/bin:$PATH
cd ~/w/$N/shopping-sessions || exit 1
T=$PWD/tools
DEVANS=$(ls -d ~/w/*/_kits/shopping-sessions/x/devans/dev-answer 2>/dev/null | head -1)
[ -d "$DEVANS" ] || echo "MISSING dev answer"
# drun DATA OUT [VAR=VAL ...]: the image with the scorer's container flags
drun() {
  local data=$1 out=$2; shift 2
  local ev=() s
  for s in "$@"; do ev+=(-e "$s"); done
  sudo rm -rf "$out"; mkdir -p "$out"; chmod 777 "$out"
  local sc; sc=$(mktemp -d); chmod 777 "$sc"
  $L sudo docker run --rm --network none --read-only --tmpfs /tmp:rw,size=64m,mode=1777 \
    --cap-drop ALL --security-opt no-new-privileges --pids-limit 4096 --cpus 8 --memory 24g --memory-swap 24g \
    --user 1000:1000 -v "$data":/data:ro -v "$out":/output -v "$sc":/scratch \
    -e DATA_DIR=/data -e OUTPUT_DIR=/output -e SCRATCH_DIR=/scratch -e HOME=/scratch -e NCPU=8 -e MEM_GB=24 \
    -e SS_VERBOSE=1 "${ev[@]}" ss-$N 2>&1 | tail -1 | grep -o "pipeline=[a-z]*"
  rm -rf "$sc"
}
echo "image: $($L sudo docker build -q -t ss-$N . 2>&1 | tail -1)"
echo "unit tests: $($L env RUSTFLAGS='-C target-cpu=x86-64-v2' cargo test --release -q 2>&1 | grep -E 'test result|FAILED|panicked' | tr '\n' ' ')"
for e in SS_DENSE=0 SS_DENSE=force; do
  echo "fixtures native avx2 [$e]: $(env $e $PY $T/run_fixtures.py fixtures -- ~/bin/$N 2>&1 | tail -1)"
  echo "fixtures native sse  [$e]: $(env $e SS_NOAVX2=1 $PY $T/run_fixtures.py fixtures -- ~/bin/$N 2>&1 | tail -1)"
  echo "fixtures image avx2  [$e]: $($L sudo IMAGE=ss-$N EXTRA_ENV=$e $PY $T/run_fixtures.py fixtures -- bash $T/docker_run.sh 2>&1 | tail -1)"
done
echo "fixtures native avx2 [SS_DENSE=force SS_K3=0]: $(env SS_DENSE=force SS_K3=0 $PY $T/run_fixtures.py fixtures -- ~/bin/$N 2>&1 | tail -1)"
echo "fixtures native avx2 [default]: $($PY $T/run_fixtures.py fixtures -- ~/bin/$N 2>&1 | tail -1)"
echo "fixtures image avx2 [SS_K3=0]: $($L sudo IMAGE=ss-$N EXTRA_ENV=SS_K3=0 $PY $T/run_fixtures.py fixtures -- bash $T/docker_run.sh 2>&1 | tail -1)"
echo "fixtures image default: $($L sudo IMAGE=ss-$N $PY $T/run_fixtures.py fixtures -- bash $T/docker_run.sh 2>&1 | tail -1)"
echo "fixtures image sse default: $($L sudo IMAGE=ss-$N EXTRA_ENV=SS_NOAVX2=1 $PY $T/run_fixtures.py fixtures -- bash $T/docker_run.sh 2>&1 | tail -1)"
# dev: dense (default) and general variants, image and native
for envs in "-" "SS_NOAVX2=1" "SS_K3=0" "SS_DENSE=force" "SS_DENSE=0 SS_X=1" "SS_DENSE=0 SS_NOAVX2=1" "SS_DENSE=0 SS_FASTEV=0" "SS_DENSE=0"; do
  set -- ; [ "$envs" != "-" ] && set -- $envs
  p=$(drun /mnt/data/dev ~/a3_odev "$@")
  echo "dev image [$envs] $p: $($PY $T/compare.py ~/a3_odev $DEVANS | tail -1)"
done
for envs in "-" "SS_NOAVX2=1" "SS_K3=0" "SS_K3=0 SS_NOAVX2=1" "SS_DENSE=0 SS_X=1" "SS_DENSE=0 SS_NOAVX2=1"; do
  set -- ; [ "$envs" != "-" ] && set -- $envs
  sudo rm -rf ~/a3_ndev
  p=$(env "$@" SS_VERBOSE=1 NCPU=8 $L ~/bin/$N /mnt/data/dev ~/a3_ndev 2>&1; echo "exit=$?")
  p=$(echo "$p" | grep -o "pipeline=[a-z]*\|exit=[0-9]*" | tr '\n' ' ')
  echo "dev native [$envs] $p: $($PY $T/compare.py ~/a3_ndev $DEVANS | tail -1)"
done
[ "$2" = quick ] && exit 0
for d in d7 d11 d23 d60; do
  p=$(drun /mnt/data/$d ~/a3_c)
  echo "compare $d ref vs $N $p: $($L $PY $T/compare.py ~/ref_$d ~/a3_c | tail -1)"
done
for envs in "SS_NOAVX2=1" "SS_K3=0" "SS_DENSE=0" "SS_DENSE=0 SS_NOAVX2=1"; do
  p=$(drun /mnt/data/d23 ~/a3_c $envs)
  echo "compare d23 ref vs $N [$envs] $p: $($L $PY $T/compare.py ~/ref_d23 ~/a3_c | tail -1)"
done
for d in d7 d11 d23 d60; do
  sudo rm -rf ~/a3_c
  p=$(env SS_VERBOSE=1 NCPU=8 $L ~/bin/$N /mnt/data/$d ~/a3_c 2>&1; echo "exit=$?")
  p=$(echo "$p" | grep -o "pipeline=[a-z]*\|exit=[0-9]*" | tr '\n' ' ')
  echo "compare $d ref vs native $N $p: $($L $PY $T/compare.py ~/ref_$d ~/a3_c | tail -1)"
done
sudo rm -rf ~/a3_c ~/a3_odev ~/a3_ndev
echo A3-DGATE-DONE
