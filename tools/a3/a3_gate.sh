#!/bin/bash
# a3 gate for ~/w/a3-TAG (native binary ~/bin/a3-TAG must exist; builds image ss-a3-TAG)
# usage: a3_gate.sh TAG [quick]
TAG=$1; N=a3-$TAG
PY=~/venv/bin/python
L="flock /tmp/bench.lock"
export PATH=$HOME/.cargo/bin:$PATH
cd ~/w/$N/shopping-sessions || exit 1
T=$PWD/tools
DEVANS=$(ls -d ~/w/*/_kits/shopping-sessions/x/devans/dev-answer 2>/dev/null | head -1)
[ -d "$DEVANS" ] || echo "MISSING dev answer"
echo "image: $($L sudo docker build -q -t ss-$N . 2>&1 | tail -1)"
echo "unit tests: $($L env RUSTFLAGS='-C target-cpu=x86-64-v2' cargo test --release -q 2>&1 | grep -E 'test result|FAILED|panicked' | tr '\n' ' ')"
echo "fixtures native avx2: $($PY $T/run_fixtures.py fixtures -- ~/bin/$N 2>&1 | tail -1)"
echo "fixtures native sse:  $(SS_NOAVX2=1 $PY $T/run_fixtures.py fixtures -- ~/bin/$N 2>&1 | tail -1)"
echo "fixtures image avx2:  $($L sudo IMAGE=ss-$N $PY $T/run_fixtures.py fixtures -- bash $T/docker_run.sh 2>&1 | tail -1)"
echo "fixtures image sse:   $($L sudo IMAGE=ss-$N EXTRA_ENV=SS_NOAVX2=1 $PY $T/run_fixtures.py fixtures -- bash $T/docker_run.sh 2>&1 | tail -1)"
for e in SS_X=1 SS_NOAVX2=1 SS_FASTEV=0; do
  sudo rm -rf ~/a3_odev
  IMAGE=ss-$N EXTRA_ENV=$e $L bash $T/docker_run.sh /mnt/data/dev ~/a3_odev > /dev/null 2>&1
  echo "dev image [$e]: $($PY $T/compare.py ~/a3_odev $DEVANS | tail -1)"
done
for e in SS_X=1 SS_NOAVX2=1; do
  rm -rf ~/a3_odev
  env $e $L ~/bin/$N /mnt/data/dev ~/a3_odev > /dev/null 2>&1
  echo "dev native [$e]: $($PY $T/compare.py ~/a3_odev $DEVANS | tail -1)"
done
[ "$2" = quick ] && exit 0
for d in d7 d11 d23 d60; do
  sudo rm -rf ~/a3_c
  IMAGE=ss-$N $L bash $T/docker_run.sh /mnt/data/$d ~/a3_c > /dev/null 2>&1
  echo "compare $d ref vs $N: $($L $PY $T/compare.py ~/ref_$d ~/a3_c | tail -1)"
done
sudo rm -rf ~/a3_c
IMAGE=ss-$N EXTRA_ENV=SS_NOAVX2=1 $L bash $T/docker_run.sh /mnt/data/d23 ~/a3_c > /dev/null 2>&1
echo "compare d23 ref vs $N (SS_NOAVX2): $($L $PY $T/compare.py ~/ref_d23 ~/a3_c | tail -1)"
sudo rm -rf ~/a3_c ~/a3_odev
echo A3-GATE-DONE
