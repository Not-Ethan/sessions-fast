#!/bin/bash
# Rebuild the Cascade Lake benchmark VM environment from a fresh Ubuntu 24.04
# image, in one command. Copy a source tree to the VM first (it needs
# shopping-sessions/ and _kits/shopping-sessions/x/{gen,devans}/), e.g.
#   tar czf src.tgz shopping-sessions _kits/shopping-sessions/x/gen _kits/shopping-sessions/x/devans
#   gcloud compute scp src.tgz ss-bench:~/ ; ssh: mkdir -p ~/w/cur && tar xzf ~/src.tgz -C ~/w/cur
#   bash ~/w/cur/shopping-sessions/tools/vm_setup.sh [BASE_SRC_DIR]
# BASE_SRC_DIR (optional): a shopping-sessions/ tree of the accepted baseline,
# built as image ss-base for output comparisons.
#
# Result:
#   packages: docker.io build-essential python3-venv linux-tools; user in group docker
#   toolchain: rustup with Rust 1.85.0 (as in the Dockerfile) + llvm-tools
#   ~/venv with polars 2.1.0 (tools/compare.py)
#   /mnt/data: tmpfs (the scorer's /data is tmpfs; /dev/shm is NOT used:
#     systemd-logind RemoveIPC wipes a user's /dev/shm files at logout)
#   datasets in /mnt/data: dev (seed 1, 1M events, 250K clicks), d7 / d11 / d23
#     (20M events, 5M clicks), d60 (seed 5, 60M events, 15M clicks)
#   ~/gen/target/release/generate; image ss-base (if BASE_SRC_DIR given)
# Benchmarks on a shared VM: wrap every timing run / build in
#   flock /tmp/bench.lock <command>
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
GENSRC=$ROOT/_kits/shopping-sessions/x/gen/generator
[ -d "$GENSRC" ] || { echo "generator source not found at $GENSRC"; exit 1; }

echo "== packages"
sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq > /dev/null
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker.io build-essential python3-venv git \
  "linux-tools-$(uname -r)" linux-tools-common > /dev/null || \
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker.io build-essential python3-venv git > /dev/null
sudo usermod -aG docker "$USER" || true

echo "== rust 1.85.0"
if [ ! -x "$HOME/.cargo/bin/rustup" ]; then
  curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.85.0 --profile minimal > /dev/null
fi
export PATH=$HOME/.cargo/bin:$PATH
rustup toolchain install 1.85.0 --profile minimal > /dev/null
rustup component add llvm-tools-preview --toolchain 1.85.0 > /dev/null || true

echo "== python venv"
[ -x "$HOME/venv/bin/python" ] || python3 -m venv "$HOME/venv"
"$HOME/venv/bin/pip" install -q polars==2.1.0

echo "== tmpfs /mnt/data"
sudo mkdir -p /mnt/data
mountpoint -q /mnt/data || sudo mount -t tmpfs -o size=14g,mode=1777 tmpfs /mnt/data

echo "== generator"
rm -rf "$HOME/gen" && cp -r "$GENSRC" "$HOME/gen"
(cd "$HOME/gen" && cargo build --release -q)
G=$HOME/gen/target/release/generate
gen() { # seed events clicks name
  [ -d "/mnt/data/$4" ] || flock /tmp/bench.lock "$G" --seed "$1" --events "$2" --clicks "$3" --output "/mnt/data/$4" --threads 8 > /dev/null
  echo "$4: $(du -sh "/mnt/data/$4" | cut -f1)"
}
gen 1 1000000 250000 dev
gen 7 20000000 5000000 d7
gen 11 20000000 5000000 d11
gen 23 20000000 5000000 d23
gen 5 60000000 15000000 d60

if [ $# -ge 1 ] && [ -d "$1" ]; then
  echo "== baseline image ss-base from $1"
  flock /tmp/bench.lock sg docker -c "docker build -q -t ss-base '$1'" > /dev/null
fi
echo "== done"
