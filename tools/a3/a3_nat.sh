#!/bin/bash
# native phase timings, interleaved: a3_nat.sh DATA ROUNDS "LABEL1|BIN1|ENV1" "LABEL2|BIN2|ENV2" ...
D=$1; N=$2; shift 2
for i in $(seq $N); do
  for spec in "$@"; do
    IFS='|' read -r label bin envs <<< "$spec"
    rm -rf /tmp/a3nat && mkdir -p /tmp/a3nat
    r=$(flock /tmp/bench.lock env $envs NCPU=8 SS_VERBOSE=1 $bin /mnt/data/$D /tmp/a3nat 2>&1 | grep -oE 'total=[0-9.]*ms|parse=[0-9.]*ms\(cpu [0-9]*|dedup=[0-9.]*ms\(cpu [0-9]*|sessions=[0-9.]*ms\(cpu [0-9]*|write=[0-9.]*ms' | tr '\n' ' ')
    echo "$label $D: $r"
  done
done
rm -rf /tmp/a3nat
