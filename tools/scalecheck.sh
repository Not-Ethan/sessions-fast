#!/bin/sh
# Scale check without a reference: generate a dataset, run the binary under
# several thread / partition / capacity settings, require identical tables,
# run the rules' self-check identities, print timings and peak RSS, delete.
# usage: scalecheck.sh GENERATE BINARY PYTHON WORKDIR SEED EVENTS CLICKS
set -eu
GEN=$1; BIN=$2; PY=$3; W=$4; SEED=$5; EV=$6; CL=$7
HERE=$(cd "$(dirname "$0")" && pwd)
D="$W/scale_$SEED"
rm -rf "$D" "$W/sc_o0" "$W/sc_o1" "$W/sc_o2"
"$GEN" --seed "$SEED" --events "$EV" --clicks "$CL" --output "$D" --threads 4 >/dev/null
du -sh "$D"
for i in 1 2 3; do
  rm -rf "$W/sc_o0"
  SS_VERBOSE=1 NCPU=8 "$BIN" "$D" "$W/sc_o0"
done
SS_VERBOSE=1 NCPU=3 SS_PE_BITS=10 SS_PS_BITS=6 SS_CHUNK=700000 "$BIN" "$D" "$W/sc_o1"
SS_VERBOSE=1 NCPU=8 SS_CAPDIV=40 SS_PE_BITS=5 "$BIN" "$D" "$W/sc_o2"
"$PY" "$HERE/compare.py" "$W/sc_o0" "$W/sc_o1"
"$PY" "$HERE/compare.py" "$W/sc_o0" "$W/sc_o2"
"$PY" "$HERE/selfcheck.py" "$W/sc_o0"
rm -rf "$D" "$W/sc_o0" "$W/sc_o1" "$W/sc_o2"
