#!/bin/bash
# One timed container run with the scorer's flags (CPU/memory scaled to the
# runner). usage: drun.sh LABEL IMAGE DATA OUT [ENV=VAL ...]
# Prints: RESULT label wall_ms docker_ms exec_ms prog_ms teardown_ms
# and the program's SS_VERBOSE line.
set -u
LABEL=$1; IMAGE=$2; DATA=$3; OUT=$4; shift 4
CPUS=${CPUS:-$(nproc)}; MEM=${MEM:-5g}; NCPU=${NCPU:-$CPUS}
rm -rf "$OUT"; mkdir -p "$OUT"; chmod 777 "$OUT"
SCR=$(mktemp -d); chmod 777 "$SCR"
ENVS=()
for kv in "$@"; do ENVS+=(-e "$kv"); done
NAME="ssb_$$_$RANDOM"
T0=$(date +%s%N)
docker run --name "$NAME" \
  --network none --read-only --tmpfs /tmp:rw,size=64m,mode=1777 \
  --cap-drop ALL --security-opt no-new-privileges \
  --pids-limit 4096 --cpus "$CPUS" --memory "$MEM" --memory-swap "$MEM" \
  --user 1000:1000 \
  -v "$DATA":/data:ro -v "$OUT":/output -v "$SCR":/scratch \
  -e DATA_DIR=/data -e OUTPUT_DIR=/output -e SCRATCH_DIR=/scratch -e HOME=/scratch \
  -e NCPU="$NCPU" -e MEM_GB="${MEM%g}" -e SS_VERBOSE=1 "${ENVS[@]}" \
  "$IMAGE" > /tmp/drun_out.txt 2> /tmp/drun_err.txt
RC=$?
[ $RC -ne 0 ] && { echo "docker run failed rc=$RC"; tail -5 /tmp/drun_err.txt; }
T1=$(date +%s%N)
INS=$(docker inspect -f '{{.State.StartedAt}} {{.State.FinishedAt}}' "$NAME")
docker rm "$NAME" > /dev/null
rm -rf "$SCR"
python3 - "$LABEL" "$T0" "$T1" "$INS" "$RC" <<'EOF'
import re, sys
label, t0, t1, ins, rc = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
def ns(s):
    m = re.match(r"(\d+)-(\d+)-(\d+)T(\d+):(\d+):(\d+)(?:\.(\d+))?Z", s)
    import calendar
    y, mo, d, h, mi, se = map(int, m.groups()[:6])
    frac = (m.group(7) or "0").ljust(9, "0")[:9]
    return calendar.timegm((y, mo, d, h, mi, se)) * 10**9 + int(frac)
a, b = ins.split()
dock = (ns(b) - ns(a)) / 1e6
err = open("/tmp/drun_err.txt").read()
m = re.search(r"unix_start_ns=(\d+) unix_end_ns=(\d+)", err)
tot = re.search(r"total=([\d.]+)ms", err)
if m:
    s, e = int(m.group(1)), int(m.group(2))
    execl, prog, tear = (s - t0) / 1e6, (e - s) / 1e6, (t1 - e) / 1e6
    # container-relative: from docker StartedAt to program start, program end to FinishedAt
    cstart, cend = (s - ns(a)) / 1e6, (ns(b) - e) / 1e6
else:
    execl = prog = tear = cstart = cend = float("nan")
print(f"RESULT {label:24s} rc={rc} wall={(t1-t0)/1e6:7.1f} docker={dock:7.1f} prog={prog:7.1f} "
      f"run->main={execl:6.1f} main_end->exit={tear:6.1f} | in-container: start->main={cstart:6.1f} main_end->finish={cend:6.1f}")
line = [l for l in err.splitlines() if l.startswith("sessions:")]
if line:
    print("   ", line[-1].replace(" | ", "\n    "))
elif err.strip():
    print("    stderr:", err[-1500:])
EOF
