#!/usr/bin/env python3
"""
Cross-check the native binary against the Python reference on generated data.

Usage: crosscheck.py GENERATE_BIN BINARY WORKDIR SEED:EVENTS:CLICKS [...]

For each spec: generate the dataset, run the reference, run the binary with
several thread counts / partition settings, compare every output with the
reference, then delete the dataset. Prints one line per spec; exit 1 on any
mismatch.
"""
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from compare import compare  # noqa: E402

REF = os.path.join(HERE, "..", "reference", "reference.py")

VARIANTS = [
    {"NCPU": "8"},
    {"NCPU": "3", "SS_PE_BITS": "2", "SS_PS_BITS": "9", "SS_CHUNK": "65536"},
    {"NCPU": "1", "SS_PE_BITS": "11", "SS_PS_BITS": "3"},
    {"NCPU": "5", "SS_CAPDIV": "50", "SS_CHUNK": "300000"},
    {"NCPU": "4", "SS_LEGACY": "1", "SS_PS_BITS": "7"},
    {"NCPU": "6", "SS_LOCBITS": "5", "SS_PS_BITS": "4", "SS_CHUNK": "100000"},
]
if os.environ.get("CROSSCHECK_VARIANTS"):
    VARIANTS = [VARIANTS[int(i)] for i in os.environ["CROSSCHECK_VARIANTS"].split(",")]


def main():
    gen, binary, work = sys.argv[1:4]
    specs = sys.argv[4:]
    failed = []
    for spec in specs:
        seed, events, clicks = spec.split(":")
        data = os.path.join(work, f"data_{seed}_{events}")
        ref_out = os.path.join(work, f"ref_{seed}_{events}")
        shutil.rmtree(data, ignore_errors=True)
        shutil.rmtree(ref_out, ignore_errors=True)
        subprocess.run([gen, "--seed", seed, "--events", events, "--clicks", clicks, "--output", data,
                        "--threads", "4"], check=True, stdout=subprocess.DEVNULL)
        t = time.time()
        r = subprocess.run([sys.executable, REF, data, ref_out], capture_output=True, text=True)
        ref_secs = time.time() - t
        if r.returncode != 0:
            print(f"{spec}: REFERENCE FAILED\n{r.stderr[-2000:]}", flush=True)
            failed.append(spec)
            continue
        warnings = [l for l in r.stderr.splitlines() if "WARNING" in l]
        results = []
        for i, var in enumerate(VARIANTS):
            out = os.path.join(work, f"fast_{seed}_{events}_{i}")
            shutil.rmtree(out, ignore_errors=True)
            env = dict(os.environ, **var)
            p = subprocess.run([binary, data, out], env=env, capture_output=True, text=True)
            if p.returncode != 0:
                results.append(f"v{i}:EXIT{p.returncode}")
                failed.append(spec)
                print(p.stderr[-2000:])
                continue
            ok = compare(ref_out, out, verbose=False)
            results.append(f"v{i}:{'ok' if ok else 'DIFF'}")
            if not ok:
                failed.append(spec)
            shutil.rmtree(out, ignore_errors=True)
        print(f"{spec}: ref {ref_secs:.0f}s {' '.join(results)} warnings={len(warnings)} {warnings[:2]}", flush=True)
        shutil.rmtree(data, ignore_errors=True)
        shutil.rmtree(ref_out, ignore_errors=True)
    print("ALL OK" if not failed else f"FAILED: {sorted(set(failed))}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
