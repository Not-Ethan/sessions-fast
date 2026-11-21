#!/usr/bin/env python3
"""
Run a pipeline on every fixture and compare with the hand-written expected tables.

Usage: run_fixtures.py FIXTURES_DIR -- CMD...
CMD gets DATA_DIR / OUTPUT_DIR / SCRATCH_DIR env vars and DATA_DIR OUTPUT_DIR as
its last two arguments, and may write either CSV or Parquet outputs.
"""
import io
import os
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare import compare  # noqa: E402


def main():
    sep = sys.argv.index("--")
    fx_dir, cmd = sys.argv[1], sys.argv[sep + 1:]
    cases = sorted(d for d in os.listdir(fx_dir) if os.path.isdir(os.path.join(fx_dir, d)))
    failed = []
    for case in cases:
        data = os.path.join(fx_dir, case, "input")
        with tempfile.TemporaryDirectory() as out, tempfile.TemporaryDirectory() as scratch:
            env = dict(os.environ, DATA_DIR=data, OUTPUT_DIR=out, SCRATCH_DIR=scratch)
            p = subprocess.run(cmd + [data, out], env=env, capture_output=True, text=True)
            if p.returncode != 0:
                failed.append(case)
                print(f"FAIL {case}: exit {p.returncode}\n{p.stderr[-2000:]}")
                continue
            buf = io.StringIO()
            with redirect_stdout(buf):
                ok = compare(os.path.join(fx_dir, case, "expected"), out, verbose=False)
            if ok:
                print(f"ok   {case}")
            else:
                failed.append(case)
                print(f"FAIL {case}\n{buf.getvalue()}")
    print(f"\n{len(cases) - len(failed)}/{len(cases)} fixtures passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
