# k3: lean per-line kernel for the dense pipeline (branch ss-k3)

Status: INTERIM REPORT 1 (Sat 18:10 ET). Work continues; this file is updated with every step.

## Design (src/k3.rs)

* Split: masks of `,` `\n` `"` over the 96 bytes at the line start (three 32-byte AVX2 blocks). Window 1 =
  bits 0..64 holds fields 0..2; window 2 = the 64 bits at offset 32, shifted right so that bit 0 is the start of
  field 3 (field 3 must start at 32..64), holds fields 3..6 whose 4th delimiter must be the newline. Quotes
  validated in closed form with fastev.rs's `window` rule. A line that fails goes to the general scanner as a
  one-line range (`ev_range`), exactly as fastev does.
* IDs: both IDs in one ymm (16 bytes ending at each field end, digits right-aligned), base-36 values and
  validity in one pass, lane masks by digit count from a table, pmaddubsw [36,1] / pmaddwd [1296,1] /
  pmuludq 36^4 -> two 40-bit values; unscramble in scalar. Canonical = prefix, 2..8 digits, no leading zero
  (value >= 36^(m-1)), value < 2^40; event number < bound; user slot dense (alias < 8, person < pmax).
* Timestamps: dates by a 28-day exact table (slot = pext of the day digits; 8+3 bytes compared), HH:MM:SS of ts
  and ingest_ts together in one xmm, 13-digit F3 in SIMD, F2 offset in SWAR, form selected by masks (no branch).
* Every check ORs into a "bad" word; one branch decides fast vs fallback. Purchases (3%) go out of line through
  main::event_pay itself. Everything in doubt -> densep::ev_line with the cut fields (the oracle).
* Hook: `densep::ev_task_x` calls `k3::ev_kernel` when SS_K3=1 (AVX2 path only). densep items made pub(crate).

## Exactness

`SS_K3CHECK=ev` runs every event task through densep's current kernel and through k3 and compares every
bucket's DE and DP sequences, the XE list and the interned user IDs:
* dev: OK, 1,072,086 lines; fast 98.56%, line fallback 1.23%, range fallback 0.21%
* d7: OK, 21,593,666 lines; fast 99.08%, line fallback 0.71%, range fallback 44,076
* d23: OK, 21,516,657 lines; fast 98.89%, line fallback 0.86%, range fallback 53,281
Adversarial unit tests: not written yet.

## Measured (ss-bench2, d23, best of 3, thread time per event line)

| stage | 1 thread | 8 threads | wall 8 thr |
|---|---|---|---|
| old split only (fastev) | 22.3 ns | 32.0 ns | 86.6 ms |
| old full (densep kernel + real EvOut sink) | 83.9 ns | 142.2 ns | 384.9 ms |
| k3 split only | 21.3 ns | 30.7 ns | 83.1 ms |
| k3 split + decode, no record | 53.5 ns | 96.3 ns | 260.4 ms |
| k3 full, null sink | 53.5 ns | 91.9 ns | 247.9 ms |
| k3 full, real EvOut sink | 71.8 ns | 118.9 ns | 323.1 ms |

Cachegrind (dev, 1 thread): k3 split ~150 instructions/line; k3 full kernel ~490 instructions/line in k3 code
(hot loop ~520 instructions, no calls, ~70 of them spills), 0.06 branch mispredicts/line in k3 code. First
version had 0.39 mispredicts/line from the compiler turning the timestamp-form select into a branch (fixed).

## Assessment so far (plain)

* Exact on dev/d7/d23, but only 16% faster than the current kernel at 8 threads (118.9 vs 142.2 ns/line).
  The 65 ns target is not met.
* The record sink alone costs ~18 ns/line on 1 thread and ~27 ns/line at 8 threads (EvOut: WC scatter into
  ~656 buckets; 42 KB of staging lines do not fit L1).
* The kernel is ~3x over the instruction budget: my per-stage estimates (~230) were far too low; the branch-free
  "compute everything" style creates high register pressure (spills) and every check costs 2-4 instructions.

## Next

1. Cut kernel instructions: fewer live values, cheaper split, cheaper checks; measure each step.
2. Measure sink variants (linear write per thread vs bucket scatter) to tell the pipeline owner what the
   scatter costs.
3. Click kernel, adversarial cargo tests.
