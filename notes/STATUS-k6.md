# k6: chunk-first (streaming masks) event-line splitter: status (restart insurance)

## Current state (06:25 ET)
- Branch ss-k6 = rebased onto origin/ss-opt4 653fdf6 (contains 86d13ad / k4-v2f); k6 code = v2 (82fdf21
  on the k4-v2e base, f617214 on ss-opt4) + the segment-end test. Tag k6-v2 after the post-rebase
  k6test, SS_K6CHECK dev5 + e23 and the e60 container output comparison (running).
- Switch: SS_K6=1, or `K6_DEFAULT = true` in main.rs (k6 is checked before k4 in `event_task`).
  Checker SS_K6CHECK=1 (every task through the old path and k6 into recording sinks).
- Result (pre-rebase v2, image ss-k6-v2, mirrored vs k4-v2e): e23 -16.5 ms median (10/10 rounds),
  e60 -29.4 ms median (7/8). Count dev5 520.6 -> 472.4 per event line (-9.3%).
- Fallback with k6 on: 0.040-0.042% of lines go to k4's slow_line (dev5 0.040, e7 0.042, e23 0.0421,
  e60 0.0414) = exactly the malformed lines (every one of them ends in the scanner: slow_line's cut,
  cut_wide and lean shape B never take one); 0.37-0.59% go from the decode to event_line (line_fb:
  decode not certain, as in k4); the last <=288 bytes of each file go to tail_fb (29-141 ranges per
  data set). With k6 on, k4's own splitters (split_lean, split, split_wide, and the hot loop task_k4)
  are dead code on the default path; k4's decode_body, line_fb, slow_line and tail_fb are what k6 uses.
- VM ss-k5, my trees under /home/Ethan/k6 (v2src = 82fdf21 tree, v2r = rebased tree, images
  ss-k6-v1, ss-k6-r, ss-k6-v2, ss-k6-v2r), scripts: bx.sh (release build of a tree), tx.sh (tests of a
  tree), cgkx.sh (callgrind + cgaddr of a tree), lb.sh (line bench), k6out.sh (container outputs vs
  ref5_*), v2chain2.sh / v2rchain.sh (validation chains).

## Files
- src/k6.rs: stage 1 (`block`, `block_slow`, `stage1`), stage 2 (`is_simple`, `simple_run`, `cut_into`,
  loop in `task_k6`), `task_any(_v)`, checker `check`.
- src/k6test.rs: k4test adapted (old path vs k6, pack2 on/off; guard pages, every chunk size, all 128
  quote patterns, malformed quotes, 0-12 fields, IDs of every shape/length 0-40, \r in every field, empty
  lines, buffer tails incl. no final newline, single-byte changes, mutations) + block/segment offsets
  0..129, lines longer than the 4 KB segment, every task cut in the last 700 bytes, and the first
  segment's end at every byte of a probe line (unquoted / quoted / partly quoted / malformed; inside a
  field, on the quotes, exactly on the newline, after it) with the buffer also ending at the probe line
  with and without its newline. 15 tests.
- k4.rs: visibility only (decode_body, line_fb, slow_line, tail_fb, MARGIN pub(crate)).
- tools/cgaddr.py: k6 stages (k6-s1 = stage 1, k6-cut, k6-loop). Its k4 decode line ranges are stale
  since k4-v2e (ts/ids/validity split wrong; totals right).

## Design (as built, v2)
- Segment = up to 4096 bytes from a line start `a`, scan end min(task end, b.len()-64, a+4096); bytes past
  the scan end in the last block masked out. Never reads past b.
- Stage 1 per 64-byte block (out-of-line function): C/N/Q masks (2 loads, 6 cmpeq, 6 movemask), D=C|N,
  FS = D<<1, Qs = Q<<1, FSs = D<<2 (carries from the previous block), op = Q&FS, od = (op - D - borrow)&D
  (= (!D + op + carry) & D: op carried to its field's end, across blocks), bad = Qs&!(FSs^D) | D&(od^Qs).
  Common block (bad==0, <=7 delimiters, 99.99% on dev5): 7 unconditional entries (tzcnt+add+store+blsr)
  and one u64 of flags = pdep(pext(Qs,D),0x01..) + 2*pdep(pext(N,D),0x01..). Else `block_slow` (exact
  loop, bad flags). Then 16 sentinel entries flagged bad.
- Stage 2 per line, first entry k: is_simple = one u64 flag load and compare. Then 6 loads (vpmovsxdq /
  vpmovzxbq) + 9 ymm ops + 4 stores write the fastev::Cut (s_j = e[k+j-1]+1+q_j, n_j = e[k+j]-s_j-q_j),
  k4::decode_body, else k4::line_fb; k += 7 (the loop-carried chain is this add).
- Not simple: first newline entry j >= k; if found: slow_line(line start) (or tail_fb if the line starts
  after b.len()-288), k = j+1; if none: next segment at the line start (slow_line if it is the segment's
  first line: longer than the scan / end of b).

## Entry format (STABLE from k6-v1; the batch decode reads it directly)
One segment = the bytes [a, s_end) of one buffer `b`, where `a` is a line start, s_end = min(task end,
b.len() - 64, a + 4096). Bytes of the last 64-byte block at or past s_end are treated as ordinary bytes.
`stage1(p = b + a, nb, lastmask, e)` fills `Ring { e: [u32; CAP], fl: [u8; CAP] }` (repr(C); fl is at
byte offset 4*CAP from e) and returns `cnt`:
- `e[i]` (u32) = offset from `a` of the i-th "entry" in byte order. Entries are every delimiter (comma or
  newline) in the segment, plus every position flagged bad by the quote rule (see k6.rs header; a bad
  position is either a delimiter or a non-delimiter byte inside the malformed line). `e[0]` = u32::MAX
  (i.e. -1: the virtual newline just before `a`). Real entries are 1..cnt-1.
- `fl[i]` (u8) flag byte of entry i: bit 0 (1) = the byte just before it is a quote (for a delimiter that
  ends a simple line's field: the field is quoted); bit 1 (2) = it is a newline; bit 2 (4) = it is a bad
  position (the line containing it is malformed). `fl[0]` = 2.
- Sentinels: entries cnt..cnt+15 have flag 4 (bad) and e = 64*nb; so an 8-byte flag load at any k <= cnt
  is in bounds and a line that runs past the scanned bytes is never simple.
- Lines: entry k-1 is always the previous line's newline (or entry 0). The line whose first entry is k
  starts at a + e[k-1] + 1 (u32 wrapping add). It is SIMPLE iff the 8 flag bytes at k, read as one
  little-endian u64 x, satisfy (x & 0x0006_0606_0606_0606) == 0x0002_0000_0000_0000, i.e. entries
  k..k+5 are neither newline nor bad and entry k+6 is a newline and not bad. A simple line has exactly
  7 fields, ends at its first newline e[k+6], and field j (0..6) is, as offsets from a:
  start s_j = e[k+j-1] + 1 + q_j, length n_j = e[k+j] - s_j - q_j, with q_j = fl[k+j] & 1.
  Every field start has >= 64 readable bytes of b after it. The next line's first entry is k + 7.
- NOT simple at k: let j = first index >= k with fl[j] & 2 (j < cnt). If it exists, the line is
  [a + e[k-1] + 1, a + e[j]] (malformed, empty, or wrong field count): it must go through the old
  routine as a one-line range (k6: `k4::slow_line`, or `k4::tail_fb` for the rest of the task if the line
  starts after b.len() - k4::MARGIN), then continue at k = j + 1. If no such j < cnt: the line runs past
  the scanned bytes; the next segment starts at that line start (if it is the segment's first line,
  i.e. starts at a, it goes to `slow_line` as above: longer than the scan or at the end of b).
- Order: lines must be handed on (decoded / sink / fallback) in increasing k; the interner and long-ID
  logs are order-sensitive.

## Measurements (dev5, NullSink line bench, 1 thread; callgrind on the x86-64-v2 profiling build)
k4 baseline re-measured on this tree (k4-v2a): 539.1/line (loop 151.6 + split 86.3 = 237.9, ids 93.1,
ts 87.4, validity 86.9, sink 13.9, other 13.2, scanner 4.7).
| version | total | k6-s1 | k6-cut | k6-loop | loop | ts | validity | ids | sink | other | line bench ms (k4 / k6) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| v0 (c59d4d4) | 558.3 | 140.9 | 54.0 | 15.3 | 29.4 | 93.9 | 101.8 | 92.4 | 12.9 | 12.9 | - |
| v1 (stage1 out of line, Cut in place) | 520.3 | 128.0 | 23.0 | 14.2 | 29.2 | 86.9 | 106.0 | 97.1 | 12.9 | 18.3 | 62.7-63.7 / 58.8-59.4 (5 rounds) |
("loop" = k4.rs decode helpers dslot/sel/rd* ~25 + prefetch; split+loop for k6 v1 = 128+23+14.2+29.2 = 194.4
vs k4 237.9. The decode costs +19 in validity inside the k6 loop: Cut read from memory + spills.)
| v1b (1401707: stage-1 state in fewer registers) | 511.3 | 122.9 | 23.0 | 14.2 | 29.2 | 86.9 | 104.8 | 97.1 | 12.9 | 15.6 | v1 58.7-59.9 / v1b 59.2-60.3 / k4 62.5-63.3 (4 rounds) |
Exactness k6-v1 (978adc9): k6test 14/14; SS_K6CHECK OK dev5 (slow_line 0.040%), e7 (21.6M lines, 0.042%),
e23 (21.5M, 0.0421%), e60 (62.9M, 0.0414%). Tagged k6-v1.
Stage 1 per 64-byte block (v1b, v2 build, ~95 instr): masks 23 (4 loads: v2 tuning splits 32-byte loads,
6 cmpeq, 6 movemask, 6 shl/or, 1 or), quote rule ~17 (3 shld, op, not+addcarry, od, 7 for bad), popcnt + 2
branches 6, flatten 32 (8 x tzcnt/add/store/blsr), flags 8 (2 pext, 2 pdep, or, store, 2 movabs), loop 5,
prefetch 1, carried state ~3. Floor of this design about 90.
NB=128 segments: no gain (line bench 59.3-60.7 ms).

## Rebased onto k4-v2e (744d257) at 05:14; decision rule from the coordinator
k6 ships only if, on one shipped-style image, mirrored order (ab_env.sh k4-first x5 + k6-first x5),
against k4-v2e: >= 5 ms on e23 over 10 rounds AND a consistent gain on e60 over 8 rounds.
Verdict due 7:00 AM with per-round lifetimes.
- rebased v1b (tree ss4, image ss-k6-r): k6test 14/14, SS_K6CHECK dev5 OK. Lifetimes vs k4-v2e (same image):
  e23 k4-first x5 (k4/k6): 882.9/873.2 875.2/864.9 876.0/871.0 874.5/871.7 849.7/842.0 (k6-k4 mean -7.1)
  e23 k6-first x5 (k6/k4): 861.5/866.4 872.8/864.3 856.2/853.5 866.4/868.0 862.7/868.0 (k6-k4 mean -0.1)
  -> 10 rounds mirrored: -3.6 ms (an order bias of ~3.6 ms favours the second run of a pair). Below the
  5 ms rule. (e60 for this image not run: replaced by v2.)
- v2 = 82fdf21 (stage 1: 7 entries, 5-op quote rule, subborrow, folded flag loads); tests 14/14.
  v2q (same + an x2 unroll that LLVM compiled worse) counted 476.6/line.
- COUNTS on the rebased tree (dev5 line bench, callgrind, v2 profiling build): k4-v2e 520.6/line;
  k6 v2 472.4 (-48.2, -9.3%: k6-s1 111.3, k6-cut 25.0, k6-loop 10.6, decode+sink+other ~325);
  v3 candidate 5317fdf (flatten's 7th blsr as the >7 check, index loop): 469.8 (k6-s1 108.6), tests 14/14.
- llvm-mca (cascadelake) of the stage-1 loop alone (v2, 84 instr/block): 23 cycles/block, IPC 3.64,
  resource-bound (RThroughput 15); ~30 cycles per line at 1.29 blocks/line.
- v2 checker dev5 OK; image ss-k6-v2 built (82fdf21, shipped Dockerfile).
  e23 k4-first x5 (k4/k6): 872.2/821.6 843.9/817.9 838.5/824.6 840.2/824.9 842.8/825.4
  e23 k6-first x5 (k6/k4): 822.8/839.4 830.2/834.7 831.8/843.2 817.6/834.4 828.0/842.2
  -> medians k4 841.2 / k6 824.75 (-16.5 ms); paired mean -18.7 (-15.1 without round 1); k6 faster 10/10.
  e60 k4-first x4 (k4/k6): 2386.5/2300.5 2366.7/2337.8 2364.7/2313.1 2350.4/2334.2
  e60 k6-first x4 (k6/k4): 2410.4/2402.9 2360.8/2427.1 2381.7/2417.8 2364.3/2370.9
  -> medians k4 2378.7 / k6 2349.3 (-29.4 ms); paired mean -35.5; k6 faster 7/8.
  VERDICT: v2 passes the rule (>= 5 ms e23 over 10 mirrored rounds, consistent on e60 over 8).
- v2 candidate 4936c59 (7 unconditional entries, flags via one constant + lea, moving pointer):
  483.0/line total (k6-s1 117.8) on the rebased tree (k4-v2e decode). NB: cgaddr's decode line ranges
  are stale after the rebase (ts/ids/validity split wrong), totals are right.
- v2 additions in test: stage-1 loop unrolled x2; bad = (qs & !(fss ^ d)) | (d & (od ^ qs)) (5 ops
  instead of 7); od via _subborrow_u64 (op - d - borrow = !d + op + carry mod 2^64, borrow = !carry).

## Lifetimes (ab_env.sh, one image, flag-free otherwise; ms)
ss-k6-v1 (978adc9, shipped Dockerfile, Rust 1.100, v3 binary):
- e23 x10 (k4/k6): 836.4/829.7 826.7/840.5 825.9/822.7 821.6/820.6 822.4/817.9 823.7/814.3 827.1/819.0
  819.4/823.6 837.8/817.0 822.4/822.3. Medians 824.8 / 821.5 (-3.3); means 826.3 / 822.8; k6 faster 7/10.
- e60 x4: 3145.9/2318.9 (k4 first run cold) 2315.8/2304.4 2307.5/2311.7 2372.8/2324.0. Medians 2344.3 /
  2315.3; rounds 2-4 means 2332.0 / 2313.4. Too noisy; more rounds queued (v2 image, 8 rounds).
So the line-bench gain (-6%, single thread, NullSink) shrinks to ~0.4% in the 8-thread program: under SMT
the instruction count decides, and k6 v1 removes only 19/line (v1b 28).
Note: an outside build of f35b7a6 (= v0, 558/line, Cut copied 3x per line) measured k6 slower (73.6-75.2
ms line bench); v0 was never the candidate. v1 = 978adc9.

## Next / open
- v3 candidate (5317fdf on the old base; cherry-pick onto ss-k6): the >7 check from the flatten's 7th
  blsr, popcnt only on the common path, index-based block loop: 469.8 vs 472.4 (-2.6/line), tests 14/14.
  Too small to show in lifetimes (~1 ms); not the candidate.
- Where stage 1 could still shrink (v2 build, ~86/block): masks 21 (19 under v3 tuning), quote rule ~16,
  flatten 28, flags 6, check + popcnt 5, loop/prefetch/state ~8. The floor of this design is ~80/block
  (~103/line); the u8-offset + block-index entry format would save ~7 adds/block but costs ~6/line in
  stage 2 and changes the published entry format (not done).
- The decode reads the Cut from memory (+~20/line vs k4's register Cut): the two-loop batch decode that
  reads the entries directly (other engineer) removes that; `simple_run` gives it batches.
