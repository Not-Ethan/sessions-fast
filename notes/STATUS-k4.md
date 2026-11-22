# k4 round two: status (restart insurance)

Branch ss-k4, worktree .claude/worktrees/k4, VM ss-k4 (build tree /home/Ethan/w/k4).
Target from the coordinator: <= 300 instructions per event line on the NullSink
line bench (from 652), report at 4:15 either way; final arbiter = flag-free
interleaved container lifetime on e60 (3+ rounds) and e23 (4+ rounds).

## Method (scoreboard)
- `/home/Ethan/w/k4/cgk.sh TAG DATA "ENVS"`: profiling build (release + debug info,
  x86-64-v2 + crt-static, same flags as the image), then
  `valgrind --tool=callgrind --dump-instr=yes` of `SS_BENCH=line NCPU=1`
  (NullSink, 3 passes) -> /home/Ethan/cg-TAG.out.
- `tools/cgaddr.py CG BIN LINES 3 HOT.s 0.5`: Ir per line by stage, classified by
  the DWARF inline chain of every instruction (addr2line -i), op classes per line
  (port-1-only scalar ops, shuffles, branches, stack moves, calls, div), and the
  hot trace (instructions run >= 0.5x per line) for llvm-mca. dev5 = 1,071,894 lines.
- llvm-mca-18 -mcpu=cascadelake -iterations=100 -bottleneck-analysis on the hot trace.

## Instructions per event line (dev5, NullSink, 1 thread)

| stage     | old path | k4 v0 (head 2418049) | lean (v3) |
|-----------|---------:|---------------------:|----------:|
| loop/glue |    163.8 |                 92.1 |           |
| split     |    122.4 |                244.3 |           |
| ts        |    142.0 |                 92.9 |           |
| ids       |    113.2 |                136.2 |           |
| validity  |     61.7 |                 92.6 |           |
| sink      |     12.1 |                  8.1 |           |
| scanner   |     16.5 |                 16.5 |           |
| other     |     18.1 |                 19.1 |           |
| **total** |  **651.6** |            **703.7** |           |

(k4 v0 split includes fastev::cut inlined + the Cut handed to the non-inlined decode.)
Op classes per line: old: port-1-only ~30, shuffles ~40, stack moves 73, branches ~53.
k4 v0: stack moves 97, call to k4::decode on every line.

llvm-mca (hot trace): old 468 instr, 264 cycles/iteration simulated, resource floor
87 (ports 0/1/5/6 ~104 uops each per line, evenly), 39% of cycles register
dependencies; k4 v0 571 instr, 328 cycles, floor 110. Port 1/5 hypothesis killed.

Waste: k4::decode not inlined (1 call/line, Cut via stack); cut_wide (15.7% of
lines) compiled at baseline calls scan::simd_masks_avx2 3x per wide line (vzeroupper);
long IDs: other_in + TLS try_with 0.24 calls/line each; no div; no memcpy/bcmp.

AVX-512: Rust 1.85 rejects the intrinsics and target_feature (stable from 1.89);
valgrind cannot run AVX-512 code either.

## Lean kernel (variant SS_K4SPLIT=3), in progress 03:40
- masks of the 128 bytes at the line start once; line end from the newline mask;
  fields 0,1 from window 1 (64 bytes at p), fields 2..6 from window 2 = the 64
  bytes ending at the newline (u128 shift); quote rule per region (`window`);
  decode forced inline (decode_body, no target_feature attribute, #[inline(always)]);
  lines it does not take -> slow_line (cut, cut_wide, scanner; cold).
- dev5 shape count (55k lines): lean shape takes ~97.5% (D2 < 64, fields 2..6
  within 63 bytes); old narrow cut 84.4%; lines <= 128 bytes 99.6%.

## Measurements of the lean kernel (dev5, NullSink, 1 thread)
| version | instr/line (callgrind) | line bench ms (best of 3, 3 interleaved rounds: old / k4 v0 / lean) | checker |
|---|---:|---|---|
| lean 1 (Cut escaped to line_fb: 15 stores/line) | 621.9 | 79.8 / 75.1 / 69.2 | dev5 OK |
| lean 2 (fallback re-splits; Cut stays in registers) | 567.0 | 79.4 / 75.2 / 65.1 (rounds: 65.1 65.5 65.7 vs old 79.4 78.5 78.9) | dev5 OK |

lean 2 by stage (inline debug info is lost for part of split_lean, which lands in
"loop", so split+loop is the reliable figure): split+loop 247.6 (incl. masks ~36,
prefetch ~16, spills ~30), ids 131.2 (2x pack_wide ~84, mix 14, long IDs ~33),
ts+validity 154.3, sink 10.9, other 16.4, scanner 4.7.

Floor estimate for this design with perfect codegen (estimate, not measured):
masks 36 + split 45 + positions 30 + ts 81 + validity 20 + ids 84 + loop/sink 15 = ~310.

## Round 3 (tag k4-v2a = 9d485b7; rebased squash d4a72ba on ss-opt4 41c0a81)
lean + pack2 + unclamped prefetch: 544.3 instr/line (split+loop 237.9, ids 95.2,
ts+validity 178.3, sink 10.0, other+scanner 21.1); line bench 63.2-64.3 ms vs old
79.3-80.2. tests 35 OK; SS_K4CHECK OK dev5 e7 e23 e60 d23; outputs = references
dev5 e7 e23 e60. Lifetime A/B (image ss-k4v2a, SS_K4=0 vs 1): running.

## Streaming masks over the chunk (the ~300 design; not started, for a second engineer)
1. Stage 1, per 64-byte block of the task chunk (sequential, branch-free, AVX2): comma, quote
   and newline masks (2 loads, 6 cmp, 6 movemask, 3 shl/or = ~17 instr per 64 B, ~22 per line)
   written to a small ring of u64 triples (L1-resident, e.g. 64 blocks = 4 KB of input).
2. Lines are indexed by a running bit cursor: the next line start = previous le + 1; its
   128-bit window is two consecutive ring words shifted by (start & 63) (shrd x3 classes).
3. Better: stage 1 also flattens delimiter (comma|newline) positions into a u16 array per
   ring (popcnt + tzcnt/blsr, ~2.5 instr per delimiter, ~18 per line), so field j of the line
   whose first delimiter index is k is [pos[k+j-1]+1, pos[k+j]); line k+7 is the next line.
4. Per-line work becomes: 2 ymm loads of 8 positions (or scalar loads), check pos[k+6] is a
   newline bit (bt on the newline ring word) and no newline among the 6 before (popcnt of n
   between), quote rule from qs at field starts / ends (bt on the quote ring, ~2 per field),
   then the same decode (ts ~85, ids ~80, validity ~20). No per-line masks, no u128 shifts,
   no tz chains, no lean/wide shape split (any line up to the ring size).
5. Estimate: stage 1 ~40 per line + split/quotes ~35 + decode ~185 + loop/sink ~15 = ~275-300.
6. Main risk: the per-line quote check without masks costs as much as it saves (2 bt + logic
   per field = ~25), and chunk/ring boundaries (lines straddling a ring refill, task ends)
   need a careful exact fallback; second risk: stage 1 and stage 2 in one loop interleave
   badly (stage 1 every 64 B, lines every ~80 B: a data-dependent inner loop = mispredicts),
   so stage 1 should run ahead over a whole ring (4 KB) and stage 2 consume it.

## Next
- lifetimes; rebased validation; scorecard under x86-64-v3; then v2b (spills, lazy 5/6,
  long-ID path, ts masks), each tagged with count + 10-round e23 timing.
