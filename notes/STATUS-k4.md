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

## Next
- round 3: unclamped prefetch, pack2 (SS_K4PACK2=1) in the lean loop.
