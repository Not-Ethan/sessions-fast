# k6: chunk-first (streaming masks) event-line splitter: status (restart insurance)

Branch ss-k6 (worktree .claude/worktrees/k6, from origin/ss-k4-rebased = k4-v2a rebased on main).
VM ss-k5 (shared: every build/test/timing under `flock /home/Ethan/vm.lock`), tree /home/Ethan/k6/ss,
scripts /home/Ethan/k6/{b.sh (release build), t.sh (tests)}. Switch SS_K6=1 (default off), checker
SS_K6CHECK=1. Target: event line 544 -> ~380 instructions (NullSink line bench, dev5, 1 thread).
Merge-ready by 8:00 AM ET; tag every checker-clean step k6-vN.

## Files
- src/k6.rs: stage 1 (`block`, `block_slow`, `stage1`), stage 2 (`cut`, loop in `task_k6`), checker.
- src/k6test.rs: k4test adapted (old path vs k6, pack2 on/off) + block/segment offsets, lines longer
  than the ring, end-of-buffer task cuts.
- k4.rs: visibility only (decode_body, line_fb, slow_line, tail_fb, MARGIN pub(crate)), commit 22c9973.

## Design (as built, v0)
- Segment = up to 4096 bytes from a line start `a`, scan end min(task end, b.len()-64, a+4096); bytes past
  the scan end in the last block masked out. Never reads past b.
- Stage 1 per 64-byte block: C/N/Q masks (2 loads, 6 cmpeq, 6 movemask), D=C|N, FS, Qs, FSs, op, Od
  (u128 add carries op through the field, across blocks), bad = (Qs&!FSs&!D) | (D&((Od^Qs)|(Qs&FSs))).
  Common block (bad==0, <=8 delimiters): 8 unconditional entries (tzcnt+add+store+blsr) and one u64 of
  flags = pdep(pext(Qs,D),0x01..) | pdep(pext(N,D),0x02..). Else `block_slow` (exact loop, bad flags).
  Entries are u32 offsets from the segment start + 1; entry 0 = virtual newline before `a`; 16 sentinel
  entries flagged bad after the last.
- Stage 2 per line, first entry k: flags of k..k+7 as one u64; simple iff (x & 0x0006060606060606) ==
  0x0002000000000000. Then 6 loads (vpmovzxdq/bq) + 8 ymm ops + 4 stores build the fastev::Cut
  (s_j = e[k+j-1]+q_j, n_j = e[k+j]-e[k+j-1]-1-2q_j), k4::decode_body, else k4::line_fb; k += 7.
- Not simple: first newline entry j >= k; if found: slow_line(line start) (or tail_fb if the line starts
  after b.len()-288), k = j+1; if none: next segment at the line start (slow_line if it is the segment's
  first line: longer than the scan / end of b).

## Measurements
(none yet)

## Next
- build + tests on ss-k5; SS_K6CHECK dev5/e7/e23/e60; callgrind line bench by stage; llvm-mca.
