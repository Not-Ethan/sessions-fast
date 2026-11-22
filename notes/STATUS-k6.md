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
- rebased v1b (tree ss4, image ss-k6-r): k6test 14/14, SS_K6CHECK dev5 OK; lifetimes queued.
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

## Next
- e23/e60 checker -> tag k6-v1 -> image ss-k6-v1, lifetimes 10x e23 + 4x e60 (k4 vs k6 one image).
- stage 1 register pressure (fss from dp via shld, cin in a register, one base register for e/fl, const
  prefetch): written, not yet measured (v1b). Then v3 scorecard + v3 line bench.
- batch surface for the two-loop decode (simple-run iterator over a segment).
