# STATUS of the main engineer (branch ss-opt4, worktree .claude/worktrees/opt4)

Updated: Sun 07:05 ET (local clock).

## Now
- Tags: m18 = 1651a22 (0.9154 s), m19 = c870b9c (0.8278 s), m20 = 0ed00d5 (0.7904 s),
  m21 = 86d13ad (0.7813 s), ss-m22-rc1 = 9fdf3fb (orchestrator gating on ss-gate).
- rc1 = m21 + release guard 653fdf6 + NCPU clamp 3981ec2 + k5-v4 cd68e89 + 7e214f6 (mixed buckets) +
  k6-v2 (K6_DEFAULT true, 8d3232f) + probe select 9fdf3fb. ss-bench mirrored: e60 1946.4 vs x18 2015.8
  (-69.4), e23 717.0 vs 740.0 (-23.0); SS_K6=0 1990.8 / 731.0.
- x18 = 62babc5 (m21 + release guard) passed xgate22: 91 lines MATCH/OK (fallback).
- m23 candidate: 63ebd99 + 738047c link union root order by 8-byte prefixes (SS_LINKPREF=0 off): exact
  (64/64 v3 tests, dev5/e23/e60 MATCH); vs rc1 ABBA e60 1944.0 vs 1958.7 (-14.6, 12 samples), e23 716.5
  vs 714.8 (+1.7 noise, 20 samples); inside one image LINKPREF on vs off: e60 -14.8, e23 -13.
- Finalize prefetch d659217 reverted (no effect).
- Link prefix one-run e60 sub-phases (x20): LINKPREF on union 49.3 ms / parse 26.9; off union 57.9 / parse 25.7.
- Extra checks on x20 (rc1 + prefix): dev5 LONGBITS=4/6, e23 LONGBITS=6 (3 reruns each), dev5/e23 NOEXEC,
  dev5 NCPU=1: all MATCH.
- The public mirror of the top cluster carries an opt-out notice on every source file: not read, not used,
  not credited (orchestrator withdrew the instruction).

## Measured (ss-bench, a3-m17 = cdadcff, SS_VERBOSE + SS_MEMACCT, one run each)
- e23: user-ID log 3.35M entries / 184K distinct long IDs; event-ID log 2.69M entries; long verify
  35.3 ms (all three verifications); log arenas 252 MB resident.
- e60: user log 9.99M / 555K distinct; event log 7.86M; verify 87.7 ms; log arenas 723 MB resident
  of 4.38 GB RSS after parse; peak RSS 5.1 GB.
- Previous agent's m17 gate (a3-m17, ~/m17_gate.log on ss-bench): all MATCH. Its life test vs m16
  (VERBOSE on): e23 0.984, e7 0.981, e60 0.977 (n=2). Its adversarial run (v5_adv, seed-offset 500,
  ~/w/v5adv17/full.log) all MATCH through case 14 at 03:24 ET.

- #6-A + event-ID check, image ss-a3-x2 (13c3b71), flag-free interleaved, switches restoring the old code
  in the same image: e60 4 rounds: new 2570.6 [2564 2569 2572 2603] vs old 2625.1 [2621 2624 2626 2632]
  (-54.4 ms, -2.1%); only the event check old 2585.2. e23 5 rounds: new 907.4 [906 906 907 912 912] vs
  old 923.4 [918 922 923 924 932] (-16.0 ms, -1.8%); only the event check old 916.0.
  Verbose e60: post-parse verify 57.0 -> 14.0 ms, event-ID check 22.6 -> 12.6 ms, parse faults 220K -> 163K.
- SS_LINKCAP=2 (link table half size): neutral on e60 (-0.2 ms).
- #22 image build (ss-bench): 146 s for both builds in parallel (1m23s each).

- m18 vs exact m17 image (flag-free): e60 4 rounds 2533.2 vs 2646.9 (-4.3%); /sessions-v3 as CMD 2498.6,
  SS_NOEXEC=1 2549.4. e23 6 rounds 878.1 vs 922.6 (-4.8%); direct 880.5; NOEXEC 896.8. Exec cost: none at
  e23; +35 ms at e60 in a fixed order (launcher after m17) -> orchestrator runs an ABBA check.
- m19 (x7) vs m18 (x3), 6 rounds: e60 2290.2 vs 2508.8 (-218.6, -8.7%); SS_K4=0 2464.4; SS_K5=0 2377.7;
  SS_SCALARPROBE=1 2305.7. e23 815.8 vs 881.5 (-65.8, -7.5%); K4=0 866.0; K5=0 838.4; scalar probe 813.5.
- Cachegrind dev5 1 thread Rust 1.100: m17 v2 1.402 G; m18 v3 1.378 G; +TLS 1.369 G; +AVX2 probe 1.3405 G.

- Session 1 (quiet, mirrored, 8 samples each): e60 m19 2287.2, m20(c7c5560) 2182.4, m21(2d599cd)
  2130.4, m21 SS_LEAN=0 2180.8, m21 SS_SUBB=32768 2100.2, pgo-on 2213.9, pgo-off 2187.9.
  e23 m19 810.5, m20 785.7, m21 763.5, m21 LEAN=0 780.8, pgo-on 786.8, pgo-off 781.1.

- The per-Parts overflow counter (6b2a0c5) is probably most of m20's -105 ms on e60 (#8 and #10 alone
  measured -9 and -5): every thread bumped one global atomic per overflowing 64-byte line, and whale
  person partitions overflow heavily (733K lines on e23, about 2.1M on e60). Grep of every other
  fetch_add in worker code (05:25): fastev SLOW/WIDE_N/CWIDE_N/CSLOW, k4 N_*, k5 N_* are all behind
  COUNT / COUNT_SLOW flags (off unless a check mode); prof only with SS_PROF; keys counters once per
  verification; link compact-index blocks once per 4096 IDs. No other shared counter in a hot loop.

- Partition counts on m20 (x12), 6 mirrored samples: e60 default 2083.4, pe11 +5.4, pe10 +52.4, ps9 +20.7;
  e23 all within 2.5 ms.

- Decode-table single probe (x13 vs m20): e23 762.7 vs 768.5 (20 samples), e60 2102.2 vs 2083.7 (6): reverted.

- Click deferral (x14): e60 x14 2081.2 vs CDEFER=1 2108.9 (8 samples); e23 760.0 vs 771.3 (12).
- Record wanted from the click LINE decoder: per valid click {user key u64, mix(user key) u64, click key
  u64 (keys::click_key), ts - (window_start - 7 days) u32, campaign index u32} = 32 bytes, appended in
  input order to a per-thread buffer, no person lookup.

- m21 stack (x15 = without v2f) vs m20, 8 mirrored samples: e60 2057.9 vs 2118.8; e23 748.8 vs 761.7.
  CDEFER=2 (512-click batches): e60 +21, e23 +4 vs x15: rejected. RELEASE=1: e60 +40, e23 +6.
- e60 peak RSS of m21 under --memory 24g: VmHWM 10.03 GB (anon 4.18 GB + mapped input).

## Known slow-but-exact paths (do not "fix" the order)
- Click IDs that do not pack are references (file, length, offset) into the click file mapping. The
  input pages are released with MADV_DONTNEED after the parse and again after every sessions
  partition is done; a tie between two such click IDs with different campaigns re-reads the bytes
  (page cache refault). Rare and slow, but exact. Never munmap the click mappings before the
  sessions partitions finish.

## Next
1. #6 B: decide on SS_KNOWN by e60/e23 flag-free A/B; if the inline probe costs more than it saves,
   try the prefetched variant in the sink (Pending carries the user field offset).
2. #16 / #3 / #4: output buffers, huge pages, nothing zeroed twice (SS_MEMACCT before/after).
3. #13 only with a replay experiment first.
