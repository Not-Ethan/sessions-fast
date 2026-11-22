# STATUS of the main engineer (branch ss-opt4, worktree .claude/worktrees/opt4)

Updated: Sun 04:50 ET.

## Now
- ss-milestone-18 = 1651a22 TAGGED (gate ~/xgate_x3.log on ss-bench: 32 MATCH, nothing else).
- Milestone 19 candidate = c870b9c (m18 + 9da10b2 TLS + 41c0a81 AVX2 probe + k5 click kernel + k4 event
  kernel, both on by default + k5 test lock). Image ss-a3-x7. Gate ~/xgate19.sh x7 running (log
  ~/xgate_x7.log); tag ss-milestone-19 when it passes.
- Milestone 20 candidate in preparation = d143e10 (c870b9c + quick-wins #8 e7b9c10, #9 a367c84,
  #10 d143e10). Image ss-a3-x8.
- #6-B parked: estimated only 7-9 ms at e60 after A; needs k4/k5/BSink changes to pass the value.

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
