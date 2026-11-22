# STATUS of the main engineer (branch ss-opt4, worktree .claude/worktrees/opt4)

Updated: Sun 05:00 ET.

## Now
- Tags: ss-milestone-18 = 1651a22 (real scorer 0.9154 s), ss-milestone-19 = c870b9c (0.8278 s).
- m20 candidate = c7c5560: m19 + #8 (e7b9c10) + #10 (d143e10) + #9 reverted (no gain, orchestrator) +
  7fa2005 (one AVX2 probe: links.rs calls k5::probe_view) + 6b2a0c5 (overflow-line statistic per Parts,
  not a contended global atomic; line_over was 1.6% of perf samples). #18 to be added when confirmed.
- m21 candidate = 2d599cd lean sink (panel row 5; SS_LEAN=0 = BSink) + 9a0a833 (SS_SUBB experiment).
- Queued on ss-bench (sess1.sh): images x9/x10/x11, PGO experiment (pgo.sh: instrumented v3, trained on
  dev5+e7, images ss-pgo-on/off), then a quiet mirrored timing session m19|m20|m21|lean0|subb32k|pgo.
- #6-B rejected (orchestrator). Row 7 (table layout) waits for the orchestrator's engine-design note.

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
