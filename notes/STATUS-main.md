# STATUS of the main engineer (branch ss-opt4, worktree .claude/worktrees/opt4)

Updated: Sun 04:10 ET.

## Now
- Candidate for ss-milestone-18 = 1651a22: #6-A (0304873, verify index by distinct values) + per-partition
  event-ID check (13c3b71) + #22 (1651a22: two builds, baseline execs /sessions-v3 on v3 CPUs; SS_NOEXEC=1
  keeps the baseline). Image ss-a3-x3 on ss-bench. Running: flag-free timing m17 vs x3 (launcher, direct
  /sessions-v3, SS_NOEXEC=1) on e60/e23, then ~/xgate.sh x3 (log ~/xgate_x3.log).
- #6-B (skip the user log for linked long IDs): inline exact probe (SS_KNOWN=1, off) skips 57% of entries
  but costs +150 ms of parse at e60: rejected in that form. A sink-prefetched version is parked until the
  quick-wins helper's #10 (bulk_run) lands.

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
