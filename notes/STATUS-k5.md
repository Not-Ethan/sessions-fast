# k5 click kernel: status (restart insurance)

Branch ss-k5, worktree /Users/Ethan_1/buildersbuildathon/.claude/worktrees/k5, VM ss-k5 (src at /home/Ethan/k5/shopping-sessions,
sync with scratch/sync.sh in the worktree (not committed: tar + gcloud scp + untar)).
VM scripts: /home/Ethan/k5cg.sh STAGES TASKS TAG (callgrind of the stage bench, only inside bstage_avx2, 1 thread, e23),
/home/Ethan/k5chk.sh DATASETS (release build + SS_K5CHECK=1).

## Facts
- Rust 1.85 (Dockerfile): AVX-512 intrinsics / target_feature avx512 unstable before 1.89 -> k5 is AVX2+BMI2.
- Hook: main.rs `click_task_avx2`: `if k5::wanted(cache) { return k5::task(b, task, ctx, lt, c); }` (SS_K5=1 and PCache off).

## Instructions per click line (callgrind, e23 first 6 tasks = 100,390 lines, 1 thread)
| stage | old | k5 v1 |
|---|---|---|
| split (cut4+loop 88, cut4_wide 50, scanner 13) | 151 | |
| timestamp | ~60 | |
| campaign | ~62 | |
| user key + click key (2x pack_wide ~45, long ~15) | ~105 | |
| person lookup (probe 54, loads/mix 16) | ~70 | |
| sink | ~32 | |
| glue (closure call per line 142, emit 31) | ~174 | |
| total | 694 | |

Old-path root causes: per-line closure not inlined (call per line, fields via memory, 40-byte ring copies);
cut4_wide<AVX2> #[inline(never)] without target_feature -> masks64_avx2 called out of line twice per wide line.

## Stage bench (previous helper) e23 8 threads thread-ns/line cumulative
split 26.4; +ts 24.0; +campaign 13.6; +user key 37.8; +click key 24.6; +person 35.2; +sink 23.0; total 184.6.

## Log
- 03:25 rebased onto origin/ss-opt4 5b7bc5d. 03:50 scoreboard sent. 04:10 k5 v1 written (fused loop, pack2, date-table ts,
  get_wide campaign, ring 16), checker SS_K5CHECK; running on dev5.

## Next
checker dev5/e7/e23/e60 -> callgrind k5 (stage 8) -> campaign hash / probe -> tests k5test.rs -> A/B.
