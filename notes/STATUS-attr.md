# STATUS-attr: session-bitmask attribution (branch `ss-attr`)

Owner: quick-wins helper. Base: ss-opt4 9fdf3fb (= `ss-m22-rc1` base). Touches `src/sess.rs` only (plus this file).

## What it is
`St::attribute_bm`, a third attribution path:
- Who takes it: persons with 1..=64 sessions and at least SS_ATTRBM_MIN (default 1) clicks, in AVX2 builds (the
  /sessions-v3 binary). An upper limit on clicks (SS_ATTRBM_MAX=16, persons with few clicks only) was tried and
  dropped: +3 instructions and -0.11 mispredicts per line, against 0 and -0.27 with no limit.
- SS_ATTRBM=0 turns it off and gives today's paths. The baseline v2 binary (SS_NOEXEC=1) keeps today's code.
- The clicks are not sorted on this path. The sort is still done for attribute_window.

How it works:
- **Per click, any order.** The click's qualifying sessions are found 8 per ymm: start <= ts <= start + 7 days, in
  the same offsets as attribute_small, with unsigned min/max_epu32 compares. Pad lanes have lo = MAX and hi = 0, so
  they never qualify.
- **Registers.** Up to 16 sessions (`bm_clicks_reg::<1 or 2>`), the lanes stay in ymm registers for all of the
  person's clicks. Above that, `bm_click` loads and stores them per click. In the v3 assembly that loop has no
  spills: about 32 instructions per click and active block, 10 per block with no qualifying session.
- **Lane update.** The candidate is ts + 1, masked to the qualifying lanes; "later" means max_epu32 changed.
- **Per session state.** Each session keeps:
  - its latest qualifying ts + 1 (0 = no click yet);
  - the campaign of the first click seen at that ts;
  - a flag set when clicks at that ts name another campaign.

  Each click is compared with the state as it was before that click.
- **Assists.** Per campaign, the masks of its clicks are ORed, so a session counts at most once per campaign. The
  result is counted per day run with popcount; `dend` ends the run.
- **Credit.**
  - No qualifying click: organic.
  - One campaign at the latest ts: that campaign.
  - Several campaigns at the latest ts: the smallest click_id among the clicks at that ts (`keys::click_cmp`), as
    attribute_small does.

## Exactness
`sess::tests::attribute_bm_matches_old_paths` compares attribute_bm, attribute_small and attribute_window cell by
cell in the accumulated cred and assist tables:
- 6000 generated persons, run twice for attribute_bm with two shuffled click orders.
- 1, 2, 3, 5, 7, 8, 9, 12, 16, 17, 32, 40, 63 and 64 sessions within a 24-day window.
- 65 sessions: the old paths only, which is what the dispatch does.
- Clicks exactly at the session start, exactly 7 days before it, and 1 ms outside each side.
- Equal-ts clicks with the same and with different campaigns, repeated clicks, zero clicks.
- Short (packed), long and odd (referenced) click ids, distinct per click as after dedup.
- Plus a five-click tie (three campaigns, long and odd ids) in all 120 arrival orders.

It passes on ARM (scalar lanes) and on x86-64-v3 under Rosetta (AVX2 lanes).

The whole-pipeline checks on ss-gate (fixtures, dev5, e7, e23, e60, f23 with the default, SS_ATTRBM=0,
SS_POISON=1 and SS_NOEXEC=1) are listed under Results when done.

## Results
- **Counts** (cachegrind, dev5, x86-64-v3 profiling build, per event line, SS_ATTRBM on vs 0):
  - whole program 992 vs 992 instructions;
  - St::person 103.7 vs 97.7; the click sort it no longer needs was the other 6;
  - simulated mispredicts 4.30 vs 4.57 for the whole program, 1.2 vs 1.4 in St::person.
- **Lifetimes** (`ab_env.sh`, one image, mirrored, SS_ATTRBM on vs 0):
  - e60: 1900.3 vs 1914.2 ms median, 6+6 rounds: -13.9 ms (-0.73%), new faster in 6 of 6. Ranges 1899-1904 vs
    1904-1916.
  - e23: 726.0 vs 733.9, 6+6 rounds: -7.9 ms (-1.1%), new faster in 4 of 6. No regression.
- **Verdict:** passes the orchestrator's rule (e60 at least 5 of 6 and -8 ms or better, e23 not worse).
- **Pre-check of the rebased tree** (5a9a6ff on ss-opt4 d9332ed, image ss-q-abm2, as reduced by the orchestrator):
  - unit tests 65/65 on ss-gate (x86-64-v2); 46/46 earlier under Rosetta on x86-64-v3;
  - dev5 vs the organisers' answer MATCH, with the default and with SS_ATTRBM=0;
  - e23 and e60 vs ref5 MATCH (default).

  The full matrix (fixtures, e7, f23, POISON, NOEXEC) is left to the milestone 23 gate.
- Tagged attr-bm-v1.

## Expectation (before measuring)
The v3 cachegrind split of St::person on dev5 (98 instructions per event line):
- attribute_small 8.2;
- attribute_window 4.9;
- assist_range 4.1;
- win_add + win_remove 1.0;
- some inlined library code on top;
- about 0.6 simulated mispredicts per line in these functions.

Generated data, 5M events:
- 92% of persons have at most 16 clicks, on average 3.8 clicks and 3.8 sessions (attribute_small);
- the other 8% have on average 53 clicks and 22 sessions (sort + attribute_window).

The bitmask path costs about 25 vector instructions per click and active 8-session block. So it should cut
instructions for small persons only slightly, and may cost more than sort + window for the large ones. Hence
SS_ATTRBM_MAX. The earlier sorted-range variant (p1) and a branch-free variant by another engineer both measured as
noise or worse.
