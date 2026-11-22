#![cfg_attr(not(target_arch = "x86_64"), allow(dead_code, unused_imports))]
//! k5: a fused click-line kernel for `main::click_task` (SS_K5).
//!
//! (work in progress: the stage bench, SS_K5BENCH=1)

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

/// k5 on when SS_K5 is not set.
pub const DEFAULT: bool = false;
/// click tasks through k5 (SS_K5; needs AVX2, see `main::click_task`).
pub static ON: AtomicBool = AtomicBool::new(false);

pub fn check(_links_in: &Input, _cl: &Input, _ctx: &Ctx) {
    eprintln!("k5check: not yet");
}

use super::{click_task, keys, par, sys, CRec, CSink, Ctx, Input, LocBits, PCache, Task, NONE, PHASE_CPU_NS, WEEK};
use crate::fastev;
use crate::links::LinkTable;
use crate::parts::Parts;

// ---------------------------------------------------------------- stage bench

struct NullC(u64);

impl CSink for NullC {
    #[inline(always)]
    fn push_click(&mut self, q: usize, r: CRec) {
        self.0 = self.0.wrapping_add(r.person ^ r.ck ^ r.ts as u64 ^ q as u64 ^ r.camp as u64);
    }
}

/// Cumulative stages of the old click path on the real splitter, with the
/// real early exits (a stage stops a line where the old path would).
#[inline(always)]
fn bstage<const AVX2: bool>(stage: u32, b: &[u8], task: Task, ctx: &Ctx, lt: &LinkTable, c: &mut Parts<CRec>, cache: &mut PCache) -> u64 {
    let safe_end = b.as_ptr() as usize + b.len();
    let mut acc = 0u64;
    let we = ctx.we;
    let cws = ctx.ws - WEEK;
    match stage {
        0 => fastev::for_each_click::<AVX2, _>(b, task.start, task.end, safe_end, &mut |f: &[&[u8]; 4]| acc = acc.wrapping_add(f[3].len() as u64 ^ f[0].len() as u64)),
        1..=4 => fastev::for_each_click::<AVX2, _>(b, task.start, task.end, safe_end, &mut |f: &[&[u8]; 4]| {
            if f[0].is_empty() || f[1].is_empty() {
                return;
            }
            let wide = f[3].as_ptr() as usize + 32 <= safe_end;
            let Some(t) = ctx.tsp.parse(f[3]) else { return };
            if t < cws || t >= we {
                return;
            }
            acc = acc.wrapping_add(t as u64);
            if stage >= 2 {
                let camp = if wide { unsafe { ctx.campaigns.get_wide(f[2]) } } else { ctx.campaigns.get(f[2]) };
                if camp == NONE {
                    return;
                }
                acc = acc.wrapping_add(camp as u64);
            }
            if stage >= 3 {
                let user = if wide { unsafe { keys::key_of_wide(f[1]) } } else { keys::key_of(f[1]) };
                acc = acc.wrapping_add(keys::mix(user));
            }
            if stage >= 4 {
                acc = acc.wrapping_add(keys::click_key(task.file, b.as_ptr(), f[0], wide));
            }
        }),
        5 => {
            let mut n = NullC(0);
            click_task::<AVX2, _>(b, task, ctx, lt, &mut n, safe_end, cache);
            acc = n.0;
        }
        _ => click_task::<AVX2, _>(b, task, ctx, lt, c, safe_end, cache),
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
unsafe fn bstage_avx2(stage: u32, b: &[u8], task: Task, ctx: &Ctx, lt: &LinkTable, c: &mut Parts<CRec>, cache: &mut PCache) -> u64 {
    bstage::<true>(stage, b, task, ctx, lt, c, cache)
}

/// SS_K5BENCH=1: thread time per click line of cumulative stages of the old
/// path at 1 and ctx.threads threads (every run printed; the first run of
/// stage 0 also maps the pages).
pub fn bench(cl: &Input, links_in: &Input, ctx: &Ctx) {
    let mut lines = 0usize;
    for t in &cl.tasks {
        let b = cl.maps[t.file as usize].bytes();
        lines += b[t.start..t.end].iter().filter(|&&c| c == b'\n').count();
    }
    let lb = LocBits::new(ctx.threads, ctx.ps_bits).unwrap();
    let empty = Input { maps: Vec::new(), tasks: Vec::new(), bytes: 0 };
    let (lt, _) = crate::parse_b(links_in, &empty, &empty, ctx, lb, None);
    let keep = sys::arenas().len();
    let ps = 1usize << ctx.ps_bits;
    let est = cl.bytes / 44 + 1;
    let names = ["split", "+ts+window", "+campaign", "+user key+mix", "+click key", "+person (null sink)", "+sink (full task)", "k5 (null sink)", "k5 full"];
    let stages: Vec<u32> = std::env::var("SS_K5STAGES").ok().map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect()).unwrap_or((0..7).collect());
    for &stage in &stages {
        for &nt in &[ctx.threads] {
            let mut runs = Vec::new();
            for _ in 0..5 {
                let next = AtomicUsize::new(0);
                PHASE_CPU_NS.store(0, Ordering::Relaxed);
                let t = Instant::now();
                let accs = par(nt, |_| {
                    let mut c: Parts<CRec> = Parts::with_wc(ps, est / (nt * ps) * 5 / 4 + 32, true);
                    let mut cache = PCache::new();
                    let mut a = 0u64;
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(&task) = cl.tasks.get(i) else { break };
                        let b = cl.maps[task.file as usize].bytes();
                        a = a.wrapping_add(run_stage(stage, b, task, ctx, &lt, &mut c, &mut cache));
                    }
                    c.finish();
                    a ^ c.total() as u64
                });
                let wall = t.elapsed().as_secs_f64() * 1e3;
                let cpu = PHASE_CPU_NS.swap(0, Ordering::Relaxed) as f64 / 1e6;
                std::hint::black_box(accs);
                sys::release_from(keep);
                runs.push((wall, cpu * 1e6 / lines as f64));
            }
            let best = runs.iter().cloned().fold((f64::MAX, 0.0), |a, r| if r.0 < a.0 { r } else { a });
            let all: Vec<String> = runs.iter().map(|r| format!("{:.1}", r.1)).collect();
            eprintln!("k5bench {:22} threads={} best wall={:6.1}ms {:6.1}ns/line  (all ns/line: {})", names[stage as usize], nt, best.0, best.1, all.join(" "));
        }
    }
    eprintln!("k5bench: {} click lines, {} MB", lines, cl.bytes >> 20);
}

fn run_stage(stage: u32, b: &[u8], task: Task, ctx: &Ctx, lt: &LinkTable, c: &mut Parts<CRec>, cache: &mut PCache) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        if ctx.avx2 {
            return unsafe { bstage_avx2(stage, b, task, ctx, lt, c, cache) };
        }
    }
    bstage::<false>(stage, b, task, ctx, lt, c, cache)
}
