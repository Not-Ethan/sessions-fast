//! SS_CDECOMP=1: cumulative click-parse stages over clicks/, best of 3, on 1
//! and NCPU threads, as thread-time per click line (each stage includes
//! everything before it; stages 1-4 decode every line, the full task only
//! what survives validation).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use crate::links::LinkTable;
use crate::parts::Parts;
use crate::{click_task, keys, par, scan, sys, CRec, CSink, Ctx, Input, LocBits, PCache, Task, PHASE_CPU_NS};

struct NullC(u64);

impl CSink for NullC {
    #[inline(always)]
    fn push_click(&mut self, q: usize, r: CRec) {
        self.0 = self.0.wrapping_add(r.person ^ r.ck ^ r.ts as u64 ^ q as u64);
    }
}

#[inline(always)]
fn stage_task<const AVX2: bool>(stage: u32, b: &[u8], task: Task, ctx: &Ctx, lt: &LinkTable, c: &mut Parts<CRec>, cache: &mut PCache) -> u64 {
    let end = b.as_ptr() as usize + b.len();
    let mut acc = 0u64;
    match stage {
        0 => scan::for_each_record_x::<4, _, AVX2>(b, task.start, task.end, |f| acc += f[3].len() as u64),
        1..=4 => scan::for_each_record_x::<4, _, AVX2>(b, task.start, task.end, |f| {
            if f[0].is_empty() || f[1].is_empty() || f[3].as_ptr() as usize + 32 > end {
                return;
            }
            acc = acc.wrapping_add(unsafe { keys::key_of_wide(f[0]) });
            if stage >= 2 {
                acc = acc.wrapping_add(keys::mix(unsafe { keys::key_of_wide(f[1]) }));
            }
            if stage >= 3 {
                acc = acc.wrapping_add(unsafe { ctx.campaigns.get_wide(f[2]) } as u64);
            }
            if stage >= 4 {
                acc = acc.wrapping_add(ctx.tsp.parse(f[3]).unwrap_or(0) as u64);
            }
        }),
        5 => {
            let mut n = NullC(0);
            click_task::<AVX2, _>(b, task, ctx, lt, &mut n, end, cache);
            acc = n.0;
        }
        _ => click_task::<AVX2, _>(b, task, ctx, lt, c, end, cache),
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
unsafe fn stage_task_avx2(stage: u32, b: &[u8], task: Task, ctx: &Ctx, lt: &LinkTable, c: &mut Parts<CRec>, cache: &mut PCache) -> u64 {
    stage_task::<true>(stage, b, task, ctx, lt, c, cache)
}

pub fn click_decomp(cl: &Input, links_in: &Input, ctx: &Ctx) {
    let mut lines = 0usize;
    for t in &cl.tasks {
        let b = cl.maps[t.file as usize].bytes();
        lines += b[t.start..t.end].iter().filter(|&&c| c == b'\n').count();
    }
    let lb = LocBits::new(ctx.threads, ctx.ps_bits).unwrap();
    let empty = Input { maps: Vec::new(), tasks: Vec::new(), bytes: 0 };
    let (lt, _) = crate::parse_b(links_in, &empty, &empty, ctx, lb);
    let keep = sys::arenas().len();
    let ps = 1usize << ctx.ps_bits;
    let est = cl.bytes / 44 + 1;
    let names = ["split", "+click_id", "+user_id", "+campaign", "+ts", "+person lookup", "+scatter (full task)"];
    for stage in 0..7u32 {
        for &nt in &[1usize, ctx.threads] {
            let mut best = (f64::MAX, 0f64);
            for _ in 0..3 {
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
                        a = a.wrapping_add(if ctx.avx2 {
                            #[cfg(target_arch = "x86_64")]
                            unsafe {
                                stage_task_avx2(stage, b, task, ctx, &lt, &mut c, &mut cache)
                            }
                            #[cfg(not(target_arch = "x86_64"))]
                            0
                        } else {
                            stage_task::<false>(stage, b, task, ctx, &lt, &mut c, &mut cache)
                        });
                    }
                    c.finish();
                    a ^ c.total() as u64
                });
                let wall = t.elapsed().as_secs_f64() * 1e3;
                let cpu = PHASE_CPU_NS.swap(0, Ordering::Relaxed) as f64 / 1e6;
                std::hint::black_box(accs);
                sys::release_from(keep);
                if wall < best.0 {
                    best = (wall, cpu);
                }
            }
            eprintln!("cdecomp {:22} threads={} wall={:7.1}ms thread-time/line={:6.1}ns", names[stage as usize], nt, best.0, best.1 * 1e6 / lines as f64);
        }
    }
    eprintln!("cdecomp: {} click lines, {} MB", lines, cl.bytes >> 20);
}
