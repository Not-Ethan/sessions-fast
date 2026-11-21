//! Dense IDs (generator structure, see notes/dense-ids.md): canonical IDs
//! `<prefix><1-8 lowercase base-36 digits>` (no leading zero unless the
//! number is 0, value < 2^40) decode to 40-bit numbers; unscrambling with the
//! data set's key makes event numbers dense. Decoding + unscrambling is an
//! injection on canonical strings for any key, so exactness never depends on
//! the key; the key only decides density.
//!
//! STEP A (measurement only, SS_DENSEBENCH=1): key recovery, event parse
//! with both IDs decoded into one bucketed 32-byte record stream, direct
//! placement with collision dedup per bucket, and person lookups in event
//! number order. Prints timings; produces no output.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use super::{ingest_rel, Ctx, Input, Task, LATE};
use crate::parts::Parts;
use crate::{keys, scan};

pub const MASK: u64 = (1 << 40) - 1;
pub const M1: u64 = 0x9E37_79B9_7F4B;
pub const M2: u64 = 0xD6E8_FEB8_6659;

/// Inverse of an odd number modulo 2^40 (Newton iteration).
pub const fn inv40(a: u64) -> u64 {
    let mut x = a; // correct to 3 bits for odd a
    let mut i = 0;
    while i < 6 {
        x = x.wrapping_mul(2u64.wrapping_sub(a.wrapping_mul(x)));
        i += 1;
    }
    x & MASK
}
pub const M1I: u64 = inv40(M1);
pub const M2I: u64 = inv40(M2);
const _: () = assert!(M1.wrapping_mul(M1I) & MASK == 1 && M2.wrapping_mul(M2I) & MASK == 1);

/// base-36 digit value of a byte, 255 if not `0-9a-z`.
static DIG: [u8; 256] = {
    let mut t = [255u8; 256];
    let mut i = 0;
    while i < 10 {
        t[b'0' as usize + i] = i as u8;
        i += 1;
    }
    let mut i = 0;
    while i < 26 {
        t[b'a' as usize + i] = 10 + i as u8;
        i += 1;
    }
    t
};

/// The 40-bit number of a canonical ID with this prefix, else None.
#[inline(always)]
pub fn decode36(s: &[u8], prefix: u8) -> Option<u64> {
    let n = s.len();
    if !(2..=9).contains(&n) || s[0] != prefix || (n > 2 && s[1] == b'0') {
        return None;
    }
    let mut v = 0u64;
    let mut bad = 0u8;
    for &c in &s[1..] {
        let d = DIG[c as usize];
        bad |= d;
        v = v * 36 + (d & 63) as u64;
    }
    if bad == 255 || bad & 0x80 != 0 || v > MASK {
        return None;
    }
    Some(v)
}

/// (v * M1) ^ key from a scrambled value v: undoes the second multiply and
/// the xorshift.
#[inline(always)]
pub fn half(v: u64) -> u64 {
    let t = v.wrapping_mul(M2I) & MASK;
    t ^ (t >> 17) ^ (t >> 34)
}

/// The number scrambled into v under `key`.
#[inline(always)]
pub fn unscramble(v: u64, key: u64) -> u64 {
    ((half(v) ^ key).wrapping_mul(M1I)) & MASK
}

/// Finds a key under which every sampled scrambled value decodes below
/// `bound`, trying n = 0, step, 2 step, ... for each of a few samples as the
/// sample's own number (in parallel). None if there is no such key.
pub fn recover(vals: &[u64], bound: u64, step: u64, threads: usize) -> Option<u64> {
    if vals.len() < 8 {
        return None;
    }
    let halves: Vec<u64> = vals.iter().map(|&v| half(v)).collect();
    let found = AtomicU64::new(u64::MAX);
    for a in 0..halves.len().min(6) {
        let ha = halves[a];
        let others: Vec<u64> = halves.iter().enumerate().filter(|&(i, _)| i != a).map(|(_, &h)| h).collect();
        let cands = bound.div_ceil(step);
        let per = cands.div_ceil(threads as u64);
        super::par(threads, |t| {
            let lo = t as u64 * per;
            let hi = (lo + per).min(cands);
            let mut i = lo;
            while i < hi {
                if found.load(Ordering::Relaxed) != u64::MAX {
                    return;
                }
                // 64 candidates per check of the flag
                let stop = (i + 64).min(hi);
                while i < stop {
                    let k = ha ^ (M1.wrapping_mul(i * step) & MASK);
                    if ((others[0] ^ k).wrapping_mul(M1I) & MASK) < bound
                        && ((others[1] ^ k).wrapping_mul(M1I) & MASK) < bound
                        && ((others[2] ^ k).wrapping_mul(M1I) & MASK) < bound
                        && others.iter().all(|&h| ((h ^ k).wrapping_mul(M1I) & MASK) < bound)
                    {
                        found.store(k, Ordering::Relaxed);
                        return;
                    }
                    i += 1;
                }
            }
        });
        let k = found.load(Ordering::Relaxed);
        if k != u64::MAX {
            return Some(k);
        }
    }
    None
}

/// One event copy in the dense stream (32 bytes), bucketed by n >> BSHIFT.
#[derive(Clone, Copy, Default)]
#[repr(C)]
struct DRec {
    /// user key (as the link table knows it); 0 for an invalid copy
    user: u64,
    n: u32,
    ingest: u32,
    ts: u32,
    pay: u32,
    _pad: u64,
}

const BSHIFT: u32 = 16;

/// Measurement-only 16-byte record: number, ingest, ts, and 32 bits of
/// payload (stand-in for a person slot; pay would go to a side stream).
#[derive(Clone, Copy, Default)]
#[repr(C)]
struct DRec16 {
    n: u32,
    ingest: u32,
    ts: u32,
    aux: u32,
}

/// SS_DENSEBENCH=16: parse into 16-byte records, lean placement (gather the
/// bucket, u32 index slots), no person lookups.
pub fn bench16(ev: &Input, ctx: &Ctx, ke: u64, bound: u64) {
    let est = ev.bytes / 70 + 1;
    let nb = (bound >> BSHIFT) as usize + 1;
    let threads = ctx.threads;
    let cap = est / (threads * nb) * 5 / 4 + 64;
    let keep = crate::sys::arenas().len();
    for round in 0..3 {
        let next = AtomicUsize::new(0);
        let tp = Instant::now();
        let parts: Vec<Parts<DRec16>> = super::par(threads, |_| {
            let mut p: Parts<DRec16> = Parts::with_wc(nb, cap, true);
            loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(&task) = ev.tasks.get(i) else { break };
                let b = ev.maps[task.file as usize].bytes();
                let safe_end = b.as_ptr() as usize + b.len();
                task16_any(b, task, ctx, ke, bound, &mut p, safe_end);
            }
            p.finish();
            p
        });
        let t_parse = tp.elapsed().as_secs_f64() * 1e3;
        let tq = Instant::now();
        let nextb = AtomicUsize::new(0);
        let kept = AtomicUsize::new(0);
        super::par(threads, |_| {
            let mut slot = vec![0u32; 1 << BSHIFT];
            let mut gat: Vec<DRec16> = Vec::new();
            let mut k = 0usize;
            loop {
                let bi = nextb.fetch_add(1, Ordering::Relaxed);
                if bi >= nb {
                    break;
                }
                gat.truncate(0);
                for p in parts.iter() {
                    for sl in p.get(bi) {
                        gat.extend_from_slice(sl);
                    }
                }
                for (i, r) in gat.iter().enumerate() {
                    let s = &mut slot[(r.n as usize) & ((1 << BSHIFT) - 1)];
                    if *s == 0 || r.ingest < gat[*s as usize - 1].ingest {
                        *s = i as u32 + 1;
                    }
                }
                for s in slot.iter_mut() {
                    if *s != 0 {
                        k += 1;
                        *s = 0;
                    }
                }
            }
            kept.fetch_add(k, Ordering::Relaxed);
        });
        let t_place = tq.elapsed().as_secs_f64() * 1e3;
        eprintln!("dense16 round {}: parse events {:.1} ms | placement+dedup {:.1} ms | kept {}", round, t_parse, t_place, kept.load(Ordering::Relaxed));
        std::mem::forget(parts);
        crate::sys::release_from(keep);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[inline(never)]
unsafe fn task16_avx2(b: &[u8], task: Task, ctx: &Ctx, ke: u64, bound: u64, p: &mut Parts<DRec16>, safe_end: usize) {
    task16_x::<true>(b, task, ctx, ke, bound, p, safe_end)
}

fn task16_any(b: &[u8], task: Task, ctx: &Ctx, ke: u64, bound: u64, p: &mut Parts<DRec16>, safe_end: usize) {
    if ctx.avx2 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            return task16_avx2(b, task, ctx, ke, bound, p, safe_end);
        }
    }
    task16_x::<false>(b, task, ctx, ke, bound, p, safe_end)
}

#[inline(always)]
fn task16_x<const AVX2: bool>(b: &[u8], task: Task, ctx: &Ctx, ke: u64, bound: u64, p: &mut Parts<DRec16>, safe_end: usize) {
    let pp: *mut Parts<DRec16> = p;
    let line = |f: &[&[u8]; 7]| {
        if f[0].is_empty() {
            return;
        }
        let (ws, we) = (ctx.ws, ctx.we);
        let Some(ing) = ctx.tsp.parse(f[3]) else { return };
        let n = match decode36(f[0], b'e') {
            Some(v) => unscramble(v, ke),
            None => u64::MAX,
        };
        if n >= bound {
            return;
        }
        let ingest = ingest_rel(ing, ctx.ws_s);
        let mut r = DRec16 { n: n as u32, ingest, ts: 0, aux: 0 };
        'v: {
            if f[1].is_empty() || f[5].is_empty() {
                break 'v;
            }
            let Some(t) = ctx.tsp.parse(f[2]) else { break 'v };
            if t < ws || t >= we {
                break 'v;
            }
            let Some(pay) = super::event_pay(f[4], || f[6], f[5], ctx, true) else { break 'v };
            if ing - t > LATE {
                break 'v;
            }
            let u = unsafe { keys::key_of_wide(f[1]) };
            r.ts = (t - ws) as u32 | (1 << 31);
            r.aux = (u as u32) ^ pay;
        }
        unsafe { (*pp).push((n >> BSHIFT) as usize, r) };
    };
    let lp = &line;
    crate::fastev::for_each_line2::<AVX2, _, _>(b, task.start, task.end, safe_end, |f| lp(f), |from, to| {
        scan::for_each_record_x::<7, _, AVX2>(b, from, to, |f| lp(f));
    });
}

/// Sample canonical (event, user) values from the first lines of the files.
fn sample(ev: &Input) -> (Vec<u64>, Vec<u64>) {
    let (mut e, mut u) = (Vec::new(), Vec::new());
    for task in &ev.tasks {
        if e.len() >= 64 {
            break;
        }
        let b = ev.maps[task.file as usize].bytes();
        let end = scan::next_line_start(b, (task.start + 16384).min(task.end).max(task.start)).min(task.end);
        scan::for_each_record_x::<7, _, false>(b, task.start, end, |f| {
            if e.len() < 64 {
                if let Some(v) = decode36(f[0], b'e') {
                    e.push(v);
                }
                if let Some(v) = decode36(f[1], b'u') {
                    u.push(v);
                }
            }
        });
    }
    (e, u)
}

pub fn bench(links_in: &Input, ev: &Input, ctx: &Ctx) {
    let t0 = Instant::now();
    let est = ev.bytes / 70 + 1;
    let bound = (2 * est as u64).max(1 << 16);
    let (es, us) = sample(ev);
    let ke = recover(&es, bound, 1, ctx.threads);
    let t_key = t0.elapsed().as_secs_f64() * 1e3;
    let ubound = 512 * (est as u64 / 4).max(1 << 12);
    let t1 = Instant::now();
    let ku = recover(&us, ubound, 512, ctx.threads);
    let t_ukey = t1.elapsed().as_secs_f64() * 1e3;
    eprintln!("dense: event key {:?} in {:.1} ms, user key {:?} in {:.1} ms (bound {}, samples {}/{})", ke.map(|k| format!("{:x}", k)), t_key, ku.map(|k| format!("{:x}", k)), t_ukey, bound, es.len(), us.len());
    let Some(ke) = ke else { return };
    if std::env::var("SS_DENSEBENCH").map_or(false, |v| v == "16") {
        return bench16(ev, ctx, ke, bound);
    }
    let lb = super::LocBits::new(ctx.threads, ctx.ps_bits).unwrap();
    let empty = Input { maps: Vec::new(), tasks: Vec::new(), bytes: 0 };
    let tl = Instant::now();
    let (lt, _) = super::parse_b(links_in, &empty, &empty, ctx, lb, None);
    let t_links = tl.elapsed().as_secs_f64() * 1e3;
    let keep = crate::sys::arenas().len();
    let nb = (bound >> BSHIFT) as usize + 1;
    let threads = ctx.threads;
    let cap = est / (threads * nb) * 5 / 4 + 64;
    let exc = AtomicUsize::new(0);
    for round in 0..3 {
        let next = AtomicUsize::new(0);
        let tp = Instant::now();
        let parts: Vec<Parts<DRec>> = super::par(threads, |_| {
            let mut p: Parts<DRec> = Parts::with_wc(nb, cap, true);
            let mut x = 0usize;
            loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(&task) = ev.tasks.get(i) else { break };
                let b = ev.maps[task.file as usize].bytes();
                let safe_end = b.as_ptr() as usize + b.len();
                task_any(b, task, ctx, ke, bound, &mut p, &mut x, safe_end);
            }
            p.finish();
            exc.fetch_add(x, Ordering::Relaxed);
            p
        });
        let t_parse = tp.elapsed().as_secs_f64() * 1e3;
        // placement + n-ordered person lookups
        let tq = Instant::now();
        let nextb = AtomicUsize::new(0);
        let kept = AtomicUsize::new(0);
        let acc = AtomicU64::new(0);
        super::par(threads, |_| {
            let mut slot = vec![0u64; 1 << BSHIFT];
            let mut outv: Vec<(u64, u32, u32)> = Vec::with_capacity(1 << BSHIFT);
            let mut a = 0u64;
            let mut k = 0usize;
            let mut last = (0u64, 0u64);
            loop {
                let bi = nextb.fetch_add(1, Ordering::Relaxed);
                if bi >= nb {
                    break;
                }
                slot.iter_mut().for_each(|s| *s = 0);
                // slot = (ingest + 1) << 32 | (thread << 26 | index) (bench: index only)
                for (t, p) in parts.iter().enumerate() {
                    for (h, sl) in p.get(bi).iter().enumerate() {
                        for (i, r) in sl.iter().enumerate() {
                            let s = &mut slot[(r.n as usize) & ((1 << BSHIFT) - 1)];
                            let enc = ((r.ingest as u64 + 1) << 32) | ((t as u64) << 27) | ((h as u64) << 26) | i as u64;
                            if *s == 0 || enc < *s {
                                *s = enc;
                            }
                        }
                    }
                }
                outv.clear();
                for &s in slot.iter() {
                    if s == 0 {
                        continue;
                    }
                    let t = ((s >> 27) & 31) as usize;
                    let h = ((s >> 26) & 1) as usize;
                    let i = (s & ((1 << 26) - 1)) as usize;
                    let r = parts[t].get(bi)[h][i];
                    k += 1;
                    if r.user != 0 {
                        // person lookup in event-number order (consecutive users repeat)
                        let p = if r.user == last.0 { last.1 } else { lt.person_hash(r.user, keys::mix(r.user)) };
                        last = (r.user, p);
                        outv.push((p, r.ts, r.pay));
                    }
                }
                for o in &outv {
                    a = a.wrapping_add(o.0 ^ o.1 as u64);
                }
            }
            kept.fetch_add(k, Ordering::Relaxed);
            acc.fetch_add(a, Ordering::Relaxed);
        });
        let t_place = tq.elapsed().as_secs_f64() * 1e3;
        let tot: usize = parts.iter().map(|p| p.total()).sum();
        eprintln!(
            "dense round {}: links {:.1} ms (once) | parse events {:.1} ms | placement+dedup+person {:.1} ms | copies {} kept {} exceptional {} | buckets {}",
            round,
            t_links,
            t_parse,
            t_place,
            tot,
            kept.load(Ordering::Relaxed),
            exc.swap(0, Ordering::Relaxed),
            nb
        );
        std::mem::forget(parts);
        crate::sys::release_from(keep);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[inline(never)]
unsafe fn task_avx2(b: &[u8], task: Task, ctx: &Ctx, ke: u64, bound: u64, p: &mut Parts<DRec>, x: &mut usize, safe_end: usize) {
    task_x::<true>(b, task, ctx, ke, bound, p, x, safe_end)
}

#[allow(clippy::too_many_arguments)]
fn task_any(b: &[u8], task: Task, ctx: &Ctx, ke: u64, bound: u64, p: &mut Parts<DRec>, x: &mut usize, safe_end: usize) {
    if ctx.avx2 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            return task_avx2(b, task, ctx, ke, bound, p, x, safe_end);
        }
    }
    task_x::<false>(b, task, ctx, ke, bound, p, x, safe_end)
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn task_x<const AVX2: bool>(b: &[u8], task: Task, ctx: &Ctx, ke: u64, bound: u64, p: &mut Parts<DRec>, x: &mut usize, safe_end: usize) {
    let pp: *mut Parts<DRec> = p;
    let xp: *mut usize = x;
    let line = |f: &[&[u8]; 7]| {
        if f[0].is_empty() {
            return;
        }
        let (ws, we) = (ctx.ws, ctx.we);
        let Some(ing) = ctx.tsp.parse(f[3]) else { return };
        let n = match decode36(f[0], b'e') {
            Some(v) => unscramble(v, ke),
            None => u64::MAX,
        };
        if n >= bound {
            unsafe { *xp += 1 };
            return;
        }
        let ingest = ingest_rel(ing, ctx.ws_s);
        let mut r = DRec { user: 0, n: n as u32, ingest, ts: 0, pay: 0, _pad: 0 };
        'v: {
            if f[1].is_empty() || f[5].is_empty() {
                break 'v;
            }
            let Some(t) = ctx.tsp.parse(f[2]) else { break 'v };
            if t < ws || t >= we {
                break 'v;
            }
            let Some(pay) = super::event_pay(f[4], || f[6], f[5], ctx, true) else { break 'v };
            if ing - t > LATE {
                break 'v;
            }
            r.user = unsafe { keys::key_of_wide(f[1]) };
            r.ts = (t - ws) as u32;
            r.pay = pay;
        }
        unsafe { (*pp).push((n >> BSHIFT) as usize, r) };
    };
    let lp = &line;
    crate::fastev::for_each_line2::<AVX2, _, _>(b, task.start, task.end, safe_end, |f| lp(f), |from, to| {
        scan::for_each_record_x::<7, _, AVX2>(b, from, to, |f| lp(f));
    });
}
