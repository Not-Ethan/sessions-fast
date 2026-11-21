//! Dense pipeline (SS_DENSE=1; SS_DENSE=force skips the size and density
//! checks, for tests). Uses the generator's ID structure (dense.rs) for
//! speed only; every result is the general pipeline's by construction:
//!
//! * IDs: a canonical event ID decodes + unscrambles to its event number n
//!   (an injection for any key). Copies with n < bound go to the dense path;
//!   any other event ID ("exceptional") keeps its exact 64-bit key and is
//!   deduplicated on its own (a different string is a different event).
//! * Users: a canonical user ID whose number u = person*512 + alias has
//!   alias < 8 and person < pmax gets slot person*8 + alias; every other user
//!   ID gets an exceptional slot xbase + id, interned by its exact key. Slots
//!   are an injection of user strings.
//! * Persons: from the link table (the links' answer for the row's own user
//!   ID): slot -> root slot; unlinked IDs are their own person. A person's
//!   name is its root's string (re-encoded for canonical roots).
//! * Events: one 16-byte record per copy {n, ingest, ts|valid, user slot |
//!   purchase flag}, bucketed by n >> 16; purchases' pay in a side stream.
//!   Per bucket, the copy with the smallest ingest wins (dedup before
//!   validation: an invalid winner drops the event); kept valid events come
//!   out in event-number order with their person.
//! * Grouping: in event-number order a person's events are mostly one run.
//!   Persons with exactly one run (counted over all buckets) are processed
//!   at once; every other person (several runs, runs cut by bucket borders,
//!   exceptional events) is gathered, sorted by person and processed whole.
//!   Clicks are grouped by person (counting sort). Each person goes through
//!   sess.rs's exact per-person routine exactly once, with all its events
//!   and all its clicks.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::Mutex;

use super::{event_pay, ingest_rel, Ctx, Input, SRec, Task, LATE};
use crate::dense::{decode36, recover, unscramble, M1, M2, MASK};
use crate::links::LinkTable;
use crate::lookup::NONE;
use crate::parts::Parts;
use crate::sess::{Cl, Out2, PersonSess};
use crate::{fastev, keys, scan};

const BSHIFT: u32 = 16;
const BSIZE: usize = 1 << BSHIFT;
const VALID: u32 = 1 << 31;
const PFLAG: u32 = 1 << 31;

/// Keys and bounds of a dense run.
#[derive(Clone, Copy)]
pub struct Dense {
    pub ke: u64,
    pub ku: u64,
    /// event numbers below this are dense
    pub bound: u64,
    /// person indexes below this get dense slots
    pub pmax: u64,
}

impl Dense {
    #[inline(always)]
    fn xbase(&self) -> u32 {
        (self.pmax * 8) as u32
    }
}

/// scramble (the generator's), to re-encode a canonical user ID.
fn scramble(v: u64, key: u64) -> u64 {
    let mut m = (v & MASK).wrapping_mul(M1) & MASK;
    m = (m ^ key) & MASK;
    m ^= m >> 17;
    m.wrapping_mul(M2) & MASK
}

fn base36(out: &mut Vec<u8>, mut v: u64) {
    let mut buf = [0u8; 13];
    let mut at = buf.len();
    loop {
        at -= 1;
        buf[at] = b"0123456789abcdefghijklmnopqrstuvwxyz"[(v % 36) as usize];
        v /= 36;
        if v == 0 {
            break;
        }
    }
    out.extend_from_slice(&buf[at..]);
}

/// Decides between the dense and the general pipeline before any heavy
/// work: None = general.
pub fn decide(ev: &Input, ctx: &Ctx) -> Option<Dense> {
    let mode = std::env::var("SS_DENSE").unwrap_or_default();
    if mode != "1" && mode != "force" {
        return None;
    }
    let force = mode == "force";
    let est = ev.bytes / 70 + 1;
    if !force && ev.bytes < (8 << 20) {
        return None;
    }
    let bound = (2 * est as u64).clamp(1 << 16, 1 << 32);
    let pmax = ((est as u64) / 8 + 4096).min(1 << 27);
    // samples: the first lines of up to 16 tasks
    let (mut es, mut us, mut lines, mut bad) = (Vec::new(), Vec::new(), 0usize, 0usize);
    for task in ev.tasks.iter().take(16) {
        let b = ev.maps[task.file as usize].bytes();
        let end = scan::next_line_start(b, (task.start + 8192).min(task.end)).min(task.end);
        scan::for_each_record_x::<7, _, false>(b, task.start, end, |f| {
            lines += 1;
            match decode36(f[0], b'e') {
                Some(v) if es.len() < 64 => es.push(v),
                Some(_) => {}
                None => bad += 1,
            }
            if let Some(v) = decode36(f[1], b'u') {
                if us.len() < 64 {
                    us.push(v);
                }
            }
        });
    }
    let keys_env = std::env::var("SS_DENSE_KEYS").ok().and_then(|v| {
        let mut it = v.split(',').map(|x| u64::from_str_radix(x.trim(), 16).ok());
        Some((it.next()??, it.next()??))
    });
    let (ke, ku) = match keys_env {
        Some(k) => k,
        None => {
            let ke = recover(&es, bound, 1, ctx.threads);
            let ku = recover(&us, 512 * pmax, 512, ctx.threads);
            match (ke, ku) {
                (Some(a), Some(b)) => (a, b),
                _ if force => (ke.unwrap_or(0), ku.unwrap_or(0)),
                _ => return None,
            }
        }
    };
    if !force && (lines < 64 || bad * 20 > lines) {
        return None;
    }
    Some(Dense { ke, ku, bound, pmax })
}

/// Exceptional user IDs: exact key -> id (sharded).
struct Xmap {
    shards: Vec<Mutex<HashMap<u64, u32>>>,
    next: AtomicU32,
    keys: Mutex<Vec<(u32, u64)>>,
}

impl Xmap {
    fn new() -> Self {
        Xmap { shards: (0..64).map(|_| Mutex::new(HashMap::new())).collect(), next: AtomicU32::new(0), keys: Mutex::new(Vec::new()) }
    }
    fn id(&self, k: u64) -> u32 {
        let mut m = self.shards[(keys::mix(k) >> 58) as usize].lock().unwrap();
        if let Some(&id) = m.get(&k) {
            return id;
        }
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        m.insert(k, id);
        self.keys.lock().unwrap().push((id, k));
        id
    }
    /// id -> key, after all interning.
    fn key_table(&self) -> Vec<u64> {
        let n = self.next.load(Ordering::Relaxed) as usize;
        let mut t = vec![0u64; n];
        for &(id, k) in self.keys.lock().unwrap().iter() {
            t[id as usize] = k;
        }
        t
    }
}

/// The slot of a user ID (non-empty), exact for every string.
#[inline(always)]
fn user_slot(s: &[u8], d: &Dense, xm: &Xmap) -> u32 {
    if let Some(v) = decode36(s, b'u') {
        let u = unscramble(v, d.ku);
        if u < (1 << 39) && (u >> 9) < d.pmax && (u & 511) < 8 {
            return ((u >> 9) * 8 + (u & 511)) as u32;
        }
    }
    user_slot_x(s, d, xm)
}

#[cold]
#[inline(never)]
fn user_slot_x(s: &[u8], d: &Dense, xm: &Xmap) -> u32 {
    d.xbase() + xm.id(keys::key_of(s))
}

/// keys::mix of the packed key of a person slot's ID string (the name key
/// the general pipeline uses).
fn pmix(p: u32, d: &Dense, xkeys: &[u64]) -> u64 {
    if p < d.xbase() {
        let u = (p as u64 / 8) * 512 + (p as u64 % 8);
        let mut s = Vec::with_capacity(10);
        s.push(b'u');
        base36(&mut s, scramble(u, d.ku));
        keys::mix(keys::key_of(&s))
    } else {
        keys::mix(xkeys[(p - d.xbase()) as usize])
    }
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct DE {
    n: u32,
    ingest: u32,
    /// ts offset | VALID (0 for an invalid copy)
    ts: u32,
    /// user slot | PFLAG for a purchase (0 for an invalid copy)
    us: u32,
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct DP {
    n: u32,
    ingest: u32,
    pay: u32,
    _p: u32,
}

/// An exceptional event copy.
#[derive(Clone, Copy)]
struct XE {
    key: u64,
    ingest: u32,
    valid: bool,
    us: u32,
    ts: u32,
    pay: u32,
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct DC {
    us: u32,
    camp: u32,
    ts: u32,
    _p: u32,
    ck: u64,
}

/// A kept valid event in event-number order.
#[derive(Clone, Copy)]
struct KE {
    p: u32,
    ts: u32,
    pay: u32,
}

struct EvOut {
    de: Parts<DE>,
    dp: Parts<DP>,
    xe: Vec<XE>,
}

/// The dense pipeline from links to user_daily part files; returns the
/// threads' campaign accumulators. `d` from `decide`.
pub fn run(links_in: &Input, ev: &Input, cl: &Input, ctx: &Ctx, d: Dense, ud: &Path, marks: &mut super::Marks) -> Vec<Out2> {
    let threads = ctx.threads;
    let xm = Xmap::new();
    // 1. links (the general code), then slot -> root slot
    let lb = super::LocBits::new(threads, ctx.ps_bits).unwrap();
    let empty = Input { maps: Vec::new(), tasks: Vec::new(), bytes: 0 };
    let (lt, _) = super::parse_b(links_in, &empty, &empty, ctx, lb, None);
    marks.mark("d.links");
    let xbase = d.xbase() as usize;
    let proot: &'static [AtomicU32] = arena(xbase);
    {
        let next = AtomicUsize::new(0);
        const CH: usize = 1 << 20;
        super::par(threads, |_| loop {
            let c = next.fetch_add(1, Ordering::Relaxed);
            if c * CH >= xbase {
                break;
            }
            for i in c * CH..((c + 1) * CH).min(xbase) {
                proot[i].store(i as u32, Ordering::Relaxed);
            }
        });
    }
    let xroot: Mutex<Vec<(u32, u32)>> = Mutex::new(Vec::new());
    link_roots(&lt, &d, &xm, proot, &xroot, threads);
    marks.mark("d.slots");
    // 2. events
    let nb = (d.bound >> BSHIFT) as usize + 1;
    let est = ev.bytes / 70 + 1;
    let cap = est / (threads * nb) * 5 / 4 + 64;
    let capp = est / 16 / (threads * nb) + 32;
    let next = AtomicUsize::new(0);
    let evo: Vec<EvOut> = super::par(threads, |_| {
        let mut o = EvOut { de: Parts::with_wc(nb, cap, true), dp: Parts::with_wc(nb, capp, true), xe: Vec::new() };
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            let Some(&task) = ev.tasks.get(i) else { break };
            let b = ev.maps[task.file as usize].bytes();
            let safe_end = b.as_ptr() as usize + b.len();
            ev_task_any(b, task, ctx, &d, &xm, &mut o, safe_end);
        }
        o.de.finish();
        o.dp.finish();
        o
    });
    marks.mark("d.events");
    // 3. clicks
    let next = AtomicUsize::new(0);
    let clo: Vec<Vec<DC>> = super::par(threads, |_| {
        let mut v = Vec::new();
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            let Some(&task) = cl.tasks.get(i) else { break };
            let b = cl.maps[task.file as usize].bytes();
            let safe_end = b.as_ptr() as usize + b.len();
            cl_task_any(b, task, ctx, &d, &xm, &mut v, safe_end);
        }
        v
    });
    marks.mark("d.clicks");
    // 4. exceptional slots' roots, the person table
    let xkeys = xm.key_table();
    let nx = xkeys.len();
    let np = xbase + nx;
    let mut px: Vec<u32> = (0..nx as u32).map(|i| d.xbase() + i).collect();
    for &(x, r) in xroot.lock().unwrap().iter() {
        px[x as usize] = r;
    }
    let person = |us: u32| -> u32 {
        let us = us as usize;
        if us < xbase {
            proot[us].load(Ordering::Relaxed)
        } else {
            px[us - xbase]
        }
    };
    // 5. placement: per bucket, smallest ingest wins; kept valid events in
    // number order with their person
    let nextb = AtomicUsize::new(0);
    let kept: Vec<Mutex<Vec<KE>>> = (0..nb).map(|_| Mutex::new(Vec::new())).collect();
    super::par(threads, |_| {
        let mut slot = vec![0u32; BSIZE];
        let mut g: Vec<DE> = Vec::new();
        let mut gp: Vec<DP> = Vec::new();
        loop {
            let bi = nextb.fetch_add(1, Ordering::Relaxed);
            if bi >= nb {
                break;
            }
            g.truncate(0);
            gp.truncate(0);
            for o in &evo {
                for sl in o.de.get(bi) {
                    g.extend_from_slice(sl);
                }
                for sl in o.dp.get(bi) {
                    gp.extend_from_slice(sl);
                }
            }
            if g.is_empty() {
                continue;
            }
            for (i, r) in g.iter().enumerate() {
                let s = &mut slot[r.n as usize & (BSIZE - 1)];
                // ties: equal-ingest copies are identical, keep the first
                if *s == 0 || r.ingest < g[*s as usize - 1].ingest {
                    *s = i as u32 + 1;
                }
            }
            gp.sort_unstable_by_key(|r| ((r.n as u64) << 32) | r.ingest as u64);
            let mut out = Vec::with_capacity(g.len());
            for s in slot.iter_mut() {
                if *s == 0 {
                    continue;
                }
                let r = g[*s as usize - 1];
                *s = 0;
                if r.ts & VALID == 0 {
                    continue;
                }
                let pay = if r.us & PFLAG != 0 {
                    let key = ((r.n as u64) << 32) | r.ingest as u64;
                    let j = gp.partition_point(|x| (((x.n as u64) << 32) | x.ingest as u64) < key);
                    assert!(j < gp.len() && gp[j].n == r.n && gp[j].ingest == r.ingest, "dense: purchase record missing");
                    gp[j].pay
                } else {
                    0
                };
                out.push(KE { p: person(r.us & !PFLAG), ts: r.ts & !VALID, pay });
            }
            *kept[bi].lock().unwrap() = out;
        }
    });
    marks.mark("d.place");
    // 6. exceptional events: dedup among themselves; winners are gathered
    let mut xbest: HashMap<u64, XE> = HashMap::new();
    for o in &evo {
        for x in &o.xe {
            match xbest.get(&x.key) {
                Some(b) if b.ingest <= x.ingest => {}
                _ => {
                    xbest.insert(x.key, *x);
                }
            }
        }
    }
    let xwin: Vec<KE> = xbest.values().filter(|x| x.valid).map(|x| KE { p: person(x.us), ts: x.ts, pay: x.pay }).collect();
    // 7. runs per person (a run cut by a bucket border counts twice)
    let runs: &'static [AtomicU32] = arena(np);
    {
        let nextb = AtomicUsize::new(0);
        super::par(threads, |_| loop {
            let bi = nextb.fetch_add(1, Ordering::Relaxed);
            if bi >= nb {
                break;
            }
            let k = kept[bi].lock().unwrap();
            let mut prev = u32::MAX;
            for e in k.iter() {
                if e.p != prev {
                    runs[e.p as usize].fetch_add(1, Ordering::Relaxed);
                    prev = e.p;
                }
            }
        });
    }
    for e in &xwin {
        runs[e.p as usize].fetch_add(2, Ordering::Relaxed);
    }
    // 8. clicks by person (counting sort)
    let cstart: &'static [AtomicU32] = arena(np + 1);
    let total_cl: usize = clo.iter().map(|v| v.len()).sum();
    {
        super::par(threads, |t| {
            if let Some(v) = clo.get(t) {
                for c in v {
                    cstart[person(c.us) as usize + 1].fetch_add(1, Ordering::Relaxed);
                }
            }
        });
        // threads beyond clo.len() (none: clo has one vector per thread)
        let mut acc = 0u32;
        for i in 0..=np {
            acc += cstart[i].load(Ordering::Relaxed);
            cstart[i].store(acc, Ordering::Relaxed);
        }
    }
    let cpos: &'static [AtomicU32] = arena(np);
    for i in 0..np {
        cpos[i].store(cstart[i].load(Ordering::Relaxed), Ordering::Relaxed);
    }
    let clicks: &'static mut [Cl] = arena_mut(total_cl);
    let cptr = clicks.as_mut_ptr() as usize;
    super::par(threads, |t| {
        if let Some(v) = clo.get(t) {
            for c in v {
                let p = person(c.us) as usize;
                let k = cpos[p].fetch_add(1, Ordering::Relaxed) as usize;
                // each position is written once (distinct fetch_add results)
                unsafe { (cptr as *mut Cl).add(k).write(Cl { person: 0, ck: c.ck, ts: c.ts, camp: c.camp }) };
            }
        }
    });
    let clicks: &[Cl] = clicks;
    marks.mark("d.group");
    // 9. persons: single-run persons from the buckets, then the gathered ones
    let est_rows = est / threads / 4 + 4096;
    let nextb = AtomicUsize::new(0);
    let gather: Mutex<Vec<KE>> = Mutex::new(xwin);
    let groups: std::sync::OnceLock<Vec<(usize, usize)>> = std::sync::OnceLock::new();
    let sorted: std::sync::OnceLock<Vec<KE>> = std::sync::OnceLock::new();
    let bar = std::sync::Barrier::new(threads);
    let nextg = AtomicUsize::new(0);
    let outs = super::par(threads, |t| {
        let mut st = PersonSess::new(ctx, est_rows);
        let mut es: Vec<SRec> = Vec::new();
        let mut cs: Vec<Cl> = Vec::new();
        let mut g: Vec<KE> = Vec::new();
        let one = |p: u32, evs: &mut dyn Iterator<Item = KE>, st: &mut PersonSess, es: &mut Vec<SRec>, cs: &mut Vec<Cl>| {
            let pm = pmix(p, &d, &xkeys);
            es.clear();
            es.extend(evs.map(|e| SRec { person: pm, ts: e.ts, pay: e.pay }));
            cs.clear();
            let (a, b) = (cstart[p as usize].load(Ordering::Relaxed) as usize, cstart[p as usize + 1].load(Ordering::Relaxed) as usize);
            cs.extend(clicks[a..b].iter().map(|c| Cl { person: pm, ..*c }));
            st.person(pm, es, cs, ctx);
        };
        loop {
            let bi = nextb.fetch_add(1, Ordering::Relaxed);
            if bi >= nb {
                break;
            }
            let k = kept[bi].lock().unwrap();
            let mut i = 0;
            while i < k.len() {
                let p = k[i].p;
                let mut j = i + 1;
                while j < k.len() && k[j].p == p {
                    j += 1;
                }
                if runs[p as usize].load(Ordering::Relaxed) == 1 {
                    one(p, &mut k[i..j].iter().copied(), &mut st, &mut es, &mut cs);
                } else {
                    g.extend_from_slice(&k[i..j]);
                }
                i = j;
            }
        }
        gather.lock().unwrap().append(&mut g);
        bar.wait();
        if t == 0 {
            let mut v = std::mem::take(&mut *gather.lock().unwrap());
            v.sort_unstable_by_key(|e| e.p);
            let mut gr = Vec::new();
            let mut i = 0;
            while i < v.len() {
                let mut j = i + 1;
                while j < v.len() && v[j].p == v[i].p {
                    j += 1;
                }
                gr.push((i, j));
                i = j;
            }
            let _ = groups.set(gr);
            let _ = sorted.set(v);
        }
        bar.wait();
        {
            let gr = groups.get().unwrap();
            let v = sorted.get().unwrap();
            loop {
                let gi = nextg.fetch_add(1, Ordering::Relaxed);
                let Some(&(a, b)) = gr.get(gi) else { break };
                one(v[a].p, &mut v[a..b].iter().copied(), &mut st, &mut es, &mut cs);
            }
        }
        st.finish(ud, t)
    });
    marks.mark("d.persons");
    outs
}

/// Root slots of every linked ID: proot for dense slots, xroot for
/// exceptional ones.
fn link_roots(lt: &LinkTable, d: &Dense, xm: &Xmap, proot: &[AtomicU32], xroot: &Mutex<Vec<(u32, u32)>>, threads: usize) {
    let nb = lt.val_blocks();
    let next = AtomicUsize::new(0);
    let xbase = d.xbase();
    super::par(threads, |_| {
        let mut local: Vec<(u32, u32)> = Vec::new();
        loop {
            let b = next.fetch_add(1, Ordering::Relaxed);
            if b >= nb {
                break;
            }
            lt.entries(b, |k, pk| {
                let s = user_slot(&keys::decode(k), d, xm);
                let r = user_slot(&keys::decode(pk), d, xm);
                if s < xbase {
                    proot[s as usize].store(r, Ordering::Relaxed);
                } else {
                    local.push((s - xbase, r));
                }
            });
        }
        xroot.lock().unwrap().extend(local);
    });
}

fn arena<T>(n: usize) -> &'static [T] {
    let p = crate::sys::alloc_arena(n.max(1) * std::mem::size_of::<T>()) as *const T;
    unsafe { std::slice::from_raw_parts(p, n) }
}

fn arena_mut<T>(n: usize) -> &'static mut [T] {
    let p = crate::sys::alloc_arena(n.max(1) * std::mem::size_of::<T>()) as *mut T;
    unsafe { std::slice::from_raw_parts_mut(p, n) }
}

// ---------------------------------------------------------------- parse tasks

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[inline(never)]
unsafe fn ev_task_avx2(b: &[u8], task: Task, ctx: &Ctx, d: &Dense, xm: &Xmap, o: &mut EvOut, safe_end: usize) {
    ev_task_x::<true>(b, task, ctx, d, xm, o, safe_end)
}

fn ev_task_any(b: &[u8], task: Task, ctx: &Ctx, d: &Dense, xm: &Xmap, o: &mut EvOut, safe_end: usize) {
    if ctx.avx2 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            return ev_task_avx2(b, task, ctx, d, xm, o, safe_end);
        }
    }
    ev_task_x::<false>(b, task, ctx, d, xm, o, safe_end)
}

/// One event line: the same steps and rules as main::event_line, with the
/// event number / user slot instead of keys.
#[inline(always)]
fn ev_line(f: &[&[u8]; 7], ctx: &Ctx, d: &Dense, xm: &Xmap, o: &mut EvOut, safe_end: usize) {
    if f[0].is_empty() {
        return;
    }
    let (ws, we) = (ctx.ws, ctx.we);
    let Some(ing) = ctx.tsp.parse(f[3]) else { return };
    let ingest = ingest_rel(ing, ctx.ws_s);
    let wide = f[6].as_ptr() as usize + 32 <= safe_end;
    let mut valid = false;
    let (mut ts, mut pay, mut us) = (0u32, 0u32, 0u32);
    'v: {
        if f[1].is_empty() || f[5].is_empty() {
            break 'v;
        }
        let Some(t) = ctx.tsp.parse(f[2]) else { break 'v };
        if t < ws || t >= we {
            break 'v;
        }
        let Some(py) = event_pay(f[4], || f[6], f[5], ctx, wide) else { break 'v };
        if ing - t > LATE {
            break 'v;
        }
        valid = true;
        ts = (t - ws) as u32;
        pay = py;
        us = user_slot(f[1], d, xm);
    }
    let n = match decode36(f[0], b'e') {
        Some(v) => unscramble(v, d.ke),
        None => u64::MAX,
    };
    if n >= d.bound {
        o.xe.push(XE { key: keys::key_of(f[0]), ingest, valid, us, ts, pay });
        return;
    }
    let b = (n >> BSHIFT) as usize;
    if valid {
        let pf = if pay != 0 { PFLAG } else { 0 };
        unsafe { o.de.push_unchecked(b, DE { n: n as u32, ingest, ts: ts | VALID, us: us | pf }) };
        if pay != 0 {
            unsafe { o.dp.push_unchecked(b, DP { n: n as u32, ingest, pay, _p: 0 }) };
        }
    } else {
        unsafe { o.de.push_unchecked(b, DE { n: n as u32, ingest, ts: 0, us: 0 }) };
    }
}

#[inline(always)]
fn ev_task_x<const AVX2: bool>(b: &[u8], task: Task, ctx: &Ctx, d: &Dense, xm: &Xmap, o: &mut EvOut, safe_end: usize) {
    let op: *mut EvOut = o;
    // the two callbacks are never active at the same time
    fastev::for_each_line2::<AVX2, _, _>(
        b,
        task.start,
        task.end,
        safe_end,
        |f| ev_line(f, ctx, d, xm, unsafe { &mut *op }, safe_end),
        |from, to| ev_range(b, from, to, ctx, d, xm, unsafe { &mut *op }, safe_end),
    );
}

/// Lines the fast splitter does not cut (out of line).
#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn ev_range(b: &[u8], from: usize, to: usize, ctx: &Ctx, d: &Dense, xm: &Xmap, o: &mut EvOut, safe_end: usize) {
    scan::for_each_record_x::<7, _, false>(b, from, to, |f| ev_line(f, ctx, d, xm, o, safe_end));
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[inline(never)]
unsafe fn cl_task_avx2(b: &[u8], task: Task, ctx: &Ctx, d: &Dense, xm: &Xmap, v: &mut Vec<DC>, safe_end: usize) {
    cl_task_x::<true>(b, task, ctx, d, xm, v, safe_end)
}

fn cl_task_any(b: &[u8], task: Task, ctx: &Ctx, d: &Dense, xm: &Xmap, v: &mut Vec<DC>, safe_end: usize) {
    if ctx.avx2 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            return cl_task_avx2(b, task, ctx, d, xm, v, safe_end);
        }
    }
    cl_task_x::<false>(b, task, ctx, d, xm, v, safe_end)
}

/// One click line: the same rules as main::click_task.
#[inline(always)]
fn cl_task_x<const AVX2: bool>(b: &[u8], task: Task, ctx: &Ctx, d: &Dense, xm: &Xmap, v: &mut Vec<DC>, safe_end: usize) {
    let we = ctx.we;
    let cws = ctx.ws - super::WEEK;
    let mut line = |f: &[&[u8]; 4]| {
        if f[0].is_empty() || f[1].is_empty() {
            return;
        }
        let wide = f[3].as_ptr() as usize + 32 <= safe_end;
        let Some(t) = ctx.tsp.parse(f[3]) else { return };
        if t < cws || t >= we {
            return;
        }
        let camp = if wide { unsafe { ctx.campaigns.get_wide(f[2]) } } else { ctx.campaigns.get(f[2]) };
        if camp == NONE {
            return;
        }
        let us = user_slot(f[1], d, xm);
        let ck = if wide { unsafe { keys::key_of_wide(f[0]) } } else { keys::key_of(f[0]) };
        v.push(DC { us, camp: camp as u32, ts: (t - cws) as u32, _p: 0, ck });
    };
    fastev::for_each_click::<AVX2, _>(b, task.start, task.end, safe_end, &mut line);
}
