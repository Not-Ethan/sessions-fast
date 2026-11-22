//! Dense pipeline (default; SS_DENSE=0 = general pipeline, SS_DENSE=force
//! skips the size and density checks, for tests). Uses the generator's ID structure (dense.rs) for
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
use crate::dense::{decode36, decode36_w, recover, unscramble, M1, M2, MASK};
use crate::links::LinkTable;
use crate::lookup::NONE;
use crate::parts::{self, Parts};
use crate::sess::{Cl, Out2, PersonSess};
use crate::{fastev, keys, scan};

/// Event-number bucket bits (SS_DBSHIFT, 12..=20, default 16; `Dense::bshift`).
const BSHIFT_DEFAULT: u32 = 16;
const VALID: u32 = 1 << 31;
const PFLAG: u32 = 1 << 31;
/// Flag in `runs`: the person has foreign clicks.
const FOREIGN: u32 = 1 << 31;
/// Single-run persons with at least this many kept events are processed
/// with the gathered ones (first, largest first).
const BIG_RUN: usize = 512;
/// Winners per prefetch block in placement.
const PF_BLOCK: usize = 16;

/// Keys and bounds of a dense run.
#[derive(Clone, Copy)]
pub struct Dense {
    pub ke: u64,
    pub ku: u64,
    /// event numbers below this are dense
    pub bound: u64,
    /// person indexes below this get dense slots
    pub pmax: u64,
    /// sizing estimates from the sample (performance only, never results):
    /// event numbers in use, user slots in use, event lines, purchase share
    pub n_est: u64,
    pub s_est: u64,
    pub lines_est: usize,
    pub pfrac: f64,
    /// event-number bucket bits
    pub bshift: u32,
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

/// Diagnostics for SS_VERBOSE: "key=value" pairs.
static STATS: Mutex<String> = Mutex::new(String::new());

fn stat(k: &str, v: impl std::fmt::Display) {
    let mut s = STATS.lock().unwrap();
    s.push_str(&format!("{}={} ", k, v));
}

pub fn stats() -> String {
    STATS.lock().unwrap().clone()
}

/// Decides between the dense and the general pipeline before any heavy
/// work: None = general.
pub fn decide(ev: &Input, ctx: &Ctx) -> Option<Dense> {
    // SS_DENSE: "1" = dense when the checks below pass, "force" = dense on
    // any input (tests), anything else (default) = general. Off by default since scorer version 5: v5 IDs pass through a keyed
    // bijection that cannot be recovered, so the dense path never engages
    // there (SS_DENSE=1 / force to use it, e.g. on version 4 data).
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
    // samples: the first lines of up to 16 tasks spread over the input
    let (mut es, mut us, mut lines, mut bad) = (Vec::new(), Vec::new(), 0usize, 0usize);
    let (mut eall, mut uall, mut sbytes, mut purch) = (Vec::new(), Vec::new(), 0usize, 0usize);
    let nt = ev.tasks.len().min(16);
    for k in 0..nt {
        let task = ev.tasks[k * ev.tasks.len() / nt];
        let b = ev.maps[task.file as usize].bytes();
        let end = scan::next_line_start(b, (task.start + 8192).min(task.end)).min(task.end);
        sbytes += end - task.start;
        scan::for_each_record_x::<7, _, false>(b, task.start, end, |f| {
            lines += 1;
            purch += (f[4] == b"purchase") as usize;
            match decode36(f[0], b'e') {
                Some(v) => {
                    if es.len() < 64 {
                        es.push(v);
                    }
                    eall.push(v);
                }
                None => bad += 1,
            }
            if let Some(v) = decode36(f[1], b'u') {
                if us.len() < 64 {
                    us.push(v);
                }
                uall.push(v);
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
    // sizing: event numbers in use ~ 90th percentile / 0.9 (robust to a
    // few outliers), user slots in use ~ largest sampled slot
    let mut ns: Vec<u64> = eall.iter().map(|&v| unscramble(v, ke)).filter(|&n| n < bound).collect();
    ns.sort_unstable();
    let n_est = if ns.len() >= 32 { (ns[ns.len() * 9 / 10] * 10 / 9 + (1 << 16)).min(bound) } else { bound };
    let xb = pmax * 8;
    let smax = uall
        .iter()
        .map(|&v| unscramble(v, ku))
        .filter(|&u| u < (1 << 39) && (u >> 9) < pmax && (u & 511) < 8)
        .map(|u| (u >> 9) * 8 + (u & 511))
        .max();
    let s_est = match smax {
        Some(m) if uall.len() >= 32 => (m + m / 20 + (1 << 16)).min(xb),
        _ => xb,
    };
    let lines_est = if sbytes > 0 && lines > 0 { (ev.bytes as f64 * lines as f64 / sbytes as f64) as usize + 1 } else { est };
    let pfrac = if lines > 0 { purch as f64 / lines as f64 } else { 1.0 / 16.0 };
    stat("d_bound", bound);
    stat("d_pmax", pmax);
    stat("d_sample_lines", lines);
    stat("d_n_est", n_est);
    stat("d_s_est", s_est);
    stat("d_lines_est", lines_est);
    stat("d_pfrac", format!("{:.4}", pfrac));
    let bshift = std::env::var("SS_DBSHIFT").ok().and_then(|v| v.parse::<u32>().ok()).filter(|b| (12..=20).contains(b)).unwrap_or(BSHIFT_DEFAULT);
    Some(Dense { ke, ku, bound, pmax, n_est, s_est, lines_est, pfrac, bshift })
}

/// Exceptional user IDs: exact key -> id (sharded).
pub(crate) struct Xmap {
    shards: Vec<Mutex<HashMap<u64, u32>>>,
    next: AtomicU32,
    keys: Mutex<Vec<(u32, u64)>>,
}

impl Xmap {
    pub(crate) fn new() -> Self {
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
    pub(crate) fn key_table(&self) -> Vec<u64> {
        let n = self.next.load(Ordering::Relaxed) as usize;
        let mut t = vec![0u64; n];
        for &(id, k) in self.keys.lock().unwrap().iter() {
            t[id as usize] = k;
        }
        t
    }
}

/// The slot of a user ID (non-empty), exact for every string. `wide`: 16
/// bytes from the start of `s` are readable.
#[inline(always)]
fn user_slot_w(s: &[u8], d: &Dense, xm: &Xmap, wide: bool) -> u32 {
    if let Some(v) = decode36_w(s, b'u', wide) {
        let u = unscramble(v, d.ku);
        if u < (1 << 39) && (u >> 9) < d.pmax && (u & 511) < 8 {
            return ((u >> 9) * 8 + (u & 511)) as u32;
        }
    }
    user_slot_x(s, d, xm)
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
pub(crate) struct DE {
    pub(crate) n: u32,
    pub(crate) ingest: u32,
    /// ts offset | VALID (0 for an invalid copy)
    pub(crate) ts: u32,
    /// user slot | PFLAG for a purchase (0 for an invalid copy)
    pub(crate) us: u32,
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub(crate) struct DP {
    pub(crate) n: u32,
    pub(crate) ingest: u32,
    pub(crate) pay: u32,
    pub(crate) _p: u32,
}

/// An exceptional event copy.
#[derive(Clone, Copy)]
pub(crate) struct XE {
    pub(crate) key: u64,
    pub(crate) ingest: u32,
    pub(crate) valid: bool,
    pub(crate) us: u32,
    pub(crate) ts: u32,
    pub(crate) pay: u32,
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub(crate) struct DC {
    pub(crate) us: u32,
    pub(crate) camp: u32,
    pub(crate) ts: u32,
    pub(crate) _p: u32,
    pub(crate) ck: u64,
    pub(crate) _q: u64,
}

/// An exceptional event's dedup winner with its person.
#[derive(Clone, Copy)]
struct KE {
    p: u32,
    ts: u32,
    pay: u32,
}

pub(crate) struct EvOut {
    pub(crate) de: Parts<DE>,
    pub(crate) dp: Parts<DP>,
    pub(crate) xe: Vec<XE>,
}

/// The dense pipeline from links to user_daily part files; returns the
/// threads' campaign accumulators. `d` from `decide`.
pub fn run(links_in: &Input, ev: &Input, cl: &Input, ctx: &Ctx, d: Dense, ud: &Path, marks: &mut super::Marks) -> Vec<Out2> {
    let threads = ctx.threads;
    let release = std::env::var("SS_NORELEASE").is_err();
    let xm = Xmap::new();
    // 1. links (the general code), then slot -> root slot (stored + 1; 0 =
    // the slot is its own person)
    let xbase = d.xbase() as usize;
    let proot: &'static [AtomicU32] = arena(xbase);
    let xroot: Mutex<Vec<(u32, u32)>> = Mutex::new(Vec::new());
    if std::env::var("SS_DLINKS").map_or(true, |v| v != "0") {
        let np = dense_links(links_in, ctx, &d, &xm, proot, &xroot, threads, marks);
        stat("d_link_pairs", np);
        if std::env::var("SS_LINKCHECK").is_ok() {
            link_check(links_in, ctx, &d, &xm, proot, &xroot, threads);
        }
    } else {
        let lb = super::LocBits::new(threads, ctx.ps_bits).unwrap();
        let empty = Input { maps: Vec::new(), tasks: Vec::new(), bytes: 0 };
        let (lt, _) = super::parse_b(links_in, &empty, &empty, ctx, lb, None);
        marks.mark("d.links");
        link_roots(&lt, &d, &xm, proot, &xroot, threads);
    }
    marks.mark("d.slots");
    // 2. events
    // bucket capacities (overflow is exact, only slower): copies spread
    // evenly over the buckets of the event numbers in use
    let bsize = 1usize << d.bshift;
    let nb = (d.bound >> d.bshift) as usize + 1;
    let est = d.lines_est;
    let used = ((d.n_est >> d.bshift) as usize + 1).min(nb);
    let cap = (est as f64 * 1.4 / (threads * used) as f64) as usize + 64;
    let capp = (cap as f64 * (d.pfrac * 1.5 + 0.01)) as usize + 32;
    let next = AtomicUsize::new(0);
    let evo: Vec<EvOut> = super::par(threads, |_| {
        let mut o = EvOut { de: Parts::with_wc(nb, cap, true), dp: Parts::with_wc(nb, capp, true), xe: Vec::new() };
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            let Some(&task) = ev.tasks.get(i) else { break };
            let b = ev.maps[task.file as usize].bytes();
            let safe_end = b.as_ptr() as usize + b.len();
            ev_task_any(b, task, ctx, &d, &xm, &mut o, safe_end);
            if release {
                // parsed: drop the chunk's page mappings now, in parallel,
                // instead of in the single-threaded exit (as main.rs)
                crate::sys::release_inner(b.as_ptr() as usize + task.start, b.as_ptr() as usize + task.end);
            }
        }
        o.de.finish();
        o.dp.finish();
        o
    });
    marks.mark("d.events");
    stat("d_over_ev", parts::WC_OVER_LINES.load(Ordering::Relaxed));
    // 3. clicks, bucketed by user slot >> 16 (exceptional slots: last bucket)
    let nbc = (xbase >> 16) + 1;
    let estc = cl.bytes / 44 + 1;
    let usedc = ((d.s_est >> 16) as usize + 1).min(nbc);
    let capc = (estc as f64 * 2.5 / (threads * usedc) as f64) as usize + 64;
    let next = AtomicUsize::new(0);
    let clo: Vec<Parts<DC>> = super::par(threads, |_| {
        let mut v: Parts<DC> = Parts::with_wc(nbc, capc, true);
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            let Some(&task) = cl.tasks.get(i) else { break };
            let b = cl.maps[task.file as usize].bytes();
            let safe_end = b.as_ptr() as usize + b.len();
            cl_task_any(b, task, ctx, &d, &xm, &mut v, nbc, safe_end);
            if release {
                crate::sys::release_inner(b.as_ptr() as usize + task.start, b.as_ptr() as usize + task.end);
            }
        }
        v.finish();
        v
    });
    // every long ID keyed: checked, decode table for names (keys.rs)
    keys::verify_longs(threads);
    marks.mark("d.clicks");
    stat("d_over_cl", parts::WC_OVER_LINES.load(Ordering::Relaxed));
    // 4. exceptional slots' roots, the person table
    let xkeys = xm.key_table();
    stat("d_xslots", xkeys.len());
    stat("d_xevents", evo.iter().map(|o| o.xe.len()).sum::<usize>());
    stat("d_copies", evo.iter().map(|o| (0..nb).map(|b| o.de.len(b)).sum::<usize>()).sum::<usize>());
    stat("d_clicks", clo.iter().map(|o| (0..nbc).map(|b| o.len(b)).sum::<usize>()).sum::<usize>());
    stat("d_nb", nb);
    let nx = xkeys.len();
    let np = xbase + nx;
    let mut px: Vec<u32> = (0..nx as u32).map(|i| d.xbase() + i).collect();
    for &(x, r) in xroot.lock().unwrap().iter() {
        px[x as usize] = r;
    }
    let person = |us: u32| -> u32 {
        let u = us as usize;
        if u < xbase {
            let v = proot[u].load(Ordering::Relaxed);
            if v == 0 {
                us
            } else {
                v - 1
            }
        } else {
            px[u - xbase]
        }
    };
    // 5. placement: per bucket, smallest ingest wins; kept valid events in
    // number order with their person, into one arena (bucket regions sized
    // by the bucket's copies)
    let mut boff = vec![0usize; nb + 1];
    for bi in 0..nb {
        boff[bi + 1] = boff[bi] + evo.iter().map(|o| o.de.len(bi)).sum::<usize>();
    }
    let kept_all: &'static mut [SRec] = arena_mut(boff[nb]);
    let kptr = kept_all.as_mut_ptr() as usize;
    let klen: Vec<AtomicUsize> = (0..nb).map(|_| AtomicUsize::new(0)).collect();
    let nextb = AtomicUsize::new(0);
    super::par(threads, |_| {
        // slot i: the winning copy of event number (bucket base + i); its
        // `n` field is reused for the purchase amount (0 until the purchase
        // record is seen); occ: occupied slots
        let mut slot = vec![DE::default(); bsize];
        let mut occ = vec![0u64; bsize / 64];
        let mut blk: Vec<(u32, u32, u32)> = Vec::with_capacity(PF_BLOCK);
        loop {
            let bi = nextb.fetch_add(1, Ordering::Relaxed);
            if bi >= nb {
                break;
            }
            if boff[bi + 1] == boff[bi] {
                continue;
            }
            for o in &evo {
                for sl in o.de.get(bi) {
                    for r in sl.iter() {
                        let i = r.n as usize & (bsize - 1);
                        let w = &mut occ[i >> 6];
                        let m = 1u64 << (i & 63);
                        // ties: equal-ingest copies are identical (rules),
                        // keep the first
                        if *w & m == 0 {
                            *w |= m;
                            slot[i] = DE { n: 0, ..*r };
                        } else if r.ingest < slot[i].ingest {
                            slot[i] = DE { n: 0, ..*r };
                        }
                    }
                }
            }
            // purchase amounts of winning purchase copies (a purchase record
            // with the winner's ingest is a copy identical to the winner)
            for o in &evo {
                for sl in o.dp.get(bi) {
                    for r in sl.iter() {
                        let i = r.n as usize & (bsize - 1);
                        assert!(occ[i >> 6] & (1u64 << (i & 63)) != 0, "dense: purchase record without event");
                        let w = &mut slot[i];
                        if w.ingest == r.ingest && w.us & PFLAG != 0 {
                            w.n = r.pay;
                        }
                    }
                }
            }
            // this bucket's region: boff[bi]..boff[bi + 1], written only here
            let out = unsafe { std::slice::from_raw_parts_mut((kptr as *mut SRec).add(boff[bi]), boff[bi + 1] - boff[bi]) };
            let mut k = 0usize;
            // valid winners in number order, in blocks: prefetch the
            // blocks' root-table entries, then resolve
            let mut flush = |blk: &mut Vec<(u32, u32, u32)>, k: &mut usize| {
                for &(us, ts, pay) in blk.iter() {
                    out[*k] = SRec { person: person(us) as u64, ts, pay };
                    *k += 1;
                }
                blk.clear();
            };
            for wi in 0..bsize / 64 {
                let mut w = occ[wi];
                occ[wi] = 0;
                while w != 0 {
                    let i = wi * 64 + w.trailing_zeros() as usize;
                    w &= w - 1;
                    let r = slot[i];
                    if r.ts & VALID == 0 {
                        continue;
                    }
                    let pay = if r.us & PFLAG != 0 {
                        assert!(r.n != 0, "dense: purchase record missing");
                        r.n
                    } else {
                        0
                    };
                    let us = r.us & !PFLAG;
                    if (us as usize) < xbase {
                        super::prefetch(&proot[us as usize] as *const AtomicU32 as *const u8);
                    }
                    blk.push((us, r.ts & !VALID, pay));
                    if blk.len() == PF_BLOCK {
                        flush(&mut blk, &mut k);
                    }
                }
            }
            flush(&mut blk, &mut k);
            klen[bi].store(k, Ordering::Relaxed);
        }
    });
    let kept_all: &[SRec] = kept_all;
    let kept = |bi: usize| -> &[SRec] { &kept_all[boff[bi]..boff[bi] + klen[bi].load(Ordering::Relaxed)] };
    marks.mark("d.place");
    // 6. exceptional events: dedup among themselves; winners join their
    // persons' gathered events (sorted by person)
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
    let mut xwin: Vec<KE> = xbest.values().filter(|x| x.valid).map(|x| KE { p: person(x.us), ts: x.ts, pay: x.pay }).collect();
    xwin.sort_unstable_by_key(|e| e.p);
    // 7. runs per person (a run cut by a bucket border counts twice; a
    // person with exceptional events counts at least twice)
    let runs: &'static [AtomicU32] = arena(np);
    {
        let nextb = AtomicUsize::new(0);
        super::par(threads, |_| loop {
            let bi = nextb.fetch_add(1, Ordering::Relaxed);
            if bi >= nb {
                break;
            }
            let mut prev = u64::MAX;
            for e in kept(bi) {
                if e.person != prev {
                    runs[e.person as usize].fetch_add(1, Ordering::Relaxed);
                    prev = e.person;
                }
            }
        });
    }
    for e in &xwin {
        runs[e.p as usize].fetch_add(2, Ordering::Relaxed);
    }
    // 8. clicks by person. Click bucket ci holds the clicks whose user slot
    // is in [ci << 16, (ci + 1) << 16) (exceptional slots: the last bucket).
    // Clicks whose person is a dense slot in the same range ("local": almost
    // all, since a person's IDs are one person index) are counting-sorted by
    // person into the bucket's region of `call`, with the person slots
    // copied into `ckey` (read-only from here: searches never read `call`,
    // which the persons phase sorts in place); the rest ("foreign") go to one
    // list sorted by person.
    let mut coff = vec![0usize; nbc + 1];
    for ci in 0..nbc {
        coff[ci + 1] = coff[ci] + clo.iter().map(|o| o.len(ci)).sum::<usize>();
    }
    let call: &'static mut [Cl] = arena_mut(coff[nbc]);
    let caddr = call.as_mut_ptr() as usize;
    let ckey: &'static mut [u32] = arena_mut(coff[nbc]);
    let kaddr = ckey.as_mut_ptr() as usize;
    let clen: Vec<AtomicUsize> = (0..nbc).map(|_| AtomicUsize::new(0)).collect();
    // largest buckets first (whales make a few buckets heavy)
    let mut corder: Vec<usize> = (0..nbc).filter(|&ci| coff[ci + 1] > coff[ci]).collect();
    corder.sort_unstable_by_key(|&ci| std::cmp::Reverse(coff[ci + 1] - coff[ci]));
    stat("d_click_bucket_max", corder.first().map_or(0, |&ci| coff[ci + 1] - coff[ci]));
    let foreign: Mutex<Vec<Cl>> = Mutex::new(Vec::new());
    // a click's person, if local to bucket ci
    let local = |c: &DC, ci: usize| -> (u32, bool) {
        let p = person(c.us);
        (p, (p as usize) < xbase && (p as usize >> 16) == ci)
    };
    // Light buckets: one worker each, two passes (count, then scatter).
    // Heavy buckets (whales): one task per (bucket, source part) for the
    // count and for the scatter, with per-part offsets in between.
    let nparts = clo.len();
    let heavy_min = coff[nbc] / (threads * 4) + 1;
    let nh = corder.iter().take_while(|&&ci| coff[ci + 1] - coff[ci] >= heavy_min).count();
    let (heavy, light) = corder.split_at(nh);
    stat("d_heavy_click_buckets", nh);
    // per (heavy bucket, part): 65536 counts, then start offsets
    let htab: &'static mut [u32] = arena_mut(nh * nparts * 65536);
    let haddr = htab.as_mut_ptr() as usize;
    let tab = |h: usize, t: usize| -> &mut [u32] {
        // (h, t) regions are disjoint; each is used by one task at a time
        unsafe { std::slice::from_raw_parts_mut((haddr as *mut u32).add((h * nparts + t) * 65536), 65536) }
    };
    {
        let next = AtomicUsize::new(0);
        super::par(threads, |_| {
            let mut cnt = vec![0u32; 65537];
            let mut fl: Vec<Cl> = Vec::new();
            loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                if k < nh * nparts {
                    let (h, t) = (k / nparts, k % nparts);
                    let ci = heavy[h];
                    let tb = tab(h, t);
                    for sl in clo[t].get(ci) {
                        for c in sl.iter() {
                            let (p, l) = local(c, ci);
                            if l {
                                tb[(p & 0xFFFF) as usize] += 1;
                            }
                        }
                    }
                    continue;
                }
                let Some(&ci) = light.get(k - nh * nparts) else { break };
                for o in &clo {
                    for sl in o.get(ci) {
                        for c in sl.iter() {
                            let (p, l) = local(c, ci);
                            if l {
                                cnt[(p & 0xFFFF) as usize + 1] += 1;
                            }
                        }
                    }
                }
                for i in 0..65536 {
                    cnt[i + 1] += cnt[i];
                }
                let n = cnt[65536] as usize;
                // this bucket's regions, written only here
                let dst = unsafe { std::slice::from_raw_parts_mut((caddr as *mut Cl).add(coff[ci]), n) };
                let dk = unsafe { std::slice::from_raw_parts_mut((kaddr as *mut u32).add(coff[ci]), n) };
                for o in &clo {
                    for sl in o.get(ci) {
                        for c in sl.iter() {
                            let (p, l) = local(c, ci);
                            let cl = Cl { person: p as u64, ck: c.ck, ts: c.ts, camp: c.camp };
                            if l {
                                let k = &mut cnt[(p & 0xFFFF) as usize];
                                dst[*k as usize] = cl;
                                dk[*k as usize] = p;
                                *k += 1;
                            } else {
                                fl.push(cl);
                            }
                        }
                    }
                }
                clen[ci].store(n, Ordering::Relaxed);
                cnt.fill(0);
            }
            foreign.lock().unwrap().append(&mut fl);
        });
    }
    if nh > 0 {
        // heavy buckets: start offsets in (person, part) order
        let next = AtomicUsize::new(0);
        super::par(threads, |_| loop {
            let h = next.fetch_add(1, Ordering::Relaxed);
            if h >= nh {
                break;
            }
            let mut run = 0u32;
            for key in 0..65536 {
                for t in 0..nparts {
                    let tb = tab(h, t);
                    let c = tb[key];
                    tb[key] = run;
                    run += c;
                }
            }
            clen[heavy[h]].store(run as usize, Ordering::Relaxed);
        });
        let next = AtomicUsize::new(0);
        super::par(threads, |_| {
            let mut fl: Vec<Cl> = Vec::new();
            loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                if k >= nh * nparts {
                    break;
                }
                let (h, t) = (k / nparts, k % nparts);
                let ci = heavy[h];
                let tb = tab(h, t);
                let n = clen[ci].load(Ordering::Relaxed);
                // each (person, part) range is written only by this task
                // (raw writes: other tasks write other indices of the region)
                let dst = (caddr as *mut Cl).wrapping_add(coff[ci]);
                let dk = (kaddr as *mut u32).wrapping_add(coff[ci]);
                for sl in clo[t].get(ci) {
                    for c in sl.iter() {
                        let (p, l) = local(c, ci);
                        let cl = Cl { person: p as u64, ck: c.ck, ts: c.ts, camp: c.camp };
                        if l {
                            let k = &mut tb[(p & 0xFFFF) as usize];
                            assert!((*k as usize) < n);
                            unsafe {
                                dst.add(*k as usize).write(cl);
                                dk.add(*k as usize).write(p);
                            }
                            *k += 1;
                        } else {
                            fl.push(cl);
                        }
                    }
                }
            }
            foreign.lock().unwrap().append(&mut fl);
        });
    }
    let mut foreign = foreign.into_inner().unwrap();
    foreign.sort_unstable_by_key(|c| c.person);
    // persons with foreign clicks: flagged in `runs` (so they are gathered,
    // and only they search the foreign list)
    for c in &foreign {
        runs[c.person as usize].fetch_or(FOREIGN, Ordering::Relaxed);
    }
    stat("d_foreign_clicks", foreign.len());
    stat("d_xwin", xwin.len());
    marks.mark("d.group");
    // 9. run descriptors (person, bucket, from, to) of every person that is
    // not exactly one run, plus a marker for persons with exceptional
    // events; grouped by person, largest groups first
    let multi: Mutex<Vec<(u32, u32, u32, u32)>> = Mutex::new(Vec::new());
    {
        let nextb = AtomicUsize::new(0);
        super::par(threads, |_| {
            let mut md: Vec<(u32, u32, u32, u32)> = Vec::new();
            loop {
                let bi = nextb.fetch_add(1, Ordering::Relaxed);
                if bi >= nb {
                    break;
                }
                let k = kept(bi);
                let mut i = 0;
                while i < k.len() {
                    let p = k[i].person;
                    let mut j = i + 1;
                    while j < k.len() && k[j].person == p {
                        j += 1;
                    }
                    // (big single-run persons are gathered too, so they start
                    // first instead of holding up a bucket at the end)
                    if runs[p as usize].load(Ordering::Relaxed) != 1 || j - i >= BIG_RUN {
                        md.push((p as u32, bi as u32, i as u32, j as u32));
                    }
                    i = j;
                }
            }
            multi.lock().unwrap().append(&mut md);
        });
    }
    let mut mv = multi.into_inner().unwrap();
    // persons that have only exceptional events still need a group
    for e in &xwin {
        mv.push((e.p, u32::MAX, 0, 0));
    }
    mv.sort_unstable();
    mv.dedup_by(|a, b| a.1 == u32::MAX && b.1 == u32::MAX && a.0 == b.0);
    let mut gr: Vec<(usize, usize, usize)> = Vec::new();
    {
        let mut i = 0;
        while i < mv.len() {
            let mut j = i + 1;
            let mut size = (mv[i].3 - mv[i].2) as usize;
            while j < mv.len() && mv[j].0 == mv[i].0 {
                size += (mv[j].3 - mv[j].2) as usize;
                j += 1;
            }
            gr.push((size, i, j));
            i = j;
        }
    }
    gr.sort_unstable_by(|a, b| b.cmp(a));
    stat("d_group_max", gr.first().map_or(0, |g| g.0));
    stat("d_groups", gr.len());
    stat("d_group_runs", mv.len());
    stat("d_kept", boff.len().min(1) * (0..nb).map(|b| klen[b].load(Ordering::Relaxed)).sum::<usize>());
    marks.mark("d.runs");
    // 10. persons: the gathered ones first (largest first), then every
    // single-run person from the buckets
    let est_rows = est / threads / 4 + 4096;
    let nextb = AtomicUsize::new(0);
    let nextg = AtomicUsize::new(0);
    // Arenas that exist now (link table, parse outputs, kept events, clicks,
    // root and run tables) are dead after the persons loop: released in
    // parallel pieces then (as sess::sessions2) instead of in the exit.
    // Arenas allocated from here on (output buffers) are not in the list.
    let keepmem = std::env::var("SS_KEEPMEM").is_ok();
    let mut pieces: Vec<(usize, usize)> = Vec::new();
    if !keepmem {
        for (a, l) in crate::sys::arenas() {
            let mut o = 0;
            while o < l {
                let n = (l - o).min(32 << 20);
                pieces.push((a + o, n));
                o += n;
            }
        }
    }
    let bar = std::sync::Barrier::new(threads);
    let next_piece = AtomicUsize::new(0);
    let old_pmix = std::env::var("SS_PMIX").is_ok();
    let outs = super::par(threads, |t| {
        let mut st = PersonSess::new(ctx, est_rows);
        let mut es: Vec<SRec> = Vec::new();
        let mut scratch: Vec<Cl> = Vec::new();
        // search cursor in the local clicks: (bucket, end of the last
        // person's range, last person)
        let mut cur: (usize, usize, u32) = (usize::MAX, 0, 0);
        // the person's clicks (its own range of `call`: every person is
        // processed once, so ranges are never shared; plus foreign ones),
        // then the routine
        let mut finish_person = |p: u32, es: &[SRec], st: &mut PersonSess, has_foreign: bool| {
            let (mut a, mut b) = (0usize, 0usize);
            if (p as usize) < xbase {
                let ci = p as usize >> 16;
                let base = coff[ci];
                let len = clen[ci].load(Ordering::Relaxed);
                let keys = unsafe { std::slice::from_raw_parts((kaddr as *const u32).add(base), len) };
                let mut i;
                if cur.0 == ci && p > cur.2 {
                    // every key before cur.1 is <= the last person < p
                    i = cur.1;
                    let lim = (i + 16).min(len);
                    while i < lim && keys[i] < p {
                        i += 1;
                    }
                    if i == lim && i < len && keys[i] < p {
                        i += keys[i..].partition_point(|&k| k < p);
                    }
                } else {
                    i = keys.partition_point(|&k| k < p);
                }
                let mut j = i;
                while j < len && keys[j] == p {
                    j += 1;
                }
                cur = (ci, j, p);
                a = base + i;
                b = base + j;
            }
            // the person's name: a dense slot's ID string straight into the
            // row buffer (name 0), else the interned key (SS_PMIX=1: every
            // name through pmix, as before)
            let pm = if (p as usize) < xbase && !old_pmix {
                let mut nb = [0u8; 9];
                let n = slot_name(p, &d, &mut nb);
                st.set_name(&nb[..n]);
                0
            } else {
                pmix(p, &d, &xkeys)
            };
            let pp = p as u64;
            let (fa, fb) = if !has_foreign { (0, 0) } else { (foreign.partition_point(|c| c.person < pp), foreign.partition_point(|c| c.person <= pp)) };
            if fa == fb {
                let cs = unsafe { std::slice::from_raw_parts_mut((caddr as *mut Cl).add(a), b - a) };
                st.person(pp, pm, es, cs, ctx);
            } else {
                scratch.clear();
                scratch.extend_from_slice(unsafe { std::slice::from_raw_parts((caddr as *const Cl).add(a), b - a) });
                scratch.extend_from_slice(&foreign[fa..fb]);
                st.person(pp, pm, es, &mut scratch, ctx);
            }
        };
        loop {
            let gi = nextg.fetch_add(1, Ordering::Relaxed);
            let Some(&(_, a, b)) = gr.get(gi) else { break };
            let p = mv[a].0;
            es.clear();
            for &(_, bi, i, j) in &mv[a..b] {
                if bi != u32::MAX {
                    es.extend_from_slice(&kept(bi as usize)[i as usize..j as usize]);
                }
            }
            let (xa, xb) = (xwin.partition_point(|e| e.p < p), xwin.partition_point(|e| e.p <= p));
            es.extend(xwin[xa..xb].iter().map(|e| SRec { person: p as u64, ts: e.ts, pay: e.pay }));
            finish_person(p, &es, &mut st, runs[p as usize].load(Ordering::Relaxed) & FOREIGN != 0);
        }
        loop {
            let bi = nextb.fetch_add(1, Ordering::Relaxed);
            if bi >= nb {
                break;
            }
            let k = kept(bi);
            let mut i = 0;
            while i < k.len() {
                let p = k[i].person;
                let mut j = i + 1;
                while j < k.len() && k[j].person == p {
                    j += 1;
                }
                if runs[p as usize].load(Ordering::Relaxed) == 1 && j - i < BIG_RUN {
                    // (runs == 1: no foreign clicks)
                    finish_person(p as u32, &k[i..j], &mut st, false);
                }
                i = j;
            }
        }
        if !keepmem {
            bar.wait();
            loop {
                let i = next_piece.fetch_add(1, Ordering::Relaxed);
                let Some(&(a, l)) = pieces.get(i) else { break };
                crate::sys::release(a, l);
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
                let mut b1 = [0u8; 16];
                let mut b2 = [0u8; 16];
                let s = key_slot(k, &mut b1, d, xm);
                let r = key_slot(pk, &mut b2, d, xm);
                if s < xbase {
                    // + 1: 0 means "own person"
                    proot[s as usize].store(r + 1, Ordering::Relaxed);
                } else {
                    local.push((s - xbase, r));
                }
            });
        }
        xroot.lock().unwrap().extend(local);
    });
}

/// The link lines of one task as slot pairs (self-links skipped): the same
/// scanner and drop rules as main::link_task (malformed lines and lines with
/// an empty side dropped), each side's slot as `user_slot_w`.
#[inline(always)]
fn dlink_task_x<const AVX2: bool>(b: &[u8], task: Task, d: &Dense, xm: &Xmap, v: &mut Vec<(u32, u32)>) {
    let safe_end = b.as_ptr() as usize + b.len();
    v.reserve((task.end - task.start) / 16 + 16);
    scan::for_each_record_x::<2, _, AVX2>(b, task.start, task.end, #[inline(always)] |f| {
        if f[0].is_empty() || f[1].is_empty() {
            return;
        }
        // f[0] starts before f[1]: 16 readable bytes from f[1] cover f[0] too
        let wide = f[1].as_ptr() as usize + 16 <= safe_end;
        let sa = user_slot_w(f[0], d, xm, wide);
        let sb = user_slot_w(f[1], d, xm, wide);
        if sa != sb {
            v.push((sa, sb));
        }
    });
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
unsafe fn dlink_task_avx2(b: &[u8], task: Task, d: &Dense, xm: &Xmap, v: &mut Vec<(u32, u32)>) {
    dlink_task_x::<true>(b, task, d, xm, v)
}

/// The ID string of a dense slot: `u` + base 36 of its scrambled number
/// (at most 8 digits: v < 2^40 < 36^8), into `buf`; returns its length.
#[inline(always)]
fn slot_name(s: u32, d: &Dense, buf: &mut [u8; 9]) -> usize {
    let u = (s as u64 / 8) * 512 + (s as u64 % 8);
    let mut v = scramble(u, d.ku);
    let mut dig = [0u8; 8];
    let mut n = 0;
    loop {
        dig[n] = b"0123456789abcdefghijklmnopqrstuvwxyz"[(v % 36) as usize];
        v /= 36;
        n += 1;
        if v == 0 || n == 8 {
            break;
        }
    }
    buf[0] = b'u';
    for i in 0..n {
        buf[1 + i] = dig[n - 1 - i];
    }
    n + 1
}

/// Packed key (keys.rs) of a slot's ID string: canonical `u` + base 36 for a
/// dense slot, the interned key for an exceptional one.
#[inline]
fn slot_key(s: u32, d: &Dense, xkeys: &[u64]) -> u64 {
    if s < d.xbase() {
        let mut buf = [0u8; 9];
        let n = slot_name(s, d, &mut buf);
        keys::key_of(&buf[..n])
    } else {
        xkeys[(s - d.xbase()) as usize]
    }
}

/// 36^k for k = 0..=8.
const P36: [u64; 9] = {
    let mut t = [1u64; 9];
    let mut k = 1;
    while k < 9 {
        t[k] = t[k - 1] * 36;
        k += 1;
    }
    t
};

/// Byte-wise order of two distinct slots' ID strings (`slot_key` order).
/// Two dense slots: both are `u` + canonical base 36 (no leading zero), so
/// with m digits each the numbers compare; otherwise the shorter one's
/// number against the longer one's leading digits (a proper prefix sorts
/// first), without building either string.
#[inline(always)]
fn slot_less(a: u32, b: u32, d: &Dense, xkeys: &[u64]) -> bool {
    let xb = d.xbase();
    if a < xb && b < xb {
        let va = scramble((a as u64 / 8) * 512 + (a as u64 % 8), d.ku);
        let vb = scramble((b as u64 / 8) * 512 + (b as u64 % 8), d.ku);
        let nd = |v: u64| -> usize {
            let mut m = 1;
            while m < 8 && v >= P36[m] {
                m += 1;
            }
            m
        };
        let (ma, mb) = (nd(va), nd(vb));
        if ma == mb {
            va < vb
        } else if ma < mb {
            // a < b unless b's first ma digits are below a's number
            vb >= va * P36[mb - ma]
        } else {
            va < vb * P36[ma - mb]
        }
    } else {
        keys::cmp(slot_key(a, d, xkeys), slot_key(b, d, xkeys)) == std::cmp::Ordering::Less
    }
}

/// Union-find over slots for `dense_links`: dense slots use `proot` itself
/// as the parent table, exceptional slots `xpar` (parent + 1; 0 = root).
struct SlotUf<'a> {
    proot: &'a [AtomicU32],
    xpar: &'a [AtomicU32],
    xbase: u32,
    d: &'a Dense,
    xkeys: &'a [u64],
}

impl SlotUf<'_> {
    #[inline(always)]
    fn ent(&self, s: u32) -> &AtomicU32 {
        if s < self.xbase {
            &self.proot[s as usize]
        } else {
            &self.xpar[(s - self.xbase) as usize]
        }
    }

    /// Root with path halving (union phase only).
    #[inline(always)]
    fn find(&self, mut x: u32) -> u32 {
        loop {
            let p = self.ent(x).load(Ordering::Acquire);
            if p == 0 {
                return x;
            }
            let p = p - 1;
            let gp = self.ent(p).load(Ordering::Acquire);
            if gp == 0 {
                return p;
            }
            // x is not a root, so it is never hooked; gp is an ancestor of x
            self.ent(x).store(gp, Ordering::Release);
            x = gp - 1;
        }
    }

    /// Root without writes (finalize: no halving store may overwrite a
    /// compressed entry).
    #[inline(always)]
    fn find_ro(&self, mut x: u32) -> u32 {
        loop {
            let p = self.ent(x).load(Ordering::Relaxed);
            if p == 0 {
                return x;
            }
            x = p - 1;
        }
    }

    /// Joins two slots; the root with the byte-wise smaller ID stays root,
    /// so every component's root is its smallest ID (the person's name).
    #[inline(always)]
    fn union(&self, a: u32, b: u32) {
        let (mut a, mut b) = (a, b);
        loop {
            a = self.find(a);
            b = self.find(b);
            if a == b {
                return;
            }
            if !slot_less(a, b, self.d, self.xkeys) {
                std::mem::swap(&mut a, &mut b);
            }
            if self.ent(b).compare_exchange(0, a + 1, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                return;
            }
        }
    }
}

/// Links straight into slots (replaces parse_b's link table + link_roots;
/// SS_DLINKS=0 restores those): parse every link line into a slot pair,
/// union-find over slots, then every linked slot's entry holds its root + 1
/// (proot for dense slots; roots keep 0 = own person) and xroot every
/// exceptional linked slot with its root. Returns the number of link pairs.
fn dense_links(links_in: &Input, ctx: &Ctx, d: &Dense, xm: &Xmap, proot: &[AtomicU32], xroot: &Mutex<Vec<(u32, u32)>>, threads: usize, marks: &mut super::Marks) -> usize {
    let next = AtomicUsize::new(0);
    let pairs: Vec<Vec<(u32, u32)>> = super::par(threads, |_| {
        let mut v = Vec::new();
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            let Some(&task) = links_in.tasks.get(i) else { break };
            let b = links_in.maps[task.file as usize].bytes();
            #[cfg(target_arch = "x86_64")]
            if ctx.avx2 {
                unsafe { dlink_task_avx2(b, task, d, xm, &mut v) };
                continue;
            }
            dlink_task_x::<false>(b, task, d, xm, &mut v);
        }
        v
    });
    // long IDs of the links checked and decodable before the union compares
    // keys (keys.rs)
    keys::verify_longs(threads);
    // every exceptional ID of a link is interned now (later ones are unlinked)
    let xkeys = xm.key_table();
    let xpar: Vec<AtomicU32> = (0..xkeys.len()).map(|_| AtomicU32::new(0)).collect();
    let uf = SlotUf { proot, xpar: &xpar, xbase: d.xbase(), d, xkeys: &xkeys };
    super::par(threads, |t| {
        const AHEAD: usize = 8;
        let v = &pairs[t];
        for k in 0..v.len() {
            if k + AHEAD < v.len() {
                let (a, b) = v[k + AHEAD];
                super::prefetch(uf.ent(a) as *const AtomicU32 as *const u8);
                super::prefetch(uf.ent(b) as *const AtomicU32 as *const u8);
            }
            uf.union(v[k].0, v[k].1);
        }
    });
    marks.mark("d.links");
    // finalize: roots of every endpoint without writes, then the stores
    let roots: Vec<Vec<(u32, u32)>> = super::par(threads, |t| {
        let mut r = Vec::new();
        for &(a, b) in &pairs[t] {
            for s in [a, b] {
                let x = uf.find_ro(s);
                if x != s {
                    r.push((s, x));
                }
            }
        }
        r
    });
    let xbase = d.xbase();
    let local: Vec<Vec<(u32, u32)>> = super::par(threads, |t| {
        let mut xl = Vec::new();
        for &(s, x) in &roots[t] {
            if s < xbase {
                proot[s as usize].store(x + 1, Ordering::Relaxed);
            } else {
                xl.push((s - xbase, x));
            }
        }
        xl
    });
    let mut xr = xroot.lock().unwrap();
    for l in local {
        xr.extend(l);
    }
    pairs.iter().map(|v| v.len()).sum()
}

/// SS_LINKCHECK (with the dense links on): the general link table +
/// link_roots on the same input and Xmap must give every slot the same
/// person as `dense_links` did. Exits nonzero on a difference.
fn link_check(links_in: &Input, ctx: &Ctx, d: &Dense, xm: &Xmap, proot: &[AtomicU32], xroot: &Mutex<Vec<(u32, u32)>>, threads: usize) {
    let lb = super::LocBits::new(threads, ctx.ps_bits).unwrap();
    let empty = Input { maps: Vec::new(), tasks: Vec::new(), bytes: 0 };
    let (lt, _) = super::parse_b(links_in, &empty, &empty, ctx, lb, None);
    let xbase = d.xbase();
    let proot2: &'static [AtomicU32] = arena(xbase as usize);
    let xroot2: Mutex<Vec<(u32, u32)>> = Mutex::new(Vec::new());
    link_roots(&lt, d, xm, proot2, &xroot2, threads);
    let nx = xm.key_table().len();
    let pmap = |pr: &[AtomicU32], xr: &Mutex<Vec<(u32, u32)>>| -> (Vec<u32>, Vec<u32>) {
        let dense: Vec<u32> = (0..xbase).map(|s| match pr[s as usize].load(Ordering::Relaxed) { 0 => s, v => v - 1 }).collect();
        let mut x: Vec<u32> = (0..nx as u32).map(|i| xbase + i).collect();
        for &(i, r) in xr.lock().unwrap().iter() {
            x[i as usize] = r;
        }
        (dense, x)
    };
    let (a, ax) = pmap(proot, xroot);
    let (b, bx) = pmap(proot2, &xroot2);
    let bad = (0..xbase as usize).filter(|&s| a[s] != b[s]).count();
    let badx = (0..nx).filter(|&i| ax[i] != bx[i]).count();
    let linked = (0..xbase as usize).filter(|&s| b[s] != s as u32).count();
    if bad != 0 || badx != 0 {
        let xkeys = xm.key_table();
        for s in (0..xbase as usize).filter(|&s| a[s] != b[s]).take(10) {
            eprintln!(
                "linkcheck: slot {} ({:?}): dense {} ({:?}) general {} ({:?})",
                s,
                String::from_utf8_lossy(&keys::decode(slot_key(s as u32, d, &xkeys))),
                a[s],
                String::from_utf8_lossy(&keys::decode(slot_key(a[s], d, &xkeys))),
                b[s],
                String::from_utf8_lossy(&keys::decode(slot_key(b[s], d, &xkeys)))
            );
        }
        eprintln!("linkcheck: MISMATCH {} dense slots, {} exceptional slots ({} linked non-root slots)", bad, badx, linked);
        std::process::exit(1);
    }
    eprintln!("linkcheck: OK {} dense slots ({} linked to another root), {} exceptional slots", xbase, linked, nx);
}

/// keys.rs's 6-bit alphabet (packed keys).
const ALPHABET: &[u8; 64] = b"-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz";

/// The user slot of the ID behind key `k` (as `user_slot` of its string).
fn key_slot(k: u64, buf: &mut [u8; 16], d: &Dense, xm: &Xmap) -> u32 {
    if keys::is_fallback(k) {
        return user_slot(&keys::decode(k), d, xm);
    }
    // keys::decode_into for a packed key
    let n = (k & 0xF) as usize;
    let codes = k >> 4;
    for (i, c) in buf.iter_mut().enumerate().take(10) {
        *c = ALPHABET[((codes >> (6 * (9 - i))) & 63) as usize];
    }
    user_slot(&buf[..n], d, xm)
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
//
// KERNEL INTERFACE (the two call sites in `run`, one per line kind):
//
//   ev_task_any(b, task, ctx, d, xm, &mut EvOut, safe_end)
//   cl_task_any(b, task, ctx, d, xm, &mut Parts<DC>, nbc, safe_end)
//
// `b` the whole mapped file, `task` the byte range [start, end) of whole
// lines (task.start at a line start), `safe_end` = b.as_ptr() + b.len()
// (reads up to there are allowed). The per-line rules are ev_line /
// the closure in cl_task_x; a replacement kernel must produce, for every
// line, exactly what they produce (order inside a bucket is free):
//
// events (EvOut), for each line with a non-empty event_id and a parsable
//   ingest time (other lines: nothing):
//   - n = event number (unscramble(decode36(event_id), d.ke)); if
//     n >= d.bound (or the ID is not canonical: n = u64::MAX) ->
//     o.xe.push(XE { key: keys::key_of(event_id), ingest, valid, us, ts,
//     pay }) (us/ts/pay 0 when not valid);
//   - else into bucket n >> d.bshift of o.de: valid copy DE { n, ingest,
//     ts: ts | VALID, us: user_slot | (PFLAG if pay != 0) }, invalid copy
//     DE { n, ingest, ts: 0, us: 0 }; a valid purchase also pushes
//     DP { n, ingest, pay } into the same bucket of o.dp.
//   valid = user and product non-empty, ts parsable and in [ws, we),
//   event_pay accepts, ingest - ts <= LATE (main::event_line's rules).
// clicks (Parts<DC>), for each line with non-empty click_id and user_id,
//   ts parsable in [ws - WEEK, we), known campaign: DC { us: user_slot,
//   camp, ts: ts - (ws - WEEK), ck: keys::key_of(click_id) } into bucket
//   min(us >> 16, nbc - 1).
// user_slot(user_id) = user_slot_w: exact for every string (exceptional
// IDs are interned in `xm`).

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[inline(never)]
unsafe fn ev_task_avx2(b: &[u8], task: Task, ctx: &Ctx, d: &Dense, xm: &Xmap, o: &mut EvOut, safe_end: usize) {
    ev_task_x::<true>(b, task, ctx, d, xm, o, safe_end)
}

pub(crate) fn ev_task_any(b: &[u8], task: Task, ctx: &Ctx, d: &Dense, xm: &Xmap, o: &mut EvOut, safe_end: usize) {
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
pub(crate) fn ev_line(f: &[&[u8]; 7], ctx: &Ctx, d: &Dense, xm: &Xmap, o: &mut EvOut, safe_end: usize) {
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
        us = user_slot_w(f[1], d, xm, wide);
    }
    let n = match decode36_w(f[0], b'e', wide) {
        Some(v) => unscramble(v, d.ke),
        None => u64::MAX,
    };
    if n >= d.bound {
        o.xe.push(XE { key: keys::key_of(f[0]), ingest, valid, us, ts, pay });
        return;
    }
    let b = (n >> d.bshift) as usize;
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
    // k3 kernel (SS_K3=1): fast lines become records directly; lines in
    // doubt go to ev_line / ev_range below
    #[cfg(target_arch = "x86_64")]
    if AVX2 && crate::k3::ON.load(Ordering::Relaxed) {
        let k = crate::k3::params(ctx, d.ke, d.ku, d.bound, d.pmax);
        let mut sink = crate::k3::EvOutSink(op, d.bshift);
        unsafe {
            crate::k3::ev_kernel::<_, _, _, { crate::k3::FULL }>(
                b,
                task.start,
                task.end,
                k,
                ctx,
                &mut sink,
                |f| ev_line(f, ctx, d, xm, &mut *op, safe_end),
                |from, to| ev_range(b, from, to, ctx, d, xm, &mut *op, safe_end),
            )
        };
        return;
    }
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
pub(crate) fn ev_range(b: &[u8], from: usize, to: usize, ctx: &Ctx, d: &Dense, xm: &Xmap, o: &mut EvOut, safe_end: usize) {
    scan::for_each_record_x::<7, _, false>(b, from, to, |f| ev_line(f, ctx, d, xm, o, safe_end));
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn cl_task_avx2(b: &[u8], task: Task, ctx: &Ctx, d: &Dense, xm: &Xmap, v: &mut Parts<DC>, nbc: usize, safe_end: usize) {
    cl_task_x::<true>(b, task, ctx, d, xm, v, nbc, safe_end)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cl_task_any(b: &[u8], task: Task, ctx: &Ctx, d: &Dense, xm: &Xmap, v: &mut Parts<DC>, nbc: usize, safe_end: usize) {
    if ctx.avx2 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            return cl_task_avx2(b, task, ctx, d, xm, v, nbc, safe_end);
        }
    }
    cl_task_x::<false>(b, task, ctx, d, xm, v, nbc, safe_end)
}

/// The click lines of one task.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn cl_task_x<const AVX2: bool>(b: &[u8], task: Task, ctx: &Ctx, d: &Dense, xm: &Xmap, v: &mut Parts<DC>, nbc: usize, safe_end: usize) {
    let vp: *mut Parts<DC> = v;
    // k3 kernel (SS_K3=1): fast lines become records directly; lines in
    // doubt go to cl_line (the rule below) / the general scanner
    #[cfg(target_arch = "x86_64")]
    if AVX2 && crate::k3::ON.load(Ordering::Relaxed) {
        let k = crate::k3::params(ctx, d.ke, d.ku, d.bound, d.pmax);
        let mut sink = crate::k3::ClSink(vp, nbc);
        unsafe {
            crate::k3::cl_kernel::<_, _, _>(
                b,
                task.start,
                task.end,
                k,
                ctx,
                &mut sink,
                |f| cl_line(f, ctx, d, xm, &mut *vp, nbc, safe_end),
                |from, to| scan::for_each_record_x::<4, _, false>(b, from, to, |f| cl_line(f, ctx, d, xm, &mut *vp, nbc, safe_end)),
            )
        };
        return;
    }
    let mut line = |f: &[&[u8]; 4]| cl_line(f, ctx, d, xm, unsafe { &mut *vp }, nbc, safe_end);
    fastev::for_each_click::<AVX2, _>(b, task.start, task.end, safe_end, &mut line);
}

/// One click line: the same rules as main::click_task.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn cl_line(f: &[&[u8]; 4], ctx: &Ctx, d: &Dense, xm: &Xmap, v: &mut Parts<DC>, nbc: usize, safe_end: usize) {
    let we = ctx.we;
    let cws = ctx.ws - super::WEEK;
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
    let us = user_slot_w(f[1], d, xm, wide);
    let ck = if wide { unsafe { keys::key_of_wide(f[0]) } } else { keys::key_of(f[0]) };
    let q = ((us >> 16) as usize).min(nbc - 1);
    v.push(q, DC { us, camp: camp as u32, ts: (t - cws) as u32, _p: 0, ck, _q: 0 });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// slot_less agrees with the byte-wise order of the packed keys of the
    /// slots' ID strings (slot_key), over random dense slots, slots near
    /// digit-count edges, and exceptional slots.
    #[test]
    fn slot_less_matches_key_order() {
        let xkeys: Vec<u64> = [&b"U1"[..], b"u0abc", b"uzzzzzzzzz", b"u", b"a", b"uZ", b"u00", b"u-1"].iter().map(|s| keys::key_of(s)).collect();
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for &pmax in &[1u64 << 10, 3_000_000, 1 << 27] {
            for k in 0..4 {
                let ku = next() & MASK;
                let d = Dense { ke: 0, ku, bound: 1 << 20, pmax, n_est: 0, s_est: 0, lines_est: 0, pfrac: 0.0, bshift: 16 };
                let xb = d.xbase();
                let pick = |r: u64| -> u32 {
                    if r % 50 == 0 {
                        xb + (r / 50 % xkeys.len() as u64) as u32
                    } else {
                        (r % xb as u64) as u32
                    }
                };
                for _ in 0..200_000 {
                    let a = pick(next());
                    let b = pick(next());
                    if a == b {
                        continue;
                    }
                    let want = keys::cmp(slot_key(a, &d, &xkeys), slot_key(b, &d, &xkeys)) == std::cmp::Ordering::Less;
                    assert_eq!(slot_less(a, b, &d, &xkeys), want, "pmax {} key {} slots {} {}", pmax, k, a, b);
                }
            }
        }
        // slots whose numbers sit on digit-count edges: search slots whose
        // scrambled value is below 36^m for small m (keys chosen so that
        // some exist) by brute force over the inverse
        let d = Dense { ke: 0, ku: 0x12345, bound: 1 << 20, pmax: 1 << 27, n_est: 0, s_est: 0, lines_est: 0, pfrac: 0.0, bshift: 16 };
        let mut edge = Vec::new();
        for m in 1..=8usize {
            for v in [P36[m - 1], P36[m - 1] + 1, P36[m] - 1, P36[m] / 2] {
                let u = unscramble(v, d.ku);
                if (u >> 9) < d.pmax && (u & 511) < 8 {
                    edge.push(((u >> 9) * 8 + (u & 511)) as u32);
                }
            }
        }
        for &a in &edge {
            for &b in &edge {
                if a != b {
                    let want = keys::cmp(slot_key(a, &d, &xkeys), slot_key(b, &d, &xkeys)) == std::cmp::Ordering::Less;
                    assert_eq!(slot_less(a, b, &d, &xkeys), want, "edge slots {} {}", a, b);
                }
            }
        }
    }
}
