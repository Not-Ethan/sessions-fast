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
use crate::dense::{decode36, decode36_w, recover, unscramble, M1, M2, MASK};
use crate::links::LinkTable;
use crate::lookup::NONE;
use crate::parts::Parts;
use crate::sess::{Cl, Out2, PersonSess};
use crate::{fastev, keys, scan};

const BSHIFT: u32 = 16;
const BSIZE: usize = 1 << BSHIFT;
const VALID: u32 = 1 << 31;
const PFLAG: u32 = 1 << 31;
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
    stat("d_bound", bound);
    stat("d_pmax", pmax);
    stat("d_sample_lines", lines);
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
    _q: u64,
}

/// An exceptional event's dedup winner with its person.
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
    // 1. links (the general code), then slot -> root slot (stored + 1; 0 =
    // the slot is its own person)
    let lb = super::LocBits::new(threads, ctx.ps_bits).unwrap();
    let empty = Input { maps: Vec::new(), tasks: Vec::new(), bytes: 0 };
    let (lt, _) = super::parse_b(links_in, &empty, &empty, ctx, lb, None);
    marks.mark("d.links");
    let xbase = d.xbase() as usize;
    let proot: &'static [AtomicU32] = arena(xbase);
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
    // 3. clicks, bucketed by user slot >> 16 (exceptional slots: last bucket)
    let nbc = (xbase >> 16) + 1;
    let estc = cl.bytes / 44 + 1;
    let capc = estc / (threads * nbc) * 5 / 4 + 64;
    let next = AtomicUsize::new(0);
    let clo: Vec<Parts<DC>> = super::par(threads, |_| {
        let mut v: Parts<DC> = Parts::with_wc(nbc, capc, true);
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            let Some(&task) = cl.tasks.get(i) else { break };
            let b = cl.maps[task.file as usize].bytes();
            let safe_end = b.as_ptr() as usize + b.len();
            cl_task_any(b, task, ctx, &d, &xm, &mut v, nbc, safe_end);
        }
        v.finish();
        v
    });
    marks.mark("d.clicks");
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
        let mut slot = vec![DE::default(); BSIZE];
        let mut occ = vec![0u64; BSIZE / 64];
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
                        let i = r.n as usize & (BSIZE - 1);
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
                        let i = r.n as usize & (BSIZE - 1);
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
            for wi in 0..BSIZE / 64 {
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
    // 8. clicks by person: each click to its person's bucket (person >> 16),
    // then per bucket a counting sort by person & 0xFFFF; clicks are kept as
    // Cl with person = the person slot
    let nbp = (np >> 16) + 1;
    let nforeign = AtomicUsize::new(0);
    let cl2: Vec<Parts<Cl>> = {
        let nextc = AtomicUsize::new(0);
        let capb = estc / (threads * nbp) * 5 / 4 + 64;
        super::par(threads, |_| {
            let mut out: Parts<Cl> = Parts::with_wc(nbp, capb, false);
            let mut nfor = 0usize;
            loop {
                let ci = nextc.fetch_add(1, Ordering::Relaxed);
                if ci >= nbc {
                    break;
                }
                for o in &clo {
                    for sl in o.get(ci) {
                        for c in sl.iter() {
                            let p = person(c.us);
                            nfor += ((p >> 16) as usize != ci) as usize;
                            out.push((p >> 16) as usize, Cl { person: p as u64, ck: c.ck, ts: c.ts, camp: c.camp });
                        }
                    }
                }
            }
            nforeign.fetch_add(nfor, Ordering::Relaxed);
            out
        })
    };
    marks.mark("d.group1");
    stat("d_foreign_clicks", nforeign.load(Ordering::Relaxed));
    stat("d_multi_persons_xwin", xwin.len());
    let mut coff = vec![0usize; nbp + 1];
    for pb in 0..nbp {
        coff[pb + 1] = coff[pb] + cl2.iter().map(|o| o.len(pb)).sum::<usize>();
    }
    let call: &'static mut [Cl] = arena_mut(coff[nbp]);
    let caddr = call.as_mut_ptr() as usize;
    // per person bucket: 65537 start offsets (relative to the bucket)
    let cidx: &'static mut [u32] = arena_mut(nbp * 65537);
    let iaddr = cidx.as_mut_ptr() as usize;
    {
        let nextp = AtomicUsize::new(0);
        super::par(threads, |_| {
            let mut cur = vec![0u32; 65536];
            loop {
                let pb = nextp.fetch_add(1, Ordering::Relaxed);
                if pb >= nbp {
                    break;
                }
                // regions written only by this bucket's iteration
                let off = unsafe { std::slice::from_raw_parts_mut((iaddr as *mut u32).add(pb * 65537), 65537) };
                let dst = unsafe { std::slice::from_raw_parts_mut((caddr as *mut Cl).add(coff[pb]), coff[pb + 1] - coff[pb]) };
                // (fresh arena memory: already zero)
                for o in &cl2 {
                    for sl in o.get(pb) {
                        for c in sl.iter() {
                            off[(c.person & 0xFFFF) as usize + 1] += 1;
                        }
                    }
                }
                for i in 0..65536 {
                    off[i + 1] += off[i];
                }
                cur.copy_from_slice(&off[..65536]);
                for o in &cl2 {
                    for sl in o.get(pb) {
                        for c in sl.iter() {
                            let k = &mut cur[(c.person & 0xFFFF) as usize];
                            dst[*k as usize] = *c;
                            *k += 1;
                        }
                    }
                }
            }
        });
    }
    // the clicks of person p: call[coff[p >> 16] + cidx[..p & 0xFFFF], ..+ 1]
    let crange = |p: u32| -> (usize, usize) {
        let pb = p as usize >> 16;
        let lo = (p & 0xFFFF) as usize;
        let base = coff[pb];
        let off = unsafe { std::slice::from_raw_parts((iaddr as *const u32).add(pb * 65537), 65537) };
        (base + off[lo] as usize, base + off[lo + 1] as usize)
    };
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
                    if runs[p as usize].load(Ordering::Relaxed) != 1 {
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
    stat("d_groups", gr.len());
    stat("d_group_runs", mv.len());
    stat("d_kept", boff.len().min(1) * (0..nb).map(|b| klen[b].load(Ordering::Relaxed)).sum::<usize>());
    marks.mark("d.runs");
    // 10. persons: the gathered ones first (largest first), then every
    // single-run person from the buckets
    let est_rows = est / threads / 4 + 4096;
    let nextb = AtomicUsize::new(0);
    let nextg = AtomicUsize::new(0);
    let outs = super::par(threads, |t| {
        let mut st = PersonSess::new(ctx, est_rows);
        let mut es: Vec<SRec> = Vec::new();
        // the person's clicks (its own range of `call`: every person is
        // processed once, so ranges are never shared), then the routine
        let finish_person = |p: u32, es: &[SRec], st: &mut PersonSess| {
            let (a, b) = crange(p);
            let cs = unsafe { std::slice::from_raw_parts_mut((caddr as *mut Cl).add(a), b - a) };
            st.person(p as u64, pmix(p, &d, &xkeys), es, cs, ctx);
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
            finish_person(p, &es, &mut st);
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
                if runs[p as usize].load(Ordering::Relaxed) == 1 {
                    finish_person(p as u32, &k[i..j], &mut st);
                }
                i = j;
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
//   - else into bucket n >> BSHIFT of o.de: valid copy DE { n, ingest,
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
#[allow(clippy::too_many_arguments)]
unsafe fn cl_task_avx2(b: &[u8], task: Task, ctx: &Ctx, d: &Dense, xm: &Xmap, v: &mut Parts<DC>, nbc: usize, safe_end: usize) {
    cl_task_x::<true>(b, task, ctx, d, xm, v, nbc, safe_end)
}

#[allow(clippy::too_many_arguments)]
fn cl_task_any(b: &[u8], task: Task, ctx: &Ctx, d: &Dense, xm: &Xmap, v: &mut Parts<DC>, nbc: usize, safe_end: usize) {
    if ctx.avx2 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            return cl_task_avx2(b, task, ctx, d, xm, v, nbc, safe_end);
        }
    }
    cl_task_x::<false>(b, task, ctx, d, xm, v, nbc, safe_end)
}

/// One click line: the same rules as main::click_task.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn cl_task_x<const AVX2: bool>(b: &[u8], task: Task, ctx: &Ctx, d: &Dense, xm: &Xmap, v: &mut Parts<DC>, nbc: usize, safe_end: usize) {
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
        let us = user_slot_w(f[1], d, xm, wide);
        let ck = if wide { unsafe { keys::key_of_wide(f[0]) } } else { keys::key_of(f[0]) };
        let q = ((us >> 16) as usize).min(nbc - 1);
        v.push(q, DC { us, camp: camp as u32, ts: (t - cws) as u32, _p: 0, ck, _q: 0 });
    };
    fastev::for_each_click::<AVX2, _>(b, task.start, task.end, safe_end, &mut line);
}
