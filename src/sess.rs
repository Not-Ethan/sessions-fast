//! Sessions phase, version 2.
//!
//! Per person partition, the surviving events and the clicks are
//! counting-sorted into buckets by the person-hash bits just below the
//! partition bits, so a bucket almost always holds one person (a bucket with
//! several is sorted by person; persons always compare exactly).
//!
//! Sessions are found without sorting events, with 30-minute slots counted
//! from window_start (slot = ts offset / 1,800,000): two events in one slot
//! are less than 30 minutes apart, so in one session; two occupied slots with
//! an empty slot between them hold events more than 30 minutes apart, so in
//! different sessions; for adjacent occupied slots the exact test
//! `min(next) - max(this) > 1,800,000` decides. Per (person, slot) only the
//! min ts, max ts and count are kept; purchases (about 3% of events) are
//! listed and assigned to their slot's session afterwards.
//!
//! Attribution walks the person's clicks sorted by ts with a sliding 7-day
//! window (the credited click among equal-ts candidates is the smallest
//! click_id). Assists are counted per activation interval of a campaign in
//! the window (a run of consecutive sessions), split by session day. Rows go
//! straight into per-thread Parquet column bodies in huge-page arenas.

use std::cmp::Ordering as CmpOrd;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::parts::Parts;
use crate::{ceil_log2, fail, keys, par, parquet, prof, sys, CRec, Ctx, SRec, DAY, DEAD, WEEK};

/// Slot width: the session gap (1,800,000 ms).
const W: u32 = crate::GAP;

/// A click in bucket order.
#[derive(Clone, Copy)]
pub(crate) struct Cl {
    pub(crate) person: u64,
    pub(crate) ck: u64,
    pub(crate) ts: u32,
    pub(crate) camp: u32,
}

#[derive(Clone, Copy)]
struct Sess {
    start: u32,
    /// day index (from the window's first day)
    day: u32,
    /// events
    cnt: u32,
    purch: u32,
    rev: u64,
    /// index of the person's first later session on a later day (or the
    /// session count)
    dend: u32,
}

/// Per (person, slot) aggregate; idle state mn = MAX, mx = 0, cnt = 0.
#[derive(Clone, Copy)]
struct Slot {
    mn: u32,
    mx: u32,
    cnt: u32,
    /// session index (valid for occupied slots after the walk)
    sidx: u32,
}

/// Append-only buffer in a lazily-faulted huge-page arena (never freed);
/// moves to a larger arena if the estimate was short.
struct Buf<T: Copy> {
    ptr: *mut T,
    len: usize,
    cap: usize,
}

impl<T: Copy> Buf<T> {
    fn new(cap: usize) -> Self {
        let cap = cap.max(64);
        Buf { ptr: sys::alloc_arena(cap * std::mem::size_of::<T>()) as *mut T, len: 0, cap }
    }
    #[cold]
    #[inline(never)]
    fn grow(&mut self, need: usize) {
        let cap = (self.cap * 2).max(need);
        let p = sys::alloc_arena(cap * std::mem::size_of::<T>()) as *mut T;
        unsafe { std::ptr::copy_nonoverlapping(self.ptr, p, self.len) };
        self.ptr = p;
        self.cap = cap;
    }
    #[inline(always)]
    fn push(&mut self, v: T) {
        if self.len == self.cap {
            self.grow(self.len + 1);
        }
        unsafe { self.ptr.add(self.len).write(v) };
        self.len += 1;
    }
    #[inline(always)]
    fn extend(&mut self, s: &[T]) {
        if self.len + s.len() > self.cap {
            self.grow(self.len + s.len());
        }
        unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(), self.ptr.add(self.len), s.len()) };
        self.len += s.len();
    }
    #[inline(always)]
    fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    #[inline(always)]
    fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
    /// Sets the length to `n` (capacity ensured; contents of new elements
    /// are whatever the arena holds, so every one must be written).
    fn set_len_for_overwrite(&mut self, n: usize) {
        if n > self.cap {
            self.len = 0;
            self.grow(n);
        }
        self.len = n;
    }
}

/// Credited sessions, purchasing sessions and revenue of one (day, campaign).
#[derive(Clone, Copy, Default)]
pub struct Cred {
    pub s: u32,
    pub p: u32,
    pub r: u64,
}

/// One thread's campaign_daily accumulators, indexed [campaign][day] (an
/// assist interval of one campaign touches consecutive days).
pub struct Out2 {
    pub nrows: usize,
    /// [camp * ndays + day]; camp ncamp = organic
    pub cred: Vec<Cred>,
    /// [camp * ndays + day]
    pub assist: Vec<u32>,
}

/// user_daily column bodies (PLAIN), built row by row.
struct Cols {
    uid: Buf<u8>,
    day: Buf<i32>,
    sess: Buf<i32>,
    evs: Buf<i32>,
    rev: Buf<i64>,
    /// definition levels of top_category, one bit per row (LSB first)
    def: Buf<u8>,
    top: Buf<u8>,
    n: usize,
}

struct St {
    hist: Vec<u32>,
    cur: Vec<u32>,
    ev: Buf<SRec>,
    chist: Vec<u32>,
    cl: Buf<Cl>,
    mixed_e: Vec<SRec>,
    /// persons with at most this many clicks use `attribute_small`
    bf_max: usize,

    /// per campaign: id of the last session it assisted (attribute_small)
    stamp: Vec<u32>,
    stamp_id: u32,
    slot: Vec<Slot>,
    bits: Vec<u64>,
    pur: Vec<(u32, u32)>,
    sess: Vec<Sess>,
    catsum: Vec<u64>,
    touched: Vec<u32>,
    wcount: Vec<u32>,
    wpos: Vec<u32>,
    /// session index at which the campaign entered the window
    since: Vec<u32>,
    active: Vec<u16>,
    /// PLAIN encoding (length + bytes) of the current person's name
    ubuf: Vec<u8>,
    /// PLAIN encoding of each category name, by rank
    cat_plain: Vec<Vec<u8>>,
    cols: Cols,
    out: Out2,
    acc: prof::Acc,
}

// sub-phase timer slots (verbose, SS_PROF)
const P_HIST: usize = 0;
const P_SCATTER: usize = 1;
const P_CLICKS: usize = 2;
const P_SLOTS: usize = 3;
const P_WALK: usize = 4;
const P_ATTR: usize = 5;
const P_ROWS: usize = 6;
const P_WRITE: usize = 7;
const P_CSORT: usize = 8;
// counters
const C_PERSONS: usize = 0;
const C_SESS: usize = 1;
const C_MIXED: usize = 2;
const C_PCLICK: usize = 3;
const C_INTERVALS: usize = 4;
const C_SMALLP: usize = 5;
const C_SMALLC: usize = 6;
const C_SMALLS: usize = 7;

fn name_slots() {
    for (i, n) in [
        (P_HIST, "s2.hist"),
        (P_SCATTER, "s2.scatter"),
        (P_CLICKS, "s2.clicks"),
        (P_SLOTS, "s2.slots"),
        (P_WALK, "s2.walk"),
        (P_CSORT, "s2.clickgather"),
        (P_ATTR, "s2.attr"),
        (P_ROWS, "s2.rows"),
        (P_WRITE, "s2.write"),
    ] {
        prof::name(i, n);
    }
    for (i, n) in [(C_PERSONS, "persons"), (C_SESS, "sessions"), (C_MIXED, "mixed_buckets"), (C_PCLICK, "person_clicks"), (C_INTERVALS, "assist_intervals"), (C_SMALLP, "small_persons"), (C_SMALLC, "small_clicks"), (C_SMALLS, "small_sessions")] {
        prof::cname(i, n);
    }
}

const IDLE: Slot = Slot { mn: u32::MAX, mx: 0, cnt: 0, sidx: 0 };

impl St {
    fn new(ctx: &Ctx, est_rows: usize, max_ev: usize, max_cl: usize) -> St {
        let nw = SLOTS_ALL / 64;
        St {
            hist: Vec::new(),
            cur: Vec::new(),
            ev: Buf::new(max_ev),
            chist: Vec::new(),
            cl: Buf::new(max_cl),
            mixed_e: Vec::new(),
            bf_max: crate::env_usize("SS_BF").unwrap_or(16),

            stamp: vec![0; ctx.ncamp.max(1)],
            stamp_id: 0,
            slot: vec![IDLE; nw * 64],
            bits: vec![0; nw],
            pur: Vec::new(),
            sess: Vec::new(),
            catsum: vec![0; ctx.ndays * 256],
            touched: Vec::new(),
            wcount: vec![0; ctx.ncamp.max(1)],
            wpos: vec![0; ctx.ncamp.max(1)],
            since: vec![0; ctx.ncamp.max(1)],
            active: Vec::new(),
            ubuf: Vec::with_capacity(64),
            cat_plain: ctx
                .cat_names
                .iter()
                .map(|c| {
                    let mut v = (c.len() as u32).to_le_bytes().to_vec();
                    v.extend_from_slice(c);
                    v
                })
                .collect(),
            cols: Cols {
                uid: Buf::new(est_rows * 16),
                day: Buf::new(est_rows),
                sess: Buf::new(est_rows),
                evs: Buf::new(est_rows),
                rev: Buf::new(est_rows),
                def: Buf::new(est_rows / 8 + 1),
                top: Buf::new(est_rows * 12),
                n: 0,
            },
            out: Out2 {
                nrows: 0,
                cred: vec![Cred::default(); ctx.ndays * (ctx.ncamp + 1)],
                assist: vec![0; ctx.ndays * ctx.ncamp],
            },
            acc: prof::Acc::new(),
        }
    }

    #[inline(never)]
    fn partition(&mut self, q: usize, sbufs: &[&Parts<SRec>], cbufs: &[&Parts<CRec>], ctx: &Ctx) {
        let n: usize = sbufs.iter().map(|t| t.len(q)).sum();
        if n == 0 {
            return;
        }
        self.acc.start();
        let bbits = ceil_log2(n / 3).clamp(4, 14).min(64 - ctx.ps_bits - 1);
        let nb = 1usize << bbits;
        let shift = 64 - ctx.ps_bits - bbits;
        let bmask = (nb - 1) as u64;
        // histogram; dead copies go to the extra bucket nb
        self.hist.clear();
        self.hist.resize(nb + 2, 0);
        // bucket of a record: the person-hash bits below the partition bits,
        // or nb for a dead copy (branch-free); always <= nb
        let bucket = |r: &SRec| -> usize {
            let dead = 0usize.wrapping_sub((r.ts == DEAD) as usize);
            (((r.person >> shift) & bmask) as usize & !dead) | (nb & dead)
        };
        {
            let hist = &mut self.hist[..nb + 2];
            for t in sbufs {
                for sl in t.get(q) {
                    for r in sl {
                        let b = bucket(r);
                        // b + 1 <= nb + 1 < hist.len()
                        unsafe { *hist.get_unchecked_mut(b + 1) += 1 };
                    }
                }
            }
        }
        for b in 0..=nb {
            self.hist[b + 1] += self.hist[b];
        }
        self.acc.lap(P_HIST);
        self.cur.clear();
        self.cur.extend_from_slice(&self.hist[..nb + 1]);
        // every one of the n records is written below (dead ones to bucket nb)
        self.ev.set_len_for_overwrite(n);
        {
            let ev = self.ev.as_mut_slice();
            let cur = &mut self.cur[..nb + 1];
            for t in sbufs {
                for sl in t.get(q) {
                    for r in sl {
                        let b = bucket(r);
                        // b <= nb < cur.len(); cur[b] < hist[b + 1] <= n = ev.len()
                        // (same buckets as the histogram over the same records)
                        unsafe {
                            let c = *cur.get_unchecked(b) as usize;
                            debug_assert!(c < ev.len());
                            *ev.get_unchecked_mut(c) = *r;
                            *cur.get_unchecked_mut(b) = c as u32 + 1;
                        }
                    }
                }
            }
        }
        self.acc.lap(P_SCATTER);
        // clicks, same buckets
        let nc: usize = cbufs.iter().map(|t| t.len(q)).sum();
        self.chist.clear();
        self.chist.resize(nb + 1, 0);
        for t in cbufs {
            for sl in t.get(q) {
                for c in sl {
                    self.chist[((c.person >> shift) & bmask) as usize + 1] += 1;
                }
            }
        }
        for b in 0..nb {
            self.chist[b + 1] += self.chist[b];
        }
        self.cur.clear();
        self.cur.extend_from_slice(&self.chist[..nb]);
        self.cl.set_len_for_overwrite(nc);
        {
            let cl = self.cl.as_mut_slice();
            for t in cbufs {
                for sl in t.get(q) {
                    for c in sl {
                        let b = ((c.person >> shift) & bmask) as usize;
                        let k = self.cur[b] as usize;
                        cl[k] = Cl { person: c.person, ck: c.ck, ts: c.ts, camp: c.camp };
                        self.cur[b] = k as u32 + 1;
                    }
                }
            }
        }
        self.acc.lap(P_CLICKS);
        let (evp, evn) = (self.ev.ptr, self.ev.len);
        let (clp, cln) = (self.cl.ptr, self.cl.len);
        // the buffers are not touched through `self` again until the next
        // partition; each bucket's click range is handed out once
        let ev: &[SRec] = unsafe { std::slice::from_raw_parts(evp, evn) };
        let cl: &mut [Cl] = unsafe { std::slice::from_raw_parts_mut(clp, cln) };
        for b in 0..nb {
            let (e0, e1) = (self.hist[b] as usize, self.hist[b + 1] as usize);
            if e0 == e1 {
                continue;
            }
            let es = &ev[e0..e1];
            let cs = &mut cl[self.chist[b] as usize..self.chist[b + 1] as usize];
            let p0 = es[0].person;
            if es.iter().all(|r| r.person == p0) {
                self.person(p0, p0, es, cs, ctx);
            } else {
                self.mixed(es, cs, ctx);
            }
        }
    }

    /// A bucket holding several persons: sort events and clicks by person.
    #[cold]
    #[inline(never)]
    fn mixed(&mut self, es: &[SRec], cs: &mut [Cl], ctx: &Ctx) {
        self.acc.count(C_MIXED, 1);
        let mut me = std::mem::take(&mut self.mixed_e);
        me.clear();
        me.extend_from_slice(es);
        me.sort_unstable_by_key(|r| r.person);
        cs.sort_unstable_by_key(|c| c.person);
        let mut i = 0;
        let mut j = 0;
        while i < me.len() {
            let p = me[i].person;
            let mut k = i;
            while k < me.len() && me[k].person == p {
                k += 1;
            }
            while j < cs.len() && cs[j].person < p {
                j += 1;
            }
            let mut l = j;
            while l < cs.len() && cs[l].person == p {
                l += 1;
            }
            self.person(p, p, &me[i..k], &mut cs[j..l], ctx);
            i = k;
            j = l;
        }
        self.mixed_e = me;
    }

    /// Campaign `c` enters the window while session `si` is being set up.
    #[inline(always)]
    fn win_add(&mut self, c: u32, si: usize) {
        let c = c as usize;
        if self.wcount[c] == 0 {
            self.wpos[c] = self.active.len() as u32;
            self.since[c] = si as u32;
            self.active.push(c as u16);
        }
        self.wcount[c] += 1;
    }

    /// A click of campaign `c` leaves the window while session `si` is being
    /// set up; if it was the campaign's last, the campaign assisted sessions
    /// since[c] .. si.
    #[inline(always)]
    fn win_remove(&mut self, c: u32, si: usize, ndays: usize) {
        let ci = c as usize;
        self.wcount[ci] -= 1;
        if self.wcount[ci] == 0 {
            self.assist_range(ci, self.since[ci] as usize, si, ndays);
            let p = self.wpos[ci] as usize;
            let last = self.active.pop().unwrap();
            if p < self.active.len() {
                self.active[p] = last;
                self.wpos[last as usize] = p as u32;
            }
        }
    }

    /// Campaign `c` assisted the person's sessions a .. b: count them per day.
    #[inline(always)]
    fn assist_range(&mut self, c: usize, a: usize, b: usize, ndays: usize) {
        let mut i = a;
        while i < b {
            let s = self.sess[i];
            let j = (s.dend as usize).min(b);
            self.out.assist[c * ndays + s.day as usize] += (j - i) as u32;
            i = j;
        }
        self.acc.count(C_INTERVALS, 1);
    }

    /// One person: events `es` (any order, all with person `p`) and the
    /// clicks of the bucket `cs` (those with person `p` are the person's).
    /// `p`: the person's value in `es` / `cs` person fields (clicks of other
    /// persons in `cs` are skipped); `name`: keys::mix of the person key.
    fn person(&mut self, p: u64, name: u64, es: &[SRec], cs: &mut [Cl], ctx: &Ctx) {
        self.acc.start();
        // 1. slots
        self.pur.clear();
        // bit w: bitmap word w is non-zero
        let mut summary = 0u64;
        {
            let slot = &mut self.slot[..SLOTS_ALL];
            let bits = &mut self.bits[..SLOTS_ALL / 64];
            for r in es {
                let t = r.ts;
                let s = (t / W) as usize;
                // s <= u32::MAX / W < SLOTS_ALL
                let a = unsafe { slot.get_unchecked_mut(s) };
                a.mn = a.mn.min(t);
                a.mx = a.mx.max(t);
                a.cnt += 1;
                unsafe { *bits.get_unchecked_mut(s >> 6) |= 1u64 << (s & 63) };
                summary |= 1u64 << (s >> 6);
                if r.pay != 0 {
                    self.pur.push((t, r.pay));
                }
            }
        }
        self.acc.lap(P_SLOTS);
        // 2. sessions: runs of occupied slots, split at gaps over W
        self.sess.clear();
        let mut last_slot = usize::MAX - 1;
        let mut last_max = 0u32;
        while summary != 0 {
            let w = summary.trailing_zeros() as usize;
            summary &= summary - 1;
            let mut word = self.bits[w];
            self.bits[w] = 0;
            while word != 0 {
                let s = w * 64 + word.trailing_zeros() as usize;
                word &= word - 1;
                let a = self.slot[s];
                // lo > last_max: every ts of slot s is above every ts of an earlier slot
                if s == last_slot + 1 && a.mn - last_max <= W {
                    let k = self.sess.len() - 1;
                    self.sess[k].cnt += a.cnt;
                } else {
                    let day = ((ctx.ws_mod_day + a.mn as u64) / DAY as u64) as u32;
                    self.sess.push(Sess { start: a.mn, day, cnt: a.cnt, purch: 0, rev: 0, dend: 0 });
                }
                self.slot[s] = Slot { sidx: (self.sess.len() - 1) as u32, ..IDLE };
                last_slot = s;
                last_max = a.mx;
            }
        }
        // day runs (sessions are in time order, so days are non-decreasing)
        let ns = self.sess.len();
        let mut end = ns as u32;
        for i in (0..ns).rev() {
            if i + 1 < ns && self.sess[i + 1].day != self.sess[i].day {
                end = i as u32 + 1;
            }
            self.sess[i].dend = end;
        }
        // purchases: revenue per session, category sums per day
        for k in 0..self.pur.len() {
            let (t, pay) = self.pur[k];
            let si = self.slot[(t / W) as usize].sidx as usize;
            let price = (pay & 0xFF_FFFF) as u64;
            let s = &mut self.sess[si];
            s.rev += price;
            s.purch = 1;
            let idx = s.day as usize * 256 + (pay >> 24) as usize;
            if self.catsum[idx] == 0 {
                self.touched.push(idx as u32);
            }
            self.catsum[idx] += price;
        }
        self.acc.lap(P_WALK);
        // 3. the person's clicks (the bucket's clicks of person p), by ts
        let mut m = cs.len();
        if let Some(k0) = cs.iter().position(|c| c.person != p) {
            m = k0;
            for k in k0 + 1..cs.len() {
                if cs[k].person == p {
                    cs[m] = cs[k];
                    m += 1;
                }
            }
        }
        let pc = &mut cs[..m];
        let small = pc.len() <= self.bf_max;
        if pc.len() > 1 && !small {
            pc.sort_unstable_by_key(|c| c.ts);
        }
        let pc = &*pc;
        self.acc.count(C_PERSONS, 1);
        self.acc.count(C_PCLICK, pc.len() as u64);
        self.acc.count(C_SESS, ns as u64);
        if small {
            self.acc.count(C_SMALLP, 1);
            self.acc.count(C_SMALLC, pc.len() as u64);
            self.acc.count(C_SMALLS, ns as u64);
        }
        self.acc.lap(P_CSORT);
        // 4. attribution
        if small {
            self.attribute_small(pc, ctx.ncamp, ctx.ndays);
        } else {
            self.attribute_window(pc, ctx.ncamp, ctx.ndays);
        }
        self.acc.lap(P_ATTR);
        // 5. one row per day
        self.ubuf.clear();
        self.ubuf.extend_from_slice(&[0; 4]);
        keys::decode_into(keys::unmix(name), &mut self.ubuf);
        let ulen = (self.ubuf.len() - 4) as u32;
        self.ubuf[..4].copy_from_slice(&ulen.to_le_bytes());
        let mut i = 0;
        while i < ns {
            let j = self.sess[i].dend as usize;
            let (mut ev, mut rev) = (0u32, 0u64);
            for s in &self.sess[i..j] {
                ev += s.cnt;
                rev += s.rev;
            }
            self.row(self.sess[i].day, (j - i) as u32, ev, rev, ctx);
            i = j;
        }
        for &idx in &self.touched {
            self.catsum[idx as usize] = 0;
        }
        self.touched.clear();
        self.acc.lap(P_ROWS);
    }

    /// Credits and assists of the current person's sessions when it has few
    /// clicks: every session checks every click (unsorted) against the rule
    /// directly. A click qualifies iff start - 7 days <= click.ts <= start
    /// (offsets: start <= c.ts <= start + 7 days); the credit is the latest
    /// qualifying ts, ties to the smallest click_id; each campaign with a
    /// qualifying click assists once (stamped per session).
    fn attribute_small(&mut self, pc: &[Cl], ncamp: usize, ndays: usize) {
        for si in 0..self.sess.len() {
            let s = self.sess[si];
            let lo = s.start;
            let hi = s.start as u64 + WEEK as u64;
            if self.stamp_id == u32::MAX {
                self.stamp.iter_mut().for_each(|x| *x = 0);
                self.stamp_id = 0;
            }
            self.stamp_id += 1;
            let sid = self.stamp_id;
            let mut best = usize::MAX;
            for (k, c) in pc.iter().enumerate() {
                if c.ts >= lo && c.ts as u64 <= hi {
                    if best == usize::MAX
                        || c.ts > pc[best].ts
                        || (c.ts == pc[best].ts && keys::cmp(c.ck, pc[best].ck) == CmpOrd::Less)
                    {
                        best = k;
                    }
                    let ci = c.camp as usize;
                    if self.stamp[ci] != sid {
                        self.stamp[ci] = sid;
                        self.out.assist[ci * ndays + s.day as usize] += 1;
                    }
                }
            }
            let credit = if best == usize::MAX { ncamp } else { pc[best].camp as usize };
            let cr = &mut self.out.cred[credit * ndays + s.day as usize];
            cr.s += 1;
            cr.p += s.purch;
            cr.r += s.rev;
        }
    }

    /// Credits and assists of the current person's sessions, its clicks
    /// sorted by ts: a sliding 7-day window, assists per activation interval.
    fn attribute_window(&mut self, pc: &[Cl], ncamp: usize, ndays: usize) {
        let ns = self.sess.len();
        let (mut add, mut rem) = (0usize, 0usize);
        for si in 0..ns {
            let s = self.sess[si];
            // click offsets c (base window_start - 7 days) qualify iff start <= c <= start + 7 days
            let hi = s.start as u64 + WEEK as u64;
            while add < pc.len() && pc[add].ts as u64 <= hi {
                self.win_add(pc[add].camp, si);
                add += 1;
            }
            while rem < add && pc[rem].ts < s.start {
                self.win_remove(pc[rem].camp, si, ndays);
                rem += 1;
            }
            let credit = if add > rem {
                // latest ts; among equal ts the smallest click_id
                let tmax = pc[add - 1].ts;
                let mut best = add - 1;
                let mut k = add - 1;
                while k > rem && pc[k - 1].ts == tmax {
                    k -= 1;
                    if keys::cmp(pc[k].ck, pc[best].ck) == CmpOrd::Less {
                        best = k;
                    }
                }
                pc[best].camp as usize
            } else {
                ncamp
            };
            let ci = credit * ndays + s.day as usize;
            let cr = &mut self.out.cred[ci];
            cr.s += 1;
            cr.p += s.purch;
            cr.r += s.rev;
        }
        // campaigns still in the window assisted every session since they entered
        while let Some(&c) = self.active.last() {
            let c = c as usize;
            self.assist_range(c, self.since[c] as usize, ns, ndays);
            self.wcount[c] = 0;
            self.active.pop();
        }
    }

    /// Appends the user_daily row of the current person for local day `day`.
    #[inline(always)]
    fn row(&mut self, day: u32, sessions: u32, events: u32, revenue: u64, ctx: &Ctx) {
        let mut top = u32::MAX;
        let mut best = 0u64;
        for &idx in &self.touched {
            if idx as usize / 256 == day as usize {
                let c = idx % 256;
                let v = self.catsum[idx as usize];
                if v > best || (v == best && c < top) {
                    best = v;
                    top = c;
                }
            }
        }
        let cols = &mut self.cols;
        cols.uid.extend(&self.ubuf);
        cols.day.push((ctx.first_day + day as i64) as i32);
        cols.sess.push(sessions as i32);
        cols.evs.push(events as i32);
        cols.rev.push(revenue as i64);
        let i = cols.n;
        if i % 8 == 0 {
            cols.def.push(0);
        }
        if top != u32::MAX {
            cols.def.as_mut_slice()[i / 8] |= 1 << (i % 8);
            cols.top.extend(&self.cat_plain[top as usize]);
        }
        cols.n += 1;
    }

    fn write(&mut self, path: &Path) {
        use parquet::Raw;
        let c = &self.cols;
        let cols = [
            Raw::Req { name: "user_id", ty: parquet::T_BYTE_ARRAY, conv: Some(parquet::CONV_UTF8), body: c.uid.as_slice() },
            Raw::Req { name: "day", ty: parquet::T_INT32, conv: Some(parquet::CONV_DATE), body: parquet::le_bytes(c.day.as_slice()) },
            Raw::Req { name: "sessions", ty: parquet::T_INT32, conv: None, body: parquet::le_bytes(c.sess.as_slice()) },
            Raw::Req { name: "events", ty: parquet::T_INT32, conv: None, body: parquet::le_bytes(c.evs.as_slice()) },
            Raw::Req { name: "revenue_cents", ty: parquet::T_INT64, conv: None, body: parquet::le_bytes(c.rev.as_slice()) },
            Raw::OptStr { name: "top_category", def: c.def.as_slice(), body: c.top.as_slice() },
        ];
        parquet::write_raw(path, c.n, &cols).unwrap_or_else(|e| fail(&format!("{}: {}", path.display(), e)));
    }
}

/// Slot arrays cover every u32 offset (u32::MAX / W < 2432), so a slot
/// index computed from any u32 is in bounds; real offsets are below the
/// window length (at most 24 days: 1152 slots, 18 bitmap words).
const SLOTS_ALL: usize = 2432;
const _: () = assert!(((u32::MAX / W) as usize) < SLOTS_ALL && SLOTS_ALL % 64 == 0 && SLOTS_ALL / 64 <= 64);

/// One thread's per-person sessions state for the dense pipeline (dense.rs):
/// the same exact per-person routine as `sessions2`, fed one person at a
/// time (all its surviving events and all its clicks).
pub(crate) struct PersonSess(St);

impl PersonSess {
    pub(crate) fn new(ctx: &Ctx, est_rows: usize) -> Self {
        PersonSess(St::new(ctx, est_rows, 64, 64))
    }

    /// One person: `es` all its surviving events (any order; only ts and
    /// pay are read), `cs` all its clicks (person field = `p`); `name` =
    /// keys::mix(person key).
    #[inline]
    pub(crate) fn person(&mut self, p: u64, name: u64, es: &[SRec], cs: &mut [Cl], ctx: &Ctx) {
        self.0.person(p, name, es, cs, ctx)
    }

    /// Writes this thread's user_daily part (thread 0 always) and returns
    /// its campaign accumulators, as `sessions2` does.
    pub(crate) fn finish(mut self, ud: &Path, t: usize) -> Out2 {
        let st = &mut self.0;
        if t == 0 || st.cols.n > 0 {
            st.acc.start();
            st.write(&ud.join(format!("part-{}.parquet", t)));
            st.acc.lap(P_WRITE);
        }
        st.acc.commit();
        st.out.nrows = st.cols.n;
        std::mem::replace(&mut st.out, Out2 { nrows: 0, cred: Vec::new(), assist: Vec::new() })
    }
}

/// Sessions, attribution and aggregation per person partition; each thread
/// then writes its own user_daily part file.
/// `dead`: further memory ranges nobody reads during this phase (the input
/// file mappings); with every arena allocated before this phase, they are
/// released in parallel once all partitions are done (SS_KEEPMEM keeps
/// them), so the single-threaded exit teardown has less to free.
pub fn sessions2(sbufs: &[&Parts<SRec>], cbufs: &[&Parts<CRec>], ctx: &Ctx, ud: &Path, dead: &[(usize, usize)]) -> Vec<Out2> {
    if prof::ON.load(Ordering::Relaxed) {
        name_slots();
    }
    let ps = 1usize << ctx.ps_bits;
    let mut order: Vec<(usize, usize)> = (0..ps).map(|q| (sbufs.iter().map(|t| t.len(q)).sum::<usize>(), q)).collect();
    let total: usize = order.iter().map(|o| o.0).sum();
    let max_cl = (0..ps).map(|q| cbufs.iter().map(|t| t.len(q)).sum::<usize>()).max().unwrap_or(0);
    // biggest partitions first, for load balance (whales)
    order.sort_unstable_by(|a, b| b.cmp(a));
    let max_ev = order.first().map_or(0, |o| o.0);
    // generous: the arenas only take memory where written
    let est_rows = total / ctx.threads / 4 + 4096;
    let next = AtomicUsize::new(0);
    // Arenas that exist now (parse and dedup outputs, link table) are only
    // read by the partition loop below; arenas allocated from here on
    // (this phase's buffers and output columns) are not in the list.
    let release = std::env::var("SS_KEEPMEM").is_err();
    let mut pieces: Vec<(usize, usize)> = Vec::new();
    if release {
        for (a, l) in sys::arenas().into_iter().chain(dead.iter().copied()) {
            let mut o = 0;
            while o < l {
                let n = (l - o).min(32 << 20);
                pieces.push((a + o, n));
                o += n;
            }
        }
    }
    let bar = std::sync::Barrier::new(ctx.threads);
    let next_piece = AtomicUsize::new(0);
    par(ctx.threads, |t| {
        let mut st = St::new(ctx, est_rows, max_ev, max_cl);
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            let Some(&(_, q)) = order.get(i) else { break };
            st.partition(q, sbufs, cbufs, ctx);
        }
        if release {
            // this thread's partition buffers are done with
            sys::release(st.ev.ptr as usize, st.ev.cap * std::mem::size_of::<SRec>());
            sys::release(st.cl.ptr as usize, st.cl.cap * std::mem::size_of::<Cl>());
            // after every thread's last partition, the inputs are dead
            bar.wait();
            loop {
                let i = next_piece.fetch_add(1, Ordering::Relaxed);
                let Some(&(a, l)) = pieces.get(i) else { break };
                sys::release(a, l);
            }
        }
        if t == 0 || st.cols.n > 0 {
            st.acc.start();
            st.write(&ud.join(format!("part-{}.parquet", t)));
            st.acc.lap(P_WRITE);
        }
        st.acc.commit();
        st.out.nrows = st.cols.n;
        st.out
    })
}

/// Merges the threads' accumulators (into the first one's arrays, already
/// resident) and writes campaign_daily from huge-page column buffers.
pub fn write_campaign_daily2(path: &Path, outs: &mut [Out2], ctx: &Ctx) {
    let ndays = ctx.ndays;
    let ncamp = ctx.ncamp;
    let nc1 = ncamp + 1;
    let (first, rest) = outs.split_first_mut().expect("at least one thread");
    for o in rest.iter() {
        for (a, b) in first.cred.iter_mut().zip(&o.cred) {
            a.s += b.s;
            a.p += b.p;
            a.r += b.r;
        }
        for (a, b) in first.assist.iter_mut().zip(&o.assist) {
            *a += *b;
        }
    }
    let est = nc1 * ndays;
    let mut ids: Buf<u8> = Buf::new(est * 16);
    let mut chs: Buf<u8> = Buf::new(est * 24);
    let mut days: Buf<i32> = Buf::new(est);
    let (mut sv, mut pv, mut rv, mut av): (Buf<i64>, Buf<i64>, Buf<i64>, Buf<i64>) = (Buf::new(est), Buf::new(est), Buf::new(est), Buf::new(est));
    let plain = |b: &mut Buf<u8>, x: &[u8]| {
        b.extend(&(x.len() as u32).to_le_bytes());
        b.extend(x);
    };
    for c in 0..nc1 {
        for d in 0..ndays {
            let k = c * ndays + d;
            let cr = first.cred[k];
            let a = if c < ncamp { first.assist[k] } else { 0 };
            if cr.s == 0 && a == 0 {
                continue;
            }
            if c < ncamp {
                plain(&mut ids, &ctx.camp_ids[c]);
                plain(&mut chs, &ctx.camp_channels[c]);
            } else {
                plain(&mut ids, b"organic");
                plain(&mut chs, b"organic");
            }
            days.push((ctx.first_day + d as i64) as i32);
            sv.push(cr.s as i64);
            pv.push(cr.p as i64);
            rv.push(cr.r as i64);
            av.push(a as i64);
        }
    }
    use parquet::Raw;
    let cols = [
        Raw::Req { name: "campaign_id", ty: parquet::T_BYTE_ARRAY, conv: Some(parquet::CONV_UTF8), body: ids.as_slice() },
        Raw::Req { name: "channel", ty: parquet::T_BYTE_ARRAY, conv: Some(parquet::CONV_UTF8), body: chs.as_slice() },
        Raw::Req { name: "day", ty: parquet::T_INT32, conv: Some(parquet::CONV_DATE), body: parquet::le_bytes(days.as_slice()) },
        Raw::Req { name: "sessions", ty: parquet::T_INT64, conv: None, body: parquet::le_bytes(sv.as_slice()) },
        Raw::Req { name: "purchasing_sessions", ty: parquet::T_INT64, conv: None, body: parquet::le_bytes(pv.as_slice()) },
        Raw::Req { name: "revenue_cents", ty: parquet::T_INT64, conv: None, body: parquet::le_bytes(rv.as_slice()) },
        Raw::Req { name: "assisted_sessions", ty: parquet::T_INT64, conv: None, body: parquet::le_bytes(av.as_slice()) },
    ];
    parquet::write_raw(path, days.len, &cols).unwrap_or_else(|e| fail(&format!("{}: {}", path.display(), e)));
}
