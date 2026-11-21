//! Identity resolution: a concurrent open-addressing table of linked IDs plus
//! a lock-free union-find over dense indices.
//!
//! Every linked ID gets a slot in the table (key -> value) and a dense index
//! (assigned on first insertion). Union-find runs on a compact parent array
//! indexed by dense index, so it stays in cache instead of chasing pointers
//! through the table. Roots are always hooked under the root with the smaller
//! key, so keys strictly decrease along parent pointers and every component's
//! root is its byte-wise smallest ID: the person's name. Finalize writes each
//! slot's person key into its value, so a lookup is one probe.

use crate::keys;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

pub struct LinkTable {
    /// [key, val] per slot. While building, val = dense index + 1 (0 while
    /// the inserting thread has not published it yet); after `finalize_vals`
    /// val = the person key.
    slots: &'static [[AtomicU64; 2]],
    mask: usize,
    /// probe start mask: !3 starts every probe at a 64-byte group of 4 slots,
    /// so one prefetched cache line usually covers the whole probe (insert
    /// and lookup use the same start, so this is exact either way)
    amask: usize,
    /// dense index -> parent dense index + 1 (0 = root)
    parent: &'static [AtomicU32],
    /// dense index -> key
    dkey: &'static [AtomicU64],
    /// dense index -> person key (the root's key; finalize scratch)
    dpkey: &'static [AtomicU64],
    /// next free block of dense indices
    next_block: AtomicU32,
    /// after finalize, slot values hold keys::mix(person key) instead of the
    /// person key (SS_OLDPH=1: raw keys, mixed at every lookup, as before)
    mixed: bool,
}

pub const FINALIZE_BLOCK: usize = 1 << 16;
const DENSE_BLOCK: u32 = 4096;

fn arena<T>(n: usize) -> &'static [T] {
    let p = crate::sys::alloc_arena(n * std::mem::size_of::<T>()) as *const T;
    unsafe { std::slice::from_raw_parts(p, n) }
}

/// A thread's current block of dense indices.
pub struct DenseAlloc {
    next: u32,
    end: u32,
}

impl DenseAlloc {
    pub fn new() -> Self {
        DenseAlloc { next: 0, end: 0 }
    }
}

impl LinkTable {
    pub fn empty() -> Self {
        LinkTable { slots: &[], mask: 0, amask: !0, parent: &[], dkey: &[], dpkey: &[], next_block: AtomicU32::new(0), mixed: false }
    }

    /// A table for `total_pairs` links built by up to `threads` threads.
    pub fn with_pairs(total_pairs: usize, threads: usize) -> Self {
        if total_pairs == 0 {
            return Self::empty();
        }
        // at most 2 distinct IDs per pair, so the load factor stays below 1
        // (typically ~0.55: ~1.3 IDs per pair)
        let f = std::env::var("SS_LINKCAP").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(4).max(2);
        let cap = (total_pairs * f + 16).next_power_of_two();
        // at most 2 new IDs per pair, plus partly used blocks
        let nd = 2 * total_pairs + (threads + 1) * DENSE_BLOCK as usize + 64;
        assert!(nd < u32::MAX as usize - DENSE_BLOCK as usize * 2, "too many links");
        LinkTable {
            slots: arena(cap),
            mask: cap - 1,
            amask: if std::env::var("SS_LINKALIGN").map_or(false, |v| v == "0") { !0 } else { !3 },
            parent: arena(nd),
            dkey: arena(nd),
            dpkey: arena(nd),
            next_block: AtomicU32::new(0),
            mixed: std::env::var("SS_OLDPH").map_or(true, |v| v != "1"),
        }
    }

    #[inline(always)]
    fn alloc(&self, a: &mut DenseAlloc) -> u32 {
        if a.next == a.end {
            let b = self.next_block.fetch_add(DENSE_BLOCK, Ordering::Relaxed);
            a.next = b;
            a.end = b + DENSE_BLOCK;
            assert!((a.end as usize) <= self.parent.len(), "dense index space exhausted");
        }
        let d = a.next;
        a.next += 1;
        d
    }

    /// First probe slot for hash h.
    #[inline(always)]
    fn start(&self, h: u64) -> usize {
        h as usize & self.mask & self.amask
    }

    /// Dense index of key k, inserting it if absent.
    #[inline(always)]
    fn insert(&self, k: u64, a: &mut DenseAlloc) -> u32 {
        let mut i = self.start(keys::mix(k));
        loop {
            let cur = self.slots[i][0].load(Ordering::Acquire);
            if cur == 0 {
                match self.slots[i][0].compare_exchange(0, k, Ordering::AcqRel, Ordering::Acquire) {
                    Ok(_) => {
                        let d = self.alloc(a);
                        self.dkey[d as usize].store(k, Ordering::Relaxed);
                        self.slots[i][1].store(d as u64 + 1, Ordering::Release);
                        return d;
                    }
                    Err(actual) if actual == k => return self.wait_dense(i),
                    Err(_) => {}
                }
            } else if cur == k {
                return self.wait_dense(i);
            }
            i = (i + 1) & self.mask;
        }
    }

    /// The dense index of slot i, once its inserting thread has published it.
    #[inline(always)]
    fn wait_dense(&self, i: usize) -> u32 {
        loop {
            let v = self.slots[i][1].load(Ordering::Acquire);
            if v != 0 {
                return (v - 1) as u32;
            }
            std::hint::spin_loop();
        }
    }

    #[inline(always)]
    fn find(&self, mut x: u32) -> u32 {
        loop {
            let p = self.parent[x as usize].load(Ordering::Acquire);
            if p == 0 {
                return x;
            }
            let p = p - 1;
            let gp = self.parent[p as usize].load(Ordering::Acquire);
            if gp == 0 {
                return p;
            }
            // path halving; x is not a root so it is never hooked again, and gp
            // is still an ancestor of x.
            self.parent[x as usize].store(gp, Ordering::Release);
            x = gp - 1;
        }
    }

    #[inline(always)]
    fn union(&self, a: u32, b: u32) {
        let (mut a, mut b) = (a, b);
        loop {
            a = self.find(a);
            b = self.find(b);
            if a == b {
                return;
            }
            let ka = self.dkey[a as usize].load(Ordering::Relaxed);
            let kb = self.dkey[b as usize].load(Ordering::Relaxed);
            if keys::cmp(ka, kb) == std::cmp::Ordering::Greater {
                std::mem::swap(&mut a, &mut b);
            }
            // hook b (larger key) under a (smaller key)
            if self.parent[b as usize].compare_exchange(0, a + 1, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                return;
            }
        }
    }

    /// Joins two IDs. Safe to call from many threads at once.
    #[inline(always)]
    #[allow(dead_code)]
    pub fn link(&self, a: u64, b: u64, alloc: &mut DenseAlloc) {
        let da = self.insert(a, alloc);
        let db = self.insert(b, alloc);
        if da != db {
            self.union(da, db);
        }
    }

    /// Dense indices of both IDs of a link, inserting them if absent.
    #[inline(always)]
    pub fn insert_pair(&self, a: u64, b: u64, alloc: &mut DenseAlloc) -> (u32, u32) {
        (self.insert(a, alloc), self.insert(b, alloc))
    }

    /// Joins two inserted IDs by dense index. Safe to call from many threads
    /// at once, also while other threads are still inserting.
    #[inline(always)]
    pub fn union_dense(&self, da: u32, db: u32) {
        if da != db {
            self.union(da, db);
        }
    }

    /// Prefetches the union-find entries of a dense index.
    #[inline(always)]
    pub fn prefetch_dense(&self, d: u32) {
        crate::prefetch(self.parent.as_ptr().wrapping_add(d as usize) as *const u8);
        crate::prefetch(self.dkey.as_ptr().wrapping_add(d as usize) as *const u8);
    }

    /// Number of blocks for `finalize_roots` (dense indices).
    pub fn root_blocks(&self) -> usize {
        (self.next_block.load(Ordering::Acquire) as usize).min(self.parent.len()).div_ceil(FINALIZE_BLOCK)
    }

    /// Number of blocks for `finalize_vals` (table slots).
    pub fn val_blocks(&self) -> usize {
        self.slots.len().div_ceil(FINALIZE_BLOCK)
    }

    /// Finalize, step 1 (after all links): person key (key of the root) of
    /// every dense index in block `b`. Blocks may run on any threads,
    /// concurrently.
    pub fn finalize_roots(&self, b: usize) {
        let n = (self.next_block.load(Ordering::Acquire) as usize).min(self.parent.len());
        for d in b * FINALIZE_BLOCK..((b + 1) * FINALIZE_BLOCK).min(n) {
            let r = self.find(d as u32) as usize;
            self.dpkey[d].store(self.dkey[r].load(Ordering::Relaxed), Ordering::Relaxed);
        }
    }

    /// Finalize, step 2 (after every block finished step 1): each occupied
    /// slot's value becomes its person key, stored as keys::mix(person key)
    /// when `mixed` (the lookups below undo or use that consistently).
    pub fn finalize_vals(&self, b: usize) {
        let n = self.slots.len();
        for i in b * FINALIZE_BLOCK..((b + 1) * FINALIZE_BLOCK).min(n) {
            if self.slots[i][0].load(Ordering::Relaxed) != 0 {
                let d = (self.slots[i][1].load(Ordering::Relaxed) - 1) as usize;
                let pk = self.dpkey[d].load(Ordering::Relaxed);
                self.slots[i][1].store(if self.mixed { keys::mix(pk) } else { pk }, Ordering::Relaxed);
            }
        }
    }

    /// The person key held in a finalized slot value.
    #[inline(always)]
    fn val_key(&self, v: u64) -> u64 {
        if self.mixed {
            keys::unmix(v)
        } else {
            v
        }
    }

    /// The person key of user key `k` (after `finalize`).
    #[inline(always)]
    pub fn person_of(&self, k: u64) -> u64 {
        if self.slots.is_empty() {
            return k;
        }
        let mut i = self.start(keys::mix(k));
        loop {
            let s = &self.slots[i];
            let kk = s[0].load(Ordering::Relaxed);
            if kk == k {
                return self.val_key(s[1].load(Ordering::Relaxed));
            }
            if kk == 0 {
                return k;
            }
            i = (i + 1) & self.mask;
        }
    }

    /// `person_of` for a key whose hash `h` = keys::mix(k) is known:
    /// Some(person key) if k is linked, None if it is not in the table.
    #[inline(always)]
    #[allow(dead_code)]
    pub fn person_of_h(&self, k: u64, h: u64) -> Option<u64> {
        if self.slots.is_empty() {
            return None;
        }
        let mut i = self.start(h);
        loop {
            let s = &self.slots[i];
            let kk = s[0].load(Ordering::Relaxed);
            if kk == k {
                return Some(self.val_key(s[1].load(Ordering::Relaxed)));
            }
            if kk == 0 {
                return None;
            }
            i = (i + 1) & self.mask;
        }
    }

    /// The person key of user key `k` (hash `h`) after `finalize`, or `k`
    /// itself when `k` is not in the table. Same result as `person_of_h`
    /// (`unwrap_or(k)`), but probes a whole 64-byte group of 4 slots per
    /// step: one compare per slot into a match mask and an empty mask, and a
    /// select instead of data-dependent branches. Exact for linear probing
    /// without deletions: a key sits after every slot of its probe sequence
    /// that precedes it, and those were all occupied when it was inserted.
    #[inline(always)]
    #[allow(dead_code)]
    pub fn person_or_self(&self, k: u64, h: u64) -> u64 {
        match self.probe(k, h) {
            Some(v) => self.val_key(v),
            None => k,
        }
    }

    /// keys::mix of the person key of user key `k`, whose hash `h` =
    /// keys::mix(k) is known (the user's own hash when it is not linked).
    /// With `mixed` the table already holds the mixed value.
    #[inline(always)]
    pub fn person_hash(&self, k: u64, h: u64) -> u64 {
        debug_assert_eq!(h, keys::mix(k));
        match self.probe(k, h) {
            Some(v) => {
                if self.mixed {
                    v
                } else {
                    keys::mix(v)
                }
            }
            None => h,
        }
    }

    /// The raw finalized slot value of key `k` (hash `h`), if `k` is in the
    /// table: the group probe described above.
    #[inline(always)]
    fn probe(&self, k: u64, h: u64) -> Option<u64> {
        if self.slots.is_empty() {
            return None;
        }
        if self.amask != !3 {
            return self.probe_slots(k, h);
        }
        let mut i = self.start(h);
        loop {
            // i is a multiple of 4 and the table holds a power of two >= 16
            // slots, so the group is in bounds
            let g = unsafe { self.slots.get_unchecked(i..i + 4) };
            let k0 = g[0][0].load(Ordering::Relaxed);
            let k1 = g[1][0].load(Ordering::Relaxed);
            let k2 = g[2][0].load(Ordering::Relaxed);
            let k3 = g[3][0].load(Ordering::Relaxed);
            let m = (k0 == k) as u32 | ((k1 == k) as u32) << 1 | ((k2 == k) as u32) << 2 | ((k3 == k) as u32) << 3;
            let z = (k0 == 0) as u32 | ((k1 == 0) as u32) << 1 | ((k2 == 0) as u32) << 2 | ((k3 == 0) as u32) << 3;
            if m | z != 0 {
                let j = (m | 16).trailing_zeros() as usize & 3;
                let v = g[j][1].load(Ordering::Relaxed);
                return if m != 0 { Some(v) } else { None };
            }
            i = (i + 4) & self.mask;
        }
    }

    /// `probe`, one slot at a time.
    #[inline(always)]
    fn probe_slots(&self, k: u64, h: u64) -> Option<u64> {
        let mut i = self.start(h);
        loop {
            let s = &self.slots[i];
            let kk = s[0].load(Ordering::Relaxed);
            if kk == k {
                return Some(s[1].load(Ordering::Relaxed));
            }
            if kk == 0 {
                return None;
            }
            i = (i + 1) & self.mask;
        }
    }

    #[inline(always)]
    pub fn prefetch_h(&self, h: u64) {
        if self.slots.is_empty() {
            return;
        }
        let i = self.start(h);
        crate::prefetch(self.slots.as_ptr().wrapping_add(i) as *const u8);
    }

    #[inline(always)]
    pub fn prefetch(&self, k: u64) {
        if self.slots.is_empty() {
            return;
        }
        let i = self.start(keys::mix(k));
        crate::prefetch(self.slots.as_ptr().wrapping_add(i) as *const u8);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The group probe agrees with the slot-by-slot probe, also at a high
    /// load factor (full groups, probes that wrap around the table end).
    #[test]
    fn group_probe_matches() {
        let mut x: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let key = |n: u64| keys::key_of(format!("u{:x}", n).as_bytes());
        for &(declared, actual, space) in &[(1000usize, 1800usize, 5000u64), (1000, 400, 3000), (50, 90, 1 << 40)] {
            let t = LinkTable::with_pairs(declared, 1);
            let mut a = DenseAlloc::new();
            let mut pairs = Vec::new();
            let mut inserted = Vec::new();
            for _ in 0..actual {
                let (u, v) = (key(next() % space + 1), key(next() % space + 1));
                let (du, dv) = t.insert_pair(u, v, &mut a);
                pairs.push((du, dv));
                inserted.push(u);
                inserted.push(v);
            }
            for &(du, dv) in &pairs {
                t.union_dense(du, dv);
            }
            for b in 0..t.root_blocks() {
                t.finalize_roots(b);
            }
            for b in 0..t.val_blocks() {
                t.finalize_vals(b);
            }
            let mut found = 0;
            for n in 1..=space.min(20_000) {
                let k = key(n);
                let h = keys::mix(k);
                let want = t.person_of_h(k, h).unwrap_or(k);
                assert_eq!(t.person_or_self(k, h), want, "key {}", n);
                found += t.person_of_h(k, h).is_some() as u32;
            }
            for _ in 0..20_000 {
                let k = key(next());
                let h = keys::mix(k);
                assert_eq!(t.person_or_self(k, h), t.person_of_h(k, h).unwrap_or(k));
            }
            for &k in &inserted {
                let h = keys::mix(k);
                let p = t.person_of_h(k, h).expect("inserted key present");
                assert_eq!(t.person_or_self(k, h), p);
                assert_eq!(t.person_hash(k, h), keys::mix(p));
                found += 1;
            }
            for n in 1..=2000u64 {
                let k = key(next());
                let h = keys::mix(k);
                assert_eq!(t.person_hash(k, h), keys::mix(t.person_of_h(k, h).unwrap_or(k)));
                let _ = n;
            }
            assert!(found > 0);
        }
        assert_eq!(LinkTable::empty().person_or_self(7, keys::mix(7)), 7);
    }
}
