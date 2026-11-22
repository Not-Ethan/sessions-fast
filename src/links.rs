//! Identity resolution: a concurrent open-addressing table of linked IDs plus
//! a lock-free union-find over compact indices.
//!
//! Every linked ID gets a slot in the table (key -> value) and a compact index
//! (assigned on first insertion). Union-find runs on a compact parent array
//! indexed by compact index, so it stays in cache instead of chasing pointers
//! through the table. Roots are always hooked under the root with the smaller
//! key, so keys strictly decrease along parent pointers and every component's
//! root is its byte-wise smallest ID: the person's name. Finalize writes each
//! slot's person key into its value, so a lookup is one probe.

use crate::keys;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

pub struct LinkTable {
    /// [key, val] per slot. While building, val = compact index + 1 (0 while
    /// the inserting thread has not published it yet); after `finalize_vals`
    /// val = the person key.
    slots: &'static [[AtomicU64; 2]],
    mask: usize,
    /// probe start mask: !3 starts every probe at a 64-byte group of 4 slots,
    /// so one prefetched cache line usually covers the whole probe (insert
    /// and lookup use the same start, so this is exact either way)
    amask: usize,
    /// compact index -> parent compact index + 1 (0 = root)
    parent: &'static [AtomicU32],
    /// compact index -> key
    dkey: &'static [AtomicU64],
    /// compact index -> the first 8 bytes of the ID (`pref8`; SS_LINKPREF=0:
    /// unused, every root comparison through keys::cmp)
    dpref: &'static [AtomicU64],
    /// compact index -> person key (the root's key; finalize scratch)
    dpkey: &'static [AtomicU64],
    /// next free block of compact indices
    next_block: AtomicU32,
    /// after finalize, slot values hold keys::mix(person key) instead of the
    /// person key (SS_OLDPH=1: raw keys, mixed at every lookup, as before)
    mixed: bool,
}

pub const FINALIZE_BLOCK: usize = 1 << 16;

/// Root order by 8-byte prefixes first (SS_LINKPREF=0: keys::cmp only, A/B).
pub static LINK_PREF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// The first 8 bytes of `s` as a big-endian number, zero-padded: for two
/// strings with different prefixes, prefix order is byte-wise string order
/// (a shorter string's padding sorts below any byte the other has there, or
/// equals a 0 byte, in which case the prefixes do not decide).
#[inline(always)]
pub fn pref8(s: &[u8]) -> u64 {
    let mut b = [0u8; 8];
    let n = s.len().min(8);
    b[..n].copy_from_slice(&s[..n]);
    u64::from_be_bytes(b)
}

/// SS_SCALARPROBE=1: the scalar group probe in AVX2 builds too (A/B).
pub static SCALAR_PROBE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// AVX2 person lookups select found / not found instead of branching on it
/// (`k5::probe_sel`; SS_PROBESEL=0: the branching form, A/B).
pub static PROBE_SEL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// a if c else b, without a branch.
#[inline(always)]
pub fn sel64(c: bool, a: u64, b: u64) -> u64 {
    let m = (c as u64).wrapping_neg();
    (a & m) | (b & !m)
}
const IDX_BLOCK: u32 = 4096;

fn arena<T>(n: usize) -> &'static [T] {
    let p = crate::sys::alloc_arena(n * std::mem::size_of::<T>()) as *const T;
    unsafe { std::slice::from_raw_parts(p, n) }
}

/// A thread's current block of compact indices.
pub struct IdxAlloc {
    next: u32,
    end: u32,
}

impl IdxAlloc {
    pub fn new() -> Self {
        IdxAlloc { next: 0, end: 0 }
    }
}

impl LinkTable {
    pub fn empty() -> Self {
        LinkTable { slots: &[], mask: 0, amask: !0, parent: &[], dkey: &[], dpref: &[], dpkey: &[], next_block: AtomicU32::new(0), mixed: false }
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
        let nd = 2 * total_pairs + (threads + 1) * IDX_BLOCK as usize + 64;
        assert!(nd < u32::MAX as usize - IDX_BLOCK as usize * 2, "too many links");
        LinkTable {
            slots: arena(cap),
            mask: cap - 1,
            amask: if std::env::var("SS_LINKALIGN").map_or(false, |v| v == "0") { !0 } else { !3 },
            parent: arena(nd),
            dkey: arena(nd),
            dpref: arena(nd),
            dpkey: arena(nd),
            next_block: AtomicU32::new(0),
            mixed: std::env::var("SS_OLDPH").map_or(true, |v| v != "1"),
        }
    }

    #[inline(always)]
    fn alloc(&self, a: &mut IdxAlloc) -> u32 {
        if a.next == a.end {
            let b = self.next_block.fetch_add(IDX_BLOCK, Ordering::Relaxed);
            a.next = b;
            a.end = b + IDX_BLOCK;
            assert!((a.end as usize) <= self.parent.len(), "compact index space exhausted");
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

    /// Compact index of key k, inserting it if absent.
    #[inline(always)]
    fn insert(&self, k: u64, pref: u64, a: &mut IdxAlloc) -> u32 {
        let mut i = self.start(keys::mix(k));
        loop {
            let cur = self.slots[i][0].load(Ordering::Acquire);
            if cur == 0 {
                match self.slots[i][0].compare_exchange(0, k, Ordering::AcqRel, Ordering::Acquire) {
                    Ok(_) => {
                        let d = self.alloc(a);
                        self.dkey[d as usize].store(k, Ordering::Relaxed);
                        self.dpref[d as usize].store(pref, Ordering::Relaxed);
                        self.slots[i][1].store(d as u64 + 1, Ordering::Release);
                        return d;
                    }
                    Err(actual) if actual == k => return self.wait_idx(i),
                    Err(_) => {}
                }
            } else if cur == k {
                return self.wait_idx(i);
            }
            i = (i + 1) & self.mask;
        }
    }

    /// The compact index of slot i, once its inserting thread has published it.
    #[inline(always)]
    fn wait_idx(&self, i: usize) -> u32 {
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
            if self.greater(a, b) {
                std::mem::swap(&mut a, &mut b);
            }
            // hook b (larger key) under a (smaller key)
            if self.parent[b as usize].compare_exchange(0, a + 1, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                return;
            }
        }
    }

    /// Whether the ID of compact index a sorts after that of b (byte-wise):
    /// by their first 8 bytes when those differ (padding zeros below every
    /// byte, so a different prefix orders the strings exactly as the bytes
    /// do), else by the keys (keys::cmp; long keys through the decode table).
    #[inline(always)]
    fn greater(&self, a: u32, b: u32) -> bool {
        if LINK_PREF.load(Ordering::Relaxed) {
            let pa = self.dpref[a as usize].load(Ordering::Relaxed);
            let pb = self.dpref[b as usize].load(Ordering::Relaxed);
            if pa != pb {
                return pa > pb;
            }
        }
        let ka = self.dkey[a as usize].load(Ordering::Relaxed);
        let kb = self.dkey[b as usize].load(Ordering::Relaxed);
        keys::cmp(ka, kb) == std::cmp::Ordering::Greater
    }

    /// Joins two IDs. Safe to call from many threads at once.
    #[inline(always)]
    #[allow(dead_code)]
    /// (Tests: every ID gets prefix 0, so roots are ordered by keys::cmp
    /// alone; a table must not mix this with `insert_pair`'s prefixes.)
    pub fn link(&self, a: u64, b: u64, alloc: &mut IdxAlloc) {
        let da = self.insert(a, 0, alloc);
        let db = self.insert(b, 0, alloc);
        if da != db {
            self.union(da, db);
        }
    }

    /// Compact indices of both IDs of a link, inserting them if absent.
    #[inline(always)]
    pub fn insert_pair(&self, a: u64, b: u64, pa: u64, pb: u64, alloc: &mut IdxAlloc) -> (u32, u32) {
        (self.insert(a, pa, alloc), self.insert(b, pb, alloc))
    }

    /// Joins two inserted IDs by compact index. Safe to call from many threads
    /// at once, also while other threads are still inserting.
    #[inline(always)]
    pub fn union_idx(&self, da: u32, db: u32) {
        if da != db {
            self.union(da, db);
        }
    }

    /// Prefetches the union-find entries of a compact index.
    #[inline(always)]
    pub fn prefetch_idx(&self, d: u32) {
        crate::prefetch(self.parent.as_ptr().wrapping_add(d as usize) as *const u8);
        crate::prefetch(self.dkey.as_ptr().wrapping_add(d as usize) as *const u8);
    }

    /// Number of blocks for `finalize_roots` (compact indices).
    pub fn root_blocks(&self) -> usize {
        (self.next_block.load(Ordering::Acquire) as usize).min(self.parent.len()).div_ceil(FINALIZE_BLOCK)
    }

    /// Number of blocks for `finalize_vals` (table slots).
    pub fn val_blocks(&self) -> usize {
        self.slots.len().div_ceil(FINALIZE_BLOCK)
    }

    /// Finalize, step 1 (after all links): person key (key of the root) of
    /// every compact index in block `b`. Blocks may run on any threads,
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
        // AVX2 builds: found / not found selected, not branched on (random
        // per line, right behind the table's cache miss); SS_PROBESEL=0 off
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        if !self.slots.is_empty() && self.amask == !3 && PROBE_SEL.load(Ordering::Relaxed) && !SCALAR_PROBE.load(Ordering::Relaxed) {
            let (found, v) = unsafe { crate::k5::probe_sel(self.slots.as_ptr() as *const u64, self.mask, k, h) };
            let r = if self.mixed { v } else { keys::mix(v) };
            return sel64(found, r, h);
        }
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
        // AVX2 builds (x86-64-v3): the 4-slot group in two ymm registers
        // (k5::probe_view: key lanes against the key and against 0; same
        // rule as below: the first matching key of the first group holding
        // a match or an empty slot); SS_SCALARPROBE=1: the scalar probe
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        if !SCALAR_PROBE.load(Ordering::Relaxed) {
            return unsafe { crate::k5::probe_view(self.slots.as_ptr() as *const u64, self.mask, k, h) };
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

    /// Read-only view for a kernel's own group probe (k5.rs): (slot array
    /// base as [key, val] u64 pairs, mask, amask, mixed); mask 0 when empty.
    /// Valid after finalize; same probe rule as `probe`.
    #[inline(always)]
    pub fn view(&self) -> (*const u64, usize, usize, bool) {
        (self.slots.as_ptr() as *const u64, if self.slots.is_empty() { 0 } else { self.mask }, self.amask, self.mixed)
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

    /// Every component's root (its person) is its byte-wise smallest ID,
    /// with root order by 8-byte prefixes and without: short, long (hex),
    /// prefix-sharing, zero-byte, upper-case and odd IDs, random links.
    #[test]
    fn union_roots_are_min() {
        let _g = crate::keys::LONG_TEST.lock().unwrap_or_else(|e| e.into_inner());
        let mut x: u64 = 0x1357_9BDF_2468_ACE1;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut ids: Vec<Vec<u8>> = Vec::new();
        for i in 0..300u64 {
            ids.push(format!("u{:x}", keys::mix(i) % 100_000).into_bytes());
            ids.push(format!("u{:016x}{:016x}", keys::mix(i + 7), keys::mix(i + 9)).into_bytes());
        }
        for s in [&b"abcdefgh"[..], b"abcdefgh1", b"abcdefg ", b"abcdefgh x", b"abcdefg", b"abcdefgi", b"ABCDEFGH", b" ", b"  ", b"u", b"u0", b"u0123456789abcdef0123456789abcdef", b"u0123456789abcdef0123456789abcdee", b"u01234567"] {
            ids.push(s.to_vec());
        }
        ids.sort();
        ids.dedup();
        let keys_: Vec<u64> = ids.iter().map(|s| keys::key_of(s)).collect();
        keys::verify_longs(1);
        assert!(!keys::collision());
        for pref_on in [true, false] {
            LINK_PREF.store(pref_on, Ordering::Relaxed);
            let n = ids.len();
            let t = LinkTable::with_pairs(n, 1);
            let mut a = IdxAlloc::new();
            // reference: union-find over indices, root = smallest string
            let mut par: Vec<usize> = (0..n).collect();
            fn root(p: &mut Vec<usize>, mut i: usize) -> usize {
                while p[i] != i {
                    i = p[i];
                }
                i
            }
            let mut idx = Vec::new();
            for _ in 0..n {
                let (i, j) = ((next() % n as u64) as usize, (next() % n as u64) as usize);
                if i == j {
                    continue;
                }
                idx.push(t.insert_pair(keys_[i], keys_[j], pref8(&ids[i]), pref8(&ids[j]), &mut a));
                let (ri, rj) = (root(&mut par, i), root(&mut par, j));
                if ri != rj {
                    // ids are sorted: the smaller index is the smaller string
                    let (lo, hi) = if ri < rj { (ri, rj) } else { (rj, ri) };
                    par[hi] = lo;
                }
            }
            for &(da, db) in &idx {
                t.union_idx(da, db);
            }
            for b in 0..t.root_blocks() {
                t.finalize_roots(b);
            }
            for b in 0..t.val_blocks() {
                t.finalize_vals(b);
            }
            for i in 0..n {
                let r = root(&mut par, i);
                let k = keys_[i];
                let want = if r == i && t.person_of_h(k, keys::mix(k)).is_none() { k } else { keys_[r] };
                assert_eq!(t.person_of(k), want, "{:?} (prefixes {})", String::from_utf8_lossy(&ids[i]), pref_on);
            }
        }
        LINK_PREF.store(true, Ordering::Relaxed);
        keys::reset_longs(0);
    }

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
            let mut a = IdxAlloc::new();
            let mut pairs = Vec::new();
            let mut inserted = Vec::new();
            for _ in 0..actual {
                let (nu, nv) = (next() % space + 1, next() % space + 1);
                let (u, v) = (key(nu), key(nv));
                let (du, dv) = t.insert_pair(u, v, pref8(format!("u{:x}", nu).as_bytes()), pref8(format!("u{:x}", nv).as_bytes()), &mut a);
                pairs.push((du, dv));
                inserted.push(u);
                inserted.push(v);
            }
            for &(du, dv) in &pairs {
                t.union_idx(du, dv);
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
