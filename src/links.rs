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
    /// dense index -> parent dense index + 1 (0 = root)
    parent: &'static [AtomicU32],
    /// dense index -> key
    dkey: &'static [AtomicU64],
    /// dense index -> root dense index (finalize scratch)
    droot: &'static [AtomicU32],
    /// next free block of dense indices
    next_block: AtomicU32,
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
        LinkTable { slots: &[], mask: 0, parent: &[], dkey: &[], droot: &[], next_block: AtomicU32::new(0) }
    }

    /// A table for `total_pairs` links built by up to `threads` threads.
    pub fn with_pairs(total_pairs: usize, threads: usize) -> Self {
        if total_pairs == 0 {
            return Self::empty();
        }
        let cap = (total_pairs * 5 / 2 + 16).next_power_of_two();
        // at most 2 new IDs per pair, plus partly used blocks
        let nd = 2 * total_pairs + (threads + 1) * DENSE_BLOCK as usize + 64;
        assert!(nd < u32::MAX as usize - DENSE_BLOCK as usize * 2, "too many links");
        LinkTable {
            slots: arena(cap),
            mask: cap - 1,
            parent: arena(nd),
            dkey: arena(nd),
            droot: arena(nd),
            next_block: AtomicU32::new(0),
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

    /// Dense index of key k, inserting it if absent.
    #[inline(always)]
    fn insert(&self, k: u64, a: &mut DenseAlloc) -> u32 {
        let mut i = keys::mix(k) as usize & self.mask;
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
    pub fn link(&self, a: u64, b: u64, alloc: &mut DenseAlloc) {
        let da = self.insert(a, alloc);
        let db = self.insert(b, alloc);
        if da != db {
            self.union(da, db);
        }
    }

    /// Number of blocks for `finalize_roots` (dense indices).
    pub fn root_blocks(&self) -> usize {
        (self.next_block.load(Ordering::Acquire) as usize).min(self.parent.len()).div_ceil(FINALIZE_BLOCK)
    }

    /// Number of blocks for `finalize_vals` (table slots).
    pub fn val_blocks(&self) -> usize {
        self.slots.len().div_ceil(FINALIZE_BLOCK)
    }

    /// Finalize, step 1 (after all links): root of every dense index in
    /// block `b`. Blocks may run on any threads, concurrently.
    pub fn finalize_roots(&self, b: usize) {
        let n = (self.next_block.load(Ordering::Acquire) as usize).min(self.parent.len());
        for d in b * FINALIZE_BLOCK..((b + 1) * FINALIZE_BLOCK).min(n) {
            self.droot[d].store(self.find(d as u32), Ordering::Relaxed);
        }
    }

    /// Finalize, step 2 (after every block finished step 1): each occupied
    /// slot's value becomes its person key.
    pub fn finalize_vals(&self, b: usize) {
        let n = self.slots.len();
        for i in b * FINALIZE_BLOCK..((b + 1) * FINALIZE_BLOCK).min(n) {
            if self.slots[i][0].load(Ordering::Relaxed) != 0 {
                let d = (self.slots[i][1].load(Ordering::Relaxed) - 1) as usize;
                let r = self.droot[d].load(Ordering::Relaxed) as usize;
                let pk = self.dkey[r].load(Ordering::Relaxed);
                self.slots[i][1].store(pk, Ordering::Relaxed);
            }
        }
    }

    /// The person key of user key `k` (after `finalize`).
    #[inline(always)]
    pub fn person_of(&self, k: u64) -> u64 {
        if self.slots.is_empty() {
            return k;
        }
        let mut i = keys::mix(k) as usize & self.mask;
        loop {
            let s = &self.slots[i];
            let kk = s[0].load(Ordering::Relaxed);
            if kk == k {
                return s[1].load(Ordering::Relaxed);
            }
            if kk == 0 {
                return k;
            }
            i = (i + 1) & self.mask;
        }
    }

    #[inline(always)]
    pub fn prefetch(&self, k: u64) {
        if self.slots.is_empty() {
            return;
        }
        let i = keys::mix(k) as usize & self.mask;
        crate::prefetch(self.slots.as_ptr().wrapping_add(i) as *const u8);
    }
}
