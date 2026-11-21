//! Identity resolution: a concurrent open-addressing table of linked IDs with a
//! lock-free union-find. Roots are always hooked under the root with the
//! smaller key, so keys strictly decrease along parent pointers and every
//! component's root is its byte-wise smallest ID: the person's name.

use crate::keys;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

pub struct LinkTable {
    /// [key, val] per slot. During union-find val = parent slot + 1 (0 = root);
    /// after `finalize` val = the person key (root's key). Lives in a zeroed,
    /// huge-page-backed arena (random probes, so TLB reach matters).
    slots: &'static [[AtomicU64; 2]],
    /// Scratch for `finalize_roots` / `finalize_vals`.
    root_of: &'static [AtomicU32],
    mask: usize,
}

pub const FINALIZE_BLOCK: usize = 1 << 16;

fn zeroed_slots(n: usize) -> &'static [[AtomicU64; 2]] {
    let p = crate::sys::alloc_arena(n * 16) as *const [AtomicU64; 2];
    unsafe { std::slice::from_raw_parts(p, n) }
}

impl LinkTable {
    pub fn empty() -> Self {
        LinkTable { slots: &[], root_of: &[], mask: 0 }
    }

    pub fn with_pairs(total_pairs: usize) -> Self {
        if total_pairs == 0 {
            return Self::empty();
        }
        let cap = (total_pairs * 5 / 2 + 16).next_power_of_two();
        let root_of = unsafe { std::slice::from_raw_parts(crate::sys::alloc_arena(cap * 4) as *const AtomicU32, cap) };
        LinkTable { slots: zeroed_slots(cap), root_of, mask: cap - 1 }
    }

    #[inline(always)]
    fn key(&self, i: usize) -> u64 {
        self.slots[i][0].load(Ordering::Relaxed)
    }

    #[inline(always)]
    fn insert(&self, k: u64) -> usize {
        let mut i = keys::mix(k) as usize & self.mask;
        loop {
            let cur = self.slots[i][0].load(Ordering::Acquire);
            if cur == k {
                return i;
            }
            if cur == 0 {
                match self.slots[i][0].compare_exchange(0, k, Ordering::AcqRel, Ordering::Acquire) {
                    Ok(_) => return i,
                    Err(actual) if actual == k => return i,
                    Err(_) => {}
                }
            }
            i = (i + 1) & self.mask;
        }
    }

    #[inline(always)]
    fn find(&self, mut x: usize) -> usize {
        loop {
            let p = self.slots[x][1].load(Ordering::Acquire);
            if p == 0 {
                return x;
            }
            let p = (p - 1) as usize;
            let gp = self.slots[p][1].load(Ordering::Acquire);
            if gp == 0 {
                return p;
            }
            // path halving; x is not a root so it is never hooked again, and gp
            // is still an ancestor of x.
            self.slots[x][1].store(gp, Ordering::Release);
            x = (gp - 1) as usize;
        }
    }

    #[inline(always)]
    fn union(&self, a: usize, b: usize) {
        let (mut a, mut b) = (a, b);
        loop {
            a = self.find(a);
            b = self.find(b);
            if a == b {
                return;
            }
            if keys::cmp(self.key(a), self.key(b)) == std::cmp::Ordering::Greater {
                std::mem::swap(&mut a, &mut b);
            }
            // hook b (larger key) under a (smaller key)
            if self.slots[b][1]
                .compare_exchange(0, a as u64 + 1, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return;
            }
        }
    }

    /// Joins two IDs. Safe to call from many threads at once.
    #[inline(always)]
    pub fn link(&self, a: u64, b: u64) {
        let ia = self.insert(a);
        let ib = self.insert(b);
        if ia != ib {
            self.union(ia, ib);
        }
    }

    /// Number of finalize blocks.
    pub fn finalize_blocks(&self) -> usize {
        self.slots.len().div_ceil(FINALIZE_BLOCK)
    }

    /// Finalize, step 1 (after all links): root slot of every occupied slot in
    /// block `b`. Blocks may run on any threads, concurrently.
    pub fn finalize_roots(&self, b: usize) {
        let n = self.slots.len();
        for i in b * FINALIZE_BLOCK..((b + 1) * FINALIZE_BLOCK).min(n) {
            if self.key(i) != 0 {
                self.root_of[i].store(self.find(i) as u32, Ordering::Relaxed);
            }
        }
    }

    /// Finalize, step 2 (after every block finished step 1): each occupied
    /// slot's value becomes its person key.
    pub fn finalize_vals(&self, b: usize) {
        let n = self.slots.len();
        for i in b * FINALIZE_BLOCK..((b + 1) * FINALIZE_BLOCK).min(n) {
            if self.key(i) != 0 {
                let rk = self.key(self.root_of[i].load(Ordering::Relaxed) as usize);
                self.slots[i][1].store(rk, Ordering::Relaxed);
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
