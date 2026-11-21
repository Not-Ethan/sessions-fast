//! Per-thread partitioned output buffers: one arena (huge-page friendly) holding
//! a fixed-capacity slice per partition, plus a heap overflow per partition for
//! skewed partitions (e.g. one holding a whale person).

use std::cell::UnsafeCell;

pub struct Parts<T: Copy> {
    arena: *mut T,
    cap: usize,
    lens: Vec<u32>,
    over: Vec<UnsafeCell<Vec<T>>>,
}

unsafe impl<T: Copy + Send> Send for Parts<T> {}
unsafe impl<T: Copy + Sync> Sync for Parts<T> {}

impl<T: Copy> Parts<T> {
    pub fn new(nparts: usize, cap: usize) -> Self {
        // SS_CAPDIV (testing only) shrinks capacities to exercise the overflow path.
        let div = std::env::var("SS_CAPDIV").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(1).max(1);
        let cap = (cap / div).clamp(1, u32::MAX as usize / 2);
        let bytes = nparts * cap * std::mem::size_of::<T>();
        let arena = crate::sys::alloc_arena(bytes) as *mut T;
        Parts { arena, cap, lens: vec![0; nparts], over: (0..nparts).map(|_| UnsafeCell::new(Vec::new())).collect() }
    }

    #[inline(always)]
    pub fn push(&mut self, p: usize, v: T) {
        let l = self.lens[p] as usize;
        if l < self.cap {
            unsafe { self.arena.add(p * self.cap + l).write(v) };
            self.lens[p] = l as u32 + 1;
        } else {
            self.over[p].get_mut().push(v);
        }
    }

    #[inline(always)]
    pub fn len(&self, p: usize) -> usize {
        self.lens[p] as usize + unsafe { (*self.over[p].get()).len() }
    }

    /// The records of partition p, as two slices (arena part, overflow part).
    #[inline(always)]
    pub fn get(&self, p: usize) -> [&[T]; 2] {
        let a = unsafe { std::slice::from_raw_parts(self.arena.add(p * self.cap), self.lens[p] as usize) };
        [a, unsafe { &*self.over[p].get() }]
    }

    /// Address of record `i` of partition `p` (i < len(p)), for in-place
    /// updates while the buffers are shared between threads. The caller must
    /// ensure no other thread accesses the same record concurrently and that
    /// no slice from `get` overlapping it is alive.
    #[inline(always)]
    pub unsafe fn slot(&self, p: usize, i: usize) -> *mut T {
        if i < self.cap {
            self.arena.add(p * self.cap + i)
        } else {
            (*self.over[p].get()).as_mut_ptr().add(i - self.cap)
        }
    }

    /// Empties every partition, keeping the memory (benchmarks).
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.lens.iter_mut().for_each(|l| *l = 0);
        self.over.iter_mut().for_each(|o| o.get_mut().clear());
    }

    pub fn total(&self) -> usize {
        (0..self.lens.len()).map(|p| self.len(p)).sum()
    }
}
