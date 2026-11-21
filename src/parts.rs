//! Per-thread partitioned output buffers: one arena (huge-page friendly) holding
//! a fixed-capacity slice per partition, plus a heap overflow per partition for
//! skewed partitions (e.g. one holding a whale person).

pub struct Parts<T: Copy> {
    arena: *mut T,
    cap: usize,
    lens: Vec<u32>,
    over: Vec<Vec<T>>,
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
        Parts { arena, cap, lens: vec![0; nparts], over: (0..nparts).map(|_| Vec::new()).collect() }
    }

    #[inline(always)]
    pub fn push(&mut self, p: usize, v: T) {
        let l = self.lens[p] as usize;
        if l < self.cap {
            unsafe { self.arena.add(p * self.cap + l).write(v) };
            self.lens[p] = l as u32 + 1;
        } else {
            self.over[p].push(v);
        }
    }

    #[inline(always)]
    pub fn len(&self, p: usize) -> usize {
        self.lens[p] as usize + self.over[p].len()
    }

    /// The records of partition p, as two slices (arena part, overflow part).
    #[inline(always)]
    pub fn get(&self, p: usize) -> [&[T]; 2] {
        let a = unsafe { std::slice::from_raw_parts(self.arena.add(p * self.cap), self.lens[p] as usize) };
        [a, &self.over[p]]
    }

    pub fn total(&self) -> usize {
        (0..self.lens.len()).map(|p| self.len(p)).sum()
    }
}
