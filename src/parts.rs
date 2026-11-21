//! Per-thread partitioned output buffers: one arena (huge-page friendly) holding
//! a fixed-capacity slice per partition, plus a heap overflow per partition for
//! skewed partitions (e.g. one holding a whale person).
//!
//! Optional write combining (records of 8, 16 or 32 bytes): each partition stages its
//! current 64-byte line in a small buffer and writes full lines with
//! non-temporal stores, so scattering to hundreds of partitions does not pay
//! a read-for-ownership miss per output line. `finish` must be called before
//! the records are read.

use std::cell::UnsafeCell;

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct Line([u64; 8]);

pub struct Parts<T: Copy> {
    arena: *mut T,
    cap: usize,
    lens: Vec<u32>,
    over: Vec<UnsafeCell<Vec<T>>>,
    /// write-combining staging lines (empty when off)
    wc: Vec<Line>,
}

unsafe impl<T: Copy + Send> Send for Parts<T> {}
unsafe impl<T: Copy + Sync> Sync for Parts<T> {}

impl<T: Copy> Parts<T> {
    /// records per 64-byte line (write combining)
    const WC_RECS: usize = 64 / std::mem::size_of::<T>();

    pub fn new(nparts: usize, cap: usize) -> Self {
        Self::with_wc(nparts, cap, false)
    }

    /// `wc` enables write combining (ignored unless T is 8, 16 or 32 bytes).
    pub fn with_wc(nparts: usize, cap: usize, wc: bool) -> Self {
        // SS_CAPDIV (testing only) shrinks capacities to exercise the overflow path.
        let div = std::env::var("SS_CAPDIV").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(1).max(1);
        let mut cap = (cap / div).clamp(1, u32::MAX as usize / 2);
        let sz = std::mem::size_of::<T>();
        let wc = wc && (sz == 8 || sz == 16 || sz == 32) && std::env::var("SS_NOWC").is_err();
        if wc {
            // whole lines per partition, so every partition starts 64-byte aligned
            cap = cap.div_ceil(Self::WC_RECS) * Self::WC_RECS;
        }
        let bytes = nparts * cap * std::mem::size_of::<T>();
        let arena = crate::sys::alloc_arena(bytes) as *mut T;
        Parts {
            arena,
            cap,
            lens: vec![0; nparts],
            over: (0..nparts).map(|_| UnsafeCell::new(Vec::new())).collect(),
            wc: if wc { vec![Line([0; 8]); nparts] } else { Vec::new() },
        }
    }

    #[inline(always)]
    pub fn push(&mut self, p: usize, v: T) {
        let l = self.lens[p] as usize;
        if !self.wc.is_empty() {
            let k = l % Self::WC_RECS;
            unsafe { (self.wc[p].0.as_mut_ptr() as *mut T).add(k).write(v) };
            self.lens[p] = l as u32 + 1;
            if k == Self::WC_RECS - 1 {
                self.flush_line(p, l + 1 - Self::WC_RECS);
            }
            return;
        }
        if l < self.cap {
            unsafe { self.arena.add(p * self.cap + l).write(v) };
            self.lens[p] = l as u32 + 1;
        } else {
            self.over[p].get_mut().push(v);
        }
    }

    /// Writes the full staged line of partition p, whose first record has
    /// index `first` (a multiple of Self::WC_RECS).
    #[inline(always)]
    fn flush_line(&mut self, p: usize, first: usize) {
        let src = &self.wc[p];
        if first < self.cap {
            // cap is a multiple of Self::WC_RECS, so the whole line fits
            let dst = unsafe { self.arena.add(p * self.cap + first) } as *mut u8;
            stream_line(dst, src);
        } else {
            let recs = unsafe { std::slice::from_raw_parts(src.0.as_ptr() as *const T, Self::WC_RECS) };
            self.over[p].get_mut().extend_from_slice(recs);
        }
    }

    /// Writes out staged records (write combining); call once after the
    /// last push and before reading.
    pub fn finish(&mut self) {
        if self.wc.is_empty() {
            return;
        }
        for p in 0..self.lens.len() {
            let l = self.lens[p] as usize;
            let k = l % Self::WC_RECS;
            let first = l - k;
            for j in 0..k {
                let v = unsafe { (self.wc[p].0.as_ptr() as *const T).add(j).read() };
                let i = first + j;
                if i < self.cap {
                    unsafe { self.arena.add(p * self.cap + i).write(v) };
                } else {
                    self.over[p].get_mut().push(v);
                }
            }
        }
        store_fence();
        self.wc = Vec::new();
    }

    #[inline(always)]
    pub fn len(&self, p: usize) -> usize {
        if !self.wc.is_empty() {
            return self.lens[p] as usize;
        }
        self.lens[p] as usize + unsafe { (*self.over[p].get()).len() }
    }

    /// The records of partition p, as two slices (arena part, overflow part).
    #[inline(always)]
    pub fn get(&self, p: usize) -> [&[T]; 2] {
        debug_assert!(self.wc.is_empty(), "get before finish");
        let a = unsafe { std::slice::from_raw_parts(self.arena.add(p * self.cap), (self.lens[p] as usize).min(self.cap)) };
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

#[inline(always)]
fn stream_line(dst: *mut u8, src: &Line) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::*;
        let s = src.0.as_ptr() as *const __m128i;
        let d = dst as *mut __m128i;
        _mm_stream_si128(d, _mm_load_si128(s));
        _mm_stream_si128(d.add(1), _mm_load_si128(s.add(1)));
        _mm_stream_si128(d.add(2), _mm_load_si128(s.add(2)));
        _mm_stream_si128(d.add(3), _mm_load_si128(s.add(3)));
    }
    #[cfg(not(target_arch = "x86_64"))]
    unsafe {
        std::ptr::copy_nonoverlapping(src.0.as_ptr() as *const u8, dst, 64);
    }
}

#[inline(always)]
fn store_fence() {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_sfence()
    }
    #[cfg(not(target_arch = "x86_64"))]
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
}
