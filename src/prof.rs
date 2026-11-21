//! Verbose-only sub-phase timers (SS_VERBOSE): each thread accumulates cycle
//! counts per sub-phase in an `Acc` and adds them to global sums when done;
//! the report converts them to summed thread-milliseconds. With timing off
//! (`ON` false) a lap is one predictable branch.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

pub static ON: AtomicBool = AtomicBool::new(false);

pub const N: usize = 16;
static SUM: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
static MAX: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
static CNT: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
static CNAMES: std::sync::Mutex<[&'static str; N]> = std::sync::Mutex::new([""; N]);
static NAMES: std::sync::Mutex<[&'static str; N]> = std::sync::Mutex::new([""; N]);
static CAL: std::sync::Mutex<Option<(Instant, u64)>> = std::sync::Mutex::new(None);

#[inline(always)]
pub fn tick() -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_rdtsc()
    }
    #[cfg(target_arch = "aarch64")]
    {
        let v: u64;
        unsafe { std::arch::asm!("mrs {}, cntvct_el0", out(reg) v, options(nomem, nostack)) };
        v
    }
}

/// Starts the tick calibration (call once, early).
pub fn calibrate_start() {
    *CAL.lock().unwrap() = Some((Instant::now(), tick()));
}

fn ns_per_tick() -> f64 {
    let g = CAL.lock().unwrap();
    let Some((i0, t0)) = *g else { return 0.0 };
    let dt = tick().wrapping_sub(t0) as f64;
    if dt <= 0.0 {
        return 0.0;
    }
    i0.elapsed().as_nanos() as f64 / dt
}

/// Names sub-phase slot `i` (for the report).
pub fn name(i: usize, n: &'static str) {
    NAMES.lock().unwrap()[i] = n;
}

/// Names counter slot `i` (for the report).
pub fn cname(i: usize, n: &'static str) {
    CNAMES.lock().unwrap()[i] = n;
}

pub struct Acc {
    on: bool,
    t: u64,
    v: [u64; N],
    c: [u64; N],
}

impl Acc {
    pub fn new() -> Acc {
        let on = ON.load(Ordering::Relaxed);
        Acc { on, t: if on { tick() } else { 0 }, v: [0; N], c: [0; N] }
    }
    #[inline(always)]
    pub fn on(&self) -> bool {
        self.on
    }
    /// Restarts the lap clock.
    #[inline(always)]
    pub fn start(&mut self) {
        if self.on {
            self.t = tick();
        }
    }
    /// Charges the time since the last lap to slot `i`.
    #[inline(always)]
    pub fn lap(&mut self, i: usize) {
        if self.on {
            let now = tick();
            self.v[i] += now.wrapping_sub(self.t);
            self.t = now;
        }
    }
    /// Adds `v` to counter `i`.
    #[inline(always)]
    pub fn count(&mut self, i: usize, v: u64) {
        if self.on {
            self.c[i] += v;
        }
    }
    /// Adds this thread's sums to the global ones.
    pub fn commit(&mut self) {
        if self.on {
            for i in 0..N {
                SUM[i].fetch_add(self.v[i], Ordering::Relaxed);
                MAX[i].fetch_max(self.v[i], Ordering::Relaxed);
                CNT[i].fetch_add(self.c[i], Ordering::Relaxed);
            }
            self.v = [0; N];
            self.c = [0; N];
        }
    }
}

/// "name=thread-ms(max-thread-ms) ..." for every named slot with time, and
/// resets the sums.
pub fn report() -> String {
    let k = ns_per_tick() / 1e6;
    let names = NAMES.lock().unwrap();
    let mut s = String::new();
    for i in 0..N {
        let v = SUM[i].swap(0, Ordering::Relaxed);
        let m = MAX[i].swap(0, Ordering::Relaxed);
        if !names[i].is_empty() && v > 0 {
            s.push_str(&format!(" {}={:.1}({:.1})", names[i], v as f64 * k, m as f64 * k));
        }
    }
    let cn = CNAMES.lock().unwrap();
    for i in 0..N {
        let v = CNT[i].swap(0, Ordering::Relaxed);
        if !cn[i].is_empty() {
            s.push_str(&format!(" #{}={}", cn[i], v));
        }
    }
    s
}
