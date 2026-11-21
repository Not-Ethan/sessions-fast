//! Minimal OS bindings (no external crates): read-only file mappings.

use std::ffi::c_void;
use std::fs::File;
use std::io;
use std::os::fd::AsRawFd;
use std::path::Path;

extern "C" {
    fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> i32;
}

const PROT_READ: i32 = 1;
const MAP_PRIVATE: i32 = 2;

pub struct Mmap {
    ptr: *const u8,
    len: usize,
    file: Option<File>,
}

unsafe impl Send for Mmap {}
unsafe impl Sync for Mmap {}

impl Mmap {
    pub fn open(path: &Path) -> io::Result<Mmap> {
        let file = File::open(path)?;
        let len = file.metadata()?.len() as usize;
        if len == 0 {
            return Ok(Mmap { ptr: std::ptr::NonNull::<u8>::dangling().as_ptr(), len: 0, file: None });
        }
        let p = unsafe { mmap(std::ptr::null_mut(), len, PROT_READ, MAP_PRIVATE, file.as_raw_fd(), 0) };
        if p as isize == -1 {
            return Err(io::Error::last_os_error());
        }
        Ok(Mmap { ptr: p as *const u8, len, file: Some(file) })
    }

    /// Reads `buf.len()` bytes at offset `off` of the file with pread.
    pub fn read_at(&self, buf: &mut [u8], off: usize) {
        use std::os::unix::fs::FileExt;
        if buf.is_empty() {
            return;
        }
        if let Err(e) = self.file.as_ref().unwrap().read_exact_at(buf, off as u64) {
            eprintln!("sessions: read failed: {}", e);
            std::process::exit(3);
        }
    }

    #[inline(always)]
    pub fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        if self.len > 0 {
            unsafe {
                munmap(self.ptr as *mut c_void, self.len);
            }
        }
    }
}

#[repr(C)]
struct Timespec {
    tv_sec: i64,
    tv_nsec: i64,
}

extern "C" {
    fn clock_gettime(clk: i32, ts: *mut Timespec) -> i32;
    fn getrusage(who: i32, usage: *mut [i64; 18]) -> i32;
}

#[cfg(target_os = "macos")]
const CLOCK_THREAD_CPUTIME_ID: i32 = 16;
#[cfg(not(target_os = "macos"))]
const CLOCK_THREAD_CPUTIME_ID: i32 = 3;

/// CPU time consumed by the calling thread, in ns.
pub fn thread_cpu_ns() -> u64 {
    let mut ts = Timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe {
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &mut ts);
    }
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

/// Wall clock (CLOCK_REALTIME) in ns since the Unix epoch.
pub fn unix_ns() -> u64 {
    let mut ts = Timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe {
        clock_gettime(0, &mut ts);
    }
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

/// Process resource usage: (user ms, sys ms, minor faults, major faults,
/// voluntary context switches, involuntary context switches).
pub fn rusage() -> (f64, f64, i64, i64, i64, i64) {
    let mut ru = [0i64; 18];
    unsafe {
        getrusage(0, &mut ru);
    }
    let ms = |s: i64, us: i64| s as f64 * 1e3 + us as f64 / 1e3;
    (ms(ru[0], ru[1]), ms(ru[2], ru[3]), ru[8], ru[9], ru[16], ru[17])
}

/// Peak resident set size of the process, in MB.
pub fn peak_rss_mb() -> u64 {
    let mut ru = [0i64; 18];
    unsafe {
        getrusage(0, &mut ru);
    }
    let v = ru[4] as u64; // ru_maxrss: bytes on macOS, KiB on Linux
    if cfg!(target_os = "macos") {
        v >> 20
    } else {
        v >> 10
    }
}

extern "C" {
    fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
}

#[cfg(target_os = "linux")]
const MAP_ANON_FLAGS: i32 = 0x20 | 0x4000; // MAP_ANONYMOUS | MAP_NORESERVE
#[cfg(not(target_os = "linux"))]
const MAP_ANON_FLAGS: i32 = 0x1000 | 0x40; // MAP_ANON | MAP_NORESERVE (macOS)
const PROT_RW: i32 = 1 | 2;

/// Zeroed anonymous memory reserved lazily (only touched pages are resident),
/// backed by transparent huge pages on Linux when the kernel allows it.
/// Never unmapped: it lives until process exit.
pub fn alloc_arena(bytes: usize) -> *mut u8 {
    let bytes = bytes.max(4096);
    let p = unsafe { mmap(std::ptr::null_mut(), bytes, PROT_RW, MAP_PRIVATE | MAP_ANON_FLAGS, -1, 0) };
    if p as isize == -1 {
        eprintln!("sessions: mmap of {} bytes failed: {}", bytes, io::Error::last_os_error());
        std::process::exit(3);
    }
    #[cfg(target_os = "linux")]
    if !NOHUGE.load(std::sync::atomic::Ordering::Relaxed) {
        unsafe {
            madvise(p, bytes, 14); // MADV_HUGEPAGE
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = madvise;
    ARENAS.lock().unwrap().push((p as usize, bytes));
    p as *mut u8
}

/// Experiment switch (SS_NOHUGE): no MADV_HUGEPAGE on arenas.
pub static NOHUGE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static ARENAS: std::sync::Mutex<Vec<(usize, usize)>> = std::sync::Mutex::new(Vec::new());

/// Unmaps every arena (diagnostic mode, between runs: nothing may use them
/// afterwards).
pub fn release_all() {
    let v = std::mem::take(&mut *ARENAS.lock().unwrap());
    for (a, l) in v {
        unsafe {
            munmap(a as *mut c_void, l);
        }
    }
}

/// Every arena as (address, length).
pub fn arenas() -> Vec<(usize, usize)> {
    ARENAS.lock().unwrap().clone()
}

/// Drops the mappings of the whole pages inside [from, to) of a read-only
/// private file mapping (MADV_DONTNEED; a later access would simply map the
/// file pages again).
pub fn release_inner(from: usize, to: usize) {
    let a = (from + 4095) & !4095;
    let b = to & !4095;
    if b > a {
        unsafe {
            madvise(a as *mut c_void, b - a, 4);
        }
    }
}

/// Releases the pages of [addr, addr + len) (MADV_DONTNEED; the start is
/// rounded down to a page boundary).
pub fn release(addr: usize, len: usize) {
    let a = addr & !4095;
    unsafe {
        madvise(a as *mut c_void, len + (addr - a), 4);
    }
}
