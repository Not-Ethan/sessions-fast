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

/// "anon=..MB thp=..MB" from /proc/self/smaps_rollup (Linux; empty elsewhere).
pub fn anon_summary() -> String {
    let Ok(t) = std::fs::read_to_string("/proc/self/smaps_rollup") else { return String::new() };
    let kb = |key: &str| -> u64 {
        t.lines()
            .find(|l| l.starts_with(key))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    };
    format!("anon={}MB thp={}MB file_rss={}MB", kb("Anonymous:") >> 10, kb("AnonHugePages:") >> 10, (kb("Rss:").saturating_sub(kb("Anonymous:"))) >> 10)
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
    LABELS.lock().unwrap().push((p as usize, bytes, LABEL.with(|l| l.get())));
    p as *mut u8
}

// ---------------------------------------------------------------- recycled memory
//
// Arenas of a finished phase are handed to later consumers instead of being
// released and faulted in (zeroed) again. Recycled memory holds stale data:
// only consumers that write everything before they read it may take it
// (`alloc_arena_dirty`); SS_POISON=1 fills it with 0xA5 when handed out, to
// prove that. Recycled arenas leave the arena list, so the parallel release
// of dead memory never drops pages in use.

static POOL: std::sync::Mutex<Vec<(usize, usize)>> = std::sync::Mutex::new(Vec::new());
pub static POISON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Recycling on (SS_RECYCLE=0: off).
pub static RECYCLE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// The arenas labelled with any of `labels` go to the pool (their data is dead).
pub fn give_back_labels(labels: &[&str]) {
    if !RECYCLE.load(std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let mut lab = LABELS.lock().unwrap();
    let mut ar = ARENAS.lock().unwrap();
    let mut pool = POOL.lock().unwrap();
    lab.retain(|&(a, l, name)| {
        if labels.contains(&name) {
            ar.retain(|&(x, _)| x != a);
            pool.push((a, l));
            false
        } else {
            true
        }
    });
    // largest first: big consumers find a region
    pool.sort_unstable_by(|x, y| y.1.cmp(&x.1));
}

/// Memory for a consumer that writes every byte before reading it: from the
/// pool if a region is large enough (stale contents), else a fresh arena.
pub fn alloc_arena_dirty(bytes: usize) -> *mut u8 {
    let need = bytes.max(4096).div_ceil(4096) * 4096;
    if RECYCLE.load(std::sync::atomic::Ordering::Relaxed) {
        let mut pool = POOL.lock().unwrap();
        if let Some(i) = pool.iter().position(|&(_, l)| l >= need) {
            let (a, l) = pool[i];
            if l - need >= (2 << 20) {
                pool[i] = (a + need, l - need);
            } else {
                pool.remove(i);
            }
            drop(pool);
            if POISON.load(std::sync::atomic::Ordering::Relaxed) {
                unsafe { std::ptr::write_bytes(a as *mut u8, 0xA5, need) };
            }
            return a as *mut u8;
        }
    }
    alloc_arena(bytes)
}

/// Experiment (SS_POPW=1): parse arenas are populated for writing in one
/// call by the thread that owns them (MADV_POPULATE_WRITE, Linux 5.14+)
/// instead of one fault per page in the hot loop.
pub static POPW: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Populates [addr, addr + len) for writing (errors ignored).
pub fn populate_write(addr: usize, len: usize) {
    #[cfg(target_os = "linux")]
    unsafe {
        madvise(addr as *mut c_void, len, 23);
    }
    #[cfg(not(target_os = "linux"))]
    let _ = (addr, len);
}

thread_local! {
    static LABEL: std::cell::Cell<&'static str> = const { std::cell::Cell::new("other") };
}
/// Every arena with the label that was current when it was allocated
/// (memory accounting, SS_MEMACCT).
static LABELS: std::sync::Mutex<Vec<(usize, usize, &'static str)>> = std::sync::Mutex::new(Vec::new());

/// Label for the arenas this thread allocates from now on (accounting only).
pub fn set_label(l: &'static str) {
    LABEL.with(|c| c.set(l));
}

#[cfg(target_os = "linux")]
extern "C" {
    fn mincore(addr: *mut c_void, len: usize, vec: *mut u8) -> i32;
}

/// SS_MEMACCT: per label, MB reserved, MB resident (touched and not
/// released), MB of it in huge pages (smaps AnonHugePages), plus the
/// process's RSS; one line for `phase`.
pub fn mem_report(phase: &str) -> String {
    #[cfg(target_os = "linux")]
    {
        use std::collections::BTreeMap;
        let labels = LABELS.lock().unwrap().clone();
        // AnonHugePages per mapping start
        let mut huge: Vec<(usize, usize, usize)> = Vec::new();
        if let Ok(t) = std::fs::read_to_string("/proc/self/smaps") {
            let (mut lo, mut hi) = (0usize, 0usize);
            for line in t.lines() {
                if let Some((range, _)) = line.split_once(' ') {
                    if let Some((a, b)) = range.split_once('-') {
                        if let (Ok(a), Ok(b)) = (usize::from_str_radix(a, 16), usize::from_str_radix(b, 16)) {
                            lo = a;
                            hi = b;
                            continue;
                        }
                    }
                }
                if let Some(v) = line.strip_prefix("AnonHugePages:") {
                    let kb: usize = v.trim().trim_end_matches("kB").trim().parse().unwrap_or(0);
                    if kb > 0 {
                        huge.push((lo, hi, kb * 1024));
                    }
                }
            }
        }
        let mut agg: BTreeMap<&str, (usize, usize, usize)> = BTreeMap::new();
        for &(a, l, lab) in &labels {
            let pages = l.div_ceil(4096);
            let mut v = vec![0u8; pages];
            let res = unsafe { mincore(a as *mut c_void, pages * 4096, v.as_mut_ptr()) };
            let resident = if res == 0 { v.iter().filter(|&&b| b & 1 != 0).count() * 4096 } else { 0 };
            // huge pages of mappings inside this arena
            let h: usize = huge.iter().filter(|&&(lo, hi, _)| lo >= a && hi <= a + l.div_ceil(4096) * 4096).map(|x| x.2).sum();
            let e = agg.entry(lab).or_insert((0, 0, 0));
            e.0 += l;
            e.1 += resident;
            e.2 += h;
        }
        let mb = |x: usize| x as f64 / (1 << 20) as f64;
        let mut out = format!("memacct {}:", phase);
        for (lab, (r, t, h)) in agg {
            out.push_str(&format!(" {}={:.0}/{:.0}/{:.0}", lab, mb(r), mb(t), mb(h)));
        }
        // total huge pages (VMAs of neighbouring arenas merge, so the per-label
        // column only counts arenas that are a VMA of their own)
        if let Ok(r) = std::fs::read_to_string("/proc/self/smaps_rollup") {
            if let Some(l) = r.lines().find(|l| l.starts_with("AnonHugePages:")) {
                let kb: usize = l["AnonHugePages:".len()..].trim().trim_end_matches("kB").trim().parse().unwrap_or(0);
                out.push_str(&format!(" AnonHuge:{:.0}", kb as f64 / 1024.0));
            }
        }
        let st = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        for k in ["VmRSS:", "RssAnon:", "RssFile:", "VmHWM:"] {
            if let Some(l) = st.lines().find(|l| l.starts_with(k)) {
                let kb: usize = l[k.len()..].trim().trim_end_matches("kB").trim().parse().unwrap_or(0);
                out.push_str(&format!(" {}{:.0}", k, kb as f64 / 1024.0));
            }
        }
        out
    }
    #[cfg(not(target_os = "linux"))]
    {
        format!("memacct {}: (linux only)", phase)
    }
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

/// Unmaps the arenas allocated after the first `keep` (benchmarks).
pub fn release_from(keep: usize) {
    let mut g = ARENAS.lock().unwrap();
    if g.len() > keep {
        let tail: Vec<(usize, usize)> = g.drain(keep..).collect();
        drop(g);
        for (a, l) in tail {
            unsafe {
                munmap(a as *mut c_void, l);
            }
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

/// Maps the pages of [addr, addr + len) of a file mapping for reading in one
/// call (MADV_POPULATE_READ, Linux 5.14+; errors ignored: then the pages are
/// simply faulted in on access).
pub fn populate_read(addr: usize, len: usize) {
    #[cfg(target_os = "linux")]
    {
        let a = addr & !4095;
        unsafe {
            madvise(a as *mut c_void, len + (addr - a), 22);
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = (addr, len);
}

/// Releases the pages of [addr, addr + len) (MADV_DONTNEED; the start is
/// rounded down to a page boundary).
pub fn release(addr: usize, len: usize) {
    let a = addr & !4095;
    unsafe {
        madvise(a as *mut c_void, len + (addr - a), 4);
    }
}


// ---------------------------------------------------------------- CPU pinning (benchmarks)

#[cfg(target_os = "linux")]
extern "C" {
    fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const u64) -> i32;
}

/// Pins the calling thread to one CPU (Linux; false elsewhere or on error).
pub fn pin_to(cpu: usize) -> bool {
    #[cfg(target_os = "linux")]
    {
        let mut mask = [0u64; 16];
        if cpu >= 1024 {
            return false;
        }
        mask[cpu / 64] |= 1u64 << (cpu % 64);
        unsafe { sched_setaffinity(0, std::mem::size_of_val(&mask), mask.as_ptr()) == 0 }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = cpu;
        false
    }
}

/// Hyperthread sibling pairs among CPUs 0..n from sysfs, or None unless
/// every CPU has exactly one sibling in that range.
pub fn sibling_pairs(n: usize) -> Option<Vec<(usize, usize)>> {
    let mut pairs = Vec::new();
    let mut seen = vec![false; n];
    for c in 0..n {
        let txt = std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{}/topology/thread_siblings_list", c)).ok()?;
        let mut cpus = Vec::new();
        for part in txt.trim().split(',') {
            if let Some((a, b)) = part.split_once('-') {
                let (a, b): (usize, usize) = (a.parse().ok()?, b.parse().ok()?);
                cpus.extend(a..=b);
            } else {
                cpus.push(part.parse().ok()?);
            }
        }
        if cpus.len() != 2 || !cpus.contains(&c) || cpus.iter().any(|&x| x >= n) {
            return None;
        }
        let (a, b) = (cpus[0].min(cpus[1]), cpus[0].max(cpus[1]));
        if !seen[a] {
            seen[a] = true;
            seen[b] = true;
            pairs.push((a, b));
        }
    }
    Some(pairs)
}
