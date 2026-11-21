//! Shopping Sessions: native pipeline. See README.md for the design.

mod keys;
mod links;
mod parts;
mod lookup;
mod parquet;
mod prof;
mod cbench;
mod scan;
mod sys;
mod ts;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use links::LinkTable;
use lookup::{IdTable, NONE};
use parts::Parts;

pub const DAY: i64 = 86_400_000;
pub const WEEK: i64 = 7 * DAY;
const GAP: u32 = 1_800_000;
const LATE: i64 = 86_400_000;

#[inline(always)]
pub fn prefetch(p: *const u8) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(p as *const i8, std::arch::x86_64::_MM_HINT_T0)
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p, options(nostack, readonly, preserves_flags))
    }
}

// ---------------------------------------------------------------- records

/// One parsed event line, routed by event_id for dedup (28 bytes).
#[derive(Clone, Copy, Default)]
#[repr(C, packed(4))]
struct ERec {
    /// keys::mix(event_id key): a bijection, so equality is exact; its top bits
    /// pick the partition and its low bits the hash-table slot.
    eid: u64,
    user: u64,
    /// ingest_ts in seconds, offset so that window_start is 2^31.
    ingest: u32,
    /// ts - window_start in ms, | OK_BIT if this copy passes validation and
    /// is not late (the offset is < 2^31: windows are limited to 24 days).
    ts: u32,
    /// purchase: price | category rank << 24; otherwise 0.
    pay: u32,
}

const OK_BIT: u32 = 1 << 31;
const _: () = assert!(std::mem::size_of::<ERec>() == 28);
const _: () = assert!(std::mem::size_of::<CRec>() == 32);
const _: () = assert!(std::mem::size_of::<SRec>() == 16);

/// A surviving event, routed by person.
#[derive(Clone, Copy)]
#[repr(C)]
struct SRec {
    /// keys::mix(person key) (a bijection): its top bits pick the partition,
    /// its low bits index the per-partition map.
    person: u64,
    ts: u32,
    pay: u32,
}

/// A valid click, routed by person. ts is relative to window_start - 7 days.
#[derive(Clone, Copy)]
#[repr(C)]
struct CRec {
    /// keys::mix(person key), as in SRec
    person: u64,
    ck: u64,
    ts: u32,
    camp: u32,
    /// padding to 32 bytes (write-combining scatter)
    _pad: u64,
}

#[derive(Clone, Copy)]
struct Click {
    ts: u32,
    camp: u32,
    ck: u64,
}

#[derive(Clone, Copy)]
struct URow {
    person: u64,
    day: i32,
    sessions: u32,
    events: u32,
    top: u32,
    revenue: u64,
}

/// Per-thread direct-mapped cache user key -> keys::mix(person key) in front
/// of the link table (activity is Zipf-like, so hot users hit). Entries hold
/// the full user key, so a hit is exact; key 0 (never a user) marks empty.
struct PCache {
    slots: Vec<(u64, u64)>,
    shift: u32,
    on: bool,
}

impl PCache {
    fn new() -> PCache {
        let bits = env_usize("SS_PCACHE").unwrap_or(0).min(20) as u32;
        PCache { slots: vec![(0, 0); 1 << bits.max(1)], shift: 64 - bits.max(1), on: bits > 0 }
    }
    /// The cached person hash of `user` (hash `h`), if present.
    #[inline(always)]
    fn get(&self, user: u64, h: u64) -> Option<u64> {
        if !self.on {
            return None;
        }
        let e = unsafe { self.slots.get_unchecked(((h << 20) >> self.shift) as usize) };
        if e.0 == user {
            Some(e.1)
        } else {
            None
        }
    }
    #[inline(always)]
    fn put(&mut self, user: u64, h: u64, hp: u64) {
        if self.on {
            let i = ((h << 20) >> self.shift) as usize;
            unsafe { *self.slots.get_unchecked_mut(i) = (user, hp) };
        }
    }
}

/// keys::mix of the person key of user key `user`, whose hash is
/// `h` = keys::mix(user) (one table probe; no second hash when the user is
/// its own person).
#[inline(always)]
fn person_hash(lt: &LinkTable, user: u64, h: u64) -> u64 {
    match lt.person_of_h(user, h) {
        Some(p) if p != user => keys::mix(p),
        _ => h,
    }
}

// ---------------------------------------------------------------- context

struct Ctx {
    tsp: ts::TsParser,
    /// experiment (SS_TSW): branch-free timestamp parser on the hot path
    tsw: bool,
    /// experiment (SS_TSH): F1/F3 decoded together, selected by length
    tsh: bool,
    avx2: bool,
    ws: i64,
    we: i64,
    ws_s: i64,
    ws_mod_day: u64,
    first_day: i64,
    ndays: usize,
    products: IdTable,
    unknown_cat: u32,
    cat_names: Vec<Vec<u8>>,
    campaigns: IdTable,
    camp_ids: Vec<Vec<u8>>,
    camp_channels: Vec<Vec<u8>>,
    ncamp: usize,
    pe_bits: u32,
    ps_bits: u32,
    threads: usize,
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok())
}

fn manifest_value(text: &str, key: &str) -> String {
    let at = text.find(&format!("\"{}\"", key)).unwrap_or_else(|| fail(&format!("manifest: no {}", key)));
    let rest = &text[at + key.len() + 2..];
    let colon = rest.find(':').unwrap_or_else(|| fail("manifest: no colon"));
    let rest = &rest[colon + 1..];
    let q1 = rest.find('"').unwrap_or_else(|| fail("manifest: no value"));
    let rest = &rest[q1 + 1..];
    let q2 = rest.find('"').unwrap_or_else(|| fail("manifest: unterminated value"));
    rest[..q2].to_string()
}

fn fail(msg: &str) -> ! {
    eprintln!("sessions: {}", msg);
    std::process::exit(2)
}

fn read_file(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| fail(&format!("{}: {}", path.display(), e)))
}

/// Data rows of a clean 2-column CSV (header skipped), as borrowed slices.
fn table_rows(text: &[u8]) -> Vec<(&[u8], &[u8])> {
    let mut rows = Vec::new();
    if text.is_empty() {
        return rows;
    }
    let h = scan::next_line_start(text, 0);
    scan::for_each_record::<2, _>(text, h, text.len(), |f| rows.push((f[0], f[1])));
    rows
}

// ---------------------------------------------------------------- input files

struct Input {
    maps: Vec<sys::Mmap>,
    tasks: Vec<Task>,
    bytes: usize,
}

#[derive(Clone, Copy)]
struct Task {
    file: u32,
    start: usize,
    end: usize,
}

fn list_csv(dir: &Path) -> Vec<PathBuf> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut v: Vec<PathBuf> = rd
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            name.ends_with(".csv") && !name.starts_with('.') && p.is_file()
        })
        .collect();
    v.sort();
    v
}

fn load_input(dir: &Path, chunk: usize) -> Input {
    let mut maps = Vec::new();
    let mut tasks = Vec::new();
    let mut bytes = 0;
    for path in list_csv(dir) {
        let m = sys::Mmap::open(&path).unwrap_or_else(|e| fail(&format!("{}: {}", path.display(), e)));
        let b = m.bytes();
        let fi = maps.len() as u32;
        if !b.is_empty() {
            let mut s = scan::next_line_start(b, 0); // skip the header line
            while s < b.len() {
                let nominal = s + chunk;
                let e = if nominal >= b.len() { b.len() } else { scan::next_line_start(b, nominal - 1) };
                tasks.push(Task { file: fi, start: s, end: e });
                s = e;
            }
            bytes += b.len();
        }
        maps.push(m);
    }
    Input { maps, tasks, bytes }
}

static PHASE_CPU_NS: AtomicU64 = AtomicU64::new(0);
static PHASE_MAX_NS: AtomicU64 = AtomicU64::new(0);
/// Summed time threads spent finished but waiting for the join (ns).
static PHASE_IDLE_NS: AtomicU64 = AtomicU64::new(0);
/// Latest thread start after the phase began (ns).
static PHASE_SPAWN_NS: AtomicU64 = AtomicU64::new(0);

fn par<R: Send>(threads: usize, f: impl Fn(usize) -> R + Sync) -> Vec<R> {
    let t0 = Instant::now();
    std::thread::scope(|s| {
        let f = &f;
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                s.spawn(move || {
                    PHASE_SPAWN_NS.fetch_max(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    let r = f(t);
                    let ns = sys::thread_cpu_ns();
                    PHASE_CPU_NS.fetch_add(ns, Ordering::Relaxed);
                    PHASE_MAX_NS.fetch_max(ns, Ordering::Relaxed);
                    (r, t0.elapsed())
                })
            })
            .collect();
        let rs: Vec<_> = hs.into_iter().map(|h| h.join().unwrap()).collect();
        let tj = t0.elapsed();
        let idle: u64 = rs.iter().map(|(_, e)| tj.saturating_sub(*e).as_nanos() as u64).sum();
        PHASE_IDLE_NS.fetch_add(idle, Ordering::Relaxed);
        rs.into_iter().map(|(r, _)| r).collect()
    })
}

/// (wall, summed thread CPU, max thread CPU) since the previous mark, in ms.
struct Marks {
    t0: Instant,
    prev: Instant,
    /// Wall clock at main() entry (ns since the epoch), for measuring exec and
    /// teardown latency from outside the process.
    unix0: u64,
    verbose: bool,
    /// minor faults at the previous mark (verbose only)
    flt: i64,
    items: Vec<(&'static str, f64, f64, f64, i64, f64, f64)>,
}

impl Marks {
    fn new(verbose: bool) -> Self {
        let unix0 = sys::unix_ns();
        let t = Instant::now();
        let flt = if verbose { sys::rusage().2 } else { 0 };
        Marks { t0: t, prev: t, unix0, verbose, flt, items: Vec::new() }
    }
    fn mark(&mut self, name: &'static str) {
        if std::env::var("SS_STOP").map_or(false, |v| v == name) {
            std::process::exit(0);
        }
        let now = Instant::now();
        let cpu = PHASE_CPU_NS.swap(0, Ordering::Relaxed) as f64 / 1e6;
        let mx = PHASE_MAX_NS.swap(0, Ordering::Relaxed) as f64 / 1e6;
        let mut df = 0;
        if self.verbose {
            let f = sys::rusage().2;
            df = f - self.flt;
            self.flt = f;
        }
        let idle = PHASE_IDLE_NS.swap(0, Ordering::Relaxed) as f64 / 1e6;
        let spawn = PHASE_SPAWN_NS.swap(0, Ordering::Relaxed) as f64 / 1e6;
        self.items.push((name, (now - self.prev).as_secs_f64() * 1e3, cpu, mx, df, idle, spawn));
        self.prev = now;
    }
    fn report(&self) -> String {
        let mut s = format!("total={:.1}ms |", self.t0.elapsed().as_secs_f64() * 1e3);
        for (n, w, c, m, f, idle, spawn) in &self.items {
            s.push_str(&format!(" {}={:.1}ms(cpu {:.0}/max {:.0} flt {} idle {:.1} spawn {:.2})", n, w, c, m, f, idle, spawn));
        }
        let (ut, st, minf, majf, nvcsw, nivcsw) = sys::rusage();
        s.push_str(&format!(
            " | rusage user={:.0}ms sys={:.0}ms minflt={} majflt={} vcsw={} ivcsw={} | unix_start_ns={} unix_end_ns={}",
            ut,
            st,
            minf,
            majf,
            nvcsw,
            nivcsw,
            self.unix0,
            sys::unix_ns()
        ));
        s
    }
}

// ---------------------------------------------------------------- parsing

/// Parses link lines of one chunk into (a, b) key pairs (self-links skipped).
fn link_task(input: &Input, task: Task, v: &mut Vec<(u64, u64)>) {
    let b = input.maps[task.file as usize].bytes();
    scan::for_each_record::<2, _>(b, task.start, task.end, |f| {
        if f[0].is_empty() || f[1].is_empty() {
            return;
        }
        let a = keys::key_of(f[0]);
        let c = keys::key_of(f[1]);
        if a != c {
            v.push((a, c));
        }
    });
}

/// The link work of one of `nl` link threads (thread `t`): parse links, build
/// the table (leader), union-find, finalize. Steps are separated by `bar`.
#[allow(clippy::too_many_arguments)]
fn link_worker(
    t: usize,
    links_in: &Input,
    lnext: &AtomicUsize,
    pairs: &[std::sync::Mutex<Vec<(u64, u64)>>],
    bar: &std::sync::Barrier,
    cell: &std::sync::OnceLock<LinkTable>,
    f1: &AtomicUsize,
    f2: &AtomicUsize,
) {
    let mut v = Vec::new();
    loop {
        let i = lnext.fetch_add(1, Ordering::Relaxed);
        let Some(&task) = links_in.tasks.get(i) else { break };
        link_task(links_in, task, &mut v);
    }
    *pairs[t].lock().unwrap() = v;
    bar.wait();
    if t == 0 {
        let total: usize = pairs.iter().map(|m| m.lock().unwrap().len()).sum();
        let _ = cell.set(LinkTable::with_pairs(total, pairs.len()));
    }
    bar.wait();
    let lt = cell.get().unwrap();
    {
        let v = pairs[t].lock().unwrap();
        let mut alloc = links::DenseAlloc::new();
        const AHEAD: usize = 8;
        for k in 0..v.len() {
            if k + AHEAD < v.len() {
                lt.prefetch(v[k + AHEAD].0);
                lt.prefetch(v[k + AHEAD].1);
            }
            lt.link(v[k].0, v[k].1, &mut alloc);
        }
    }
    bar.wait();
    let nb = lt.root_blocks();
    loop {
        let b = f1.fetch_add(1, Ordering::Relaxed);
        if b >= nb {
            break;
        }
        lt.finalize_roots(b);
    }
    bar.wait();
    let nb = lt.val_blocks();
    loop {
        let b = f2.fetch_add(1, Ordering::Relaxed);
        if b >= nb {
            break;
        }
        lt.finalize_vals(b);
    }
    bar.wait();
}

#[inline(always)]
fn ingest_rel(ms: i64, ws_s: i64) -> u32 {
    let s = ms.div_euclid(1000) - ws_s + (1i64 << 31);
    s.clamp(0, u32::MAX as i64) as u32
}

struct ParseOut {
    e: (Parts<ERec>, u32),
    c: Parts<CRec>,
}

/// Receives every parsed event copy: its event_id (`keys::mix` of the key),
/// ingest (`ingest_rel`), and for a copy that passes validation and is not
/// late, (user key, ts - window_start in ms, pay).
trait EvSink {
    fn copy(&mut self, eid: u64, ingest: u32, valid: Option<(u64, u32, u32)>);
}

/// Legacy pipeline: one ERec per copy, routed by event_id.
impl EvSink for (Parts<ERec>, u32) {
    #[inline(always)]
    fn copy(&mut self, eid: u64, ingest: u32, valid: Option<(u64, u32, u32)>) {
        let mut r = ERec { eid, ingest, ..Default::default() };
        if let Some((user, ts, pay)) = valid {
            r.user = user;
            r.ts = ts | OK_BIT;
            r.pay = pay;
        }
        self.0.push((eid >> (64 - self.1)) as usize, r);
    }
}

/// Type and price of a valid event: Some(pay) for view/cart (0) or a valid
/// purchase (price | category rank << 24).
#[inline(always)]
fn event_pay(ty: &[u8], price: &[u8], product: &[u8], ctx: &Ctx, wide: bool) -> Option<u32> {
    let n4 = ty.len();
    let is_purchase;
    let is_vc;
    if wide {
        let w = u64::from_le(unsafe { std::ptr::read_unaligned(ty.as_ptr() as *const u64) });
        let w4 = w & 0xFFFF_FFFF;
        is_purchase = n4 == 8 && w == u64::from_le_bytes(*b"purchase");
        is_vc = n4 == 4 && (w4 == u32::from_le_bytes(*b"view") as u64 || w4 == u32::from_le_bytes(*b"cart") as u64);
    } else {
        is_purchase = ty == b"purchase";
        is_vc = ty == b"view" || ty == b"cart";
    }
    if is_purchase {
        let price = ts::parse_price(price)?;
        let mut cat = if wide { unsafe { ctx.products.get_wide(product) } } else { ctx.products.get(product) } as u32;
        if cat == NONE as u32 {
            cat = ctx.unknown_cat;
        }
        Some(price | (cat << 24))
    } else if is_vc {
        Some(0)
    } else {
        None
    }
}

/// One event line (event_id non-empty). With `WIDE`, 32 bytes from the start
/// of every field must be readable.
#[inline(always)]
fn event_line<const WIDE: bool, S: EvSink>(f: &[&[u8]; 7], ctx: &Ctx, sink: &mut S) {
    let (ws, we) = (ctx.ws, ctx.we);
    let ing = if WIDE && ctx.tsw { unsafe { ctx.tsp.parse_wide_f1(f[3]) } } else { ctx.tsp.parse(f[3]) };
    let Some(ing) = ing else { return };
    let eid = keys::mix(if WIDE { unsafe { keys::key_of_wide(f[0]) } } else { keys::key_of(f[0]) });
    let ingest = ingest_rel(ing, ctx.ws_s);
    let valid = 'v: {
        if f[1].is_empty() || f[5].is_empty() {
            break 'v None;
        }
        let t = if WIDE && ctx.tsh {
            unsafe { ctx.tsp.parse_hyb(f[2]) }
        } else if WIDE && ctx.tsw {
            unsafe { ctx.tsp.parse_wide(f[2]) }
        } else {
            ctx.tsp.parse(f[2])
        };
        let Some(t) = t else { break 'v None };
        if t < ws || t >= we {
            break 'v None;
        }
        let Some(pay) = event_pay(f[4], f[6], f[5], ctx, WIDE) else { break 'v None };
        if ing - t > LATE {
            break 'v None;
        }
        let user = if WIDE { unsafe { keys::key_of_wide(f[1]) } } else { keys::key_of(f[1]) };
        Some((user, (t - ws) as u32, pay))
    };
    sink.copy(eid, ingest, valid);
}

#[inline(never)]
#[cold]
fn event_line_slow<S: EvSink>(f: &[&[u8]; 7], ctx: &Ctx, sink: &mut S) {
    event_line::<false, S>(f, ctx, sink)
}

/// One chunk of events/: parse, validate, hand every copy to `sink`.
/// `safe_end` is the end address of readable memory after `b` (>= its end).
#[inline(always)]
fn event_task<const AVX2: bool, S: EvSink>(b: &[u8], task: Task, ctx: &Ctx, sink: &mut S, safe_end: usize) {
    scan::for_each_record_x::<7, _, AVX2>(b, task.start, task.end, #[inline(always)] |f| {
        if f[0].is_empty() {
            return;
        }
        if f[6].as_ptr() as usize + 32 <= safe_end {
            event_line::<true, S>(f, ctx, sink);
        } else {
            event_line_slow(f, ctx, sink);
        }
    });
}

/// Receives every valid click, routed by person (partition `q`).
trait CSink {
    fn push_click(&mut self, q: usize, r: CRec);
}

impl CSink for Parts<CRec> {
    #[inline(always)]
    fn push_click(&mut self, q: usize, r: CRec) {
        self.push(q, r)
    }
}

/// One chunk of clicks/: parse, validate, resolve the person, route by person.
#[inline(always)]
fn click_task<const AVX2: bool, C: CSink>(b: &[u8], task: Task, ctx: &Ctx, lt: &LinkTable, c: &mut C, safe_end: usize, cache: &mut PCache) {
    let we = ctx.we;
    let cws = ctx.ws - WEEK;
    // Person lookups go through a small ring so the link-table slot can be
    // prefetched a few clicks ahead.
    const RING: usize = 16;
    let mut ring = [(CRec { person: 0, ck: 0, ts: 0, camp: 0, _pad: 0 }, 0u64); RING];
    let mut rh = 0usize;
    let mut rn = 0usize;
    let emit = |c: &mut C, cache: &mut PCache, (mut r, h): (CRec, u64)| {
        let user = r.person;
        r.person = person_hash(lt, user, h);
        cache.put(user, h, r.person);
        c.push_click((r.person >> (64 - ctx.ps_bits)) as usize, r);
    };
    scan::for_each_record_x::<4, _, AVX2>(b, task.start, task.end, #[inline(always)] |f| {
        if f[0].is_empty() || f[1].is_empty() {
            return;
        }
        let wide = f[3].as_ptr() as usize + 32 <= safe_end;
        let t = if wide && ctx.tsh {
            unsafe { ctx.tsp.parse_hyb(f[3]) }
        } else if wide && ctx.tsw {
            unsafe { ctx.tsp.parse_wide(f[3]) }
        } else {
            ctx.tsp.parse(f[3])
        };
        let Some(t) = t else { return };
        if t < cws || t >= we {
            return;
        }
        let camp = if wide { unsafe { ctx.campaigns.get_wide(f[2]) } } else { ctx.campaigns.get(f[2]) };
        if camp == NONE {
            return;
        }
        let user = if wide { unsafe { keys::key_of_wide(f[1]) } } else { keys::key_of(f[1]) };
        let h = keys::mix(user);
        let ck = if wide { unsafe { keys::key_of_wide(f[0]) } } else { keys::key_of(f[0]) };
        if let Some(hp) = cache.get(user, h) {
            c.push_click((hp >> (64 - ctx.ps_bits)) as usize, CRec { person: hp, ck, ts: (t - cws) as u32, camp: camp as u32, _pad: 0 });
            return;
        }
        lt.prefetch_h(h);
        // `person` holds the user key until the lookup in `emit`
        let r = (CRec { person: user, ck, ts: (t - cws) as u32, camp: camp as u32, _pad: 0 }, h);
        if rn == RING {
            emit(c, cache, ring[rh]);
            ring[rh] = r;
            rh = (rh + 1) % RING;
        } else {
            ring[(rh + rn) % RING] = r;
            rn += 1;
        }
    });
    for k in 0..rn {
        emit(c, cache, ring[(rh + k) % RING]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
unsafe fn event_task_avx2<S: EvSink>(b: &[u8], task: Task, ctx: &Ctx, e: &mut S, safe_end: usize) {
    event_task::<true, S>(b, task, ctx, e, safe_end)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
unsafe fn click_task_avx2<C: CSink>(b: &[u8], task: Task, ctx: &Ctx, lt: &LinkTable, c: &mut C, safe_end: usize, cache: &mut PCache) {
    click_task::<true, C>(b, task, ctx, lt, c, safe_end, cache)
}

/// AVX2 + BMI1/2 available and not disabled with SS_NOAVX2.
fn use_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::env::var("SS_NOAVX2").is_err()
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("bmi1")
            && std::arch::is_x86_feature_detected!("bmi2")
            && std::arch::is_x86_feature_detected!("lzcnt")
            && std::arch::is_x86_feature_detected!("popcnt")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Links (on up to 2 threads) overlapped with event parsing (on the rest, and
/// on the link threads once they are done); click tasks, which need the
/// finished link table, come last and wait for it.
fn parse_all(links_in: &Input, ev: &Input, cl: &Input, ctx: &Ctx) -> (LinkTable, Vec<ParseOut>) {
    let pe = 1usize << ctx.pe_bits;
    let ps = 1usize << ctx.ps_bits;
    let est_lines = ev.bytes / 70 + 1;
    let cap_e = est_lines / (ctx.threads * pe) * 3 / 2 + 64;
    let est_clicks = cl.bytes / 45 + 1;
    let cap_c = est_clicks / (ctx.threads * ps) * 3 / 2 + 64;
    let ntasks = ev.tasks.len() + cl.tasks.len();
    let next = AtomicUsize::new(0);
    let avx2 = ctx.avx2;
    let nl = ctx.threads.min(2);
    let lnext = AtomicUsize::new(0);
    let pairs: Vec<std::sync::Mutex<Vec<(u64, u64)>>> = (0..nl).map(|_| std::sync::Mutex::new(Vec::new())).collect();
    let bar = std::sync::Barrier::new(nl);
    let cell: std::sync::OnceLock<LinkTable> = std::sync::OnceLock::new();
    let (f1, f2) = (AtomicUsize::new(0), AtomicUsize::new(0));
    let ready = std::sync::atomic::AtomicBool::new(false);
    // Experiment (SS_READ): pread each chunk into a per-thread buffer instead
    // of reading the file mapping.
    let use_read = std::env::var("SS_READ").is_ok();
    let max_task = ev.tasks.iter().chain(cl.tasks.iter()).map(|t| t.end - t.start).max().unwrap_or(0);
    let outs = par(ctx.threads, |t| {
        let mut out = ParseOut { e: (Parts::new(pe, cap_e), ctx.pe_bits), c: Parts::new(ps, cap_c) };
        let mut pcache = PCache::new();
        let mut rbuf: Vec<u8> = if use_read { vec![0u8; max_task + 64] } else { Vec::new() };
        if t < nl {
            link_worker(t, links_in, &lnext, &pairs, &bar, &cell, &f1, &f2);
            if t == 0 {
                ready.store(true, Ordering::Release);
            }
        }
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= ntasks {
                break;
            }
            if i < ev.tasks.len() {
                let mut task = ev.tasks[i];
                let mut b = ev.maps[task.file as usize].bytes();
                let mut safe_end = b.as_ptr() as usize + b.len();
                if use_read {
                    let n = task.end - task.start;
                    ev.maps[task.file as usize].read_at(&mut rbuf[..n], task.start);
                    safe_end = rbuf.as_ptr() as usize + rbuf.len();
                    b = &rbuf[..n];
                    task = Task { file: task.file, start: 0, end: n };
                }
                if avx2 {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        event_task_avx2(b, task, ctx, &mut out.e, safe_end)
                    }
                } else {
                    event_task::<false, _>(b, task, ctx, &mut out.e, safe_end)
                }
            } else {
                while !ready.load(Ordering::Acquire) {
                    std::thread::yield_now();
                }
                let lt = cell.get().unwrap();
                let mut task = cl.tasks[i - ev.tasks.len()];
                let mut b = cl.maps[task.file as usize].bytes();
                let mut safe_end = b.as_ptr() as usize + b.len();
                if use_read {
                    let n = task.end - task.start;
                    cl.maps[task.file as usize].read_at(&mut rbuf[..n], task.start);
                    safe_end = rbuf.as_ptr() as usize + rbuf.len();
                    b = &rbuf[..n];
                    task = Task { file: task.file, start: 0, end: n };
                }
                if avx2 {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        click_task_avx2(b, task, ctx, lt, &mut out.c, safe_end, &mut pcache)
                    }
                } else {
                    click_task::<false, _>(b, task, ctx, lt, &mut out.c, safe_end, &mut pcache)
                }
            }
        }
        out
    });
    (cell.into_inner().unwrap(), outs)
}

// ---------------------------------------------------------------- dedup

#[derive(Clone, Copy, Default)]
struct Slot {
    eid: u64,
    ingest: u32,
    idx: u32,
}


/// Dedup one bucket of records (all copies of an event_id are in the same
/// bucket): keeps the copy with the smallest ingest (ties: any, the copies are
/// identical) and appends the kept copies that passed validation to `kept`.
#[inline(always)]
fn dedup_bucket(recs: &[ERec], table: &mut Vec<Slot>, kept: &mut Vec<(u64, u32, u32)>) {
    let n = recs.len();
    if n == 0 {
        return;
    }
    let cap = (n + n / 4 + 16).next_power_of_two();
    let mask = cap - 1;
    table.clear();
    table.resize(cap, Slot::default());
    for (j, r) in recs.iter().enumerate() {
        let mut i = r.eid as usize & mask;
        loop {
            let s = &mut table[i];
            if s.eid == 0 {
                *s = Slot { eid: r.eid, ingest: r.ingest, idx: j as u32 };
                break;
            }
            if s.eid == r.eid {
                if r.ingest < s.ingest {
                    s.ingest = r.ingest;
                    s.idx = j as u32;
                }
                break;
            }
            i = (i + 1) & mask;
        }
    }
    for s in table.iter() {
        if s.eid != 0 {
            let r = recs[s.idx as usize];
            if r.ts & OK_BIT != 0 {
                kept.push((r.user, r.ts & !OK_BIT, r.pay));
            }
        }
    }
}

/// Records per sub-bucket targeted by the in-cache split in `dedup`.
const SUB_TARGET: usize = 24 * 1024;

fn dedup(e: &[Parts<ERec>], lt: &LinkTable, ctx: &Ctx) -> Vec<Parts<SRec>> {
    let pe = 1usize << ctx.pe_bits;
    let ps = 1usize << ctx.ps_bits;
    let total: usize = e.iter().map(|t| t.total()).sum();
    let cap_s = total / (ctx.threads * ps) * 3 / 2 + 64;
    // biggest partitions first
    let mut order: Vec<(usize, usize)> = (0..pe).map(|p| (e.iter().map(|t| t.len(p)).sum::<usize>(), p)).collect();
    order.sort_unstable_by(|a, b| b.cmp(a));
    let next = AtomicUsize::new(0);
    par(ctx.threads, |_| {
        let mut out: Parts<SRec> = Parts::new(ps, cap_s);
        let mut table: Vec<Slot> = Vec::new();
        let mut kept: Vec<(u64, u32, u32)> = Vec::new();
        let mut scratch: Vec<ERec> = Vec::new();
        let mut counts: Vec<usize> = Vec::new();
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            let Some(&(n, p)) = order.get(i) else { break };
            if n == 0 {
                continue;
            }
            kept.clear();
            let sub_bits = ceil_log2(n.div_ceil(SUB_TARGET)).min(12);
            if sub_bits == 0 {
                let mut slices = e.iter().flat_map(|t| t.get(p)).filter(|sl| !sl.is_empty());
                let first = slices.next().unwrap();
                if slices.next().is_none() {
                    dedup_bucket(first, &mut table, &mut kept);
                } else {
                    scratch.clear();
                    for t in e {
                        for sl in t.get(p) {
                            scratch.extend_from_slice(sl);
                        }
                    }
                    dedup_bucket(&scratch, &mut table, &mut kept);
                }
            } else {
                // split by the hash bits just below the partition bits
                let shift = 64 - ctx.pe_bits - sub_bits;
                let smask = (1usize << sub_bits) - 1;
                counts.clear();
                counts.resize((1 << sub_bits) + 1, 0);
                for t in e {
                    for sl in t.get(p) {
                        for r in sl {
                            counts[((r.eid >> shift) as usize & smask) + 1] += 1;
                        }
                    }
                }
                for b in 0..(1 << sub_bits) {
                    counts[b + 1] += counts[b];
                }
                scratch.clear();
                scratch.reserve(n);
                unsafe { scratch.set_len(n) };
                let mut cur: Vec<usize> = counts[..(1 << sub_bits)].to_vec();
                for t in e {
                    for sl in t.get(p) {
                        for r in sl {
                            let b = (r.eid >> shift) as usize & smask;
                            unsafe { *scratch.get_unchecked_mut(cur[b]) = *r };
                            cur[b] += 1;
                        }
                    }
                }
                for b in 0..(1 << sub_bits) {
                    dedup_bucket(&scratch[counts[b]..counts[b + 1]], &mut table, &mut kept);
                }
            }
            const AHEAD: usize = 16;
            for k in 0..kept.len() {
                if k + AHEAD < kept.len() {
                    lt.prefetch(kept[k + AHEAD].0);
                }
                let (user, ts, pay) = kept[k];
                let person = keys::mix(lt.person_of(user));
                out.push((person >> (64 - ctx.ps_bits)) as usize, SRec { person, ts, pay });
            }
        }
        out
    })
}


// ---------------------------------------------------------------- pipeline B
//
// Parse emits, per event copy, a 16-byte dedup record routed by event_id
// (ARec) and, for a copy that passes validation, a 16-byte session record
// routed by person (SRec; the person is resolved at parse, links come first).
// Dedup then only has to find the losing copies and mark their SRec dead.

/// Dedup stream record: one per parsed event copy, routed by event_id.
#[derive(Clone, Copy, Default)]
#[repr(C)]
struct ARec {
    /// keys::mix(event_id key); never 0
    eid: u64,
    ingest: u32,
    /// Location of the copy's SRec (thread | person partition | index), or
    /// LOC_NONE if the copy is invalid.
    loc: u32,
}

const LOC_NONE: u32 = u32::MAX;
/// `SRec::ts` of a copy that lost dedup (real offsets are < 2^31).
const DEAD: u32 = u32::MAX;

#[derive(Clone, Copy)]
struct LocBits {
    qbits: u32,
    ibits: u32,
    /// indices must be < imax (so that no location equals LOC_NONE)
    imax: usize,
}

impl LocBits {
    fn new(threads: usize, ps_bits: u32) -> Option<LocBits> {
        let tbits = ceil_log2(threads);
        let ibits = env_usize("SS_LOCBITS").map(|b| b as u32).unwrap_or(32u32.saturating_sub(tbits + ps_bits));
        if ibits < 1 || tbits + ps_bits + ibits > 32 {
            return None;
        }
        Some(LocBits { qbits: ps_bits, ibits, imax: (1usize << ibits) - 1 })
    }
}

#[derive(Clone, Copy, Default)]
struct Pending {
    eid: u64,
    user: u64,
    /// keys::mix(user)
    h: u64,
    ingest: u32,
    ts: u32,
    pay: u32,
}

/// Valid copies queued between the link-table prefetch and the lookup
/// (build-time override SS_BRING, power of two).
const BRING: usize = const_usize(option_env!("SS_BRING"), 16);

const fn const_usize(s: Option<&str>, default: usize) -> usize {
    match s {
        None => default,
        Some(s) => {
            let b = s.as_bytes();
            let mut v = 0usize;
            let mut i = 0;
            while i < b.len() {
                v = v * 10 + (b[i] - b'0') as usize;
                i += 1;
            }
            v
        }
    }
}

/// One parse thread's pipeline-B output. Valid copies wait in a small ring
/// so that the link-table slot of their user is prefetched before lookup.
struct BSink<'a> {
    a: Parts<ARec>,
    s: Parts<SRec>,
    ring: [Pending; BRING],
    rh: usize,
    rn: usize,
    tbase: u32,
    overflow: bool,
    lt: &'a LinkTable,
    pe_bits: u32,
    ps_bits: u32,
    lb: LocBits,
    /// benchmark only: skip the person lookup (person = user)
    nolookup: bool,
    cache: PCache,
}

impl BSink<'_> {
    #[inline(always)]
    fn emit(&mut self, p: Pending) {
        let person = if self.nolookup { p.h } else { person_hash(self.lt, p.user, p.h) };
        self.cache.put(p.user, p.h, person);
        self.emit_hp(p, person);
    }

    /// Writes the session record and the dedup record of a valid copy whose
    /// person hash is known.
    #[inline(always)]
    fn emit_hp(&mut self, p: Pending, person: u64) {
        let q = (person >> (64 - self.ps_bits)) as usize;
        let i = self.s.len(q);
        self.s.push(q, SRec { person, ts: p.ts, pay: p.pay });
        let loc = if i < self.lb.imax {
            self.tbase | ((q as u32) << self.lb.ibits) | i as u32
        } else {
            self.overflow = true;
            LOC_NONE
        };
        self.a.push((p.eid >> (64 - self.pe_bits)) as usize, ARec { eid: p.eid, ingest: p.ingest, loc });
    }

    fn flush(&mut self) {
        while self.rn > 0 {
            let p = self.ring[self.rh];
            self.rh = (self.rh + 1) % BRING;
            self.rn -= 1;
            self.emit(p);
        }
    }
}

impl EvSink for BSink<'_> {
    #[inline(always)]
    fn copy(&mut self, eid: u64, ingest: u32, valid: Option<(u64, u32, u32)>) {
        match valid {
            None => self.a.push((eid >> (64 - self.pe_bits)) as usize, ARec { eid, ingest, loc: LOC_NONE }),
            Some((user, ts, pay)) => {
                let h = keys::mix(user);
                let p = Pending { eid, user, h, ingest, ts, pay };
                if let Some(hp) = self.cache.get(user, h) {
                    self.emit_hp(p, hp);
                    return;
                }
                self.lt.prefetch_h(h);
                if self.rn == BRING {
                    let old = self.ring[self.rh];
                    self.ring[self.rh] = p;
                    self.rh = (self.rh + 1) % BRING;
                    self.emit(old);
                } else {
                    self.ring[(self.rh + self.rn) % BRING] = p;
                    self.rn += 1;
                }
            }
        }
    }
}

struct BOut {
    a: Parts<ARec>,
    s: Parts<SRec>,
    c: Parts<CRec>,
    overflow: bool,
}

/// Pipeline B parse: links on all threads first, then events and clicks.
fn parse_b(links_in: &Input, ev: &Input, cl: &Input, ctx: &Ctx, lb: LocBits) -> (LinkTable, Vec<BOut>) {
    let pe = 1usize << ctx.pe_bits;
    let ps = 1usize << ctx.ps_bits;
    // Capacities with modest slack: huge pages zero the slack too, and a
    // partition that fills up continues in its (exact) overflow vector.
    let est_lines = ev.bytes / 72 + 1;
    let cap_a = est_lines / (ctx.threads * pe) * 5 / 4 + 32;
    let cap_s = est_lines / (ctx.threads * ps) * 5 / 4 + 32;
    let est_clicks = cl.bytes / 44 + 1;
    let cap_c = est_clicks / (ctx.threads * ps) * 5 / 4 + 32;
    let release = std::env::var("SS_NORELEASE").is_err();
    let ntasks = ev.tasks.len() + cl.tasks.len();
    let next = AtomicUsize::new(0);
    let avx2 = ctx.avx2;
    let nl = ctx.threads;
    let lnext = AtomicUsize::new(0);
    let pairs: Vec<std::sync::Mutex<Vec<(u64, u64)>>> = (0..nl).map(|_| std::sync::Mutex::new(Vec::new())).collect();
    let bar = std::sync::Barrier::new(nl);
    let cell: std::sync::OnceLock<LinkTable> = std::sync::OnceLock::new();
    let (f1, f2) = (AtomicUsize::new(0), AtomicUsize::new(0));
    let t0 = Instant::now();
    let outs = par(ctx.threads, |t| {
        link_worker(t, links_in, &lnext, &pairs, &bar, &cell, &f1, &f2);
        if t == 0 {
            LINKS_US.store(t0.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        let lt = cell.get().unwrap();
        let mut sink = BSink {
            a: Parts::with_wc(pe, cap_a, true),
            s: Parts::with_wc(ps, cap_s, true),
            ring: [Pending::default(); BRING],
            rh: 0,
            rn: 0,
            tbase: ((t as u64) << (lb.qbits + lb.ibits)) as u32,
            overflow: false,
            lt,
            pe_bits: ctx.pe_bits,
            ps_bits: ctx.ps_bits,
            lb,
            nolookup: false,
            cache: PCache::new(),
        };
        let mut c: Parts<CRec> = Parts::with_wc(ps, cap_c, true);
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= ntasks {
                break;
            }
            if i < ev.tasks.len() {
                let task = ev.tasks[i];
                let b = ev.maps[task.file as usize].bytes();
                let safe_end = b.as_ptr() as usize + b.len();
                if avx2 {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        event_task_avx2(b, task, ctx, &mut sink, safe_end)
                    }
                } else {
                    event_task::<false, _>(b, task, ctx, &mut sink, safe_end)
                }
                sink.flush();
                if release {
                    // the chunk is parsed: drop its page mappings now, in
                    // parallel, instead of in the single-threaded exit
                    sys::release_inner(b.as_ptr() as usize + task.start, b.as_ptr() as usize + task.end);
                }
            } else {
                let task = cl.tasks[i - ev.tasks.len()];
                let b = cl.maps[task.file as usize].bytes();
                let safe_end = b.as_ptr() as usize + b.len();
                if avx2 {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        click_task_avx2(b, task, ctx, lt, &mut c, safe_end, &mut sink.cache)
                    }
                } else {
                    click_task::<false, _>(b, task, ctx, lt, &mut c, safe_end, &mut sink.cache)
                }
                if release {
                    sys::release_inner(b.as_ptr() as usize + task.start, b.as_ptr() as usize + task.end);
                }
            }
        }
        sink.a.finish();
        sink.s.finish();
        c.finish();
        BOut { a: sink.a, s: sink.s, c, overflow: sink.overflow }
    });
    (cell.into_inner().unwrap(), outs)
}

/// Wall time of the links step of `parse_b` (verbose report).
static LINKS_US: AtomicU64 = AtomicU64::new(0);

/// Records per sub-bucket targeted by the in-cache split in `dedup_b`.
const SUB_B: usize = 8 * 1024;

/// Marks the SRec at `loc` dead (no-op for LOC_NONE).
#[inline(always)]
fn kill(loc: u32, sb: &[BOut], lb: LocBits) {
    if loc != LOC_NONE {
        let t = (loc as u64 >> (lb.qbits + lb.ibits)) as usize;
        let q = ((loc >> lb.ibits) as usize) & ((1usize << lb.qbits) - 1);
        let i = (loc as usize) & ((1usize << lb.ibits) - 1);
        // Each SRec belongs to exactly one event copy, whose ARec is in
        // exactly one event_id partition, handled by one thread: no two
        // threads ever write the same record, and nothing reads SRecs
        // until dedup has finished.
        unsafe { (*sb[t].s.slot(q, i)).ts = DEAD };
    }
}

/// Inserts one copy into the dedup table: keeps the copy with the smallest
/// ingest (on a tie, the one already there; equal-ingest copies are
/// identical) and marks the loser's SRec dead.
#[inline(always)]
fn dedup_insert(table: &mut [ARec], mask: usize, r: &ARec, sb: &[BOut], lb: LocBits) {
    let mut i = r.eid as usize & mask;
    loop {
        let slot = unsafe { table.get_unchecked_mut(i) };
        if slot.eid == 0 {
            *slot = *r;
            return;
        }
        if slot.eid == r.eid {
            if r.ingest < slot.ingest {
                kill(slot.loc, sb, lb);
                slot.ingest = r.ingest;
                slot.loc = r.loc;
            } else {
                kill(r.loc, sb, lb);
            }
            return;
        }
        i = (i + 1) & mask;
    }
}

fn dedup_b(sb: &[BOut], ctx: &Ctx, lb: LocBits) {
    let pe = 1usize << ctx.pe_bits;
    let mut order: Vec<(usize, usize)> = (0..pe).map(|p| (sb.iter().map(|t| t.a.len(p)).sum::<usize>(), p)).collect();
    order.sort_unstable_by(|a, b| b.cmp(a));
    let next = AtomicUsize::new(0);
    par(ctx.threads, |_| {
        let mut table: Vec<ARec> = Vec::new();
        let mut scratch: Vec<ARec> = Vec::new();
        let mut counts: Vec<usize> = Vec::new();
        let reset = |table: &mut Vec<ARec>, n: usize| -> usize {
            let cap = (n + n / 4 + 16).next_power_of_two();
            table.clear();
            table.resize(cap, ARec::default());
            cap - 1
        };
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            let Some(&(n, p)) = order.get(i) else { break };
            if n == 0 {
                continue;
            }
            let sub_bits = ceil_log2(n.div_ceil(SUB_B)).min(12);
            if sub_bits == 0 {
                let mask = reset(&mut table, n);
                for t in sb {
                    for sl in t.a.get(p) {
                        for r in sl {
                            dedup_insert(&mut table, mask, r, sb, lb);
                        }
                    }
                }
                continue;
            }
            // split by the hash bits just below the partition bits
            let shift = 64 - ctx.pe_bits - sub_bits;
            let smask = (1usize << sub_bits) - 1;
            counts.clear();
            counts.resize((1 << sub_bits) + 1, 0);
            for t in sb {
                for sl in t.a.get(p) {
                    for r in sl {
                        counts[((r.eid >> shift) as usize & smask) + 1] += 1;
                    }
                }
            }
            for b in 0..(1 << sub_bits) {
                counts[b + 1] += counts[b];
            }
            scratch.clear();
            scratch.resize(n, ARec::default());
            let mut cur: Vec<usize> = counts[..(1 << sub_bits)].to_vec();
            for t in sb {
                for sl in t.a.get(p) {
                    for r in sl {
                        let b = (r.eid >> shift) as usize & smask;
                        unsafe { *scratch.get_unchecked_mut(cur[b]) = *r };
                        cur[b] += 1;
                    }
                }
            }
            for b in 0..(1 << sub_bits) {
                let bucket = &scratch[counts[b]..counts[b + 1]];
                if bucket.is_empty() {
                    continue;
                }
                let mask = reset(&mut table, bucket.len());
                for r in bucket {
                    dedup_insert(&mut table, mask, r, sb, lb);
                }
            }
        }
    });
}

// ---------------------------------------------------------------- sessions

struct PMap {
    keys: Vec<u64>,
    vals: Vec<u32>,
    mask: usize,
    len: usize,
}

impl PMap {
    fn new() -> Self {
        PMap { keys: Vec::new(), vals: Vec::new(), mask: 0, len: 0 }
    }
    fn reset(&mut self, expected: usize) {
        let cap = (expected * 2).next_power_of_two().max(64);
        if self.keys.len() != cap {
            self.keys = vec![0; cap];
            self.vals = vec![0; cap];
        } else {
            self.keys.fill(0);
        }
        self.mask = cap - 1;
        self.len = 0;
    }
    fn grow(&mut self) {
        let old_k = std::mem::take(&mut self.keys);
        let old_v = std::mem::take(&mut self.vals);
        let cap = old_k.len() * 2;
        self.keys = vec![0; cap];
        self.vals = vec![0; cap];
        self.mask = cap - 1;
        for (k, v) in old_k.into_iter().zip(old_v) {
            if k != 0 {
                let mut i = k as usize & self.mask;
                while self.keys[i] != 0 {
                    i = (i + 1) & self.mask;
                }
                self.keys[i] = k;
                self.vals[i] = v;
            }
        }
    }
    /// Returns the value for k, inserting `self.len` if absent.
    #[inline(always)]
    fn insert(&mut self, k: u64) -> (u32, bool) {
        if (self.len + 1) * 2 > self.keys.len() {
            self.grow();
        }
        let mut i = k as usize & self.mask;
        loop {
            let kk = self.keys[i];
            if kk == k {
                return (self.vals[i], false);
            }
            if kk == 0 {
                self.keys[i] = k;
                let v = self.len as u32;
                self.vals[i] = v;
                self.len += 1;
                return (v, true);
            }
            i = (i + 1) & self.mask;
        }
    }
    #[inline(always)]
    fn get(&self, k: u64) -> Option<u32> {
        let mut i = k as usize & self.mask;
        loop {
            let kk = self.keys[i];
            if kk == k {
                return Some(self.vals[i]);
            }
            if kk == 0 {
                return None;
            }
            i = (i + 1) & self.mask;
        }
    }
}

struct SOut {
    rows: Vec<URow>,
    nrows: usize,
    /// [(ncamp + 1) * ndays] of (sessions, purchasing, revenue); index ncamp = organic.
    cred: Vec<[u64; 3]>,
    /// [ncamp * ndays] assisted sessions.
    assist: Vec<u64>,
}

struct SState {
    map: PMap,
    persons: Vec<u64>,
    ridx: Vec<u32>,
    start: Vec<u32>,
    cursor: Vec<u32>,
    ev: Vec<u64>,
    cidx: Vec<u32>,
    cstart: Vec<u32>,
    clicks: Vec<Click>,
    catsum: Vec<u64>,
    touched: Vec<u8>,
    wcount: Vec<u32>,
    wpos: Vec<u32>,
    active: Vec<u16>,
    /// sub-phase timers (verbose)
    acc: prof::Acc,
}

impl SState {
    fn new(ctx: &Ctx) -> Self {
        SState {
            map: PMap::new(),
            persons: Vec::new(),
            ridx: Vec::new(),
            start: Vec::new(),
            cursor: Vec::new(),
            ev: Vec::new(),
            cidx: Vec::new(),
            cstart: Vec::new(),
            clicks: Vec::new(),
            catsum: vec![0; 256],
            touched: Vec::new(),
            wcount: vec![0; ctx.ncamp.max(1)],
            wpos: vec![0; ctx.ncamp.max(1)],
            active: Vec::new(),
            acc: prof::Acc::new(),
        }
    }

    #[inline(always)]
    fn win_add(&mut self, c: u32) {
        let c = c as usize;
        if self.wcount[c] == 0 {
            self.wpos[c] = self.active.len() as u32;
            self.active.push(c as u16);
        }
        self.wcount[c] += 1;
    }

    #[inline(always)]
    fn win_remove(&mut self, c: u32) {
        let c = c as usize;
        self.wcount[c] -= 1;
        if self.wcount[c] == 0 {
            let p = self.wpos[c] as usize;
            let last = self.active.pop().unwrap();
            if p < self.active.len() {
                self.active[p] = last;
                self.wpos[last as usize] = p as u32;
            }
        }
    }

    fn flush_day(&mut self, person: u64, day: u32, sessions: u32, events: u32, revenue: u64, ctx: &Ctx, out: &mut SOut) {
        self.acc.lap(S_WALK);
        let mut top = u32::MAX;
        let mut best = 0u64;
        for &c in &self.touched {
            let v = self.catsum[c as usize];
            if v > best || (v == best && (c as u32) < top) {
                best = v;
                top = c as u32;
            }
        }
        for &c in &self.touched {
            self.catsum[c as usize] = 0;
        }
        self.touched.clear();
        out.rows.push(URow { person, day: (ctx.first_day + day as i64) as i32, sessions, events, top, revenue });
        self.acc.lap(S_ROWS);
    }

    fn partition(
        &mut self,
        q: usize,
        sbufs: &[&Parts<SRec>],
        cbufs: &[&Parts<CRec>],
        ctx: &Ctx,
        out: &mut SOut,
    ) {
        let n: usize = sbufs.iter().map(|t| t.len(q)).sum();
        if n == 0 {
            return;
        }
        self.acc.start();
        self.map.reset(n / 8 + 16);
        self.persons.clear();
        self.ridx.clear();
        for t in sbufs {
            for sl in t.get(q) {
                for r in sl {
                    if r.ts == DEAD {
                        continue;
                    }
                    let (idx, new) = self.map.insert(r.person);
                    if new {
                        self.persons.push(r.person);
                    }
                    self.ridx.push(idx);
                }
            }
        }
        let np = self.persons.len();
        self.acc.lap(S_MAP);
        self.start.clear();
        self.start.resize(np + 1, 0);
        for &i in &self.ridx {
            self.start[i as usize + 1] += 1;
        }
        for k in 0..np {
            self.start[k + 1] += self.start[k];
        }
        self.cursor.clear();
        self.cursor.extend_from_slice(&self.start[..np]);
        self.ev.clear();
        self.ev.resize(n, 0);
        let mut k = 0;
        for t in sbufs {
            for sl in t.get(q) {
                for r in sl {
                    if r.ts == DEAD {
                        continue;
                    }
                    let i = self.ridx[k] as usize;
                    let c = self.cursor[i] as usize;
                    self.ev[c] = ((r.ts as u64) << 32) | r.pay as u64;
                    self.cursor[i] += 1;
                    k += 1;
                }
            }
        }
        self.acc.lap(S_CSORT);
        // clicks of these persons
        self.cidx.clear();
        self.cstart.clear();
        self.cstart.resize(np + 1, 0);
        for t in cbufs {
            for sl in t.get(q) {
                for c in sl {
                    match self.map.get(c.person) {
                        Some(i) => {
                            self.cidx.push(i);
                            self.cstart[i as usize + 1] += 1;
                        }
                        None => self.cidx.push(u32::MAX),
                    }
                }
            }
        }
        for k in 0..np {
            self.cstart[k + 1] += self.cstart[k];
        }
        let nc = self.cstart[np] as usize;
        self.cursor.clear();
        self.cursor.extend_from_slice(&self.cstart[..np]);
        self.clicks.clear();
        self.clicks.resize(nc, Click { ts: 0, camp: 0, ck: 0 });
        let mut k = 0;
        for t in cbufs {
            for sl in t.get(q) {
                for c in sl {
                    let i = self.cidx[k];
                    k += 1;
                    if i != u32::MAX {
                        let pos = self.cursor[i as usize] as usize;
                        self.clicks[pos] = Click { ts: c.ts, camp: c.camp, ck: c.ck };
                        self.cursor[i as usize] += 1;
                    }
                }
            }
        }
        self.acc.lap(S_CGROUP);
        let mut ev = std::mem::take(&mut self.ev);
        let mut clicks = std::mem::take(&mut self.clicks);
        if !self.acc.on() {
            for p in 0..np {
                let es = &mut ev[self.start[p] as usize..self.start[p + 1] as usize];
                let cs = &mut clicks[self.cstart[p] as usize..self.cstart[p + 1] as usize];
                person_sort(es, cs);
                self.person(self.persons[p], es, cs, ctx, out);
            }
        } else {
            // timed: the same work in three passes
            for p in 0..np {
                ev[self.start[p] as usize..self.start[p + 1] as usize].sort_unstable();
            }
            self.acc.lap(S_ESORT);
            for p in 0..np {
                person_sort(&mut [], &mut clicks[self.cstart[p] as usize..self.cstart[p + 1] as usize]);
            }
            self.acc.lap(S_CLSORT);
            for p in 0..np {
                let es = &mut ev[self.start[p] as usize..self.start[p + 1] as usize];
                let cs = &mut clicks[self.cstart[p] as usize..self.cstart[p + 1] as usize];
                self.person(self.persons[p], es, cs, ctx, out);
            }
            self.acc.lap(S_WALK);
        }
        self.ev = ev;
        self.clicks = clicks;
    }

    /// Sessions, attribution and rows of one person whose events and clicks
    /// are sorted (`person_sort`).
    fn person(&mut self, person: u64, ev: &mut [u64], cl: &mut [Click], ctx: &Ctx, out: &mut SOut) {
        let ndays = ctx.ndays;
        let organic = ctx.ncamp;
        let (mut add, mut rem) = (0usize, 0usize);
        let mut cur_day = u32::MAX;
        let (mut d_sessions, mut d_events, mut d_rev) = (0u32, 0u32, 0u64);
        let mut i = 0;
        while i < ev.len() {
            let start = (ev[i] >> 32) as u32;
            let day = ((ctx.ws_mod_day + start as u64) / DAY as u64) as u32;
            if day != cur_day {
                if cur_day != u32::MAX {
                    self.flush_day(person, cur_day, d_sessions, d_events, d_rev, ctx, out);
                }
                cur_day = day;
                d_sessions = 0;
                d_events = 0;
                d_rev = 0;
            }
            let mut last = start;
            let mut rev = 0u64;
            let mut purch = false;
            let mut j = i;
            while j < ev.len() {
                let v = ev[j];
                let t = (v >> 32) as u32;
                if t - last > GAP {
                    break;
                }
                last = t;
                let pay = v as u32;
                if pay != 0 {
                    let price = (pay & 0xFF_FFFF) as u64;
                    let cat = (pay >> 24) as usize;
                    rev += price;
                    purch = true;
                    if self.catsum[cat] == 0 {
                        self.touched.push(cat as u8);
                    }
                    self.catsum[cat] += price;
                }
                j += 1;
            }
            d_sessions += 1;
            d_events += (j - i) as u32;
            d_rev += rev;
            // attribution: click offsets c (base ws - 7d) qualify iff start <= c <= start + 7d
            let hi = start as u64 + WEEK as u64;
            while add < cl.len() && cl[add].ts as u64 <= hi {
                self.win_add(cl[add].camp);
                add += 1;
            }
            while rem < add && cl[rem].ts < start {
                self.win_remove(cl[rem].camp);
                rem += 1;
            }
            let credit = if add > rem {
                let tmax = cl[add - 1].ts;
                let mut k = add - 1;
                while k > rem && cl[k - 1].ts == tmax {
                    k -= 1;
                }
                cl[k].camp as usize
            } else {
                organic
            };
            let ci = credit * ndays + day as usize;
            out.cred[ci][0] += 1;
            out.cred[ci][1] += purch as u64;
            out.cred[ci][2] += rev;
            for &c in &self.active {
                out.assist[c as usize * ndays + day as usize] += 1;
            }
            i = j;
        }
        if cur_day != u32::MAX {
            self.flush_day(person, cur_day, d_sessions, d_events, d_rev, ctx, out);
        }
        while rem < add {
            self.win_remove(cl[rem].camp);
            rem += 1;
        }
    }
}

/// Sorts a person's events by (ts, pay) and clicks by (ts, click_id).
#[inline(always)]
fn person_sort(ev: &mut [u64], cl: &mut [Click]) {
    ev.sort_unstable();
    if cl.len() > 1 {
        cl.sort_unstable_by(|a, b| a.ts.cmp(&b.ts).then_with(|| keys::cmp(a.ck, b.ck)));
    }
}

// sub-phase timer slots of the sessions phase (verbose)
const S_MAP: usize = 0;
const S_CSORT: usize = 1;
const S_CGROUP: usize = 2;
const S_ESORT: usize = 3;
const S_CLSORT: usize = 4;
const S_WALK: usize = 5;
const S_ROWS: usize = 6;
const S_ENCODE: usize = 7;
const S_FWRITE: usize = 8;

fn name_session_slots() {
    for (i, n) in [
        (S_MAP, "s.map"),
        (S_CSORT, "s.countsort"),
        (S_CGROUP, "s.clickgroup"),
        (S_ESORT, "s.evsort"),
        (S_CLSORT, "s.clicksort"),
        (S_WALK, "s.walk"),
        (S_ROWS, "s.rows"),
        (S_ENCODE, "s.encode"),
        (S_FWRITE, "s.fwrite"),
    ] {
        prof::name(i, n);
    }
}

/// Sessions, attribution and aggregation per person partition; each thread then
/// writes its own user_daily part file (overlapping with threads still busy).
fn sessions(sbufs: &[&Parts<SRec>], cbufs: &[&Parts<CRec>], ctx: &Ctx, ud: &Path) -> Vec<SOut> {
    let ps = 1usize << ctx.ps_bits;
    // biggest partitions first, for load balance (whales)
    let mut order: Vec<(usize, usize)> =
        (0..ps).map(|q| (sbufs.iter().map(|t| t.len(q)).sum::<usize>(), q)).collect();
    order.sort_unstable_by(|a, b| b.cmp(a));
    let next = AtomicUsize::new(0);
    par(ctx.threads, |t| {
        let mut st = SState::new(ctx);
        let mut out = SOut {
            rows: Vec::new(),
            nrows: 0,
            cred: vec![[0; 3]; (ctx.ncamp + 1) * ctx.ndays],
            assist: vec![0; ctx.ncamp * ctx.ndays],
        };
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            let Some(&(_, q)) = order.get(i) else { break };
            st.partition(q, sbufs, cbufs, ctx, &mut out);
        }
        if t == 0 || !out.rows.is_empty() {
            write_user_daily(&ud.join(format!("part-{}.parquet", t)), &out.rows, ctx, &mut st.acc);
        }
        st.acc.commit();
        out.nrows = out.rows.len();
        out.rows = Vec::new();
        out
    })
}

// ---------------------------------------------------------------- output

fn write_user_daily(path: &Path, rows: &[URow], ctx: &Ctx, acc: &mut prof::Acc) {
    acc.start();
    let n = rows.len();
    let mut ubytes = Vec::with_capacity(n * 10 + 16);
    let mut uends = Vec::with_capacity(n);
    let mut days = Vec::with_capacity(n);
    let mut sess = Vec::with_capacity(n);
    let mut evs = Vec::with_capacity(n);
    let mut revs = Vec::with_capacity(n);
    let mut cbytes = Vec::with_capacity(n * 8 + 16);
    let mut cends = Vec::with_capacity(n);
    let mut cvalid = Vec::with_capacity(n);
    for r in rows {
        keys::decode_into(keys::unmix(r.person), &mut ubytes);
        uends.push(ubytes.len() as u32);
        days.push(r.day);
        sess.push(r.sessions as i32);
        evs.push(r.events as i32);
        revs.push(r.revenue as i64);
        if r.top != u32::MAX {
            cbytes.extend_from_slice(&ctx.cat_names[r.top as usize]);
            cvalid.push(true);
        } else {
            cvalid.push(false);
        }
        cends.push(cbytes.len() as u32);
    }
    use parquet::{Column, Data};
    let cols = [
        Column { name: "user_id", data: Data::Str { bytes: &ubytes, ends: &uends, valid: None } },
        Column { name: "day", data: Data::I32 { values: &days, date: true } },
        Column { name: "sessions", data: Data::I32 { values: &sess, date: false } },
        Column { name: "events", data: Data::I32 { values: &evs, date: false } },
        Column { name: "revenue_cents", data: Data::I64 { values: &revs } },
        Column { name: "top_category", data: Data::Str { bytes: &cbytes, ends: &cends, valid: Some(&cvalid) } },
    ];
    let buf = parquet::encode(n, &cols);
    acc.lap(S_ENCODE);
    std::fs::write(path, &buf).unwrap_or_else(|e| fail(&format!("{}: {}", path.display(), e)));
    acc.lap(S_FWRITE);
}

fn write_campaign_daily(path: &Path, souts: &[SOut], ctx: &Ctx) {
    let ndays = ctx.ndays;
    let mut cred = vec![[0u64; 3]; (ctx.ncamp + 1) * ndays];
    let mut assist = vec![0u64; ctx.ncamp * ndays];
    for so in souts {
        for (a, b) in cred.iter_mut().zip(&so.cred) {
            a[0] += b[0];
            a[1] += b[1];
            a[2] += b[2];
        }
        for (a, b) in assist.iter_mut().zip(&so.assist) {
            *a += *b;
        }
    }
    let mut ibytes = Vec::new();
    let mut iends = Vec::new();
    let mut chbytes = Vec::new();
    let mut chends = Vec::new();
    let mut days = Vec::new();
    let (mut s, mut ps, mut rv, mut asv) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for c in 0..=ctx.ncamp {
        for d in 0..ndays {
            let cr = cred[c * ndays + d];
            let a = if c < ctx.ncamp { assist[c * ndays + d] } else { 0 };
            if cr[0] == 0 && a == 0 {
                continue;
            }
            if c < ctx.ncamp {
                ibytes.extend_from_slice(&ctx.camp_ids[c]);
                chbytes.extend_from_slice(&ctx.camp_channels[c]);
            } else {
                ibytes.extend_from_slice(b"organic");
                chbytes.extend_from_slice(b"organic");
            }
            iends.push(ibytes.len() as u32);
            chends.push(chbytes.len() as u32);
            days.push((ctx.first_day + d as i64) as i32);
            s.push(cr[0] as i64);
            ps.push(cr[1] as i64);
            rv.push(cr[2] as i64);
            asv.push(a as i64);
        }
    }
    use parquet::{Column, Data};
    let cols = [
        Column { name: "campaign_id", data: Data::Str { bytes: &ibytes, ends: &iends, valid: None } },
        Column { name: "channel", data: Data::Str { bytes: &chbytes, ends: &chends, valid: None } },
        Column { name: "day", data: Data::I32 { values: &days, date: true } },
        Column { name: "sessions", data: Data::I64 { values: &s } },
        Column { name: "purchasing_sessions", data: Data::I64 { values: &ps } },
        Column { name: "revenue_cents", data: Data::I64 { values: &rv } },
        Column { name: "assisted_sessions", data: Data::I64 { values: &asv } },
    ];
    parquet::write_file(path, days.len(), &cols).unwrap_or_else(|e| fail(&format!("{}: {}", path.display(), e)));
}

// ---------------------------------------------------------------- main

/// Build-time smoke test: the static binary runs, threads spawn, parsers work.
fn self_test() {
    assert_eq!(ts::parse_ts(b"2026-10-03T19:58:10-04:00"), Some(1791071890000));
    assert_eq!(keys::decode(keys::key_of(b"u42huhoug")), b"u42huhoug".to_vec());
    let mut n = 0;
    scan::for_each_record::<2, _>(b"a,\"b\"\nc,d\"\n", 0, 11, |f| {
        assert_eq!(f[1], b"b");
        n += 1
    });
    assert_eq!(n, 1);
    let r = par(4, |t| t * 2);
    assert_eq!(r, vec![0, 2, 4, 6]);
    eprintln!("self-test ok");
}

fn ceil_log2(x: usize) -> u32 {
    if x <= 1 {
        0
    } else {
        usize::BITS - (x - 1).leading_zeros()
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 2 && args[1] == "--self-test" {
        self_test();
        return;
    }
    let (data_dir, out_dir) = if args.len() >= 3 {
        (PathBuf::from(&args[args.len() - 2]), PathBuf::from(&args[args.len() - 1]))
    } else {
        (
            PathBuf::from(std::env::var("DATA_DIR").unwrap_or_else(|_| "/data".into())),
            PathBuf::from(std::env::var("OUTPUT_DIR").unwrap_or_else(|_| "/output".into())),
        )
    };
    if std::env::var("SS_DIAG").is_ok() {
        diag(&data_dir);
    }
    let threads = env_usize("NCPU")
        .filter(|&n| n > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(8, |n| n.get()))
        .min(64);
    let verbose = std::env::var("SS_VERBOSE").is_ok();
    let report = run(&data_dir, &out_dir, threads, verbose);
    if verbose {
        eprintln!("{}", report);
    }
    // Skip destructors (mmaps, big vectors): the OS reclaims everything.
    std::process::exit(0);
}

/// The whole pipeline on one dataset. Returns the verbose report line.
fn run(data_dir: &Path, out_dir: &Path, threads: usize, verbose: bool) -> String {
    sys::NOHUGE.store(std::env::var("SS_NOHUGE").is_ok(), Ordering::Relaxed);
    scan::USE_V1.store(std::env::var("SS_SCAN1").is_ok(), Ordering::Relaxed);
    LINKS_US.store(0, Ordering::Relaxed);
    let mut marks = Marks::new(verbose);
    // SS_PROF (with SS_VERBOSE): sub-phase timers; perturbs the timing a little
    let profile = verbose && std::env::var("SS_PROF").is_ok();
    prof::ON.store(profile, Ordering::Relaxed);
    if profile {
        prof::calibrate_start();
        name_session_slots();
    }

    // manifest + lookups
    let manifest = std::fs::read_to_string(data_dir.join("manifest.json")).unwrap_or_else(|e| fail(&format!("manifest: {}", e)));
    let ws = ts::parse_ts(manifest_value(&manifest, "window_start").as_bytes()).unwrap_or_else(|| fail("bad window_start"));
    let we = ts::parse_ts(manifest_value(&manifest, "window_end").as_bytes()).unwrap_or_else(|| fail("bad window_end"));
    if we <= ws || we - ws >= (1i64 << 31) {
        fail("unsupported window length (over 24 days)");
    }
    let ptext = read_file(&data_dir.join("products.csv"));
    let prows = table_rows(&ptext);
    // categories: distinct names (few), then ranked byte-wise
    let mut cat_list: Vec<&[u8]> = Vec::new();
    let mut prow_cat: Vec<u32> = Vec::with_capacity(prows.len());
    let mut last: Option<(&[u8], u32)> = None;
    for &(_, c) in &prows {
        let idx = match last {
            Some((lc, li)) if lc == c => li,
            _ => {
                let i = match cat_list.iter().position(|&x| x == c) {
                    Some(i) => i as u32,
                    None => {
                        cat_list.push(c);
                        if cat_list.len() > 255 {
                            fail("more than 255 categories is not supported");
                        }
                        (cat_list.len() - 1) as u32
                    }
                };
                last = Some((c, i));
                i
            }
        };
        prow_cat.push(idx);
    }
    if !cat_list.iter().any(|&c| c == b"unknown") {
        cat_list.push(b"unknown");
    }
    let mut sorted: Vec<(&[u8], usize)> = cat_list.iter().enumerate().map(|(i, &c)| (c, i)).collect();
    sorted.sort();
    let mut rank_of = vec![0u32; cat_list.len()];
    for (r, &(_, i)) in sorted.iter().enumerate() {
        rank_of[i] = r as u32;
    }
    let cat_names: Vec<Vec<u8>> = sorted.iter().map(|(c, _)| c.to_vec()).collect();
    let unknown_cat = sorted.iter().position(|(c, _)| *c == b"unknown").unwrap() as u32;
    let pentries: Vec<(&[u8], u16)> = prows.iter().zip(&prow_cat).map(|(&(p, _), &ci)| (p, rank_of[ci as usize] as u16)).collect();
    let ctext = read_file(&data_dir.join("campaigns.csv"));
    let crows = table_rows(&ctext);
    if crows.len() >= NONE as usize {
        fail("too many campaigns");
    }
    let centries: Vec<(&[u8], u16)> = crows.iter().enumerate().map(|(i, &(c, _))| (c, i as u16)).collect();
    marks.mark("lookups");
    let chunk = env_usize("SS_CHUNK").unwrap_or(2 << 20);
    let links_in = load_input(&data_dir.join("links"), chunk);
    let events_in = load_input(&data_dir.join("events"), chunk);
    let clicks_in = load_input(&data_dir.join("clicks"), chunk);
    marks.mark("mmap");
    let est_lines = events_in.bytes / 75 + 1;
    let pe_bits = env_usize("SS_PE_BITS").map(|b| b as u32).unwrap_or_else(|| ceil_log2(est_lines / 4096).clamp(2, 10));
    let ps_bits = env_usize("SS_PS_BITS").map(|b| b as u32).unwrap_or_else(|| ceil_log2(est_lines / 16384).clamp(2, 10));

    let first_day = ws.div_euclid(DAY);
    let ndays = ((we - 1).div_euclid(DAY) - first_day + 1) as usize;
    let ctx = Ctx {
        tsp: ts::TsParser::new(ws - 10 * DAY, we + 10 * DAY),
        tsw: std::env::var("SS_TSW").is_ok(),
        tsh: std::env::var("SS_TSH").is_ok(),
        avx2: use_avx2(),
        ws,
        we,
        ws_s: ws.div_euclid(1000),
        ws_mod_day: ws.rem_euclid(DAY) as u64,
        first_day,
        ndays,
        products: IdTable::build(&pentries),
        unknown_cat,
        cat_names,
        campaigns: IdTable::build(&centries),
        camp_ids: crows.iter().map(|r| r.0.to_vec()).collect(),
        camp_channels: crows.iter().map(|r| r.1.to_vec()).collect(),
        ncamp: crows.len(),
        pe_bits,
        ps_bits,
        threads,
    };
    marks.mark("setup");
    if std::env::var("SS_DECOMP").is_ok() {
        decomp_bench(&events_in, &links_in, &ctx);
        std::process::exit(0);
    }
    if std::env::var("SS_CDECOMP").is_ok() {
        cbench::click_decomp(&clicks_in, &links_in, &ctx);
        std::process::exit(0);
    }
    if std::env::var("SS_FLOOR").is_ok() {
        floor_bench(&events_in, &ctx);
        std::process::exit(0);
    }
    if let Ok(stage) = std::env::var("SS_BENCH") {
        for st in stage.split(',') {
            bench_stages(st, &events_in, &links_in, &ctx);
        }
        std::process::exit(0);
    }

    // Pipeline B unless disabled (SS_LEGACY) or unusable (location bits);
    // if a location index ever overflows, B's results are discarded and the
    // legacy pipeline runs from scratch.
    let mut pipeline = "B";
    let lb = if std::env::var("SS_LEGACY").is_ok() { None } else { LocBits::new(threads, ps_bits) };
    let mut b_result = None;
    if let Some(lb) = lb {
        let (lt, bouts) = parse_b(&links_in, &events_in, &clicks_in, &ctx, lb);
        marks.mark("parse");
        if bouts.iter().any(|o| o.overflow) {
            pipeline = "B-overflow->legacy";
            std::mem::forget(bouts);
            std::mem::forget(lt);
        } else {
            dedup_b(&bouts, &ctx, lb);
            marks.mark("dedup");
            b_result = Some((lt, bouts));
        }
    } else {
        pipeline = "legacy";
    }
    let mut legacy_bufs = None;
    if b_result.is_none() {
        let (lt, pouts) = parse_all(&links_in, &events_in, &clicks_in, &ctx);
        marks.mark("parse");
        let (ebufs, cbufs): (Vec<Parts<ERec>>, Vec<Parts<CRec>>) = pouts.into_iter().map(|po| (po.e.0, po.c)).unzip();
        let sbufs = dedup(&ebufs, &lt, &ctx);
        marks.mark("dedup");
        // Not freed: unmapping costs more than it saves; the process exits soon.
        std::mem::forget(ebufs);
        legacy_bufs = Some((lt, sbufs, cbufs));
    }
    let (sbufs, cbufs): (Vec<&Parts<SRec>>, Vec<&Parts<CRec>>) = match (&b_result, &legacy_bufs) {
        (Some((_, bouts)), _) => (bouts.iter().map(|o| &o.s).collect(), bouts.iter().map(|o| &o.c).collect()),
        (None, Some((_, sb, cb))) => (sb.iter().collect(), cb.iter().collect()),
        _ => unreachable!(),
    };

    let ud = out_dir.join("user_daily");
    let cd = out_dir.join("campaign_daily");
    std::fs::create_dir_all(&ud).unwrap_or_else(|e| fail(&format!("{}: {}", ud.display(), e)));
    std::fs::create_dir_all(&cd).unwrap_or_else(|e| fail(&format!("{}: {}", cd.display(), e)));
    let souts = sessions(&sbufs, &cbufs, &ctx, &ud);
    marks.mark("sessions");
    write_campaign_daily(&cd.join("part-0.parquet"), &souts, &ctx);
    marks.mark("write");
    if std::env::var("SS_FREE").is_ok() {
        // Experiment: release arena pages on all threads (MADV_DONTNEED takes
        // the mmap lock shared) so that the kernel's single-threaded exit
        // teardown has less to free.
        let mut pieces = Vec::new();
        let mut regions = sys::arenas();
        if std::env::var("SS_FREE").map_or(false, |v| v != "arenas") {
            for inp in [&links_in, &events_in, &clicks_in] {
                for m in &inp.maps {
                    let b = m.bytes();
                    if !b.is_empty() {
                        regions.push((b.as_ptr() as usize, b.len()));
                    }
                }
            }
        }
        for (a, l) in regions {
            let mut o = 0;
            while o < l {
                let n = (l - o).min(64 << 20);
                pieces.push((a + o, n));
                o += n;
            }
        }
        let nx = AtomicUsize::new(0);
        par(threads, |_| loop {
            let i = nx.fetch_add(1, Ordering::Relaxed);
            let Some(&(a, l)) = pieces.get(i) else { break };
            sys::release(a, l);
        });
        marks.mark("free");
    }

    if verbose {
        let rows: usize = souts.iter().map(|s| s.nrows).sum();
        return format!(
            "sessions: {} | pipeline={} links={:.1}ms threads={} avx2={} pe_bits={} ps_bits={} user_rows={} fallback_keys={} peak_rss={}MB {} | sub:{}",
            marks.report(),
            pipeline,
            LINKS_US.load(Ordering::Relaxed) as f64 / 1e3,
            threads,
            ctx.avx2,
            pe_bits,
            ps_bits,
            rows,
            keys::any_fallback(),
            sys::peak_rss_mb(),
            sys::anon_summary(),
            prof::report()
        );
    }
    String::new()
}

/// Diagnostic helper: writes `copies` independent copies of the dataset into
/// `rep`: in copy k every user / event / click ID gets the suffix char
/// 'A' + k (inside its quotes), so copies share no IDs.
fn build_replica(data_dir: &Path, rep: &Path, copies: usize) {
    let _ = std::fs::remove_dir_all(rep);
    let mut jobs = Vec::new();
    for (sub, idx) in [("events", &[0usize, 1][..]), ("clicks", &[0, 1][..]), ("links", &[0, 1][..])] {
        std::fs::create_dir_all(rep.join(sub)).unwrap_or_else(|e| fail(&format!("diag mkdir: {}", e)));
        for f in list_csv(&data_dir.join(sub)) {
            for k in 0..copies {
                jobs.push((f.clone(), rep.join(sub).join(format!("r{:02}_{}", k, f.file_name().unwrap().to_string_lossy())), idx, k));
            }
        }
    }
    for f in ["manifest.json", "products.csv", "campaigns.csv"] {
        let _ = std::fs::copy(data_dir.join(f), rep.join(f));
    }
    let next = AtomicUsize::new(0);
    par(8, |_| loop {
        let i = next.fetch_add(1, Ordering::Relaxed);
        let Some((src, dst, idx, k)) = jobs.get(i) else { break };
        let text = read_file(src);
        let suffix = b'A' + *k as u8;
        let mut out = Vec::with_capacity(text.len() + text.len() / 20);
        let mut first = true;
        for line in text.split_inclusive(|&b| b == b'\n') {
            if first {
                out.extend_from_slice(line); // header
                first = false;
                continue;
            }
            let mut field = 0usize;
            let mut start = 0usize;
            for (j, &c) in line.iter().enumerate() {
                if c == b',' || c == b'\n' || j + 1 == line.len() {
                    let end = if c == b',' || c == b'\n' { j } else { j + 1 };
                    out.extend_from_slice(&line[start..end]);
                    if idx.contains(&field) && end > start {
                        if line[end - 1] == b'"' && end - start >= 2 {
                            out.pop();
                            out.push(suffix);
                            out.push(b'"');
                        } else {
                            out.push(suffix);
                        }
                    }
                    if end == j {
                        out.push(c);
                    }
                    field += 1;
                    start = j + 1;
                }
            }
        }
        std::fs::write(dst, &out).unwrap_or_else(|e| fail(&format!("diag write: {}", e)));
    });
}

/// SS_DIAG=1 (diagnostic submissions only): machine summary, then the
/// pipeline under several configurations on the dataset and on a 20x
/// replica of it built from symlinks in SCRATCH_DIR; exits with status 3 so
/// the scorer reports stderr. Never used for a real run.
fn diag(data_dir: &Path) -> ! {
    let cpu = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let model = cpu.lines().find(|l| l.starts_with("model name")).unwrap_or("?").to_string();
    let flags = cpu.lines().find(|l| l.starts_with("flags")).unwrap_or("");
    let has = |f: &str| flags.split_whitespace().any(|x| x == f);
    eprintln!(
        "diag: {} | ncpu={} avx2={} avx512bw={} bmi2={} thp={}",
        model,
        std::thread::available_parallelism().map_or(0, |n| n.get()),
        has("avx2"),
        has("avx512bw"),
        has("bmi2"),
        std::fs::read_to_string("/sys/kernel/mm/transparent_hugepage/enabled").unwrap_or_default().trim()
    );
    let scratch = PathBuf::from(std::env::var("SCRATCH_DIR").unwrap_or_else(|_| "/scratch".into()));
    let rep = scratch.join("rep20");
    let t = Instant::now();
    build_replica(data_dir, &rep, 20);
    eprintln!("diag replica: {:.0}ms", t.elapsed().as_secs_f64() * 1e3);
    let short = |r: &str| -> String {
        let mut o = String::new();
        for key in ["total=", "parse=", "dedup=", "sessions=", "write=", "links=", "pipeline=", "user_rows=", "peak_rss="] {
            if let Some(i) = r.find(key) {
                let rest = &r[i..];
                let end = rest.find(|c: char| c == ' ' || c == '(').unwrap_or(rest.len());
                o.push_str(&rest[..end]);
                if let Some(j) = rest.find("(cpu ") {
                    if j == end {
                        let c = &rest[j + 5..];
                        o.push_str(&format!("/c{}", &c[..c.find('/').unwrap_or(0)]));
                    }
                }
                o.push(' ');
            }
        }
        o
    };
    let configs: [(&str, &[(&str, &str)], usize, bool); 10] = [
        ("dev-B8", &[], 8, false),
        ("x20-B8", &[], 8, true),
        ("x20-B8", &[], 8, true),
        ("x20-L8", &[("SS_LEGACY", "1")], 8, true),
        ("x20-B4", &[], 4, true),
        ("x20-B6", &[], 6, true),
        ("x20-B8-nowc", &[("SS_NOWC", "1")], 8, true),
        ("x20-B8-tsw", &[("SS_TSW", "1")], 8, true),
        ("x20-L8", &[("SS_LEGACY", "1")], 8, true),
        ("x20-B8", &[], 8, true),
    ];
    for (i, (name, env, threads, x20)) in configs.iter().enumerate() {
        for (k, v) in env.iter() {
            std::env::set_var(k, v);
        }
        let out = scratch.join(format!("diag_out{}", i));
        let _ = std::fs::remove_dir_all(&out);
        let t = Instant::now();
        let r = run(if *x20 { &rep } else { data_dir }, &out, *threads, true);
        eprintln!("diag {:14} wall={:.0}ms {}", name, t.elapsed().as_secs_f64() * 1e3, short(&r));
        for (k, _) in env.iter() {
            std::env::remove_var(k);
        }
        sys::release_all();
    }
    std::process::exit(3)
}


// ---------------------------------------------------------------- floor benchmarks
// SS_FLOOR=1: lower bounds for field splitting over events/, single thread
// and on NCPU threads (thread-time per line): stream (64-byte loads),
// masks (comma/newline/quote masks + popcounts; AVX2 and AVX-512BW), and the
// full split step (scan).

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn floor_stream(b: &[u8]) -> u64 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_si256();
    let n = b.len() / 64 * 64;
    let mut i = 0;
    while i < n {
        let x = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
        let y = _mm256_loadu_si256(b.as_ptr().add(i + 32) as *const __m256i);
        acc = _mm256_add_epi64(acc, _mm256_add_epi64(x, y));
        i += 64;
    }
    let mut o = [0u64; 4];
    _mm256_storeu_si256(o.as_mut_ptr() as *mut __m256i, acc);
    o[0] ^ o[1] ^ o[2] ^ o[3]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,popcnt")]
unsafe fn floor_masks_avx2(b: &[u8]) -> u64 {
    use std::arch::x86_64::*;
    let vc = _mm256_set1_epi8(b',' as i8);
    let vn = _mm256_set1_epi8(b'\n' as i8);
    let vq = _mm256_set1_epi8(b'"' as i8);
    let n = b.len() / 64 * 64;
    let mut acc = 0u64;
    let mut i = 0;
    while i < n {
        let x = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
        let y = _mm256_loadu_si256(b.as_ptr().add(i + 32) as *const __m256i);
        let c = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(x, vc)) as u32 as u64) | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(y, vc)) as u32 as u64) << 32);
        let l = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(x, vn)) as u32 as u64) | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(y, vn)) as u32 as u64) << 32);
        let q = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(x, vq)) as u32 as u64) | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(y, vq)) as u32 as u64) << 32);
        acc += (c.count_ones() + l.count_ones() + q.count_ones()) as u64;
        i += 64;
    }
    acc
}

/// AVX-512BW masks via inline asm (the intrinsics are not stable in the
/// pinned toolchain). Bench only; ends with vzeroupper.
#[cfg(target_arch = "x86_64")]
unsafe fn floor_masks_avx512(b: &[u8]) -> u64 {
    let n = b.len() / 64;
    if n == 0 {
        return 0;
    }
    let acc: u64;
    std::arch::asm!(
        "mov {t:e}, 44",
        "vpbroadcastb zmm1, {t:e}",
        "mov {t:e}, 10",
        "vpbroadcastb zmm2, {t:e}",
        "mov {t:e}, 34",
        "vpbroadcastb zmm3, {t:e}",
        "xor {acc}, {acc}",
        "2:",
        "vmovdqu64 zmm0, [{p}]",
        "vpcmpeqb k1, zmm0, zmm1",
        "vpcmpeqb k2, zmm0, zmm2",
        "vpcmpeqb k3, zmm0, zmm3",
        "kmovq {t}, k1",
        "popcnt {t}, {t}",
        "add {acc}, {t}",
        "kmovq {t}, k2",
        "popcnt {t}, {t}",
        "add {acc}, {t}",
        "kmovq {t}, k3",
        "popcnt {t}, {t}",
        "add {acc}, {t}",
        "add {p}, 64",
        "dec {n}",
        "jnz 2b",
        "vzeroupper",
        p = inout(reg) b.as_ptr() => _,
        n = inout(reg) n => _,
        t = out(reg) _,
        acc = out(reg) acc,
        out("xmm0") _, out("xmm1") _, out("xmm2") _, out("xmm3") _,
        out("k1") _, out("k2") _, out("k3") _,
        options(nostack, readonly),
    );
    acc
}

fn floor_one(stage: &str, b: &[u8], task: Task, ctx: &Ctx) -> u64 {
    let s = &b[task.start..task.end];
    #[cfg(target_arch = "x86_64")]
    unsafe {
        match stage {
            "stream" => return floor_stream(s),
            "masks-avx2" => return floor_masks_avx2(s),
            "masks-avx512" => return floor_masks_avx512(s),
            "scan" => {
                let mut acc = 0u64;
                if ctx.avx2 {
                    floor_scan_avx2(b, task, &mut acc);
                } else {
                    scan::for_each_record_x::<7, _, false>(b, task.start, task.end, |f| acc += f[3].len() as u64);
                }
                return acc;
            }
            _ => {}
        }
    }
    let _ = (s, ctx, stage);
    fail("unknown floor stage")
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
unsafe fn floor_scan_avx2(b: &[u8], task: Task, acc: &mut u64) {
    scan::for_each_record_x::<7, _, true>(b, task.start, task.end, |f| *acc += f[3].len() as u64);
}

fn floor_bench(ev: &Input, ctx: &Ctx) {
    // event lines (for per-line figures)
    let mut lines = 0usize;
    for t in &ev.tasks {
        let b = ev.maps[t.file as usize].bytes();
        lines += b[t.start..t.end].iter().filter(|&&c| c == b'\n').count();
    }
    let has512 = {
        #[cfg(target_arch = "x86_64")]
        {
            std::arch::is_x86_feature_detected!("avx512bw")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    };
    for stage in ["stream", "masks-avx2", "masks-avx512", "scan"] {
        if stage == "masks-avx512" && !has512 {
            continue;
        }
        for &nt in &[1usize, ctx.threads] {
            let mut best = (f64::MAX, 0f64);
            for _ in 0..3 {
                let next = AtomicUsize::new(0);
                PHASE_CPU_NS.store(0, Ordering::Relaxed);
                let t = Instant::now();
                let accs = par(nt, |_| {
                    let mut a = 0u64;
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(&task) = ev.tasks.get(i) else { break };
                        a = a.wrapping_add(floor_one(stage, ev.maps[task.file as usize].bytes(), task, ctx));
                    }
                    a
                });
                let wall = t.elapsed().as_secs_f64() * 1e3;
                let cpu = PHASE_CPU_NS.swap(0, Ordering::Relaxed) as f64 / 1e6;
                std::hint::black_box(accs);
                if wall < best.0 {
                    best = (wall, cpu);
                }
            }
            eprintln!(
                "floor {:13} threads={} wall={:7.1}ms thread-time/line={:5.1}ns ({} lines, {} MB)",
                stage,
                nt,
                best.0,
                best.1 * 1e6 / lines as f64,
                lines,
                ev.bytes >> 20
            );
        }
    }
}


// ---------------------------------------------------------------- parse decomposition
// SS_DECOMP=1: cumulative event-parse stages, single thread and NCPU threads,
// thread-time per event line (each stage includes everything before it).

#[inline(always)]
fn decomp_task<const AVX2: bool>(stage: u32, b: &[u8], task: Task, ctx: &Ctx, sink: &mut BSink) -> u64 {
    let mut acc = 0u64;
    let end = b.as_ptr() as usize + b.len();
    match stage {
        // split only
        0 => scan::for_each_record_x::<7, _, AVX2>(b, task.start, task.end, |f| acc += f[3].len() as u64),
        // + event id, + user id, + ts, + ingest
        1..=4 => scan::for_each_record_x::<7, _, AVX2>(b, task.start, task.end, |f| {
            if f[6].as_ptr() as usize + 32 > end || f[0].is_empty() || f[1].is_empty() {
                return;
            }
            acc = acc.wrapping_add(keys::mix(unsafe { keys::key_of_wide(f[0]) }));
            if stage >= 2 {
                acc = acc.wrapping_add(unsafe { keys::key_of_wide(f[1]) });
            }
            if stage >= 3 {
                acc = acc.wrapping_add(ctx.tsp.parse(f[2]).unwrap_or(0) as u64);
            }
            if stage >= 4 {
                acc = acc.wrapping_add(ctx.tsp.parse(f[3]).unwrap_or(0) as u64);
            }
        }),
        // + type / product / price / validation (full line logic, null sink)
        5 => {
            let mut ns = NullSink(0);
            event_task::<AVX2, _>(b, task, ctx, &mut ns, end);
            acc = ns.0;
        }
        // + record scatter (dedup + session records, no person lookup), + lookup
        _ => {
            event_task::<AVX2, _>(b, task, ctx, sink, end);
            sink.flush();
        }
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
unsafe fn decomp_task_avx2(stage: u32, b: &[u8], task: Task, ctx: &Ctx, sink: &mut BSink) -> u64 {
    decomp_task::<true>(stage, b, task, ctx, sink)
}

fn decomp_bench(ev: &Input, links_in: &Input, ctx: &Ctx) {
    let mut lines = 0usize;
    for t in &ev.tasks {
        let b = ev.maps[t.file as usize].bytes();
        lines += b[t.start..t.end].iter().filter(|&&c| c == b'\n').count();
    }
    let lb = LocBits::new(ctx.threads, ctx.ps_bits).unwrap();
    let empty = Input { maps: Vec::new(), tasks: Vec::new(), bytes: 0 };
    let (lt, _) = parse_b(links_in, &empty, &empty, ctx, lb);
    let keep = sys::arenas().len();
    let pe = 1usize << ctx.pe_bits;
    let ps = 1usize << ctx.ps_bits;
    let est = ev.bytes / 72 + 1;
    let names = ["split", "+event_id", "+user_id", "+ts", "+ingest", "+type/product/price/valid", "+scatter(no lookup)", "+person lookup"];
    for stage in 0..8u32 {
        for &nt in &[1usize, ctx.threads] {
            let mut best = (f64::MAX, 0f64);
            for _ in 0..3 {
                let next = AtomicUsize::new(0);
                PHASE_CPU_NS.store(0, Ordering::Relaxed);
                let t = Instant::now();
                let accs = par(nt, |ti| {
                    let mut sink = BSink {
                        a: Parts::with_wc(pe, est / (nt * pe) * 5 / 4 + 32, true),
                        s: Parts::with_wc(ps, est / (nt * ps) * 5 / 4 + 32, true),
                        ring: [Pending::default(); BRING],
                        rh: 0,
                        rn: 0,
                        tbase: ((ti as u64) << (lb.qbits + lb.ibits)) as u32,
                        overflow: false,
                        lt: &lt,
                        pe_bits: ctx.pe_bits,
                        ps_bits: ctx.ps_bits,
                        lb,
                        nolookup: stage == 6,
                        cache: PCache::new(),
                    };
                    let mut a = 0u64;
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(&task) = ev.tasks.get(i) else { break };
                        let b = ev.maps[task.file as usize].bytes();
                        a = a.wrapping_add(if ctx.avx2 {
                            #[cfg(target_arch = "x86_64")]
                            unsafe {
                                decomp_task_avx2(stage, b, task, ctx, &mut sink)
                            }
                            #[cfg(not(target_arch = "x86_64"))]
                            0
                        } else {
                            decomp_task::<false>(stage, b, task, ctx, &mut sink)
                        });
                    }
                    sink.a.finish();
                    sink.s.finish();
                    a ^ sink.a.total() as u64
                });
                let wall = t.elapsed().as_secs_f64() * 1e3;
                let cpu = PHASE_CPU_NS.swap(0, Ordering::Relaxed) as f64 / 1e6;
                std::hint::black_box(accs);
                sys::release_from(keep);
                if wall < best.0 {
                    best = (wall, cpu);
                }
            }
            eprintln!(
                "decomp {:28} threads={} wall={:7.1}ms thread-time/line={:6.1}ns",
                names[stage as usize],
                nt,
                best.0,
                best.1 * 1e6 / lines as f64
            );
        }
    }
    eprintln!("decomp: {} event lines, {} MB", lines, ev.bytes >> 20);
}

/// Null sink for the stage benchmark.
struct NullSink(u64);
impl EvSink for NullSink {
    #[inline(always)]
    fn copy(&mut self, eid: u64, ingest: u32, valid: Option<(u64, u32, u32)>) {
        self.0 = self.0.wrapping_add(eid ^ ingest as u64);
        if let Some((u, t, p)) = valid {
            self.0 = self.0.wrapping_add(u ^ t as u64 ^ p as u64);
        }
    }
}

#[inline(always)]
fn bench_task<const AVX2: bool>(stage: &str, b: &[u8], task: Task, ctx: &Ctx, parts: &mut (Parts<ERec>, u32), a16: &mut Parts<ARec>) -> u64 {
    let mut acc = 0u64;
    let end = b.as_ptr() as usize + b.len();
    match stage {
        "touch" => {
            let mut s = task.start;
            while s < task.end {
                acc += b[s] as u64;
                s += 4096;
            }
        }
        "scan" => scan::for_each_record_x::<7, _, AVX2>(b, task.start, task.end, |f| acc += f[3].len() as u64),
        "ts" => scan::for_each_record_x::<7, _, AVX2>(b, task.start, task.end, |f| {
            acc += ctx.tsp.parse(f[3]).unwrap_or(0) as u64 + ctx.tsp.parse(f[2]).unwrap_or(0) as u64
        }),
        "tsw" => scan::for_each_record_x::<7, _, AVX2>(b, task.start, task.end, |f| {
            if f[6].as_ptr() as usize + 32 <= end {
                acc += unsafe { ctx.tsp.parse_wide_f1(f[3]).unwrap_or(0) as u64 + ctx.tsp.parse_wide(f[2]).unwrap_or(0) as u64 }
            }
        }),
        "keys" => scan::for_each_record_x::<7, _, AVX2>(b, task.start, task.end, |f| {
            if !f[0].is_empty() && !f[1].is_empty() {
                acc += keys::key_of(f[0]) ^ keys::key_of(f[1])
            }
        }),
        "keysw" => scan::for_each_record_x::<7, _, AVX2>(b, task.start, task.end, |f| {
            if !f[0].is_empty() && !f[1].is_empty() && f[6].as_ptr() as usize + 32 <= end {
                acc += unsafe { keys::key_of_wide(f[0]) ^ keys::key_of_wide(f[1]) }
            }
        }),
        "line" => {
            let mut ns = NullSink(0);
            event_task::<AVX2, _>(b, task, ctx, &mut ns, end);
            acc = ns.0;
        }
        "ev" | "evre" => event_task::<AVX2, _>(b, task, ctx, parts, end),
        "push28" | "push16" | "push16wc" => {
            // one record per line into 256 partitions chosen by a hash
            let mut i = task.start as u64;
            scan::for_each_record_x::<7, _, AVX2>(b, task.start, task.end, |f| {
                i = i.wrapping_add(f[3].len() as u64 + 1);
                let h = keys::mix(i);
                if stage == "push28" {
                    parts.0.push((h >> 56) as usize, ERec { eid: h, user: 0, ingest: 0, ts: 0, pay: 0 });
                } else {
                    a16.push((h >> 56) as usize, ARec { eid: h, ingest: 0, loc: 0 });
                }
            })
        }
        _ => fail("unknown SS_BENCH stage"),
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
unsafe fn bench_task_avx2(stage: &str, b: &[u8], task: Task, ctx: &Ctx, parts: &mut (Parts<ERec>, u32), a16: &mut Parts<ARec>) -> u64 {
    bench_task::<true>(stage, b, task, ctx, parts, a16)
}

/// SS_BENCH=STAGE: single-threaded stage benchmark over events/, best of 3
/// passes (the first pass also maps the pages in).
fn bench_stages(stage: &str, ev: &Input, links_in: &Input, ctx: &Ctx) {
    let est = ev.bytes / 70 + 1;
    let mut best = f64::MAX;
    let mut acc = 0u64;
    let lines: usize = ev.tasks.len();
    if stage == "evb" || stage == "evbre" {
        return bench_b(stage, ev, links_in, ctx);
    }
    let mut reused = (Parts::<ERec>::new(1 << ctx.pe_bits, est / (1 << ctx.pe_bits) * 3 / 2 + 64), ctx.pe_bits);
    for _ in 0..3 {
        let mut fresh;
        let parts = if stage == "evre" {
            reused.0.reset();
            &mut reused
        } else {
            fresh = (Parts::<ERec>::new(1 << ctx.pe_bits, est / (1 << ctx.pe_bits) * 3 / 2 + 64), ctx.pe_bits);
            &mut fresh
        };
        let mut a16 = Parts::<ARec>::with_wc(256, est / 256 * 3 / 2 + 64, stage == "push16wc");
        let t = Instant::now();
        for task in &ev.tasks {
            let b = ev.maps[task.file as usize].bytes();
            acc = acc.wrapping_add(if ctx.avx2 {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    bench_task_avx2(stage, b, *task, ctx, parts, &mut a16)
                }
                #[cfg(not(target_arch = "x86_64"))]
                0
            } else {
                bench_task::<false>(stage, b, *task, ctx, parts, &mut a16)
            });
        }
        a16.finish();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
        acc = acc.wrapping_add(parts.0.total() as u64);
    }
    eprintln!("bench {:6} {:8.1} ms  ({} tasks, {} MB, avx2={}, acc {})", stage, best, lines, ev.bytes >> 20, ctx.avx2, acc % 1000);
}

/// Pipeline-B parse sink (with person lookups), single thread.
fn bench_b(stage: &str, ev: &Input, links_in: &Input, ctx: &Ctx) {
    let lb = LocBits::new(1, ctx.ps_bits).unwrap();
    let empty = Input { maps: Vec::new(), tasks: Vec::new(), bytes: 0 };
    let t0 = Instant::now();
    let (lt, _) = parse_b(links_in, &empty, &empty, ctx, lb);
    let link_ms = t0.elapsed().as_secs_f64() * 1e3;
    let pe = 1usize << ctx.pe_bits;
    let ps = 1usize << ctx.ps_bits;
    let est = ev.bytes / 70 + 1;
    let mk = |lt| BSink {
        a: Parts::new(pe, est / pe * 3 / 2 + 64),
        s: Parts::new(ps, est / ps * 3 / 2 + 64),
        ring: [Pending::default(); BRING],
        rh: 0,
        rn: 0,
        tbase: 0,
        overflow: false,
        lt,
        pe_bits: ctx.pe_bits,
        ps_bits: ctx.ps_bits,
        lb,
        nolookup: std::env::var("SS_BENCH_NOLOOKUP").is_ok(),
        cache: PCache::new(),
    };
    let mut reused = mk(&lt);
    let mut best = f64::MAX;
    for _ in 0..3 {
        let mut fresh;
        let sink = if stage == "evbre" {
            reused.a.reset();
            reused.s.reset();
            &mut reused
        } else {
            fresh = mk(&lt);
            &mut fresh
        };
        let t = Instant::now();
        for task in &ev.tasks {
            let b = ev.maps[task.file as usize].bytes();
            let end = b.as_ptr() as usize + b.len();
            if ctx.avx2 {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    event_task_avx2(b, *task, ctx, sink, end)
                }
            } else {
                event_task::<false, _>(b, *task, ctx, sink, end)
            }
            sink.flush();
        }
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    eprintln!("bench {:6} {:8.1} ms  (links {:.1} ms)", stage, best, link_ms);
}
