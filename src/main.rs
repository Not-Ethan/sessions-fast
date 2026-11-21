//! Shopping Sessions: native pipeline. See README.md for the design.

mod keys;
mod links;
mod parts;
mod lookup;
mod parquet;
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

/// A surviving event, routed by person.
#[derive(Clone, Copy)]
#[repr(C)]
struct SRec {
    person: u64,
    ts: u32,
    pay: u32,
}

/// A valid click, routed by person. ts is relative to window_start - 7 days.
#[derive(Clone, Copy)]
#[repr(C)]
struct CRec {
    person: u64,
    ck: u64,
    ts: u32,
    camp: u32,
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

#[inline(always)]
fn part(k: u64, bits: u32) -> usize {
    (keys::mix(k) >> (64 - bits)) as usize
}

// ---------------------------------------------------------------- context

struct Ctx {
    tsp: ts::TsParser,
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

fn par<R: Send>(threads: usize, f: impl Fn(usize) -> R + Sync) -> Vec<R> {
    std::thread::scope(|s| {
        let f = &f;
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                s.spawn(move || {
                    let r = f(t);
                    let ns = sys::thread_cpu_ns();
                    PHASE_CPU_NS.fetch_add(ns, Ordering::Relaxed);
                    PHASE_MAX_NS.fetch_max(ns, Ordering::Relaxed);
                    r
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
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
    items: Vec<(&'static str, f64, f64, f64, i64)>,
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
        self.items.push((name, (now - self.prev).as_secs_f64() * 1e3, cpu, mx, df));
        self.prev = now;
    }
    fn report(&self) -> String {
        let mut s = format!("total={:.1}ms |", self.t0.elapsed().as_secs_f64() * 1e3);
        for (n, w, c, m, f) in &self.items {
            s.push_str(&format!(" {}={:.1}ms(cpu {:.0}/max {:.0} flt {})", n, w, c, m, f));
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
        let _ = cell.set(LinkTable::with_pairs(total));
    }
    bar.wait();
    let lt = cell.get().unwrap();
    {
        let v = pairs[t].lock().unwrap();
        const AHEAD: usize = 8;
        for k in 0..v.len() {
            if k + AHEAD < v.len() {
                lt.prefetch(v[k + AHEAD].0);
                lt.prefetch(v[k + AHEAD].1);
            }
            lt.link(v[k].0, v[k].1);
        }
    }
    bar.wait();
    let nb = lt.finalize_blocks();
    loop {
        let b = f1.fetch_add(1, Ordering::Relaxed);
        if b >= nb {
            break;
        }
        lt.finalize_roots(b);
    }
    bar.wait();
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
    e: Parts<ERec>,
    c: Parts<CRec>,
}

/// One chunk of events/: parse, validate, route by event_id.
#[inline(always)]
fn event_task<const AVX2: bool>(b: &[u8], task: Task, ctx: &Ctx, e: &mut Parts<ERec>) {
    let (ws, we) = (ctx.ws, ctx.we);
    scan::for_each_record_x::<7, _, AVX2>(b, task.start, task.end, |f| {
        if f[0].is_empty() {
            return;
        }
        let Some(ing) = ctx.tsp.parse(f[3]) else { return };
        let eid = keys::mix(keys::key_of(f[0]));
        let mut r = ERec { eid, ingest: ingest_rel(ing, ctx.ws_s), ..Default::default() };
        'v: {
            if f[1].is_empty() || f[5].is_empty() {
                break 'v;
            }
            let Some(t) = ctx.tsp.parse(f[2]) else { break 'v };
            if t < ws || t >= we {
                break 'v;
            }
            let ty = f[4];
            let mut pay = 0u32;
            if ty == b"purchase" {
                let Some(price) = ts::parse_price(f[6]) else { break 'v };
                let mut cat = ctx.products.get(f[5]) as u32;
                if cat == NONE as u32 {
                    cat = ctx.unknown_cat;
                }
                pay = price | (cat << 24);
            } else if ty != b"view" && ty != b"cart" {
                break 'v;
            }
            if ing - t > LATE {
                break 'v;
            }
            r.user = keys::key_of(f[1]);
            r.ts = (t - ws) as u32 | OK_BIT;
            r.pay = pay;
        }
        e.push((eid >> (64 - ctx.pe_bits)) as usize, r);
    });
}

/// One chunk of clicks/: parse, validate, resolve the person, route by person.
#[inline(always)]
fn click_task<const AVX2: bool>(b: &[u8], task: Task, ctx: &Ctx, lt: &LinkTable, c: &mut Parts<CRec>) {
    let we = ctx.we;
    let cws = ctx.ws - WEEK;
    // Person lookups go through a small ring so the link-table slot can be
    // prefetched a few clicks ahead.
    const RING: usize = 16;
    let mut ring = [CRec { person: 0, ck: 0, ts: 0, camp: 0 }; RING];
    let mut rh = 0usize;
    let mut rn = 0usize;
    let emit = |c: &mut Parts<CRec>, mut r: CRec| {
        r.person = lt.person_of(r.person);
        c.push(part(r.person, ctx.ps_bits), r);
    };
    scan::for_each_record_x::<4, _, AVX2>(b, task.start, task.end, |f| {
        if f[0].is_empty() || f[1].is_empty() {
            return;
        }
        let Some(t) = ctx.tsp.parse(f[3]) else { return };
        if t < cws || t >= we {
            return;
        }
        let camp = ctx.campaigns.get(f[2]);
        if camp == NONE {
            return;
        }
        let user = keys::key_of(f[1]);
        lt.prefetch(user);
        // `person` holds the user key until the lookup in `emit`
        let r = CRec { person: user, ck: keys::key_of(f[0]), ts: (t - cws) as u32, camp: camp as u32 };
        if rn == RING {
            emit(c, ring[rh]);
            ring[rh] = r;
            rh = (rh + 1) % RING;
        } else {
            ring[(rh + rn) % RING] = r;
            rn += 1;
        }
    });
    for k in 0..rn {
        emit(c, ring[(rh + k) % RING]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
unsafe fn event_task_avx2(b: &[u8], task: Task, ctx: &Ctx, e: &mut Parts<ERec>) {
    event_task::<true>(b, task, ctx, e)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
unsafe fn click_task_avx2(b: &[u8], task: Task, ctx: &Ctx, lt: &LinkTable, c: &mut Parts<CRec>) {
    click_task::<true>(b, task, ctx, lt, c)
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
        let mut out = ParseOut { e: Parts::new(pe, cap_e), c: Parts::new(ps, cap_c) };
        let mut rbuf: Vec<u8> = if use_read { vec![0u8; max_task] } else { Vec::new() };
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
                if use_read {
                    let n = task.end - task.start;
                    ev.maps[task.file as usize].read_at(&mut rbuf[..n], task.start);
                    b = &rbuf[..n];
                    task = Task { file: task.file, start: 0, end: n };
                }
                if avx2 {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        event_task_avx2(b, task, ctx, &mut out.e)
                    }
                } else {
                    event_task::<false>(b, task, ctx, &mut out.e)
                }
            } else {
                while !ready.load(Ordering::Acquire) {
                    std::thread::yield_now();
                }
                let lt = cell.get().unwrap();
                let mut task = cl.tasks[i - ev.tasks.len()];
                let mut b = cl.maps[task.file as usize].bytes();
                if use_read {
                    let n = task.end - task.start;
                    cl.maps[task.file as usize].read_at(&mut rbuf[..n], task.start);
                    b = &rbuf[..n];
                    task = Task { file: task.file, start: 0, end: n };
                }
                if avx2 {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        click_task_avx2(b, task, ctx, lt, &mut out.c)
                    }
                } else {
                    click_task::<false>(b, task, ctx, lt, &mut out.c)
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
                let person = lt.person_of(user);
                out.push(part(person, ctx.ps_bits), SRec { person, ts, pay });
            }
        }
        out
    })
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
                let mut i = keys::mix(k) as usize & self.mask;
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
        let mut i = keys::mix(k) as usize & self.mask;
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
        let mut i = keys::mix(k) as usize & self.mask;
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
    }

    fn partition(
        &mut self,
        q: usize,
        sbufs: &[Parts<SRec>],
        cbufs: &[Parts<CRec>],
        ctx: &Ctx,
        out: &mut SOut,
    ) {
        let n: usize = sbufs.iter().map(|t| t.len(q)).sum();
        if n == 0 {
            return;
        }
        self.map.reset(n / 8 + 16);
        self.persons.clear();
        self.ridx.clear();
        for t in sbufs {
            for sl in t.get(q) {
                for r in sl {
                    let (idx, new) = self.map.insert(r.person);
                    if new {
                        self.persons.push(r.person);
                    }
                    self.ridx.push(idx);
                }
            }
        }
        let np = self.persons.len();
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
                    let i = self.ridx[k] as usize;
                    let c = self.cursor[i] as usize;
                    self.ev[c] = ((r.ts as u64) << 32) | r.pay as u64;
                    self.cursor[i] += 1;
                    k += 1;
                }
            }
        }
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
        let mut ev = std::mem::take(&mut self.ev);
        let mut clicks = std::mem::take(&mut self.clicks);
        for p in 0..np {
            let es = &mut ev[self.start[p] as usize..self.start[p + 1] as usize];
            let cs = &mut clicks[self.cstart[p] as usize..self.cstart[p + 1] as usize];
            self.person(self.persons[p], es, cs, ctx, out);
        }
        self.ev = ev;
        self.clicks = clicks;
    }

    fn person(&mut self, person: u64, ev: &mut [u64], cl: &mut [Click], ctx: &Ctx, out: &mut SOut) {
        ev.sort_unstable();
        if cl.len() > 1 {
            cl.sort_unstable_by(|a, b| a.ts.cmp(&b.ts).then_with(|| keys::cmp(a.ck, b.ck)));
        }
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

/// Sessions, attribution and aggregation per person partition; each thread then
/// writes its own user_daily part file (overlapping with threads still busy).
fn sessions(sbufs: &[Parts<SRec>], cbufs: &[Parts<CRec>], ctx: &Ctx, ud: &Path) -> Vec<SOut> {
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
            write_user_daily(&ud.join(format!("part-{}.parquet", t)), &out.rows, ctx);
        }
        out.nrows = out.rows.len();
        out.rows = Vec::new();
        out
    })
}

// ---------------------------------------------------------------- output

fn write_user_daily(path: &Path, rows: &[URow], ctx: &Ctx) {
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
        keys::decode_into(r.person, &mut ubytes);
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
    parquet::write_file(path, n, &cols).unwrap_or_else(|e| fail(&format!("{}: {}", path.display(), e)));
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
    if args.len() == 4 && args[1] == "--bench" {
        bench(&args[2], Path::new(&args[3]));
        return;
    }
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
    let threads = env_usize("NCPU")
        .filter(|&n| n > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(8, |n| n.get()))
        .min(64);
    let verbose = std::env::var("SS_VERBOSE").is_ok();
    sys::NOHUGE.store(std::env::var("SS_NOHUGE").is_ok(), Ordering::Relaxed);
    let mut marks = Marks::new(verbose);

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
    let pe_bits = env_usize("SS_PE_BITS").map(|b| b as u32).unwrap_or_else(|| ceil_log2(est_lines / 4096).clamp(2, 8));
    let ps_bits = env_usize("SS_PS_BITS").map(|b| b as u32).unwrap_or_else(|| ceil_log2(est_lines / 4096).clamp(2, 8));

    let first_day = ws.div_euclid(DAY);
    let ndays = ((we - 1).div_euclid(DAY) - first_day + 1) as usize;
    let ctx = Ctx {
        tsp: ts::TsParser::new(ws - 10 * DAY, we + 10 * DAY),
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

    let (lt, pouts) = parse_all(&links_in, &events_in, &clicks_in, &ctx);
    marks.mark("parse");

    let (ebufs, cbufs): (Vec<Parts<ERec>>, Vec<Parts<CRec>>) = pouts.into_iter().map(|po| (po.e, po.c)).unzip();
    let sbufs = dedup(&ebufs, &lt, &ctx);
    marks.mark("dedup");
    // Not freed: unmapping costs more than it saves; the process exits soon.
    std::mem::forget(ebufs);

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
        for (a, l) in sys::arenas() {
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
        eprintln!(
            "sessions: {} | threads={} avx2={} pe_bits={} ps_bits={} user_rows={} fallback_keys={} peak_rss={}MB",
            marks.report(),
            threads,
            ctx.avx2,
            pe_bits,
            ps_bits,
            rows,
            keys::any_fallback(),
            sys::peak_rss_mb()
        );
    }
    // Skip destructors (mmaps, big vectors): the OS reclaims everything.
    std::process::exit(0);
}

/// Hidden micro-benchmark: `sessions --bench STAGE DATA`. Parses events/ with
/// increasing amounts of work per line, single-threaded, 3 passes.
#[allow(dead_code)]
fn bench(stage: &str, data: &Path) {
    let ev = load_input(&data.join("events"), 2 << 20);
    let ws = 0i64;
    let tsp = ts::TsParser::new(1_700_000_000_000, 1_900_000_000_000);
    let mut acc = 0u64;
    let t = Instant::now();
    let pbits = env_usize("SS_PE_BITS").unwrap_or(11) as u32;
    let mut parts: Vec<Vec<ERec>> = (0..(1usize << pbits)).map(|_| Vec::with_capacity(16)).collect();
    for _ in 0..3 {
        for p in parts.iter_mut() {
            p.clear();
        }
        for task in &ev.tasks {
            let b = ev.maps[task.file as usize].bytes();
            match stage {
                "none" => acc += b[task.start] as u64,
                "probe" => acc += ts::ss_probe_parse(&tsp, &b[task.start..task.start + 20]).unwrap_or(0) as u64,
                "touch" => {
                    let mut s = task.start;
                    while s < task.end {
                        acc += b[s] as u64;
                        s += 4096;
                    }
                }
                "lines" => {
                    let mut s = task.start;
                    while s < task.end {
                        let e = scan::next_line_start(b, s);
                        acc += (e - s) as u64;
                        s = e;
                    }
                }
                "scan" => scan::for_each_record::<7, _>(b, task.start, task.end, |f| acc += f[3].len() as u64),
                "ts" => scan::for_each_record::<7, _>(b, task.start, task.end, |f| {
                    acc += tsp.parse(f[3]).unwrap_or(0) as u64 + tsp.parse(f[2]).unwrap_or(0) as u64
                }),
                "keys" => scan::for_each_record::<7, _>(b, task.start, task.end, |f| {
                    acc += keys::key_of(f[0]) ^ keys::key_of(f[1])
                }),
                "all" | "push" => scan::for_each_record::<7, _>(b, task.start, task.end, |f| {
                    let i = tsp.parse(f[3]).unwrap_or(0);
                    let t = tsp.parse(f[2]).unwrap_or(0);
                    let e = keys::key_of(f[0]);
                    let u = keys::key_of(f[1]);
                    let r = ERec { eid: e, user: u, ingest: ingest_rel(i, ws), ts: t as u32, pay: f[4].len() as u32 };
                    if stage == "push" {
                        parts[part(e, pbits)].push(r);
                    } else {
                        acc += { r.eid } ^ { r.user } ^ { r.ts } as u64 ^ { r.ingest } as u64;
                    }
                }),
                _ => fail("unknown stage"),
            }
        }
    }
    let n: usize = parts.iter().map(|p| p.len()).sum();
    eprintln!("bench {}: {:.1}ms per pass (acc {} n {})", stage, t.elapsed().as_secs_f64() * 1e3 / 3.0, acc, n);
}
