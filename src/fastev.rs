//! Event lines, fast path: a line-local splitter for the 7-field event lines.
//!
//! The general scanner (scan.rs) indexes every byte of a chunk (masks,
//! flattened delimiter and newline positions, quote parity per block) and
//! then slices lines out of those arrays. Event lines have an almost fixed
//! shape, so this path splits one line at a time, from its own bytes:
//!
//! * window 1: masks of `,` `\n` `"` over the 64 bytes at the line start;
//!   its first three delimiters end fields 0, 1, 2 (event_id, user_id, ts);
//! * window 2: the same masks over the 64 bytes at the start of field 3; its
//!   first four delimiters end fields 3..=6 (ingest_ts, event_type,
//!   product_id, price_cents), the fourth must be the line's newline.
//!
//! A line takes this path only if, inside the bytes the two windows cover up
//! to the newline, the first six delimiters are commas, the seventh is a
//! newline, and every field holds no quote or exactly two, as its first and
//! last byte (the dialect's rule; see `window`). Then the line has exactly 7
//! well-formed fields, the newline found is the line's first one, and each
//! field unquoted from its first byte is exactly the slice the scanner
//! yields. Every other line (any other shape, a field too long for the
//! windows, a line within `MARGIN` bytes of the end of the buffer) is handed
//! to the general scanner on its own: same lines, same fields by
//! construction. The fields then go through the unchanged `event_line`.
//! The next line start is taken from the newline masks of the 128 bytes at
//! the line start, which keeps the line-to-line dependency chain short; the
//! field walk must find the same newline.
//!
//! The fast path never reads beyond `b` (nor beyond `safe_end`).

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use super::{event_line, event_line_slow, Ctx, EvSink, Task};
use crate::scan;

/// Use this path for event tasks (SS_FASTEV; see `main::run`).
pub static ON: AtomicBool = AtomicBool::new(false);
/// Interleaved cursors per task, 1 or 2 (SS_FASTK).
pub static CURSORS: AtomicUsize = AtomicUsize::new(1);

/// Lines handed to the general scanner by `slow_line` (statistics).
pub static SLOW: AtomicUsize = AtomicUsize::new(0);

/// Readable bytes needed from a line start for the fast path: window 1 (64),
/// window 2 at most 65 bytes further (64), and 32 bytes from the start of
/// any field (the WIDE decoders), with slack.
const MARGIN: usize = 192;

// ---------------------------------------------------------------- masks

// (The compiler splits these 32-byte loads into two 16-byte loads and an
// insert for the x86-64-v2 target; single vmovdqu loads via inline asm were
// measured no faster on the twin, so the intrinsics stay.)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn masks64_avx2(p: *const u8) -> (u64, u64, u64) {
    use std::arch::x86_64::*;
    let a = _mm256_loadu_si256(p as *const __m256i);
    let b = _mm256_loadu_si256(p.add(32) as *const __m256i);
    let m = |v: __m256i, ch: u8| -> u64 { _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, _mm256_set1_epi8(ch as i8))) as u32 as u64 };
    (m(a, b',') | (m(b, b',') << 32), m(a, b'\n') | (m(b, b'\n') << 32), m(a, b'"') | (m(b, b'"') << 32))
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn masks64_base(p: *const u8) -> (u64, u64, u64) {
    use std::arch::x86_64::*;
    let vc = _mm_set1_epi8(b',' as i8);
    let vn = _mm_set1_epi8(b'\n' as i8);
    let vq = _mm_set1_epi8(b'"' as i8);
    let mut c = 0u64;
    let mut n = 0u64;
    let mut q = 0u64;
    for k in 0..4 {
        let v = _mm_loadu_si128(p.add(16 * k) as *const __m128i);
        c |= (_mm_movemask_epi8(_mm_cmpeq_epi8(v, vc)) as u32 as u64) << (16 * k);
        n |= (_mm_movemask_epi8(_mm_cmpeq_epi8(v, vn)) as u32 as u64) << (16 * k);
        q |= (_mm_movemask_epi8(_mm_cmpeq_epi8(v, vq)) as u32 as u64) << (16 * k);
    }
    (c, n, q)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn masks64_base(p: *const u8) -> (u64, u64, u64) {
    use std::arch::aarch64::*;
    #[inline(always)]
    unsafe fn movemask(a: uint8x16_t, b: uint8x16_t, c: uint8x16_t, d: uint8x16_t) -> u64 {
        const BITS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        let m = vld1q_u8(BITS.as_ptr());
        let s0 = vpaddq_u8(vandq_u8(a, m), vandq_u8(b, m));
        let s1 = vpaddq_u8(vandq_u8(c, m), vandq_u8(d, m));
        let s0 = vpaddq_u8(s0, s1);
        let s0 = vpaddq_u8(s0, s0);
        vgetq_lane_u64(vreinterpretq_u64_u8(s0), 0)
    }
    let v0 = vld1q_u8(p);
    let v1 = vld1q_u8(p.add(16));
    let v2 = vld1q_u8(p.add(32));
    let v3 = vld1q_u8(p.add(48));
    let f = |ch: u8| {
        let c = vdupq_n_u8(ch);
        movemask(vceqq_u8(v0, c), vceqq_u8(v1, c), vceqq_u8(v2, c), vceqq_u8(v3, c))
    };
    (f(b','), f(b'\n'), f(b'"'))
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline(always)]
unsafe fn masks64_base(p: *const u8) -> (u64, u64, u64) {
    let (mut c, mut n, mut q) = (0u64, 0u64, 0u64);
    for i in 0..64 {
        let b = *p.add(i);
        c |= ((b == b',') as u64) << i;
        n |= ((b == b'\n') as u64) << i;
        q |= ((b == b'"') as u64) << i;
    }
    (c, n, q)
}

/// (comma, newline, quote) masks of the 64 bytes at `p` (all readable).
#[inline(always)]
unsafe fn masks64<const AVX2: bool>(p: *const u8) -> (u64, u64, u64) {
    #[cfg(target_arch = "x86_64")]
    {
        if AVX2 {
            return masks64_avx2(p);
        }
    }
    masks64_base(p)
}

// ---------------------------------------------------------------- one line

/// A line cut by the fast path: field j is `n[j]` bytes at offset `s[j]`
/// from the line start (already unquoted); `len` is the line length
/// including its newline. Meaningful only if `ok`.
#[derive(Clone, Copy)]
struct Cut {
    ok: bool,
    s: [usize; 7],
    n: [usize; 7],
    len: usize,
}

/// Lowest set bit of `x` and everything below it (all bits if `x == 0`).
#[inline(always)]
fn upto(x: u64) -> u64 {
    x ^ x.wrapping_sub(1)
}

#[inline(always)]
fn blsr(x: u64) -> u64 {
    x & x.wrapping_sub(1)
}

#[inline(always)]
fn tz(x: u64) -> usize {
    x.trailing_zeros() as usize
}

/// Field structure of one 64-byte window that starts at a field start,
/// whose region `r` (bits [0, last used delimiter]) holds whole fields:
/// (violation bits, content-start mask, content-end mask).
///
/// With `fs` the field starts and `qs` the quoted ones (a quote as first
/// byte), adding `qs` to `!d` carries each start bit up to the delimiter that
/// ends its field (fields are separated by delimiters, the zeros of `!d`), so
/// `cl` = the last byte of every quoted field. The window's quotes in `r`
/// must be exactly `qs | cl`, and `qs & cl` empty (a one-byte quoted field):
/// every field holds no quote, or exactly two, as its first and last byte.
/// Then field j (j-th field of the window) spans
/// [j-th bit of `cs`, j-th bit of `ce`) unquoted: `cs` = starts, moved one
/// byte right when quoted; `ce` = delimiters, moved one byte left when the
/// field is quoted.
#[inline(always)]
fn window(d: u64, q: u64, r: u64) -> (u64, u64, u64) {
    let fs = (d << 1) | 1;
    let qs = q & fs & r;
    let cl = ((!d).wrapping_add(qs) & d) >> 1;
    let bad = ((q ^ (qs | cl)) | (qs & cl)) & r;
    let cs = (fs & !q) | (qs << 1);
    let ce = (d & !(cl << 1)) | cl;
    (bad, cs, ce)
}

/// Cuts the line at `p` (a line start with MARGIN readable bytes).
/// Branch-free; positions use wrapping arithmetic and are garbage unless ok.
#[inline(always)]
unsafe fn cut<const AVX2: bool>(p: *const u8) -> Cut {
    // window 1: fields 0..=2 (its region ends at the third delimiter)
    let (c1, n1, q1) = masks64::<AVX2>(p);
    let d1 = c1 | n1;
    let d1c = blsr(blsr(d1));
    let r1 = upto(d1c);
    let (b1, cs1, ce1) = window(d1, q1, r1);
    // window 2: fields 3..=6 from the start of field 3 (region: 4 delimiters)
    let w = tz(d1c) + 1; // <= 65
    let (c2, n2, q2) = masks64::<AVX2>(p.add(w));
    let d2 = c2 | n2;
    let d2d = blsr(blsr(blsr(d2)));
    let r2 = upto(d2d);
    let (b2, cs2, ce2) = window(d2, q2, r2);
    // the first six delimiters are commas, the seventh is the newline
    let last = d2d & d2d.wrapping_neg();
    let nl = (n1 & r1) | ((n2 ^ last) & r2);
    // The line end also from the newline masks of the 128 bytes at p, off
    // the field chain: the next line start depends only on these loads. On
    // a cut line both are its first newline (offset <= 128).
    let (_, n1b, _) = masks64::<AVX2>(p.add(64));
    let le = if n1 != 0 { tz(n1) } else { 64 + tz(n1b) };
    let ok = (d1c != 0) & (d2d != 0) & ((b1 | b2 | nl) == 0) & (w + tz(d2d) == le);
    let mut s = [0usize; 7];
    let mut n = [0usize; 7];
    let (mut cs, mut ce) = (cs1, ce1);
    for j in 0..3 {
        s[j] = tz(cs);
        n[j] = tz(ce).wrapping_sub(tz(cs));
        cs = blsr(cs);
        ce = blsr(ce);
    }
    let (mut cs, mut ce) = (cs2, ce2);
    for j in 3..7 {
        s[j] = w + tz(cs);
        n[j] = tz(ce).wrapping_sub(tz(cs));
        cs = blsr(cs);
        ce = blsr(ce);
    }
    Cut { ok, s, n, len: le + 1 }
}

/// The fields of a cut line (`c.ok`).
#[inline(always)]
unsafe fn fields<'a>(p: *const u8, c: &Cut) -> [&'a [u8]; 7] {
    let f = |j: usize| std::slice::from_raw_parts(p.add(c.s[j]), c.n[j]);
    [f(0), f(1), f(2), f(3), f(4), f(5), f(6)]
}

/// The general scanner on the single line at `*off`; advances `*off` past it.
#[cold]
#[inline(never)]
fn slow_line<'a, const AVX2: bool, FS: FnMut(&[&'a [u8]; 7])>(b: &'a [u8], off: &mut usize, fs: &mut FS) {
    SLOW.fetch_add(1, Ordering::Relaxed);
    let e = scan::next_line_start(b, *off);
    scan::for_each_record_x::<7, _, AVX2>(b, *off, e, |x| fs(x));
    *off = e;
}

/// Lines from `off` while `off < end` and `off <= lim` (fast path, `ff`),
/// the rest of [off, end) with the general scanner (`fs`).
#[inline(always)]
fn run1<'a, const AVX2: bool, FF: FnMut(&[&'a [u8]; 7]), FS: FnMut(&[&'a [u8]; 7])>(
    b: &'a [u8],
    mut off: usize,
    end: usize,
    lim: usize,
    ff: &mut FF,
    fs: &mut FS,
) {
    let base = b.as_ptr();
    while off < end && off <= lim {
        unsafe {
            let p = base.add(off);
            let c = cut::<AVX2>(p);
            if c.ok {
                ff(&fields(p, &c));
                off += c.len;
            } else {
                slow_line::<AVX2, FS>(b, &mut off, fs);
            }
        }
    }
    if off < end {
        scan::for_each_record_x::<7, _, AVX2>(b, off, end, |x| fs(x));
    }
}

/// Two cursors over two line-aligned halves of [start, end), one line of
/// each per step.
#[inline(always)]
fn run2<'a, const AVX2: bool, FF: FnMut(&[&'a [u8]; 7]), FS: FnMut(&[&'a [u8]; 7])>(
    b: &'a [u8],
    start: usize,
    end: usize,
    lim: usize,
    ff: &mut FF,
    fs: &mut FS,
) {
    let base = b.as_ptr();
    // a line start in (start, end]
    let mid = scan::next_line_start(b, start + (end - start) / 2 - 1).min(end);
    let (mut a, mut c) = (start, mid);
    // a < mid <= c, so c <= lim covers both
    while a < mid && c < end && c <= lim {
        unsafe {
            let (pa, pc) = (base.add(a), base.add(c));
            let ca = cut::<AVX2>(pa);
            let cc = cut::<AVX2>(pc);
            if ca.ok & cc.ok {
                ff(&fields(pa, &ca));
                ff(&fields(pc, &cc));
                a += ca.len;
                c += cc.len;
            } else {
                if ca.ok {
                    ff(&fields(pa, &ca));
                    a += ca.len;
                } else {
                    slow_line::<AVX2, FS>(b, &mut a, fs);
                }
                if cc.ok {
                    ff(&fields(pc, &cc));
                    c += cc.len;
                } else {
                    slow_line::<AVX2, FS>(b, &mut c, fs);
                }
            }
        }
    }
    run1::<AVX2, FF, FS>(b, a, mid, lim, ff, fs);
    run1::<AVX2, FF, FS>(b, c, end, lim, ff, fs);
}

/// Every well-formed line of exactly 7 fields in `b[start..end)`, as its
/// unquoted fields: the same lines and fields as
/// `scan::for_each_record_x::<7, _, AVX2>(b, start, end, ..)`, in the same
/// order with one cursor; with two (k = 2), the lines of two halves
/// interleave (each half in order). Lines cut by the fast path go to `ff`
/// (then 32 bytes from the start of every field are inside `b`), the others
/// to `fs`. `start` must be a line start and `end` a line start or
/// `b.len()`; `safe_end` is the end address of readable memory (>= the end
/// of `b`); the fast path never reads beyond `b`.
#[inline(always)]
pub fn for_each_line2<'a, const AVX2: bool, FF: FnMut(&[&'a [u8]; 7]), FS: FnMut(&[&'a [u8]; 7])>(
    b: &'a [u8],
    start: usize,
    end: usize,
    safe_end: usize,
    k: usize,
    mut ff: FF,
    mut fs: FS,
) {
    assert!(start <= end && end <= b.len() && safe_end >= b.as_ptr() as usize + b.len());
    // last line start for the fast path
    let lim = b.len().wrapping_sub(MARGIN);
    if b.len() < MARGIN || start >= end {
        scan::for_each_record_x::<7, _, AVX2>(b, start, end, fs);
        return;
    }
    const MINSUB: usize = 16 << 10;
    if k >= 2 && end - start >= 2 * MINSUB {
        run2::<AVX2, FF, FS>(b, start, end, lim, &mut ff, &mut fs)
    } else {
        run1::<AVX2, FF, FS>(b, start, end, lim, &mut ff, &mut fs)
    }
}

/// `for_each_line2` with one callback for all lines.
#[inline(always)]
pub fn for_each_line<'a, const AVX2: bool, F: FnMut(&[&'a [u8]; 7])>(b: &'a [u8], start: usize, end: usize, safe_end: usize, k: usize, mut f: F) {
    // the two callbacks are never active at the same time
    let fp: *mut F = &mut f;
    for_each_line2::<AVX2, _, _>(b, start, end, safe_end, k, #[inline(always)] |x| unsafe { (*fp)(x) }, |x| unsafe { (*fp)(x) });
}

/// `main::event_task` on the fast splitter (same copies; with two cursors
/// in a different order).
#[inline(always)]
pub fn event_task_fast<const AVX2: bool, S: EvSink>(b: &[u8], task: Task, ctx: &Ctx, sink: &mut S, safe_end: usize) {
    let k = CURSORS.load(Ordering::Relaxed);
    // The callbacks are never active at the same time; both reach the sink
    // through this pointer.
    let sp: *mut S = sink;
    for_each_line2::<AVX2, _, _>(
        b,
        task.start,
        task.end,
        safe_end,
        k,
        #[inline(always)]
        |f| {
            // fast-path line: 32 bytes from every field start are readable
            if !f[0].is_empty() {
                event_line::<true, S, _>(f, f[0], ctx, unsafe { &mut *sp });
            }
        },
        |f| {
            if f[0].is_empty() {
                return;
            }
            if f[6].as_ptr() as usize + 32 <= safe_end {
                event_line::<true, S, _>(f, f[0], ctx, unsafe { &mut *sp });
            } else {
                event_line_slow(f, ctx, unsafe { &mut *sp });
            }
        },
    );
}

// ---------------------------------------------------------------- checking

/// Records every copy handed to the sink.
#[derive(Default)]
pub struct RecSink(pub Vec<(u64, u32, Option<(u64, u32, u32)>)>);
impl EvSink for RecSink {
    #[inline(always)]
    fn copy(&mut self, eid: u64, ingest: u32, valid: Option<(u64, u32, u32)>) {
        self.0.push((eid, ingest, valid));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
unsafe fn task_fast_avx2<S: EvSink>(b: &[u8], task: Task, ctx: &Ctx, sink: &mut S, safe_end: usize) {
    event_task_fast::<true, S>(b, task, ctx, sink, safe_end)
}

/// `event_task_fast` with run-time dispatch (as `main::event_task_avx2`).
pub fn task_fast_any<S: EvSink>(b: &[u8], task: Task, ctx: &Ctx, sink: &mut S, safe_end: usize) {
    if ctx.avx2 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            return task_fast_avx2(b, task, ctx, sink, safe_end);
        }
    }
    event_task_fast::<false, S>(b, task, ctx, sink, safe_end)
}

/// SS_EVCHECK: every event task through the general path and through this
/// path with 1 and 2 cursors; the copies must be identical (same order with
/// one cursor, same multiset otherwise). Exits nonzero on a difference.
pub fn check(ev: &super::Input, ctx: &Ctx) {
    let saved = CURSORS.load(Ordering::Relaxed);
    let mut lines = 0usize;
    let mut copies = 0usize;
    for task in &ev.tasks {
        let b = ev.maps[task.file as usize].bytes();
        let safe_end = b.as_ptr() as usize + b.len();
        let mut want = RecSink::default();
        super::event_task_scan_any(b, *task, ctx, &mut want, safe_end);
        lines += b[task.start..task.end].iter().filter(|&&c| c == b'\n').count();
        copies += want.0.len();
        for k in 1..=2 {
            CURSORS.store(k, Ordering::Relaxed);
            let mut got = RecSink::default();
            task_fast_any(b, *task, ctx, &mut got, safe_end);
            let same = if k == 1 {
                got.0 == want.0
            } else {
                let mut a = got.0.clone();
                let mut w = want.0.clone();
                a.sort_unstable();
                w.sort_unstable();
                a == w
            };
            if !same {
                eprintln!("evcheck: MISMATCH file {} [{}, {}) cursors {} ({} vs {} copies)", task.file, task.start, task.end, k, got.0.len(), want.0.len());
                std::process::exit(1);
            }
        }
    }
    CURSORS.store(saved, Ordering::Relaxed);
    eprintln!(
        "evcheck: OK {} tasks, {} lines, {} copies identical (cursors 1-2, avx2={}); {} lines took the general path",
        ev.tasks.len(),
        lines,
        copies,
        ctx.avx2,
        SLOW.load(Ordering::Relaxed) / 2
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: u64) -> impl FnMut() -> u64 {
        let mut x = seed | 1;
        move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        }
    }

    /// One event field value of kind j (0..7), often well-formed, sometimes
    /// hostile.
    fn value(j: usize, r: &mut impl FnMut() -> u64) -> Vec<u8> {
        let k = r() % 100;
        let ws = 1_828_224_000_000i64; // 2027-12-09T00:00:00Z
        let civil = |ms: i64| -> String {
            let days = ms.div_euclid(86_400_000);
            let sod = ms.rem_euclid(86_400_000) / 1000;
            let (y, m, d) = civil_from_days(days);
            format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}", y, m, d, sod / 3600, sod / 60 % 60, sod % 60)
        };
        if k < 4 {
            // hostile soup
            let alpha = b"ab,\"\n\r 0Z-:T\xd9\xa1";
            return (0..r() % 40).map(|_| alpha[(r() % alpha.len() as u64) as usize]).collect();
        }
        if k < 7 {
            return Vec::new();
        }
        if k < 9 {
            // long values (windows overflow)
            return (0..30 + r() % 60).map(|_| b"xyz019"[(r() % 6) as usize]).collect();
        }
        let ms = ws - 3 * 86_400_000 + (r() % (20 * 86_400_000)) as i64;
        match j {
            0 | 1 => {
                let n = 1 + r() % 13;
                let alpha: &[u8] = if r() % 20 == 0 { b"aZ_-09 \xc3\xa9" } else { b"abcdefghij0123456789" };
                let mut v: Vec<u8> = (0..n).map(|_| alpha[(r() % alpha.len() as u64) as usize]).collect();
                v[0] = if j == 0 { b'e' } else { b'u' };
                v
            }
            2 | 3 => match (r() % 10, j) {
                (0..=5, _) | (_, 3) => format!("{}Z", civil(ms)).into_bytes(),
                (6..=7, _) => format!("{:0w$}", ms, w = if r() % 8 == 0 { 16 } else { 13 }).into_bytes(),
                _ => {
                    let oh = r() % 25;
                    format!("{}{}{:02}:{:02}", civil(ms), if r() % 2 == 0 { '+' } else { '-' }, oh, r() % 61).into_bytes()
                }
            },
            4 => [&b"view"[..], b"cart", b"purchase", b"Purchase", b"cart ", b"vie"][(r() % 6) as usize].to_vec(),
            5 => format!("p{}", r() % 60_000).into_bytes(),
            _ => match r() % 6 {
                0 => b"".to_vec(),
                1 => b"1000\r".to_vec(),
                2 => format!("{}", r() % 20_000_000).into_bytes(),
                3 => b"0250".to_vec(),
                _ => format!("{}", 100 + r() % 50_000).into_bytes(),
            },
        }
    }

    fn civil_from_days(z: i64) -> (i64, i64, i64) {
        let z = z + 719_468;
        let era = z.div_euclid(146_097);
        let doe = z - era * 146_097;
        let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = doy - (153 * mp + 2) / 5 + 1;
        let m = if mp < 10 { mp + 3 } else { mp - 9 };
        (yoe + era * 400 + (m <= 2) as i64, m, d)
    }

    /// Random event text: mostly 7-field lines with random quoting, plus
    /// wrong field counts, stray quotes, empty lines, `\r`, long fields.
    fn text(r: &mut impl FnMut() -> u64, lines: usize) -> Vec<u8> {
        let mut t = Vec::new();
        for _ in 0..lines {
            let mode = r() % 50;
            if mode == 0 {
                t.push(b'\n');
                continue;
            }
            let nf = match mode {
                1 => 6,
                2 => 8,
                3 => 1 + (r() % 12) as usize,
                _ => 7,
            };
            for j in 0..nf {
                if j > 0 {
                    t.push(b',');
                }
                let v = value(j.min(6), r);
                let quoted = r() % 5 == 0;
                if quoted {
                    t.push(b'"');
                }
                t.extend_from_slice(&v);
                if quoted && r() % 40 != 0 {
                    t.push(b'"');
                }
                if r() % 200 == 0 {
                    t.push(b'"');
                }
            }
            if r() % 100 == 0 {
                t.push(b'\r');
            }
            t.push(b'\n');
        }
        if r() % 3 == 0 {
            t.pop();
        }
        t
    }

    fn rows(b: &[u8], start: usize, end: usize, fast: Option<usize>) -> Vec<Vec<Vec<u8>>> {
        let mut out = Vec::new();
        let push = |f: &[&[u8]; 7]| out.push(f.iter().map(|s| s.to_vec()).collect());
        let safe_end = b.as_ptr() as usize + b.len();
        match fast {
            Some(k) => for_each_line::<false, _>(b, start, end, safe_end, k, push),
            None => scan::for_each_record_x::<7, _, false>(b, start, end, push),
        }
        out
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
    unsafe fn rows_avx2(b: &[u8], start: usize, end: usize, k: usize) -> Vec<Vec<Vec<u8>>> {
        let mut out = Vec::new();
        let safe_end = b.as_ptr() as usize + b.len();
        for_each_line::<true, _>(b, start, end, safe_end, k, |f: &[&[u8]; 7]| out.push(f.iter().map(|s| s.to_vec()).collect()));
        out
    }

    /// Same lines and fields as the general scanner, whole buffers and
    /// line-aligned chunks, 1-2 cursors, both SIMD paths where available.
    #[test]
    fn splitter_matches_scanner() {
        let mut r = rng(0xFA57_E7E7_0000_0001);
        #[cfg(target_arch = "x86_64")]
        let avx2 = std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("bmi2");
        let mut total = 0usize;
        for round in 0..600 {
            let n = match round % 4 {
                0 => (r() % 8) as usize,
                1 => (r() % 200) as usize,
                _ => 300 + (r() % 3000) as usize,
            };
            let t = text(&mut r, n);
            let want = rows(&t, 0, t.len(), None);
            total += want.len();
            assert_eq!(rows(&t, 0, t.len(), Some(1)), want, "round {}", round);
            for k in 2..=2 {
                let mut a = rows(&t, 0, t.len(), Some(k));
                let mut w = want.clone();
                a.sort();
                w.sort();
                assert_eq!(a, w, "round {} k {}", round, k);
            }
            #[cfg(target_arch = "x86_64")]
            if avx2 {
                for k in 1..=2 {
                    let mut a = unsafe { rows_avx2(&t, 0, t.len(), k) };
                    let mut w = want.clone();
                    if k > 1 {
                        a.sort();
                        w.sort();
                    }
                    assert_eq!(a, w, "avx2 round {} k {}", round, k);
                }
            }
            // line-aligned chunks
            let mut s0 = 0;
            let mut got = Vec::new();
            while s0 < t.len() {
                let nominal = s0 + 1 + (r() % 5000) as usize;
                let e = if nominal >= t.len() { t.len() } else { scan::next_line_start(&t, nominal - 1) };
                got.extend(rows(&t, s0, e, Some(1)));
                s0 = e;
            }
            assert_eq!(got, want, "chunked round {}", round);
        }
        assert!(total > 100_000, "{}", total);
    }

    /// Every single-byte change of a well-formed line, at every position,
    /// in the middle of a buffer (so the fast path sees it).
    #[test]
    fn single_byte_mutations() {
        let mut r = rng(77);
        let line = b"\"e42huhoug\",uduffvac,\"2027-12-19T18:13:05Z\",2027-12-19T18:14:52Z,\"purchase\",p13649,\"2918\"\n";
        let mut pre = Vec::new();
        for _ in 0..4 {
            pre.extend_from_slice(b"e1,u1,1829024243000,2027-12-17T06:18:47Z,cart,p2,\n");
        }
        for pos in 0..line.len() {
            for v in [b',', b'"', b'\n', b'\r', b'x', b' ', 0u8, 0xFF] {
                let mut l = line.to_vec();
                l[pos] = v;
                let mut t = pre.clone();
                t.extend_from_slice(&l);
                for _ in 0..8 {
                    t.extend_from_slice(&text(&mut r, 1));
                    if t.last() != Some(&b'\n') {
                        t.push(b'\n');
                    }
                }
                t.extend_from_slice(&[b'z'; 300]);
                t.push(b'\n');
                assert_eq!(rows(&t, 0, t.len(), Some(1)), rows(&t, 0, t.len(), None), "pos {} byte {}", pos, v);
            }
        }
    }
}
