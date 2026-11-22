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

use super::{event_line, Ctx, EvSink, Task};
use crate::scan;

/// Use this path for event tasks (SS_FASTEV; see `main::run`).
pub static ON: AtomicBool = AtomicBool::new(false);

/// Lines handed to the general scanner by `slow_end` (statistics).
pub static SLOW: AtomicUsize = AtomicUsize::new(0);
/// Click lines through the one-window splitter (`for_each_click`) unless
/// SS_CLICKFAST=0 (see `main::run`).
pub static CLICKS: AtomicBool = AtomicBool::new(false);
/// Click lines handed to the general scanner (statistics).
pub static CSLOW: AtomicUsize = AtomicUsize::new(0);
/// Lines cut by the 128-byte splitters (statistics, counted like SLOW).
pub static WIDE_N: AtomicUsize = AtomicUsize::new(0);
pub static CWIDE_N: AtomicUsize = AtomicUsize::new(0);
/// SLOW / CSLOW are counted only while set (the checks): a shared counter
/// bumped by every thread for every slow line costs far more than the line.
pub static COUNT_SLOW: AtomicBool = AtomicBool::new(false);
/// Software prefetch distance (bytes ahead of the current line) for the
/// line splitters; 0 = none (SS_PFDIST, default 1024).
pub static PF_DIST: AtomicUsize = AtomicUsize::new(1024);
/// A second prefetch per line, 64 bytes further (lines average ~1.4 cache
/// lines; on unless SS_PF2=0).
pub static PF2: AtomicBool = AtomicBool::new(true);
/// The 128-byte splitters for lines the 64-byte windows cannot cut (long
/// IDs, scorer version 5) unless SS_WIDECUT=0.
pub static WIDE: AtomicBool = AtomicBool::new(true);

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
pub(crate) struct Cut {
    pub(crate) ok: bool,
    pub(crate) s: [usize; 7],
    pub(crate) n: [usize; 7],
    pub(crate) len: usize,
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
pub(crate) unsafe fn cut<const AVX2: bool>(p: *const u8) -> Cut {
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

// ---------------------------------------------------------------- wide lines
//
// Lines with long IDs (scorer version 5: 33-byte IDs) do not fit the 64-byte
// windows. The wide splitters use the same rule over 128 bits (`window128`,
// identical to `window` but on u128): event window 1 covers the 128 bytes at
// the line start and must hold fields 0..=2; window 2 (64 bytes at the start
// of field 3) is unchanged. A click line must fit 128 bytes. Then, as for
// the narrow cut, the line has exactly its fields and they are exactly the
// scanner's; anything else still goes to the scanner.

/// Readable bytes from a line start for the wide event cut: window 1 (128),
/// window 2 at most 129 bytes in (64), 32 bytes from any field start (field
/// 6 starts at most 129 + 64), with slack.
pub(crate) const MARGIN_W: usize = 256;
/// The same for the wide click cut: the window (128) + 32 from a field start.
const MARGIN_CW: usize = 192;

#[inline(always)]
fn upto128(x: u128) -> u128 {
    x ^ x.wrapping_sub(1)
}

#[inline(always)]
fn blsr128(x: u128) -> u128 {
    x & x.wrapping_sub(1)
}

#[inline(always)]
fn tz128(x: u128) -> usize {
    x.trailing_zeros() as usize
}

/// `window` on a 128-byte window.
#[inline(always)]
fn window128(d: u128, q: u128, r: u128) -> (u128, u128, u128) {
    let fs = (d << 1) | 1;
    let qs = q & fs & r;
    let cl = ((!d).wrapping_add(qs) & d) >> 1;
    let bad = ((q ^ (qs | cl)) | (qs & cl)) & r;
    let cs = (fs & !q) | (qs << 1);
    let ce = (d & !(cl << 1)) | cl;
    (bad, cs, ce)
}

#[inline(always)]
unsafe fn masks128<const AVX2: bool>(p: *const u8) -> (u128, u128, u128) {
    let (c0, n0, q0) = masks64::<AVX2>(p);
    let (c1, n1, q1) = masks64::<AVX2>(p.add(64));
    (c0 as u128 | ((c1 as u128) << 64), n0 as u128 | ((n1 as u128) << 64), q0 as u128 | ((q1 as u128) << 64))
}

/// `cut` with a 128-byte window 1 (a line start with MARGIN_W readable bytes).
#[inline(never)]
pub(crate) unsafe fn cut_wide<const AVX2: bool>(p: *const u8) -> Cut {
    let (c1, n1, q1) = masks128::<AVX2>(p);
    let d1 = c1 | n1;
    let d1c = blsr128(blsr128(d1));
    let r1 = upto128(d1c);
    let (b1, cs1, ce1) = window128(d1, q1, r1);
    let w = tz128(d1c) + 1; // <= 129
    let (c2, n2, q2) = masks64::<AVX2>(p.add(w));
    let d2 = c2 | n2;
    let d2d = blsr(blsr(blsr(d2)));
    let r2 = upto(d2d);
    let (b2, cs2, ce2) = window(d2, q2, r2);
    let last = d2d & d2d.wrapping_neg();
    let bad = (b1 != 0) | (n1 & r1 != 0) | ((b2 | ((n2 ^ last) & r2)) != 0);
    let ok = (d1c != 0) & (d2d != 0) & !bad;
    let mut s = [0usize; 7];
    let mut n = [0usize; 7];
    let (mut cs, mut ce) = (cs1, ce1);
    for j in 0..3 {
        s[j] = tz128(cs);
        n[j] = tz128(ce).wrapping_sub(tz128(cs));
        cs = blsr128(cs);
        ce = blsr128(ce);
    }
    let (mut cs, mut ce) = (cs2, ce2);
    for j in 3..7 {
        s[j] = w + tz(cs);
        n[j] = tz(ce).wrapping_sub(tz(cs));
        cs = blsr(cs);
        ce = blsr(ce);
    }
    Cut { ok, s, n, len: w + tz(d2d) + 1 }
}

/// `cut4` on a 128-byte window (MARGIN_CW readable bytes).
#[inline(never)]
unsafe fn cut4_wide<const AVX2: bool>(p: *const u8) -> Cut4 {
    let (c, n, q) = masks128::<AVX2>(p);
    let d = c | n;
    let d3 = blsr128(blsr128(blsr128(d)));
    let r = upto128(d3);
    let (b, cs, ce) = window128(d, q, r);
    let last = d3 & d3.wrapping_neg();
    let nl = (n ^ last) & r;
    let ok = (d3 != 0) & ((b | nl) == 0);
    let mut s = [0usize; 4];
    let mut ln = [0usize; 4];
    let (mut cs, mut ce) = (cs, ce);
    for j in 0..4 {
        s[j] = tz128(cs);
        ln[j] = tz128(ce).wrapping_sub(tz128(cs));
        cs = blsr128(cs);
        ce = blsr128(ce);
    }
    Cut4 { ok, s, n: ln, len: tz128(d3) + 1 }
}

/// End of the line at `off` (the position after its newline, or the end of
/// `b`): a line the fast path did not cut.
#[cold]
#[inline(never)]
fn slow_end(b: &[u8], off: usize) -> usize {
    if COUNT_SLOW.load(Ordering::Relaxed) {
        SLOW.fetch_add(1, Ordering::Relaxed);
    }
    scan::next_line_start(b, off)
}

/// Every line of `b[start..end)`: lines cut by the fast path go to `ff` as
/// their unquoted fields (then 32 bytes from the start of every field are
/// inside `b`); every other part of the range goes to `fr(from, to)` as a
/// line-aligned byte range for the general scanner (a single line the fast
/// path did not cut, the lines within MARGIN of the end of `b`, or all of it
/// for a small buffer). In order, and together exactly the lines and fields
/// of `scan::for_each_record_x::<7, _, AVX2>(b, start, end, ..)`. `start`
/// must be a line start and `end` a line start or `b.len()`; `safe_end` is
/// the end address of readable memory (>= the end of `b`); the fast path
/// never reads beyond `b`.
#[inline(always)]
pub fn for_each_line2<'a, const AVX2: bool, FF: FnMut(&[&'a [u8]; 7]), FR: FnMut(usize, usize)>(
    b: &'a [u8],
    start: usize,
    end: usize,
    safe_end: usize,
    mut ff: FF,
    mut fr: FR,
) {
    assert!(start <= end && end <= b.len() && safe_end >= b.as_ptr() as usize + b.len());
    if start >= end {
        return;
    }
    if b.len() < MARGIN {
        fr(start, end);
        return;
    }
    // last line start for the fast path
    let lim = b.len() - MARGIN;
    let lim_w = b.len().saturating_sub(MARGIN_W);
    let wide = WIDE.load(Ordering::Relaxed) && b.len() >= MARGIN_W;
    let base = b.as_ptr();
    let pf = PF_DIST.load(Ordering::Relaxed);
    let pf2 = PF2.load(Ordering::Relaxed);
    // prefetch addresses stay inside b (a prefetch never faults, but keep it tidy)
    let pf_lim = (base as usize + b.len()).saturating_sub(64);
    let mut off = start;
    while off < end && off <= lim {
        unsafe {
            let p = base.add(off);
            if pf != 0 {
                super::prefetch((p as usize + pf).min(pf_lim) as *const u8);
                if pf2 {
                    super::prefetch((p as usize + pf + 64).min(pf_lim) as *const u8);
                }
            }
            let c = cut::<AVX2>(p);
            if c.ok {
                ff(&fields(p, &c));
                off += c.len;
            } else {
                if wide && off <= lim_w {
                    let c = cut_wide::<AVX2>(p);
                    if c.ok {
                        if COUNT_SLOW.load(Ordering::Relaxed) {
                            WIDE_N.fetch_add(1, Ordering::Relaxed);
                        }
                        ff(&fields(p, &c));
                        off += c.len;
                        continue;
                    }
                }
                let e = slow_end(b, off);
                fr(off, e);
                off = e;
            }
        }
    }
    if off < end {
        fr(off, end);
    }
}

/// `main::event_task` on the fast splitter: the same copies in the same
/// order. Lines the fast path does not cut go through the general path
/// (`main::event_task_scan_ni`, one out-of-line instance per sink type).
#[inline(always)]
pub fn event_task_fast<const AVX2: bool, S: EvSink>(b: &[u8], task: Task, ctx: &Ctx, sink: &mut S, safe_end: usize) {
    // The callbacks are never active at the same time; both reach the sink
    // through this pointer.
    let sp: *mut S = sink;
    for_each_line2::<AVX2, _, _>(
        b,
        task.start,
        task.end,
        safe_end,
        #[inline(always)]
        |f| {
            // fast-path line: 32 bytes from every field start are readable
            if !f[0].is_empty() {
                event_line::<true, S, _>(f, f[0], ctx, unsafe { &mut *sp });
            }
        },
        |from, to| super::event_task_scan_ni::<AVX2, S>(b, Task { file: task.file, start: from, end: to }, ctx, unsafe { &mut *sp }, safe_end),
    );
}

// ---------------------------------------------------------------- click lines
//
// A click line `click_id,user_id,campaign_id,ts` is short (about 50 bytes),
// so one 64-byte window from the line start normally holds the whole line:
// its first four delimiters must be three commas and the line's newline,
// with every field holding no quote or two, as its first and last byte
// (`window`). Then the line has exactly 4 well-formed fields and the newline
// found is its first one. Any other line, and the lines within MARGIN_C
// bytes of the end of the buffer, go through the general scanner.

/// Readable bytes needed from a click line start: the window (64) and 32
/// bytes from any field start inside it (the WIDE decoders), with slack.
const MARGIN_C: usize = 128;

/// A click line cut from one window; meaningful only if `ok`.
#[derive(Clone, Copy)]
struct Cut4 {
    ok: bool,
    s: [usize; 4],
    n: [usize; 4],
    len: usize,
}

#[inline(always)]
unsafe fn cut4<const AVX2: bool>(p: *const u8) -> Cut4 {
    let (c, n, q) = masks64::<AVX2>(p);
    let d = c | n;
    // the fourth delimiter (lowest bit of d3) must be the newline
    let d3 = blsr(blsr(blsr(d)));
    let r = upto(d3);
    let (b, cs, ce) = window(d, q, r);
    let last = d3 & d3.wrapping_neg();
    let nl = (n ^ last) & r;
    let ok = (d3 != 0) & ((b | nl) == 0);
    let mut s = [0usize; 4];
    let mut ln = [0usize; 4];
    let (mut cs, mut ce) = (cs, ce);
    for j in 0..4 {
        s[j] = tz(cs);
        ln[j] = tz(ce).wrapping_sub(tz(cs));
        cs = blsr(cs);
        ce = blsr(ce);
    }
    Cut4 { ok, s, n: ln, len: tz(d3) + 1 }
}

/// The fields of a cut click line (`c.ok`).
#[inline(always)]
unsafe fn fields4<'a>(p: *const u8, c: &Cut4) -> [&'a [u8]; 4] {
    let f = |j: usize| std::slice::from_raw_parts(p.add(c.s[j]), c.n[j]);
    [f(0), f(1), f(2), f(3)]
}

/// The general scanner on click lines b[from..to) (out of line: the fallback
/// for single lines, buffer tails and small buffers).
#[cold]
#[inline(never)]
fn click_scan<'a, const AVX2: bool, F: FnMut(&[&'a [u8]; 4])>(b: &'a [u8], from: usize, to: usize, f: &mut F) {
    scan::for_each_record_x::<4, _, AVX2>(b, from, to, |x| f(x));
}

/// Calls `f` with the unquoted fields of every well-formed 4-field line in
/// `b[start..end)`, in order: the same lines and fields as
/// `scan::for_each_record_x::<4, _, AVX2>(b, start, end, f)`. Lines cut by
/// the fast path have 32 readable bytes after every field start inside `b`.
/// `start` must be a line start and `end` a line start or `b.len()`;
/// `safe_end` (>= the end of `b`) as for `for_each_line2`; the fast path
/// never reads beyond `b`.
#[inline(always)]
pub fn for_each_click<'a, const AVX2: bool, F: FnMut(&[&'a [u8]; 4])>(b: &'a [u8], start: usize, end: usize, safe_end: usize, f: &mut F) {
    assert!(start <= end && end <= b.len() && safe_end >= b.as_ptr() as usize + b.len());
    if start >= end {
        return;
    }
    if b.len() < MARGIN_C {
        click_scan::<AVX2, F>(b, start, end, f);
        return;
    }
    let lim = b.len() - MARGIN_C;
    let lim_w = b.len().saturating_sub(MARGIN_CW);
    let wide = WIDE.load(Ordering::Relaxed) && b.len() >= MARGIN_CW;
    let base = b.as_ptr();
    let pf = PF_DIST.load(Ordering::Relaxed);
    let pf_lim = (base as usize + b.len()).saturating_sub(64);
    let mut off = start;
    while off < end && off <= lim {
        unsafe {
            let p = base.add(off);
            if pf != 0 {
                super::prefetch((p as usize + pf).min(pf_lim) as *const u8);
            }
            let c = cut4::<AVX2>(p);
            if c.ok {
                f(&fields4(p, &c));
                off += c.len;
            } else {
                if wide && off <= lim_w {
                    let c = cut4_wide::<AVX2>(p);
                    if c.ok {
                        if COUNT_SLOW.load(Ordering::Relaxed) {
                            CWIDE_N.fetch_add(1, Ordering::Relaxed);
                        }
                        f(&fields4(p, &c));
                        off += c.len;
                        continue;
                    }
                }
                if COUNT_SLOW.load(Ordering::Relaxed) {
                    CSLOW.fetch_add(1, Ordering::Relaxed);
                }
                let e = scan::next_line_start(b, off);
                click_scan::<AVX2, F>(b, off, e, f);
                off = e;
            }
        }
    }
    if off < end {
        click_scan::<AVX2, F>(b, off, end, f);
    }
}

/// SS_CLICKCHECK: every click task through `main::click_task` with the
/// general scanner and with the click splitter; the click records must be
/// identical, in the same order. Exits nonzero on a difference.
pub fn click_check(links_in: &super::Input, cl: &super::Input, ctx: &Ctx) {
    COUNT_SLOW.store(true, Ordering::Relaxed);
    use super::{CRec, CSink, PCache};
    struct RecC(Vec<(usize, u64, u64, u32, u32)>);
    impl CSink for RecC {
        #[inline(always)]
        fn push_click(&mut self, q: usize, r: CRec) {
            self.0.push((q, r.person, r.ck, r.ts, r.camp));
        }
    }
    let lb = super::LocBits::new(ctx.threads, ctx.ps_bits).unwrap();
    let empty = super::Input { maps: Vec::new(), tasks: Vec::new(), bytes: 0 };
    let (lt, _) = super::parse_b(links_in, &empty, &empty, ctx, lb, None);
    let saved = CLICKS.load(Ordering::Relaxed);
    let (mut lines, mut recs) = (0usize, 0usize);
    for task in &cl.tasks {
        let b = cl.maps[task.file as usize].bytes();
        let safe_end = b.as_ptr() as usize + b.len();
        let run = |fast: bool| {
            CLICKS.store(fast, Ordering::Relaxed);
            let mut r = RecC(Vec::new());
            let mut cache = PCache::new();
            super::click_task_any(b, *task, ctx, &lt, &mut r, safe_end, &mut cache);
            r.0
        };
        let want = run(false);
        let got = run(true);
        lines += b[task.start..task.end].iter().filter(|&&c| c == b'\n').count();
        recs += want.len();
        if got != want {
            eprintln!("clickcheck: MISMATCH file {} [{}, {}) ({} vs {} records)", task.file, task.start, task.end, got.len(), want.len());
            std::process::exit(1);
        }
    }
    CLICKS.store(saved, Ordering::Relaxed);
    eprintln!(
        "clickcheck: OK {} tasks, {} lines, {} click records identical (avx2={}); {} lines took the general path, {} the 128-byte splitter",
        cl.tasks.len(),
        lines,
        recs,
        ctx.avx2,
        CSLOW.load(Ordering::Relaxed),
        CWIDE_N.load(Ordering::Relaxed)
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
#[inline(never)]
unsafe fn task_fast_avx2<S: EvSink>(b: &[u8], task: Task, ctx: &Ctx, sink: &mut S, safe_end: usize) {
    event_task_fast::<true, S>(b, task, ctx, sink, safe_end)
}

/// `event_task_fast` with run-time dispatch (as `main::event_task_avx2`).
#[inline(never)]
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
/// path; the copies must be identical, in the same order. Exits nonzero on
/// a difference.
pub fn check(ev: &super::Input, ctx: &Ctx) {
    COUNT_SLOW.store(true, Ordering::Relaxed);
    let mut lines = 0usize;
    let mut copies = 0usize;
    for task in &ev.tasks {
        let b = ev.maps[task.file as usize].bytes();
        let safe_end = b.as_ptr() as usize + b.len();
        let mut want = RecSink::default();
        super::event_task_scan_any(b, *task, ctx, &mut want, safe_end);
        lines += b[task.start..task.end].iter().filter(|&&c| c == b'\n').count();
        copies += want.0.len();
        let mut got = RecSink::default();
        task_fast_any(b, *task, ctx, &mut got, safe_end);
        if got.0 != want.0 {
            eprintln!("evcheck: MISMATCH file {} [{}, {}) ({} vs {} copies)", task.file, task.start, task.end, got.0.len(), want.0.len());
            std::process::exit(1);
        }
    }
    eprintln!(
        "evcheck: OK {} tasks, {} lines, {} copies identical (avx2={}); {} lines took the general path, {} the 128-byte splitter",
        ev.tasks.len(),
        lines,
        copies,
        ctx.avx2,
        SLOW.load(Ordering::Relaxed),
        WIDE_N.load(Ordering::Relaxed)
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
                // a quarter of the IDs in the version 5 long form (prefix +
                // 32 hex digits), sometimes one digit more or less
                let n = if r() % 4 == 0 { 31 + r() % 5 } else { 1 + r() % 13 };
                let alpha: &[u8] = if r() % 20 == 0 { b"aZ_-09 \xc3\xa9" } else if n > 30 { b"0123456789abcdef" } else { b"abcdefghij0123456789" };
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

    /// The lines of b[start..end) through `for_each_line2` (fast lines and
    /// general-path ranges, in order).
    fn fast_rows<const AVX2: bool>(b: &[u8], start: usize, end: usize) -> Vec<Vec<Vec<u8>>> {
        let out = std::cell::RefCell::new(Vec::new());
        let safe_end = b.as_ptr() as usize + b.len();
        let push = |f: &[&[u8]; 7]| out.borrow_mut().push(f.iter().map(|s| s.to_vec()).collect::<Vec<_>>());
        for_each_line2::<AVX2, _, _>(b, start, end, safe_end, push, |from, to| scan::for_each_record_x::<7, _, AVX2>(b, from, to, push));
        out.into_inner()
    }

    /// The general scanner's lines (`fast` false) or the fast path's.
    fn rows(b: &[u8], start: usize, end: usize, fast: bool) -> Vec<Vec<Vec<u8>>> {
        if fast {
            return fast_rows::<false>(b, start, end);
        }
        let mut out = Vec::new();
        scan::for_each_record_x::<7, _, false>(b, start, end, |f: &[&[u8]; 7]| out.push(f.iter().map(|s| s.to_vec()).collect()));
        out
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
    unsafe fn rows_avx2(b: &[u8], start: usize, end: usize) -> Vec<Vec<Vec<u8>>> {
        fast_rows::<true>(b, start, end)
    }

    /// Same lines and fields as the general scanner, in the same order, whole
    /// buffers and line-aligned chunks, both SIMD paths where available.
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
            let want = rows(&t, 0, t.len(), false);
            total += want.len();
            assert_eq!(rows(&t, 0, t.len(), true), want, "round {}", round);
            #[cfg(target_arch = "x86_64")]
            if avx2 {
                assert_eq!(unsafe { rows_avx2(&t, 0, t.len()) }, want, "avx2 round {}", round);
            }
            // line-aligned chunks
            let mut s0 = 0;
            let mut got = Vec::new();
            while s0 < t.len() {
                let nominal = s0 + 1 + (r() % 5000) as usize;
                let e = if nominal >= t.len() { t.len() } else { scan::next_line_start(&t, nominal - 1) };
                got.extend(rows(&t, s0, e, true));
                s0 = e;
            }
            assert_eq!(got, want, "chunked round {}", round);
        }
        assert!(total > 100_000, "{}", total);
    }

    fn click_text(r: &mut impl FnMut() -> u64, lines: usize) -> Vec<u8> {
        let mut t = Vec::new();
        for _ in 0..lines {
            let mode = r() % 40;
            if mode == 0 {
                t.push(b'\n');
                continue;
            }
            let nf = match mode {
                1 => 3,
                2 => 5,
                3 => 1 + (r() % 8) as usize,
                _ => 4,
            };
            for j in 0..nf {
                if j > 0 {
                    t.push(b',');
                }
                let v: Vec<u8> = match (j, r() % 30) {
                    (_, 0) => (0..r() % 30).map(|_| b"ab,\"\n\r 0Z-:T"[(r() % 12) as usize]).collect(),
                    (_, 1) => Vec::new(),
                    (_, 2) => (0..40 + r() % 40).map(|_| b"xyz019"[(r() % 6) as usize]).collect(),
                    (0, 3..=9) | (1, 3..=9) => {
                        // version 5 long form, sometimes a digit more or less
                        let mut v: Vec<u8> = (0..31 + r() % 5).map(|_| b"0123456789abcdef"[(r() % 16) as usize]).collect();
                        v[0] = if j == 0 { b'k' } else { b'u' };
                        v
                    }
                    (0, _) | (1, _) => (0..1 + r() % 12).map(|_| b"kuabcdefghij0123456789"[(r() % 22) as usize]).collect(),
                    (2, _) => format!("c_{}", r() % 2500).into_bytes(),
                    _ => value(2, r),
                };
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

    fn click_rows<const AVX2: bool>(b: &[u8], start: usize, end: usize, fast: bool) -> Vec<Vec<Vec<u8>>> {
        let mut out = Vec::new();
        let mut push = |f: &[&[u8]; 4]| out.push(f.iter().map(|s| s.to_vec()).collect::<Vec<_>>());
        let safe_end = b.as_ptr() as usize + b.len();
        if fast {
            for_each_click::<AVX2, _>(b, start, end, safe_end, &mut push);
        } else {
            scan::for_each_record_x::<4, _, AVX2>(b, start, end, push);
        }
        out
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
    unsafe fn click_rows_avx2(b: &[u8], start: usize, end: usize) -> Vec<Vec<Vec<u8>>> {
        click_rows::<true>(b, start, end, true)
    }

    /// The click splitter yields the general scanner's lines and fields, in
    /// order: adversarial text, whole and in line-aligned chunks, both SIMD
    /// paths where available, and every single-byte change of a click line.
    #[test]
    fn click_splitter_matches_scanner() {
        let mut r = rng(0xC11C_C11C_0000_0001);
        #[cfg(target_arch = "x86_64")]
        let avx2 = std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("bmi2");
        let mut total = 0usize;
        for round in 0..600 {
            let n = match round % 4 {
                0 => (r() % 8) as usize,
                1 => (r() % 200) as usize,
                _ => 300 + (r() % 4000) as usize,
            };
            let t = click_text(&mut r, n);
            let want = click_rows::<false>(&t, 0, t.len(), false);
            total += want.len();
            assert_eq!(click_rows::<false>(&t, 0, t.len(), true), want, "round {}", round);
            #[cfg(target_arch = "x86_64")]
            if avx2 {
                assert_eq!(unsafe { click_rows_avx2(&t, 0, t.len()) }, want, "avx2 round {}", round);
            }
            let mut s0 = 0;
            let mut got = Vec::new();
            while s0 < t.len() {
                let nominal = s0 + 1 + (r() % 3000) as usize;
                let e = if nominal >= t.len() { t.len() } else { scan::next_line_start(&t, nominal - 1) };
                got.extend(click_rows::<false>(&t, s0, e, true));
                s0 = e;
            }
            assert_eq!(got, want, "chunked round {}", round);
        }
        assert!(total > 100_000, "{}", total);
        let line = b"\"kc1ni90tn\",ucnon1osd,\"c_1190\",\"2027-12-06T10:41:30+05:30\"\n";
        for pos in 0..line.len() {
            for v in [b',', b'"', b'\n', b'\r', b'x', b' ', 0u8, 0xFF] {
                let mut l = line.to_vec();
                l[pos] = v;
                let mut t = b"k1,u1,c_2,1829024243000\n".repeat(3);
                t.extend_from_slice(&l);
                t.extend_from_slice(&click_text(&mut r, 6));
                if t.last() != Some(&b'\n') {
                    t.push(b'\n');
                }
                t.extend_from_slice(&[b'z'; 200]);
                t.push(b'\n');
                assert_eq!(click_rows::<false>(&t, 0, t.len(), true), click_rows::<false>(&t, 0, t.len(), false), "pos {} byte {}", pos, v);
            }
        }
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
                assert_eq!(rows(&t, 0, t.len(), true), rows(&t, 0, t.len(), false), "pos {} byte {}", pos, v);
            }
        }
    }
}
