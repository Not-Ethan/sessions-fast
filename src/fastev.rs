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
//! construction. The fields then go through the unchanged `event_line`
//! (`event_fast`, opt-in, is the same logic with experimental decoders).
//! The next line start is taken from the newline masks of the 128 bytes at
//! the line start, which keeps the line-to-line dependency chain short; the
//! field walk must find the same newline.
//!
//! The fast path never reads beyond `b` (nor beyond `safe_end`).

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use super::{event_line, event_line_slow, ingest_rel, Ctx, EvSink, Task, LATE};
use crate::{keys, scan, ts};

/// Use this path for event tasks (SS_FASTEV; see `main::run`).
pub static ON: AtomicBool = AtomicBool::new(false);
/// Interleaved cursors per task, 1 or 2 (SS_FASTK).
pub static CURSORS: AtomicUsize = AtomicUsize::new(1);
/// Decode fast-path lines with `event_fast` instead of `event_line`
/// (SS_FASTDEC=1).
pub static DEC: AtomicBool = AtomicBool::new(false);
/// With DEC: event ts through `ts_sel` (SS_FASTSEL=1).
pub static SEL: AtomicBool = AtomicBool::new(false);
/// With DEC and AVX2: event_id and user_id packed together (`pack2_avx2`,
/// SS_FASTPK=1).
pub static PK: AtomicBool = AtomicBool::new(false);

/// Lines handed to the general scanner by `slow_line` (statistics).
pub static SLOW: AtomicUsize = AtomicUsize::new(0);

/// Readable bytes needed from a line start for the fast path: window 1 (64),
/// window 2 at most 65 bytes further (64), and 32 bytes from the start of
/// any field (the WIDE decoders), with slack.
const MARGIN: usize = 192;

// ---------------------------------------------------------------- masks

/// Unaligned 32-byte load as one instruction: for the x86-64-v2 target the
/// compiler splits `_mm256_loadu_si256` into two 16-byte loads and an insert
/// (build with SS_NOASMLOAD set to get that back, for comparison).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load32(p: *const u8) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    if option_env!("SS_NOASMLOAD").is_some() {
        return _mm256_loadu_si256(p as *const __m256i);
    }
    let v: __m256i;
    std::arch::asm!("vmovdqu {0}, ymmword ptr [{1}]", out(ymm_reg) v, in(reg) p, options(pure, readonly, nostack, preserves_flags));
    v
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn masks64_avx2(p: *const u8) -> (u64, u64, u64) {
    use std::arch::x86_64::*;
    let a = load32(p);
    let b = load32(p.add(32));
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

// ---------------------------------------------------------------- decoding

/// One exact date: `YYYY-MM-` as a little-endian word, `DD` (2 bytes), and
/// its midnight in epoch ms.
#[derive(Clone, Copy, Default)]
struct DSlot {
    w0: u64,
    dd: u64,
    ms: i64,
}

/// Exact-date table for F1 timestamps: the days from 2 before the window to
/// 6 after it (at most 64), in slots indexed by the day-of-month digits
/// (`dslot`). A slot matches only if all ten bytes `YYYY-MM-DD` equal its
/// date, which is valid by construction (and checked against
/// `ts::days_from_civil`); unused slots repeat a real entry, so a match is
/// always exact. Dates that are not in the table take `TsParser::parse`.
pub struct Days {
    key: (i64, i64),
    on: bool,
    slots: [DSlot; 64],
}

#[inline(always)]
fn dslot(w1: u64) -> usize {
    // (d0 & 3) << 4 | (d1 & 15): distinct for "01".."31"
    (((w1 & 3) << 4) | ((w1 >> 8) & 15)) as usize
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

impl Days {
    pub fn new(ws: i64, we: i64) -> Days {
        let mut slots = [DSlot::default(); 64];
        let mut used = [false; 64];
        let mut first: Option<DSlot> = None;
        let lo = ws.div_euclid(86_400_000) - 2;
        let hi = we.div_euclid(86_400_000) + 6;
        let mut day = lo;
        while day <= hi && day < lo + 64 {
            let (y, m, d) = civil_from_days(day);
            if (0..=9999).contains(&y) && ts::days_from_civil(y, m, d) == day {
                let txt = format!("{:04}-{:02}-{:02}", y, m, d);
                let t = txt.as_bytes();
                let w0 = u64::from_le_bytes(t[..8].try_into().unwrap());
                let w1 = u16::from_le_bytes([t[8], t[9]]) as u64;
                let e = DSlot { w0, dd: w1, ms: day * 86_400_000 };
                let k = dslot(w1);
                slots[k] = e;
                used[k] = true;
                first.get_or_insert(e);
            }
            day += 1;
        }
        if let Some(e) = first {
            for k in 0..64 {
                if !used[k] {
                    slots[k] = e;
                }
            }
        }
        Days { key: (ws, we), on: first.is_some(), slots }
    }
}

thread_local! {
    static DAYS: std::cell::RefCell<Option<Box<Days>>> = const { std::cell::RefCell::new(None) };
}

/// Bytes `at..at + 8` of `p` as a little-endian u64.
#[inline(always)]
unsafe fn rd(p: *const u8, at: usize) -> u64 {
    u64::from_le(std::ptr::read_unaligned(p.add(at) as *const u64))
}

/// Nonzero iff some byte lane selected by `lanes` is not an ASCII digit.
#[inline(always)]
fn digits_bad(w: u64, lanes: u64) -> u64 {
    let hi = (w & 0xF0F0_F0F0_F0F0_F0F0) ^ 0x3030_3030_3030_3030;
    let lo = ((w & 0x0F0F_0F0F_0F0F_0F0F) + 0x0606_0606_0606_0606) & 0xF0F0_F0F0_F0F0_F0F0;
    (hi | lo) & lanes
}

/// An F1 timestamp `YYYY-MM-DDTHH:MM:SSZ` (20 bytes at `b`, the `Z`
/// already checked) through the date table: None if the date is not in the
/// table; else Some(parse_ts result), which is Some(ms) iff `T`, the colons,
/// the six time digits and h <= 23, m <= 59, s <= 59 are right.
#[inline(always)]
unsafe fn f1(days: &Days, b: *const u8) -> Option<Option<i64>> {
    let w0 = rd(b, 0); // YYYY-MM-
    let w1 = rd(b, 8); // DDTHH:MM
    let e = days.slots.get_unchecked(dslot(w1));
    let hit = days.on & (e.w0 == w0) & (e.dd == (w1 & 0xFFFF));
    let w = rd(b, 11); // HH:MM:SS
    let bad = digits_bad(w, 0xFFFF_00FF_FF00_FFFF) | ((w & 0x0000_FF00_00FF_0000) ^ 0x0000_3A00_003A_0000) | (((w1 >> 16) & 0xFF) ^ b'T' as u64);
    // lane i = 10 * digit_i + digit_{i+1}: h in lane 0, m in lane 3, s in 6
    let v = w & 0x0F0F_0F0F_0F0F_0F0F;
    let pr = v.wrapping_mul(10).wrapping_add(v >> 8);
    // lane + (128 - limit) has its top bit set iff the value is over the limit
    // (values <= 99 when the digits are good: no carry between lanes)
    let over = pr.wrapping_add(0x0044_0000_4400_0068) & 0x0080_0000_8000_0080;
    let tod = (pr & 0xFF) * 3600 + ((pr >> 24) & 0xFF) * 60 + ((pr >> 48) & 0xFF);
    if !hit {
        return None;
    }
    Some(if (bad | over) == 0 { Some(e.ms + tod as i64 * 1000) } else { None })
}

/// `tsp.parse(b)` (= `ts::parse_ts`), F1 through the date table. 32 bytes
/// from the start of `b` must be readable.
#[inline(always)]
fn ts_fast(tsp: &ts::TsParser, days: &Days, b: &[u8]) -> Option<i64> {
    if b.len() == 20 && b[19] == b'Z' {
        if let Some(v) = unsafe { f1(days, b.as_ptr()) } {
            return v;
        }
    }
    tsp.parse(b)
}

/// 8 ASCII digits (first char in the lowest byte) to their value (as ts.rs).
#[inline(always)]
fn parse8(w: u64) -> u64 {
    let l = w.wrapping_sub(0x3030_3030_3030_3030);
    let c = (l.wrapping_mul(10).wrapping_add(l >> 8)) & 0x00FF_00FF_00FF_00FF;
    let c = (c.wrapping_mul(100).wrapping_add(c >> 16)) & 0x0000_FFFF_0000_FFFF;
    (c.wrapping_mul(10000).wrapping_add(c >> 32)) & 0xFFFF_FFFF
}

/// `ts_fast` with F1 (date table) and 13-digit F3 decoded side by side and
/// selected by length, so the F1/F3 choice needs no branch; other shapes and
/// F1 dates outside the table take `tsp.parse`. 32 bytes from the start of
/// `b` must be readable.
#[inline(always)]
fn ts_sel(tsp: &ts::TsParser, days: &Days, b: &[u8]) -> Option<i64> {
    let n = b.len();
    let p = b.as_ptr();
    unsafe {
        let is1 = (n == 20) & (*p.add(19) == b'Z');
        let is3 = n == 13;
        if !(is1 | is3) {
            return tsp.parse(b);
        }
        let w0 = rd(p, 0);
        let w1 = rd(p, 8);
        let e = days.slots.get_unchecked(dslot(w1));
        let hit = days.on & (e.w0 == w0) & (e.dd == (w1 & 0xFFFF));
        let w = rd(p, 11);
        let bad = digits_bad(w, 0xFFFF_00FF_FF00_FFFF) | ((w & 0x0000_FF00_00FF_0000) ^ 0x0000_3A00_003A_0000) | (((w1 >> 16) & 0xFF) ^ b'T' as u64);
        let v = w & 0x0F0F_0F0F_0F0F_0F0F;
        let pr = v.wrapping_mul(10).wrapping_add(v >> 8);
        let over = pr.wrapping_add(0x0044_0000_4400_0068) & 0x0080_0000_8000_0080;
        let tod = (pr & 0xFF) * 3600 + ((pr >> 24) & 0xFF) * 60 + ((pr >> 48) & 0xFF);
        let ok1 = (bad | over) == 0;
        let v1 = e.ms + tod as i64 * 1000;
        // F3: 13 digits, as TsParser::parse
        let wb = rd(p, 5);
        let ok3 = (digits_bad(w0, !0) | digits_bad(wb, !0)) == 0;
        let v3 = (parse8(w0) * 100_000 + parse8((wb & !0xFF_FFFF) | 0x30_3030)) as i64;
        if is1 & !hit {
            return tsp.parse(b);
        }
        let ok = if is3 { ok3 } else { ok1 };
        let v = if is3 { v3 } else { v1 };
        if ok {
            Some(v)
        } else {
            None
        }
    }
}

/// `keys::pack_wide` of two strings at once, one per 128-bit lane: the same
/// operations lane by lane, so the same keys. 16 bytes from the start of
/// each must be readable.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn pack2_avx2(a: &[u8], b: &[u8]) -> (Option<u64>, Option<u64>) {
    use std::arch::x86_64::*;
    let (na, nb) = (a.len(), b.len());
    let v = _mm256_inserti128_si256::<1>(
        _mm256_castsi128_si256(_mm_loadu_si128(a.as_ptr() as *const __m128i)),
        _mm_loadu_si128(b.as_ptr() as *const __m128i),
    );
    let idx = _mm256_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    let nv = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(_mm_set1_epi8(na as i8)), _mm_set1_epi8(nb as i8));
    let inlen = _mm256_cmpgt_epi8(nv, idx);
    let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(v), _mm256_set1_epi8(0x0F));
    let lo = _mm256_and_si256(v, _mm256_set1_epi8(0x0F));
    let lut_hi = _mm256_setr_epi8(0, 0, 1, 2, 4, 8, 4, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 4, 8, 4, 16, 0, 0, 0, 0, 0, 0, 0, 0);
    let lut_lo = _mm256_setr_epi8(
        26, 30, 30, 30, 30, 30, 30, 30, 30, 30, 28, 4, 4, 5, 4, 12, 26, 30, 30, 30, 30, 30, 30, 30, 30, 30, 28, 4, 4, 5, 4, 12,
    );
    let cls = _mm256_and_si256(_mm256_shuffle_epi8(lut_hi, hi), _mm256_shuffle_epi8(lut_lo, lo));
    let invalid = _mm256_and_si256(_mm256_cmpeq_epi8(cls, _mm256_setzero_si256()), inlen);
    let off = _mm256_shuffle_epi8(
        _mm256_setr_epi8(
            0, 0, -0x2D, -0x2F, -0x36, -0x36, -0x3B, -0x3B, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0x2D, -0x2F, -0x36, -0x36, -0x3B, -0x3B, 0, 0, 0, 0, 0,
            0, 0, 0,
        ),
        hi,
    );
    let us = _mm256_and_si256(_mm256_cmpeq_epi8(v, _mm256_set1_epi8(0x5F)), _mm256_set1_epi8(-4));
    let codes = _mm256_and_si256(_mm256_add_epi8(_mm256_add_epi8(v, off), us), inlen);
    let t = _mm256_maddubs_epi16(codes, _mm256_set1_epi16(0x0140));
    let u = _mm256_madd_epi16(t, _mm256_set1_epi32(0x0001_1000));
    let l = |x: i32| x as u32 as u64;
    let (l0, l1, l2) = (l(_mm256_extract_epi32::<0>(u)), l(_mm256_extract_epi32::<1>(u)), l(_mm256_extract_epi32::<2>(u)));
    let (l4, l5, l6) = (l(_mm256_extract_epi32::<4>(u)), l(_mm256_extract_epi32::<5>(u)), l(_mm256_extract_epi32::<6>(u)));
    let ka = (((l0 << 36) | (l1 << 12) | (l2 >> 12)) << 4) | na as u64;
    let kb = (((l4 << 36) | (l5 << 12) | (l6 >> 12)) << 4) | nb as u64;
    let mm = _mm256_movemask_epi8(invalid) as u32;
    let oka = ((mm & 0xFFFF) == 0) & (na.wrapping_sub(1) < 10);
    let okb = ((mm >> 16) == 0) & (nb.wrapping_sub(1) < 10);
    (if oka { Some(ka) } else { None }, if okb { Some(kb) } else { None })
}

/// `event_line::<true>` for a fast-path line (f[0] non-empty, 32 readable
/// bytes from every field start), with `ts_fast` for both timestamps
/// (`ts_sel` for ts if `sel`) and, if `pk` (AVX2 only), both IDs packed
/// together. Same steps, same order, same results.
#[inline(always)]
fn event_fast<const AVX2: bool, S: EvSink>(f: &[&[u8]; 7], ctx: &Ctx, days: &Days, sink: &mut S, sel: bool, pk: bool) {
    let (ws, we) = (ctx.ws, ctx.we);
    let Some(ing) = ts_fast(&ctx.tsp, days, f[3]) else { return };
    #[allow(unused_mut)]
    let (mut ka, mut kb) = (None, None);
    #[cfg(target_arch = "x86_64")]
    if AVX2 && pk {
        (ka, kb) = unsafe { pack2_avx2(f[0], f[1]) };
    }
    let _ = pk;
    let eid = keys::mix(match ka {
        Some(k) => k,
        None => unsafe { keys::key_of_wide(f[0]) },
    });
    let ingest = ingest_rel(ing, ctx.ws_s);
    let valid = 'v: {
        if f[1].is_empty() || f[5].is_empty() {
            break 'v None;
        }
        let t = if sel { ts_sel(&ctx.tsp, days, f[2]) } else { ts_fast(&ctx.tsp, days, f[2]) };
        let Some(t) = t else { break 'v None };
        if t < ws || t >= we {
            break 'v None;
        }
        let Some(pay) = super::event_pay(f[4], || f[6], f[5], ctx, true) else { break 'v None };
        if ing - t > LATE {
            break 'v None;
        }
        let user = match kb {
            Some(k) => k,
            None => unsafe { keys::key_of_wide(f[1]) },
        };
        Some((user, (t - ws) as u32, pay))
    };
    sink.copy(eid, ingest, valid);
}

/// `main::event_task` on the fast splitter (same copies; with two cursors
/// in a different order).
#[inline(always)]
pub fn event_task_fast<const AVX2: bool, S: EvSink>(b: &[u8], task: Task, ctx: &Ctx, sink: &mut S, safe_end: usize) {
    let k = CURSORS.load(Ordering::Relaxed);
    if DEC.load(Ordering::Relaxed) {
        return event_task_dec::<AVX2, S>(b, task, ctx, sink, safe_end, k);
    }
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

/// `event_task_fast` with `event_fast` for the fast-path lines.
#[inline(always)]
fn event_task_dec<const AVX2: bool, S: EvSink>(b: &[u8], task: Task, ctx: &Ctx, sink: &mut S, safe_end: usize, k: usize) {
    let days: *const Days = DAYS.with(|c| {
        let mut c = c.borrow_mut();
        if c.as_ref().map_or(true, |d| d.key != (ctx.ws, ctx.we)) {
            *c = Some(Box::new(Days::new(ctx.ws, ctx.we)));
        }
        &**c.as_ref().unwrap() as *const Days
    });
    // the box lives in this thread's slot until a later task with another
    // window replaces it; this task is done with it by then
    let days = unsafe { &*days };
    let (sel, pk) = (SEL.load(Ordering::Relaxed), PK.load(Ordering::Relaxed));
    let sp: *mut S = sink;
    for_each_line2::<AVX2, _, _>(
        b,
        task.start,
        task.end,
        safe_end,
        k,
        #[inline(always)]
        |f| {
            if !f[0].is_empty() {
                event_fast::<AVX2, S>(f, ctx, days, unsafe { &mut *sp }, sel, pk);
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
    let saved_dec = DEC.load(Ordering::Relaxed);
    let saved_sel = SEL.load(Ordering::Relaxed);
    let saved_pk = PK.load(Ordering::Relaxed);
    let mut lines = 0usize;
    let mut copies = 0usize;
    for task in &ev.tasks {
        let b = ev.maps[task.file as usize].bytes();
        let safe_end = b.as_ptr() as usize + b.len();
        let mut want = RecSink::default();
        super::event_task_scan_any(b, *task, ctx, &mut want, safe_end);
        lines += b[task.start..task.end].iter().filter(|&&c| c == b'\n').count();
        copies += want.0.len();
        for (k, dec, sel, pk) in
            [(1, false, false, false), (2, false, false, false), (1, true, false, false), (2, true, false, false), (1, true, true, false), (1, true, false, true), (2, true, true, true)]
        {
            CURSORS.store(k, Ordering::Relaxed);
            DEC.store(dec, Ordering::Relaxed);
            SEL.store(sel, Ordering::Relaxed);
            PK.store(pk, Ordering::Relaxed);
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
                eprintln!("evcheck: MISMATCH file {} [{}, {}) cursors {} dec {} sel {} pk {} ({} vs {} copies)", task.file, task.start, task.end, k, dec, sel, pk, got.0.len(), want.0.len());
                std::process::exit(1);
            }
        }
    }
    CURSORS.store(saved, Ordering::Relaxed);
    DEC.store(saved_dec, Ordering::Relaxed);
    SEL.store(saved_sel, Ordering::Relaxed);
    PK.store(saved_pk, Ordering::Relaxed);
    eprintln!(
        "evcheck: OK {} tasks, {} lines, {} copies identical (cursors 1-2, event_line and event_fast variants, avx2={}); {} lines took the general path",
        ev.tasks.len(),
        lines,
        copies,
        ctx.avx2,
        SLOW.load(Ordering::Relaxed) / 7
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

    /// The date table + SWAR time decoder agrees with `ts::parse_ts` on F1
    /// shapes around several windows: every byte value at every position of
    /// valid timestamps, random mutations, all dates of the table range.
    #[test]
    fn f1_matches_parse_ts() {
        let mut r = rng(0xF1F1);
        let mut hits = 0u64;
        for wi in 0..40 {
            // windows from 2025 to 2028 (Feb 29 2028 included), up to 24 days
            let ws = ts::parse_ts(b"2025-01-01T00:00:00Z").unwrap() + ((r() % (4 * 366)) as i64) * 86_400_000;
            let ws = if wi == 0 { ts::parse_ts(b"2028-02-20T00:00:00Z").unwrap() } else { ws };
            let we = ws + (1 + (r() % 24) as i64) * 86_400_000;
            let days = Days::new(ws, we);
            let tsp = ts::TsParser::new(ws - 10 * 86_400_000, we + 10 * 86_400_000);
            // ts_sel / ts_fast on any field (32 readable bytes after its start)
            let check_any = |s: &[u8]| {
                let mut buf = s.to_vec();
                buf.extend_from_slice(&[b'7'; 40]);
                let b = &buf[..s.len()];
                let want = ts::parse_ts(b);
                assert_eq!(ts_sel(&tsp, &days, b), want, "sel {:?}", String::from_utf8_lossy(b));
                assert_eq!(ts_fast(&tsp, &days, b), want, "fast {:?}", String::from_utf8_lossy(b));
            };
            for _ in 0..3000 {
                let ms = ws - 2 * 86_400_000 + (r() % (30 * 86_400_000)) as i64;
                let mut t = match r() % 4 {
                    0 => format!("{}", ms),
                    1 => format!("{:013}", r() % 10_000_000_000_000),
                    2 => format!("{:0w$}", ms, w = 13 + (r() % 4) as usize),
                    _ => {
                        let (y, m, d) = civil_from_days(ms.div_euclid(86_400_000));
                        let sod = ms.rem_euclid(86_400_000) / 1000;
                        format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}{}", y, m, d, sod / 3600, sod / 60 % 60, sod % 60,
                                ["Z", "+01:00", "-10:30", "z", ""][(r() % 5) as usize])
                    }
                }
                .into_bytes();
                if r() % 3 == 0 && !t.is_empty() {
                    let at = (r() as usize) % t.len();
                    t[at] = b"0123456789Z:-T. x\"\xff"[(r() % 19) as usize];
                }
                check_any(&t);
            }
            let check = |s: &[u8], hits: &mut u64| {
                check_any(s);
                let mut buf = s.to_vec();
                buf.extend_from_slice(&[b'9'; 40]);
                let b = &buf[..20];
                if b[19] != b'Z' {
                    return; // f1's precondition (ts_fast checks it)
                }
                let want = ts::parse_ts(b);
                let got = unsafe { f1(&days, b.as_ptr()) };
                if let Some(v) = got {
                    *hits += 1;
                    assert_eq!(v, want, "{:?}", String::from_utf8_lossy(b));
                }
            };
            for day in (ws / 86_400_000 - 5)..(we / 86_400_000 + 9) {
                let (y, m, d) = civil_from_days(day);
                let sod = (r() % 86_400) as i64;
                let base = format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, m, d, sod / 3600, sod / 60 % 60, sod % 60).into_bytes();
                check(&base, &mut hits);
                for at in 0..20 {
                    for v in [b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'9', b':', b'-', b'T', b'/', b' ', 0, 0xFF, b'Z', b'"'] {
                        let mut t = base.clone();
                        t[at] = v;
                        check(&t, &mut hits);
                    }
                }
                for (h, mi, s) in [(23, 59, 59), (24, 0, 0), (0, 60, 0), (0, 0, 60), (19, 0, 0), (29, 0, 0), (99, 99, 99), (0, 0, 0)] {
                    let t = format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, m, d, h, mi, s).into_bytes();
                    check(&t, &mut hits);
                }
            }
            for _ in 0..20_000 {
                let mut t = format!(
                    "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
                    2024 + r() % 6, r() % 14, r() % 33, r() % 26, r() % 62, r() % 62
                )
                .into_bytes();
                if r() % 2 == 0 {
                    let at = (r() % 20) as usize;
                    t[at] = (r() & 0xFF) as u8;
                }
                check(&t, &mut hits);
            }
        }
        assert!(hits > 100_000, "{}", hits);
    }

    /// `pack2_avx2` gives `keys::pack` of each string, whatever follows it.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn pack2_matches_pack() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut r = rng(0x9AC2);
        let alpha: &[u8] = b"-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz";
        let one = |r: &mut dyn FnMut() -> u64| -> Vec<u8> {
            let n = (r() % 14) as usize;
            (0..n)
                .map(|_| {
                    let x = r();
                    match x % 8 {
                        0 => (x >> 8) as u8,
                        1 => alpha[((x >> 8) % 64) as usize].wrapping_add(((x >> 16) % 3) as u8).wrapping_sub(1),
                        _ => alpha[((x >> 8) % 64) as usize],
                    }
                })
                .collect()
        };
        let mut ok = 0u64;
        for round in 0..2_000_000u64 {
            let a = one(&mut r);
            let b = one(&mut r);
            let mut buf = a.clone();
            buf.extend((0..20).map(|_| if round % 2 == 0 { (r() & 0xFF) as u8 } else { alpha[(r() % 64) as usize] }));
            let off = buf.len();
            buf.extend_from_slice(&b);
            buf.extend((0..20).map(|_| (r() & 0xFF) as u8));
            let (ka, kb) = unsafe { pack2_avx2(&buf[..a.len()], &buf[off..off + b.len()]) };
            assert_eq!(ka, keys::pack(&a), "{:?}", a);
            assert_eq!(kb, keys::pack(&b), "{:?}", b);
            ok += ka.is_some() as u64;
        }
        assert!(ok > 500_000, "{}", ok);
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
