#![cfg_attr(not(target_arch = "x86_64"), allow(dead_code, unused_imports))]
//! k4: a fused event-line kernel for `main::event_task` (SS_K4).
//!
//! The old path per event line: the line splitter (fastev.rs `cut`, else
//! `cut_wide`, else the general scanner) hands the seven unquoted fields to
//! `main::event_line`, which parses ingest_ts and ts with the general
//! timestamp parser, keys event_id and user_id, checks type / product /
//! price (`event_pay`) and hands one copy to the sink.
//!
//! k4 splits the line the same way (its own splitter, `split`; see below)
//! and then decodes every field it needs in one branch-light pass:
//! * ingest_ts: F1 whose date `YYYY-MM-DDT` is byte-exact one of the 28
//!   dates of the kernel's date table (`DTab`, built from the window and
//!   checked against `ts::parse_ts`) and whose `HH:MM:SS` is well-formed and
//!   in range (`tvec`, both timestamps' times in one ymm);
//! * ts: F1 / F2 with the date in the table, or exactly 13 digits (F3);
//! * event type exactly as `event_pay` (wide), user / product emptiness;
//! * the window and the late limit exactly as `event_line`, on offsets from
//!   window_start;
//! * event_id and (valid copies only) user_id through `keys::pack_wide`,
//!   and for every ID that does not pack through `keys::other_ev` /
//!   `keys::other_main` (exactly `key_of_ev_wide` / `key_of_wide`: long
//!   IDs hashed and logged, the rest interned);
//! * purchases: `main::event_pay` itself (out of line).
//! A copy is produced by the kernel only when every input of the decision is
//! certain: ingest_ts decoded, and ts decoded unless the copy is certainly
//! invalid for another reason (empty user / product, a bad type). Every other
//! cut line goes to `main::event_line` with its fields (the old routine), and
//! a line the splitter cannot cut to the general scanner as a one-line
//! range, exactly as `fastev::event_task_fast`. Lines within `MARGIN` bytes
//! of the end of the buffer go through `fastev::event_task_fast` itself.
//! Side effects (long-ID logs, the interner) happen for the same lines in
//! the same order as on the old path: event_id is keyed iff ingest_ts
//! parses, user_id iff the copy is valid.
//!
//! The checker (SS_K4CHECK=1, `check`) runs every event task through the old
//! path and through k4 into recording sinks and compares the copies.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::{Ctx, EvSink, Input, Task, DAY};
#[cfg(target_arch = "x86_64")]
use super::{event_line, LATE};
use crate::{fastev, scan};

/// event tasks through k4 (SS_K4; needs AVX2, see `main::event_task`).
pub static ON: AtomicBool = AtomicBool::new(false);
/// Count lines per outcome (the checker; off in timed runs).
pub static COUNT: AtomicBool = AtomicBool::new(false);
pub static N_FAST: AtomicUsize = AtomicUsize::new(0);
pub static N_LINE: AtomicUsize = AtomicUsize::new(0);
pub static N_RANGE: AtomicUsize = AtomicUsize::new(0);
pub static N_TAIL: AtomicUsize = AtomicUsize::new(0);

/// Readable bytes from a line start for the kernel: the wide cut (256),
/// and 32 bytes from the start of any field it yields.
const MARGIN: usize = 288;

// ---------------------------------------------------------------- date table

/// One date of the table.
#[derive(Clone, Copy)]
#[repr(C, align(32))]
struct DEnt {
    /// bytes 0..8 `YYYY-MM-` (little-endian)
    pat: u64,
    /// bytes 8..11 `DDT` (24 bits; 0xFFFF_FFFF for an unused slot: never matches)
    dd: u32,
    /// `main::ingest_rel` of the day's midnight (whole seconds; near range
    /// for every second of the day)
    rel_s: u32,
    /// midnight - window_start (ms)
    off_ms: i64,
    _p: i64,
}

/// 64 slots indexed by the day-of-month digits (`dslot`): 28 consecutive
/// dates never share a day of month, so each has its own slot.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct DTab([DEnt; 64]);

/// Slot of the date bytes 8..12 as a little-endian u32 (bytes 8, 9 = `DD`):
/// low 2 bits of the tens digit, low 4 bits of the units digit (distinct
/// for "00".."39").
#[inline(always)]
fn dslot(d4: u32) -> usize {
    ((d4 & 3) | ((d4 >> 6) & 0x3C)) as usize
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

impl DTab {
    /// The 28 dates from `first_day` (epoch days). A date is left out (its
    /// slot never matches) unless `ts::parse_ts` gives its midnight exactly,
    /// every second of it is in `ingest_rel`'s near range and its offset from
    /// window_start is representable.
    fn new(first_day: i64, ws: i64, ws_s: i64) -> DTab {
        let mut t = DTab([DEnt { pat: 0, dd: 0xFFFF_FFFF, rel_s: 0, off_ms: 0, _p: 0 }; 64]);
        for day in first_day..first_day + 28 {
            let (y, m, d) = civil_from_days(day);
            if !(0..=9999).contains(&y) {
                continue;
            }
            let txt = format!("{:04}-{:02}-{:02}T00:00:00Z", y, m, d);
            let tb = txt.as_bytes();
            let Some(ms) = day.checked_mul(DAY) else { continue };
            if crate::ts::parse_ts(tb) != Some(ms) {
                continue;
            }
            let dsec = ms.div_euclid(1000) - ws_s;
            if !(-(1i64 << 30)..(1i64 << 30) - 86_400).contains(&dsec) {
                continue;
            }
            let Some(off) = ms.checked_sub(ws) else { continue };
            if off.checked_add(DAY).is_none() || off.checked_sub(DAY).is_none() {
                continue;
            }
            let pat = u64::from_le_bytes(tb[..8].try_into().unwrap());
            let dd = u32::from_le_bytes([tb[8], tb[9], tb[10], 0]);
            let k = dslot(u32::from_le_bytes([tb[8], tb[9], tb[10], tb[11]]));
            assert!(t.0[k].dd == 0xFFFF_FFFF, "k4: date slot collision");
            t.0[k] = DEnt { pat, dd, rel_s: (dsec + (1i64 << 31)) as u32, off_ms: off, _p: 0 };
        }
        t
    }
}

/// Kernel parameters of a window.
pub struct K4 {
    ws: i64,
    we: i64,
    /// we - ws (ms; 0 if empty)
    span: u64,
    tab: DTab,
}

static PARAMS: Mutex<Vec<&'static K4>> = Mutex::new(Vec::new());

/// The (cached) parameters for ctx's window.
pub fn params(ctx: &Ctx) -> &'static K4 {
    let mut v = PARAMS.lock().unwrap();
    for &k in v.iter() {
        if k.ws == ctx.ws && k.we == ctx.we {
            return k;
        }
    }
    let wd = ctx.ws.div_euclid(DAY);
    let k: &'static K4 = Box::leak(Box::new(K4 {
        ws: ctx.ws,
        we: ctx.we,
        span: (ctx.we.saturating_sub(ctx.ws)).max(0) as u64,
        // ts and ingest dates from 2 days before the window start
        tab: DTab::new(wd - 2, ctx.ws, ctx.ws_s),
    }));
    v.push(k);
    k
}

// ---------------------------------------------------------------- helpers

#[inline(always)]
unsafe fn rd64(p: *const u8) -> u64 {
    u64::from_le(std::ptr::read_unaligned(p as *const u64))
}

#[inline(always)]
unsafe fn rd32(p: *const u8) -> u32 {
    u32::from_le(std::ptr::read_unaligned(p as *const u32))
}

/// a if c else b, without a branch.
#[inline(always)]
fn sel(c: bool, a: u64, b: u64) -> u64 {
    let m = (c as u64).wrapping_neg();
    (a & m) | (b & !m)
}

/// The date `YYYY-MM-DDT` at q looked up in `tab`: (nonzero unless the 11
/// bytes are exactly a table date, its entry). Reads q..q+12.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn date(q: *const u8, tab: &DTab) -> (u64, &DEnt) {
    let a = rd64(q);
    let d4 = rd32(q.add(8));
    let e = tab.0.get_unchecked(dslot_pext(d4));
    ((a ^ e.pat) | ((d4 & 0xFF_FFFF) ^ e.dd) as u64, e)
}

/// `dslot` in one instruction: PEXT of bits 0, 1 and 8..12 gives the same
/// six bits in the same order ((d4 & 3) | ((d4 >> 6) & 0x3C)). BMI2: only
/// inside the kernel's target-feature context (k4 runs only with AVX2+BMI2).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn dslot_pext(d4: u32) -> usize {
    let k = _pext_u32(d4, 0x0F03) as usize;
    debug_assert_eq!(k, dslot(d4));
    k
}

/// The 16 bytes from a + 11 (ts) and from b + 11 (ingest_ts) at once:
/// `HH:MM:SS`, byte 19 (`Z` or the F2 sign) and the F2 offset `HH:MM`.
/// Returns (seconds of day of a, F2 offset minutes of a, seconds of day of
/// b, lane mask, range mask): lane mask bit i (a: 0..16, b: 16..32) is set
/// iff byte 11 + i is a digit where `HH:MM:SS?HH:MM` has a digit, or equals
/// the `:` / `Z` / `:` at 2, 5, 8, 11; range mask bits 2w, 2w+1 are set iff
/// word w of [hh, mm, ss, 0, oh, om, 0, 0] exceeds [23, 59, 59, 0, 23, 59, 0, 0]
/// (b: + 16). Values are meaningful only where the lanes are digits.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn tvec(a: *const u8, b: *const u8) -> (u32, u32, u32, u32, u32) {
    let va = _mm_loadu_si128(a.add(11) as *const __m128i);
    let vb = _mm_loadu_si128(b.add(11) as *const __m128i);
    let v = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(va), vb);
    let d = _mm256_sub_epi8(v, _mm256_set1_epi8(b'0' as i8));
    let isd = _mm256_cmpeq_epi8(_mm256_min_epu8(d, _mm256_set1_epi8(9)), d);
    const C: i8 = b':' as i8;
    const Z: i8 = b'Z' as i8;
    let pat = _mm256_setr_epi8(0, 0, C, 0, 0, C, 0, 0, Z, 0, 0, C, 0, 0, 0, 0, 0, 0, C, 0, 0, C, 0, 0, Z, 0, 0, C, 0, 0, 0, 0);
    let iseq = _mm256_cmpeq_epi8(v, pat);
    const E: i8 = -1;
    let eqlanes = _mm256_setr_epi8(0, 0, E, 0, 0, E, 0, 0, E, 0, 0, E, 0, 0, 0, 0, 0, 0, E, 0, 0, E, 0, 0, E, 0, 0, E, 0, 0, 0, 0);
    let okm = _mm256_movemask_epi8(_mm256_blendv_epi8(isd, iseq, eqlanes)) as u32;
    let dg = _mm256_shuffle_epi8(d, _mm256_setr_epi8(0, 1, 3, 4, 6, 7, -1, -1, 9, 10, 12, 13, -1, -1, -1, -1, 0, 1, 3, 4, 6, 7, -1, -1, 9, 10, 12, 13, -1, -1, -1, -1));
    let w = _mm256_maddubs_epi16(dg, _mm256_set1_epi16(0x010A)); // [hh, mm, ss, 0, oh, om, 0, 0]
    let rm = _mm256_movemask_epi8(_mm256_cmpgt_epi16(w, _mm256_setr_epi16(23, 59, 59, 0, 23, 59, 0, 0, 23, 59, 59, 0, 23, 59, 0, 0))) as u32;
    let s = _mm256_madd_epi16(w, _mm256_setr_epi16(3600, 60, 1, 0, 60, 1, 0, 0, 3600, 60, 1, 0, 60, 1, 0, 0));
    let t = _mm256_hadd_epi32(s, s); // per lane [sec, omin, sec, omin]
    let ra = _mm256_extract_epi64::<0>(t) as u64;
    let rb = _mm256_extract_epi64::<2>(t) as u64;
    (ra as u32, (ra >> 32) as u32, rb as u32, okm, rm)
}

/// 13 digits at q: (value, nonzero unless all 13 bytes are digits). Reads
/// q-3..q+13 (decode: field 2 starts at byte 3 or later, event_id being
/// non-empty).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn f3_13(q: *const u8) -> (u64, u32) {
    // lanes 3..16 = q[0..13] (lanes 0..3 = q[-3..0], masked out below)
    let v = _mm_loadu_si128(q.sub(3) as *const __m128i);
    let d = _mm_sub_epi8(v, _mm_set1_epi8(b'0' as i8));
    let ok = _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_min_epu8(d, _mm_set1_epi8(9)), d)) as u32;
    let dz = _mm_and_si128(d, _mm_setr_epi8(0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
    let w = _mm_maddubs_epi16(dz, _mm_set1_epi16(0x010A)); // 2-digit groups
    let g4 = _mm_madd_epi16(w, _mm_set1_epi32(0x0001_0064)); // [100, 1]: 4-digit groups
    let p = _mm_packus_epi32(g4, g4);
    let g8 = _mm_madd_epi16(p, _mm_set1_epi32(0x0001_2710)); // [10000, 1]: 8-digit groups
    let r = _mm_cvtsi128_si64(g8) as u64;
    ((r & 0xFFFF_FFFF) * 100_000_000 + (r >> 32), !ok & 0xFFF8)
}

/// `keys::pack_wide` of two strings at once, one per 128-bit lane (16
/// readable bytes from each start): (key a, a packs, key b, b packs), the
/// same operations as `pack_wide` lane by lane, so `oka == pack_wide(a).is_some()`
/// and then `ka == pack_wide(a).unwrap()` (likewise b).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn pack2(a: *const u8, na: usize, b: *const u8, nb: usize) -> (u64, bool, u64, bool) {
    let va = _mm_loadu_si128(a as *const __m128i);
    let vb = _mm_loadu_si128(b as *const __m128i);
    let v = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(va), vb);
    let idx = _mm256_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    let nv = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(_mm_set1_epi8(na as i8)), _mm_set1_epi8(nb as i8));
    let inlen = _mm256_cmpgt_epi8(nv, idx);
    let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(v), _mm256_set1_epi8(0x0F));
    let lo = _mm256_and_si256(v, _mm256_set1_epi8(0x0F));
    // the class and offset tables of keys::pack_wide in both lanes
    let lut_hi = _mm256_broadcastsi128_si256(_mm_setr_epi8(0, 0, 1, 2, 4, 8, 4, 16, 0, 0, 0, 0, 0, 0, 0, 0));
    let lut_lo = _mm256_broadcastsi128_si256(_mm_setr_epi8(
        2 | 8 | 16,
        2 | 4 | 8 | 16,
        2 | 4 | 8 | 16,
        2 | 4 | 8 | 16,
        2 | 4 | 8 | 16,
        2 | 4 | 8 | 16,
        2 | 4 | 8 | 16,
        2 | 4 | 8 | 16,
        2 | 4 | 8 | 16,
        2 | 4 | 8 | 16,
        4 | 8 | 16,
        4,
        4,
        1 | 4,
        4,
        4 | 8,
    ));
    let cls = _mm256_and_si256(_mm256_shuffle_epi8(lut_hi, hi), _mm256_shuffle_epi8(lut_lo, lo));
    let invalid = _mm256_and_si256(_mm256_cmpeq_epi8(cls, _mm256_setzero_si256()), inlen);
    let off = _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_setr_epi8(0, 0, -0x2D, -0x2F, -0x36, -0x36, -0x3B, -0x3B, 0, 0, 0, 0, 0, 0, 0, 0)), hi);
    let us = _mm256_and_si256(_mm256_cmpeq_epi8(v, _mm256_set1_epi8(0x5F)), _mm256_set1_epi8(-4));
    let codes = _mm256_and_si256(_mm256_add_epi8(_mm256_add_epi8(v, off), us), inlen);
    let t = _mm256_maddubs_epi16(codes, _mm256_set1_epi16(0x0140));
    let u = _mm256_madd_epi16(t, _mm256_set1_epi32(0x0001_1000));
    // per lane, dwords [l0, l1, l2, l3] (24-bit groups of 4 codes); the key of
    // pack_wide is (((l0 << 36) | (l1 << 12) | (l2 >> 12)) << 4) | n
    //   = l0 << 40 | l1 << 16 | ((l2 >> 12) << 4) | n: bytes 0..8 of the key
    // are l2 bytes 1, 2, l1 bytes 0..3, l0 bytes 0..3 (one byte shuffle); key
    // bits 0..4 then hold l2 bits 8..12, replaced by n
    let kv = _mm256_shuffle_epi8(u, _mm256_setr_epi8(9, 10, 4, 5, 6, 0, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1, 9, 10, 4, 5, 6, 0, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1));
    let ka = (_mm_cvtsi128_si64(_mm256_castsi256_si128(kv)) as u64 & !0xF) | na as u64;
    let kb = (_mm_cvtsi128_si64(_mm256_extracti128_si256::<1>(kv)) as u64 & !0xF) | nb as u64;
    let inv = _mm256_movemask_epi8(invalid) as u32;
    let oka = ((inv & 0xFFFF) == 0) & (na.wrapping_sub(1) < 10);
    let okb = ((inv >> 16) == 0) & (nb.wrapping_sub(1) < 10);
    (ka, oka, kb, okb)
}

// ---------------------------------------------------------------- split

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

/// fastev.rs's `window` (the dialect's quote rule over one window whose bit
/// 0 is a field start): (violations, content starts, content ends).
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

/// `window` in the shorter form (fs given; region r): with qr = q & r and qs
/// = qr & fs the quoted starts, the quotes in r must be exactly qs and the
/// closing quotes cl, so bad = (qr & !fs) ^ cl (a one-byte quoted field puts
/// its cl on a field start, which qr & !fs lacks); quoted starts are never
/// adjacent, nor are closing quotes, so qs * 3 = qs | qs << 1 and cl * 3 =
/// cl | cl << 1: cs = fs ^ (qs * 3), ce = d ^ (cl * 3). Same (bad == 0, and
/// then the same starts / ends inside r) as `window` (the checker and the
/// tests compare them).
#[inline(always)]
fn window_r(d: u64, q: u64, r: u64, fs: u64) -> (u64, u64, u64) {
    let qr = q & r;
    let qs = qr & fs;
    let cl = ((!d).wrapping_add(qs) & d) >> 1;
    let bad = (qr & !fs) ^ cl;
    let cs = fs ^ qs.wrapping_mul(3);
    let ce = d ^ cl.wrapping_mul(3);
    (bad, cs, ce)
}

/// (comma, newline, quote) masks of the 64 bytes at p.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn masks64(p: *const u8) -> (u64, u64, u64) {
    let a = _mm256_loadu_si256(p as *const __m256i);
    let b = _mm256_loadu_si256(p.add(32) as *const __m256i);
    let m = |v: __m256i, ch: u8| -> u64 { _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, _mm256_set1_epi8(ch as i8))) as u32 as u64 };
    (m(a, b',') | (m(b, b',') << 32), m(a, b'\n') | (m(b, b'\n') << 32), m(a, b'"') | (m(b, b'"') << 32))
}

/// 64 bits from bit `w` (1..=65) of the 128-bit mask (hi, lo).
#[inline(always)]
fn shift128(lo: u64, hi: u64, w: usize) -> u64 {
    ((((hi as u128) << 64) | lo as u128) >> (w & 127)) as u64
}

/// `fastev::cut` with window 2 taken from the masks of the 128 bytes at p
/// (no second set of loads): window 1 = the 64 bytes at p, its third
/// delimiter ends field 2 at w - 1 <= 63, so window 2 = bits [w, w + 64) of
/// those 128 bytes, exactly cut's window 2 (w + 64 <= 128). Hence the same
/// `ok`, fields and length as `cut` for every line (128 readable bytes).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[inline]
unsafe fn split(c0: u64, n0: u64, q0: u64, c1: u64, n1: u64, q1: u64) -> fastev::Cut {
    // window 1: fields 0..=2
    let d1 = c0 | n0;
    let d1c = blsr(blsr(d1));
    let r1 = upto(d1c);
    let (b1, cs1, ce1) = window(d1, q0, r1);
    // window 2: fields 3..=6 from the start of field 3
    let w = tz(d1c) + 1; // <= 65 (65: no third delimiter, not ok)
    let n2 = shift128(n0, n1, w);
    let d2 = shift128(c0, c1, w) | n2;
    let q2 = shift128(q0, q1, w);
    let d2d = blsr(blsr(blsr(d2)));
    let r2 = upto(d2d);
    let (b2, cs2, ce2) = window(d2, q2, r2);
    // the first six delimiters are commas, the seventh is the newline
    let last = d2d & d2d.wrapping_neg();
    let nl = (n0 & r1) | ((n2 ^ last) & r2);
    // the line end from the newline masks alone (off the field chain)
    let le = if n0 != 0 { tz(n0) } else { 64 + tz(n1) };
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
    fastev::Cut { ok, s, n, len: le + 1 }
}

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

/// `window` on 128 bits.
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

/// `fastev::cut_wide` from the masks of the 128 bytes at p (already
/// computed for `split`) and of the 64 bytes at p + 128: window 1 = those
/// 128 bytes; window 2 = bits [w, w + 64) of the 192 bytes (w <= 128 when
/// ok), exactly cut_wide's window 2 (its masks of the 64 bytes at p + w).
/// Hence the same `ok`, fields and length as `cut_wide` (192 readable bytes).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[inline(never)]
unsafe fn split_wide(p: *const u8, c0: u64, n0: u64, q0: u64, c1: u64, n1: u64, q1: u64) -> fastev::Cut {
    let (c2, n2, q2) = masks64(p.add(128));
    let c = c0 as u128 | ((c1 as u128) << 64);
    let n = n0 as u128 | ((n1 as u128) << 64);
    let q = q0 as u128 | ((q1 as u128) << 64);
    let d1 = c | n;
    let d1c = blsr128(blsr128(d1));
    let r1 = upto128(d1c);
    let (b1, cs1, ce1) = window128(d1, q, r1);
    let w = tz128(d1c) + 1; // <= 129 (129: not ok)
    let hiw = w >= 64;
    let pick = |x0: u64, x1: u64, x2: u64| if hiw { shift128(x1, x2, w - 64) } else { shift128(x0, x1, w) };
    let cw = pick(c0, c1, c2);
    let nw = pick(n0, n1, n2);
    let qw = pick(q0, q1, q2);
    let d2 = cw | nw;
    let d2d = blsr(blsr(blsr(d2)));
    let r2 = upto(d2d);
    let (b2, cs2, ce2) = window(d2, qw, r2);
    let last = d2d & d2d.wrapping_neg();
    let bad = (b1 != 0) | (n & r1 != 0) | ((b2 | ((nw ^ last) & r2)) != 0);
    let ok = (d1c != 0) & (d2d != 0) & !bad;
    let mut s = [0usize; 7];
    let mut ln = [0usize; 7];
    let (mut cs, mut ce) = (cs1, ce1);
    for j in 0..3 {
        s[j] = tz128(cs);
        ln[j] = tz128(ce).wrapping_sub(tz128(cs));
        cs = blsr128(cs);
        ce = blsr128(ce);
    }
    let (mut cs, mut ce) = (cs2, ce2);
    for j in 3..7 {
        s[j] = w + tz(cs);
        ln[j] = tz(ce).wrapping_sub(tz(cs));
        cs = blsr(cs);
        ce = blsr(ce);
    }
    fastev::Cut { ok, s, n: ln, len: w + tz(d2d) + 1 }
}

/// The lean split (variant 3): one set of masks of the 128 bytes at p, the
/// line end from the newline masks, fields 0..K from window 1 (the 64 bytes
/// at p) and fields K..=6 from window 2 = the 64 bytes that end at the line's
/// newline (bit i of window 2 = byte le - 63 + i). K = 2 (shape A, ~94% of
/// lines) or 3 (shape B: fields 2..=6 longer than 63 bytes, short IDs).
/// Requirements (else not ok; the caller tries shape B, then the old
/// splitters): a newline within the 128 bytes (le <= 127), the K-th delimiter
/// DK below 64 and le - DK <= 64 (window 2 holds every byte after DK),
/// exactly 6 - K delimiters strictly between DK and le, and the dialect's
/// quote rule (`window`) in both regions: window 1 over [0, DK], window 2
/// over (DK, le]. Then the line's delimiters before le are exactly six (all
/// commas, le being the first newline) and every field holds no quote or
/// exactly two, as its first and last byte: the fields and length of
/// `fastev::cut` / `cut_wide` for every line it accepts (the checker and the
/// tests compare them).
#[inline(always)]
unsafe fn split_lean<const K: usize>(c0: u64, n0: u64, q0: u64, c1: u64, n1: u64, q1: u64) -> fastev::Cut {
    let le = if n0 != 0 { tz(n0) } else { 64 + tz(n1) }; // 128: no newline
    let d0 = c0 | n0;
    let d1 = c1 | n1;
    // window 1: fields 0..K (region [0, DK])
    let mut x1 = blsr(d0);
    if K == 3 {
        x1 = blsr(x1);
    }
    let r1 = upto(x1);
    let (b1, cs1, ce1) = window_r(d0, q0, r1, (d0 << 1) | 1);
    let ek = tz(x1); // DK (64 if absent)
    // window 2: bit i = byte le - 63 + i (zeros below byte 0)
    let sh = 127usize.wrapping_sub(le) & 127;
    let dw = ((((d1 as u128) << 64) | d0 as u128) << sh >> 64) as u64;
    let qw = ((((q1 as u128) << 64) | q0 as u128) << sh >> 64) as u64;
    // DK at bit bk = bkp1 - 1 of window 2; bkp1 = 0: DK is the byte just
    // below window 2, whose bit 0 is then field K's start (`fs` bit 0)
    let bkp1 = (ek + 64).wrapping_sub(le); // valid iff <= 63
    let r2 = (!0u64).wrapping_shl(bkp1 as u32); // bits (bk, 63]
    let fs = (dw << 1) | 1;
    let (b2bad, cs2, ce2) = window_r(dw, qw, r2, fs);
    let (cs2, ce2) = (cs2 & r2, ce2 & r2);
    // delimiters in (DK, le]: 6 - K commas, then the newline at bit 63
    let y = dw & r2;
    let mut yr = blsr(blsr(blsr(y)));
    if K == 2 {
        yr = blsr(yr);
    }
    let ok = (le < 128) & (x1 != 0) & (bkp1 < 64) & (yr == 1u64 << 63) & ((b1 | b2bad) == 0);
    let mut s = [0usize; 7];
    let mut n = [0usize; 7];
    let (mut cs, mut ce) = (cs1, ce1);
    for j in 0..K {
        s[j] = tz(cs);
        n[j] = tz(ce).wrapping_sub(tz(cs));
        cs = blsr(cs);
        ce = blsr(ce);
    }
    let w = le.wrapping_sub(63);
    let (mut cs, mut ce) = (cs2, ce2);
    for j in K..7 {
        s[j] = w.wrapping_add(tz(cs));
        n[j] = tz(ce).wrapping_sub(tz(cs));
        cs = blsr(cs);
        ce = blsr(ce);
    }
    fastev::Cut { ok, s, n, len: le + 1 }
}

/// Split variants (SS_K4SPLIT): 0 = fastev's `cut` and `cut_wide` (the
/// default of round one), 1 = k4's `split` and fastev's `cut_wide`,
/// 2 = k4's `split` and `split_wide` (masks shared), 3 = `split_lean`, the
/// lines it does not take through `slow_line` (cut, cut_wide, scanner);
/// the default.
pub static SPLIT: AtomicUsize = AtomicUsize::new(3);
/// The lean variant tries the second shape (`split_lean::<3>`) in
/// `slow_line` before the old splitters (SS_K4SHAPEB=0: off).
pub static SPLIT_B: AtomicBool = AtomicBool::new(true);
/// Pack both IDs in one ymm (`pack2`); SS_K4PACK2=0 turns it off (default on:
/// fewer instructions and faster in the lean loop).
pub static PACK2: AtomicBool = AtomicBool::new(true);

const VIEW: u32 = u32::from_le_bytes(*b"view");
const CART: u32 = u32::from_le_bytes(*b"cart");
const PURCHASE: u64 = u64::from_le_bytes(*b"purchase");

/// `keys::key_of_ev_wide` (16 bytes from q readable).
#[inline(always)]
unsafe fn key_ev(f: &[u8]) -> u64 {
    match crate::keys::pack_wide(f) {
        Some(k) => k,
        None => crate::keys::other_ev(f),
    }
}

/// `keys::key_of_wide` (16 bytes from q readable).
#[inline(always)]
unsafe fn key_user(f: &[u8]) -> u64 {
    match crate::keys::pack_wide(f) {
        Some(k) => k,
        None => crate::keys::other_main(f),
    }
}

/// A purchase whose other inputs make the copy valid: price and category
/// through `main::event_pay` itself.
#[cold]
#[inline(never)]
fn purchase(ty: &[u8], price: &[u8], product: &[u8], ctx: &Ctx) -> Option<u32> {
    super::event_pay(ty, || price, product, ctx, true)
}

#[inline(always)]
unsafe fn field<'a>(p: *const u8, s: usize, n: usize) -> &'a [u8] {
    std::slice::from_raw_parts(p.add(s), n)
}

// ---------------------------------------------------------------- one line

/// Line outcomes of one task (statistics).
#[derive(Clone, Copy, Default, Debug)]
pub struct Stats {
    /// copies (or certain skips) decided by the kernel
    pub fast: usize,
    /// cut lines handed to `event_line`
    pub line_fb: usize,
    /// lines handed to the general scanner
    pub range_fb: usize,
    /// lines at the buffer end through `fastev::event_task_fast` (ranges)
    pub tail: usize,
    /// lines neither lean shape took (the old splitters in `slow_line`; also
    /// counted in fast / line_fb / range_fb)
    pub slow: usize,
}

/// The cut line at p (fields `c.s[j]`, `c.n[j]`; 32 bytes readable from every
/// field start): the copy if every input of it is certain (true), else
/// nothing (false: the caller hands the fields to `event_line`). No side
/// effect when it returns false.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[inline]
unsafe fn decode<S: EvSink, const P2: bool>(k: &K4, ctx: &Ctx, p: *const u8, c: &fastev::Cut, sink: &mut S) -> bool {
    decode_body::<S, P2>(k, ctx, p, c, sink)
}

/// `decode` without the target-feature attribute, so that it can be forced
/// inline into the lean loop (which has the features; the SIMD helpers it
/// calls inline there).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn decode_body<S: EvSink, const P2: bool>(k: &K4, ctx: &Ctx, p: *const u8, c: &fastev::Cut, sink: &mut S) -> bool {
    let p2 = P2;
    let n0 = c.n[0];
    if n0 == 0 {
        // the old path skips a line with an empty event_id (no side effect)
        return true;
    }
    // both IDs packed at once (pure; used only as key_of_ev_wide / key_of_wide would)
    let (ka, oka, kb, okb) = if p2 { pack2(p.add(c.s[0]), n0, p.add(c.s[1]), c.n[1]) } else { (0, false, 0, false) };
    let ws = k.ws;
    let q2 = p.add(c.s[2]);
    let q3 = p.add(c.s[3]);
    let (n2, n3) = (c.n[2], c.n[3]);
    let (dbad2, d2e) = date(q2, &k.tab);
    let (dbad3, d3e) = date(q3, &k.tab);
    let (sec2, omin, sec3, tok, trng) = tvec(q2, q3);
    // both checks in one word: bad separator / digit lanes (bits 0..32) and
    // out-of-range words (bits 32..64); only zero / nonzero matters below
    let bad = (!tok) as u64 | ((trng as u64) << 32);
    const IMASK: u64 = 0x1FF_0000 | (0x3F_0000 << 32);
    const F1MASK: u64 = 0x1FF | (0x3F << 32);
    const F2MASK: u64 = 0x3EFF | (0xF3F << 32);
    // ingest_ts: F1, date in the table, time well-formed and in range
    let ibad = (bad & IMASK) | dbad3 | (n3 ^ 20) as u64;
    // ts: F1 / F2 (date in the table) or 13 digits, as an offset from ws
    let civ = d2e.off_ms.wrapping_add(sec2 as i64 * 1000);
    let b19 = *q2.add(19) as u32;
    // F2: '+' subtracts the offset, '-' adds it (0x2C - sign = 1 / -1)
    let f2v = civ.wrapping_sub((omin as i64 * 60_000).wrapping_mul(0x2C - b19 as i64));
    let (f3v, f3bad) = f3_13(q2);
    let is13 = n2 == 13;
    let is25 = n2 == 25;
    // F1 (20 bytes) or F2 (25 bytes, sign at 19): lanes and words of the form,
    // date in the table, the length / sign
    let fmask = sel(is25, F2MASK, F1MASK);
    let fform = sel(is25, (b19.wrapping_sub(0x2B) & 0xFFFF_FFFD) as u64, (n2 ^ 20) as u64);
    let f12bad = (bad & fmask) | dbad2 | fform;
    let t_off = sel(is13, (f3v as i64).wrapping_sub(ws) as u64, sel(is25, f2v as u64, civ as u64)) as i64;
    let tbad = sel(is13, f3bad as u64, f12bad);
    // type exactly as event_pay (wide): 8 bytes from the field start
    let n4 = c.n[4];
    let w4 = rd64(p.add(c.s[4]));
    let is_vc = (n4 == 4) & (((w4 as u32) == VIEW) | ((w4 as u32) == CART));
    let is_pu = (n4 == 8) & (w4 == PURCHASE);
    // certainly invalid whatever ts holds
    let inv = (c.n[1] == 0) | (c.n[5] == 0) | !(is_vc | is_pu);
    if (ibad != 0) | (!inv & (tbad != 0)) {
        return false;
    }
    let ingest = d3e.rel_s.wrapping_add(sec3);
    let f0 = field(p, c.s[0], n0);
    let eid = crate::keys::mix(if !p2 {
        key_ev(f0)
    } else if oka {
        ka
    } else {
        crate::keys::other_ev(f0)
    });
    let ing_off = d3e.off_ms.wrapping_add(sec3 as i64 * 1000);
    let ok = !inv & ((t_off as u64) < k.span) & (ing_off.wrapping_sub(t_off) <= LATE);
    if !ok {
        sink.copy(eid, ingest, None);
        return true;
    }
    let mut pay = 0u32;
    if is_pu {
        match purchase(field(p, c.s[4], n4), field(p, c.s[6], c.n[6]), field(p, c.s[5], c.n[5]), ctx) {
            Some(v) => pay = v,
            None => {
                sink.copy(eid, ingest, None);
                return true;
            }
        }
    }
    let f1 = field(p, c.s[1], c.n[1]);
    let user = if !p2 {
        key_user(f1)
    } else if okb {
        kb
    } else {
        crate::keys::other_main(f1)
    };
    sink.copy(eid, ingest, Some((user, t_off as u32, pay)));
    true
}

/// A cut line the kernel is not certain about: `event_line` with its fields
/// (exactly `fastev::event_task_fast`'s per-line call).
#[cfg(target_arch = "x86_64")]
#[cold]
#[inline(never)]
fn line_fb<S: EvSink>(p: *const u8, c: &fastev::Cut, ctx: &Ctx, sink: &mut S) {
    let f: [&[u8]; 7] = unsafe { std::array::from_fn(|j| field(p, c.s[j], c.n[j])) };
    if !f[0].is_empty() {
        event_line::<true, S, _>(&f, f[0], ctx, sink);
    }
}

/// `line_fb` for a line `split_lean` took (the same cut, recomputed).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[cold]
#[inline(never)]
unsafe fn line_fb_lean<S: EvSink>(p: *const u8, ctx: &Ctx, sink: &mut S) {
    let (c0, n0, q0) = masks64(p);
    let (c1, n1, q1) = masks64(p.add(64));
    let c = split_lean::<2>(c0, n0, q0, c1, n1, q1);
    debug_assert!(c.ok);
    line_fb(p, &c, ctx, sink);
}

// ---------------------------------------------------------------- task

/// One event task through k4 (same copies, same order as
/// `fastev::event_task_fast` + `event_line`).
///
/// # Safety
/// The CPU must support AVX2, BMI1, BMI2, LZCNT and POPCNT; `safe_end` as
/// for `fastev::event_task_fast`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[inline]
pub unsafe fn event_task_k4<S: EvSink>(b: &[u8], task: Task, ctx: &Ctx, sink: &mut S, safe_end: usize) {
    let st = task_dispatch(b, task, ctx, sink, safe_end);
    if COUNT.load(Ordering::Relaxed) {
        N_FAST.fetch_add(st.fast, Ordering::Relaxed);
        N_LINE.fetch_add(st.line_fb, Ordering::Relaxed);
        N_RANGE.fetch_add(st.range_fb, Ordering::Relaxed);
        N_TAIL.fetch_add(st.tail, Ordering::Relaxed);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[inline]
unsafe fn task_dispatch<S: EvSink>(b: &[u8], task: Task, ctx: &Ctx, sink: &mut S, safe_end: usize) -> Stats {
    task_variant(b, task, ctx, sink, safe_end, SPLIT.load(Ordering::Relaxed), PACK2.load(Ordering::Relaxed))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[inline]
unsafe fn task_variant<S: EvSink>(b: &[u8], task: Task, ctx: &Ctx, sink: &mut S, safe_end: usize, spl: usize, p2: bool) -> Stats {
    match (spl, p2) {
        (1, true) => task_k4::<S, 1, true>(b, task, ctx, sink, safe_end),
        (2, false) => task_k4::<S, 2, false>(b, task, ctx, sink, safe_end),
        (2, true) => task_k4::<S, 2, true>(b, task, ctx, sink, safe_end),
        (3, false) => task_k4::<S, 3, false>(b, task, ctx, sink, safe_end),
        (3, true) => task_k4::<S, 3, true>(b, task, ctx, sink, safe_end),
        _ => task_k4::<S, 0, false>(b, task, ctx, sink, safe_end),
    }
}

/// One task with split variant `SPL` (see `SPLIT`) and `pack2` iff `P2`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[inline]
unsafe fn task_k4<S: EvSink, const SPL: usize, const P2: bool>(b: &[u8], task: Task, ctx: &Ctx, sink: &mut S, safe_end: usize) -> Stats {
    let mut st = Stats::default();
    let (start, end) = (task.start, task.end);
    assert!(start <= end && end <= b.len() && safe_end >= b.as_ptr() as usize + b.len());
    if start >= end {
        return st;
    }
    if b.len() < MARGIN {
        st.tail += 1;
        tail_fb(b, task, ctx, sink, safe_end);
        return st;
    }
    let k = params(ctx);
    let lim = b.len() - MARGIN;
    let base = b.as_ptr();
    let pf = fastev::PF_DIST.load(Ordering::Relaxed);
    let pf2 = fastev::PF2.load(Ordering::Relaxed);
    let pf_lim = (base as usize + b.len()).saturating_sub(64);
    let mut off = start;
    while off < end && off <= lim {
        let p = base.add(off);
        if SPL == 3 {
            // a prefetch never faults: no clamp to the buffer (wrapping address, never dereferenced)
            if pf != 0 {
                super::prefetch(p.wrapping_add(pf));
                if pf2 {
                    super::prefetch(p.wrapping_add(pf + 64));
                }
            }
        } else if pf != 0 {
            super::prefetch((p as usize + pf).min(pf_lim) as *const u8);
            if pf2 {
                super::prefetch((p as usize + pf + 64).min(pf_lim) as *const u8);
            }
        }
        if SPL == 3 {
            let (c0, n0, q0) = masks64(p);
            let (c1, n1, q1) = masks64(p.add(64));
            let c = split_lean::<2>(c0, n0, q0, c1, n1, q1);
            if c.ok {
                if decode_body::<S, P2>(k, ctx, p, &c, sink) {
                    st.fast += 1;
                } else {
                    // the cut is recomputed there (not handed over: it stays in registers here)
                    st.line_fb += 1;
                    line_fb_lean(p, ctx, sink);
                }
                off += c.len;
            } else {
                off = slow_line::<S, P2>(b, off, task.file, k, ctx, sink, safe_end, &mut st);
            }
            continue;
        }
        let mut c;
        if SPL == 0 {
            c = fastev::cut::<true>(p);
            if !c.ok {
                c = fastev::cut_wide::<true>(p);
            }
        } else {
            let (c0, n0, q0) = masks64(p);
            let (c1, n1, q1) = masks64(p.add(64));
            c = split(c0, n0, q0, c1, n1, q1);
            if !c.ok {
                c = if SPL == 2 { split_wide(p, c0, n0, q0, c1, n1, q1) } else { fastev::cut_wide::<true>(p) };
            }
        }
        if !c.ok {
            let e = range_end(b, off);
            st.range_fb += 1;
            super::event_task_scan_ni::<true, S>(b, Task { file: task.file, start: off, end: e }, ctx, sink, safe_end);
            off = e;
            continue;
        }
        if decode::<S, P2>(k, ctx, p, &c, sink) {
            st.fast += 1;
        } else {
            st.line_fb += 1;
            line_fb(p, &c, ctx, sink);
        }
        off += c.len;
    }
    if off < end {
        st.tail += 1;
        tail_fb(b, Task { file: task.file, start: off, end }, ctx, sink, safe_end);
    }
    st
}

/// A line `split_lean` did not take (lean variant): `fastev::cut`, else
/// `cut_wide`, then `decode` / `event_line`, else the scanner on the line,
/// exactly as variant 0 does for every line. Returns the next line start.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn slow_line<S: EvSink, const P2: bool>(b: &[u8], off: usize, file: u32, k: &K4, ctx: &Ctx, sink: &mut S, safe_end: usize, st: &mut Stats) -> usize {
    let p = b.as_ptr().add(off);
    // the second lean shape first (fields 0..=2 in the first 64 bytes); out of
    // the hot loop so that the loop's code stays that of shape A alone
    if SPLIT_B.load(Ordering::Relaxed) {
        let (c0, n0, q0) = masks64(p);
        let (c1, n1, q1) = masks64(p.add(64));
        let c = split_lean::<3>(c0, n0, q0, c1, n1, q1);
        if c.ok {
            if decode::<S, P2>(k, ctx, p, &c, sink) {
                st.fast += 1;
            } else {
                st.line_fb += 1;
                line_fb(p, &c, ctx, sink);
            }
            return off + c.len;
        }
    }
    st.slow += 1;
    let mut c = fastev::cut::<true>(p);
    if !c.ok {
        c = fastev::cut_wide::<true>(p);
    }
    if !c.ok {
        let e = range_end(b, off);
        st.range_fb += 1;
        super::event_task_scan_ni::<true, S>(b, Task { file, start: off, end: e }, ctx, sink, safe_end);
        return e;
    }
    if decode::<S, P2>(k, ctx, p, &c, sink) {
        st.fast += 1;
    } else {
        st.line_fb += 1;
        line_fb(p, &c, ctx, sink);
    }
    off + c.len
}

/// The old path on a range (buffer ends, small buffers), out of line.
#[cfg(target_arch = "x86_64")]
#[cold]
#[inline(never)]
fn tail_fb<S: EvSink>(b: &[u8], task: Task, ctx: &Ctx, sink: &mut S, safe_end: usize) {
    fastev::task_fast_any(b, task, ctx, sink, safe_end)
}

/// End of the line at `off` (after its newline, or the end of `b`).
#[cold]
#[inline(never)]
fn range_end(b: &[u8], off: usize) -> usize {
    scan::next_line_start(b, off)
}

/// `event_task_k4` with a run-time check (checks and tests): None without AVX2.
pub fn task_any<S: EvSink>(b: &[u8], task: Task, ctx: &Ctx, sink: &mut S, safe_end: usize) -> Option<Stats> {
    task_any_v(b, task, ctx, sink, safe_end, SPLIT.load(Ordering::Relaxed), PACK2.load(Ordering::Relaxed))
}

/// `task_any` for one variant (split `spl`, `pack2` iff `p2`; tests).
pub fn task_any_v<S: EvSink>(b: &[u8], task: Task, ctx: &Ctx, sink: &mut S, safe_end: usize, spl: usize, p2: bool) -> Option<Stats> {
    #[cfg(target_arch = "x86_64")]
    {
        if ctx.avx2 {
            return Some(unsafe { task_k4_avx2(b, task, ctx, sink, safe_end, spl, p2) });
        }
    }
    let _ = (b, task, ctx, sink, safe_end, spl, p2);
    None
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[inline(never)]
unsafe fn task_k4_avx2<S: EvSink>(b: &[u8], task: Task, ctx: &Ctx, sink: &mut S, safe_end: usize, spl: usize, p2: bool) -> Stats {
    task_variant(b, task, ctx, sink, safe_end, spl, p2)
}

// ---------------------------------------------------------------- checking

fn show(b: &[u8]) -> String {
    let s = b.escape_ascii().to_string();
    if s.len() > 400 {
        format!("{}...[{} bytes]", &s[..400], b.len())
    } else {
        s
    }
}

/// SS_K4CHECK=1: every event task through the old path
/// (`fastev::event_task_fast` + `event_line`) and through k4 into recording
/// sinks; the copy() calls must be identical, argument by argument, in the
/// same order. Prints the line counts per outcome; exits nonzero on any
/// difference (after showing the first lines that differ on their own).
pub fn check(ev: &Input, ctx: &Ctx) {
    if !ctx.avx2 {
        eprintln!("k4check: needs AVX2 + BMI1/2 (k4 is never used without them)");
        std::process::exit(1);
    }
    let (mut lines, mut copies, mut valid) = (0usize, 0usize, 0usize);
    let mut tot = Stats::default();
    let mut bad = 0usize;
    for task in &ev.tasks {
        let b = ev.maps[task.file as usize].bytes();
        let safe_end = b.as_ptr() as usize + b.len();
        let mut want = fastev::RecSink::default();
        fastev::task_fast_any(b, *task, ctx, &mut want, safe_end);
        let mut got = fastev::RecSink::default();
        let st = task_any(b, *task, ctx, &mut got, safe_end).unwrap();
        tot.fast += st.fast;
        tot.line_fb += st.line_fb;
        tot.range_fb += st.range_fb;
        tot.tail += st.tail;
        tot.slow += st.slow;
        lines += b[task.start..task.end].iter().filter(|&&c| c == b'\n').count();
        copies += want.0.len();
        valid += want.0.iter().filter(|c| c.2.is_some()).count();
        if got.0 != want.0 {
            bad += 1;
            eprintln!("k4check: MISMATCH file {} [{}, {}) ({} k4 vs {} old copies)", task.file, task.start, task.end, got.0.len(), want.0.len());
            let i = got.0.iter().zip(&want.0).position(|(a, b)| a != b).unwrap_or(got.0.len().min(want.0.len()));
            eprintln!("  first differing copy #{}: k4 {:?} old {:?}", i, got.0.get(i), want.0.get(i));
            // the lines that differ on their own
            let mut l = task.start;
            let mut shown = 0;
            while l < task.end && shown < 5 {
                let e = scan::next_line_start(b, l).min(task.end);
                let one = Task { file: task.file, start: l, end: e };
                let mut a = fastev::RecSink::default();
                fastev::task_fast_any(b, one, ctx, &mut a, safe_end);
                let mut g = fastev::RecSink::default();
                task_any(b, one, ctx, &mut g, safe_end);
                if a.0 != g.0 {
                    eprintln!("  line at {}: \"{}\"\n    old: {:?}\n    k4:  {:?}", l, show(&b[l..e]), a.0, g.0);
                    shown += 1;
                }
                l = e;
            }
            if shown == 0 {
                eprintln!("  (no single line differs on its own)");
            }
            if bad >= 3 {
                break;
            }
        }
    }
    let pct = |x: usize| x as f64 * 100.0 / lines.max(1) as f64;
    eprintln!(
        "k4check: {} {} tasks, {} lines, {} copies ({} valid); k4 fast {} ({:.3}%), event_line fallback {} ({:.3}%), scanner lines {} ({:.4}%), buffer-end ranges {}; taken by neither lean shape {} ({:.3}%)",
        if bad == 0 { "OK" } else { "FAILED" },
        ev.tasks.len(),
        lines,
        copies,
        valid,
        tot.fast,
        pct(tot.fast),
        tot.line_fb,
        pct(tot.line_fb),
        tot.range_fb,
        pct(tot.range_fb),
        tot.tail,
        tot.slow,
        pct(tot.slow)
    );
    if bad != 0 {
        std::process::exit(1);
    }
}
