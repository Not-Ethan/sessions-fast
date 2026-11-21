//! k3: lean per-line kernels for the dense pipeline's event and click lines
//! (notes/k3-kernel.md).
//!
//! A line takes the fast path only when it is recognised with certainty as
//! one of the shapes the generator writes; it is then decoded branch-free
//! into the dense record. Every other line goes, out of line, to the exact
//! routines of densep.rs (the oracle): a line whose fields were cut but
//! whose values are in any doubt to `ev_line` with those fields, a line the
//! splitter could not cut to the general scanner as a one-line range.
//!
//! Event fast path (all of it must hold, else the line falls back):
//! * split: the 96 bytes at the line start hold the whole line; the first
//!   three delimiters end fields 0..2 inside the first 64 bytes, field 3
//!   starts at offset 32..64 and the 64 bytes from there hold fields 3..6,
//!   whose fourth delimiter is the newline; the first six delimiters are
//!   commas; every field holds no quote or two, as its first and last byte
//!   (fastev.rs's `window` rule). Then the fields are exactly the scanner's.
//! * event_id: `e` + 2..8 lowercase base-36 digits, no leading zero, value
//!   < 2^40, unscrambled number < bound (else ev_line: XE path).
//! * ingest_ts: F1 whose date is in the kernel's 28-day date table.
//! * ts: F1 / F2 (date in the table) or exactly 13 digits (F3).
//! * a valid copy's user_id: canonical, unscrambled into a dense slot.
//! Everything else that decides the record (empty user / product, window,
//! event type, late) is computed exactly; purchases (price, product) go
//! through main::event_pay itself.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::{Ctx, DAY, LATE, WEEK};
use crate::dense::{M1, M2, MASK};
use crate::scan;

/// SS_K3: the dense pipeline uses these kernels (densep.rs hooks).
pub static ON: AtomicBool = AtomicBool::new(false);

const M1I: u64 = crate::dense::M1I;
const M2I: u64 = crate::dense::M2I;
const _: () = assert!(M1.wrapping_mul(M1I) & MASK == 1 && M2.wrapping_mul(M2I) & MASK == 1);

pub const VALID: u32 = 1 << 31;
pub const PFLAG: u32 = 1 << 31;

/// Readable bytes after a fast-path line start (garbage positions of a line
/// that fails the split included): masks 96, field reads <= 65 + 64 + 27.
const MARGIN: usize = 192;
/// Lines starting before this offset of the buffer take the general path
/// (the IDs are read with 16-byte loads that end at the field end).
const FRONT: usize = 16;

/// Receives the records of fast-path event lines.
pub trait EvRec {
    /// One event copy with a dense number n: for a valid copy ts = offset |
    /// VALID, us = user slot | PFLAG if pay != 0; for an invalid copy ts =
    /// us = pay = 0.
    fn ev(&mut self, n: u32, ingest: u32, ts: u32, us: u32, pay: u32);
}

// ---------------------------------------------------------------- tables

/// One date of the kernel's date table.
#[derive(Clone, Copy, Default)]
#[repr(C, align(32))]
struct DEnt {
    /// bytes 0..8 `YYYY-MM-` (little-endian)
    pat: u64,
    /// bytes 8..11 `DDT` (24 bits; 0xFFFF_FFFF for an unused slot)
    dd: u32,
    /// ingest_rel of the day's midnight (seconds)
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

/// Slot of the day digits `DD` (bytes 8, 9 as a little-endian u16):
/// distinct for "00".."39".
#[inline(always)]
fn dslot(dd: u32) -> usize {
    ((dd & 3) | ((dd >> 6) & 0x3C)) as usize
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
    /// The 28 dates from `first_day` (epoch days); slots of dates that
    /// cannot be represented stay unused (they never match).
    fn new(first_day: i64, ws: i64, ws_s: i64) -> DTab {
        let mut t = DTab([DEnt { pat: 0, dd: 0xFFFF_FFFF, rel_s: 0, off_ms: 0, _p: 0 }; 64]);
        for day in first_day..first_day + 28 {
            let (y, m, d) = civil_from_days(day);
            if !(0..=9999).contains(&y) {
                continue;
            }
            let txt = format!("{:04}-{:02}-{:02}T00:00:00Z", y, m, d);
            let tb = txt.as_bytes();
            let ms = day * DAY;
            // every entry is checked against the general parser
            if crate::ts::parse_ts(tb) != Some(ms) {
                continue;
            }
            // main::ingest_rel's near range for every second of the day
            let dsec = ms.div_euclid(1000) - ws_s;
            let rel = dsec + (1i64 << 31);
            let off = ms.checked_sub(ws);
            if !(-(1i64 << 30)..(1i64 << 30) - 86_400).contains(&dsec) || off.is_none() {
                continue;
            }
            let pat = u64::from_le_bytes(tb[..8].try_into().unwrap());
            let dd = u32::from_le_bytes([tb[8], tb[9], tb[10], 0]);
            let k = dslot(u16::from_le_bytes([tb[8], tb[9]]) as u32);
            assert!(t.0[k].dd == 0xFFFF_FFFF, "k3: date slot collision");
            t.0[k] = DEnt { pat, dd, rel_s: rel as u32, off_ms: off.unwrap(), _p: 0 };
        }
        t
    }
}

/// Kernel parameters of one run (keys, bounds, window, date tables).
pub struct K3 {
    pub ke: u64,
    pub ku: u64,
    pub bound: u64,
    pub pmax: u64,
    ws: i64,
    /// we - ws (events' window length, ms)
    span: u64,
    /// ws - WEEK and we - (ws - WEEK) (clicks)
    cws: i64,
    cspan: u64,
    ev_tab: DTab,
    cl_tab: DTab,
}

impl K3 {
    pub fn new(ctx: &Ctx, ke: u64, ku: u64, bound: u64, pmax: u64) -> K3 {
        let (ws, we) = (ctx.ws, ctx.we);
        let wd = ws.div_euclid(DAY);
        let cws = ws - WEEK;
        K3 {
            ke,
            ku,
            bound,
            pmax,
            ws,
            span: (we - ws).max(0) as u64,
            cws,
            cspan: (we - cws).max(0) as u64,
            // events: ts and ingest dates from 2 days before the window
            ev_tab: DTab::new(wd - 2, ws, ctx.ws_s),
            // clicks: from 9 days before the window start
            cl_tab: DTab::new(wd - 9, ws, ctx.ws_s),
        }
    }
}

static PARAMS: Mutex<Vec<&'static K3>> = Mutex::new(Vec::new());

/// The (cached) parameters for these keys and ctx's window.
pub fn params(ctx: &Ctx, ke: u64, ku: u64, bound: u64, pmax: u64) -> &'static K3 {
    let mut v = PARAMS.lock().unwrap();
    for &k in v.iter() {
        if k.ke == ke && k.ku == ku && k.bound == bound && k.pmax == pmax && k.ws == ctx.ws && k.span == (ctx.we - ctx.ws).max(0) as u64 {
            return k;
        }
    }
    let k: &'static K3 = Box::leak(Box::new(K3::new(ctx, ke, ku, bound, pmax)));
    v.push(k);
    k
}

/// Lane masks for a right-aligned ID of m digits (m <= 8): bytes 16-m..16.
#[repr(C, align(16))]
struct Lm([[u8; 16]; 16]);
static LM: Lm = {
    let mut t = [[0u8; 16]; 16];
    let mut m = 0;
    while m <= 8 {
        let mut i = 16 - m;
        while i < 16 {
            t[m][i] = 0xFF;
            i += 1;
        }
        m += 1;
    }
    Lm(t)
};
static NEED: [u32; 16] = {
    let mut t = [0u32; 16];
    let mut m = 0;
    while m <= 8 {
        t[m] = (0xFFFFu32 << (16 - m)) & 0xFFFF;
        m += 1;
    }
    t
};

/// Per digit count m (index m & 63): the digit lanes of a right-aligned ID
/// (`need`, bit 40 set for m outside 2..=8: never satisfied) and 36^(m-1)
/// (an m-digit value below it has a leading zero).
#[derive(Clone, Copy)]
#[repr(C, align(16))]
struct IdM {
    need: u64,
    pow: u64,
}
static IDM: [IdM; 64] = {
    let mut t = [IdM { need: 1 << 40, pow: 0 }; 64];
    let mut m = 2;
    let mut pw = 36u64;
    while m <= 8 {
        t[m] = IdM { need: ((0xFFFFu64 << (16 - m)) & 0xFFFF), pow: pw };
        pw *= 36;
        m += 1;
    }
    t
};

// ---------------------------------------------------------------- helpers

#[inline(always)]
unsafe fn rd64(p: *const u8) -> u64 {
    u64::from_le(std::ptr::read_unaligned(p as *const u64))
}
#[inline(always)]
unsafe fn rd32(p: *const u8) -> u32 {
    u32::from_le(std::ptr::read_unaligned(p as *const u32))
}
#[inline(always)]
unsafe fn rd16(p: *const u8) -> u32 {
    u16::from_le(std::ptr::read_unaligned(p as *const u16)) as u32
}
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

/// fastev.rs's `window`: (violations, content starts, content ends) of a
/// 64-bit window whose bit 0 is a field start and whose region `r` holds
/// whole fields.
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

/// Nonzero iff some byte lane selected by `lanes` is not an ASCII digit.
#[inline(always)]
fn digits_bad(w: u64, lanes: u64) -> u64 {
    let hi = (w & 0xF0F0_F0F0_F0F0_F0F0) ^ 0x3030_3030_3030_3030;
    let lo = ((w & 0x0F0F_0F0F_0F0F_0F0F) + 0x0606_0606_0606_0606) & 0xF0F0_F0F0_F0F0_F0F0;
    (hi | lo) & lanes
}

/// 8 ASCII digits (first char in the lowest byte) to their value.
#[inline(always)]
fn parse8(w: u64) -> u64 {
    let l = w.wrapping_sub(0x3030_3030_3030_3030);
    let c = (l.wrapping_mul(10).wrapping_add(l >> 8)) & 0x00FF_00FF_00FF_00FF;
    let c = (c.wrapping_mul(100).wrapping_add(c >> 16)) & 0x0000_FFFF_0000_FFFF;
    (c.wrapping_mul(10000).wrapping_add(c >> 32)) & 0xFFFF_FFFF
}

/// The number scrambled into v (40 bits) under key.
#[inline(always)]
fn unscramble(v: u64, key: u64) -> u64 {
    let t = v.wrapping_mul(M2I) & MASK;
    ((t ^ (t >> 17) ^ (t >> 34) ^ key).wrapping_mul(M1I)) & MASK
}

/// a if c else b, without a branch.
#[inline(always)]
fn sel(c: bool, a: u64, b: u64) -> u64 {
    let m = (c as u64).wrapping_neg();
    (a & m) | (b & !m)
}

/// Nonzero unless 2 <= m <= 8.
#[inline(always)]
fn range28(m: usize) -> u64 {
    let x = m.wrapping_sub(2) as u64;
    (x | x.wrapping_add(1)) >> 3
}

/// 36^k (k <= 7): an m-digit value below 36^(m-1) has a leading zero.
static POW36: [u64; 16] = {
    let mut t = [u64::MAX; 16];
    let mut k = 0;
    let mut v = 1u64;
    while k <= 8 {
        t[k] = v;
        v *= 36;
        k += 1;
    }
    t
};

/// The date `YYYY-MM-DDT` at q looked up in `tab`: (nonzero unless the 11
/// bytes are exactly a table date, its entry). Reads q..q+12.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
#[inline]
unsafe fn date(q: *const u8, tab: &DTab) -> (u64, &DEnt) {
    let a = rd64(q);
    let d4 = rd32(q.add(8));
    let e = tab.0.get_unchecked(_pext_u32(d4, 0x0F03) as usize);
    ((a ^ e.pat) | ((d4 & 0xFF_FFFF) ^ e.dd) as u64, e)
}

/// `HH:MM:SS` at a + 11 and at b + 11 at once: (seconds of day of a, of b,
/// nonzero unless both are well-formed with hh <= 23, mm, ss <= 59).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn times2(a: *const u8, b: *const u8) -> (u32, u32, u32) {
    let v = _mm_set_epi64x(rd64(b.add(11)) as i64, rd64(a.add(11)) as i64);
    let x = _mm_xor_si128(v, _mm_set1_epi64x(0x3030_3A30_303A_3030));
    // digit lanes <= 9, ':' lanes == 0
    let lim = _mm_set1_epi64x(0x0909_0009_0900_0909);
    let okm = _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_min_epu8(x, lim), x)) as u32;
    let dg = _mm_shuffle_epi8(x, _mm_setr_epi8(0, 1, 3, 4, 6, 7, -1, -1, 8, 9, 11, 12, 14, 15, -1, -1));
    let w = _mm_maddubs_epi16(dg, _mm_set1_epi16(0x010A)); // [hh, mm, ss, 0] x 2
    let rm = _mm_movemask_epi8(_mm_cmpgt_epi16(w, _mm_setr_epi16(23, 59, 59, 0, 23, 59, 59, 0))) as u32;
    let s = _mm_madd_epi16(w, _mm_setr_epi16(3600, 60, 1, 0, 3600, 60, 1, 0));
    let t = _mm_hadd_epi32(s, s);
    let r = _mm_cvtsi128_si64(t) as u64;
    (r as u32, (r >> 32) as u32, (okm ^ 0xFFFF) | rm)
}

/// The 16 bytes from a + 11 (ts) and from b + 11 (ingest_ts) at once:
/// `HH:MM:SS`, byte 19 (`Z` or the F2 sign) and the F2 offset `HH:MM`.
/// Returns (seconds of day of a, F2 offset minutes of a, seconds of day of
/// b, lane mask, range mask): lane mask bit i (a: 0..16, b: 16..32) is set
/// iff byte 11 + i is a digit where `HH:MM:SS?HH:MM` has a digit, or equals
/// the `:` / `Z` / `:` at 2, 5, 8, 11; range mask bits 2w, 2w+1 are set iff
/// word w of [hh, mm, ss, 0, oh, om, 0, 0] exceeds [23, 59, 59, 0, 23, 59, 0, 0]
/// (b: + 16).
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
/// the 16 bytes ending at q + 13.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn f3_13(q: *const u8) -> (u64, u32) {
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn masks32(p: *const u8) -> (u64, u64, u64) {
    let v = _mm256_loadu_si256(p as *const __m256i);
    let c = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, _mm256_set1_epi8(b',' as i8))) as u32 as u64;
    let n = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, _mm256_set1_epi8(b'\n' as i8))) as u32 as u64;
    let q = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, _mm256_set1_epi8(b'"' as i8))) as u32 as u64;
    (c, n, q)
}

/// Both IDs at once: the 16 bytes ending at `ea` (event_id) and at `eb`
/// (user_id), digits right-aligned, m digits each (index into LM, <= 15):
/// (value a, value b, digit-lane mask: bit i set iff byte i of the two
/// 16-byte windows is a lowercase base-36 digit). Values are meaningful
/// only for m <= 8 and all m lanes digits.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn ids2(ea: *const u8, eb: *const u8, ma: usize, mb: usize) -> (u64, u64, u32) {
    let va = _mm_loadu_si128(ea.sub(16) as *const __m128i);
    let vb = _mm_loadu_si128(eb.sub(16) as *const __m128i);
    let v = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(va), vb);
    let c0 = _mm256_sub_epi8(v, _mm256_set1_epi8(b'0' as i8));
    let ca = _mm256_sub_epi8(v, _mm256_set1_epi8(b'a' as i8));
    let isd = _mm256_cmpeq_epi8(_mm256_min_epu8(c0, _mm256_set1_epi8(9)), c0);
    let isl = _mm256_cmpeq_epi8(_mm256_min_epu8(ca, _mm256_set1_epi8(25)), ca);
    // letters: c0 = 0x31..0x4A -> minus 0x27 = 10..35
    let val = _mm256_sub_epi8(c0, _mm256_and_si256(isl, _mm256_set1_epi8(0x27)));
    let okm = _mm256_movemask_epi8(_mm256_or_si256(isd, isl)) as u32;
    let lm = _mm256_inserti128_si256::<1>(
        _mm256_castsi128_si256(_mm_load_si128(LM.0.as_ptr().add(ma) as *const __m128i)),
        _mm_load_si128(LM.0.as_ptr().add(mb) as *const __m128i),
    );
    let dv = _mm256_and_si256(val, lm);
    let w = _mm256_maddubs_epi16(dv, _mm256_set1_epi16(0x0124)); // [36, 1] per byte pair
    let q = _mm256_madd_epi16(w, _mm256_set1_epi32(0x0001_0510)); // [1296, 1] per word pair
    let hi = _mm256_mul_epu32(q, _mm256_set_epi64x(1_679_616, 0, 1_679_616, 0));
    let lo = _mm256_srli_epi64::<32>(q);
    let s = _mm256_add_epi64(hi, lo);
    (_mm256_extract_epi64::<1>(s) as u64, _mm256_extract_epi64::<3>(s) as u64, okm)
}

// ---------------------------------------------------------------- events

/// Line counters of one kernel call (for statistics and the checker).
#[derive(Clone, Copy, Default, Debug)]
pub struct Stats {
    pub fast: usize,
    /// cut lines given to the exact per-line routine
    pub line_fb: usize,
    /// lines given to the general scanner
    pub range_fb: usize,
}

/// What the kernel computes per line (benchmark stages): 0 = everything,
/// 1 = split only, 2 = split + decode without a record.
pub const FULL: u32 = 0;
pub const SPLIT: u32 = 1;
pub const DECODE: u32 = 2;

/// The fields of a cut line (all seven), from the content masks.
#[inline(always)]
unsafe fn fields7<'a>(p: *const u8, cs1: u64, ce1: u64, s3r: usize, cs2: u64, ce2: u64) -> [&'a [u8]; 7] {
    let mut f: [&[u8]; 7] = [&[]; 7];
    let (mut cs, mut ce) = (cs1, ce1);
    for x in f.iter_mut().take(3) {
        *x = std::slice::from_raw_parts(p.add(tz(cs)), tz(ce) - tz(cs));
        cs = blsr(cs);
        ce = blsr(ce);
    }
    let p2 = p.add(s3r);
    let (mut cs, mut ce) = (cs2, ce2);
    for x in f.iter_mut().skip(3) {
        *x = std::slice::from_raw_parts(p2.add(tz(cs)), tz(ce) - tz(cs));
        cs = blsr(cs);
        ce = blsr(ce);
    }
    f
}

/// Every line of b[start..end): fast lines to `sink`, cut lines in doubt to
/// `fl` (their seven unquoted fields, exactly the scanner's), every other
/// part of the range to `fr` as a line-aligned range for the general
/// scanner. In order. `start` must be a line start and `end` a line start
/// or b.len(). Reads only inside `b`. Requires AVX2 + BMI1/2 (caller
/// checks, e.g. ctx.avx2).
///
/// # Safety
/// The CPU must support AVX2, BMI1, BMI2, LZCNT and POPCNT.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[inline(never)]
pub unsafe fn ev_kernel<'a, S: EvRec, FL: FnMut(&[&'a [u8]; 7]), FR: FnMut(usize, usize), const MODE: u32>(
    b: &'a [u8],
    start: usize,
    end: usize,
    k: &K3,
    ctx: &Ctx,
    sink: &mut S,
    mut fl: FL,
    mut fr: FR,
) -> (Stats, u64) {
    let mut st = Stats::default();
    let mut acc = 0u64;
    assert!(start <= end && end <= b.len());
    if start >= end {
        return (st, acc);
    }
    if b.len() < MARGIN + FRONT {
        st.range_fb += 1;
        fr(start, end);
        return (st, acc);
    }
    let base = b.as_ptr();
    let lim = b.len() - MARGIN;
    let mut off = start;
    while off < end && off < FRONT {
        let e = scan::next_line_start(b, off);
        st.range_fb += 1;
        fr(off, e);
        off = e;
    }
    let tab = &k.ev_tab;
    let (ke, ku, ws) = (k.ke, k.ku, k.ws);
    // n >= bound  <=>  bit 63 of n + (2^63 - bound) (n < 2^40)
    let nb_add = (1u64 << 63).wrapping_sub(k.bound);
    // dense user: u < pmax * 512 and alias (u & 511) < 8
    let pu_add = (1u64 << 63).wrapping_sub(k.pmax << 9);
    let span1 = k.span as i64 - 1;
    while off < end && off <= lim {
        let p = base.add(off);
        // ---- split
        let (c0, n0, q0) = masks32(p);
        let (c1, n1, q1) = masks32(p.add(32));
        let (c2, n2, q2) = masks32(p.add(64));
        let d1 = c0 | n0 | ((c1 | n1) << 32);
        let q01 = q0 | (q1 << 32);
        let d1c = blsr(blsr(d1));
        let r1 = upto(d1c);
        let (b1, cs1, ce1) = window(d1, q01, r1);
        let s3r = tz(d1c) + 1; // raw start of field 3 (65: no third delimiter)
        let sh = s3r.wrapping_sub(32); // window 2 = the 64 bits at p + 32, shifted
        let shc = (sh & 63) as u32;
        let n12 = (n1 | (n2 << 32)) >> shc;
        let d2 = ((c1 | (c2 << 32)) >> shc) | n12;
        let q2w = (q1 | (q2 << 32)) >> shc;
        let d2d = blsr(blsr(blsr(d2)));
        let r2 = upto(d2d);
        let (b2, cs2, ce2) = window(d2, q2w, r2);
        let last = d2d & d2d.wrapping_neg();
        // newline among fields 0..2 (n0, n1 within r1) or a non-final delimiter / a final comma in window 2
        let nlbad = ((n0 | (n1 << 32)) & r1) | ((n12 ^ last) & r2);
        let sbad = (sh >> 5) as u64 | (d2d == 0) as u64 | b1 | b2 | nlbad;
        let le = s3r + tz(d2d); // the newline
        if MODE == SPLIT {
            if sbad == 0 {
                st.fast += 1;
                acc = acc.wrapping_add((tz(cs1) ^ tz(ce2) ^ tz(blsr(cs2))) as u64);
                off += le + 1;
            } else {
                let e = slow_end(b, off);
                st.range_fb += 1;
                fr(off, e);
                off = e;
            }
            continue;
        }
        // Positions are garbage unless sbad == 0; every read stays inside
        // MARGIN. Every check ORs a nonzero value into a "bad" word when it
        // fails.
        // ---- IDs: event number, user slot
        let s0 = tz(cs1);
        let e0 = tz(ce1);
        let cs1b = blsr(cs1);
        let ce1b = blsr(ce1);
        let s1 = tz(cs1b);
        let e1 = tz(ce1b);
        let m0 = e0.wrapping_sub(s0).wrapping_sub(1);
        let m1 = e1.wrapping_sub(s1).wrapping_sub(1);
        let (xa, xb, okm) = ids2(p.add(e0), p.add(e1), m0 & 15, m1 & 15);
        let ia = IDM.get_unchecked(m0 & 63);
        let ib = IDM.get_unchecked(m1 & 63);
        let ev_n = unscramble(xa, ke);
        // canonical event_id (prefix, 2..8 digits, no leading zero, < 2^40)
        // with a dense number
        let ebad = (ia.need & !(okm as u64))
            | (*p.add(s0) ^ b'e') as u64
            | (xa >> 40)
            | (xa.wrapping_sub(ia.pow) >> 63)
            | (ev_n.wrapping_add(nb_add) >> 63);
        let u = unscramble(xb, ku);
        let slot = ((u >> 9) * 8 + (u & 511)) as u32;
        let ubad = (ib.need & !((okm >> 16) as u64))
            | (*p.add(s1) ^ b'u') as u64
            | (xb >> 40)
            | (xb.wrapping_sub(ib.pow) >> 63)
            | (u.wrapping_add(pu_add) >> 63)
            | (u & 0x1F8);
        // ---- ts (F1 / F2 / 13-digit F3) and ingest_ts (F1)
        let s2 = tz(blsr(cs1b));
        let e2 = tz(blsr(ce1b));
        let p2 = p.add(s3r);
        let s3 = tz(cs2);
        let e3 = tz(ce2);
        let q2 = p.add(s2);
        let q3 = p2.add(s3);
        let n2 = e2.wrapping_sub(s2);
        let (dbad2, d2e) = date(q2, tab);
        let (dbad3, d3e) = date(q3, tab);
        let (sec2, omin, sec3, tok, trng) = tvec(q2, q3);
        let civ = d2e.off_ms.wrapping_add(sec2 as i64 * 1000);
        let b19 = *q2.add(19) as u32;
        // F2: '+' subtracts the offset, '-' adds it (0x2C - sign = 1 / -1)
        let f2v = civ.wrapping_sub((omin as i64 * 60_000).wrapping_mul(0x2C - b19 as i64));
        let (f3, f3bad) = f3_13(q2);
        let ntok = !tok as u64;
        let f1bad = (ntok & 0x1FF) | (trng & 0x3F) as u64 | dbad2 | (n2 ^ 20) as u64;
        let f2bad = (ntok & 0x3EFF) | (trng & 0xF3F) as u64 | dbad2 | (b19.wrapping_sub(0x2B) & 0xFFFF_FFFD) as u64;
        let is13 = n2 == 13;
        let is25 = n2 == 25;
        let t_off = sel(is13, (f3 as i64).wrapping_sub(ws) as u64, sel(is25, f2v as u64, civ as u64)) as i64;
        let tbad = sel(is13, f3bad as u64, sel(is25, f2bad, f1bad));
        let ing_off = d3e.off_ms.wrapping_add(sec3 as i64 * 1000);
        let ingest = d3e.rel_s.wrapping_add(sec3);
        let ibad = (ntok & 0x1FF_0000) | (trng & 0x3F_0000) as u64 | dbad3 | (e3.wrapping_sub(s3) ^ 20) as u64;
        // ---- type and validity (inv nonzero: an invalid copy)
        let cs2b = blsr(cs2);
        let ce2b = blsr(ce2);
        let s4 = tz(cs2b);
        let e4 = tz(ce2b);
        let s5 = tz(blsr(cs2b));
        let e5 = tz(blsr(ce2b));
        let w4 = rd64(p2.add(s4));
        let n4 = e4.wrapping_sub(s4) as u64;
        let w4l = w4 as u32;
        let vcbad = (((w4l ^ u32::from_le_bytes(*b"view")) as u64) * ((w4l ^ u32::from_le_bytes(*b"cart")) as u64)) | (n4 ^ 4);
        let pubad = (w4 ^ u64::from_le_bytes(*b"purchase")) | (n4 ^ 8);
        let inv = vcbad.min(pubad)
            | ((e1.wrapping_sub(s1) as i64).wrapping_sub(1) as u64 >> 63)
            | ((e5.wrapping_sub(s5) as i64).wrapping_sub(1) as u64 >> 63)
            | (t_off as u64 >> 63)
            | (span1.wrapping_sub(t_off) as u64 >> 63)
            | (LATE.wrapping_sub(ing_off.wrapping_sub(t_off)) as u64 >> 63);
        let v0 = inv == 0;
        // one decision: a valid copy also needs a dense user and is not a purchase
        let pre = sbad | ebad | ibad | tbad;
        let all = pre | sel(v0, ubad | (pubad == 0) as u64, 0);
        if MODE == DECODE {
            if all == 0 {
                st.fast += 1;
                acc = acc.wrapping_add(ev_n ^ (slot as u64) ^ (ingest as u64) ^ (t_off as u64) ^ (v0 as u64));
                off += le + 1;
                continue;
            }
        } else if all == 0 {
            st.fast += 1;
            let ts = sel(v0, (t_off as u32 | VALID) as u64, 0) as u32;
            let us = sel(v0, slot as u64, 0) as u32;
            sink.ev(ev_n as u32, ingest, ts, us, 0);
            off += le + 1;
            continue;
        }
        // ---- out of line: purchases, doubts, lines the splitter did not cut
        if sbad == 0 {
            let f = fields7(p, cs1, ce1, s3r, cs2, ce2);
            if pre == 0 && v0 && ubad == 0 && pubad == 0 && MODE == FULL {
                st.fast += 1;
                purchase(&f, ctx, sink, ev_n as u32, ingest, t_off as u32, slot);
            } else {
                st.line_fb += 1;
                fl(&f);
            }
            off += le + 1;
        } else {
            let e = slow_end(b, off);
            st.range_fb += 1;
            fr(off, e);
            off = e;
        }
    }
    if off < end {
        st.range_fb += 1;
        fr(off, end);
    }
    (st, acc)
}

// ---------------------------------------------------------------- clicks

/// Receives the records of fast-path click lines (valid clicks only).
pub trait ClRec {
    fn cl(&mut self, us: u32, camp: u32, ts: u32, ck: u64);
}

/// Left-aligned lane masks for a click_id of n bytes (n <= 10): bytes 0..n.
static LM0: Lm = {
    let mut t = [[0u8; 16]; 16];
    let mut n = 0;
    while n <= 10 {
        let mut i = 0;
        while i < n {
            t[n][i] = 0xFF;
            i += 1;
        }
        n += 1;
    }
    Lm(t)
};
/// Per click_id length n (index n & 63): the lanes that must hold [0-9a-z]
/// (bit 40 set for n outside 1..=10: never satisfied).
static NEED0: [u64; 64] = {
    let mut t = [1u64 << 40; 64];
    let mut n = 1;
    while n <= 10 {
        t[n] = (1u64 << n) - 1;
        n += 1;
    }
    t
};

/// click_id (16 bytes from its start `ca`, n0 bytes) and user_id (16 bytes
/// ending at `ue`, mb digits) at once: (keys::pack of the click_id when all
/// its bytes are [0-9a-z] (without the length), the user value, lane mask:
/// bit i set iff byte i of the two windows is [0-9a-z]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn ids_cl(ca: *const u8, ue: *const u8, n0: usize, mb: usize) -> (u64, u64, u32) {
    let va = _mm_loadu_si128(ca as *const __m128i);
    let vb = _mm_loadu_si128(ue.sub(16) as *const __m128i);
    let v = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(va), vb);
    let c0 = _mm256_sub_epi8(v, _mm256_set1_epi8(b'0' as i8));
    let ca_ = _mm256_sub_epi8(v, _mm256_set1_epi8(b'a' as i8));
    let isd = _mm256_cmpeq_epi8(_mm256_min_epu8(c0, _mm256_set1_epi8(9)), c0);
    let isl = _mm256_cmpeq_epi8(_mm256_min_epu8(ca_, _mm256_set1_epi8(25)), ca_);
    let okm = _mm256_movemask_epi8(_mm256_or_si256(isd, isl)) as u32;
    // lane 0: keys.rs codes ('0'..'9' -> 1..10, 'a'..'z' -> 38..63) = c0 + 1
    // - (letter ? 12 : 0); lane 1: base-36 values = c0 - (letter ? 0x27 : 0)
    let add = _mm256_setr_epi64x(0x0101_0101_0101_0101, 0x0101_0101_0101_0101, 0, 0);
    let kk = _mm256_setr_epi64x(0x0C0C_0C0C_0C0C_0C0C, 0x0C0C_0C0C_0C0C_0C0C, 0x2727_2727_2727_2727, 0x2727_2727_2727_2727);
    let val = _mm256_sub_epi8(_mm256_add_epi8(c0, add), _mm256_and_si256(isl, kk));
    let lm = _mm256_inserti128_si256::<1>(
        _mm256_castsi128_si256(_mm_load_si128(LM0.0.as_ptr().add(n0) as *const __m128i)),
        _mm_load_si128(LM.0.as_ptr().add(mb) as *const __m128i),
    );
    let dv = _mm256_and_si256(val, lm);
    // lane 0: [64, 1] then [4096, 1]; lane 1: [36, 1] then [1296, 1]
    let w = _mm256_maddubs_epi16(dv, _mm256_setr_epi64x(0x0140_0140_0140_0140, 0x0140_0140_0140_0140, 0x0124_0124_0124_0124, 0x0124_0124_0124_0124));
    let q = _mm256_madd_epi16(w, _mm256_setr_epi64x(0x0001_1000_0001_1000, 0x0001_1000_0001_1000, 0x0001_0510_0001_0510, 0x0001_0510_0001_0510));
    // click_id: dwords 0, 1, 2 of lane 0 hold codes 0-3, 4-7, 8-9 (x 4096)
    let lo = _mm256_castsi256_si128(q);
    let d01 = _mm_cvtsi128_si64(lo) as u64;
    let d2 = _mm_extract_epi32::<2>(lo) as u32 as u64;
    let key60 = ((d01 & 0xFF_FFFF) << 36) | ((d01 >> 32) << 12) | (d2 >> 12);
    // user: dwords 2, 3 of lane 1
    let hi = _mm256_mul_epu32(q, _mm256_set_epi64x(1_679_616, 0, 1_679_616, 0));
    let s = _mm256_add_epi64(hi, _mm256_srli_epi64::<32>(q));
    (key60, _mm256_extract_epi64::<3>(s) as u64, okm)
}

/// One timestamp's 16 bytes from a + 11, as `tvec` (lane a only).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn tvec1(a: *const u8) -> (u32, u32, u32, u32) {
    let v = _mm_loadu_si128(a.add(11) as *const __m128i);
    let d = _mm_sub_epi8(v, _mm_set1_epi8(b'0' as i8));
    let isd = _mm_cmpeq_epi8(_mm_min_epu8(d, _mm_set1_epi8(9)), d);
    const C: i8 = b':' as i8;
    const Z: i8 = b'Z' as i8;
    let iseq = _mm_cmpeq_epi8(v, _mm_setr_epi8(0, 0, C, 0, 0, C, 0, 0, Z, 0, 0, C, 0, 0, 0, 0));
    const E: i8 = -1;
    let okm = _mm_movemask_epi8(_mm_blendv_epi8(isd, iseq, _mm_setr_epi8(0, 0, E, 0, 0, E, 0, 0, E, 0, 0, E, 0, 0, 0, 0))) as u32;
    let dg = _mm_shuffle_epi8(d, _mm_setr_epi8(0, 1, 3, 4, 6, 7, -1, -1, 9, 10, 12, 13, -1, -1, -1, -1));
    let w = _mm_maddubs_epi16(dg, _mm_set1_epi16(0x010A));
    let rm = _mm_movemask_epi8(_mm_cmpgt_epi16(w, _mm_setr_epi16(23, 59, 59, 0, 23, 59, 0, 0))) as u32;
    let s = _mm_madd_epi16(w, _mm_setr_epi16(3600, 60, 1, 0, 60, 1, 0, 0));
    let r = _mm_cvtsi128_si64(_mm_hadd_epi32(s, s)) as u64;
    (r as u32, (r >> 32) as u32, okm, rm)
}

/// Every click line of b[start..end): fast lines (valid clicks) to `sink`;
/// cut lines in doubt to `fl` (their four unquoted fields, exactly the
/// scanner's); every other part to `fr` as a line-aligned range for the
/// general scanner. Lines that are certainly dropped (empty IDs, ts outside
/// the click window, unknown campaign) produce nothing, as the oracle.
///
/// Fast path: the 64 bytes at the line start hold the whole line (4th
/// delimiter = newline, quotes as fastev's `window`); click_id 1..10 bytes of
/// [0-9a-z] (packed as keys::pack); user_id canonical with a dense slot; ts
/// F1 / F2 with the date in the click table or 13 digits; campaign through
/// IdTable::get_wide itself.
///
/// # Safety
/// The CPU must support AVX2, BMI1, BMI2, LZCNT and POPCNT.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[inline(never)]
pub unsafe fn cl_kernel<'a, S: ClRec, FL: FnMut(&[&'a [u8]; 4]), FR: FnMut(usize, usize)>(
    b: &'a [u8],
    start: usize,
    end: usize,
    k: &K3,
    ctx: &Ctx,
    sink: &mut S,
    mut fl: FL,
    mut fr: FR,
) -> Stats {
    const MARGIN_C: usize = 128;
    let mut st = Stats::default();
    assert!(start <= end && end <= b.len());
    if start >= end {
        return st;
    }
    if b.len() < MARGIN_C + FRONT {
        st.range_fb += 1;
        fr(start, end);
        return st;
    }
    let base = b.as_ptr();
    let lim = b.len() - MARGIN_C;
    let mut off = start;
    while off < end && off < FRONT {
        let e = scan::next_line_start(b, off);
        st.range_fb += 1;
        fr(off, e);
        off = e;
    }
    let tab = &k.cl_tab;
    let (ku, ws) = (k.ku, k.ws);
    let pu_add = (1u64 << 63).wrapping_sub(k.pmax << 9);
    let cspan1 = k.cspan as i64 - 1;
    let wk = ws - k.cws; // ms offsets: window_start -> click window start
    while off < end && off <= lim {
        let p = base.add(off);
        let (c0, n0, q0) = masks32(p);
        let (c1, n1, q1) = masks32(p.add(32));
        let n = n0 | (n1 << 32);
        let d = c0 | (c1 << 32) | n;
        let q = q0 | (q1 << 32);
        let d3 = blsr(blsr(blsr(d)));
        let r = upto(d3);
        let (wb, cs, ce) = window(d, q, r);
        let last = d3 & d3.wrapping_neg();
        let sbad = (d3 == 0) as u64 | wb | ((n ^ last) & r);
        let le = tz(d3);
        // positions (garbage unless sbad == 0; reads stay inside MARGIN_C)
        let s0 = tz(cs);
        let e0 = tz(ce);
        let csb = blsr(cs);
        let ceb = blsr(ce);
        let s1 = tz(csb);
        let e1 = tz(ceb);
        let csc = blsr(csb);
        let cec = blsr(ceb);
        let s2 = tz(csc);
        let s3 = tz(blsr(csc));
        let e3 = tz(blsr(cec));
        // ---- IDs
        let n0c = e0.wrapping_sub(s0);
        let m1 = e1.wrapping_sub(s1).wrapping_sub(1);
        let (key60, xb, okm) = ids_cl(p.add(s0), p.add(e1), n0c & 15, m1 & 15);
        let ck = (key60 << 4) | n0c as u64;
        let kbad = *NEED0.get_unchecked(n0c & 63) & !(okm as u64);
        let ib = IDM.get_unchecked(m1 & 63);
        let u = unscramble(xb, ku);
        let slot = ((u >> 9) * 8 + (u & 511)) as u32;
        let ubad = (ib.need & !((okm >> 16) as u64))
            | (*p.add(s1) ^ b'u') as u64
            | (xb >> 40)
            | (xb.wrapping_sub(ib.pow) >> 63)
            | (u.wrapping_add(pu_add) >> 63)
            | (u & 0x1F8);
        // ---- ts
        let q3 = p.add(s3);
        let n3 = e3.wrapping_sub(s3);
        let (dbad, de) = date(q3, tab);
        let (sec, omin, tok, trng) = tvec1(q3);
        let civ = de.off_ms.wrapping_add(sec as i64 * 1000);
        let b19 = *q3.add(19) as u32;
        let f2v = civ.wrapping_sub((omin as i64 * 60_000).wrapping_mul(0x2C - b19 as i64));
        let (f3, f3bad) = f3_13(q3);
        let ntok = !tok as u64;
        let f1bad = (ntok & 0x1FF) | (trng & 0x3F) as u64 | dbad | (n3 ^ 20) as u64;
        let f2bad = (ntok & 0x3EFF) | (trng & 0xF3F) as u64 | dbad | (b19.wrapping_sub(0x2B) & 0xFFFF_FFFD) as u64;
        let is13 = n3 == 13;
        let is25 = n3 == 25;
        // ms after the click window start
        let t_off = sel(is13, (f3 as i64).wrapping_sub(k.cws) as u64, sel(is25, f2v.wrapping_add(wk) as u64, civ.wrapping_add(wk) as u64)) as i64;
        let tbad = sel(is13, f3bad as u64, sel(is25, f2bad, f1bad));
        let outw = (t_off as u64 >> 63) | (cspan1.wrapping_sub(t_off) as u64 >> 63);
        // campaign (IdTable, exact for every string): only for lines that
        // could be recorded
        let all = sbad | kbad | ubad | tbad | outw;
        if all == 0 {
            let camp = ctx.campaigns.get_wide(std::slice::from_raw_parts(p.add(s2), tz(cec) - s2));
            if camp != crate::lookup::NONE {
                sink.cl(slot, camp as u32, t_off as u32, ck);
            }
            st.fast += 1;
            off += le + 1;
            continue;
        }
        // certainly dropped before any side effect of the oracle (empty
        // click_id / user_id, or a recognised ts outside the window)
        if (sbad == 0) & ((n0c == 0) | (e1 == s1) | ((tbad == 0) & (outw != 0))) {
            st.fast += 1;
            off += le + 1;
            continue;
        }
        if sbad == 0 {
            let f: [&[u8]; 4] = [
                std::slice::from_raw_parts(p.add(s0), n0c),
                std::slice::from_raw_parts(p.add(s1), e1.wrapping_sub(s1)),
                std::slice::from_raw_parts(p.add(s2), tz(cec) - s2),
                std::slice::from_raw_parts(q3, n3),
            ];
            st.line_fb += 1;
            fl(&f);
            off += le + 1;
        } else {
            let e = slow_end(b, off);
            st.range_fb += 1;
            fr(off, e);
            off = e;
        }
    }
    if off < end {
        st.range_fb += 1;
        fr(off, end);
    }
    st
}

/// The dense pipeline's click output (Parts<DC>, nbc buckets) as a sink.
pub struct ClSink(pub *mut Parts<crate::densep::DC>, pub usize);

impl ClRec for ClSink {
    #[inline(always)]
    fn cl(&mut self, us: u32, camp: u32, ts: u32, ck: u64) {
        let v = unsafe { &mut *self.0 };
        let q = ((us >> 16) as usize).min(self.1 - 1);
        v.push(q, crate::densep::DC { us, camp, ts, _p: 0, ck, _q: 0 });
    }
}

/// A recognised purchase whose other fields make it valid: price and
/// category through main::event_pay (as ev_line).
#[cold]
#[inline(never)]
fn purchase<S: EvRec>(f: &[&[u8]; 7], ctx: &Ctx, sink: &mut S, n: u32, ingest: u32, ts: u32, slot: u32) {
    match super::event_pay(f[4], || f[6], f[5], ctx, true) {
        Some(pay) => {
            let pf = if pay != 0 { PFLAG } else { 0 };
            sink.ev(n, ingest, ts | VALID, slot | pf, pay)
        }
        None => sink.ev(n, ingest, 0, 0, 0),
    }
}

#[cold]
#[inline(never)]
fn slow_end(b: &[u8], off: usize) -> usize {
    scan::next_line_start(b, off)
}

// ---------------------------------------------------------------- densep glue

/// The dense pipeline's event output as a record sink (through a raw
/// pointer: the fallback closures reach the same EvOut).
pub struct EvOutSink(pub *mut crate::densep::EvOut, pub u32);

impl EvRec for EvOutSink {
    #[inline(always)]
    fn ev(&mut self, n: u32, ingest: u32, ts: u32, us: u32, pay: u32) {
        use crate::densep::{DE, DP};
        let o = unsafe { &mut *self.0 };
        let b = (n >> self.1) as usize;
        unsafe { o.de.push_unchecked(b, DE { n, ingest, ts, us }) };
        if pay != 0 {
            unsafe { o.dp.push_unchecked(b, DP { n, ingest, pay, _p: 0 }) };
        }
    }
}

// ---------------------------------------------------------------- checking, benchmarks

use super::{Input, Task};
use crate::densep::{self, Dense, EvOut, Xmap};
use crate::parts::Parts;

fn dense_or_exit(ev: &Input, ctx: &Ctx) -> Dense {
    if !ctx.avx2 {
        eprintln!("k3: needs AVX2/BMI2");
        std::process::exit(1);
    }
    match densep::decide(ev, ctx) {
        Some(d) => d,
        None => {
            eprintln!("k3: input not dense (set SS_DENSE=1, or SS_DENSE=force for small inputs)");
            std::process::exit(1)
        }
    }
}

/// The k3 event kernel on one task into `o`, fallbacks to densep's exact
/// routines (with `xm`).
#[cfg(target_arch = "x86_64")]
pub fn ev_task_k3(b: &[u8], task: Task, ctx: &Ctx, d: &Dense, xm: &Xmap, o: &mut EvOut) -> Stats {
    let k = params(ctx, d.ke, d.ku, d.bound, d.pmax);
    let safe_end = b.as_ptr() as usize + b.len();
    let op: *mut EvOut = o;
    let mut sink = EvOutSink(op, d.bshift);
    unsafe {
        ev_kernel::<_, _, _, FULL>(
            b,
            task.start,
            task.end,
            k,
            ctx,
            &mut sink,
            |f| densep::ev_line(f, ctx, d, xm, &mut *op, safe_end),
            |from, to| densep::ev_range(b, from, to, ctx, d, xm, &mut *op, safe_end),
        )
        .0
    }
}

fn add(a: &mut Stats, b: Stats) {
    a.fast += b.fast;
    a.line_fb += b.line_fb;
    a.range_fb += b.range_fb;
}

/// SS_K3CHECK=ev: every event task through densep's kernel and through k3;
/// the records (every bucket's DE and DP sequences, the exceptional copies,
/// the interned user IDs) must be identical. Exits nonzero on a difference.
pub fn check(what: &str, ev: &Input, cl: &Input, ctx: &Ctx) {
    let d = dense_or_exit(ev, ctx);
    if what.contains("ev") {
        check_ev(ev, ctx, &d);
    }
    if what.contains("cl") {
        check_cl(cl, ctx, &d);
    }
}

/// The k3 click kernel on one task into `v` (nbc buckets), fallbacks to
/// densep::cl_line (with `xm`).
#[cfg(target_arch = "x86_64")]
pub fn cl_task_k3(b: &[u8], task: Task, ctx: &Ctx, d: &Dense, xm: &Xmap, v: &mut Parts<crate::densep::DC>, nbc: usize) -> Stats {
    let k = params(ctx, d.ke, d.ku, d.bound, d.pmax);
    let safe_end = b.as_ptr() as usize + b.len();
    let vp: *mut Parts<crate::densep::DC> = v;
    let mut sink = ClSink(vp, nbc);
    unsafe {
        cl_kernel(
            b,
            task.start,
            task.end,
            k,
            ctx,
            &mut sink,
            |f| densep::cl_line(f, ctx, d, xm, &mut *vp, nbc, safe_end),
            |from, to| scan::for_each_record_x::<4, _, false>(b, from, to, |f| densep::cl_line(f, ctx, d, xm, &mut *vp, nbc, safe_end)),
        )
    }
}

fn dc_list(v: &Parts<crate::densep::DC>, nbc: usize) -> Vec<(usize, u32, u32, u32, u64)> {
    let mut out = Vec::new();
    for i in 0..nbc {
        for r in v.get(i).iter().flat_map(|s| s.iter()) {
            out.push((i, r.us, r.camp, r.ts, r.ck));
        }
    }
    out
}

/// SS_K3CHECK=cl: every click task through densep's click kernel and
/// through k3; the click records (per bucket, in order) and the interned
/// user IDs must be identical.
fn check_cl(cl: &Input, ctx: &Ctx, d: &Dense) {
    let nbc = ((d.pmax * 8) >> 16) as usize + 1;
    let mut a: Parts<crate::densep::DC> = Parts::with_wc(nbc, 64, false);
    let mut b: Parts<crate::densep::DC> = Parts::with_wc(nbc, 64, false);
    let (xa, xb) = (Xmap::new(), Xmap::new());
    let mut tot = Stats::default();
    let (mut recs, mut lines) = (0usize, 0usize);
    let saved = ON.load(Ordering::Relaxed);
    ON.store(false, Ordering::Relaxed);
    for task in &cl.tasks {
        let buf = cl.maps[task.file as usize].bytes();
        let safe_end = buf.as_ptr() as usize + buf.len();
        a.reset();
        b.reset();
        densep::cl_task_any(buf, *task, ctx, d, &xa, &mut a, nbc, safe_end);
        #[cfg(target_arch = "x86_64")]
        add(&mut tot, cl_task_k3(buf, *task, ctx, d, &xb, &mut b, nbc));
        lines += buf[task.start..task.end].iter().filter(|&&c| c == b'\n').count();
        let (la, lb) = (dc_list(&a, nbc), dc_list(&b, nbc));
        recs += la.len();
        if la != lb {
            eprintln!("k3check: MISMATCH clicks file {} [{}, {}) ({} vs {} records)", task.file, task.start, task.end, la.len(), lb.len());
            let mut l = task.start;
            let mut shown = 0;
            while l < task.end && shown < 5 {
                let e = scan::next_line_start(buf, l);
                let one = Task { file: task.file, start: l, end: e };
                a.reset();
                b.reset();
                densep::cl_task_any(buf, one, ctx, d, &xa, &mut a, nbc, safe_end);
                #[cfg(target_arch = "x86_64")]
                cl_task_k3(buf, one, ctx, d, &xb, &mut b, nbc);
                let (x, y) = (dc_list(&a, nbc), dc_list(&b, nbc));
                if x != y {
                    eprintln!("line at {}: {:?}\n  old: {:?}\n  k3:  {:?}", l, String::from_utf8_lossy(&buf[l..e]), x, y);
                    shown += 1;
                }
                l = e;
            }
            std::process::exit(1);
        }
    }
    if xa.key_table() != xb.key_table() {
        eprintln!("k3check: MISMATCH interned user IDs (clicks)");
        std::process::exit(1);
    }
    ON.store(saved, Ordering::Relaxed);
    eprintln!(
        "k3check cl: OK {} tasks, {} lines, {} click records identical; k3 fast (records + certain drops) {} ({:.3}%), line fallback {} ({:.3}%), range fallback {}",
        cl.tasks.len(),
        lines,
        recs,
        tot.fast,
        tot.fast as f64 * 100.0 / lines.max(1) as f64,
        tot.line_fb,
        tot.line_fb as f64 * 100.0 / lines.max(1) as f64,
        tot.range_fb
    );
}

struct NullCl(u64);
impl ClRec for NullCl {
    #[inline(always)]
    fn cl(&mut self, us: u32, camp: u32, ts: u32, ck: u64) {
        self.0 = self.0.wrapping_add((us ^ camp ^ ts) as u64 ^ ck);
    }
}

/// SS_K3BENCH=cl[:stages]: click-parse stages (old, k3-null, k3).
fn bench_cl(stages: &str, cl: &Input, ctx: &Ctx, d: &Dense) {
    let mut lines = 0usize;
    for t in &cl.tasks {
        let b = cl.maps[t.file as usize].bytes();
        lines += b[t.start..t.end].iter().filter(|&&c| c == b'\n').count();
    }
    let rounds = std::env::var("SS_K3ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(3usize);
    let thr: Vec<usize> = match std::env::var("SS_K3THREADS") {
        Ok(v) => v.split(',').filter_map(|x| x.parse().ok()).collect(),
        Err(_) => vec![1, ctx.threads],
    };
    let nbc = ((d.pmax * 8) >> 16) as usize + 1;
    let estc = cl.bytes / 44 + 1;
    let keep = crate::sys::arenas().len();
    for &nt in &thr {
        for stage in stages.split(',') {
            let mut best = (f64::MAX, 0f64);
            for _ in 0..rounds {
                let next = std::sync::atomic::AtomicUsize::new(0);
                let xm = Xmap::new();
                let capc = estc / (nt * nbc) * 5 / 4 + 64;
                super::PHASE_CPU_NS.store(0, Ordering::Relaxed);
                let t0 = std::time::Instant::now();
                let r = super::par(nt, |_| {
                    let mut v: Parts<crate::densep::DC> = Parts::with_wc(nbc, capc, true);
                    let mut nc = NullCl(0);
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(&task) = cl.tasks.get(i) else { break };
                        let b = cl.maps[task.file as usize].bytes();
                        let safe_end = b.as_ptr() as usize + b.len();
                        let vp: *mut Parts<crate::densep::DC> = &mut v;
                        #[cfg(target_arch = "x86_64")]
                        match stage {
                            "old" => densep::cl_task_any(b, task, ctx, d, &xm, unsafe { &mut *vp }, nbc, safe_end),
                            "k3" => {
                                cl_task_k3(b, task, ctx, d, &xm, unsafe { &mut *vp }, nbc);
                            }
                            "k3-null" => unsafe {
                                let k = params(ctx, d.ke, d.ku, d.bound, d.pmax);
                                cl_kernel(
                                    b,
                                    task.start,
                                    task.end,
                                    k,
                                    ctx,
                                    &mut nc,
                                    |f| densep::cl_line(f, ctx, d, &xm, &mut *vp, nbc, safe_end),
                                    |from, to| scan::for_each_record_x::<4, _, false>(b, from, to, |f| densep::cl_line(f, ctx, d, &xm, &mut *vp, nbc, safe_end)),
                                );
                            },
                            _ => panic!("k3bench: unknown click stage {}", stage),
                        }
                    }
                    v.finish();
                    nc.0 ^ v.total() as u64
                });
                let wall = t0.elapsed().as_secs_f64() * 1e3;
                let cpu = super::PHASE_CPU_NS.swap(0, Ordering::Relaxed) as f64;
                std::mem::forget(r);
                crate::sys::release_from(keep);
                if wall < best.0 {
                    best = (wall, cpu);
                }
            }
            eprintln!("k3bench cl {:10} threads={} wall={:7.1}ms thread-time/line={:6.1}ns", stage, nt, best.0, best.1 / lines as f64);
        }
    }
    eprintln!("k3bench: {} click lines, {} MB", lines, cl.bytes >> 20);
}

fn same_de(a: &Parts<crate::densep::DE>, b: &Parts<crate::densep::DE>, nb: usize) -> Option<usize> {
    for i in 0..nb {
        let x: Vec<[u32; 4]> = a.get(i).iter().flat_map(|s| s.iter()).map(|r| [r.n, r.ingest, r.ts, r.us]).collect();
        let y: Vec<[u32; 4]> = b.get(i).iter().flat_map(|s| s.iter()).map(|r| [r.n, r.ingest, r.ts, r.us]).collect();
        if x != y {
            return Some(i);
        }
    }
    None
}

fn same_dp(a: &Parts<crate::densep::DP>, b: &Parts<crate::densep::DP>, nb: usize) -> Option<usize> {
    for i in 0..nb {
        let x: Vec<[u32; 3]> = a.get(i).iter().flat_map(|s| s.iter()).map(|r| [r.n, r.ingest, r.pay]).collect();
        let y: Vec<[u32; 3]> = b.get(i).iter().flat_map(|s| s.iter()).map(|r| [r.n, r.ingest, r.pay]).collect();
        if x != y {
            return Some(i);
        }
    }
    None
}

fn xe_list(o: &EvOut) -> Vec<(u64, u32, bool, u32, u32, u32)> {
    o.xe.iter().map(|x| (x.key, x.ingest, x.valid, x.us, x.ts, x.pay)).collect()
}

fn check_ev(ev: &Input, ctx: &Ctx, d: &Dense) {
    let nb = (d.bound >> d.bshift) as usize + 1;
    let mk = || EvOut { de: Parts::with_wc(nb, 64, false), dp: Parts::with_wc(nb, 8, false), xe: Vec::new() };
    let (mut a, mut b) = (mk(), mk());
    let (xa, xb) = (Xmap::new(), Xmap::new());
    let mut tot = Stats::default();
    let (mut ncopies, mut lines) = (0usize, 0usize);
    let saved = ON.load(Ordering::Relaxed);
    ON.store(false, Ordering::Relaxed);
    for task in &ev.tasks {
        let buf = ev.maps[task.file as usize].bytes();
        let safe_end = buf.as_ptr() as usize + buf.len();
        for o in [&mut a, &mut b] {
            o.de.reset();
            o.dp.reset();
            o.xe.clear();
        }
        densep::ev_task_any(buf, *task, ctx, d, &xa, &mut a, safe_end);
        #[cfg(target_arch = "x86_64")]
        add(&mut tot, ev_task_k3(buf, *task, ctx, d, &xb, &mut b));
        lines += buf[task.start..task.end].iter().filter(|&&c| c == b'\n').count();
        ncopies += a.de.total() + a.xe.len();
        let bad = if let Some(i) = same_de(&a.de, &b.de, nb) {
            Some(format!("DE bucket {}", i))
        } else if let Some(i) = same_dp(&a.dp, &b.dp, nb) {
            Some(format!("DP bucket {}", i))
        } else if xe_list(&a) != xe_list(&b) {
            Some("XE".to_string())
        } else {
            None
        };
        if let Some(what) = bad {
            eprintln!("k3check: MISMATCH ({}) file {} [{}, {})", what, task.file, task.start, task.end);
            // the first line that differs on its own
            let mut l = task.start;
            let mut shown = 0;
            while l < task.end && shown < 5 {
                let e = scan::next_line_start(buf, l);
                let one = Task { file: task.file, start: l, end: e };
                for o in [&mut a, &mut b] {
                    o.de.reset();
                    o.dp.reset();
                    o.xe.clear();
                }
                densep::ev_task_any(buf, one, ctx, d, &xa, &mut a, safe_end);
                #[cfg(target_arch = "x86_64")]
                ev_task_k3(buf, one, ctx, d, &xb, &mut b);
                let dump = |o: &EvOut| -> Vec<String> {
                    let mut v = Vec::new();
                    for i in 0..nb {
                        for r in o.de.get(i).iter().flat_map(|s| s.iter()) {
                            v.push(format!("DE n={} ing={} ts={:#x} us={:#x}", r.n, r.ingest, r.ts, r.us));
                        }
                        for r in o.dp.get(i).iter().flat_map(|s| s.iter()) {
                            v.push(format!("DP n={} ing={} pay={}", r.n, r.ingest, r.pay));
                        }
                    }
                    for x in &o.xe {
                        v.push(format!("XE key={:#x} ing={} valid={} us={} ts={} pay={}", x.key, x.ingest, x.valid, x.us, x.ts, x.pay));
                    }
                    v
                };
                let (da, db) = (dump(&a), dump(&b));
                if da != db {
                    eprintln!("line at {}: {:?}\n  old: {:?}\n  k3:  {:?}", l, String::from_utf8_lossy(&buf[l..e]), da, db);
                    shown += 1;
                }
                l = e;
            }
            std::process::exit(1);
        }
    }
    if xa.key_table() != xb.key_table() {
        eprintln!("k3check: MISMATCH interned user IDs");
        std::process::exit(1);
    }
    ON.store(saved, Ordering::Relaxed);
    eprintln!(
        "k3check ev: OK {} tasks, {} lines, {} copies identical; k3 fast {} ({:.3}%, purchases incl.), line fallback {} ({:.3}%), range fallback {}",
        ev.tasks.len(),
        lines,
        ncopies,
        tot.fast,
        tot.fast as f64 * 100.0 / lines.max(1) as f64,
        tot.line_fb,
        tot.line_fb as f64 * 100.0 / lines.max(1) as f64,
        tot.range_fb
    );
}

struct NullEv(u64);
impl EvRec for NullEv {
    #[inline(always)]
    fn ev(&mut self, n: u32, ingest: u32, ts: u32, us: u32, pay: u32) {
        self.0 = self.0.wrapping_add((n ^ ingest ^ ts ^ us ^ pay) as u64);
    }
}

/// Benchmark sink: 16-byte records written one after the other (no buckets).
struct LinEv {
    p: *mut [u32; 4],
    i: usize,
    cap: usize,
}
impl EvRec for LinEv {
    #[inline(always)]
    fn ev(&mut self, n: u32, ingest: u32, ts: u32, us: u32, _pay: u32) {
        if self.i < self.cap {
            unsafe { self.p.add(self.i).write([n, ingest, ts, us]) };
            self.i += 1;
        }
    }
}

/// Benchmark sink: EvOut's write-combining scatter with 2^shift numbers per bucket.
struct BucketEv {
    de: Parts<crate::densep::DE>,
    shift: u32,
}
impl EvRec for BucketEv {
    #[inline(always)]
    fn ev(&mut self, n: u32, ingest: u32, ts: u32, us: u32, _pay: u32) {
        unsafe { self.de.push_unchecked((n >> self.shift) as usize, crate::densep::DE { n, ingest, ts, us }) };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[inline(never)]
unsafe fn old_split_avx2(b: &[u8], task: Task) -> u64 {
    let safe_end = b.as_ptr() as usize + b.len();
    let mut acc = 0u64;
    let ap: *mut u64 = &mut acc;
    crate::fastev::for_each_line2::<true, _, _>(
        b,
        task.start,
        task.end,
        safe_end,
        |f| *ap = (*ap).wrapping_add((f[0].len() ^ f[6].len()) as u64),
        |from, to| scan::for_each_record_x::<7, _, true>(b, from, to, |f| *ap = (*ap).wrapping_add((f[0].len() ^ f[6].len()) as u64)),
    );
    acc
}

/// One benchmark stage on one task.
#[cfg(target_arch = "x86_64")]
fn ev_stage(stage: &str, b: &[u8], task: Task, ctx: &Ctx, d: &Dense, xm: &Xmap, o: &mut EvOut, lin: &mut LinEv, bk: &mut BucketEv) -> u64 {
    let safe_end = b.as_ptr() as usize + b.len();
    let k = params(ctx, d.ke, d.ku, d.bound, d.pmax);
    let op: *mut EvOut = o;
    let mut ns = NullEv(0);
    unsafe {
        match stage {
            "old-split" => old_split_avx2(b, task),
            "old" => {
                densep::ev_task_any(b, task, ctx, d, xm, &mut *op, safe_end);
                0
            }
            "k3-split" => {
                ev_kernel::<_, _, _, SPLIT>(b, task.start, task.end, k, ctx, &mut ns, |f| densep::ev_line(f, ctx, d, xm, &mut *op, safe_end), |from, to| densep::ev_range(b, from, to, ctx, d, xm, &mut *op, safe_end)).1
            }
            "k3-decode" => {
                ev_kernel::<_, _, _, DECODE>(b, task.start, task.end, k, ctx, &mut ns, |f| densep::ev_line(f, ctx, d, xm, &mut *op, safe_end), |from, to| densep::ev_range(b, from, to, ctx, d, xm, &mut *op, safe_end)).1
            }
            "k3-null" => {
                ev_kernel::<_, _, _, FULL>(b, task.start, task.end, k, ctx, &mut ns, |f| densep::ev_line(f, ctx, d, xm, &mut *op, safe_end), |from, to| densep::ev_range(b, from, to, ctx, d, xm, &mut *op, safe_end));
                ns.0
            }
            "k3" => {
                ev_task_k3(b, task, ctx, d, xm, &mut *op);
                0
            }
            "k3-lin" => {
                ev_kernel::<_, _, _, FULL>(b, task.start, task.end, k, ctx, lin, |f| densep::ev_line(f, ctx, d, xm, &mut *op, safe_end), |from, to| densep::ev_range(b, from, to, ctx, d, xm, &mut *op, safe_end));
                lin.i as u64
            }
            s if s.starts_with("k3-b") => {
                ev_kernel::<_, _, _, FULL>(b, task.start, task.end, k, ctx, bk, |f| densep::ev_line(f, ctx, d, xm, &mut *op, safe_end), |from, to| densep::ev_range(b, from, to, ctx, d, xm, &mut *op, safe_end));
                0
            }
            _ => panic!("k3bench: unknown stage {}", stage),
        }
    }
}

/// SS_K3BENCH=ev[:stage,stage,...]: event-parse stages, best of 3
/// (SS_K3ROUNDS), on 1 and NCPU threads (SS_K3THREADS): wall time and
/// thread time per event line.
pub fn bench(what: &str, ev: &Input, cl: &Input, ctx: &Ctx) {
    let d = dense_or_exit(ev, ctx);
    if let Some(st) = what.strip_prefix("cl") {
        let st = st.strip_prefix(':').unwrap_or("");
        return bench_cl(if st.is_empty() { "old,k3-null,k3" } else { st }, cl, ctx, &d);
    }
    let (kind, stages) = what.split_once(':').unwrap_or((what, "old-split,old,k3-split,k3-decode,k3-null,k3"));
    if kind != "ev" {
        eprintln!("k3bench: unknown kind {}", kind);
        std::process::exit(1);
    }
    ON.store(false, Ordering::Relaxed);
    let mut lines = 0usize;
    for t in &ev.tasks {
        let b = ev.maps[t.file as usize].bytes();
        lines += b[t.start..t.end].iter().filter(|&&c| c == b'\n').count();
    }
    let rounds = std::env::var("SS_K3ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(3usize);
    let thr: Vec<usize> = match std::env::var("SS_K3THREADS") {
        Ok(v) => v.split(',').filter_map(|x| x.parse().ok()).collect(),
        Err(_) => vec![1, ctx.threads],
    };
    let nb = (d.bound >> d.bshift) as usize + 1;
    let est = ev.bytes / 70 + 1;
    let keep = crate::sys::arenas().len();
    for &nt in &thr {
        for stage in stages.split(',') {
            let mut best = (f64::MAX, 0f64);
            let mut acc = 0u64;
            for _ in 0..rounds {
                let next = std::sync::atomic::AtomicUsize::new(0);
                let xm = Xmap::new();
                let cap = est / (nt * nb) * 5 / 4 + 64;
                let capp = est / 16 / (nt * nb) + 32;
                super::PHASE_CPU_NS.store(0, Ordering::Relaxed);
                let t0 = std::time::Instant::now();
                let r = super::par(nt, |_| {
                    let mut o = EvOut { de: Parts::with_wc(nb, cap, true), dp: Parts::with_wc(nb, capp, true), xe: Vec::new() };
                    let lcap = est / nt * 5 / 4 + 1024;
                    let mut lin = LinEv { p: crate::sys::alloc_arena(lcap * 16) as *mut [u32; 4], i: 0, cap: lcap };
                    let shift: u32 = stage.strip_prefix("k3-b").and_then(|x| x.parse().ok()).unwrap_or(16);
                    let nbk = (d.bound >> shift) as usize + 1;
                    let mut bk = BucketEv { de: Parts::with_wc(nbk, est / (nt * nbk) * 5 / 4 + 64, true), shift };
                    let mut a = 0u64;
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(&task) = ev.tasks.get(i) else { break };
                        let b = ev.maps[task.file as usize].bytes();
                        #[cfg(target_arch = "x86_64")]
                        {
                            a = a.wrapping_add(ev_stage(stage, b, task, ctx, &d, &xm, &mut o, &mut lin, &mut bk));
                        }
                    }
                    o.de.finish();
                    o.dp.finish();
                    bk.de.finish();
                    a ^ (o.de.total() + o.xe.len() + bk.de.total()) as u64
                });
                let wall = t0.elapsed().as_secs_f64() * 1e3;
                let cpu = super::PHASE_CPU_NS.swap(0, Ordering::Relaxed) as f64;
                acc = r.iter().fold(0, |x, &y| x ^ y);
                std::mem::forget(r);
                crate::sys::release_from(keep);
                if wall < best.0 {
                    best = (wall, cpu);
                }
            }
            eprintln!("k3bench ev {:10} threads={} wall={:7.1}ms thread-time/line={:6.1}ns (acc {})", stage, nt, best.0, best.1 / lines as f64, acc % 997);
        }
    }
    eprintln!("k3bench: {} event lines, {} MB", lines, ev.bytes >> 20);
}
