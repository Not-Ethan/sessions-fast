#![cfg_attr(not(target_arch = "x86_64"), allow(dead_code, unused_imports))]
//! k5: a fused click-line kernel for `main::click_task` (SS_K5=1).
//!
//! The old click path costs about 694 instructions per click line
//! (callgrind, e23): a splitter that calls a per-line closure (fields
//! through memory), a 128-byte retry that calls its mask routine out of
//! line, two separate ID packers, a 40-byte ring entry copied twice. This
//! kernel does the same work in one inlined per-task loop:
//!
//! * split: comma / newline / quote masks of the 64 bytes at the line start
//!   (AVX2); when they hold fewer than four delimiters, the next 64 bytes
//!   too (128-bit masks, inline). The field rule is `fastev::window`
//!   verbatim (first three delimiters commas, the fourth the newline, every
//!   field holds no quote or exactly two, as first and last byte): then the
//!   line has exactly 4 fields and they are exactly the scanner's.
//! * timestamp: F1 / F2 with the date looked up as 11 exact bytes in a
//!   28-date table (built and verified with `ctx.tsp.parse` itself) and
//!   `HH:MM:SS` + offset decoded in one 128-bit vector; F3 13 digits in one
//!   vector. Any other shape, or a date not in the table, goes to
//!   `ctx.tsp.parse` (the old routine).
//! * campaign: an exact (bytes, length) hash table of the campaign IDs
//!   (`CTab`, built from `ctx.campaigns.get` itself; IDs are opaque bytes).
//! * both IDs packed at once, one per 128-bit lane (`pack2`, the same
//!   operations as `keys::pack_wide`). A user ID that does not pack goes to
//!   `keys::key_of`, a click ID that does not pack to `keys::click_key`,
//!   only for valid lines, in line order (their side effects as before).
//! * person: the same 16-deep FIFO ring (prefetch at parse, lookup 16
//!   valid lines later), so records reach the sink in line order, as
//!   before; the lookup is `LinkTable::person_hash` with the 4-slot group
//!   compared in two ymm registers (`Probe`, through `LinkTable::view`).
//!
//! Every line the kernel does not cut (any other shape, a line within
//! MARGIN bytes of the end of the buffer) goes through the general scanner
//! on its own and the old per-line logic (`slow_fields`), into the same
//! ring. `SS_K5CHECK=1` runs every click task through both paths and
//! compares the records one by one.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

/// k5 on when SS_K5 is not set.
pub const DEFAULT: bool = true;
/// click tasks through k5 (SS_K5; needs AVX2, see `main::click_task_avx2`).
pub static ON: AtomicBool = AtomicBool::new(false);
/// statistics (checks only): lines cut by the kernel / handed to the scanner
pub static COUNT: AtomicBool = AtomicBool::new(false);
pub static N_FAST: AtomicUsize = AtomicUsize::new(0);
pub static N_WIDE: AtomicUsize = AtomicUsize::new(0);
pub static N_SLOW: AtomicUsize = AtomicUsize::new(0);
pub static N_TSSLOW: AtomicUsize = AtomicUsize::new(0);

use super::{click_task, keys, par, sys, CRec, CSink, Ctx, Input, LocBits, PCache, Task, DAY, NONE, PHASE_CPU_NS, WEEK};
use crate::fastev;
use crate::links::LinkTable;
use crate::parts::Parts;
use crate::scan;

/// Readable bytes from a kernel line start: the 128-byte window, and 32
/// bytes from any field start (< 128), with slack.
const MARGIN: usize = 192;

/// Person lookups are deferred this many valid lines (prefetch distance).
const RING: usize = 16;

// ---------------------------------------------------------------- date table

/// One date: bytes 0..8 `YYYY-MM-` and 8..11 `DDT` exactly, and the epoch
/// ms of its midnight.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
struct DEnt {
    pat: u64,
    /// `DDT` (24 bits); 0xFFFF_FFFF for an unused slot (never matches)
    dd: u32,
    _p: u32,
    ms: i64,
    _q: i64,
}

#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct DTab([DEnt; 64]);

/// Slot of bytes 8..12 as a little-endian u32 (bytes 8, 9 = `DD`): low 2
/// bits of the tens digit, low 4 bits of the units digit (distinct for
/// "00".."39"; 28 consecutive dates have distinct days of month).
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

/// Kernel parameters of a context (window and timestamp parser).
pub struct K5 {
    ws: i64,
    we: i64,
    tab: DTab,
    camp: CTab,
}

/// Campaign IDs of at most 16 bytes as exact (bytes, length) -> index:
/// every ID of ctx.camp_ids with its `ctx.campaigns.get` value. A string
/// that is not one of them gets NONE from `ctx.campaigns.get` too (map
/// mode: exact keys; direct mode: prefix + canonical decimal, a unique
/// spelling), so a miss here is NONE. IDs are compared as opaque bytes.
struct CTab {
    slots: Vec<CSlot>,
    shift: u32,
    mask: usize,
}

#[derive(Clone, Copy)]
#[repr(C, align(32))]
struct CSlot {
    w0: u64,
    w1: u64,
    /// length; u32::MAX for an empty slot
    len: u32,
    v: u32,
}

#[inline(always)]
fn chash(w0: u64, w1: u64, len: usize) -> u64 {
    (w0 ^ w1.rotate_left(31) ^ ((len as u64) << 58)).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

impl CTab {
    fn new(ctx: &Ctx) -> CTab {
        let n = ctx.camp_ids.len();
        let size = (n * 2).next_power_of_two().max(16);
        let bits = size.trailing_zeros();
        let mut t = CTab { slots: vec![CSlot { w0: 0, w1: 0, len: u32::MAX, v: 0 }; size], shift: 64 - bits, mask: size - 1 };
        for id in &ctx.camp_ids {
            if id.len() > 16 {
                continue;
            }
            let v = ctx.campaigns.get(id);
            if v == NONE {
                continue;
            }
            let mut b = [0u8; 16];
            b[..id.len()].copy_from_slice(id);
            let w0 = u64::from_le_bytes(b[..8].try_into().unwrap());
            let w1 = u64::from_le_bytes(b[8..].try_into().unwrap());
            let mut i = (chash(w0, w1, id.len()) >> t.shift) as usize;
            loop {
                let e = &mut t.slots[i];
                if e.len == u32::MAX {
                    *e = CSlot { w0, w1, len: id.len() as u32, v: v as u32 };
                    break;
                }
                if e.w0 == w0 && e.w1 == w1 && e.len as usize == id.len() {
                    assert_eq!(e.v, v as u32, "k5: one campaign ID, two indices");
                    break;
                }
                i = (i + 1) & t.mask;
            }
        }
        t
    }
}

static PARAMS: Mutex<Vec<(usize, &'static K5)>> = Mutex::new(Vec::new());

/// The 28 dates from the day before window_start - 7 days. A date enters
/// only if `ctx.tsp.parse` gives `YYYY-MM-DDT00:00:00Z` its midnight and
/// the midnight +- 3 days stays far from overflow.
fn build(ctx: &Ctx) -> K5 {
    let mut t = DTab([DEnt { pat: 0, dd: 0xFFFF_FFFF, _p: 0, ms: 0, _q: 0 }; 64]);
    let first = (ctx.ws - WEEK).div_euclid(DAY) - 1;
    for day in first..first + 28 {
        let (y, m, d) = civil_from_days(day);
        if !(0..=9999).contains(&y) {
            continue;
        }
        let txt = format!("{:04}-{:02}-{:02}T00:00:00Z", y, m, d);
        let tb = txt.as_bytes();
        let Some(ms) = day.checked_mul(DAY) else { continue };
        if ctx.tsp.parse(tb) != Some(ms) || ms.checked_add(3 * DAY).is_none() || ms.checked_sub(3 * DAY).is_none() {
            continue;
        }
        let k = dslot(u32::from_le_bytes([tb[8], tb[9], tb[10], tb[11]]));
        assert!(t.0[k].dd == 0xFFFF_FFFF, "k5: date slot collision");
        t.0[k] = DEnt { pat: u64::from_le_bytes(tb[..8].try_into().unwrap()), dd: u32::from_le_bytes([tb[8], tb[9], tb[10], 0]), _p: 0, ms, _q: 0 };
    }
    K5 { ws: ctx.ws, we: ctx.we, tab: t, camp: CTab::new(ctx) }
}

/// The (cached) parameters for `ctx` (keyed by its window and the address
/// of its timestamp parser; `forget` drops them, for tests).
pub fn params(ctx: &Ctx) -> &'static K5 {
    let id = &ctx.tsp as *const _ as usize;
    let mut v = PARAMS.lock().unwrap();
    for &(i, k) in v.iter() {
        if i == id && k.ws == ctx.ws && k.we == ctx.we {
            return k;
        }
    }
    let k: &'static K5 = Box::leak(Box::new(build(ctx)));
    v.push((id, k));
    k
}

/// Drops the cached parameters (tests build many contexts).
#[allow(dead_code)]
pub fn forget() {
    PARAMS.lock().unwrap().clear();
}

// ---------------------------------------------------------------- the kernel

#[cfg(target_arch = "x86_64")]
mod imp {
    use super::*;
    use std::arch::x86_64::*;

    #[inline(always)]
    unsafe fn rd64(p: *const u8) -> u64 {
        u64::from_le(std::ptr::read_unaligned(p as *const u64))
    }

    #[inline(always)]
    unsafe fn rd32(p: *const u8) -> u32 {
        u32::from_le(std::ptr::read_unaligned(p as *const u32))
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

    /// `fastev::window` in the shorter form (k4's `window_r`): with qr = q & r and
    /// qs = qr & fs the quoted starts, the quotes in r must be exactly qs and
    /// the closing quotes cl, so bad = (qr & !fs) ^ cl; quoted starts are
    /// never adjacent, nor are closing quotes: cs = fs ^ (qs * 3), ce = d ^
    /// (cl * 3). Same bad == 0 and the same starts / ends inside r.
    #[inline(always)]
    fn window_r(d: u64, q: u64, r: u64) -> (u64, u64, u64) {
        let fs = (d << 1) | 1;
        let qr = q & r;
        let qs = qr & fs;
        let cl = ((!d).wrapping_add(qs) & d) >> 1;
        let bad = (qr & !fs) ^ cl;
        let cs = fs ^ qs.wrapping_mul(3);
        let ce = d ^ cl.wrapping_mul(3);
        (bad, cs, ce)
    }

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

    /// (comma, newline, quote) masks of the 64 bytes at p.
    #[inline(always)]
    unsafe fn masks64(p: *const u8) -> (u64, u64, u64) {
        let a = _mm256_loadu_si256(p as *const __m256i);
        let b = _mm256_loadu_si256(p.add(32) as *const __m256i);
        let m = |v: __m256i, ch: u8| -> u64 { _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, _mm256_set1_epi8(ch as i8))) as u32 as u64 };
        (m(a, b',') | (m(b, b',') << 32), m(a, b'\n') | (m(b, b'\n') << 32), m(a, b'"') | (m(b, b'"') << 32))
    }

    /// `HH:MM:SS?HH:MM` at q + 11: (seconds of day, offset minutes, lane
    /// mask, range mask). Lane mask bit i (0..16) is set iff byte 11 + i is
    /// a digit where `HH:MM:SS?HH:MM` has a digit, or equals the `:` / `Z` /
    /// `:` at 2, 5, 8, 11; range mask bits 2w, 2w+1 are set iff word w of
    /// [hh, mm, ss, 0, oh, om, 0, 0] exceeds [23, 59, 59, 0, 23, 59, 0, 0].
    /// (k4's `tvec`, one lane.) Reads q..q+27.
    #[inline(always)]
    unsafe fn tvec1(q: *const u8) -> (u32, u32, u32, u32) {
        let v = _mm_loadu_si128(q.add(11) as *const __m128i);
        let d = _mm_sub_epi8(v, _mm_set1_epi8(b'0' as i8));
        let isd = _mm_cmpeq_epi8(_mm_min_epu8(d, _mm_set1_epi8(9)), d);
        const C: i8 = b':' as i8;
        const Z: i8 = b'Z' as i8;
        const E: i8 = -1;
        let iseq = _mm_cmpeq_epi8(v, _mm_setr_epi8(0, 0, C, 0, 0, C, 0, 0, Z, 0, 0, C, 0, 0, 0, 0));
        let okm = _mm_movemask_epi8(_mm_blendv_epi8(isd, iseq, _mm_setr_epi8(0, 0, E, 0, 0, E, 0, 0, E, 0, 0, E, 0, 0, 0, 0))) as u32;
        let dg = _mm_shuffle_epi8(d, _mm_setr_epi8(0, 1, 3, 4, 6, 7, -1, -1, 9, 10, 12, 13, -1, -1, -1, -1));
        let w = _mm_maddubs_epi16(dg, _mm_set1_epi16(0x010A)); // [hh, mm, ss, 0, oh, om, 0, 0]
        let rm = _mm_movemask_epi8(_mm_cmpgt_epi16(w, _mm_setr_epi16(23, 59, 59, 0, 23, 59, 0, 0))) as u32;
        let s = _mm_madd_epi16(w, _mm_setr_epi16(3600, 60, 1, 0, 60, 1, 0, 0));
        let t = _mm_hadd_epi32(s, s); // [sec, omin, sec, omin]
        let r = _mm_cvtsi128_si64(t) as u64;
        (r as u32, (r >> 32) as u32, okm, rm)
    }

    /// 13 digits at q: (value, nonzero unless all 13 bytes are digits).
    /// (k4's `f3_13`.) Reads q-3..q+13: the timestamp is the fourth field,
    /// so it starts at byte 3 or later of the line.
    #[inline(always)]
    unsafe fn f3_13(q: *const u8) -> (u64, u32) {
        // lanes 3..16 = q[0..13] (lanes 0..3 = q[-3..0], masked out below)
        let v = _mm_loadu_si128(q.sub(3) as *const __m128i);
        let d = _mm_sub_epi8(v, _mm_set1_epi8(b'0' as i8));
        let ok = _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_min_epu8(d, _mm_set1_epi8(9)), d)) as u32;
        let dz = _mm_and_si128(d, _mm_setr_epi8(0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
        let w = _mm_maddubs_epi16(dz, _mm_set1_epi16(0x010A));
        let g4 = _mm_madd_epi16(w, _mm_set1_epi32(0x0001_0064));
        let p = _mm_packus_epi32(g4, g4);
        let g8 = _mm_madd_epi16(p, _mm_set1_epi32(0x0001_2710));
        let r = _mm_cvtsi128_si64(g8) as u64;
        ((r & 0xFFFF_FFFF) * 100_000_000 + (r >> 32), !ok & 0xFFF8)
    }

    /// The timestamp field at q (n bytes, 32 readable): `ctx.tsp.parse` of it.
    #[inline(always)]
    unsafe fn ts(k: &K5, ctx: &Ctx, q: *const u8, n: usize) -> Option<i64> {
        if n == 13 {
            let (v, bad) = f3_13(q);
            if bad == 0 {
                return Some(v as i64);
            }
        } else if (n == 20) | (n == 25) {
            let a = rd64(q);
            let d4 = rd32(q.add(8));
            // dslot(d4) in one instruction: bits 0, 1 and 8..12 in order
            let e = k.tab.0.get_unchecked(_pext_u32(d4, 0x0F03) as usize);
            let dm = (a ^ e.pat) | ((d4 & 0xFF_FFFF) ^ e.dd) as u64;
            let (sec, omin, okm, rm) = tvec1(q);
            let b19 = *q.add(19);
            if n == 20 {
                if (dm == 0) & ((okm & 0x1FF) == 0x1FF) & ((rm & 0x3F) == 0) {
                    return Some(e.ms + sec as i64 * 1000);
                }
            } else if (dm == 0) & ((okm & 0x3EFF) == 0x3EFF) & ((rm & 0xF3F) == 0) & ((b19 == b'+') | (b19 == b'-')) {
                let base = e.ms + sec as i64 * 1000;
                let off = omin as i64 * 60_000;
                return Some(if b19 == b'+' { base - off } else { base + off });
            }
        }
        ts_slow(ctx, q, n)
    }

    #[cold]
    #[inline(never)]
    fn ts_slow(ctx: &Ctx, q: *const u8, n: usize) -> Option<i64> {
        if COUNT.load(Ordering::Relaxed) {
            N_TSSLOW.fetch_add(1, Ordering::Relaxed);
        }
        ctx.tsp.parse(unsafe { std::slice::from_raw_parts(q, n) })
    }

    /// k4's `pack2`: `keys::pack_wide` of two strings at once, one per
    /// 128-bit lane (16 readable bytes from each start): (key a, a packs,
    /// key b, b packs).
    #[inline(always)]
    pub(crate) unsafe fn pack2(a: *const u8, na: usize, b: *const u8, nb: usize) -> (u64, bool, u64, bool) {
        let va = _mm_loadu_si128(a as *const __m128i);
        let vb = _mm_loadu_si128(b as *const __m128i);
        let v = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(va), vb);
        let idx = _mm256_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let nv = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(_mm_set1_epi8(na as i8)), _mm_set1_epi8(nb as i8));
        let inlen = _mm256_cmpgt_epi8(nv, idx);
        let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(v), _mm256_set1_epi8(0x0F));
        let lo = _mm256_and_si256(v, _mm256_set1_epi8(0x0F));
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
        // per lane, dwords [l0, l1, l2, l3]; the key of pack_wide is
        // l0 << 40 | l1 << 16 | ((l2 >> 12) << 4) | n: key bytes 0..8 are
        // l2 bytes 1, 2, l1 bytes 0..3, l0 bytes 0..3 (one byte shuffle), key
        // bits 0..4 then hold l2 bits 8..12, replaced by n (k4's pack2)
        let kv = _mm256_shuffle_epi8(u, _mm256_setr_epi8(9, 10, 4, 5, 6, 0, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1, 9, 10, 4, 5, 6, 0, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1));
        let ka = (_mm_cvtsi128_si64(_mm256_castsi256_si128(kv)) as u64 & !0xF) | na as u64;
        let kb = (_mm_cvtsi128_si64(_mm256_extracti128_si256::<1>(kv)) as u64 & !0xF) | nb as u64;
        let inv = _mm256_movemask_epi8(invalid) as u32;
        let oka = ((inv & 0xFFFF) == 0) & (na.wrapping_sub(1) < 10);
        let okb = ((inv >> 16) == 0) & (nb.wrapping_sub(1) < 10);
        (ka, oka, kb, okb)
    }

    /// `ctx.campaigns.get` of the field at q (n bytes, 16 readable).
    #[inline(always)]
    unsafe fn campaign(k: &K5, ctx: &Ctx, q: *const u8, n: usize) -> u16 {
        if n > 16 {
            return ctx.campaigns.get(std::slice::from_raw_parts(q, n));
        }
        let w0 = _bzhi_u64(rd64(q), (n * 8) as u32);
        let w1 = if n > 8 { _bzhi_u64(rd64(q.add(8)), (n * 8 - 64) as u32) } else { 0 };
        let t = &k.camp;
        let mut i = (chash(w0, w1, n) >> t.shift) as usize;
        loop {
            let e = t.slots.get_unchecked(i);
            if (e.w0 == w0) & (e.w1 == w1) & (e.len as usize == n) {
                return e.v as u16;
            }
            if e.len == u32::MAX {
                return NONE;
            }
            i = (i + 1) & t.mask;
        }
    }

    /// The link table through its view: `LinkTable::person_hash` with the
    /// 4-slot group compared in two ymm registers (same probe sequence,
    /// same rule: the first matching key of the first group holding a match
    /// or an empty slot). Usable by the event path as well (call from an
    /// AVX2 `#[target_feature]` function so that it inlines).
    pub(crate) struct Probe<'a> {
        lt: &'a LinkTable,
        slots: *const u64,
        mask: usize,
        amask: usize,
        mixed: bool,
        own: bool,
    }

    impl<'a> Probe<'a> {
        pub(crate) fn new(lt: &'a LinkTable) -> Probe<'a> {
            let (slots, mask, amask, mixed) = lt.view();
            Probe { lt, slots, mask, amask, mixed, own: mask != 0 && amask == !3 }
        }
        #[inline(always)]
        pub(crate) unsafe fn prefetch(&self, h: u64) {
            if self.own {
                crate::prefetch(self.slots.wrapping_add(2 * (h as usize & self.mask & self.amask)) as *const u8);
            } else {
                self.lt.prefetch_h(h);
            }
        }
        #[inline(always)]
        pub(crate) unsafe fn person_hash(&self, k: u64, h: u64) -> u64 {
            if !self.own {
                return self.lt.person_hash(k, h);
            }
            if !crate::links::PROBE_SEL.load(Ordering::Relaxed) {
                return match probe_view(self.slots, self.mask, k, h) {
                    Some(v) => {
                        if self.mixed {
                            v
                        } else {
                            keys::mix(v)
                        }
                    }
                    None => h,
                };
            }
            // found / not found selected, not branched on (it is random per
            // line, and the branch sits right behind the table's cache miss)
            let (found, v) = probe_sel(self.slots, self.mask, k, h);
            let r = if self.mixed { v } else { keys::mix(v) };
            crate::links::sel64(found, r, h)
        }
    }

    /// `probe_view` with the result selected instead of branched on:
    /// (k is in the table, its raw slot value if so, else some in-bounds
    /// slot's value). The group loop ends at the first group holding a
    /// match or an empty slot, as `probe_view`'s. Inlines into AVX2 code only.
    #[inline(always)]
    pub(crate) unsafe fn probe_sel(slots: *const u64, mask: usize, k: u64, h: u64) -> (bool, u64) {
        let kk = _mm256_set1_epi64x(k as i64);
        let z = _mm256_setzero_si256();
        let mut i = h as usize & mask & !3;
        loop {
            let g = slots.add(2 * i) as *const __m256i;
            let a = _mm256_loadu_si256(g);
            let b = _mm256_loadu_si256(g.add(1));
            let m = (_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(a, kk))) | (_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(b, kk))) << 4)) as u32 & 0x55;
            let e = (_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(a, z))) | (_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(b, z))) << 4)) as u32 & 0x55;
            if (m | e) != 0 {
                // the first matching slot's value (slot 0's if none matches:
                // m's bits are at 0, 2, 4, 6, so bit 8 gives index 8 & 7 = 0)
                let v = *slots.add(2 * i + ((m | 0x100).trailing_zeros() & 7) as usize + 1);
                return (m != 0, v);
            }
            i = (i + 4) & mask;
        }
    }

    /// `LinkTable::probe` over `LinkTable::view()` (`slots`, `mask`; only
    /// for amask == !3 and mask != 0): the raw finalized slot value of key
    /// `k` (hash `h`), if `k` is in the table. Same probe sequence (4-slot
    /// groups from h & mask & !3, wrapping), same rule: in the first group
    /// holding a match or an empty slot, the first matching slot's value,
    /// or None if the group holds no match. Each group is compared in two
    /// ymm registers (keys at even u64 lanes). Inlines into AVX2 code only.
    #[inline(always)]
    pub(crate) unsafe fn probe_view(slots: *const u64, mask: usize, k: u64, h: u64) -> Option<u64> {
        let kk = _mm256_set1_epi64x(k as i64);
        let z = _mm256_setzero_si256();
        let mut i = h as usize & mask & !3;
        loop {
            let g = slots.add(2 * i) as *const __m256i;
            let a = _mm256_loadu_si256(g);
            let b = _mm256_loadu_si256(g.add(1));
            let m = (_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(a, kk))) | (_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(b, kk))) << 4)) as u32 & 0x55;
            let e = (_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(a, z))) | (_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(b, z))) << 4)) as u32 & 0x55;
            if m != 0 {
                return Some(*slots.add(2 * i + m.trailing_zeros() as usize + 1));
            }
            if e != 0 {
                return None;
            }
            i = (i + 4) & mask;
        }
    }

    /// A valid click before its person lookup.
    #[derive(Clone, Copy, Default)]
    struct Ent {
        user: u64,
        h: u64,
        ck: u64,
        ts: u32,
        camp: u32,
    }

    /// The FIFO of valid clicks awaiting their person lookup.
    struct Ring {
        e: [Ent; RING],
        n: usize,
    }

    impl Ring {
        #[inline(always)]
        unsafe fn emit<C: CSink>(lt: &Probe, ps_shift: u32, c: &mut C, e: Ent) {
            let person = lt.person_hash(e.user, e.h);
            c.push_click((person >> ps_shift) as usize, CRec { person, ck: e.ck, ts: e.ts, camp: e.camp, _pad: 0 });
        }
        #[inline(always)]
        unsafe fn push<C: CSink>(&mut self, lt: &Probe, ps_shift: u32, c: &mut C, e: Ent) {
            lt.prefetch(e.h);
            let i = self.n % RING;
            if self.n >= RING {
                Self::emit(lt, ps_shift, c, self.e[i]);
            }
            self.e[i] = e;
            self.n += 1;
        }
        #[inline(always)]
        unsafe fn finish<C: CSink>(&mut self, lt: &Probe, ps_shift: u32, c: &mut C) {
            for k in self.n.saturating_sub(RING)..self.n {
                Self::emit(lt, ps_shift, c, self.e[k % RING]);
            }
            self.n = 0;
        }
    }

    /// Per-task constants.
    struct Env<'a> {
        ctx: &'a Ctx,
        lt: Probe<'a>,
        k: &'static K5,
        file: u32,
        base: *const u8,
        cws: i64,
        we: i64,
        ps_shift: u32,
    }

    /// The old per-line logic (`main::click_task`'s closure) on the fields
    /// of a line the kernel did not cut.
    #[inline(always)]
    fn slow_fields<C: CSink>(env: &Env, ring: &mut Ring, c: &mut C, f: &[&[u8]; 4]) {
        if f[0].is_empty() || f[1].is_empty() {
            return;
        }
        let ctx = env.ctx;
        let Some(t) = ctx.tsp.parse(f[3]) else { return };
        if t < env.cws || t >= env.we {
            return;
        }
        let camp = ctx.campaigns.get(f[2]);
        if camp == NONE {
            return;
        }
        let user = keys::key_of(f[1]);
        let h = keys::mix(user);
        let ck = keys::click_key(env.file, env.base, f[0], false);
        unsafe { ring.push(&env.lt, env.ps_shift, c, Ent { user, h, ck, ts: (t - env.cws) as u32, camp: camp as u32 }) };
    }

    /// Lines b[from..to) through the general scanner and `slow_fields`.
    #[cold]
    #[inline(never)]
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
    unsafe fn slow_range<C: CSink>(env: &Env, ring: &mut Ring, c: &mut C, b: &[u8], from: usize, to: usize) {
        if COUNT.load(Ordering::Relaxed) {
            N_SLOW.fetch_add(1, Ordering::Relaxed);
        }
        scan::for_each_record_x::<4, _, true>(b, from, to, |f| slow_fields(env, ring, c, f));
    }

    /// One cut line: field j is l[j] bytes at p + s[j] (unquoted).
    #[inline(always)]
    unsafe fn line<C: CSink>(env: &Env, ring: &mut Ring, c: &mut C, p: *const u8, s: [usize; 4], l: [usize; 4]) {
        if (l[0] == 0) | (l[1] == 0) {
            return;
        }
        let ctx = env.ctx;
        let Some(t) = ts(env.k, ctx, p.add(s[3]), l[3]) else { return };
        if (t < env.cws) | (t >= env.we) {
            return;
        }
        let camp = campaign(env.k, ctx, p.add(s[2]), l[2]);
        if camp == NONE {
            return;
        }
        let (kc, okc, ku, oku) = pack2(p.add(s[0]), l[0], p.add(s[1]), l[1]);
        let user = if oku { ku } else { user_slow(std::slice::from_raw_parts(p.add(s[1]), l[1])) };
        let ck = if okc { kc } else { keys::click_key(env.file, env.base, std::slice::from_raw_parts(p.add(s[0]), l[0]), true) };
        let h = keys::mix(user);
        ring.push(&env.lt, env.ps_shift, c, Ent { user, h, ck, ts: (t - env.cws) as u32, camp: camp as u32 });
    }

    /// The key of a user ID that does not pack (long or interned; logs).
    #[inline(never)]
    fn user_slow(f: &[u8]) -> u64 {
        keys::key_of(f)
    }

    /// `main::click_task` (PCache off) with the kernel: the same records,
    /// in the same order.
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
    pub unsafe fn task<C: CSink>(b: &[u8], task: Task, ctx: &Ctx, lt: &LinkTable, c: &mut C) {
        assert!(task.start <= task.end && task.end <= b.len());
        let env = Env {
            ctx,
            lt: Probe::new(lt),
            k: params(ctx),
            file: task.file,
            base: b.as_ptr(),
            cws: ctx.ws - WEEK,
            we: ctx.we,
            ps_shift: 64 - ctx.ps_bits,
        };
        let mut ring = Ring { e: [Ent::default(); RING], n: 0 };
        let count = COUNT.load(Ordering::Relaxed);
        let base = b.as_ptr();
        let lim = b.len().saturating_sub(MARGIN);
        let pf = fastev::PF_DIST.load(Ordering::Relaxed);
        let mut off = task.start;
        if b.len() >= MARGIN {
            while off < task.end && off <= lim {
                let p = base.add(off);
                if pf != 0 {
                    // a prefetch never faults: no clamp (wrapping address, never dereferenced)
                    crate::prefetch(p.wrapping_add(pf));
                }
                let (c0, n0, q0) = masks64(p);
                let d = c0 | n0;
                let d3 = blsr(blsr(blsr(d)));
                if d3 != 0 {
                    // the line fits the 64-byte window
                    let r = upto(d3);
                    let (bad, cs, ce) = window_r(d, q0, r);
                    let last = d3 & d3.wrapping_neg();
                    let nl = (n0 ^ last) & r;
                    if (bad | nl) == 0 {
                        let (cs1, ce1) = (blsr(cs), blsr(ce));
                        let (cs2, ce2) = (blsr(cs1), blsr(ce1));
                        let (cs3, ce3) = (blsr(cs2), blsr(ce2));
                        let s = [tz(cs), tz(cs1), tz(cs2), tz(cs3)];
                        let e = [tz(ce), tz(ce1), tz(ce2), tz(ce3)];
                        let l = [e[0].wrapping_sub(s[0]), e[1].wrapping_sub(s[1]), e[2].wrapping_sub(s[2]), e[3].wrapping_sub(s[3])];
                        if count {
                            N_FAST.fetch_add(1, Ordering::Relaxed);
                        }
                        line(&env, &mut ring, c, p, s, l);
                        off += tz(d3) + 1;
                        continue;
                    }
                } else {
                    // a line over 64 bytes: the 128-byte window
                    let (c1, n1, q1) = masks64(p.add(64));
                    let d = d as u128 | ((c1 | n1) as u128) << 64;
                    let n = n0 as u128 | (n1 as u128) << 64;
                    let q = q0 as u128 | (q1 as u128) << 64;
                    let d3 = blsr128(blsr128(blsr128(d)));
                    let r = upto128(d3);
                    let (bad, cs, ce) = window128(d, q, r);
                    let last = d3 & d3.wrapping_neg();
                    let nl = (n ^ last) & r;
                    if (d3 != 0) & ((bad | nl) == 0) {
                        let (cs1, ce1) = (blsr128(cs), blsr128(ce));
                        let (cs2, ce2) = (blsr128(cs1), blsr128(ce1));
                        let (cs3, ce3) = (blsr128(cs2), blsr128(ce2));
                        let s = [tz128(cs), tz128(cs1), tz128(cs2), tz128(cs3)];
                        let e = [tz128(ce), tz128(ce1), tz128(ce2), tz128(ce3)];
                        let l = [e[0].wrapping_sub(s[0]), e[1].wrapping_sub(s[1]), e[2].wrapping_sub(s[2]), e[3].wrapping_sub(s[3])];
                        if count {
                            N_WIDE.fetch_add(1, Ordering::Relaxed);
                        }
                        line(&env, &mut ring, c, p, s, l);
                        off += tz128(d3) + 1;
                        continue;
                    }
                }
                // not cut: this line through the scanner and the old logic
                let e = scan::next_line_start(b, off);
                slow_range(&env, &mut ring, c, b, off, e);
                off = e;
            }
        }
        if off < task.end {
            slow_range(&env, &mut ring, c, b, off, task.end);
        }
        ring.finish(&env.lt, env.ps_shift, c);
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
pub(crate) use imp::{probe_sel, probe_view, Probe};

/// Whether `main::click_task_avx2` should take k5 for this task.
#[inline(always)]
pub fn wanted(cache: &PCache) -> bool {
    ON.load(Ordering::Relaxed) && !cache.on
}

/// k5 on one click task (`main::click_task` with PCache off: the same
/// records in the same order). Call only with AVX2 + BMI1/2 enabled.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn task<C: CSink>(b: &[u8], task: Task, ctx: &Ctx, lt: &LinkTable, c: &mut C) {
    imp::task(b, task, ctx, lt, c)
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
pub(crate) unsafe fn pack2(a: *const u8, na: usize, b: *const u8, nb: usize) -> (u64, bool, u64, bool) {
    #[target_feature(enable = "avx2")]
    unsafe fn f(a: *const u8, na: usize, b: *const u8, nb: usize) -> (u64, bool, u64, bool) {
        imp::pack2(a, na, b, nb)
    }
    f(a, na, b, nb)
}

// ---------------------------------------------------------------- checking

/// Records every click handed to the sink.
#[derive(Default)]
pub struct RecC(pub Vec<(usize, u64, u64, u32, u32)>);
impl CSink for RecC {
    #[inline(always)]
    fn push_click(&mut self, q: usize, r: CRec) {
        self.0.push((q, r.person, r.ck, r.ts, r.camp));
    }
}

/// Records with every side-table click reference (keys::click_key for a
/// click_id of 1024+ bytes: an index into a global table that grows with
/// every call) replaced by one placeholder: running both paths in one
/// process gives the same click_id two indices. Both paths make such
/// references with the same `keys::click_key` call (k5 never cuts a line
/// that long), so only the placeholder is compared.
pub fn norm(mut v: Vec<(usize, u64, u64, u32, u32)>) -> Vec<(usize, u64, u64, u32, u32)> {
    for r in v.iter_mut() {
        if r.2 & 0xF == 0 && r.2 >> 54 == 1023 {
            r.2 = 1023 << 54;
        }
    }
    v
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[inline(never)]
unsafe fn old_avx2<C: CSink>(b: &[u8], task: Task, ctx: &Ctx, lt: &LinkTable, c: &mut C, safe_end: usize) {
    let mut cache = PCache { slots: vec![(0, 0); 2], shift: 63, on: false };
    click_task::<true, C>(b, task, ctx, lt, c, safe_end, &mut cache)
}

/// The old path on a task (oracle; PCache off).
#[cfg(target_arch = "x86_64")]
pub fn old_task<C: CSink>(b: &[u8], task: Task, ctx: &Ctx, lt: &LinkTable, c: &mut C, safe_end: usize) {
    unsafe { old_avx2(b, task, ctx, lt, c, safe_end) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[inline(never)]
pub(crate) unsafe fn task_k5<C: CSink>(b: &[u8], task: Task, ctx: &Ctx, lt: &LinkTable, c: &mut C) {
    imp::task(b, task, ctx, lt, c)
}

/// SS_K5CHECK=1: every click task through the old path and through k5;
/// the records must be identical, in the same order. Panics on the first
/// difference with the line that differs on its own.
#[cfg(target_arch = "x86_64")]
pub fn check(links_in: &Input, cl: &Input, ctx: &Ctx) {
    assert!(ctx.avx2, "k5check needs AVX2");
    COUNT.store(true, Ordering::Relaxed);
    let lb = LocBits::new(ctx.threads, ctx.ps_bits).unwrap();
    let empty = Input { maps: Vec::new(), tasks: Vec::new(), bytes: 0 };
    let (lt, _) = crate::parse_b(links_in, &empty, &empty, ctx, lb, None);
    let (mut lines, mut recs) = (0usize, 0usize);
    let next = AtomicUsize::new(0);
    let t0 = Instant::now();
    let res = par(ctx.threads, |_| {
        let (mut l, mut r) = (0usize, 0usize);
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            let Some(&task) = cl.tasks.get(i) else { break };
            let b = cl.maps[task.file as usize].bytes();
            let safe_end = b.as_ptr() as usize + b.len();
            let mut want = RecC::default();
            old_task(b, task, ctx, &lt, &mut want, safe_end);
            let mut got = RecC::default();
            unsafe { task_k5(b, task, ctx, &lt, &mut got) };
            l += b[task.start..task.end].iter().filter(|&&c| c == b'\n').count();
            r += want.0.len();
            let (want, got) = (RecC(norm(want.0)), RecC(norm(got.0)));
            if got.0 != want.0 {
                let culprit = culprits(b, task, ctx, &lt);
                let k = got.0.iter().zip(want.0.iter()).position(|(a, b)| a != b).unwrap_or(got.0.len().min(want.0.len()));
                eprintln!(
                    "k5check: MISMATCH file {} [{}, {}): {} k5 vs {} old records, first difference at record {}: k5 {:?} old {:?}\n{}",
                    task.file,
                    task.start,
                    task.end,
                    got.0.len(),
                    want.0.len(),
                    k,
                    got.0.get(k),
                    want.0.get(k),
                    culprit
                );
                std::process::exit(1);
            }
        }
        (l, r)
    });
    for (l, r) in res {
        lines += l;
        recs += r;
    }
    let (f, w, s, t) = (N_FAST.load(Ordering::Relaxed), N_WIDE.load(Ordering::Relaxed), N_SLOW.load(Ordering::Relaxed), N_TSSLOW.load(Ordering::Relaxed));
    eprintln!(
        "k5check: OK {} tasks, {} lines, {} click records identical ({:.0} ms); k5 lines: 64-byte {} ({:.2}%), 128-byte {} ({:.2}%), scanner ranges {} ({:.3}%), ts fallbacks {} ({:.3}%)",
        cl.tasks.len(),
        lines,
        recs,
        t0.elapsed().as_secs_f64() * 1e3,
        f,
        f as f64 * 100.0 / lines as f64,
        w,
        w as f64 * 100.0 / lines as f64,
        s,
        s as f64 * 100.0 / lines as f64,
        t,
        t as f64 * 100.0 / lines as f64
    );
}

#[cfg(not(target_arch = "x86_64"))]
pub fn check(_links_in: &Input, _cl: &Input, _ctx: &Ctx) {
    eprintln!("k5check: k5 is x86-64 only");
}

/// The lines of a task whose records differ on their own (at most 5).
#[cfg(target_arch = "x86_64")]
pub fn culprits(b: &[u8], t: Task, ctx: &Ctx, lt: &LinkTable) -> String {
    let safe_end = b.as_ptr() as usize + b.len();
    let mut s = String::new();
    let mut l = t.start;
    let mut shown = 0;
    while l < t.end && shown < 5 {
        let e = scan::next_line_start(b, l).min(t.end);
        let lt1 = Task { file: t.file, start: l, end: e };
        let mut a = RecC::default();
        old_task(b, lt1, ctx, lt, &mut a, safe_end);
        let mut k = RecC::default();
        unsafe { task_k5(b, lt1, ctx, lt, &mut k) };
        let (a, k) = (RecC(norm(a.0)), RecC(norm(k.0)));
        if a.0 != k.0 {
            s += &format!("  line at {} ({} bytes before the buffer end): \"{}\"\n    old: {:?}\n    k5:  {:?}\n", l, b.len() - l, b[l..e].escape_ascii(), a.0, k.0);
            shown += 1;
        }
        l = e;
    }
    if shown == 0 {
        s += "  (no single line differs on its own)\n";
    }
    s
}

// ---------------------------------------------------------------- stage bench

struct NullC(u64);

impl CSink for NullC {
    #[inline(always)]
    fn push_click(&mut self, q: usize, r: CRec) {
        self.0 = self.0.wrapping_add(r.person ^ r.ck ^ r.ts as u64 ^ q as u64 ^ r.camp as u64);
    }
}

/// Cumulative stages of the old click path on the real splitter, with the
/// real early exits (a stage stops a line where the old path would);
/// stages 7, 8: k5 with a null sink, k5 into the real sink.
#[inline(always)]
fn bstage<const AVX2: bool>(stage: u32, b: &[u8], task: Task, ctx: &Ctx, lt: &LinkTable, c: &mut Parts<CRec>, cache: &mut PCache) -> u64 {
    let safe_end = b.as_ptr() as usize + b.len();
    let mut acc = 0u64;
    let we = ctx.we;
    let cws = ctx.ws - WEEK;
    match stage {
        0 => fastev::for_each_click::<AVX2, _>(b, task.start, task.end, safe_end, &mut |f: &[&[u8]; 4]| acc = acc.wrapping_add(f[3].len() as u64 ^ f[0].len() as u64)),
        1..=4 => fastev::for_each_click::<AVX2, _>(b, task.start, task.end, safe_end, &mut |f: &[&[u8]; 4]| {
            if f[0].is_empty() || f[1].is_empty() {
                return;
            }
            let wide = f[3].as_ptr() as usize + 32 <= safe_end;
            let Some(t) = ctx.tsp.parse(f[3]) else { return };
            if t < cws || t >= we {
                return;
            }
            acc = acc.wrapping_add(t as u64);
            if stage >= 2 {
                let camp = if wide { unsafe { ctx.campaigns.get_wide(f[2]) } } else { ctx.campaigns.get(f[2]) };
                if camp == NONE {
                    return;
                }
                acc = acc.wrapping_add(camp as u64);
            }
            if stage >= 3 {
                let user = if wide { unsafe { keys::key_of_wide(f[1]) } } else { keys::key_of(f[1]) };
                acc = acc.wrapping_add(keys::mix(user));
            }
            if stage >= 4 {
                acc = acc.wrapping_add(keys::click_key(task.file, b.as_ptr(), f[0], wide));
            }
        }),
        5 => {
            let mut n = NullC(0);
            click_task::<AVX2, _>(b, task, ctx, lt, &mut n, safe_end, cache);
            acc = n.0;
        }
        6 => click_task::<AVX2, _>(b, task, ctx, lt, c, safe_end, cache),
        #[cfg(target_arch = "x86_64")]
        7 => {
            let mut n = NullC(0);
            unsafe { imp::task(b, task, ctx, lt, &mut n) };
            acc = n.0;
        }
        #[cfg(target_arch = "x86_64")]
        8 => unsafe { imp::task(b, task, ctx, lt, c) },
        _ => {}
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
unsafe fn bstage_avx2(stage: u32, b: &[u8], task: Task, ctx: &Ctx, lt: &LinkTable, c: &mut Parts<CRec>, cache: &mut PCache) -> u64 {
    bstage::<true>(stage, b, task, ctx, lt, c, cache)
}

/// SS_K5BENCH=1: thread time per click line of cumulative stages of the old
/// path (SS_K5STAGES, default 0..=8; 7, 8 = k5) at ctx.threads threads;
/// SS_K5RUNS runs each (default 5), SS_K5TASKS limits the tasks.
pub fn bench(cl: &Input, links_in: &Input, ctx: &Ctx) {
    let stages: Vec<u32> = std::env::var("SS_K5STAGES").ok().map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect()).unwrap_or((0..9).collect());
    let nruns = super::env_usize("SS_K5RUNS").unwrap_or(5);
    let ntasks = super::env_usize("SS_K5TASKS").unwrap_or(usize::MAX).min(cl.tasks.len());
    let lines: usize = cl.tasks[..ntasks].iter().map(|t| cl.maps[t.file as usize].bytes()[t.start..t.end].iter().filter(|&&c| c == b'\n').count()).sum();
    let lb = LocBits::new(ctx.threads, ctx.ps_bits).unwrap();
    let empty = Input { maps: Vec::new(), tasks: Vec::new(), bytes: 0 };
    let (lt, _) = crate::parse_b(links_in, &empty, &empty, ctx, lb, None);
    let keep = sys::arenas().len();
    let ps = 1usize << ctx.ps_bits;
    let est = cl.bytes / 44 + 1;
    let names = ["split", "+ts+window", "+campaign", "+user key+mix", "+click key", "+person (null sink)", "+sink (full task)", "k5 (null sink)", "k5 full"];
    for &stage in &stages {
        let nt = ctx.threads;
        let mut runs = Vec::new();
        for _ in 0..nruns {
            let next = AtomicUsize::new(0);
            PHASE_CPU_NS.store(0, Ordering::Relaxed);
            let t = Instant::now();
            let accs = par(nt, |_| {
                let mut c: Parts<CRec> = Parts::with_wc(ps, est / (nt * ps) * 5 / 4 + 32, true);
                let mut cache = PCache::new();
                let mut a = 0u64;
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&task) = cl.tasks[..ntasks].get(i) else { break };
                    let b = cl.maps[task.file as usize].bytes();
                    a = a.wrapping_add(run_stage(stage, b, task, ctx, &lt, &mut c, &mut cache));
                }
                c.finish();
                a ^ c.total() as u64
            });
            let wall = t.elapsed().as_secs_f64() * 1e3;
            let cpu = PHASE_CPU_NS.swap(0, Ordering::Relaxed) as f64 / 1e6;
            std::hint::black_box(accs);
            sys::release_from(keep);
            runs.push((wall, cpu * 1e6 / lines as f64));
        }
        let best = runs.iter().cloned().fold((f64::MAX, 0.0), |a, r| if r.0 < a.0 { r } else { a });
        let all: Vec<String> = runs.iter().map(|r| format!("{:.1}", r.1)).collect();
        let nm = names.get(stage as usize).copied().unwrap_or("none");
        eprintln!("k5bench {:22} threads={} best wall={:6.1}ms {:6.1}ns/line  (all ns/line: {})", nm, nt, best.0, best.1, all.join(" "));
    }
    eprintln!("k5bench: {} click lines, {} MB", lines, cl.bytes >> 20);
}

fn run_stage(stage: u32, b: &[u8], task: Task, ctx: &Ctx, lt: &LinkTable, c: &mut Parts<CRec>, cache: &mut PCache) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        if ctx.avx2 {
            return unsafe { bstage_avx2(stage, b, task, ctx, lt, c, cache) };
        }
    }
    if stage >= 7 {
        return 0;
    }
    bstage::<false>(stage, b, task, ctx, lt, c, cache)
}
