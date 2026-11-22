//! IDs as 64-bit keys.
//!
//! A non-empty string of at most 10 bytes drawn from the 64-symbol alphabet
//! `-0-9A-Z_a-z` is packed exactly: 6-bit codes (assigned in ASCII order),
//! left-aligned in the top 60 bits, with the length in the low 4 bits. For two
//! packed keys, numeric order equals byte-wise string order and equality equals
//! string equality.
//!
//! A "long" ID, exactly one byte of that alphabet followed by exactly 32
//! lowercase hexadecimal digits (`[0-9a-f]`; one ID in eight in the current
//! data), gets `(h << 4) | 0xE` with h a 60-bit keyed hash of (first byte,
//! 128-bit value), computed without any shared state. Every thread logs
//! (key, first byte, value) of each long ID it keys; `verify_longs` (after the
//! links are parsed, and after all parsing) checks that no key stands for two
//! different IDs and builds the table that decodes long keys. If two IDs ever
//! share a key, `collision()` is set and the caller reruns the whole job with
//! another seed (`reset_longs`); after `LONG_TRIES` seeds long IDs go to the
//! interner. So long keys are equal iff the strings are equal, for any input.
//! Their order: long vs long by (first byte, value), which is the byte-wise
//! order of the strings since all have the same width and hex digits sort in
//! value order; anything else by comparing the strings.
//!
//! Every other string is interned in a global side table and gets the key
//! `(index << 4) | 0xF`. Such keys are equal iff the strings are equal; their
//! order is resolved by comparing the actual bytes (rare, slow path).
//!
//! The low 4 bits tell the kinds apart for every bit pattern: 1..=10 packed
//! (the length), 0xE long, 0xF interned; 0 and 11..=13 are never produced.
//! Key 0 is never produced and serves as "empty".

use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering as AO};
use std::sync::Mutex;

use crate::parts::Parts;

const ALPHABET: &[u8; 64] = b"-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz";
const INVALID: u8 = 0x80;

static CODE: [u8; 256] = {
    let mut t = [INVALID; 256];
    let mut i = 0;
    while i < 64 {
        t[ALPHABET[i] as usize] = i as u8;
        i += 1;
    }
    t
};

pub const FALLBACK_TAG: u64 = 0xF;

struct Interner {
    map: HashMap<Vec<u8>, u64>,
    strings: Vec<Vec<u8>>,
}

static INTERNER: Mutex<Option<Interner>> = Mutex::new(None);
static ANY_FALLBACK: AtomicBool = AtomicBool::new(false);

#[inline(always)]
pub fn pack(s: &[u8]) -> Option<u64> {
    let n = s.len();
    if n == 0 || n > 10 {
        return None;
    }
    let mut acc = 0u64;
    let mut bad = 0u8;
    for &c in s {
        let code = CODE[c as usize];
        bad |= code;
        acc = (acc << 6) | (code as u64 & 63);
    }
    if bad & INVALID != 0 {
        return None;
    }
    Some((acc << (6 * (10 - n) + 4)) | n as u64)
}

/// `pack(s)` for a string at most 10 bytes long whose first 16 bytes from
/// its start are readable (they may extend past `s`). Branch-free.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn pack_wide(s: &[u8]) -> Option<u64> {
    use std::arch::x86_64::*;
    // x86-64-v2 baseline: SSSE3 / SSE4.1 are always available.
    let n = s.len();
    let v = _mm_loadu_si128(s.as_ptr() as *const __m128i);
    let idx = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    let inlen = _mm_cmpgt_epi8(_mm_set1_epi8(n as i8), idx);
    let hi = _mm_and_si128(_mm_srli_epi16(v, 4), _mm_set1_epi8(0x0F));
    let lo = _mm_and_si128(v, _mm_set1_epi8(0x0F));
    // class bits: b0 '-' (hi 2), b1 digits (hi 3), b2 hi 4/6 low 1-F,
    // b3 hi 5 low 0-A or F, b4 hi 7 low 0-A
    let lut_hi = _mm_setr_epi8(0, 0, 1, 2, 4, 8, 4, 16, 0, 0, 0, 0, 0, 0, 0, 0);
    let lut_lo = _mm_setr_epi8(
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
    );
    let cls = _mm_and_si128(_mm_shuffle_epi8(lut_hi, hi), _mm_shuffle_epi8(lut_lo, lo));
    let invalid = _mm_and_si128(_mm_cmpeq_epi8(cls, _mm_setzero_si128()), inlen);
    let off = _mm_shuffle_epi8(
        _mm_setr_epi8(0, 0, -0x2D, -0x2F, -0x36, -0x36, -0x3B, -0x3B, 0, 0, 0, 0, 0, 0, 0, 0),
        hi,
    );
    let us = _mm_and_si128(_mm_cmpeq_epi8(v, _mm_set1_epi8(0x5F)), _mm_set1_epi8(-4));
    let codes = _mm_and_si128(_mm_add_epi8(_mm_add_epi8(v, off), us), inlen);
    let t = _mm_maddubs_epi16(codes, _mm_set1_epi16(0x0140));
    let u = _mm_madd_epi16(t, _mm_set1_epi32(0x0001_1000));
    let l0 = _mm_cvtsi128_si32(u) as u32 as u64;
    let l1 = _mm_extract_epi32::<1>(u) as u32 as u64;
    let l2 = _mm_extract_epi32::<2>(u) as u32 as u64;
    let key = (((l0 << 36) | (l1 << 12) | (l2 >> 12)) << 4) | n as u64;
    if (_mm_movemask_epi8(invalid) == 0) & (n.wrapping_sub(1) < 10) {
        Some(key)
    } else {
        None
    }
}

/// `pack(s)` for a string whose first 16 bytes from its start are readable.
#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn pack_wide(s: &[u8]) -> Option<u64> {
    let n = s.len();
    let p = s.as_ptr();
    let mut acc = 0u64;
    let mut bad = 0u8;
    for i in 0..10 {
        let code = CODE[*p.add(i) as usize];
        let m = if i < n { 0xFF } else { 0 };
        bad |= code & m;
        acc = (acc << 6) | (code & m & 63) as u64;
    }
    if (bad & INVALID == 0) & (n.wrapping_sub(1) < 10) {
        Some((acc << 4) | n as u64)
    } else {
        None
    }
}

/// `key_of(s)` with the 16-byte readability guarantee of `pack_wide`.
#[inline(always)]
pub unsafe fn key_of_wide(s: &[u8]) -> u64 {
    // length first: a long ID never packs
    if s.len() > 10 {
        return other(s);
    }
    match pack_wide(s) {
        Some(k) => k,
        None => other(s),
    }
}

/// Key of a non-empty string.
#[inline(always)]
pub fn key_of(s: &[u8]) -> u64 {
    match pack(s) {
        Some(k) => k,
        None => other(s),
    }
}

/// Key of a string that does not pack: long form or interned.
#[inline(always)]
fn other(s: &[u8]) -> u64 {
    other_in::<false>(s)
}

/// `other` for event IDs (EV): long ones are logged to the event log, which
/// is checked only for the keys dedup met more than once (`verify_ev_longs`):
/// event IDs are compared only there and never decoded.
#[inline(always)]
fn other_in<const EV: bool>(s: &[u8]) -> u64 {
    if s.len() == 33 && !LONG_OFF.load(AO::Relaxed) {
        if let Some((hi, lo)) = hex32(&s[1..]) {
            if CODE[s[0] as usize] & INVALID == 0 {
                return long_key::<EV>(s[0], hi, lo);
            }
        }
    }
    intern(s)
}

/// `other` for a user ID that `pack_wide` did not pack: `key_of_wide(s)`
/// is exactly `pack_wide(s).unwrap_or_else(|| other_main(s))` (k4.rs).
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
#[inline(never)]
pub fn other_main(s: &[u8]) -> u64 {
    other_in::<false>(s)
}

/// `other_in::<true>` for an event ID that `pack_wide` did not pack:
/// `key_of_ev_wide(s)` is exactly `pack_wide(s).unwrap_or_else(|| other_ev(s))`.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
#[inline(never)]
pub fn other_ev(s: &[u8]) -> u64 {
    other_in::<true>(s)
}

/// `key_of` for an event_id (see `other_in`).
#[inline(always)]
pub fn key_of_ev(s: &[u8]) -> u64 {
    match pack(s) {
        Some(k) => k,
        None => other_in::<true>(s),
    }
}

/// `key_of_wide` for an event_id (see `other_in`).
#[inline(always)]
pub unsafe fn key_of_ev_wide(s: &[u8]) -> u64 {
    if s.len() > 10 {
        return other_in::<true>(s);
    }
    match pack_wide(s) {
        Some(k) => k,
        None => other_in::<true>(s),
    }
}

// ---------------------------------------------------------------- long IDs

/// Hash seeds tried in turn (rerun after a collision); then the interner.
pub const LONG_TRIES: u64 = 3;
static LONG_SEED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0x5EED_0000_0000_0001);
/// Long IDs go to the interner (after LONG_TRIES collisions, or SS_LONGOFF).
static LONG_OFF: AtomicBool = AtomicBool::new(false);
/// Hash bits kept (SS_LONGBITS, tests: few bits force collisions), <= 60.
static LONG_BITS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(60);
static COLLISION: AtomicBool = AtomicBool::new(false);
/// Capacity of a thread's log (entries; `set_log_hints`): first for the link
/// phase, then (after the first verification) for the rest.
static LOG_HINT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1024);
static LOG_HINT_NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1024);

/// Expected long IDs per worker thread while parsing links, and while
/// parsing events and clicks (sizing only: logs are partitioned arenas, and
/// an arena much larger than its log would be zeroed for nothing).
pub fn set_log_hints(links: usize, rest: usize, ev: usize) {
    LOG_HINT.store(links.clamp(1024, 1 << 26), AO::Relaxed);
    LOG_HINT_NEXT.store(rest.clamp(1024, 1 << 26), AO::Relaxed);
    EV_HINT.store(ev.clamp(1024, 1 << 26), AO::Relaxed);
}
static EV_HINT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1024);

/// One logged long ID: key, first byte, the 32 hex digits as two u64.
#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
struct LEnt {
    key: u64,
    pre: u64,
    hi: u64,
    lo: u64,
}

/// Log partitions: a thread's log is split by the key's top TAB_BITS bits
/// as it is written, so the check needs no scatter pass.
const TAB_BITS: u32 = 8;
const NPART: usize = 1 << TAB_BITS;

thread_local! {
    /// This thread's log (Box::into_raw; null = none yet). A raw pointer in a
    /// Cell: no destructor to register, so each access is a plain TLS load
    /// (a RefCell<Option<..>> went through LocalKey::try_with every time).
    static LOG: std::cell::Cell<*mut Parts<LEnt>> = const { std::cell::Cell::new(std::ptr::null_mut()) };
}
thread_local! {
    /// This thread's log of long event IDs (as LOG).
    static EVLOG: std::cell::Cell<*mut Parts<LEnt>> = const { std::cell::Cell::new(std::ptr::null_mut()) };
}
/// Logs of finished threads (`flush_thread`).
static LOGS: Mutex<Vec<Parts<LEnt>>> = Mutex::new(Vec::new());
static EVLOGS: Mutex<Vec<Parts<LEnt>>> = Mutex::new(Vec::new());

/// One partition of the decode table: the distinct long IDs of the
/// partition in an open-addressing table of entries (key 0 = empty slot; no
/// key is 0), so a lookup touches one entry's line, not an index then an
/// entry. Load at most 2/3.
struct PartTab {
    tab: Vec<LEnt>,
    mask: usize,
    n: usize,
}
/// The decode table of long keys, NPART partitions by the key's top bits.
struct LongTab {
    parts: Vec<PartTab>,
}
static TAB: std::sync::atomic::AtomicPtr<LongTab> = std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

/// Slot of a key in a partition index (bits below the partition bits; the
/// key's low 4 bits are the tag).
#[inline(always)]
fn slot_of(k: u64, mask: usize) -> usize {
    ((k >> 4) as usize) & mask
}

/// Settings from the environment (SS_LONGBITS, SS_LONGOFF); call once at start.
pub fn long_settings() {
    if let Some(b) = std::env::var("SS_LONGBITS").ok().and_then(|v| v.parse::<u32>().ok()) {
        LONG_BITS.store(b.clamp(1, 60), AO::Relaxed);
    }
    if std::env::var("SS_LONGOFF").is_ok() {
        LONG_OFF.store(true, AO::Relaxed);
    }
    VERIFY_OCC.store(std::env::var("SS_VERIFYOCC").map_or(false, |v| v == "1"), AO::Relaxed);
    KNOWN_REQ.store(std::env::var("SS_KNOWN").map_or(false, |v| v == "1"), AO::Relaxed);
    EV_SET1.store(std::env::var("SS_EVSET1").map_or(false, |v| v == "1"), AO::Relaxed);
}

/// SS_VERIFYOCC=1: the verification index sized from occurrences (as before).
static VERIFY_OCC: AtomicBool = AtomicBool::new(false);
/// Largest initial verification index per partition (slots; tests: small,
/// to exercise the growth).
static VERIFY_START: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(4096);
/// Wall ns of the verifications of this run: links, after the parse, event IDs.
pub static VERIFY_SPLIT: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];
static VERIFY_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// 32 lowercase hex digits as (first 16, last 16) values; None for any other byte.
#[inline(always)]
fn hex32(s: &[u8]) -> Option<(u64, u64)> {
    debug_assert!(s.len() == 32);
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let a = hex16_sse(s.as_ptr());
        let b = hex16_sse(s.as_ptr().add(16));
        if let (Some(x), Some(y)) = (a, b) {
            return Some((x, y));
        }
        return None;
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let mut v = [0u64; 2];
        for (h, c) in s.iter().enumerate() {
            let d = match c {
                b'0'..=b'9' => c - b'0',
                b'a'..=b'f' => c - b'a' + 10,
                _ => return None,
            };
            v[h / 16] = (v[h / 16] << 4) | d as u64;
        }
        Some((v[0], v[1]))
    }
}

/// 16 lowercase hex digits at p (16 readable bytes) as a value (first digit
/// most significant), None unless all 16 are `[0-9a-f]`. SSE4.1 (x86-64-v2).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hex16_sse(p: *const u8) -> Option<u64> {
    use std::arch::x86_64::*;
    let v = _mm_loadu_si128(p as *const __m128i);
    let d = _mm_sub_epi8(v, _mm_set1_epi8(b'0' as i8));
    let isd = _mm_cmpeq_epi8(_mm_min_epu8(d, _mm_set1_epi8(9)), d);
    let l = _mm_sub_epi8(v, _mm_set1_epi8(b'a' as i8));
    let isl = _mm_cmpeq_epi8(_mm_min_epu8(l, _mm_set1_epi8(5)), l);
    if _mm_movemask_epi8(_mm_or_si128(isd, isl)) != 0xFFFF {
        return None;
    }
    let nib = _mm_or_si128(_mm_and_si128(isd, d), _mm_and_si128(isl, _mm_add_epi8(l, _mm_set1_epi8(10))));
    // pairs (first digit high): 16 * a + b per 16-bit lane
    let w = _mm_maddubs_epi16(nib, _mm_set1_epi16(0x0110));
    let b = _mm_packus_epi16(w, w);
    Some((_mm_cvtsi128_si64(b) as u64).swap_bytes())
}

#[inline(always)]
fn long_hash(pre: u64, hi: u64, lo: u64) -> u64 {
    let seed = LONG_SEED.load(AO::Relaxed);
    let a = mix(hi ^ seed);
    let b = mix(lo.wrapping_add(seed.rotate_left(17)) ^ pre.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    mix(a ^ b.rotate_left(29))
}

#[inline(always)]
fn long_key<const EV: bool>(pre: u8, hi: u64, lo: u64) -> u64 {
    let h = long_hash(pre as u64, hi, lo) >> 4;
    let bits = LONG_BITS.load(AO::Relaxed);
    let h = if bits >= 60 { h } else { h & ((1u64 << bits) - 1) };
    let key = (h << 4) | 0xE;
    // a user ID that is exactly one of the linked long IDs (verified with
    // the links, in the decode table) needs no log entry: its key was
    // logged with the links, so the final verification still sees every
    // (key, value) pair that occurs (SS_KNOWN=1, experiment)
    if !EV && KNOWN_ON.load(AO::Relaxed) && known_exact(key, pre as u64, hi, lo) {
        return key;
    }
    // each static accessed directly (a LocalKey chosen at run time went
    // through LocalKey::try_with, about 70 instructions per long ID)
    let mut p = if EV { EVLOG.get() } else { LOG.get() };
    if p.is_null() {
        p = new_log_for::<EV>();
    }
    unsafe { (*p).push_unchecked((key >> (64 - TAB_BITS)) as usize, LEnt { key, pre: pre as u64, hi, lo }) };
    key
}

/// Linked long IDs skip the user log while parsing events and clicks
/// (between the link verification and the final one; SS_KNOWN=1).
static KNOWN_ON: AtomicBool = AtomicBool::new(false);
static KNOWN_REQ: AtomicBool = AtomicBool::new(false);

/// The decode table (built by the link verification) holds exactly this ID.
#[inline(always)]
fn known_exact(key: u64, pre: u64, hi: u64, lo: u64) -> bool {
    let t = TAB.load(AO::Acquire);
    if t.is_null() {
        return false;
    }
    let t = unsafe { &*t };
    let pt = unsafe { t.parts.get_unchecked((key >> (64 - TAB_BITS)) as usize) };
    let mut s = slot_of(key, pt.mask);
    loop {
        let e = unsafe { pt.tab.get_unchecked(s) };
        if e.key == key {
            return (e.pre == pre) & (e.hi == hi) & (e.lo == lo);
        }
        if e.key == 0 {
            return false;
        }
        s = (s + 1) & pt.mask;
    }
}

/// This thread's first long ID of its kind: its log.
#[cold]
#[inline(never)]
fn new_log_for<const EV: bool>() -> *mut Parts<LEnt> {
    let p = new_log(if EV { EV_HINT.load(AO::Relaxed) } else { LOG_HINT.load(AO::Relaxed) });
    if EV {
        EVLOG.set(p);
    } else {
        LOG.set(p);
    }
    p
}

#[cold]
#[inline(never)]
fn new_log(hint: usize) -> *mut Parts<LEnt> {
    crate::sys::set_label("log");
    let p = Parts::with_wc(NPART, hint / NPART * 5 / 4 + 64, true);
    crate::sys::set_label("other");
    Box::into_raw(Box::new(p))
}

#[inline(always)]
pub fn is_long(k: u64) -> bool {
    k & 0xF == 0xE
}

/// Moves this thread's log of long IDs to the shared list (main::par calls
/// it at the end of every worker; also before a verification).
pub fn flush_thread() {
    for (log, logs) in [(&LOG, &LOGS), (&EVLOG, &EVLOGS)] {
        let p = log.with(|l| l.replace(std::ptr::null_mut()));
        if !p.is_null() {
            let mut v = unsafe { Box::from_raw(p) };
            v.finish();
            if v.total() > 0 {
                logs.lock().unwrap().push(*v);
            }
        }
    }
}

/// Drops this thread's unflushed logs and every flushed one.
fn clear_logs() {
    for (log, logs) in [(&LOG, &LOGS), (&EVLOG, &EVLOGS)] {
        let p = log.with(|l| l.replace(std::ptr::null_mut()));
        if !p.is_null() {
            drop(unsafe { Box::from_raw(p) });
        }
        logs.lock().unwrap().clear();
    }
}

/// Start of a job (main::run): logs and table of a previous job (the far
/// ingest_ts rerun) are dropped; seed and interner mode are kept.
pub fn new_run() {
    clear_logs();
    KNOWN_ON.store(false, AO::Relaxed);
    VERIFY_CALLS.store(0, AO::Relaxed);
    TAB.store(std::ptr::null_mut(), AO::Release);
}

/// The whole pages inside the decode table's buffers, as (address, length):
/// dead once every name is written (sess::sessions2 releases them with its
/// other dead memory, in parallel, instead of the single-threaded exit).
pub fn long_regions() -> Vec<(usize, usize)> {
    let t = TAB.load(AO::Acquire);
    if t.is_null() {
        return Vec::new();
    }
    let t = unsafe { &*t };
    let mut v = Vec::new();
    let mut add = |a: usize, l: usize| {
        let s = (a + 4095) & !4095;
        let e = (a + l) & !4095;
        if e > s {
            v.push((s, e - s));
        }
    };
    for pt in &t.parts {
        add(pt.tab.as_ptr() as usize, pt.tab.capacity() * std::mem::size_of::<LEnt>());
    }
    v
}

/// A key collision was seen by a verification (the job must run again).
pub fn collision() -> bool {
    COLLISION.load(AO::Relaxed)
}

/// Before a rerun after a collision: next seed (or the interner after
/// LONG_TRIES seeds), logs and tables cleared. Returns the attempt number.
pub fn reset_longs(attempt: u64) {
    clear_logs();
    KNOWN_ON.store(false, AO::Relaxed);
    TAB.store(std::ptr::null_mut(), AO::Release);
    COLLISION.store(false, AO::Relaxed);
    if attempt >= LONG_TRIES {
        LONG_OFF.store(true, AO::Relaxed);
    } else {
        LONG_SEED.store(mix(0x5EED_0000_0000_0001 ^ attempt.wrapping_mul(0xA24B_AED4_963E_E407)), AO::Relaxed);
    }
}

/// Checks every long ID logged so far (all threads, flushed) and builds the
/// decode table: sets `collision()` if one key stands for two IDs. Returns
/// the number of distinct long IDs (0 and no work when there are none).
pub fn verify_longs(threads: usize) -> usize {
    let t0 = std::time::Instant::now();
    KNOWN_ON.store(false, AO::Relaxed);
    let d = verify_longs_inner(threads);
    LOG_HINT.store(LOG_HINT_NEXT.load(AO::Relaxed), AO::Relaxed);
    let ns = t0.elapsed().as_nanos() as u64;
    VERIFY_NS.fetch_add(ns, AO::Relaxed);
    let call = VERIFY_CALLS.fetch_add(1, AO::Relaxed);
    VERIFY_SPLIT[call.min(1)].store(ns, AO::Relaxed);
    // after the link verification the decode table holds exactly the
    // linked long IDs: from now until the next verification, user IDs found
    // in it are not logged
    if call == 0 && KNOWN_REQ.load(AO::Relaxed) && !LONG_OFF.load(AO::Relaxed) {
        KNOWN_ON.store(true, AO::Relaxed);
    }
    d
}

/// Wall time spent in verify_longs (ns) and the last count of logged
/// entries / distinct long IDs (statistics for SS_VERBOSE).
pub static VERIFY_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static LONG_ENTRIES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static LONG_DISTINCT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// "long_entries=.. long_distinct=.. long_verify_ms=.." for the verbose report.
pub fn long_stats() -> String {
    format!(
        "long_entries={} long_distinct={} long_ev_entries={} long_ev_checked={} long_verify_ms={:.1} (links {:.1} parse {:.1} ev {:.1})",
        LONG_ENTRIES.load(AO::Relaxed),
        LONG_DISTINCT.load(AO::Relaxed),
        EV_ENTRIES.load(AO::Relaxed),
        EV_CHECKED.load(AO::Relaxed),
        VERIFY_NS.load(AO::Relaxed) as f64 / 1e6,
        VERIFY_SPLIT[0].load(AO::Relaxed) as f64 / 1e6,
        VERIFY_SPLIT[1].load(AO::Relaxed) as f64 / 1e6,
        VERIFY_SPLIT[2].load(AO::Relaxed) as f64 / 1e6
    )
}

fn verify_longs_inner(threads: usize) -> usize {
    flush_thread();
    let lists = LOGS.lock().unwrap();
    let total: usize = lists.iter().map(|v| v.total()).sum();
    LONG_ENTRIES.store(total as u64, AO::Relaxed);
    if total == 0 {
        return 0;
    }
    // per partition (all threads' logs): an open-addressing index keyed by
    // the key; an equal key with a different value is a collision
    let coll = AtomicBool::new(false);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let done: Mutex<Vec<(usize, PartTab)>> = Mutex::new(Vec::with_capacity(NPART));
    crate::par(threads, |_| {
        let mut mine = Vec::new();
        loop {
            let p = next.fetch_add(1, AO::Relaxed);
            if p >= NPART {
                break;
            }
            let n: usize = lists.iter().map(|v| v.len(p)).sum();
            // sized by the distinct values, not the occurrences (a long ID
            // occurs ~18 times on average): the table starts small and
            // doubles above load 2/3 (SS_VERIFYOCC=1: sized from occurrences)
            let mut size = if VERIFY_OCC.load(AO::Relaxed) { (2 * n).next_power_of_two().max(16) } else { (2 * n).next_power_of_two().clamp(16, VERIFY_START.load(AO::Relaxed)) };
            let mut mask = size - 1;
            let mut tab: Vec<LEnt> = vec![LEnt::default(); size];
            let mut cnt = 0usize;
            let mut bad = false;
            for v in lists.iter() {
                for sl in v.get(p) {
                    for e in sl.iter() {
                        let mut s = slot_of(e.key, mask);
                        loop {
                            let f = unsafe { tab.get_unchecked_mut(s) };
                            if f.key == e.key {
                                bad |= *f != *e;
                                break;
                            }
                            if f.key == 0 {
                                *f = *e;
                                cnt += 1;
                                if 3 * cnt > 2 * size {
                                    // grow: reinsert the distinct entries
                                    size *= 2;
                                    mask = size - 1;
                                    let old = std::mem::replace(&mut tab, vec![LEnt::default(); size]);
                                    for f in old.iter().filter(|f| f.key != 0) {
                                        let mut t = slot_of(f.key, mask);
                                        while tab[t].key != 0 {
                                            t = (t + 1) & mask;
                                        }
                                        tab[t] = *f;
                                    }
                                }
                                break;
                            }
                            s = (s + 1) & mask;
                        }
                    }
                }
            }
            if bad {
                coll.store(true, AO::Relaxed);
            }
            mine.push((p, PartTab { tab, mask, n: cnt }));
        }
        done.lock().unwrap().append(&mut mine);
    });
    drop(lists);
    if coll.load(AO::Relaxed) {
        COLLISION.store(true, AO::Relaxed);
    }
    let mut done = done.into_inner().unwrap();
    done.sort_unstable_by_key(|d| d.0);
    let parts: Vec<PartTab> = done.into_iter().map(|d| d.1).collect();
    let distinct = parts.iter().map(|t| t.n).sum();
    LONG_DISTINCT.store(distinct as u64, AO::Relaxed);
    // black_box: rustc 1.85's LLVM otherwise turns the boxed table into a
    // global with an i8 "initialised" flag and fails to select its atomic swap
    let tab = std::hint::black_box(Box::into_raw(Box::new(LongTab { parts })));
    TAB.store(tab, AO::Release);
    distinct
}

/// SS_EVSET1=1: the event-ID check builds one global key set on one thread.
static EV_SET1: AtomicBool = AtomicBool::new(false);

/// Open-addressing set of keys (keys are hashes already: their own bits
/// index it; 0 = empty, never a key); returns (slots, mask).
fn build_set(m: &[u64]) -> (Vec<u64>, usize) {
    let size = (2 * m.len()).next_power_of_two().max(16);
    let mask = size - 1;
    let mut t = vec![0u64; size];
    for &k in m {
        let mut s = slot_of(k, mask);
        while t[s] != 0 && t[s] != k {
            s = (s + 1) & mask;
        }
        t[s] = k;
    }
    (t, mask)
}

#[inline(always)]
fn set_has(t: &[u64], mask: usize, k: u64) -> bool {
    let mut s = slot_of(k, mask);
    loop {
        let x = unsafe { *t.get_unchecked(s) };
        if x == k {
            return true;
        }
        if x == 0 {
            return false;
        }
        s = (s + 1) & mask;
    }
}

pub static EV_ENTRIES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static EV_CHECKED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The check of long event IDs (logged by key_of_ev). With `multi` (the
/// keys that dedup met more than once): only their entries are compared.
/// That is complete: two different event IDs with one key would be two
/// records with one key in dedup, so the key would be in `multi`. Without
/// (pipelines whose dedup does not collect them): every entry, as
/// verify_longs. Sets `collision()` like verify_longs.
pub fn verify_ev_longs(threads: usize, multi: Option<&[u64]>) {
    let t0 = std::time::Instant::now();
    flush_thread();
    let lists = EVLOGS.lock().unwrap();
    let total: usize = lists.iter().map(|v| v.total()).sum();
    EV_ENTRIES.store(total as u64, AO::Relaxed);
    if total == 0 {
        return;
    }
    if multi.is_some_and(|m| m.is_empty()) {
        return;
    }
    // the keys met more than once, split by the log partition (top TAB_BITS
    // bits; one counting pass): each partition then builds its own small set
    // (cache-sized) on its own thread (SS_EVSET1=1: one global set and a
    // bitmap built on this thread, as before)
    let global = EV_SET1.load(AO::Relaxed);
    let (mstart, mkeys): (Vec<usize>, Vec<u64>) = match multi {
        Some(m) if !global => {
            let mut cnt = vec![0usize; NPART + 1];
            for &k in m {
                cnt[(k >> (64 - TAB_BITS)) as usize + 1] += 1;
            }
            for p in 0..NPART {
                cnt[p + 1] += cnt[p];
            }
            let mut cur = cnt.clone();
            let mut keys = vec![0u64; m.len()];
            for &k in m {
                let p = (k >> (64 - TAB_BITS)) as usize;
                keys[cur[p]] = k;
                cur[p] += 1;
            }
            (cnt, keys)
        }
        _ => (Vec::new(), Vec::new()),
    };
    // global variant: the keys as an open-addressing set (keys are hashes
    // already: their own bits index it; 0 = empty, never a key)
    let set: Option<(Vec<u64>, usize)> = multi.filter(|_| global).map(|m| build_set(m));
    // a bitmap over other key bits in front of the set (256 KB, in cache;
    // most entries are not in the set and stop here)
    const BM_BITS: u32 = 21;
    let bm: Vec<u64> = match multi {
        Some(m) if global => {
            let mut b = vec![0u64; 1 << (BM_BITS - 6)];
            for &k in m {
                let i = ((k >> 24) & ((1 << BM_BITS) - 1)) as usize;
                b[i >> 6] |= 1 << (i & 63);
            }
            b
        }
        _ => Vec::new(),
    };
    let maybe = |k: u64| -> bool {
        let i = ((k >> 24) & ((1 << BM_BITS) - 1)) as usize;
        bm[i >> 6] >> (i & 63) & 1 != 0
    };
    // the entries to compare, per partition, then a sort and run check
    let next = std::sync::atomic::AtomicUsize::new(0);
    let coll = AtomicBool::new(false);
    let checked = std::sync::atomic::AtomicU64::new(0);
    crate::par(threads, |_| {
        let mut v: Vec<LEnt> = Vec::new();
        loop {
            let p = next.fetch_add(1, AO::Relaxed);
            if p >= NPART {
                break;
            }
            v.clear();
            if multi.is_some() && !global {
                let mk = &mkeys[mstart[p]..mstart[p + 1]];
                if mk.is_empty() {
                    // no key of this partition was met twice: nothing to compare
                    continue;
                }
                let (t, mask) = build_set(mk);
                for l in lists.iter() {
                    for sl in l.get(p) {
                        v.extend(sl.iter().filter(|e| set_has(&t, mask, e.key)));
                    }
                }
            } else {
                for l in lists.iter() {
                    for sl in l.get(p) {
                        match &set {
                            Some((t, mask)) => v.extend(sl.iter().filter(|e| maybe(e.key) && set_has(t, *mask, e.key))),
                            None => v.extend_from_slice(sl),
                        }
                    }
                }
            }
            checked.fetch_add(v.len() as u64, AO::Relaxed);
            v.sort_unstable_by_key(|e| e.key);
            for i in 1..v.len() {
                if v[i].key == v[i - 1].key && v[i] != v[i - 1] {
                    coll.store(true, AO::Relaxed);
                }
            }
        }
    });
    EV_CHECKED.store(checked.load(AO::Relaxed), AO::Relaxed);
    if coll.load(AO::Relaxed) {
        COLLISION.store(true, AO::Relaxed);
    }
    VERIFY_NS.fetch_add(t0.elapsed().as_nanos() as u64, AO::Relaxed);
    VERIFY_SPLIT[2].store(t0.elapsed().as_nanos() as u64, AO::Relaxed);
}

/// (first byte, hi, lo) of a long key, from the decode table.
fn long_value(k: u64) -> (u64, u64, u64) {
    let t = TAB.load(AO::Acquire);
    assert!(!t.is_null(), "long key without a decode table");
    let t = unsafe { &*t };
    let pt = &t.parts[(k >> (64 - TAB_BITS)) as usize];
    let mut s = slot_of(k, pt.mask);
    loop {
        let e = &pt.tab[s];
        if e.key == k {
            return (e.pre, e.hi, e.lo);
        }
        assert!(e.key != 0, "long key missing from the decode table");
        s = (s + 1) & pt.mask;
    }
}

fn long_string(k: u64, out: &mut Vec<u8>) {
    let (pre, hi, lo) = long_value(k);
    out.push(pre as u8);
    for v in [hi, lo] {
        for i in (0..16).rev() {
            out.push(b"0123456789abcdef"[((v >> (4 * i)) & 15) as usize]);
        }
    }
}

#[cold]
#[inline(never)]
fn intern(s: &[u8]) -> u64 {
    let mut guard = INTERNER.lock().unwrap();
    let it = guard.get_or_insert_with(|| Interner { map: HashMap::new(), strings: Vec::new() });
    if let Some(&k) = it.map.get(s) {
        return k;
    }
    let k = ((it.strings.len() as u64 + 1) << 4) | FALLBACK_TAG;
    it.strings.push(s.to_vec());
    it.map.insert(s.to_vec(), k);
    ANY_FALLBACK.store(true, AO::Relaxed);
    k
}

/// Not a packed key: interned (0xF) or long (0xE).
#[inline(always)]
pub fn is_fallback(k: u64) -> bool {
    k & 0xF >= 0xE
}

pub fn any_fallback() -> bool {
    ANY_FALLBACK.load(AO::Relaxed)
}

/// Appends the string of `k` to `out`.
#[inline]
pub fn decode_into(k: u64, out: &mut Vec<u8>) {
    if is_long(k) {
        long_string(k, out);
        return;
    }
    if k & 0xF == FALLBACK_TAG {
        let guard = INTERNER.lock().unwrap();
        let it = guard.as_ref().expect("fallback key without interner");
        out.extend_from_slice(&it.strings[((k >> 4) - 1) as usize]);
        return;
    }
    let n = (k & 0xF) as usize;
    let codes = k >> 4;
    let mut buf = [0u8; 10];
    for (i, c) in buf.iter_mut().enumerate() {
        *c = ALPHABET[((codes >> (6 * (9 - i))) & 63) as usize];
    }
    out.extend_from_slice(&buf[..n]);
}

pub fn decode(k: u64) -> Vec<u8> {
    let mut v = Vec::with_capacity(12);
    decode_into(k, &mut v);
    v
}

/// Byte-wise order of the strings behind two keys.
#[inline(always)]
pub fn cmp(a: u64, b: u64) -> Ordering {
    if !is_fallback(a) && !is_fallback(b) {
        a.cmp(&b)
    } else {
        cmp_slow(a, b)
    }
}

#[cold]
#[inline(never)]
fn cmp_slow(a: u64, b: u64) -> Ordering {
    if a == b {
        return Ordering::Equal;
    }
    if is_long(a) && is_long(b) {
        // same width, hex digits in value order: (first byte, value) order
        return long_value(a).cmp(&long_value(b));
    }
    if a & 0xF != FALLBACK_TAG && b & 0xF != FALLBACK_TAG {
        // long vs packed: both strings on the stack (no allocation)
        let (mut x, mut y) = ([0u8; 33], [0u8; 33]);
        let n = stack_string(a, &mut x);
        let m = stack_string(b, &mut y);
        return x[..n].cmp(&y[..m]);
    }
    decode(a).cmp(&decode(b))
}

/// The string of a long or packed key into `out`; returns its length.
#[inline]
fn stack_string(k: u64, out: &mut [u8; 33]) -> usize {
    if is_long(k) {
        let (pre, hi, lo) = long_value(k);
        out[0] = pre as u8;
        for (j, v) in [hi, lo].into_iter().enumerate() {
            for i in 0..16 {
                out[1 + 16 * j + i] = b"0123456789abcdef"[((v >> (4 * (15 - i))) & 15) as usize];
            }
        }
        33
    } else {
        let n = (k & 0xF) as usize;
        let codes = k >> 4;
        for (i, c) in out.iter_mut().enumerate().take(10) {
            *c = ALPHABET[((codes >> (6 * (9 - i))) & 63) as usize];
        }
        n
    }
}

// ---------------------------------------------------------------- click keys
//
// A click_id is only ever compared, byte-wise, to break a tie between two
// clicks with the same ts. A click_id that packs (keys::pack: at most 10
// bytes of the 64-symbol alphabet) is carried as its packed key (low 4 bits
// 1..=10, numeric order = byte order, equal iff the strings are equal).
// Any other click_id (the 33-byte hex form, odd shapes) is carried as a
// reference to its bytes in the input: ((file, length, offset) << 4, low 4
// bits 0) into the click file mapping, which stays mapped for the whole run
// (pages dropped after parsing are read back from the file); fields that do
// not fit the encoding are copied to a side table (file 1023). Such click IDs
// are never hashed or logged. A comparison involving a reference compares the
// bytes: exact for every shape, quoted or not.

/// (base address, length) of every click file mapping of the current run.
static CMAPS_PTR: std::sync::atomic::AtomicPtr<Vec<(usize, usize)>> = std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());
static CSIDE: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());
const CREF_SIDE: u64 = 1023;

/// The click file mappings of this run (main::run, before the clicks are parsed).
pub fn set_click_maps(maps: Vec<(usize, usize)>) {
    // a leaked snapshot for lock-free reads
    let p = Box::into_raw(Box::new(maps));
    CMAPS_PTR.store(std::hint::black_box(p), AO::Release);
    CSIDE.lock().unwrap().clear();
}

/// The click key of click_id field `f` (non-empty) of click file `file`,
/// whose mapping starts at `base`; `wide`: 16 bytes from `f` are readable.
#[inline(always)]
pub fn click_key(file: u32, base: *const u8, f: &[u8], wide: bool) -> u64 {
    let k = if wide { unsafe { pack_wide(f) } } else { pack(f) };
    if let Some(k) = k {
        return k;
    }
    let off = f.as_ptr() as usize - base as usize;
    let len = f.len();
    if (file as u64) < CREF_SIDE && len < 1024 && (off as u64) < (1 << 40) {
        (((file as u64) << 50) | ((len as u64) << 40) | off as u64) << 4
    } else {
        click_ref_side(f)
    }
}

#[cold]
#[inline(never)]
fn click_ref_side(f: &[u8]) -> u64 {
    let mut g = CSIDE.lock().unwrap();
    g.push(f.to_vec());
    ((CREF_SIDE << 50) | (g.len() as u64 - 1)) << 4
}

/// Byte-wise order of the click_ids behind two click keys.
#[inline(always)]
pub fn click_cmp(a: u64, b: u64) -> Ordering {
    if a == b {
        return Ordering::Equal;
    }
    if (a & 0xF != 0) & (b & 0xF != 0) {
        // both packed
        return a.cmp(&b);
    }
    click_cmp_bytes(a, b)
}

#[inline(never)]
fn click_cmp_bytes(a: u64, b: u64) -> Ordering {
    // the side table is locked only if a side reference is involved
    let side_needed = |h: u64| h & 0xF == 0 && (h >> 4) >> 50 == CREF_SIDE;
    let guard = if side_needed(a) || side_needed(b) { Some(CSIDE.lock().unwrap()) } else { None };
    let maps = unsafe { &*CMAPS_PTR.load(AO::Acquire) };
    let (mut ba, mut bb) = ([0u8; 33], [0u8; 33]);
    let bytes = |h: u64, buf: &mut [u8; 33]| -> &[u8] {
        if h & 0xF != 0 {
            let n = stack_string(h, buf);
            return unsafe { std::slice::from_raw_parts(buf.as_ptr(), n) };
        }
        let r = h >> 4;
        let file = r >> 50;
        if file == CREF_SIDE {
            let v = &guard.as_ref().unwrap()[(r & ((1 << 50) - 1)) as usize];
            return unsafe { std::slice::from_raw_parts(v.as_ptr(), v.len()) };
        }
        let (base, mlen) = maps[file as usize];
        let len = ((r >> 40) & 1023) as usize;
        let off = (r & ((1 << 40) - 1)) as usize;
        assert!(off + len <= mlen, "click reference outside its file");
        unsafe { std::slice::from_raw_parts((base + off) as *const u8, len) }
    };
    let x = bytes(a, &mut ba);
    let y = bytes(b, &mut bb);
    x.cmp(y)
}

/// A well-mixed 64-bit hash of a key (splitmix64 finalizer, a bijection).
#[inline(always)]
pub fn mix(k: u64) -> u64 {
    let mut z = k;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

/// Inverse of `mix`.
#[inline]
pub fn unmix(h: u64) -> u64 {
    let mut z = h;
    z = z ^ (z >> 31) ^ (z >> 62);
    z = z.wrapping_mul(0x319642b2d24d8ec3);
    z = z ^ (z >> 27) ^ (z >> 54);
    z = z.wrapping_mul(0x96de1b173f119089);
    z ^ (z >> 30) ^ (z >> 60)
}

/// Tests that change or depend on the global long-key state (seed, bits,
/// interner mode) hold this: one at a time (keys tests, k4test, k5test).
#[cfg(test)]
pub(crate) static LONG_TEST: Mutex<()> = Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unmix_inverts_mix() {
        let mut x: u64 = 0x0123_4567_89AB_CDEF;
        for _ in 0..2_000_000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            assert_eq!(unmix(mix(x)), x);
            assert_eq!(mix(unmix(x)), x);
        }
        for v in [0u64, 1, u64::MAX, 1 << 63, 0xF, 0x10] {
            assert_eq!(unmix(mix(v)), v);
        }
    }

    /// `pack_wide` agrees with `pack` on every string, whatever follows it.
    #[test]
    fn pack_wide_matches_pack() {
        let mut x: u64 = 0xDEAD_BEEF_1234_5677;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut ok = 0u64;
        for round in 0..5_000_000u64 {
            let n = (next() % 14) as usize;
            let mut s = Vec::with_capacity(n + 20);
            for _ in 0..n {
                let r = next();
                let c = match r % 8 {
                    0 => (r >> 8) as u8,
                    1 => ALPHABET[((r >> 8) % 64) as usize].wrapping_add(((r >> 16) % 3) as u8).wrapping_sub(1),
                    _ => ALPHABET[((r >> 8) % 64) as usize],
                };
                s.push(c);
            }
            let want = pack(&s);
            ok += want.is_some() as u64;
            for _ in 0..20 {
                s.push(if round % 2 == 0 { (next() & 0xFF) as u8 } else { ALPHABET[(next() % 64) as usize] });
            }
            assert_eq!(unsafe { pack_wide(&s[..n]) }, want, "{:?}", &s[..n]);
        }
        assert!(ok > 1_000_000);
        // every single byte value at every position
        for n in 1..=11usize {
            for pos in 0..n {
                for c in 0..=255u8 {
                    let mut s = vec![b'a'; n];
                    s[pos] = c;
                    let want = pack(&s);
                    s.extend_from_slice(&[0xFF; 20]);
                    assert_eq!(unsafe { pack_wide(&s[..n]) }, want);
                }
            }
        }
    }

    use super::LONG_TEST;

    fn hex_id(pre: u8, hi: u64, lo: u64) -> Vec<u8> {
        let mut s = vec![pre];
        s.extend_from_slice(format!("{:016x}{:016x}", hi, lo).as_bytes());
        s
    }

    /// Long keys: exact shape test, round trip through the decode table,
    /// equality and byte-wise order against the strings, mixed with packed,
    /// interned and near-long shapes.
    #[test]
    fn long_keys_exact() {
        let _g = LONG_TEST.lock().unwrap_or_else(|e| e.into_inner());
        reset_longs(0);
        LONG_OFF.store(false, AO::Relaxed);
        LONG_BITS.store(60, AO::Relaxed);
        let base = 0x0123_4567_89ab_cdefu64;
        let mut samples: Vec<Vec<u8>> = Vec::new();
        for &(hi, lo) in &[(base, base), (base, base ^ 1), (base ^ 1, base), (base, base ^ (1 << 63)), (base ^ (1 << 63), base), (0, 0), (u64::MAX, u64::MAX), (0, u64::MAX), (u64::MAX, 0)] {
            for &pre in b"uekU-_z0" {
                samples.push(hex_id(pre, hi, lo));
            }
        }
        // near shapes: interner (uppercase hex, 31 / 33 / 34 / 64 digits, bad byte, non-alphabet prefix)
        let mut up = hex_id(b'u', base, base);
        up[5] = b'A';
        samples.push(up.clone());
        samples.push(b"u0123456789abcdef0123456789abcde".to_vec());
        samples.push(b"u0123456789abcdef0123456789abcdef0".to_vec());
        samples.push(b"u0123456789abcdef0123456789abcdef01".to_vec());
        samples.push([b"u".as_slice(), &[b'0'; 64]].concat());
        let mut g = hex_id(b'u', base, base);
        g[32] = b'g';
        samples.push(g);
        samples.push(hex_id(b' ', base, base));
        samples.push(hex_id(0xC3, base, base));
        // short and packed IDs that interleave in byte order with the long ones
        for s in [&b"u"[..], b"u0", b"u01", b"u0123456789", b"u0123", b"uz", b"u1", b"e0", b"k", b"U", b"u-", b"zzzzzzzzzz", b"u0123456789abcdef0123456789abcdef"] {
            samples.push(s.to_vec());
        }
        let keys: Vec<u64> = samples.iter().map(|s| key_of(s)).collect();
        // long form only for the exact shape
        for (s, &k) in samples.iter().zip(&keys) {
            let long_shape = s.len() == 33 && CODE[s[0] as usize] & INVALID == 0 && s[1..].iter().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(c));
            assert_eq!(is_long(k), long_shape, "{:?}", String::from_utf8_lossy(s));
        }
        assert_eq!(verify_longs(2), samples.iter().zip(&keys).filter(|(_, &k)| is_long(k)).map(|(s, _)| s.clone()).collect::<std::collections::HashSet<_>>().len());
        assert!(!collision());
        for (s, &k) in samples.iter().zip(&keys) {
            assert_eq!(&decode(k), s, "{:?}", String::from_utf8_lossy(s));
        }
        for i in 0..samples.len() {
            for j in 0..samples.len() {
                assert_eq!(cmp(keys[i], keys[j]), samples[i].cmp(&samples[j]), "{:?} {:?}", String::from_utf8_lossy(&samples[i]), String::from_utf8_lossy(&samples[j]));
                assert_eq!(keys[i] == keys[j], samples[i] == samples[j]);
            }
        }
        // the wide variant agrees
        for s in &samples {
            let mut w = s.clone();
            let n = w.len();
            w.extend_from_slice(&[0xAB; 16]);
            assert_eq!(unsafe { key_of_wide(&w[..n]) }, key_of(s));
        }
        reset_longs(0);
    }

    /// Few hash bits: keys collide, verification says so; the next seeds
    /// and finally the interner (as main's rerun loop does) give exact keys.
    #[test]
    fn long_key_collisions_detected() {
        let _g = LONG_TEST.lock().unwrap_or_else(|e| e.into_inner());
        reset_longs(0);
        LONG_OFF.store(false, AO::Relaxed);
        LONG_BITS.store(4, AO::Relaxed);
        let ids: Vec<Vec<u8>> = (0..200u64).map(|i| hex_id(b'u', mix(i), mix(i + 1000))).collect();
        let mut attempt = 0;
        loop {
            let keys: Vec<u64> = ids.iter().map(|s| key_of(s)).collect();
            verify_longs(3);
            if !collision() {
                // exact: distinct IDs have distinct keys and decode back
                let set: std::collections::HashSet<u64> = keys.iter().copied().collect();
                assert_eq!(set.len(), ids.len());
                for (s, &k) in ids.iter().zip(&keys) {
                    assert_eq!(&decode(k), s);
                }
                break;
            }
            attempt += 1;
            assert!(attempt <= LONG_TRIES, "no exact keys after the interner fallback");
            reset_longs(attempt);
        }
        assert_eq!(attempt, LONG_TRIES, "4 hash bits for 200 IDs must collide for every seed");
        assert!(LONG_OFF.load(AO::Relaxed));
        LONG_BITS.store(60, AO::Relaxed);
        LONG_OFF.store(false, AO::Relaxed);
        reset_longs(0);
    }

    /// The verification index grows from a small start: distinct values,
    /// repeats and forced collisions give the same verdicts and decode table
    /// as an index sized from the occurrences.
    #[test]
    fn verify_index_grows() {
        let _g = LONG_TEST.lock().unwrap_or_else(|e| e.into_inner());
        for &(bits, start) in &[(60u32, 16usize), (60, 4096), (9, 16), (9, 1 << 20)] {
            reset_longs(0);
            LONG_OFF.store(false, AO::Relaxed);
            LONG_BITS.store(bits, AO::Relaxed);
            VERIFY_START.store(start, AO::Relaxed);
            // 20000 distinct IDs, each 1 to 4 times, interleaved
            let ids: Vec<Vec<u8>> = (0..20_000u64).map(|i| hex_id(b"uUk-"[(i % 4) as usize], mix(i ^ 0x55), mix(i + 99))).collect();
            let mut keys = Vec::new();
            for r in 0..4u64 {
                for (i, s) in ids.iter().enumerate() {
                    if (i as u64 + r) % 4 >= r {
                        keys.push((i, key_of(s)));
                    }
                }
            }
            let d = verify_longs(3);
            let distinct_keys: std::collections::HashSet<u64> = keys.iter().map(|&(_, k)| k).collect();
            assert_eq!(d, distinct_keys.len());
            if bits == 60 {
                assert!(!collision());
                assert_eq!(d, ids.len());
                for &(i, k) in keys.iter().step_by(7) {
                    assert_eq!(decode(k), ids[i]);
                }
            } else {
                // 2^9 keys for 20000 IDs: collisions must be seen
                assert!(collision());
            }
        }
        VERIFY_START.store(4096, AO::Relaxed);
        LONG_BITS.store(60, AO::Relaxed);
        reset_longs(0);
    }

    /// Event-ID log: a collision is found by the full check and by the check
    /// of the keys met more than once, and not reported for equal IDs.
    #[test]
    fn event_long_keys_checked() {
        let _g = LONG_TEST.lock().unwrap_or_else(|e| e.into_inner());
        reset_longs(0);
        LONG_OFF.store(false, AO::Relaxed);
        LONG_BITS.store(4, AO::Relaxed);
        let ids: Vec<Vec<u8>> = (0..100u64).map(|i| hex_id(b'e', mix(i + 7), mix(i + 77))).collect();
        let keys: Vec<u64> = ids.iter().map(|s| key_of_ev(s)).collect();
        // the same ID again (a duplicate copy) never counts as a collision
        let again = key_of_ev(&ids[0]);
        assert_eq!(again, keys[0]);
        // keys met more than once: with 4 bits, distinct IDs share keys
        let mut multi: Vec<u64> = keys.clone();
        multi.sort_unstable();
        let dup: Vec<u64> = multi.windows(2).filter(|w| w[0] == w[1]).map(|w| w[0]).collect();
        assert!(!dup.is_empty());
        verify_ev_longs(2, Some(&[]));
        assert!(!collision());
        verify_ev_longs(2, Some(&[keys[0]]));
        let c0 = collision();
        COLLISION.store(false, AO::Relaxed);
        verify_ev_longs(2, Some(&dup));
        assert!(collision(), "targeted check must find the shared keys");
        COLLISION.store(false, AO::Relaxed);
        verify_ev_longs(2, None);
        assert!(collision(), "full check must find the shared keys");
        // keys[0] alone: a collision iff another ID has its key
        assert_eq!(c0, keys.iter().skip(1).any(|&k| k == keys[0]));
        // exact keys (60 bits): no collision
        reset_longs(0);
        LONG_BITS.store(60, AO::Relaxed);
        let keys: Vec<u64> = ids.iter().map(|s| key_of_ev(s)).collect();
        let _ = key_of_ev(&ids[3]);
        verify_ev_longs(2, Some(&keys));
        assert!(!collision());
        verify_ev_longs(2, None);
        assert!(!collision());
        reset_longs(0);
    }

    /// Click references: ties compare the referenced bytes, exactly the
    /// byte-wise order of the click_ids, for quoted / unquoted copies, long,
    /// short and odd shapes, and fields that go to the side table.
    #[test]
    fn click_refs_order() {
        let _g = LONG_TEST.lock().unwrap_or_else(|e| e.into_inner());
        let ids: Vec<Vec<u8>> = vec![
            b"k1".to_vec(), b"k10".to_vec(), b"k0123456789abcdef0123456789abcdef".to_vec(), b"k0123456789abcdef0123456789abcdee".to_vec(),
            b"K0123456789abcdef0123456789abcdef".to_vec(), b"k0".to_vec(), b"k".to_vec(), b" k1".to_vec(), b"k1 ".to_vec(), b"\xc3\xa9".to_vec(),
            vec![b'k'; 5000], b"k0123456789ABCDEF0123456789abcdef".to_vec(),
        ];
        // a "file": every id twice, once quoted
        let mut buf = Vec::new();
        let mut pos = Vec::new();
        for id in &ids {
            pos.push((buf.len(), id.len()));
            buf.extend_from_slice(id);
            buf.push(b',');
            buf.push(b'"');
            pos.push((buf.len(), id.len()));
            buf.extend_from_slice(id);
            buf.extend_from_slice(b"\",\n");
        }
        set_click_maps(vec![(buf.as_ptr() as usize, buf.len())]);
        let refs: Vec<u64> = pos.iter().map(|&(o, n)| click_key(0, buf.as_ptr(), &buf[o..o + n], false)).collect();
        // a file index beyond the encoding goes to the side table (a short
        // id is packed: use a long one)
        let side = click_key(5000, buf.as_ptr(), &buf[pos[4].0..pos[4].0 + pos[4].1], false);
        for i in 0..refs.len() {
            let si = &buf[pos[i].0..pos[i].0 + pos[i].1];
            assert_eq!(click_cmp(refs[i], side), si.cmp(&ids[2][..]));
            for j in 0..refs.len() {
                let sj = &buf[pos[j].0..pos[j].0 + pos[j].1];
                assert_eq!(click_cmp(refs[i], refs[j]), si.cmp(sj), "{} {}", i, j);
            }
        }
        set_click_maps(Vec::new());
    }

    /// The SSE hex decoder agrees with a scalar one on random strings.
    #[test]
    fn hex32_matches_scalar() {
        let mut x: u64 = 0x1234_5678_9ABC_DEF1;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let alpha = b"0123456789abcdefABCDEFgz/:`@G \xff";
        for _ in 0..300_000 {
            let mut s = [0u8; 32];
            let bad = next() % 4 == 0;
            for c in s.iter_mut() {
                *c = b"0123456789abcdef"[(next() % 16) as usize];
            }
            if bad {
                let k = (next() % 32) as usize;
                s[k] = if next() % 2 == 0 { alpha[(next() % alpha.len() as u64) as usize] } else { (next() & 0xFF) as u8 };
            }
            let want = (|| {
                let mut v = [0u64; 2];
                for (h, c) in s.iter().enumerate() {
                    let d = match c {
                        b'0'..=b'9' => c - b'0',
                        b'a'..=b'f' => c - b'a' + 10,
                        _ => return None,
                    };
                    v[h / 16] = (v[h / 16] << 4) | d as u64;
                }
                Some((v[0], v[1]))
            })();
            assert_eq!(hex32(&s), want, "{:?}", String::from_utf8_lossy(&s));
        }
    }

    #[test]
    fn roundtrip_and_order() {
        let samples: Vec<&[u8]> = vec![
            b"u1", b"U1", b"u1-", b"u", b"-", b"zzzzzzzzzz", b"a", b"a-", b"a-b", b"a0", b"aa", b"u42huhoug",
            b" u1", b"u1 ", b"A\tB", b"user_id_a", b"user_id_b", b"zzzzzzzzzzz", b"e3\r", b"_", b"Z", b"0",
        ];
        let keys: Vec<u64> = samples.iter().map(|s| key_of(s)).collect();
        for (s, &k) in samples.iter().zip(&keys) {
            assert_eq!(&decode(k)[..], *s);
            assert_ne!(k, 0);
        }
        for i in 0..samples.len() {
            for j in 0..samples.len() {
                assert_eq!(cmp(keys[i], keys[j]), samples[i].cmp(samples[j]), "{:?} {:?}", samples[i], samples[j]);
                assert_eq!(keys[i] == keys[j], samples[i] == samples[j]);
            }
        }
    }
}
