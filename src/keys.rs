//! IDs as 64-bit keys.
//!
//! A non-empty string of at most 10 bytes drawn from the 64-symbol alphabet
//! `-0-9A-Z_a-z` is packed exactly: 6-bit codes (assigned in ASCII order),
//! left-aligned in the top 60 bits, with the length in the low 4 bits. For two
//! packed keys, numeric order equals byte-wise string order and equality equals
//! string equality.
//!
//! Every other string is interned in a global side table and gets the key
//! `(index << 4) | 0xF`. Such keys are equal iff the strings are equal; their
//! order is resolved by comparing the actual bytes (rare, slow path).
//!
//! Key 0 is never produced and serves as "empty".

use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering as AO};
use std::sync::Mutex;

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
    match pack_wide(s) {
        Some(k) => k,
        None => intern(s),
    }
}

/// Key of a non-empty string.
#[inline(always)]
pub fn key_of(s: &[u8]) -> u64 {
    match pack(s) {
        Some(k) => k,
        None => intern(s),
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

#[inline(always)]
pub fn is_fallback(k: u64) -> bool {
    k & 0xF == FALLBACK_TAG
}

pub fn any_fallback() -> bool {
    ANY_FALLBACK.load(AO::Relaxed)
}

/// Appends the string of `k` to `out`.
#[inline]
pub fn decode_into(k: u64, out: &mut Vec<u8>) {
    if is_fallback(k) {
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
    if (a & 0xF) != FALLBACK_TAG && (b & 0xF) != FALLBACK_TAG {
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
    decode(a).cmp(&decode(b))
}

/// A well-mixed 64-bit hash of a key (splitmix64 finalizer, a bijection).
#[inline(always)]
pub fn mix(k: u64) -> u64 {
    let mut z = k;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

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
