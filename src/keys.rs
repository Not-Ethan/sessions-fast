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
