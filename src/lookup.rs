//! Exact string -> small integer lookup for product and campaign IDs.
//!
//! Fast path: when every key in the file is `PREFIX` + a canonical decimal
//! (first digit 1-9, value < 2^24), a direct array indexed by the number is
//! used. A probe string that is not of that canonical form cannot equal any
//! key, so it is "not found". Otherwise a hash map on the exact bytes is used.

use std::collections::HashMap;

pub const NONE: u16 = u16::MAX;

pub struct IdTable {
    prefix: Vec<u8>,
    direct: Vec<u16>,
    map: HashMap<Vec<u8>, u16>,
    use_direct: bool,
    /// `get_wide` fast path: direct table with a prefix of at most 8 bytes,
    /// held as a little-endian word and lane mask.
    wide: bool,
    pword: u64,
    pmask: u64,
}

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

/// 8 ASCII digits (first char in the lowest byte) to their value.
#[inline(always)]
fn parse8(w: u64) -> u64 {
    let l = w.wrapping_sub(0x3030_3030_3030_3030);
    let c = (l.wrapping_mul(10).wrapping_add(l >> 8)) & 0x00FF_00FF_00FF_00FF;
    let c = (c.wrapping_mul(100).wrapping_add(c >> 16)) & 0x0000_FFFF_0000_FFFF;
    (c.wrapping_mul(10000).wrapping_add(c >> 32)) & 0xFFFF_FFFF
}

#[inline(always)]
fn canonical_number(s: &[u8]) -> Option<usize> {
    if s.is_empty() || s.len() > 7 || s[0] == b'0' {
        return None;
    }
    let mut v = 0usize;
    for &c in s {
        let d = c.wrapping_sub(b'0');
        if d >= 10 {
            return None;
        }
        v = v * 10 + d as usize;
    }
    Some(v)
}

impl IdTable {
    pub fn build<K: AsRef<[u8]>>(entries: &[(K, u16)]) -> IdTable {
        let mut map = HashMap::new();
        // Detect PREFIX + canonical decimal for every key.
        let mut use_direct = !entries.is_empty();
        let mut prefix: Vec<u8> = Vec::new();
        if let Some((first, _)) = entries.first() {
            let first = first.as_ref();
            let plen = first.iter().position(|c| c.is_ascii_digit()).unwrap_or(first.len());
            prefix = first[..plen].to_vec();
        }
        let mut maxn = 0usize;
        if use_direct {
            for (k, _) in entries {
                let k = k.as_ref();
                if !k.starts_with(&prefix) {
                    use_direct = false;
                    break;
                }
                match canonical_number(&k[prefix.len()..]) {
                    Some(n) if n < (1 << 24) => maxn = maxn.max(n),
                    _ => {
                        use_direct = false;
                        break;
                    }
                }
            }
        }
        let mut direct = Vec::new();
        if use_direct {
            direct = vec![NONE; maxn + 1];
            for (k, v) in entries {
                let n = canonical_number(&k.as_ref()[prefix.len()..]).unwrap();
                direct[n] = *v;
            }
        } else {
            map.reserve(entries.len());
            for (k, v) in entries {
                map.insert(k.as_ref().to_vec(), *v);
            }
        }
        let wide = use_direct && prefix.len() <= 8;
        let mut pword = 0u64;
        let mut pmask = 0u64;
        if wide {
            for (i, &c) in prefix.iter().enumerate() {
                pword |= (c as u64) << (8 * i);
                pmask |= 0xFF << (8 * i);
            }
        }
        IdTable { prefix, direct, map, use_direct, wide, pword, pmask }
    }

    /// Same as `get(s)`; 16 bytes from the start of `s` must be readable
    /// (they may extend past `s`).
    #[inline(always)]
    pub unsafe fn get_wide(&self, s: &[u8]) -> u16 {
        if !self.wide {
            return self.get(s);
        }
        let p = s.as_ptr();
        let pl = self.prefix.len();
        let k = s.len().wrapping_sub(pl); // digit count, must be 1..=7
        let w = rd(p, 0);
        let d = rd(p, pl);
        let kk = k.min(7);
        let lanes = (1u64 << (8 * kk)) - 1;
        let first = (d & 0xFF) as u8;
        let ok = ((w & self.pmask) == self.pword)
            & (k.wrapping_sub(1) < 7)
            & (digits_bad(d, lanes) == 0)
            & (first != b'0');
        let sh = 8 * (8 - kk as u32); // 8..=64 for kk = 7..=0
        let aligned = if sh < 64 { (d << sh) | (0x3030_3030_3030_3030 & ((1u64 << sh) - 1)) } else { 0x3030_3030_3030_3030 };
        let n = parse8(aligned) as usize;
        if ok && n < self.direct.len() {
            *self.direct.get_unchecked(n)
        } else {
            NONE
        }
    }

    #[inline(always)]
    pub fn get(&self, s: &[u8]) -> u16 {
        if self.use_direct {
            let pl = self.prefix.len();
            if s.len() > pl && s[..pl] == self.prefix[..] {
                if let Some(n) = canonical_number(&s[pl..]) {
                    if n < self.direct.len() {
                        return self.direct[n];
                    }
                }
            }
            NONE
        } else {
            self.map.get(s).copied().unwrap_or(NONE)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_and_map() {
        let e: Vec<(Vec<u8>, u16)> = vec![(b"p1".to_vec(), 0), (b"p2".to_vec(), 1), (b"p10".to_vec(), 2)];
        let t = IdTable::build(&e);
        assert!(t.use_direct);
        assert_eq!(t.get(b"p1"), 0);
        assert_eq!(t.get(b"p10"), 2);
        for miss in [&b"p01"[..], b"P1", b"p1 ", b" p1", b"p", b"p3", b"", b"p99999999", b"q1"] {
            assert_eq!(t.get(miss), NONE);
        }
        // get_wide agrees with get on probes followed by arbitrary bytes
        let mut x: u64 = 0x5EED_0000_0000_0001;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let tables: Vec<IdTable> = vec![
            IdTable::build(&(1..50_000u32).map(|i| (format!("p{}", i).into_bytes(), (i % 21) as u16)).collect::<Vec<_>>()),
            IdTable::build(&(1..2001u32).map(|i| (format!("c_{}", i).into_bytes(), i as u16)).collect::<Vec<_>>()),
            IdTable::build(&(1..300u32).map(|i| (format!("abcdefgh{}", i * 7).into_bytes(), i as u16)).collect::<Vec<_>>()),
            IdTable::build(&(1..300u32).map(|i| (format!("q{}", i * 33331).into_bytes(), i as u16)).collect::<Vec<_>>()),
        ];
        let alpha = b"0123456789pcq_P C\"\r-abcdefgh";
        for t in &tables {
            for round in 0..1_000_000u64 {
                let r = next();
                let mut s: Vec<u8> = match r % 4 {
                    0 => format!("{}{}", String::from_utf8_lossy(&t.prefix), (r >> 8) % 120_000).into_bytes(),
                    1 => format!("{}{:0w$}", String::from_utf8_lossy(&t.prefix), (r >> 8) % 1000, w = ((r >> 40) % 9) as usize).into_bytes(),
                    _ => (0..(r >> 8) % 12).map(|_| alpha[(next() % alpha.len() as u64) as usize]).collect(),
                };
                if r % 7 == 0 && !s.is_empty() {
                    let at = (next() as usize) % s.len();
                    s[at] = alpha[(next() as usize) % alpha.len()];
                }
                let n = s.len();
                for _ in 0..20 {
                    s.push(if round % 2 == 0 { (next() & 0xFF) as u8 } else { alpha[(next() as usize) % alpha.len()] });
                }
                assert_eq!(unsafe { t.get_wide(&s[..n]) }, t.get(&s[..n]), "{:?}", String::from_utf8_lossy(&s[..n]));
            }
        }
        let e2: Vec<(Vec<u8>, u16)> = vec![(b"c_1".to_vec(), 0), (b"x9".to_vec(), 1)];
        let t2 = IdTable::build(&e2);
        assert!(!t2.use_direct);
        assert_eq!(t2.get(b"x9"), 1);
        assert_eq!(t2.get(b"c_1"), 0);
        assert_eq!(t2.get(b"c_01"), NONE);
    }
}
