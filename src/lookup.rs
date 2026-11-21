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
    pub fn build(entries: &[(Vec<u8>, u16)]) -> IdTable {
        let mut map = HashMap::with_capacity(entries.len());
        for (k, v) in entries {
            map.insert(k.clone(), *v);
        }
        // Detect PREFIX + canonical decimal for every key.
        let mut use_direct = !entries.is_empty();
        let mut prefix: Vec<u8> = Vec::new();
        if let Some((first, _)) = entries.first() {
            let plen = first.iter().position(|c| c.is_ascii_digit()).unwrap_or(first.len());
            prefix = first[..plen].to_vec();
        }
        let mut maxn = 0usize;
        if use_direct {
            for (k, _) in entries {
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
                let n = canonical_number(&k[prefix.len()..]).unwrap();
                direct[n] = *v;
            }
        }
        IdTable { prefix, direct, map, use_direct }
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
        let e2: Vec<(Vec<u8>, u16)> = vec![(b"c_1".to_vec(), 0), (b"x9".to_vec(), 1)];
        let t2 = IdTable::build(&e2);
        assert!(!t2.use_direct);
        assert_eq!(t2.get(b"x9"), 1);
        assert_eq!(t2.get(b"c_1"), 0);
        assert_eq!(t2.get(b"c_01"), NONE);
    }
}
