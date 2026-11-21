//! Timestamps (rules.md "Timestamps") and other small field parsers.

const DIM: [u32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

#[inline(always)]
fn is_leap(y: u32) -> bool {
    y % 4 == 0 && (y % 100 != 0 || y % 400 == 0)
}

pub fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * ((m + 9) % 12) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

#[inline(always)]
fn d2(b: &[u8], i: usize) -> Option<u32> {
    let a = b[i].wrapping_sub(b'0');
    let c = b[i + 1].wrapping_sub(b'0');
    if a < 10 && c < 10 {
        Some(a as u32 * 10 + c as u32)
    } else {
        None
    }
}

/// `YYYY-MM-DDTHH:MM:SS` at the start of `b` (len >= 19) as epoch ms.
#[inline(always)]
fn parse_civil(b: &[u8]) -> Option<i64> {
    if b[4] != b'-' || b[7] != b'-' || b[10] != b'T' || b[13] != b':' || b[16] != b':' {
        return None;
    }
    let y = d2(b, 0)? * 100 + d2(b, 2)?;
    let mo = d2(b, 5)?;
    let d = d2(b, 8)?;
    let h = d2(b, 11)?;
    let mi = d2(b, 14)?;
    let s = d2(b, 17)?;
    if mo == 0 || mo > 12 || d == 0 || h > 23 || mi > 59 || s > 59 {
        return None;
    }
    let dim = DIM[(mo - 1) as usize] + (mo == 2 && is_leap(y)) as u32;
    if d > dim {
        return None;
    }
    let days = days_from_civil(y as i64, mo as i64, d as i64);
    Some((((days * 24 + h as i64) * 60 + mi as i64) * 60 + s as i64) * 1000)
}

/// Parses F1 / F2 / F3 into epoch milliseconds. `None` is a bad timestamp
/// (including an F3 value too large for a signed 64-bit integer, which is
/// outside every window anyway).
#[inline(always)]
pub fn parse_ts(b: &[u8]) -> Option<i64> {
    let n = b.len();
    if n == 20 && b[19] == b'Z' {
        return parse_civil(b);
    }
    if n == 25 && (b[19] == b'+' || b[19] == b'-') && b[22] == b':' {
        let base = parse_civil(b)?;
        let oh = d2(b, 20)?;
        let om = d2(b, 23)?;
        if oh > 23 || om > 59 {
            return None;
        }
        let off = (oh * 60 + om) as i64 * 60_000;
        return Some(if b[19] == b'+' { base - off } else { base + off });
    }
    parse_f3(b)
}

#[inline(always)]
fn parse_f3(b: &[u8]) -> Option<i64> {
    if b.is_empty() {
        return None;
    }
    let mut i = 0;
    while i < b.len() && b[i] == b'0' {
        i += 1;
    }
    let rest = &b[i..];
    if rest.len() > 19 {
        return None; // too long for i64 (or not digits): bad either way
    }
    let mut v: u64 = 0;
    for &c in rest {
        let d = c.wrapping_sub(b'0');
        if d >= 10 {
            return None;
        }
        v = v * 10 + d as u64;
    }
    if v > i64::MAX as u64 {
        return None;
    }
    Some(v as i64)
}

/// A purchase price: ASCII digits only, value 1..=10_000_000.
#[inline(always)]
pub fn parse_price(b: &[u8]) -> Option<u32> {
    if b.is_empty() {
        return None;
    }
    let mut v: u32 = 0;
    for &c in b {
        let d = c.wrapping_sub(b'0');
        if d >= 10 {
            return None;
        }
        v = v * 10 + d as u32;
        if v > 10_000_000 {
            return None;
        }
    }
    if v == 0 {
        None
    } else {
        Some(v)
    }
}


// ------------------------------------------------------------------ fast path

#[inline(always)]
fn load8(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

/// True iff every byte lane selected by `lanes` (0xFF per lane) is an ASCII digit.
#[inline(always)]
fn digits_ok(w: u64, lanes: u64) -> bool {
    let hi = (w & 0xF0F0_F0F0_F0F0_F0F0) ^ 0x3030_3030_3030_3030;
    let lo = ((w & 0x0F0F_0F0F_0F0F_0F0F) + 0x0606_0606_0606_0606) & 0xF0F0_F0F0_F0F0_F0F0;
    (hi | lo) & lanes == 0
}

/// Lane i of the result (as a byte) = 10 * digit_i + digit_{i+1}.
#[inline(always)]
fn pairs(w: u64) -> u64 {
    let v = w & 0x0F0F_0F0F_0F0F_0F0F;
    v.wrapping_mul(10).wrapping_add(v >> 8)
}

#[inline(always)]
fn lane(w: u64, i: u32) -> u32 {
    ((w >> (8 * i)) & 0xFF) as u32
}

/// 8 ASCII digits (first char in the lowest byte) to their value.
#[inline(always)]
fn parse8(w: u64) -> u64 {
    let l = w.wrapping_sub(0x3030_3030_3030_3030);
    let c = (l.wrapping_mul(10).wrapping_add(l >> 8)) & 0x00FF_00FF_00FF_00FF;
    let c = (c.wrapping_mul(100).wrapping_add(c >> 16)) & 0x0000_FFFF_0000_FFFF;
    (c.wrapping_mul(10000).wrapping_add(c >> 32)) & 0xFFFF_FFFF
}

/// Timestamp parser with a cache of epoch days for a range of whole months,
/// so the common case needs no calendar arithmetic. Results are identical to
/// `parse_ts` for every input (the cache only covers whole months, and dates
/// outside it use the general path).
pub struct TsParser {
    base_month: i64,
    nmonths: i64,
    /// [month * 32 + day] -> epoch day, or i32::MIN for an impossible date.
    days: Vec<i32>,
}

const DIGIT_LANES_W0: u64 = 0x00FF_FF00_FFFF_FFFF; // YYYY-MM-  lanes 0-3, 5-6
const DIGIT_LANES_W1: u64 = 0xFFFF_00FF_FF00_FFFF; // DDTHH:MM  lanes 0-1, 3-4, 6-7
const SEP_MASK_W0: u64 = 0xFF00_00FF_0000_0000;
const SEP_W0: u64 = ((b'-' as u64) << 56) | ((b'-' as u64) << 32);
const SEP_MASK_W1: u64 = 0x0000_FF00_00FF_0000;
const SEP_W1: u64 = ((b':' as u64) << 40) | ((b'T' as u64) << 16);
// w2 = bytes 12..20: lane 4 ':' (16), lanes 5-6 digits (17, 18)
const DIGIT_LANES_W2: u64 = 0x00FF_FF00_0000_0000;
const SEP_MASK_W2: u64 = 0x0000_00FF_0000_0000;
const SEP_W2: u64 = (b':' as u64) << 32;

impl TsParser {
    /// Caches every month touching [from_ms, to_ms].
    pub fn new(from_ms: i64, to_ms: i64) -> TsParser {
        let civil_month = |ms: i64| -> i64 {
            // months since year 0 of the UTC date of ms
            let days = ms.div_euclid(86_400_000);
            let (y, m) = civil_year_month(days);
            y * 12 + (m - 1)
        };
        let lo = civil_month(from_ms).max(0);
        let hi = civil_month(to_ms).min(9999 * 12 + 11);
        let nmonths = (hi - lo + 1).max(0);
        let mut days = vec![i32::MIN; (nmonths * 32) as usize];
        for mi in 0..nmonths {
            let ym = lo + mi;
            let (y, m) = (ym / 12, ym % 12 + 1);
            let dim = DIM[(m - 1) as usize] as i64 + (m == 2 && is_leap(y as u32)) as i64;
            for d in 1..=dim {
                days[(mi * 32 + d) as usize] = days_from_civil(y, m, d) as i32;
            }
        }
        TsParser { base_month: lo, nmonths, days }
    }

    /// Fields 0..19 of an F1/F2 form, `n` = 20 or 25 (already checked).
    #[inline(always)]
    fn civil(&self, b: &[u8]) -> Option<i64> {
        let w0 = load8(b, 0);
        let w1 = load8(b, 8);
        let w2 = load8(b, 12);
        if (w0 & SEP_MASK_W0) != SEP_W0 || (w1 & SEP_MASK_W1) != SEP_W1 || (w2 & SEP_MASK_W2) != SEP_W2 {
            return None;
        }
        if !digits_ok(w0, DIGIT_LANES_W0) || !digits_ok(w1, DIGIT_LANES_W1) || !digits_ok(w2, DIGIT_LANES_W2) {
            return None;
        }
        let p0 = pairs(w0);
        let p1 = pairs(w1);
        let p2 = pairs(w2);
        let y = lane(p0, 0) * 100 + lane(p0, 2);
        let mo = lane(p0, 5);
        let d = lane(p1, 0);
        let h = lane(p1, 3);
        let mi = lane(p1, 6);
        let s = lane(p2, 5);
        if mo.wrapping_sub(1) >= 12 || d.wrapping_sub(1) >= 31 || h > 23 || mi > 59 || s > 59 {
            return None;
        }
        let month = (y as i64) * 12 + (mo as i64 - 1) - self.base_month;
        let days = if (month as u64) < self.nmonths as u64 {
            let v = self.days[(month * 32 + d as i64) as usize];
            if v == i32::MIN {
                return None;
            }
            v as i64
        } else {
            let dim = DIM[(mo - 1) as usize] + (mo == 2 && is_leap(y)) as u32;
            if d > dim {
                return None;
            }
            days_from_civil(y as i64, mo as i64, d as i64)
        };
        Some(((days * 24 + h as i64) * 60 + mi as i64) * 60_000 + s as i64 * 1000)
    }

    #[inline(always)]
    pub fn parse(&self, b: &[u8]) -> Option<i64> {
        let n = b.len();
        if n == 20 {
            if b[19] == b'Z' {
                return self.civil(b);
            }
        } else if n == 13 {
            let wa = load8(b, 0);
            let wb = load8(b, 5);
            if digits_ok(wa, !0) && digits_ok(wb, !0) {
                let tail = (wb & !0xFF_FFFF) | 0x30_3030;
                return Some((parse8(wa) * 100_000 + parse8(tail)) as i64);
            }
            return None;
        } else if n == 25 {
            let sign = b[19];
            if (sign == b'+' || sign == b'-') && b[22] == b':' {
                let base = self.civil(b)?;
                let w3 = load8(b, 17); // lanes 3-4 HH, 5 ':', 6-7 MM
                if !digits_ok(w3, 0xFFFF_00FF_FF00_0000) {
                    return None;
                }
                let p3 = pairs(w3);
                let (oh, om) = (lane(p3, 3), lane(p3, 6));
                if oh > 23 || om > 59 {
                    return None;
                }
                let off = (oh * 60 + om) as i64 * 60_000;
                return Some(if sign == b'+' { base - off } else { base + off });
            }
        }
        parse_ts(b)
    }
}

fn civil_year_month(z: i64) -> (i64, i64) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + (m <= 2) as i64, m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forms() {
        assert_eq!(parse_ts(b"2026-10-03T23:58:10Z"), Some(1791071890000));
        assert_eq!(parse_ts(b"2026-10-03T19:58:10-04:00"), Some(1791071890000));
        assert_eq!(parse_ts(b"1791071890000"), Some(1791071890000));
        assert_eq!(parse_ts(b"0001791071890000"), Some(1791071890000));
        assert_eq!(parse_ts(b"2026-10-04T01:30:00+02:00"), parse_ts(b"2026-10-03T23:30:00Z"));
        assert_eq!(parse_ts(b"2026-10-03T23:30:00-00:00"), parse_ts(b"2026-10-03T23:30:00Z"));
        for bad in [
            &b""[..], b"2026-10-03T23:58:10.000Z", b"2026-10-03T23:58:10z", b"2026-10-03t23:58:10Z",
            b"2026-10-03 23:58:10Z", b"2026-10-03T23:58:10", b"1791071890000.0", b"1.79e12", b"-5",
            b"2026-13-03T23:58:10Z", b"2026-02-30T23:58:10Z", b"2027-02-29T23:58:10Z", b"2026-10-03T24:00:00Z",
            b"2026-10-03T23:58:60Z", b"2026-10-03T23:58:10+24:00", b"2026-10-03T23:58:10+05:60",
            b"2026-10-03T23:58:10+0200", b"179102160000000000000", b"\xd9\xa1", b"2026-10-03T23:58:10Z\r",
        ] {
            assert_eq!(parse_ts(bad), None, "{:?}", String::from_utf8_lossy(bad));
        }
        assert!(parse_ts(b"2028-02-29T00:00:00Z").is_some());
        assert_eq!(parse_ts(b"86400000"), Some(86_400_000));
        assert_eq!(parse_ts(b"00000000000000000000000000001"), Some(1));
        assert_eq!(parse_ts(b"9223372036854775807"), Some(i64::MAX));
        assert_eq!(parse_ts(b"9223372036854775808"), None);
    }


    #[test]
    fn fast_matches_general() {
        // xorshift fuzz around a window, plus mutations of valid forms
        let ws = parse_ts(b"2028-02-20T00:00:00Z").unwrap();
        let p = TsParser::new(ws - 10 * 86_400_000, ws + 24 * 86_400_000);
        let mut x: u64 = 0x9E3779B97F4A7C15;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let alphabet = b"0123456789-+:TZtz .e";
        let mut count = 0;
        for _ in 0..3_000_000 {
            let r = next();
            let ms = ws + (r % (60 * 86_400_000)) as i64 - 20 * 86_400_000;
            let days = ms.div_euclid(86_400_000);
            let (y, m) = civil_year_month(days);
            let dom = days - days_from_civil(y, m, 1) + 1;
            let sod = ms.rem_euclid(86_400_000) / 1000;
            let mut s = match r % 4 {
                0 => format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, m, dom, sod / 3600, sod / 60 % 60, sod % 60),
                1 => format!(
                    "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}{}{:02}:{:02}",
                    y, m, dom, sod / 3600, sod / 60 % 60, sod % 60,
                    if r & 64 == 0 { '+' } else { '-' }, (r >> 8) % 26, (r >> 16) % 62
                ),
                2 => format!("{}", ms),
                _ => format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", 1990 + (r >> 9) % 100, (r >> 20) % 14,
                             (r >> 30) % 33, (r >> 40) % 26, (r >> 45) % 62, (r >> 50) % 62),
            }
            .into_bytes();
            // random mutation
            if (r >> 56) % 3 == 0 && !s.is_empty() {
                let at = (next() as usize) % s.len();
                s[at] = alphabet[(next() as usize) % alphabet.len()];
            }
            if (r >> 58) % 7 == 0 {
                s.truncate((next() as usize) % (s.len() + 1));
            }
            assert_eq!(p.parse(&s), parse_ts(&s), "{:?}", String::from_utf8_lossy(&s));
            count += p.parse(&s).is_some() as u32;
        }
        assert!(count > 1_000_000);
        for y in [0u32, 1, 1899, 1970, 2000, 2024, 2100, 9999] {
            for m in 1..=13u32 {
                for d in 0..=32u32 {
                    let s = format!("{:04}-{:02}-{:02}T12:34:56Z", y, m, d);
                    assert_eq!(p.parse(s.as_bytes()), parse_ts(s.as_bytes()), "{}", s);
                }
            }
        }
    }

    #[test]
    fn prices() {
        assert_eq!(parse_price(b"0250"), Some(250));
        assert_eq!(parse_price(b"10000000"), Some(10_000_000));
        assert_eq!(parse_price(b"00000000000010000000"), Some(10_000_000));
        assert_eq!(parse_price(b"1"), Some(1));
        for bad in [&b""[..], b"0", b"00", b"10000001", b"250000000", b"19.99", b"-5", b"+500", b"1e3", b"1000\r"] {
            assert_eq!(parse_price(bad), None);
        }
    }
}

#[inline(never)]
#[no_mangle]
pub fn ss_probe_parse(p: &TsParser, b: &[u8]) -> Option<i64> {
    p.parse(b)
}
