//! CSV dialect of rules.md: `\n` ends a line, `,` always separates fields,
//! a field may be wrapped in double quotes (syntax, stripped), any other quote
//! makes the line malformed, a wrong field count makes it malformed.
//!
//! Lines are scanned 64 bytes at a time with SIMD masks for `,` `\n` `"`.

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn simd_masks(p: *const u8) -> (u64, u64, u64) {
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
unsafe fn simd_masks(p: *const u8) -> (u64, u64, u64) {
    use std::arch::aarch64::*;
    #[inline(always)]
    unsafe fn movemask(a: uint8x16_t, b: uint8x16_t, c: uint8x16_t, d: uint8x16_t) -> u64 {
        const BITS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        let m = vld1q_u8(BITS.as_ptr());
        let a = vandq_u8(a, m);
        let b = vandq_u8(b, m);
        let c = vandq_u8(c, m);
        let d = vandq_u8(d, m);
        let s0 = vpaddq_u8(a, b);
        let s1 = vpaddq_u8(c, d);
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
unsafe fn simd_masks(p: *const u8) -> (u64, u64, u64) {
    let mut c = 0u64;
    let mut n = 0u64;
    let mut q = 0u64;
    for i in 0..64 {
        let b = *p.add(i);
        c |= ((b == b',') as u64) << i;
        n |= ((b == b'\n') as u64) << i;
        q |= ((b == b'"') as u64) << i;
    }
    (c, n, q)
}

/// Masks of the 64 bytes at `pos`, with bits at or beyond `end` cleared.
#[inline(always)]
fn masks(buf: &[u8], pos: usize, end: usize) -> (u64, u64, u64) {
    let (c, n, q) = if pos + 64 <= buf.len() {
        unsafe { simd_masks(buf.as_ptr().add(pos)) }
    } else {
        let mut tmp = [0u8; 64];
        let k = buf.len() - pos;
        tmp[..k].copy_from_slice(&buf[pos..]);
        unsafe { simd_masks(tmp.as_ptr()) }
    };
    let rem = end - pos;
    if rem < 64 {
        let m = (1u64 << rem) - 1;
        (c & m, n & m, q & m)
    } else {
        (c, n, q)
    }
}

/// Calls `f` with the unquoted fields of every well-formed line of exactly
/// `N` fields in `buf[start..end]`. `start` must be a line start and `end` a
/// line start or `buf.len()` (a final line without `\n` is still a line).
/// An empty field is passed as an empty slice (null).
///
/// Quote rule: a field is quoted iff its first byte is `"`; it must then be at
/// least 2 bytes long and end with `"`. The line is well-formed iff all quoted
/// fields satisfy that and the line holds exactly 2 quotes per quoted field
/// (so no quote appears anywhere else).
///
/// Two stages per 4 KiB window: (1) SIMD masks, with the positions of all
/// delimiters and of all newlines flattened into arrays by fixed-width
/// unrolled loops; (2) one iteration per line, branch-free field slicing.
#[inline(always)]
pub fn for_each_record<const N: usize, F: FnMut(&[&[u8]; N])>(buf: &[u8], start: usize, end: usize, mut f: F) {
    assert!(end <= buf.len() && end - start < (u32::MAX as usize) - 8192);
    const WBLK: usize = 64;
    const WIN: usize = WBLK * 64;
    let r = &buf[start..];
    let rlen = end - start;
    // carry (<= N + 1) + up to WIN delimiters + flatten overrun
    let mut dl = [0u32; WIN + 96];
    let mut nlp = [0u32; WIN + 16];
    let mut qmask = [0u64; WBLK];
    let mut qpre = [0u32; WBLK + 1];
    let mut carry = 0usize;
    let mut ls: usize = 0; // current line start (relative)
    let mut q_ls: u64 = 0; // quotes before ls
    let mut qbase: u64 = 0; // quotes before the window
    let mut w = 0usize;
    while w < rlen {
        let wend = (w + WIN).min(rlen);
        let nblk = (wend - w).div_ceil(64);
        let mut nd = carry;
        let mut nn = 0usize;
        for b in 0..nblk {
            let p = w + 64 * b;
            let (cm, nl, qm) = masks(buf, start + p, start + wend);
            qmask[b] = qm;
            qpre[b + 1] = qpre[b] + qm.count_ones();
            flatten8(&mut dl, &mut nd, cm | nl, p as u32);
            flatten4(&mut nlp, &mut nn, nl, p as u32);
        }
        let mut k = 0usize;
        for &le in &nlp[..nn] {
            let le = le as usize;
            let off = le - w;
            let bi = off >> 6;
            let q_le = qbase + qpre[bi] as u64 + (qmask[bi] & ((1u64 << (off & 63)) - 1)).count_ones() as u64;
            if k + N <= nd && dl[k + N - 1] as usize == le {
                line::<N, F>(r, ls, &dl[k..k + N], q_le - q_ls, &mut f);
                k += N;
            } else {
                while dl[k] as usize != le {
                    k += 1;
                }
                k += 1;
            }
            ls = le + 1;
            q_ls = q_le;
        }
        let keep = (nd - k).min(N + 1);
        dl.copy_within(k..k + keep, 0);
        carry = keep;
        qbase += qpre[nblk] as u64;
        w = wend;
    }
    if rlen > 0 && end == buf.len() && buf[end - 1] != b'\n' && carry + 1 == N {
        // final line without '\n': its N-1 commas are carried; the end closes it
        line_safe::<N, F>(r, rlen, ls, &dl[..N - 1], qbase - q_ls, &mut f);
    }
}

#[inline(always)]
fn flatten8(out: &mut [u32], n: &mut usize, mut m: u64, base: u32) {
    let cnt = m.count_ones() as usize;
    let mut i = *n;
    let target = i + cnt;
    loop {
        for _ in 0..8 {
            unsafe { *out.get_unchecked_mut(i) = base + m.trailing_zeros() };
            m &= m.wrapping_sub(1);
            i += 1;
        }
        if i >= target {
            break;
        }
    }
    *n = target;
}

#[inline(always)]
fn flatten4(out: &mut [u32], n: &mut usize, mut m: u64, base: u32) {
    let cnt = m.count_ones() as usize;
    let mut i = *n;
    let target = i + cnt;
    loop {
        for _ in 0..4 {
            unsafe { *out.get_unchecked_mut(i) = base + m.trailing_zeros() };
            m &= m.wrapping_sub(1);
            i += 1;
        }
        if i >= target {
            break;
        }
    }
    *n = target;
}

/// One line whose N delimiter positions (relative to `r`) are in `d`; every
/// d[j] is a real delimiter position inside `r`, so r[s] and r[e - 1] exist.
#[inline(always)]
fn line<const N: usize, F: FnMut(&[&[u8]; N])>(r: &[u8], ls: usize, d: &[u32], qline: u64, f: &mut F) {
    let mut fs: [&[u8]; N] = [&[]; N];
    let mut s = ls;
    let mut nq = 0u64;
    let mut bad = false;
    let p = r.as_ptr();
    for j in 0..N {
        let e = d[j] as usize;
        let first = unsafe { *p.add(s) };
        let last = unsafe { *p.add(e.max(1) - 1) };
        let q = (first == b'"') as usize;
        bad |= (q != 0) & ((e - s < 2) | (last != b'"'));
        nq += 2 * q as u64;
        let a = s + q;
        let len = (e - q).saturating_sub(a);
        fs[j] = unsafe { std::slice::from_raw_parts(p.add(a), len) };
        s = e + 1;
    }
    if !bad && nq == qline {
        f(&fs);
    }
}

/// Same as `line` for the final unterminated line: the last field ends at `rlen`.
fn line_safe<const N: usize, F: FnMut(&[&[u8]; N])>(r: &[u8], rlen: usize, ls: usize, commas: &[u32], qline: u64, f: &mut F) {
    let mut fs: [&[u8]; N] = [&[]; N];
    let mut s = ls;
    let mut nq = 0u64;
    for j in 0..N {
        let e = if j + 1 < N { commas[j] as usize } else { rlen };
        let field = &r[s..e];
        if field.first() == Some(&b'"') {
            if field.len() < 2 || field[field.len() - 1] != b'"' {
                return;
            }
            fs[j] = &field[1..field.len() - 1];
            nq += 2;
        } else {
            fs[j] = field;
        }
        s = e + 1;
    }
    if nq == qline {
        f(&fs);
    }
}

/// Position just after the first `\n` at or after `from`, or `buf.len()`.
pub fn next_line_start(buf: &[u8], from: usize) -> usize {
    match buf[from.min(buf.len())..].iter().position(|&b| b == b'\n') {
        Some(i) => from + i + 1,
        None => buf.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect<const N: usize>(text: &[u8]) -> Vec<Vec<Vec<u8>>> {
        let mut out = Vec::new();
        for_each_record::<N, _>(text, 0, text.len(), |f| out.push(f.iter().map(|s| s.to_vec()).collect()));
        out
    }


    fn naive<const N: usize>(text: &[u8]) -> Vec<Vec<Vec<u8>>> {
        let mut out = Vec::new();
        let mut lines: Vec<&[u8]> = text.split(|&b| b == b'\n').collect();
        if text.last() == Some(&b'\n') || text.is_empty() {
            lines.pop();
        }
        'l: for l in lines {
            let parts: Vec<&[u8]> = l.split(|&b| b == b',').collect();
            if parts.len() != N {
                continue;
            }
            let mut row = Vec::new();
            for p in parts {
                if p.contains(&b'"') {
                    if p.len() >= 2 && p[0] == b'"' && p[p.len() - 1] == b'"' && !p[1..p.len() - 1].contains(&b'"') {
                        row.push(p[1..p.len() - 1].to_vec());
                    } else {
                        continue 'l;
                    }
                } else {
                    row.push(p.to_vec());
                }
            }
            out.push(row);
        }
        out
    }

    #[test]
    fn fuzz_against_naive() {
        let mut x: u64 = 12345;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let alpha = b"ab,,\"\"\n \r1";
        for round in 0..20000 {
            let len = (next() % 300) as usize;
            let mut t: Vec<u8> = (0..len).map(|_| alpha[(next() % alpha.len() as u64) as usize]).collect();
            if round % 3 == 0 {
                // mostly well-formed lines with random quoting
                t.clear();
                for _ in 0..(next() % 12) {
                    for j in 0..3 {
                        if j > 0 {
                            t.push(b',');
                        }
                        let fl = (next() % 40) as usize;
                        let q = next() % 4 == 0;
                        if q {
                            t.push(b'"');
                        }
                        for _ in 0..fl {
                            t.push(b"xyz09 "[(next() % 6) as usize]);
                        }
                        if q {
                            t.push(b'"');
                        }
                    }
                    t.push(b'\n');
                }
                if next() % 2 == 0 {
                    t.pop();
                }
            }
            let want3 = naive::<3>(&t);
            assert_eq!(collect::<3>(&t), want3, "{:?}", String::from_utf8_lossy(&t));
            assert_eq!(collect::<2>(&t), naive::<2>(&t), "{:?}", String::from_utf8_lossy(&t));
            // chunked: split at line starts
            let mut got = Vec::new();
            let mut s0 = 0;
            while s0 < t.len() {
                let nominal = s0 + 1 + (next() % 70) as usize;
                let e = if nominal >= t.len() { t.len() } else { next_line_start(&t, nominal - 1) };
                for_each_record::<3, _>(&t, s0, e, |f| got.push(f.iter().map(|s| s.to_vec()).collect::<Vec<_>>()));
                s0 = e;
            }
            assert_eq!(got, want3);
        }
    }


    #[test]
    fn fuzz_large_windows() {
        let mut x: u64 = 987654321;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for _round in 0..300 {
            let mut t = Vec::new();
            let lines = 50 + next() % 400;
            for _ in 0..lines {
                match next() % 10 {
                    0 => {
                        // very long line, many commas, sometimes spanning several windows
                        let n = (next() % 9000) as usize;
                        for _ in 0..n {
                            t.push(if next() % 3 == 0 { b',' } else { b'z' });
                        }
                    }
                    1 => {
                        // long single field crossing windows
                        let n = (next() % 6000) as usize;
                        t.extend(std::iter::repeat(b'y').take(n));
                        t.extend_from_slice(b",a,\"b\"");
                    }
                    2 => {}
                    _ => {
                        for j in 0..3 {
                            if j > 0 {
                                t.push(b',');
                            }
                            let q = next() % 5 == 0;
                            if q {
                                t.push(b'"');
                            }
                            for _ in 0..(next() % 30) {
                                t.push(b"ab\"1"[(next() % if next() % 50 == 0 { 4 } else { 3 }) as usize]);
                            }
                            if q {
                                t.push(b'"');
                            }
                        }
                    }
                }
                t.push(b'\n');
            }
            if next() % 2 == 0 {
                t.pop();
                t.extend_from_slice(b"p,q,\"r\"");
            }
            let want = naive::<3>(&t);
            assert_eq!(collect::<3>(&t), want);
            let mut got = Vec::new();
            let mut s0 = 0;
            while s0 < t.len() {
                let nominal = s0 + 1 + (next() % 20000) as usize;
                let e = if nominal >= t.len() { t.len() } else { next_line_start(&t, nominal - 1) };
                for_each_record::<3, _>(&t, s0, e, |f| got.push(f.iter().map(|s| s.to_vec()).collect::<Vec<_>>()));
                s0 = e;
            }
            assert_eq!(got, want);
        }
    }

    #[test]
    fn dialect() {
        let t = b"u1,u2,u3\nu1,\n,u3\nu1,u5\"\n\"u4\",\"u1\"\nu4,u1\nu6,u6\n\nu1\n\"\",x\n\"a\"b,c\na\"\"b,c\n\"\"\",x\n\" u1\" ,x\nlast,line";
        let got = collect::<2>(t);
        let want: Vec<Vec<&[u8]>> = vec![
            vec![b"u1", b""],
            vec![b"", b"u3"],
            vec![b"u4", b"u1"],
            vec![b"u4", b"u1"],
            vec![b"u6", b"u6"],
            vec![b"", b"x"],
            vec![b"last", b"line"],
        ];
        let want: Vec<Vec<Vec<u8>>> = want.into_iter().map(|r| r.into_iter().map(|s| s.to_vec()).collect()).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn long_lines_and_blocks() {
        // Lines crossing 64-byte blocks, long fields, quotes across blocks.
        let mut t = Vec::new();
        let mut want = Vec::new();
        for i in 0..500 {
            let a = "x".repeat(i % 150);
            let b = format!("{}", i);
            if i % 3 == 0 {
                t.extend_from_slice(format!("\"{}\",{}\n", a, b).as_bytes());
            } else {
                t.extend_from_slice(format!("{},\"{}\"\n", a, b).as_bytes());
            }
            want.push(vec![a.into_bytes(), b.into_bytes()]);
            if i % 7 == 0 {
                t.extend_from_slice(b"bad\"quote,1\n");
            }
        }
        assert_eq!(collect::<2>(&t), want);
    }
}
