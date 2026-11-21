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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn simd_masks_avx2(p: *const u8) -> (u64, u64, u64) {
    use std::arch::x86_64::*;
    let a = _mm256_loadu_si256(p as *const __m256i);
    let b = _mm256_loadu_si256(p.add(32) as *const __m256i);
    let m = |v: __m256i, ch: u8| -> u64 {
        _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, _mm256_set1_epi8(ch as i8))) as u32 as u64
    };
    (
        m(a, b',') | (m(b, b',') << 32),
        m(a, b'\n') | (m(b, b'\n') << 32),
        m(a, b'"') | (m(b, b'"') << 32),
    )
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

#[inline(always)]
unsafe fn simd_masks_x<const AVX2: bool>(p: *const u8) -> (u64, u64, u64) {
    #[cfg(target_arch = "x86_64")]
    {
        if AVX2 {
            return simd_masks_avx2(p);
        }
    }
    simd_masks(p)
}

/// Masks of the 64 bytes at `pos`, with bits at or beyond `end` cleared.
#[inline(always)]
fn masks<const AVX2: bool>(buf: &[u8], pos: usize, end: usize) -> (u64, u64, u64) {
    let (c, n, q) = if pos + 64 <= buf.len() {
        unsafe { simd_masks_x::<AVX2>(buf.as_ptr().add(pos)) }
    } else {
        let mut tmp = [0u8; 64];
        let k = buf.len() - pos;
        tmp[..k].copy_from_slice(&buf[pos..]);
        unsafe { simd_masks_x::<AVX2>(tmp.as_ptr()) }
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
pub fn for_each_record<'a, const N: usize, F: FnMut(&[&'a [u8]; N])>(buf: &'a [u8], start: usize, end: usize, f: F) {
    for_each_record_x::<N, F, false>(buf, start, end, f)
}

/// Experiment switch (SS_SCAN1): use the previous scanner.
pub static USE_V1: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// `for_each_record`, with AVX2 masks when `AVX2` (the caller must run with
/// AVX2 enabled, e.g. inside a `#[target_feature(enable = "avx2")]` function).
#[inline(always)]
pub fn for_each_record_x<'a, const N: usize, F: FnMut(&[&'a [u8]; N]), const AVX2: bool>(
    buf: &'a [u8],
    start: usize,
    end: usize,
    f: F,
) {
    if USE_V1.load(std::sync::atomic::Ordering::Relaxed) {
        for_each_record_x_v1::<N, F, AVX2>(buf, start, end, f)
    } else {
        for_each_record_x_v2::<N, F, AVX2>(buf, start, end, f)
    }
}

/// Inclusive prefix XOR of the bits of x (bit i = parity of bits 0..=i).
#[inline(always)]
fn prefix_xor(mut x: u64) -> u64 {
    x ^= x << 1;
    x ^= x << 2;
    x ^= x << 4;
    x ^= x << 8;
    x ^= x << 16;
    x ^= x << 32;
    x
}

/// Scanner, version 2: delimiters and newlines are flattened as in v1, but
/// quote validity is decided per 64-byte block with masks instead of per
/// field. A line is well-formed iff it has N fields, every quote in it is
/// the first or the last byte of a field, and every field holds an even
/// number of quotes (equivalently: the quote parity counted from the line
/// start is 0 at every delimiter). This is the same rule as v1's "a quoted
/// field is >= 2 bytes and ends with a quote, and the line holds exactly 2
/// quotes per quoted field": under either rule each field holds 0 quotes or
/// exactly 2, at its first and last byte. Blocks get a violation mask; lines
/// touching no violation (almost all) need no per-field quote check, and a
/// field is unquoted by `start += q; end -= q` from its first byte.
#[inline(always)]
pub fn for_each_record_x_v2<'a, const N: usize, F: FnMut(&[&'a [u8]; N]), const AVX2: bool>(
    buf: &'a [u8],
    start: usize,
    end: usize,
    mut f: F,
) {
    assert!(end <= buf.len() && end - start < (u32::MAX as usize) - 8192);
    const WBLK: usize = 64;
    const WIN: usize = WBLK * 64;
    let r = &buf[start..];
    let rlen = end - start;
    let mut dl = [0u32; WIN + 96];
    let mut nlp = [0u32; WIN + 16];
    let mut dmask = [0u64; WBLK];
    let mut nmask = [0u64; WBLK];
    let mut qmask = [0u64; WBLK];
    let mut vmask = [0u64; WBLK];
    let mut carry = 0usize;
    let mut ls: usize = 0; // current line start (relative)
    // block-to-block state
    let mut d_prev = 1u64; // delimiter at the position before the block (start: a line start)
    let mut par = 0u64; // quote parity since the last newline, before the block
    let mut v_open = false; // the unfinished line at the window start had a violation
    let mut w = 0usize;
    while w < rlen {
        let wend = (w + WIN).min(rlen);
        let nblk = (wend - w).div_ceil(64);
        let mut nd = carry;
        let mut nn = 0usize;
        for b in 0..nblk {
            let p = w + 64 * b;
            let (cm, nl, qm) = masks::<AVX2>(buf, start + p, start + wend);
            dmask[b] = cm | nl;
            nmask[b] = nl;
            qmask[b] = qm;
            flatten8(&mut dl, &mut nd, cm | nl, p as u32);
            flatten4(&mut nlp, &mut nn, nl, p as u32);
        }
        // delimiter at the first byte after the window (end of the task: none;
        // a quote there is handled by the final-line path)
        let d_next = if wend < rlen { matches!(r[wend], b',' | b'\n') as u64 } else { 0 };
        let mut any_v = 0u64;
        for b in 0..nblk {
            let d = dmask[b];
            let q = qmask[b];
            let fs = (d << 1) | d_prev; // first byte of a field
            let dn = if b + 1 < nblk { dmask[b + 1] & 1 } else { d_next };
            let fl = (d >> 1) | (dn << 63); // last byte of a field
            let outside = q & !(fs | fl);
            let mut px = prefix_xor(q) ^ 0u64.wrapping_sub(par);
            let mut nm = nmask[b];
            while nm != 0 {
                let j = nm.trailing_zeros();
                let pj = (px >> j) & 1;
                let above = if j == 63 { 0 } else { !0u64 << (j + 1) };
                px ^= 0u64.wrapping_sub(pj) & above;
                nm &= nm - 1;
            }
            let v = outside | (px & d);
            vmask[b] = v;
            any_v |= v;
            par = if (nmask[b] >> 63) & 1 == 1 { 0 } else { px >> 63 };
            d_prev = d >> 63;
        }
        let mut k = 0usize;
        for &le in &nlp[..nn] {
            let le = le as usize;
            if k + N <= nd && dl[k + N - 1] as usize == le {
                let bad = (ls < w && v_open) || (any_v != 0 && has_v(&vmask, w, ls.max(w), le + 1));
                if !bad {
                    line_v2::<N, F>(r, ls, &dl[k..k + N], &mut f);
                }
                k += N;
            } else {
                while dl[k] as usize != le {
                    k += 1;
                }
                k += 1;
            }
            ls = le + 1;
        }
        // the unfinished line (from ls) carries any violation into the next window
        v_open = (ls < w && v_open) || (any_v != 0 && ls < wend && has_v(&vmask, w, ls.max(w), wend));
        let keep = (nd - k).min(N + 1);
        dl.copy_within(k..k + keep, 0);
        carry = keep;
        w = wend;
    }
    if rlen > 0 && end == buf.len() && buf[end - 1] != b'\n' && carry + 1 == N {
        // final line without '\n': its N-1 commas are carried; the end closes it
        let qline = r[ls..rlen].iter().filter(|&&c| c == b'"').count() as u64;
        line_safe::<N, F>(r, rlen, ls, &dl[..N - 1], qline, &mut f);
    }
}

/// Any violation bit at positions [from, to) (relative to the scan start),
/// for a window starting at w.
#[cold]
#[inline(never)]
fn has_v(vmask: &[u64], w: usize, from: usize, to: usize) -> bool {
    let mut pos = from;
    while pos < to {
        let off = pos - w;
        let b = off >> 6;
        let lo = off & 63;
        let hi = (to - w - 64 * b).min(64);
        let m = if hi - lo == 64 { !0u64 } else { ((1u64 << (hi - lo)) - 1) << lo };
        if vmask[b] & m != 0 {
            return true;
        }
        pos = w + 64 * (b + 1);
    }
    false
}

/// One well-formed line (quote rule already checked by masks): unquote each
/// field from its first byte.
#[inline(always)]
fn line_v2<'a, const N: usize, F: FnMut(&[&'a [u8]; N])>(r: &'a [u8], ls: usize, d: &[u32], f: &mut F) {
    let mut fs: [&'a [u8]; N] = [&[]; N];
    let mut s = ls;
    let p = r.as_ptr();
    for j in 0..N {
        let e = d[j] as usize;
        let q = (unsafe { *p.add(s) } == b'"') as usize;
        fs[j] = unsafe { std::slice::from_raw_parts(p.add(s + q), e - s - 2 * q) };
        s = e + 1;
    }
    f(&fs);
}

/// Scanner, version 1 (per-field quote checks).
#[inline(always)]
pub fn for_each_record_x_v1<'a, const N: usize, F: FnMut(&[&'a [u8]; N]), const AVX2: bool>(
    buf: &'a [u8],
    start: usize,
    end: usize,
    mut f: F,
) {
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
            let (cm, nl, qm) = masks::<AVX2>(buf, start + p, start + wend);
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
fn line<'a, const N: usize, F: FnMut(&[&'a [u8]; N])>(r: &'a [u8], ls: usize, d: &[u32], qline: u64, f: &mut F) {
    let mut fs: [&'a [u8]; N] = [&[]; N];
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
fn line_safe<'a, const N: usize, F: FnMut(&[&'a [u8]; N])>(r: &'a [u8], rlen: usize, ls: usize, commas: &[u32], qline: u64, f: &mut F) {
    let mut fs: [&'a [u8]; N] = [&[]; N];
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

    fn collect_x<const N: usize, const V2: bool>(t: &[u8], s0: usize, e: usize, out: &mut Vec<Vec<Vec<u8>>>) {
        let push = |f: &[&[u8]; N]| out.push(f.iter().map(|s| s.to_vec()).collect());
        if V2 {
            for_each_record_x_v2::<N, _, false>(t, s0, e, push)
        } else {
            for_each_record_x_v1::<N, _, false>(t, s0, e, push)
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
    unsafe fn collect_avx2<const N: usize>(t: &[u8], s0: usize, e: usize, out: &mut Vec<Vec<Vec<u8>>>) {
        for_each_record_x_v2::<N, _, true>(t, s0, e, |f: &[&[u8]; N]| out.push(f.iter().map(|s| s.to_vec()).collect()))
    }

    /// v2 agrees with v1 and with the naive splitter on adversarial quote /
    /// newline / comma soup, whole and split into random line-aligned chunks,
    /// with windows crossed by long lines; scalar and (if present) AVX2 masks.
    #[test]
    fn v2_matches_v1_and_naive() {
        let mut x: u64 = 0x5CA9_0000_0000_0001;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        #[cfg(target_arch = "x86_64")]
        let avx2 = std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("bmi2");
        for round in 0..40000u64 {
            let mut t = Vec::new();
            let mode = round % 5;
            let lines = 1 + next() % 120;
            for _ in 0..lines {
                if mode == 4 && next() % 20 == 0 {
                    // long line crossing windows
                    let n = (next() % 9000) as usize;
                    for _ in 0..n {
                        t.push(b"z,\"a"[(next() % 4) as usize]);
                    }
                } else {
                    let nf = 1 + next() % 5;
                    for j in 0..nf {
                        if j > 0 {
                            t.push(b',');
                        }
                        let fl = (next() % 12) as usize;
                        let quoted = next() % 3 == 0;
                        if quoted {
                            t.push(b'"');
                        }
                        for _ in 0..fl {
                            let c = match mode {
                                0 => b"ab\"1 "[(next() % 5) as usize],
                                1 => b"xy"[(next() % 2) as usize],
                                _ => b"q\"\"r,"[(next() % 5) as usize],
                            };
                            t.push(c);
                        }
                        if quoted && next() % 7 != 0 {
                            t.push(b'"');
                        }
                    }
                }
                if next() % 13 != 0 || mode != 3 {
                    t.push(b'\n');
                }
            }
            if next() % 3 == 0 {
                t.pop();
            }
            for nfields in [2usize, 3, 4] {
                let want = match nfields {
                    2 => naive::<2>(&t),
                    3 => naive::<3>(&t),
                    _ => naive::<4>(&t),
                };
                let (mut a, mut b) = (Vec::new(), Vec::new());
                match nfields {
                    2 => {
                        collect_x::<2, true>(&t, 0, t.len(), &mut a);
                        collect_x::<2, false>(&t, 0, t.len(), &mut b);
                    }
                    3 => {
                        collect_x::<3, true>(&t, 0, t.len(), &mut a);
                        collect_x::<3, false>(&t, 0, t.len(), &mut b);
                    }
                    _ => {
                        collect_x::<4, true>(&t, 0, t.len(), &mut a);
                        collect_x::<4, false>(&t, 0, t.len(), &mut b);
                    }
                }
                assert_eq!(a, want, "v2 vs naive N={} {:?}", nfields, String::from_utf8_lossy(&t));
                assert_eq!(b, want, "v1 vs naive N={}", nfields);
                #[cfg(target_arch = "x86_64")]
                if avx2 {
                    let mut c = Vec::new();
                    unsafe {
                        match nfields {
                            2 => collect_avx2::<2>(&t, 0, t.len(), &mut c),
                            3 => collect_avx2::<3>(&t, 0, t.len(), &mut c),
                            _ => collect_avx2::<4>(&t, 0, t.len(), &mut c),
                        }
                    }
                    assert_eq!(c, want, "v2 avx2 vs naive N={}", nfields);
                }
            }
            // chunked
            let want = naive::<3>(&t);
            let mut got = Vec::new();
            let mut s0 = 0;
            while s0 < t.len() {
                let nominal = s0 + 1 + (next() % 3000) as usize;
                let e = if nominal >= t.len() { t.len() } else { next_line_start(&t, nominal - 1) };
                collect_x::<3, true>(&t, s0, e, &mut got);
                s0 = e;
            }
            assert_eq!(got, want, "v2 chunked");
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
