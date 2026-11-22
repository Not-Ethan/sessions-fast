#![cfg_attr(not(target_arch = "x86_64"), allow(dead_code, unused_imports))]
//! k6: the chunk-first ("streaming masks") event-line splitter (SS_K6).
//!
//! k4 splits every line from masks of the 128 bytes at its start (two mask
//! sets per line, a tz/blsr chain per field, 128-bit shifts). k6 indexes the
//! task's bytes once, sequentially, in segments of up to `SEG` bytes that
//! start at a line start (stage 1), and then reads every line's seven field
//! boundaries from that index (stage 2):
//!
//! * stage 1, per 64-byte block (branch-free in the common case): comma,
//!   newline and quote masks; the dialect's quote rule as a mask `bad` of
//!   positions that make their line malformed (see `block`); the positions
//!   of every delimiter (comma or newline) and every `bad` bit, in order,
//!   flattened into `e` (u32, offset from the segment start), and one
//!   flag byte per entry in `fl`: bit 0 = the byte before it is a quote,
//!   bit 1 = it is a newline, bit 2 = it is a `bad` position.
//! * stage 2, per line (its first entry has index k; entry k - 1 is the
//!   previous line's newline, or the virtual entry 0 at the segment start):
//!   the line is simple iff entries k..k+5 are neither newlines nor bad and
//!   entry k+6 is a newline and not bad (one 8-byte load and compare). Then
//!   the line has exactly seven fields, ends at its first newline, and has
//!   no quote except a pair around a whole field (the closing one is flag
//!   bit 0 of the field's terminating entry); field j is
//!   [e[k+j-1] + 1 + q_j, e[k+j] - q_j) (offsets from the segment start),
//!   computed for all seven fields at once in two ymm, written into a
//!   `fastev::Cut` and handed to k4's own `decode_body` (and to k4's
//!   `line_fb` when it is not certain), so timestamps, IDs, validity,
//!   purchases and the sink are k4's. The next line's first entry is k + 7.
//! * every other line (wrong delimiter count, a bad quote, an empty line, a
//!   line not wholly inside the segment's scanned bytes) goes to k4's
//!   `slow_line` as a one-line range (lines starting within `k4::MARGIN`
//!   bytes of the buffer end: the rest of the task to k4's `tail_fb`), in
//!   order; a line cut off by the end of the segment's scanned bytes starts
//!   the next segment.
//!
//! The scan never reads beyond `b` (it stops 64 bytes before its end, and
//! at the task end), so every field start of a simple line has 64 readable
//! bytes after it (the decode reads 32).
//!
//! The quote rule as masks (the dialect: a field is quoted iff its first
//! byte is a quote; then its last byte is a quote and its length is at
//! least 2; no other quote anywhere in the line). With D the delimiters, FS
//! the field starts (D << 1, and the segment start), Qs the quotes moved one
//! byte up (bit i = byte i-1 is a quote), FSs = FS << 1, op = the quotes at
//! field starts and Od the delimiters that end a field starting with a quote
//! (op carried through the field by an add to !D, across blocks):
//! * Qs & !FSs & !D: a quote at i that is not a field's first byte and is
//!   not followed by a delimiter (flagged at i + 1, inside the same line);
//! * D & Qs & FSs: a field that is a single quote;
//! * D & (Od ^ Qs): a field that starts with a quote but does not end with
//!   one, or ends with a quote but does not start with one.
//! A line with none of these at a position in it has every quote as the
//! first or the last byte of a field of length >= 2 that starts and ends
//! with one: exactly the rule (the checker and the tests compare against
//! the old path).

use std::mem::MaybeUninit;
use std::ptr::addr_of_mut;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::{Ctx, EvSink, Input, Task};
use crate::k4::{self, Stats};
use crate::{fastev, scan};

/// event tasks through k6 (SS_K6; needs AVX2, see `main::event_task`).
pub static ON: AtomicBool = AtomicBool::new(false);
/// Count lines per outcome (the checker; off in timed runs).
pub static COUNT: AtomicBool = AtomicBool::new(false);
pub static N_FAST: AtomicUsize = AtomicUsize::new(0);
pub static N_LINE: AtomicUsize = AtomicUsize::new(0);
pub static N_RANGE: AtomicUsize = AtomicUsize::new(0);
pub static N_TAIL: AtomicUsize = AtomicUsize::new(0);
/// Lines k6 handed to k4's `slow_line` (counted while `COUNT` is set).
pub static N_SLOW: AtomicUsize = AtomicUsize::new(0);

/// 64-byte blocks per segment.
const NB: usize = 64;
/// Bytes per segment.
const SEG: usize = NB * 64;
/// Entries of a segment: the virtual entry, at most one per byte, the
/// unconditional writes past the last (7 entries, 8 flag bytes) and the
/// sentinels (16).
const CAP: usize = SEG + 64;
/// Sentinel entries after the last real one (flag `F_BAD`).
const NSENT: usize = 16;
/// Byte offset of `Ring::fl` from `Ring::e`.
const FL_OFF: usize = 4 * CAP;
/// Stage 1 software prefetch distance (bytes ahead of the block).
const PFD: usize = 1024;

/// Flag bits of an entry (see STATUS-k6.md, "Entry format"): the byte
/// before it is a quote / it is a newline / it is a bad position.
#[allow(dead_code)]
pub(crate) const F_Q: u8 = 1;
pub(crate) const F_NL: u8 = 2;
pub(crate) const F_BAD: u8 = 4;
const L1: u64 = 0x0101_0101_0101_0101;
/// Flag bytes of entries k..k+7: no newline or bad flag in bytes 0..5, the
/// newline flag (and no bad flag) in byte 6.
pub(crate) const PAT_M: u64 = 0x0006_0606_0606_0606;
pub(crate) const PAT: u64 = 0x0002_0000_0000_0000;

/// The index of one segment.
#[repr(C, align(64))]
pub(crate) struct Ring {
    pub(crate) e: [u32; CAP],
    pub(crate) fl: [u8; CAP],
}

/// Stage 1 state carried from block to block.
struct S1 {
    /// previous block's delimiters and quotes; borrow of `od` (1 - carry)
    dp: u64,
    qp: u64,
    bin: u8,
    /// next entry index
    kb: usize,
}

#[inline(always)]
unsafe fn rd64(p: *const u8) -> u64 {
    u64::from_le(std::ptr::read_unaligned(p as *const u64))
}


// ---------------------------------------------------------------- stage 1

/// One block: 64 bytes at p + 64 * blk (bytes outside `vm` count as
/// ordinary bytes). Appends its entries and flags at `st.kb`.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn block(st: &mut S1, q: *const u8, base: u32, vm: u64, e: *mut u32) {
    // the flags live at a fixed offset from the entries (one base register)
    let fl = (e as *mut u8).add(FL_OFF);
    let a = _mm256_loadu_si256(q as *const __m256i);
    let b = _mm256_loadu_si256(q.add(32) as *const __m256i);
    let mm = |v: __m256i, ch: u8| -> u64 { _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, _mm256_set1_epi8(ch as i8))) as u32 as u64 };
    let c = (mm(a, b',') | (mm(b, b',') << 32)) & vm;
    let n = (mm(a, b'\n') | (mm(b, b'\n') << 32)) & vm;
    let qq = (mm(a, b'"') | (mm(b, b'"') << 32)) & vm;
    let d = c | n;
    let fs = (d << 1) | (st.dp >> 63);
    let qs = (qq << 1) | (st.qp >> 63);
    // fs << 1 with the previous block's fs bit 63 (= its d bit 62)
    let fss = (d << 2) | (st.dp >> 62);
    let op = qq & fs;
    // od: op carried through its field to the delimiter that ends it, i.e.
    // (!d + op + carry) & d. As op - d - borrow (the same sum mod 2^64; the
    // borrow out is the inverse of the carry out, so the state is the borrow).
    let mut sum = 0u64;
    let bout = _subborrow_u64(st.bin, op, d, &mut sum);
    let od = sum & d;
    // bad (see the header): d & qs & fss (a one-byte quoted field) and
    // !d & qs & !fss (an inner quote), together qs & !(fss ^ d); and at a
    // delimiter od ^ qs (a field that starts with a quote xor ends with one)
    let bad = (qs & !(fss ^ d)) | (d & (od ^ qs));
    st.dp = d;
    st.qp = qq;
    st.bin = bout;
    let kb = st.kb;
    let cnt = d.count_ones() as usize;
    // at most 7 delimiters in a block: 99.99% of dev5's blocks (8 or more, or a
    // bad position: the exact loop)
    if (bad != 0) | (cnt > 7) {
        st.kb = kb + block_slow(d | bad, qs, n, bad, base, e.add(kb), fl.add(kb));
        return;
    }
    let ep = e.add(kb);
    let mut f = d;
    for i in 0..7 {
        *ep.add(i) = base + f.trailing_zeros();
        f &= f.wrapping_sub(1);
    }
    // flag bytes: bit 0 from qs, bit 1 from n (one constant; the add is a lea)
    let fl8 = _pdep_u64(_pext_u64(qs, d), L1).wrapping_add(_pdep_u64(_pext_u64(n, d), L1).wrapping_mul(2));
    std::ptr::write_unaligned(fl.add(kb) as *mut u64, fl8.to_le());
    st.kb = kb + cnt;
}

/// A block with more than 7 entries or a bad position: every entry of `f`
/// (exact count), flags in groups of 8. Returns the entry count.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[cold]
#[inline(never)]
unsafe fn block_slow(f: u64, qs: u64, n: u64, bad: u64, base: u32, ep: *mut u32, flp: *mut u8) -> usize {
    let cnt = f.count_ones() as usize;
    let mut x = f;
    let mut i = 0;
    while x != 0 {
        *ep.add(i) = base + x.trailing_zeros();
        x &= x - 1;
        i += 1;
    }
    let fq = _pext_u64(qs, f);
    let fnl = _pext_u64(n, f);
    let fb = _pext_u64(bad, f);
    let mut g = 0;
    while g < cnt {
        let v = _pdep_u64(fq >> g, L1) | _pdep_u64(fnl >> g, L1 << 1) | _pdep_u64(fb >> g, L1 << 2);
        std::ptr::write_unaligned(flp.add(g) as *mut u64, v.to_le());
        g += 8;
    }
    cnt
}

/// Stage 1 over the `nb` blocks at p (the segment; p is a line start; bytes
/// of the last block outside `lastmask` are ignored). Returns the number of
/// entries including the virtual entry 0 (the newline before p, offset -1);
/// sentinels follow. Out of line: its loop gets every register.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[inline(never)]
pub(crate) unsafe fn stage1(p: *const u8, nb: usize, lastmask: u64, e: *mut u32) -> usize {
    let fl = (e as *mut u8).add(FL_OFF);
    *e = u32::MAX;
    *fl = F_NL;
    // the byte before p ends a line: a delimiter there, no quote, no carry
    let mut st = S1 { dp: 1 << 63, qp: 0, bin: 1, kb: 1 };
    let mut q = p;
    let mut base = 0u32;
    // the nb - 1 full blocks, then the last (masked) block (unrolling x2
    // measured no fewer instructions)
    let qend = p.add((nb - 1) * 64);
    while q < qend {
        super::prefetch(q.wrapping_add(PFD));
        block(&mut st, q, base, !0u64, e);
        q = q.add(64);
        base += 64;
    }
    block(&mut st, q, base, lastmask, e);
    let kb = st.kb;
    let big = (nb * 64) as u32;
    for i in 0..NSENT {
        *e.add(kb + i) = big;
        *fl.add(kb + i) = F_BAD;
    }
    kb
}

// ---------------------------------------------------------------- stage 2

/// The line whose first entry is k is simple (7 fields, quotes only around
/// whole fields, wholly inside the scanned bytes). k <= the entry count.
#[inline(always)]
pub(crate) unsafe fn is_simple(fl: *const u8, k: usize) -> bool {
    rd64(fl.add(k)) & PAT_M == PAT
}

/// A batch of simple lines from entry k: returns k' = k + 7m, m <= max,
/// such that the lines k, k+7, ..., k'-7 are simple and (if m < max) the
/// line k' is not. For a batch decode reading the entries directly.
#[allow(dead_code)]
#[inline(always)]
pub(crate) unsafe fn simple_run(fl: *const u8, k: usize, max: usize) -> usize {
    let mut j = k;
    let end = k + 7 * max;
    while j < end && is_simple(fl, j) {
        j += 7;
    }
    j
}

/// The fields of the simple line whose first entry is k, into `*c` (a
/// `fastev::Cut` relative to the segment start): s_j = e[k+j-1] + 1 + q_j,
/// n_j = e[k+j] - s_j - q_j (q_j = flag bit 0 of entry k+j), lanes 0..3 and
/// 3..6. Entries are sign-extended (the virtual entry is -1).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn cut_into(c: *mut fastev::Cut, e: *const u32, fl: *const u8, k: usize) {
    let ld = |i: usize| _mm256_cvtepi32_epi64(_mm_loadu_si128(e.add(i) as *const __m128i));
    let one = _mm256_set1_epi64x(1);
    // fields 0..3: flags of entries k..k+3 are 0 or 1 (simple line)
    let alo = ld(k - 1);
    let blo = ld(k);
    let qlo = _mm256_cvtepu8_epi64(_mm_loadl_epi64(fl.add(k) as *const __m128i));
    // fields 3..6: entry k+6 carries the newline flag
    let ahi = ld(k + 2);
    let bhi = ld(k + 3);
    let qhi = _mm256_and_si256(_mm256_cvtepu8_epi64(_mm_loadl_epi64(fl.add(k + 3) as *const __m128i)), one);
    let slo = _mm256_add_epi64(_mm256_add_epi64(alo, qlo), one);
    let shi = _mm256_add_epi64(_mm256_add_epi64(ahi, qhi), one);
    let nlo = _mm256_sub_epi64(_mm256_sub_epi64(blo, slo), qlo);
    let nhi = _mm256_sub_epi64(_mm256_sub_epi64(bhi, shi), qhi);
    let sp = addr_of_mut!((*c).s) as *mut usize;
    let np = addr_of_mut!((*c).n) as *mut usize;
    _mm256_storeu_si256(sp as *mut __m256i, slo);
    _mm256_storeu_si256(sp.add(3) as *mut __m256i, shi);
    _mm256_storeu_si256(np as *mut __m256i, nlo);
    _mm256_storeu_si256(np.add(3) as *mut __m256i, nhi);
}

// ---------------------------------------------------------------- task

/// One event task through k6 (same copies, same order as
/// `fastev::event_task_fast` + `event_line`).
///
/// # Safety
/// The CPU must support AVX2, BMI1, BMI2, LZCNT and POPCNT; `safe_end` as
/// for `fastev::event_task_fast`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[inline]
pub unsafe fn event_task_k6<S: EvSink>(b: &[u8], task: Task, ctx: &Ctx, sink: &mut S, safe_end: usize) {
    let st = if k4::PACK2.load(Ordering::Relaxed) { task_k6::<S, true>(b, task, ctx, sink, safe_end) } else { task_k6::<S, false>(b, task, ctx, sink, safe_end) };
    if COUNT.load(Ordering::Relaxed) {
        N_FAST.fetch_add(st.fast, Ordering::Relaxed);
        N_LINE.fetch_add(st.line_fb, Ordering::Relaxed);
        N_RANGE.fetch_add(st.range_fb, Ordering::Relaxed);
        N_TAIL.fetch_add(st.tail, Ordering::Relaxed);
    }
}

/// One task (`pack2` iff `P2`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[inline]
unsafe fn task_k6<S: EvSink, const P2: bool>(b: &[u8], task: Task, ctx: &Ctx, sink: &mut S, safe_end: usize) -> Stats {
    let mut st = Stats::default();
    let (start, end) = (task.start, task.end);
    assert!(start <= end && end <= b.len() && safe_end >= b.as_ptr() as usize + b.len());
    if start >= end {
        return st;
    }
    if b.len() < k4::MARGIN {
        st.tail += 1;
        k4::tail_fb(b, task, ctx, sink, safe_end);
        return st;
    }
    let kp = k4::params(ctx);
    // last line start for slow_line (its cuts read up to k4::MARGIN bytes)
    let lim = b.len() - k4::MARGIN;
    // the scan stops 64 bytes before the end of b
    let scan_lim = b.len() - 64;
    let base = b.as_ptr();
    let mut ring = MaybeUninit::<Ring>::uninit();
    let e = addr_of_mut!((*ring.as_mut_ptr()).e) as *mut u32;
    let fl = addr_of_mut!((*ring.as_mut_ptr()).fl) as *mut u8;
    debug_assert_eq!(fl as usize - e as usize, FL_OFF);
    // the cut of the current line (written by cut_into, read by the decode)
    let mut cut = fastev::Cut { ok: true, s: [0; 7], n: [0; 7], len: 0 };
    let cp: *mut fastev::Cut = &mut cut;
    let mut off = start;
    let mut slow = 0usize;
    'seg: while off < end && off <= lim {
        // a segment from the line start `off`
        let a = off;
        let s_end = end.min(scan_lim).min(a + SEG);
        let nbytes = s_end - a;
        let nb = nbytes.div_ceil(64);
        let r = nbytes & 63;
        let lastmask = if r == 0 { !0u64 } else { (1u64 << r) - 1 };
        let cnt = stage1(base.add(a), nb, lastmask, e);
        let p = base.add(a);
        let mut k = 1usize;
        loop {
            if is_simple(fl, k) {
                cut_into(cp, e, fl, k);
                if k4::decode_body::<S, P2>(kp, ctx, p, &*cp, sink) {
                    st.fast += 1;
                } else {
                    st.line_fb += 1;
                    k4::line_fb(p, &*cp, ctx, sink);
                }
                k += 7;
                continue;
            }
            // not simple: the line's newline among the entries, if any
            let ls = a + (*e.add(k - 1)).wrapping_add(1) as usize;
            let mut j = k;
            while j < cnt && *fl.add(j) & F_NL == 0 {
                j += 1;
            }
            if j < cnt {
                if ls > lim {
                    off = ls;
                    break 'seg;
                }
                slow += 1;
                let nx = k4::slow_line::<S, P2>(b, ls, task.file, kp, ctx, sink, safe_end, &mut st);
                debug_assert_eq!(nx, a + *e.add(j) as usize + 1);
                let _ = nx;
                k = j + 1;
            } else {
                // the line runs past the scanned bytes: the next segment
                // starts at it, unless it is the segment's first line (a
                // line longer than the scan, or at the end of b)
                off = if ls == a {
                    slow += 1;
                    k4::slow_line::<S, P2>(b, ls, task.file, kp, ctx, sink, safe_end, &mut st)
                } else {
                    ls
                };
                break;
            }
        }
    }
    if off < end {
        st.tail += 1;
        k4::tail_fb(b, Task { file: task.file, start: off, end }, ctx, sink, safe_end);
    }
    if COUNT.load(Ordering::Relaxed) {
        N_SLOW.fetch_add(slow, Ordering::Relaxed);
    }
    st
}

/// `event_task_k6` with a run-time check (checks and tests): None without AVX2.
pub fn task_any<S: EvSink>(b: &[u8], task: Task, ctx: &Ctx, sink: &mut S, safe_end: usize) -> Option<Stats> {
    task_any_v(b, task, ctx, sink, safe_end, k4::PACK2.load(Ordering::Relaxed))
}

/// `task_any` with `pack2` iff `p2` (tests).
pub fn task_any_v<S: EvSink>(b: &[u8], task: Task, ctx: &Ctx, sink: &mut S, safe_end: usize, p2: bool) -> Option<Stats> {
    #[cfg(target_arch = "x86_64")]
    {
        if ctx.avx2 {
            return Some(unsafe { task_k6_avx2(b, task, ctx, sink, safe_end, p2) });
        }
    }
    let _ = (b, task, ctx, sink, safe_end, p2);
    None
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[inline(never)]
unsafe fn task_k6_avx2<S: EvSink>(b: &[u8], task: Task, ctx: &Ctx, sink: &mut S, safe_end: usize, p2: bool) -> Stats {
    if p2 {
        task_k6::<S, true>(b, task, ctx, sink, safe_end)
    } else {
        task_k6::<S, false>(b, task, ctx, sink, safe_end)
    }
}

// ---------------------------------------------------------------- checking

fn show(b: &[u8]) -> String {
    let s = b.escape_ascii().to_string();
    if s.len() > 400 {
        format!("{}...[{} bytes]", &s[..400], b.len())
    } else {
        s
    }
}

/// SS_K6CHECK=1: every event task through the old path
/// (`fastev::event_task_fast` + `event_line`) and through k6 into recording
/// sinks; the copy() calls must be identical, argument by argument, in the
/// same order. Prints the line counts per outcome; exits nonzero on any
/// difference (after showing the first lines that differ on their own).
pub fn check(ev: &Input, ctx: &Ctx) {
    if !ctx.avx2 {
        eprintln!("k6check: needs AVX2 + BMI1/2 (k6 is never used without them)");
        std::process::exit(1);
    }
    let (mut lines, mut copies, mut valid) = (0usize, 0usize, 0usize);
    let mut tot = Stats::default();
    let mut bad = 0usize;
    for task in &ev.tasks {
        let b = ev.maps[task.file as usize].bytes();
        let safe_end = b.as_ptr() as usize + b.len();
        let mut want = fastev::RecSink::default();
        fastev::task_fast_any(b, *task, ctx, &mut want, safe_end);
        let mut got = fastev::RecSink::default();
        let st = task_any(b, *task, ctx, &mut got, safe_end).unwrap();
        tot.fast += st.fast;
        tot.line_fb += st.line_fb;
        tot.range_fb += st.range_fb;
        tot.tail += st.tail;
        lines += b[task.start..task.end].iter().filter(|&&c| c == b'\n').count();
        copies += want.0.len();
        valid += want.0.iter().filter(|c| c.2.is_some()).count();
        if got.0 != want.0 {
            bad += 1;
            eprintln!("k6check: MISMATCH file {} [{}, {}) ({} k6 vs {} old copies)", task.file, task.start, task.end, got.0.len(), want.0.len());
            let i = got.0.iter().zip(&want.0).position(|(a, b)| a != b).unwrap_or(got.0.len().min(want.0.len()));
            eprintln!("  first differing copy #{}: k6 {:?} old {:?}", i, got.0.get(i), want.0.get(i));
            let mut l = task.start;
            let mut shown = 0;
            while l < task.end && shown < 5 {
                let e = scan::next_line_start(b, l).min(task.end);
                let one = Task { file: task.file, start: l, end: e };
                let mut a = fastev::RecSink::default();
                fastev::task_fast_any(b, one, ctx, &mut a, safe_end);
                let mut g = fastev::RecSink::default();
                task_any(b, one, ctx, &mut g, safe_end);
                if a.0 != g.0 {
                    eprintln!("  line at {}: \"{}\"\n    old: {:?}\n    k6:  {:?}", l, show(&b[l..e]), a.0, g.0);
                    shown += 1;
                }
                l = e;
            }
            if shown == 0 {
                eprintln!("  (no single line differs on its own)");
            }
            if bad >= 3 {
                break;
            }
        }
    }
    let pct = |x: usize| x as f64 * 100.0 / lines.max(1) as f64;
    eprintln!(
        "k6check: {} {} tasks, {} lines, {} copies ({} valid); decoded {} ({:.3}%), event_line fallback {} ({:.3}%), scanner lines {} ({:.4}%), buffer-end ranges {}; lines through slow_line {} ({:.4}%)",
        if bad == 0 { "OK" } else { "FAILED" },
        ev.tasks.len(),
        lines,
        copies,
        valid,
        tot.fast,
        pct(tot.fast),
        tot.line_fb,
        pct(tot.line_fb),
        tot.range_fb,
        pct(tot.range_fb),
        tot.tail,
        N_SLOW.load(Ordering::Relaxed),
        pct(N_SLOW.load(Ordering::Relaxed))
    );
    if bad != 0 {
        std::process::exit(1);
    }
}
