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
//!   flattened into `e` (u32, offset from the segment start + 1), and one
//!   flag byte per entry in `fl`: bit 0 = the byte before it is a quote,
//!   bit 1 = it is a newline, bit 2 = it is a `bad` position.
//! * stage 2, per line (its first entry has index k; entry k - 1 is the
//!   previous line's newline, or the virtual entry 0 at the segment start):
//!   the line is simple iff entries k..k+5 are neither newlines nor bad and
//!   entry k+6 is a newline and not bad (one 8-byte load and compare). Then
//!   the line has exactly seven fields, ends at its first newline, and has
//!   no quote except a pair around a whole field (the closing one is flag
//!   bit 0 of the field's terminating entry); field j is
//!   [e[k+j-1] + q_j, e[k+j] - 1 - q_j) (offsets from the segment start),
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
/// unconditional writes past the last (8) and the sentinels (16).
const CAP: usize = SEG + 64;
/// Sentinel entries after the last real one (flag `F_BAD`).
const NSENT: usize = 16;

const F_NL: u8 = 2;
const F_BAD: u8 = 4;
const L1: u64 = 0x0101_0101_0101_0101;
/// Flag bytes of entries k..k+7: no newline or bad flag in bytes 0..5, the
/// newline flag (and no bad flag) in byte 6.
const PAT_M: u64 = 0x0006_0606_0606_0606;
const PAT: u64 = 0x0002_0000_0000_0000;

/// The index of one segment.
#[repr(C, align(64))]
struct Ring {
    e: [u32; CAP],
    fl: [u8; CAP],
}

/// Stage 1 state carried from block to block.
struct S1 {
    /// previous block's delimiters, quotes, field starts; carry of `od`
    dp: u64,
    qp: u64,
    fsp: u64,
    cin: u64,
    /// next entry index
    kb: usize,
}

#[inline(always)]
unsafe fn rd64(p: *const u8) -> u64 {
    u64::from_le(std::ptr::read_unaligned(p as *const u64))
}

#[inline(always)]
unsafe fn rd32(p: *const u8) -> u32 {
    u32::from_le(std::ptr::read_unaligned(p as *const u32))
}

// ---------------------------------------------------------------- stage 1

/// One block: 64 bytes at p + 64 * blk (bytes outside `vm` count as
/// ordinary bytes). Appends its entries and flags at `st.kb`.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn block(st: &mut S1, p: *const u8, blk: usize, vm: u64, e: *mut u32, fl: *mut u8) {
    let q = p.add(blk * 64);
    let a = _mm256_loadu_si256(q as *const __m256i);
    let b = _mm256_loadu_si256(q.add(32) as *const __m256i);
    let mm = |v: __m256i, ch: u8| -> u64 { _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, _mm256_set1_epi8(ch as i8))) as u32 as u64 };
    let c = (mm(a, b',') | (mm(b, b',') << 32)) & vm;
    let n = (mm(a, b'\n') | (mm(b, b'\n') << 32)) & vm;
    let qq = (mm(a, b'"') | (mm(b, b'"') << 32)) & vm;
    let d = c | n;
    let fs = (d << 1) | (st.dp >> 63);
    let qs = (qq << 1) | (st.qp >> 63);
    let fss = (fs << 1) | (st.fsp >> 63);
    let op = qq & fs;
    let sum = (!d as u128) + (op as u128) + (st.cin as u128);
    let od = (sum as u64) & d;
    let bad = (qs & !fss & !d) | (d & ((od ^ qs) | (qs & fss)));
    st.dp = d;
    st.qp = qq;
    st.fsp = fs;
    st.cin = (sum >> 64) as u64;
    let base1 = (blk * 64 + 1) as u32;
    let kb = st.kb;
    let cnt = d.count_ones() as usize;
    if (bad != 0) | (cnt > 8) {
        st.kb = kb + block_slow(d | bad, qs, n, bad, base1, e.add(kb), fl.add(kb));
        return;
    }
    let ep = e.add(kb);
    let mut f = d;
    for i in 0..8 {
        *ep.add(i) = base1 + f.trailing_zeros();
        f &= f.wrapping_sub(1);
    }
    let fl8 = _pdep_u64(_pext_u64(qs, d), L1) | _pdep_u64(_pext_u64(n, d), L1 << 1);
    std::ptr::write_unaligned(fl.add(kb) as *mut u64, fl8.to_le());
    st.kb = kb + cnt;
}

/// A block with more than 8 entries or a bad position: every entry of `f`
/// (exact count), flags in groups of 8. Returns the entry count.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[cold]
#[inline(never)]
unsafe fn block_slow(f: u64, qs: u64, n: u64, bad: u64, base1: u32, ep: *mut u32, flp: *mut u8) -> usize {
    let cnt = f.count_ones() as usize;
    let mut x = f;
    let mut i = 0;
    while x != 0 {
        *ep.add(i) = base1 + x.trailing_zeros();
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
/// entries including the virtual entry 0 (the newline before p); sentinels
/// follow.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn stage1(p: *const u8, nb: usize, lastmask: u64, e: *mut u32, fl: *mut u8, pf: usize) -> usize {
    *e = 0;
    *fl = F_NL;
    // the byte before p ends a line: a delimiter there, no quote, no carry
    let mut st = S1 { dp: 1 << 63, qp: 0, fsp: 0, cin: 0, kb: 1 };
    for blk in 0..nb - 1 {
        super::prefetch(p.wrapping_add(blk * 64 + pf));
        block(&mut st, p, blk, !0u64, e, fl);
    }
    block(&mut st, p, nb - 1, lastmask, e, fl);
    let kb = st.kb;
    let big = (nb * 64 + 1) as u32;
    for i in 0..NSENT {
        *e.add(kb + i) = big;
        *fl.add(kb + i) = F_BAD;
    }
    kb
}

// ---------------------------------------------------------------- stage 2

/// The fields of the simple line whose first entry is k, as a `fastev::Cut`
/// relative to the segment start: s_j = e[k+j-1] + q_j, n_j = e[k+j] -
/// e[k+j-1] - 1 - 2 q_j (q_j = flag bit 0 of entry k+j), lanes 0..3 and 3..6.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn cut(e: *const u32, fl: *const u8, k: usize) -> fastev::Cut {
    let mut c = MaybeUninit::<fastev::Cut>::uninit();
    let cp = c.as_mut_ptr();
    let ld = |i: usize| _mm256_cvtepu32_epi64(_mm_loadu_si128(e.add(i) as *const __m128i));
    let one = _mm256_set1_epi64x(1);
    // fields 0..3: flags of entries k..k+3 are 0 or 1 (simple line)
    let alo = ld(k - 1);
    let blo = ld(k);
    let qlo = _mm256_cvtepu8_epi64(_mm_cvtsi32_si128(rd32(fl.add(k)) as i32));
    // fields 3..6: entry k+6 carries the newline flag
    let ahi = ld(k + 2);
    let bhi = ld(k + 3);
    let qhi = _mm256_and_si256(_mm256_cvtepu8_epi64(_mm_cvtsi32_si128(rd32(fl.add(k + 3)) as i32)), one);
    let slo = _mm256_add_epi64(alo, qlo);
    let shi = _mm256_add_epi64(ahi, qhi);
    let nlo = _mm256_sub_epi64(_mm256_sub_epi64(blo, _mm256_add_epi64(slo, qlo)), one);
    let nhi = _mm256_sub_epi64(_mm256_sub_epi64(bhi, _mm256_add_epi64(shi, qhi)), one);
    let sp = addr_of_mut!((*cp).s) as *mut usize;
    let np = addr_of_mut!((*cp).n) as *mut usize;
    _mm256_storeu_si256(sp as *mut __m256i, slo);
    _mm256_storeu_si256(sp.add(3) as *mut __m256i, shi);
    _mm256_storeu_si256(np as *mut __m256i, nlo);
    _mm256_storeu_si256(np.add(3) as *mut __m256i, nhi);
    addr_of_mut!((*cp).ok).write(true);
    addr_of_mut!((*cp).len).write(0);
    c.assume_init()
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
    let pf = fastev::PF_DIST.load(Ordering::Relaxed);
    let mut ring = MaybeUninit::<Ring>::uninit();
    let e = addr_of_mut!((*ring.as_mut_ptr()).e) as *mut u32;
    let fl = addr_of_mut!((*ring.as_mut_ptr()).fl) as *mut u8;
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
        let cnt = stage1(base.add(a), nb, lastmask, e, fl, pf);
        let p = base.add(a);
        let mut k = 1usize;
        loop {
            let x = rd64(fl.add(k));
            if x & PAT_M == PAT {
                let c = cut(e, fl, k);
                if k4::decode_body::<S, P2>(kp, ctx, p, &c, sink) {
                    st.fast += 1;
                } else {
                    st.line_fb += 1;
                    k4::line_fb(p, &c, ctx, sink);
                }
                k += 7;
                continue;
            }
            // not simple: the line's newline among the entries, if any
            let ls = a + *e.add(k - 1) as usize;
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
                debug_assert_eq!(nx, a + *e.add(j) as usize);
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
