//! Adversarial tests of the k4 event kernel (k4.rs) against the old path
//! (the oracle: `fastev::task_fast_any`, i.e. `fastev::event_task_fast` +
//! `main::event_line`).
//!
//! Every test builds event CSV text, places it in a buffer that ends exactly
//! at a PROT_NONE page (and, separately, one that starts right after a
//! PROT_NONE page), cuts it into tasks as `main::load_input` would at many
//! chunk sizes (one task per line up to one task for the whole buffer), runs
//! both paths on every task into recording sinks and requires the same
//! copy() calls, argument by argument, in the same order. A failure names
//! the task, the line(s) that differ on their own (exact bytes) and both
//! copy lists.
//!
//! k4 is x86-64 AVX2/BMI2 only: the tests exist on x86_64 and return early
//! (with a note) when the CPU lacks the features.
//!
//! Run (release, on the x86 benchmark VM): `cargo test --release k4test`.
//! SS_K4TEST_SEED=k moves the random tests to other seeds, SS_K4TEST_SCALE=k
//! sets the number of random lines (default 4).

#[cfg(not(target_arch = "x86_64"))]
#[test]
fn k4test_skipped_not_x86() {
    eprintln!("k4test: k4 is x86-64 only; nothing to test on this architecture");
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use crate::fastev::{self, RecSink};
    use crate::k4::{self, Stats};
    use crate::lookup::IdTable;
    use crate::{scan, Ctx, Task, DAY, LATE};
    use std::ffi::c_void;

    const NPROD: usize = 3000;

    // ------------------------------------------------------------ PRNG

    /// splitmix64 (deterministic, no crates).
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Rng {
            Rng(seed ^ 0x2545_F491_4F6C_DD1D ^ seed0().wrapping_mul(0x9E37_79B9_7F4A_7C15))
        }
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n.max(1)
        }
        fn range(&mut self, lo: i64, hi: i64) -> i64 {
            lo + self.below((hi - lo).max(1) as u64) as i64
        }
        fn chance(&mut self, p: f64) -> bool {
            ((self.next() >> 11) as f64) < p * (1u64 << 53) as f64
        }
        fn pick<'a, T>(&mut self, v: &'a [T]) -> &'a T {
            &v[self.below(v.len() as u64) as usize]
        }
    }

    fn seed0() -> u64 {
        std::env::var("SS_K4TEST_SEED").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0)
    }

    fn scale() -> usize {
        std::env::var("SS_K4TEST_SCALE").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(4usize).max(1)
    }

    // ------------------------------------------------------------ environment

    fn ms_of(s: &str) -> i64 {
        crate::ts::parse_ts(s.as_bytes()).unwrap_or_else(|| panic!("bad test timestamp {}", s))
    }

    /// A Ctx as main::run builds it for window [ws, we): NPROD products
    /// (p1.., 20 categories + unknown), a few campaigns.
    fn mk_ctx(ws: i64, we: i64) -> Ctx {
        let first_day = ws.div_euclid(DAY);
        let ndays = ((we - 1).div_euclid(DAY) - first_day + 1).max(1) as usize;
        let mut cat_names: Vec<Vec<u8>> = (0..20).map(|i| format!("cat{:02}", i).into_bytes()).collect();
        cat_names.push(b"unknown".to_vec());
        let pentries: Vec<(Vec<u8>, u16)> = (1..=NPROD).map(|i| (format!("p{}", i).into_bytes(), (i % 20) as u16)).collect();
        let camp_ids: Vec<Vec<u8>> = (1..=10).map(|i| format!("c_{}", i).into_bytes()).collect();
        let centries: Vec<(Vec<u8>, u16)> = camp_ids.iter().enumerate().map(|(i, c)| (c.clone(), i as u16)).collect();
        Ctx {
            tsp: crate::ts::TsParser::new(ws - 10 * DAY, we + 10 * DAY),
            tsw: false,
            tsh: false,
            avx2: crate::use_avx2(),
            ws,
            we,
            ws_s: ws.div_euclid(1000),
            ws_mod_day: ws.rem_euclid(DAY) as u64,
            first_day,
            ndays,
            products: IdTable::build(&pentries),
            unknown_cat: 20,
            cat_names,
            campaigns: IdTable::build(&centries),
            camp_channels: camp_ids.iter().map(|_| b"search".to_vec()).collect(),
            ncamp: camp_ids.len(),
            camp_ids,
            pe_bits: 2,
            ps_bits: 2,
            threads: 1,
        }
    }

    /// Windows the tests run in: a generator-like 14-day window, one over
    /// February 29 2028, one starting at a non-midnight instant over a year
    /// end, a 1-day and a 24-day window.
    fn envs() -> Vec<Ctx> {
        let w = |s: &str, ms: i64| {
            let ws = ms_of(s);
            mk_ctx(ws, ws + ms)
        };
        vec![
            w("2027-12-10T00:00:00Z", 14 * DAY),
            w("2028-02-20T00:00:00Z", 14 * DAY),
            w("2025-12-25T05:30:07Z", 10 * DAY + 1234),
            w("2026-03-31T00:00:00Z", DAY),
            w("2026-01-01T00:00:00Z", 24 * DAY),
        ]
    }

    fn have_cpu() -> bool {
        let ok = crate::use_avx2();
        if !ok {
            eprintln!("k4test: CPU without AVX2/BMI2 (or SS_NOAVX2 set): k4 is never used; test skipped");
        }
        ok
    }

    // ------------------------------------------------------------ guarded buffers

    extern "C" {
        fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut c_void;
        fn munmap(addr: *mut c_void, len: usize) -> i32;
        fn mprotect(addr: *mut u8, len: usize, prot: i32) -> i32;
        fn fork() -> i32;
        fn waitpid(pid: i32, status: *mut i32, options: i32) -> i32;
        fn _exit(code: i32) -> !;
    }
    #[cfg(target_os = "linux")]
    const MAP_ANON: i32 = 0x20;
    #[cfg(not(target_os = "linux"))]
    const MAP_ANON: i32 = 0x1000;
    const PG: usize = 4096;

    /// Text in anonymous pages between two PROT_NONE pages, ending exactly
    /// at the second (`at_end`) or starting exactly after the first: a read
    /// one byte past the end (or before the start) faults.
    struct GBuf {
        map: *mut u8,
        maplen: usize,
        off: usize,
        len: usize,
    }

    impl GBuf {
        fn new(text: &[u8], at_end: bool) -> GBuf {
            let pages = text.len().div_ceil(PG).max(1);
            let maplen = (pages + 2) * PG;
            unsafe {
                let map = mmap(std::ptr::null_mut(), maplen, 3, 2 | MAP_ANON, -1, 0) as *mut u8;
                assert!(map as isize != -1, "mmap failed");
                assert_eq!(mprotect(map, PG, 0), 0, "mprotect failed");
                assert_eq!(mprotect(map.add((pages + 1) * PG), PG, 0), 0, "mprotect failed");
                let off = if at_end { (pages + 1) * PG - text.len() } else { PG };
                std::ptr::copy_nonoverlapping(text.as_ptr(), map.add(off), text.len());
                GBuf { map, maplen, off, len: text.len() }
            }
        }
        fn bytes(&self) -> &[u8] {
            unsafe { std::slice::from_raw_parts(self.map.add(self.off), self.len) }
        }
    }

    impl Drop for GBuf {
        fn drop(&mut self) {
            unsafe { munmap(self.map as *mut c_void, self.maplen) };
        }
    }

    // ------------------------------------------------------------ comparison

    type Copies = Vec<(u64, u32, Option<(u64, u32, u32)>)>;

    fn add(a: &mut Stats, b: Stats) {
        a.fast += b.fast;
        a.line_fb += b.line_fb;
        a.range_fb += b.range_fb;
        a.tail += b.tail;
    }

    fn show(b: &[u8]) -> String {
        let s = b.escape_ascii().to_string();
        if s.len() > 600 {
            format!("{}...[{} bytes]", &s[..600], b.len())
        } else {
            s
        }
    }

    fn list<T: std::fmt::Debug>(v: &[T]) -> String {
        if v.len() > 20 {
            format!("{:?} ... ({} copies)", &v[..20], v.len())
        } else {
            format!("{:?}", v)
        }
    }

    /// The kernel variants (split, pack2): every one is checked.
    const VARIANTS: &[(usize, bool)] = &[(2, true), (0, false), (1, true), (2, false), (3, false), (3, true)];

    /// Both paths on one task -> (old copies, k4 copies, k4 stats of the
    /// default variant); the k4 copies of every variant must be identical
    /// (else the first differing variant's copies are returned).
    fn run_one(ctx: &Ctx, buf: &[u8], t: Task) -> (Copies, Copies, Stats) {
        let safe_end = buf.as_ptr() as usize + buf.len();
        let mut a = RecSink::default();
        fastev::task_fast_any(buf, t, ctx, &mut a, safe_end);
        let mut first = None;
        for &(spl, p2) in VARIANTS {
            let mut b = RecSink::default();
            let st = k4::task_any_v(buf, t, ctx, &mut b, safe_end, spl, p2).expect("k4 needs AVX2");
            if b.0 != a.0 {
                eprintln!("k4test: variant split={} pack2={} differs", spl, p2);
                return (a.0, b.0, st);
            }
            first.get_or_insert((b.0, st));
        }
        let (b, st) = first.unwrap();
        (a.0, b, st)
    }

    /// The lines of task t that differ on their own.
    fn culprits(ctx: &Ctx, buf: &[u8], t: Task) -> String {
        let mut s = String::new();
        let mut l = t.start;
        let mut shown = 0;
        while l < t.end && shown < 5 {
            let end = scan::next_line_start(buf, l).min(t.end);
            let (a, b, _) = run_one(ctx, buf, Task { file: 0, start: l, end });
            if a != b {
                s += &format!("  line at {} ({} bytes before the buffer end): \"{}\"\n    old: {}\n    k4:  {}\n", l, buf.len() - l, show(&buf[l..end]), list(&a), list(&b));
                shown += 1;
            }
            l = end;
        }
        if shown == 0 {
            s += "  (no single line differs on its own: the difference depends on the task context)\n";
        }
        s
    }

    fn cmp(ctx: &Ctx, buf: &[u8], tasks: &[Task], what: &str) -> Result<Stats, String> {
        let mut st = Stats::default();
        for &t in tasks {
            let (a, b, s) = run_one(ctx, buf, t);
            add(&mut st, s);
            if a != b {
                let mut rep = format!(
                    "k4test {}: MISMATCH in task [{}, {}) of a {}-byte buffer (window {}..{})\n  old: {}\n  k4:  {}\n",
                    what,
                    t.start,
                    t.end,
                    buf.len(),
                    ctx.ws,
                    ctx.we,
                    list(&a),
                    list(&b)
                );
                rep += &culprits(ctx, buf, t);
                if t.end - t.start <= 2000 {
                    rep += &format!("  task text: \"{}\"\n", show(&buf[t.start..t.end]));
                }
                return Err(rep);
            }
        }
        Ok(st)
    }

    /// Tasks as main::load_input cuts them with this chunk.
    fn chunk_tasks(b: &[u8], chunk: usize) -> Vec<Task> {
        let mut v = Vec::new();
        let mut s = 0;
        while s < b.len() {
            let nominal = s + chunk;
            let e = if nominal >= b.len() { b.len() } else { scan::next_line_start(b, nominal - 1) };
            v.push(Task { file: 0, start: s, end: e });
            s = e;
        }
        v
    }

    /// chunk 1 = one task per line; the last = the whole buffer.
    const CHUNKS: &[usize] = &[1, 2, 7, 64, 100, 191, 192, 193, 255, 256, 257, 287, 288, 289, 300, 777, 4096, 1 << 40];
    const CHUNKS_FEW: &[usize] = &[1, 300, 1 << 40];

    /// Text in both guarded placements, every chunking; panics with the
    /// report on a difference. Returns k4's stats of the whole-buffer task
    /// (end-aligned placement).
    fn check_text(ctx: &Ctx, text: &[u8], chunks: &[usize], what: &str) -> Stats {
        let mut whole = Stats::default();
        for at_end in [true, false] {
            let g = GBuf::new(text, at_end);
            let b = g.bytes();
            for &c in chunks {
                let tasks = chunk_tasks(b, c);
                match cmp(ctx, b, &tasks, &format!("{} [{}, chunk {}]", what, if at_end { "end at guard" } else { "start after guard" }, c)) {
                    Ok(st) => {
                        if at_end && c == 1 << 40 {
                            whole = st;
                        }
                    }
                    Err(rep) => panic!("{}", rep),
                }
            }
        }
        whole
    }

    /// Lines (no newline at their end) after a header line and before a
    /// 320-byte back filler, so that every line can take the kernel.
    fn framed(lines: &[Vec<u8>]) -> Vec<u8> {
        let mut t = Vec::new();
        t.extend_from_slice(b"event_id,user_id,ts,ingest_ts,event_type,product_id,price_cents\n");
        for l in lines {
            t.extend_from_slice(l);
            t.push(b'\n');
        }
        for _ in 0..5 {
            t.extend_from_slice(&[b'x'; 63]);
            t.push(b'\n');
        }
        t
    }

    /// Lines made for each window by `mk(ctx, seed)`, in batches of `per`
    /// (each framed), every chunking, every window.
    fn check_gen(seed: u64, per: usize, chunks: &[usize], what: &str, mk: impl Fn(&Ctx, u64) -> Vec<Vec<u8>>) -> Stats {
        let mut tot = Stats::default();
        let mut nlines = 0;
        for (ei, ctx) in envs().iter().enumerate() {
            let lines = mk(ctx, seed * 1000 + ei as u64);
            nlines += lines.len();
            for (i, part) in lines.chunks(per.max(1)).enumerate() {
                let st = check_text(ctx, &framed(part), chunks, &format!("{} env {} batch {}", what, ei, i));
                add(&mut tot, st);
            }
        }
        eprintln!(
            "k4test {}: {} lines over {} windows identical; k4 whole-buffer: fast {} event_line {} scanner {} tail ranges {}",
            what,
            nlines,
            envs().len(),
            tot.fast,
            tot.line_fb,
            tot.range_fb,
            tot.tail
        );
        tot
    }

    // ------------------------------------------------------------ values

    fn civil(z: i64) -> (i64, i64, i64) {
        let z = z + 719_468;
        let era = z.div_euclid(146_097);
        let doe = z - era * 146_097;
        let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = doy - (153 * mp + 2) / 5 + 1;
        let m = if mp < 10 { mp + 3 } else { mp - 9 };
        (yoe + era * 400 + (m <= 2) as i64, m, d)
    }

    /// `YYYY-MM-DDTHH:MM:SS` of ms (floored to the second).
    fn civ(ms: i64) -> String {
        let (y, m, d) = civil(ms.div_euclid(DAY));
        let s = ms.rem_euclid(DAY) / 1000;
        format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}", y, m, d, s / 3600, s / 60 % 60, s % 60)
    }

    fn f1(ms: i64) -> String {
        format!("{}Z", civ(ms))
    }

    /// F2 of ms with offset `off` minutes (local = UTC + off).
    fn f2(ms: i64, off: i64) -> String {
        let a = off.abs();
        format!("{}{}{:02}:{:02}", civ(ms + off * 60_000), if off < 0 { '-' } else { '+' }, a / 60, a % 60)
    }

    fn f3(ms: i64, width: usize) -> String {
        format!("{:0w$}", ms, w = width)
    }

    fn b36(mut v: u64, n: usize) -> String {
        let al = b"0123456789abcdefghijklmnopqrstuvwxyz";
        let mut s = vec![b'0'; n];
        for i in (0..n).rev() {
            s[i] = al[(v % 36) as usize];
            v /= 36;
        }
        String::from_utf8(s).unwrap()
    }

    fn hex32(r: &mut Rng) -> String {
        format!("{:016x}{:016x}", r.next(), r.next())
    }

    /// An ID as the generator writes it: prefix + 6..8 base-36 digits, or one
    /// in eight prefix + 32 lowercase hex digits.
    fn gen_id(r: &mut Rng, prefix: char) -> String {
        if r.chance(0.125) {
            format!("{}{}", prefix, hex32(r))
        } else {
            let n = 6 + r.below(3) as usize;
            format!("{}{}", prefix, b36(r.next(), n))
        }
    }

    /// IDs of many shapes (prefix `p`): packed, long, near-long, odd bytes.
    fn odd_id(r: &mut Rng, p: char) -> Vec<u8> {
        let h = hex32(r);
        match r.below(24) {
            0 => Vec::new(),
            1 => format!("{}", p).into_bytes(),
            2 => format!("{}{}", p, b36(r.next(), 9)).into_bytes(), // 10 bytes: packs
            3 => format!("{}{}", p, b36(r.next(), 10)).into_bytes(), // 11 bytes: interned
            4 => format!("{}{}", p, &h[..31]).into_bytes(),
            5 => format!("{}{}0", p, h).into_bytes(),
            6 => format!("{}{}", p, h.to_uppercase()).into_bytes(),
            7 => {
                let mut s = format!("{}{}", p, h).into_bytes();
                let k = 1 + r.below(32) as usize;
                s[k] = *r.pick(b"ABCDEFgxz-_G/:`@ \xff");
                s
            }
            8 => format!(" {}", h).into_bytes(),
            9 => {
                let mut s = format!("{}{}", p, h).into_bytes();
                s[0] = *r.pick(b" \xc3-_0Z\xff\t");
                s
            }
            10 => format!("{}{}", p.to_ascii_uppercase(), b36(r.next(), 7)).into_bytes(),
            11 => format!("{}-{}_{}", p, b36(r.next(), 3), b36(r.next(), 2)).into_bytes(),
            12 => format!("{} {}", p, b36(r.next(), 5)).into_bytes(),
            13 => format!("{}{}\r", p, b36(r.next(), 6)).into_bytes(),
            14 => vec![p as u8, 0xd9, 0xa1, b'1'],
            15 => (0..r.below(40) as usize).map(|_| *r.pick(b"abcdef0123456789uU-_ ")).collect(),
            16 => format!("{}{}", p, "0".repeat(32)).into_bytes(),
            17 => format!("{}{}", p, "f".repeat(32)).into_bytes(),
            18 => format!("{}{}{}", p, h, h).into_bytes(),
            19 => format!("{}{}", p, &h[..16]).into_bytes(),
            _ => gen_id(r, p).into_bytes(),
        }
    }

    const PRICES: &[&str] = &[
        "", "1", "250", "0250", "10000000", "10000001", "0", "00", "-5", "+5", "19.99", "1e3", " 250", "250 ", "1000\r", "00000000000010000000", "4294967296", "99999999999999999999", "25 0", "\u{662}\u{665}\u{660}", "9999999", "123456789012345678901234567890",
    ];
    const TYPES: &[&str] = &["view", "cart", "purchase", "Purchase", "PURCHASE", "View", "cart ", " cart", "click", "", "views", "purchas", "purchasee", "vie", "car", "purchas\r", "view\r", "viewcart", "cartview", "purchase\r"];

    /// One generated event line's fields (mostly valid, generator dirt).
    fn gen_fields(ctx: &Ctx, r: &mut Rng) -> Vec<String> {
        let (ws, we) = (ctx.ws, ctx.we);
        let t = r.range(ws - DAY, we + DAY);
        let ts = match r.below(100) {
            0..=57 => f1(t),
            58..=67 => f2(t, r.range(-600, 721)),
            68..=97 => f3(t, 13),
            98 => f3(t, 13 + r.below(5) as usize),
            _ => f1(t).replace('T', " "),
        };
        let tt = t.div_euclid(1000) * 1000;
        let delay = match r.below(100) {
            0..=89 => r.range(0, 600_000),
            90..=93 => LATE + r.range(-2000, 2001),
            94..=96 => r.range(LATE, 5 * DAY),
            _ => -r.range(0, 3_600_000),
        };
        let ing = f1(tt + delay);
        let ty = match r.below(100) {
            0..=60 => "view".to_string(),
            61..=90 => "cart".to_string(),
            91..=97 => "purchase".to_string(),
            _ => r.pick(TYPES).to_string(),
        };
        let prod = match r.below(100) {
            0..=95 => format!("p{}", 1 + r.below(NPROD as u64)),
            96 => String::new(),
            97 => format!("P{}", 1 + r.below(NPROD as u64)),
            98 => format!("p{}", NPROD as u64 + 1 + r.below(100_000)),
            _ => format!("p0{}", r.below(99)),
        };
        let price = match r.below(10) {
            0..=6 => format!("{}", 100 + r.below(49_901)),
            7 => String::new(),
            _ => r.pick(PRICES).to_string(),
        };
        let user = if r.chance(0.01) { String::new() } else { gen_id(r, 'u') };
        vec![gen_id(r, 'e'), user, ts, ing, ty, prod, price]
    }

    /// Fields joined, field j quoted iff bit j of qmask.
    fn join(f: &[String], qmask: u32) -> Vec<u8> {
        let mut v = Vec::new();
        for (j, x) in f.iter().enumerate() {
            if j > 0 {
                v.push(b',');
            }
            let q = qmask >> j & 1 != 0;
            if q {
                v.push(b'"');
            }
            v.extend_from_slice(x.as_bytes());
            if q {
                v.push(b'"');
            }
        }
        v
    }

    fn join_raw(f: &[Vec<u8>]) -> Vec<u8> {
        let mut v = Vec::new();
        for (j, x) in f.iter().enumerate() {
            if j > 0 {
                v.push(b',');
            }
            v.extend_from_slice(x);
        }
        v
    }

    fn rand_q(r: &mut Rng) -> u32 {
        let mut m = 0;
        for j in 0..7 {
            if r.chance(0.2) {
                m |= 1 << j;
            }
        }
        m
    }

    fn with(f: &[String], i: usize, v: &str) -> Vec<String> {
        let mut g = f.to_vec();
        g[i] = v.to_string();
        g
    }

    /// Random single-byte edits (replace / insert / delete) with bytes that
    /// matter to the dialect and the decoders.
    fn mutate(r: &mut Rng, l: &[u8]) -> Vec<u8> {
        let mut v = l.to_vec();
        let alpha = b",\"\n\r 0123456789:-+TZtzaefu.\xff\x00";
        for _ in 0..1 + r.below(3) {
            if v.is_empty() {
                v.push(*r.pick(alpha));
                continue;
            }
            let k = r.below(v.len() as u64) as usize;
            match r.below(3) {
                0 => v[k] = *r.pick(alpha),
                1 => v.insert(k, *r.pick(alpha)),
                _ => {
                    v.remove(k);
                }
            }
        }
        // a newline inside makes two lines: keep them (both are lines)
        v
    }

    // ------------------------------------------------------------ tests

    /// The guard page faults on a read one byte past the buffer end (so the
    /// placements below really test reads past the end).
    #[test]
    fn k4test_guard_pages_fault() {
        let g = GBuf::new(b"e1,u1,1829024243000,2027-12-17T06:18:47Z,cart,p2,\n", true);
        let b = g.bytes();
        unsafe {
            let pid = fork();
            assert!(pid >= 0);
            if pid == 0 {
                let x = std::ptr::read_volatile(b.as_ptr().add(b.len()));
                _exit(x as i32 & 0 | 3);
            }
            let mut st = 0i32;
            waitpid(pid, &mut st, 0);
            let sig = st & 0x7f;
            assert!(sig == 11 || sig == 10, "read past the guarded buffer did not fault (status {:#x})", st);
        }
    }

    /// Generated lines (generator-like values and dirt, random quoting):
    /// the bulk of the kernel's fast path.
    #[test]
    fn k4test_generated_lines() {
        if !have_cpu() {
            return;
        }
        // the keys tests change the global long-key state (hash bits, seed)
        let _g = crate::keys::LONG_TEST.lock().unwrap_or_else(|e| e.into_inner());
        let st = check_gen(1, 1000, CHUNKS, "generated", |ctx, seed| {
            let mut r = Rng::new(seed);
            let mut lines = Vec::new();
            for _ in 0..3000 * scale() {
                let f = gen_fields(ctx, &mut r);
                lines.push(join(&f, rand_q(&mut r)));
            }
            lines
        });
        assert!(st.fast * 10 > 3000 * scale() * 5 * 8, "fast path not exercised: {:?}", st);
    }

    /// Every quote pattern of the seven fields, malformed quotes in every
    /// field, field counts 0..12, empty fields at every position, `""`.
    #[test]
    fn k4test_quotes_counts_empty() {
        if !have_cpu() {
            return;
        }
        // the keys tests change the global long-key state (hash bits, seed)
        let _g = crate::keys::LONG_TEST.lock().unwrap_or_else(|e| e.into_inner());
        check_gen(2, 400, CHUNKS, "quotes", |ctx, seed| {
            let mut r = Rng::new(seed);
            let mut lines = Vec::new();
            for _ in 0..6 {
                let f = gen_fields(ctx, &mut r);
                for q in 0..128u32 {
                    lines.push(join(&f, q));
                }
                let raw: Vec<Vec<u8>> = f.iter().map(|x| x.as_bytes().to_vec()).collect();
                for j in 0..7 {
                    let mut g = raw.clone();
                    let x = g[j].clone();
                    for v in [
                        [b"\"".as_slice(), &x].concat(),                           // unclosed
                        [x.as_slice(), b"\""].concat(),                           // trailing quote
                        [b"\"".as_slice(), &x, b"\"\""].concat(),                 // extra quote
                        b"\"".to_vec(),                                           // one-byte quote field
                        b"\"\"".to_vec(),                                         // empty quoted
                        b"\"\"\"\"".to_vec(),                                     // four quotes
                        [b"\"".as_slice(), &x, b"\"x"].concat(),                  // byte after the closing quote
                        [b"x\"".as_slice(), &x, b"\""].concat(),                  // byte before the opening quote
                        [&x[..x.len() / 2], b"\"", &x[x.len() / 2..]].concat(),   // quote inside
                        Vec::new(),                                               // empty
                    ] {
                        g[j] = v;
                        lines.push(join_raw(&g));
                    }
                }
                for nf in 0..13usize {
                    let g: Vec<Vec<u8>> = (0..nf).map(|j| raw[j.min(6)].clone()).collect();
                    lines.push(join_raw(&g));
                    let mut h = join(&f, rand_q(&mut r));
                    for _ in 7..nf {
                        h.extend_from_slice(b",x");
                    }
                    lines.push(h);
                }
                lines.push(Vec::new());
                lines.push(b",,,,,,".to_vec());
                lines.push(b"\"\",\"\",\"\",\"\",\"\",\"\",\"\"".to_vec());
                lines.push(join(&f, 0).iter().map(|&c| if c == b',' { b';' } else { c }).collect());
            }
            lines
        });
    }

    /// event_id and user_id of every shape (packed, long hex, near-long,
    /// upper case, non-alphabet bytes, every length 0..40), valid and
    /// invalid copies.
    #[test]
    fn k4test_id_shapes() {
        if !have_cpu() {
            return;
        }
        // the keys tests change the global long-key state (hash bits, seed)
        let _g = crate::keys::LONG_TEST.lock().unwrap_or_else(|e| e.into_inner());
        let st = check_gen(3, 1500, CHUNKS_FEW, "ids", |ctx, seed| {
            let mut r = Rng::new(seed);
            let mut lines = Vec::new();
            for i in 0..4000 * scale() {
                let mut f: Vec<Vec<u8>> = gen_fields(ctx, &mut r).into_iter().map(|s| s.into_bytes()).collect();
                if i % 3 != 1 {
                    f[0] = odd_id(&mut r, 'e');
                }
                if i % 3 != 0 {
                    f[1] = odd_id(&mut r, 'u');
                }
                if i % 2 == 0 {
                    // make the copy valid as far as the other fields go
                    f[4] = b"view".to_vec();
                    f[5] = b"p7".to_vec();
                }
                let q = rand_q(&mut r);
                let g: Vec<Vec<u8>> = f.iter().enumerate().map(|(j, x)| if q >> j & 1 != 0 { [b"\"".as_slice(), x, b"\""].concat() } else { x.clone() }).collect();
                lines.push(join_raw(&g));
            }
            // every length 0..40 over a few alphabets, both fields
            for n in 0..41usize {
                for al in [&b"abcdefghijklmnopqrstuvwxyz0123456789"[..], b"0123456789abcdef", b"ABCDEF0123456789", b"-_aZ09", b"a \xc3\xa9"] {
                    let id: Vec<u8> = (0..n).map(|_| *r.pick(al)).collect();
                    let mut f: Vec<Vec<u8>> = gen_fields(ctx, &mut r).into_iter().map(|s| s.into_bytes()).collect();
                    f[4] = b"cart".to_vec();
                    f[5] = b"p9".to_vec();
                    let mut g = f.clone();
                    g[0] = id.clone();
                    lines.push(join_raw(&g));
                    let mut g = f.clone();
                    g[1] = id.clone();
                    lines.push(join_raw(&g));
                }
            }
            lines
        });
        assert!(st.fast > 4000 * scale(), "fast path not exercised: {:?}", st);
    }

    /// ts values of every form, valid and invalid, at the window edges.
    fn ts_values(ctx: &Ctx, r: &mut Rng) -> Vec<String> {
        let (ws, we) = (ctx.ws, ctx.we);
        let mut v = Vec::new();
        let ws_s = ws.div_euclid(1000) * 1000;
        let edges = [ws - 1, ws, ws + 1, we - 1, we, we + 1, ws_s, ws_s + 1000, ws - 1000, ws - 2 * DAY, ws - 3 * DAY, we + 5 * DAY, we + 20 * DAY, ws + DAY / 2];
        for &t in &edges {
            v.push(f3(t, 13));
            v.push(f3(t, 14));
            v.push(f3(t, 16));
            v.push(f1(t));
            v.push(f1(t + 999));
            for off in [-600, -1, 0, 1, 59, 60, 330, 720, 23 * 60 + 59] {
                v.push(f2(t, off));
            }
            v.push(f2(t, 0).replace('+', "-"));
        }
        let t = r.range(ws, we);
        let base = f1(t);
        for bad in [
            base.replace('Z', "z"),
            base.replace('T', "t"),
            base.replace('T', " "),
            base.replace('Z', ""),
            base.replace('Z', ".000Z"),
            format!("{}\r", base),
            format!("{}+24:00", &base[..19]),
            format!("{}+05:60", &base[..19]),
            format!("{}+0200", &base[..19]),
            format!("{}+02:0", &base[..19]),
            format!("{}+02:000", &base[..19]),
            format!("{}*02:00", &base[..19]),
            format!("{}+2:00 ", &base[..19]),
            format!("{}.0", f3(t, 13)),
            "1.79e12".to_string(),
            "-5".to_string(),
            String::new(),
            "86400000".to_string(),
            "0".to_string(),
            "9223372036854775807".to_string(),
            "9223372036854775808".to_string(),
            "00000000000000000000001".to_string(),
            "\u{662}\u{665}\u{660}".to_string(),
            "１７９１０７１８９００００".to_string(),
        ] {
            v.push(bad);
        }
        // impossible calendar values in and around the window's dates
        let (y, m, d) = civil(t.div_euclid(DAY));
        for (mm, dd, hh, mi, ss) in [(m, d, 24, 0, 0), (m, d, 23, 60, 0), (m, d, 23, 59, 60), (13, d, 1, 1, 1), (0, d, 1, 1, 1), (m, 0, 1, 1, 1), (m, 32, 1, 1, 1), (2, 29, 1, 1, 1), (2, 30, 1, 1, 1), (m, 31, 1, 1, 1), (m, d, 0, 0, 0), (m, d, 23, 59, 59)] {
            v.push(format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, mm, dd, hh, mi, ss));
            v.push(format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}+01:00", y, mm, dd, hh, mi, ss));
        }
        // every F3 length 1..22 with leading zeros, and digits around
        for w in 1..23usize {
            v.push(f3(t, w));
            v.push(f3(r.below(1000) as i64, w));
        }
        // one byte changed at every position of each form
        for s in [f1(t), f2(t, 330), f3(t, 13)] {
            for k in 0..s.len() {
                for c in [b'0', b'9', b'/', b':', b'-', b'+', b'T', b'Z', b' ', b'"', b'a'] {
                    let mut b = s.clone().into_bytes();
                    b[k] = c;
                    if let Ok(x) = String::from_utf8(b) {
                        v.push(x);
                    }
                }
            }
        }
        v
    }

    /// Every ts value with ingest values of every kind (valid near, late
    /// limit +- 1 ms and 1 s, before ts, invalid and other forms), valid and
    /// invalid copies, random quoting.
    #[test]
    fn k4test_timestamps() {
        if !have_cpu() {
            return;
        }
        // the keys tests change the global long-key state (hash bits, seed)
        let _g = crate::keys::LONG_TEST.lock().unwrap_or_else(|e| e.into_inner());
        let mut lines = Vec::new();
        let mut r = Rng::new(4);
        // build per window (values depend on it), check per window
        for (ei, ctx) in envs().iter().enumerate() {
            let mut ls = Vec::new();
            let tsv = ts_values(ctx, &mut r);
            for ts in &tsv {
                let tms = crate::ts::parse_ts(ts.as_bytes()).unwrap_or(ctx.ws + DAY);
                let ing0 = tms.div_euclid(1000) * 1000;
                let mut ings = vec![f1(ing0 + 1000 * r.range(0, 600)), f1(tms + LATE), f1(tms + LATE + 1000), f1(tms + LATE - 1000), f1(ing0 - 3_600_000)];
                if r.chance(0.2) {
                    ings.push(f3(ing0 + 5000, 13));
                    ings.push(f2(ing0, 60));
                    ings.push(ts.clone());
                    ings.push(f1(ing0).replace('Z', "z"));
                    ings.push(String::new());
                }
                for ing in ings {
                    let mut f = gen_fields(ctx, &mut r);
                    f[2] = ts.clone();
                    f[3] = ing;
                    if r.chance(0.8) {
                        f[1] = gen_id(&mut r, 'u');
                        f[4] = (*r.pick(&["view", "cart", "purchase"])).to_string();
                        f[5] = format!("p{}", 1 + r.below(NPROD as u64));
                        f[6] = format!("{}", 1 + r.below(10_000_000));
                    }
                    ls.push(join(&f, rand_q(&mut r)));
                }
            }
            // late limit exactly, +- 1 ms (ts F3 with milliseconds, ingest F1)
            for _ in 0..200 {
                let ing = ctx.ws + 1000 * r.range(DAY / 1000, (ctx.we - ctx.ws) / 1000 + DAY / 1000);
                for dt in [-1i64, 0, 1, 999, 1000, 1001] {
                    let t = ing - LATE + dt;
                    let mut f = gen_fields(ctx, &mut r);
                    f[1] = gen_id(&mut r, 'u');
                    f[2] = if dt % 1000 == 0 && r.chance(0.5) { f1(t) } else { f3(t, 13) };
                    f[3] = f1(ing);
                    f[4] = "view".to_string();
                    f[5] = "p1".to_string();
                    ls.push(join(&f, rand_q(&mut r)));
                }
            }
            let mut tot = Stats::default();
            for (i, part) in ls.chunks(1500).enumerate() {
                add(&mut tot, check_text(ctx, &framed(part), CHUNKS_FEW, &format!("timestamps env {} batch {}", ei, i)));
            }
            eprintln!("k4test timestamps env {}: {} lines identical; fast {} event_line {} scanner {}", ei, ls.len(), tot.fast, tot.line_fb, tot.range_fb);
            assert!(tot.fast * 4 > ls.len(), "fast path not exercised: {:?}", tot);
            lines.extend(ls);
        }
        assert!(lines.len() > 10_000);
    }

    /// Types, products, prices, `\r` at the end of every field.
    #[test]
    fn k4test_types_prices_cr() {
        if !have_cpu() {
            return;
        }
        // the keys tests change the global long-key state (hash bits, seed)
        let _g = crate::keys::LONG_TEST.lock().unwrap_or_else(|e| e.into_inner());
        check_gen(5, 2000, CHUNKS_FEW, "types-prices-cr", |ctx, seed| {
            let mut r = Rng::new(seed);
            let mut lines = Vec::new();
            for ty in TYPES {
                for pr in PRICES {
                    let mut f = gen_fields(ctx, &mut r);
                    f[1] = gen_id(&mut r, 'u');
                    f[2] = f1(r.range(ctx.ws, ctx.we));
                    f[3] = f1(crate::ts::parse_ts(f[2].as_bytes()).unwrap() + 60_000);
                    f[4] = ty.to_string();
                    f[6] = pr.to_string();
                    for prod in ["p1", "p3000", "p3001", "", "P1", "p01", "p0", "\"p5\""] {
                        let g = with(&f, 5, prod);
                        lines.push(join(&g, rand_q(&mut r)));
                    }
                }
            }
            for _ in 0..500 {
                let f = gen_fields(ctx, &mut r);
                for j in 0..7 {
                    let mut g = f.clone();
                    g[j].push('\r');
                    lines.push(join(&g, rand_q(&mut r)));
                    lines.push(join(&g, 1 << j));
                }
                lines.push([join(&f, 0), b"\r".to_vec()].concat());
            }
            lines
        });
    }

    /// Lines of every length around the windows: long fields in every
    /// position (product, price, IDs, type), up to 400 bytes.
    #[test]
    fn k4test_long_lines() {
        if !have_cpu() {
            return;
        }
        // the keys tests change the global long-key state (hash bits, seed)
        let _g = crate::keys::LONG_TEST.lock().unwrap_or_else(|e| e.into_inner());
        check_gen(6, 1500, CHUNKS_FEW, "long-lines", |ctx, seed| {
            let mut r = Rng::new(seed);
            let mut lines = Vec::new();
            for _ in 0..300 {
                let f = gen_fields(ctx, &mut r);
                for j in 0..7 {
                    for extra in [1usize, 10, 20, 30, 40, 50, 60, 70, 100, 150, 250] {
                        let mut g = f.clone();
                        let pad: String = (0..extra).map(|_| *r.pick(&['a', '0', 'z', '9'])).collect();
                        g[j] = format!("{}{}", g[j], pad);
                        lines.push(join(&g, rand_q(&mut r)));
                    }
                }
                // two long IDs and an F2 ts: the widest generator line
                let mut g = f.clone();
                g[0] = format!("e{}", hex32(&mut r));
                g[1] = format!("u{}", hex32(&mut r));
                g[2] = f2(r.range(ctx.ws, ctx.we), -540);
                lines.push(join(&g, 0x7F));
                lines.push(join(&g, rand_q(&mut r)));
            }
            lines
        });
    }

    /// Random byte mutations of generated lines.
    #[test]
    fn k4test_mutations() {
        if !have_cpu() {
            return;
        }
        // the keys tests change the global long-key state (hash bits, seed)
        let _g = crate::keys::LONG_TEST.lock().unwrap_or_else(|e| e.into_inner());
        check_gen(7, 2000, CHUNKS_FEW, "mutations", |ctx, seed| {
            let mut r = Rng::new(seed);
            let mut lines = Vec::new();
            for _ in 0..8000 * scale() {
                let f = gen_fields(ctx, &mut r);
                let l = join(&f, rand_q(&mut r));
                lines.push(mutate(&mut r, &l));
            }
            lines
        });
    }

    /// Every single-byte change of one generator line, at every position.
    #[test]
    fn k4test_single_byte_changes() {
        if !have_cpu() {
            return;
        }
        // the keys tests change the global long-key state (hash bits, seed)
        let _g = crate::keys::LONG_TEST.lock().unwrap_or_else(|e| e.into_inner());
        check_gen(8, 3000, CHUNKS_FEW, "single-byte", |ctx, seed| {
            let mut lines = Vec::new();
            let ts = f2(ctx.ws + 3 * DAY + 4321_000, 330);
            let ing = f1(ctx.ws + 3 * DAY + 4400_000);
            let mut r = Rng::new(seed);
            let lid = format!("e{}", hex32(&mut r));
            let base = [
                format!("\"e42huhoug\",uduffvac,\"{}\",{},\"purchase\",p1364,\"2918\"", ts, ing),
                format!("{},\"u{}\",{},\"{}\",view,p9,", lid, hex32(&mut r), f3(ctx.ws + 5 * DAY + 17, 13), ing),
            ];
            for line in &base {
                let lb = line.as_bytes();
                for pos in 0..lb.len() {
                    for v in [b',', b'"', b'\n', b'\r', b'x', b' ', 0u8, 0xFF, b'0', b'9', b':', b'-', b'Z', b'T'] {
                        let mut l = lb.to_vec();
                        l[pos] = v;
                        lines.push(l);
                    }
                }
            }
            lines
        });
    }

    /// Buffer tails: the buffer ending (at the guard page) at every offset
    /// inside and after the last lines, with and without a final newline,
    /// buffers shorter than the kernel's margin, every chunk size 1..400.
    #[test]
    fn k4test_buffer_tails() {
        if !have_cpu() {
            return;
        }
        // the keys tests change the global long-key state (hash bits, seed)
        let _g = crate::keys::LONG_TEST.lock().unwrap_or_else(|e| e.into_inner());
        let ctx = &envs()[0];
        let mut r = Rng::new(9);
        let mut text = Vec::new();
        for _ in 0..40 {
            let f = gen_fields(ctx, &mut r);
            text.extend_from_slice(&join(&f, rand_q(&mut r)));
            text.push(b'\n');
        }
        let n = text.len();
        let mut st = Stats::default();
        // every truncation of the last ~700 bytes (and short buffers)
        let from = n.saturating_sub(700);
        for cut in (from..=n).chain(0..from.min(400)) {
            let t = &text[..cut];
            add(&mut st, check_text(ctx, t, &[1, 1 << 40], &format!("tail cut {}", cut)));
        }
        // every chunk size on the whole text
        let g = GBuf::new(&text, true);
        let b = g.bytes();
        for c in 1..400usize {
            let tasks = chunk_tasks(b, c);
            if let Err(rep) = cmp(ctx, b, &tasks, &format!("tails chunk {}", c)) {
                panic!("{}", rep);
            }
        }
        eprintln!("k4test tails: identical; fast {} event_line {} scanner {} tail ranges {}", st.fast, st.line_fb, st.range_fb, st.tail);
    }

    /// `k4::pack2` agrees with `keys::pack` in both lanes on every string,
    /// whatever follows it (random strings, every byte value at every
    /// position, every length 0..40).
    #[test]
    fn k4test_pack2_matches_pack() {
        if !have_cpu() {
            return;
        }
        // the keys tests change the global long-key state (hash bits, seed)
        let _g = crate::keys::LONG_TEST.lock().unwrap_or_else(|e| e.into_inner());
        let al = b"-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz";
        let mut r = Rng::new(11);
        let check = |a: &[u8], b: &[u8], r: &mut Rng| {
            let mut ba = a.to_vec();
            let mut bb = b.to_vec();
            for _ in 0..20 {
                ba.push(r.next() as u8);
                bb.push(*r.pick(al));
            }
            let (ka, oka, kb, okb) = unsafe { crate::k4::pack2(ba.as_ptr(), a.len(), bb.as_ptr(), b.len()) };
            let (wa, wb) = (crate::keys::pack(a), crate::keys::pack(b));
            assert_eq!(if oka { Some(ka) } else { None }, wa, "{:?}", a);
            assert_eq!(if okb { Some(kb) } else { None }, wb, "{:?}", b);
        };
        let gen = |r: &mut Rng| -> Vec<u8> {
            let n = r.below(14) as usize;
            (0..n)
                .map(|_| {
                    let x = r.next();
                    match x % 8 {
                        0 => (x >> 8) as u8,
                        1 => al[((x >> 8) % 64) as usize].wrapping_add(((x >> 16) % 3) as u8).wrapping_sub(1),
                        _ => al[((x >> 8) % 64) as usize],
                    }
                })
                .collect()
        };
        for _ in 0..2_000_000 {
            let a = gen(&mut r);
            let b = gen(&mut r);
            check(&a, &b, &mut r);
        }
        for n in 0..41usize {
            for pos in 0..n.min(12) {
                for c in 0..=255u8 {
                    let mut s = vec![b'a'; n];
                    s[pos] = c;
                    let o = vec![b'z'; (n + pos) % 12];
                    check(&s, &o, &mut r);
                    check(&o, &s, &mut r);
                }
            }
        }
    }

    /// k4's date table against the general parser: every second-of-day
    /// boundary of every table date, and dates just outside the table.
    #[test]
    fn k4test_date_edges() {
        if !have_cpu() {
            return;
        }
        // the keys tests change the global long-key state (hash bits, seed)
        let _g = crate::keys::LONG_TEST.lock().unwrap_or_else(|e| e.into_inner());
        let mut r = Rng::new(10);
        for ctx in envs() {
            let mut lines = Vec::new();
            let d0 = ctx.ws.div_euclid(DAY);
            for day in d0 - 4..d0 + 31 {
                for sod in [0i64, 1, 59, 60, 3599, 3600, 43_200, 86_399] {
                    let t = day * DAY + sod * 1000;
                    let mut f = gen_fields(&ctx, &mut r);
                    f[1] = gen_id(&mut r, 'u');
                    f[4] = "view".to_string();
                    f[5] = "p3".to_string();
                    f[2] = if r.chance(0.5) { f1(t) } else { f2(t, r.range(-600, 721)) };
                    f[3] = f1(t + 1000 * r.range(-1, 3 * 86_400));
                    lines.push(join(&f, rand_q(&mut r)));
                    f[3] = f1(t);
                    f[2] = f1(t - LATE);
                    lines.push(join(&f, 0));
                }
            }
            check_text(&ctx, &framed(&lines), CHUNKS_FEW, "date-edges");
        }
    }
}
