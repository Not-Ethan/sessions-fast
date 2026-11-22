//! Tests of the k5 click kernel (k5.rs) against the old path (the oracle:
//! `main::click_task`, PCache off, through `k5::old_task`).
//!
//! Every test builds click CSV text, places it in a buffer that ends exactly
//! at a PROT_NONE page (and, separately, one that starts right after a
//! PROT_NONE page), cuts it into tasks as `main::load_input` would at many
//! chunk sizes, runs both paths on every task into recording sinks and
//! requires the same push_click() calls, value by value, in the same order.
//! A failure names the task, the lines that differ on their own (exact
//! bytes) and both record lists.
//!
//! k5 is x86-64 AVX2/BMI2 only: the tests exist on x86_64 and return early
//! (with a note) when the CPU lacks the features.
//! Run (release, on the x86 VM): `cargo test --release k5test`.
//! SS_K5TEST_SEED=k moves the random tests to other seeds.

#[cfg(not(target_arch = "x86_64"))]
#[test]
fn k5test_skipped_not_x86() {
    eprintln!("k5test: k5 is x86-64 only; nothing to test on this architecture");
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use crate::k5::{self, RecC};
    use crate::links::LinkTable;
    use crate::lookup::IdTable;
    use crate::{scan, Ctx, Task, DAY};
    use std::ffi::c_void;

    // ------------------------------------------------------------ PRNG

    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Rng {
            Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
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
            (self.next() >> 11) as f64 / (1u64 << 53) as f64 <= p
        }
        fn pick<'a, T>(&mut self, v: &'a [T]) -> &'a T {
            &v[self.below(v.len() as u64) as usize]
        }
    }

    fn seed0() -> u64 {
        std::env::var("SS_K5TEST_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(1)
    }

    // ------------------------------------------------------------ contexts

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

    fn ms_of(s: &str) -> i64 {
        crate::ts::parse_ts(s.as_bytes()).unwrap()
    }

    fn mk_ctx(ws: i64, we: i64) -> Ctx {
        let first_day = ws.div_euclid(DAY);
        let ndays = ((we - 1).div_euclid(DAY) - first_day + 1).max(1) as usize;
        let mut cat_names: Vec<Vec<u8>> = (0..20).map(|i| format!("cat{:02}", i).into_bytes()).collect();
        cat_names.push(b"unknown".to_vec());
        let pentries: Vec<(Vec<u8>, u16)> = (1..=50).map(|i| (format!("p{}", i).into_bytes(), (i % 20) as u16)).collect();
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

    /// A generator-like 14-day window, one over February 29 2028, one
    /// starting at a non-midnight instant over a year end (not whole
    /// seconds), a 1-day and a 24-day window (dates beyond the kernel's
    /// 28-date table).
    fn envs() -> Vec<Ctx> {
        let w = |s: &str, ms: i64| {
            let ws = ms_of(s);
            mk_ctx(ws, ws + ms)
        };
        vec![
            w("2026-10-01T00:00:00Z", 14 * DAY),
            w("2028-03-02T00:00:00Z", 14 * DAY),
            w("2025-12-25T05:30:07Z", 10 * DAY + 1234),
            w("2026-03-31T00:00:00Z", DAY),
            w("2026-01-01T00:00:00Z", 24 * DAY),
        ]
    }

    fn have_cpu() -> bool {
        let ok = crate::use_avx2();
        if !ok {
            eprintln!("k5test: CPU without AVX2/BMI2 (or SS_NOAVX2 set): k5 is never used; test skipped");
        }
        ok
    }

    // ------------------------------------------------------------ guarded buffers

    extern "C" {
        fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut c_void;
        fn munmap(addr: *mut c_void, len: usize) -> i32;
        fn mprotect(addr: *mut u8, len: usize, prot: i32) -> i32;
    }
    #[cfg(target_os = "linux")]
    const MAP_ANON: i32 = 0x20;
    #[cfg(not(target_os = "linux"))]
    const MAP_ANON: i32 = 0x1000;
    const PG: usize = 4096;

    /// Text between two PROT_NONE pages, ending exactly at the second
    /// (`at_end`) or starting exactly after the first.
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
                assert_eq!(mprotect(map, PG, 0), 0);
                assert_eq!(mprotect(map.add((pages + 1) * PG), PG, 0), 0);
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

    type Recs = Vec<(usize, u64, u64, u32, u32)>;

    fn run_one(ctx: &Ctx, lt: &LinkTable, buf: &[u8], t: Task) -> (Recs, Recs) {
        let safe_end = buf.as_ptr() as usize + buf.len();
        let mut a = RecC::default();
        k5::old_task(buf, t, ctx, lt, &mut a, safe_end);
        let mut b = RecC::default();
        unsafe { k5::task_k5(buf, t, ctx, lt, &mut b) };
        (k5::norm(a.0), k5::norm(b.0))
    }

    fn list<T: std::fmt::Debug>(v: &[T]) -> String {
        if v.len() > 12 {
            format!("{:?} ... ({} records)", &v[..12], v.len())
        } else {
            format!("{:?}", v)
        }
    }

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

    const CHUNKS: &[usize] = &[1, 2, 7, 64, 100, 191, 192, 193, 255, 256, 257, 300, 777, 4096, 1 << 40];
    const CHUNKS_FEW: &[usize] = &[1, 300, 1 << 40];

    /// Text in both guarded placements, every chunking; panics on a difference.
    /// Returns the number of records of the whole-buffer task.
    fn check_text(ctx: &Ctx, text: &[u8], chunks: &[usize], what: &str) -> usize {
        let lt = LinkTable::empty();
        let mut n = 0;
        for at_end in [true, false] {
            let g = GBuf::new(text, at_end);
            let b = g.bytes();
            for &c in chunks {
                for t in chunk_tasks(b, c) {
                    let (a, k) = run_one(ctx, &lt, b, t);
                    if a != k {
                        let mut rep = format!(
                            "k5test {} [{}, chunk {}]: MISMATCH in task [{}, {}) of a {}-byte buffer (window {}..{})\n  old: {}\n  k5:  {}\n",
                            what,
                            if at_end { "end at guard" } else { "start after guard" },
                            c,
                            t.start,
                            t.end,
                            b.len(),
                            ctx.ws,
                            ctx.we,
                            list(&a),
                            list(&k)
                        );
                        rep += &k5::culprits(b, t, ctx, &lt);
                        panic!("{}", rep);
                    }
                    if at_end && c == 1 << 40 {
                        n += a.len();
                    }
                }
            }
        }
        n
    }

    /// Lines after a header and before a 320-byte filler (so that every
    /// line can take the kernel); `bare` leaves both out.
    fn framed(lines: &[Vec<u8>], bare: bool) -> Vec<u8> {
        let mut t = Vec::new();
        if !bare {
            t.extend_from_slice(b"click_id,user_id,campaign_id,ts\n");
        }
        for l in lines {
            t.extend_from_slice(l);
            t.push(b'\n');
        }
        if !bare {
            for _ in 0..5 {
                t.extend_from_slice(&[b'x'; 63]);
                t.push(b'\n');
            }
        }
        t
    }

    fn check_lines(ctx: &Ctx, lines: &[Vec<u8>], per: usize, chunks: &[usize], what: &str) -> usize {
        let mut n = 0;
        for (i, part) in lines.chunks(per.max(1)).enumerate() {
            n += check_text(ctx, &framed(part, false), chunks, &format!("{} batch {}", what, i));
        }
        n
    }

    // ------------------------------------------------------------ values

    fn civ(ms: i64) -> String {
        let (y, m, d) = civil(ms.div_euclid(DAY));
        let s = ms.rem_euclid(DAY) / 1000;
        format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}", y, m, d, s / 3600, s / 60 % 60, s % 60)
    }
    fn f1(ms: i64) -> String {
        format!("{}Z", civ(ms))
    }
    fn f2(ms: i64, off: i64) -> String {
        let a = off.abs();
        format!("{}{}{:02}:{:02}", civ(ms + off * 60_000), if off < 0 { '-' } else { '+' }, a / 60, a % 60)
    }
    fn f3(ms: i64) -> String {
        format!("{:013}", ms)
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

    /// IDs of every shape: short, long (letter + 32 hex), upper-case long,
    /// 11 bytes, odd bytes, empty.
    fn id(r: &mut Rng, pre: char) -> String {
        match r.below(16) {
            0 | 1 => format!("{}{:016x}{:016x}", pre, r.next(), r.next()),
            2 => format!("{}{:016X}{:016x}", pre, r.next(), r.next()),
            3 => format!("{}{}", pre, b36(r.next(), 10)),
            4 => (*r.pick(&["u.1", "U_AB-c", "a b", "x\"y", "é1", "1234567890", "-", "_", "Z", "abcdefghijk", "\t1"])).to_string(),
            5 if r.chance(0.3) => String::new(),
            6 => format!("{}{:016x}{:015x}", pre, r.next(), r.next() >> 4),
            7 => format!("{}{:016x}{:016x}0", pre, r.next(), r.next()),
            _ => format!("{}{}", pre, b36(r.next(), 1 + r.below(8) as usize)),
        }
    }

    fn camp(r: &mut Rng) -> String {
        if r.chance(0.85) {
            format!("c_{}", 1 + r.below(10))
        } else {
            (*r.pick(&["c_11", "C_1", "c_01", "c_", "", "c_1 ", "c_1x", "c_0", "c_10000000", "x_1", "c_-1"])).to_string()
        }
    }

    /// A timestamp near the window: every form, the window edges, and bad forms.
    fn tstamp(r: &mut Rng, ctx: &Ctx) -> String {
        let cws = ctx.ws - 7 * DAY;
        let edges = [cws - 1000, cws - 1, cws, cws + 1, ctx.we - 1000, ctx.we - 1, ctx.we, ctx.we + 1000];
        let ms = if r.chance(0.2) { *r.pick(&edges) } else { r.range(cws - 2 * DAY, ctx.we + 2 * DAY) };
        let off = r.range(-24 * 60 + 1, 24 * 60);
        match r.below(20) {
            0..=8 => f1(ms),
            9..=10 => f2(ms, off),
            11..=15 => f3(ms),
            16 => {
                // bad / odd variants of a valid form
                let s = f1(ms);
                let mut b = s.into_bytes();
                let i = r.below(b.len() as u64) as usize;
                b[i] = *r.pick(b"0123459:+-TZz /.\x00\xff");
                String::from_utf8_lossy(&b).into_owned()
            }
            17 => {
                let s = f2(ms, off);
                let mut b = s.into_bytes();
                let i = r.below(b.len() as u64) as usize;
                b[i] = *r.pick(b"0123459:+-TZz /.");
                String::from_utf8_lossy(&b).into_owned()
            }
            18 => {
                let s = f3(ms);
                let mut b = s.into_bytes();
                let i = r.below(b.len() as u64) as usize;
                b[i] = *r.pick(b"0123459:+-a /.");
                String::from_utf8_lossy(&b).into_owned()
            }
            _ => (*r.pick(&[
                "2026-10-03T24:00:00Z",
                "2026-10-03T23:60:00Z",
                "2026-10-03T23:59:60Z",
                "2026-10-32T00:00:00Z",
                "2026-13-01T00:00:00Z",
                "2026-02-29T00:00:00Z",
                "2028-02-29T12:00:00Z",
                "2028-02-30T12:00:00Z",
                "2026-10-03t23:58:10Z",
                "2026-10-03T23:58:10z",
                "2026-10-03T23:58:10",
                "2026-10-03 23:58:10Z",
                "2026-10-03T23:58:10+24:00",
                "2026-10-03T23:58:10+23:60",
                "2026-10-03T23:58:10+2300",
                "2026-10-03T23:58:10*01:00",
                "2026-10-03T23:58:10.123Z",
                " 2026-10-03T23:58:10Z",
                "2026-10-03T23:58:10Z ",
                "175952989000",
                "17595298900001",
                "-175952989000",
                "",
                "1759529890000\r",
                "2026-10-03T23:58:10Z\r",
            ]))
            .to_string(),
        }
    }

    fn q(r: &mut Rng, s: &str, quote: bool) -> String {
        let _ = r;
        if quote {
            format!("\"{}\"", s)
        } else {
            s.to_string()
        }
    }

    fn line(r: &mut Rng, ctx: &Ctx, mask: u32) -> Vec<u8> {
        let f = [id(r, 'k'), id(r, 'u'), camp(r), tstamp(r, ctx)];
        let v: Vec<String> = (0..4).map(|j| q(r, &f[j], mask >> j & 1 == 1)).collect();
        v.join(",").into_bytes()
    }

    // ------------------------------------------------------------ tests

    /// All 16 quote combinations over random fields of every shape.
    #[test]
    fn k5test_quotes_and_shapes() {
        if !have_cpu() {
            return;
        }
        let mut r = Rng::new(seed0());
        let mut tot = 0;
        for (ei, ctx) in envs().iter().enumerate() {
            let mut lines = Vec::new();
            for i in 0..1600u32 {
                lines.push(line(&mut r, ctx, i % 16));
            }
            tot += check_lines(ctx, &lines, 400, CHUNKS_FEW, &format!("quotes env {}", ei));
        }
        eprintln!("k5test quotes_and_shapes: identical, {} records", tot);
        assert!(tot > 1000);
    }

    /// Valid lines at and around both window edges in every form.
    #[test]
    fn k5test_window_edges() {
        if !have_cpu() {
            return;
        }
        let mut tot = 0;
        for (ei, ctx) in envs().iter().enumerate() {
            let cws = ctx.ws - 7 * DAY;
            let mut lines = Vec::new();
            for (k, &ms) in [cws - 1000, cws - 1, cws, cws + 1, cws + 999, ctx.we - 1001, ctx.we - 1000, ctx.we - 1, ctx.we, ctx.we + 1].iter().enumerate() {
                for ts in [f1(ms), f2(ms, 0), f2(ms, 330), f2(ms, -600), f2(ms, 23 * 60 + 59), f2(ms, -(23 * 60 + 59)), f3(ms)] {
                    lines.push(format!("k{},u{},c_{},{}", k, k, 1 + k % 10, ts).into_bytes());
                    lines.push(format!("\"k{}\",\"u{}\",\"c_{}\",\"{}\"", k, k, 1 + k % 10, ts).into_bytes());
                }
            }
            tot += check_lines(ctx, &lines, 1000, CHUNKS, &format!("edges env {}", ei));
        }
        eprintln!("k5test window_edges: identical, {} records", tot);
        assert!(tot > 50);
    }

    /// Odd lines: \r, wrong field counts, quotes inside fields, empty
    /// lines, a 5000-byte ID, lines over 64 and over 128 bytes.
    #[test]
    fn k5test_odd_lines() {
        if !have_cpu() {
            return;
        }
        let big = "u".repeat(5000);
        let long = format!("u{}", "a".repeat(32));
        let mut lines: Vec<String> = vec![
            "k1,u1,c_1,2026-10-03T23:58:10Z\r".into(),
            "k1,u1,c_1,1759535890000\r".into(),
            "k1,u1,c_1".into(),
            "k1,u1,c_1,2026-10-03T23:58:10Z,x".into(),
            "k1,u1,c_1,2026-10-03T23:58:10Z,".into(),
            "".into(),
            ",,,".into(),
            ",u1,c_1,2026-10-03T23:58:10Z".into(),
            "k1,,c_1,2026-10-03T23:58:10Z".into(),
            "\"\",u1,c_1,2026-10-03T23:58:10Z".into(),
            "k1,\"\",c_1,2026-10-03T23:58:10Z".into(),
            "k\"1,u1,c_1,2026-10-03T23:58:10Z".into(),
            "\"k1,u1,c_1,2026-10-03T23:58:10Z".into(),
            "\"k1\"x,u1,c_1,2026-10-03T23:58:10Z".into(),
            "\"k\"\"1\",u1,c_1,2026-10-03T23:58:10Z".into(),
            "\"k,1\",u1,c_1,2026-10-03T23:58:10Z".into(),
            "\"\"\",u1,c_1,2026-10-03T23:58:10Z".into(),
            "k1,u1,\"c_1\",\"2026-10-03T23:58:10Z".into(),
            format!("k1,{},c_1,2026-10-03T23:58:10Z", big),
            format!("{},u1,c_1,2026-10-03T23:58:10Z", big),
            format!("k1,u1,c_1,{}", big),
            format!("k{},{},c_1,2026-10-03T23:58:10+05:30", "a".repeat(32), long),
            format!("\"k{}\",\"{}\",\"c_10\",\"2026-10-03T23:58:10+05:30\"", "a".repeat(32), long),
            format!("\"k{}\",\"{}x\",\"c_10\",\"2026-10-03T23:58:10+05:30\"", "a".repeat(40), long),
            format!("k{},{},c_1,2026-10-03T23:58:10Z", "b".repeat(60), "c".repeat(60)),
            format!("k{},{},c_1,2026-10-03T23:58:10Z", "b".repeat(62), "c".repeat(62)),
            format!("k{},u1,c_1,2026-10-03T23:58:10Z", "b".repeat(120)),
        ];
        // every line length from 40 to 140 around the 64 / 128 window edges
        for n in 0..100 {
            lines.push(format!("k{},u1,c_1,2026-10-03T23:58:10Z", "a".repeat(n)));
            lines.push(format!("\"k{}\",u1,\"c_1\",1759535890000", "a".repeat(n)));
        }
        let lines: Vec<Vec<u8>> = lines.into_iter().map(|s| s.into_bytes()).collect();
        let mut tot = 0;
        for (ei, ctx) in envs().iter().enumerate() {
            tot += check_lines(ctx, &lines, 40, CHUNKS, &format!("odd env {}", ei));
            // the same lines without header and filler: lines at the buffer ends
            for l in &lines {
                tot += check_text(ctx, &framed(std::slice::from_ref(l), true), CHUNKS_FEW, &format!("odd bare env {}", ei));
            }
        }
        eprintln!("k5test odd_lines: identical, {} records", tot);
    }

    /// Many random lines in long buffers (every chunking), including
    /// random bytes.
    #[test]
    fn k5test_random() {
        if !have_cpu() {
            return;
        }
        let mut r = Rng::new(seed0() ^ 77);
        let mut tot = 0;
        for (ei, ctx) in envs().iter().enumerate() {
            let mut lines = Vec::new();
            for _ in 0..4000 {
                let mask = (r.next() & 15) as u32 * (r.chance(0.6) as u32);
                let mut l = line(&mut r, ctx, mask);
                if r.chance(0.02) {
                    let i = r.below(l.len().max(1) as u64) as usize;
                    if i < l.len() {
                        l[i] = *r.pick(b",\"\n\rx\x00");
                    }
                }
                lines.push(l);
            }
            tot += check_lines(ctx, &lines, 2000, CHUNKS, &format!("random env {}", ei));
        }
        eprintln!("k5test random: identical, {} records", tot);
        assert!(tot > 1000);
    }

    /// pack2 agrees with keys::pack_wide on both lanes for every string.
    #[test]
    fn k5test_pack2() {
        if !have_cpu() {
            return;
        }
        let mut r = Rng::new(seed0() ^ 5);
        let al = b"-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz";
        for _ in 0..2_000_000 {
            let mut s = [[0u8; 32]; 2];
            let mut n = [0usize; 2];
            for k in 0..2 {
                n[k] = r.below(13) as usize;
                for i in 0..32 {
                    let x = r.next();
                    s[k][i] = if x % 8 == 0 { (x >> 8) as u8 } else { al[((x >> 8) % 64) as usize] };
                }
            }
            let (ka, oka, kb, okb) = unsafe { k5::pack2(s[0].as_ptr(), n[0], s[1].as_ptr(), n[1]) };
            let wa = crate::keys::pack(&s[0][..n[0]]);
            let wb = crate::keys::pack(&s[1][..n[1]]);
            assert_eq!(oka, wa.is_some(), "{:?}", &s[0][..n[0]]);
            assert_eq!(okb, wb.is_some(), "{:?}", &s[1][..n[1]]);
            if let Some(w) = wa {
                assert_eq!(ka, w);
            }
            if let Some(w) = wb {
                assert_eq!(kb, w);
            }
        }
    }

    /// The kernel's timestamp decode equals ctx.tsp.parse on random and
    /// perturbed timestamps of every form (via one-line tasks).
    #[test]
    fn k5test_timestamps() {
        if !have_cpu() {
            return;
        }
        let mut r = Rng::new(seed0() ^ 99);
        let mut tot = 0;
        for (ei, ctx) in envs().iter().enumerate() {
            let mut lines = Vec::new();
            for i in 0..20_000 {
                lines.push(format!("k{},u{},c_1,{}", i % 97, i % 31, tstamp(&mut r, ctx)).into_bytes());
            }
            tot += check_lines(ctx, &lines, 5000, &[1 << 40], &format!("ts env {}", ei));
        }
        eprintln!("k5test timestamps: identical, {} records", tot);
        assert!(tot > 10_000);
    }

    /// The group probe (k5::Probe) equals LinkTable::person_hash for linked
    /// and unlinked keys, on tables from tiny (long probe runs) to large.
    #[test]
    fn k5test_probe() {
        if !have_cpu() {
            return;
        }
        #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
        unsafe fn ph(p: &k5::Probe, k: u64, h: u64) -> u64 {
            p.person_hash(k, h)
        }
        let mut r = Rng::new(seed0() ^ 1234);
        for &npairs in &[1usize, 3, 10, 100, 1000, 50_000] {
            let t = LinkTable::with_pairs(npairs, 1);
            let mut alloc = crate::links::IdxAlloc::new();
            let key = |r: &mut Rng| (r.next() & !0xF) | (1 + r.below(10));
            let mut keys_in = Vec::new();
            for _ in 0..npairs {
                let a = key(&mut r);
                let b = if keys_in.is_empty() || r.chance(0.5) { key(&mut r) } else { *r.pick(&keys_in) };
                t.link(a, b, &mut alloc);
                keys_in.push(a);
                keys_in.push(b);
            }
            for b in 0..t.root_blocks() {
                t.finalize_roots(b);
            }
            for b in 0..t.val_blocks() {
                t.finalize_vals(b);
            }
            let p = k5::Probe::new(&t);
            for i in 0..200_000 {
                let k = if i % 2 == 0 && !keys_in.is_empty() { *r.pick(&keys_in) } else { key(&mut r) };
                let h = crate::keys::mix(k);
                assert_eq!(unsafe { ph(&p, k, h) }, t.person_hash(k, h), "key {:x} table of {} pairs", k, npairs);
            }
        }
        let e = LinkTable::empty();
        let p = k5::Probe::new(&e);
        assert_eq!(unsafe { ph(&p, 0x15, crate::keys::mix(0x15)) }, crate::keys::mix(0x15));
    }

    /// probe_view (the liftable group probe) against the table's own
    /// lookups: found iff person_of_h finds the key, and the same value
    /// through person_hash; adversarial keys (unmix of chosen hashes) pile
    /// dozens of keys into one probe group, including the last group of the
    /// table (wrap-around), next to random keys.
    #[test]
    fn k5test_probe_adversarial() {
        if !have_cpu() {
            return;
        }
        #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
        unsafe fn pv(t: &LinkTable, k: u64, h: u64) -> Option<u64> {
            let (slots, mask, amask, _) = t.view();
            assert!(mask != 0 && amask == !3);
            k5::probe_view(slots, mask, k, h)
        }
        #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
        unsafe fn ph(p: &k5::Probe, k: u64, h: u64) -> u64 {
            p.person_hash(k, h)
        }
        let mut r = Rng::new(seed0() ^ 4321);
        for &(npairs, crowd) in &[(4usize, 6usize), (16, 30), (64, 100), (5000, 300)] {
            let t = LinkTable::with_pairs(npairs + crowd, 1);
            let (_, mask, _, _) = t.view();
            let mut alloc = crate::links::IdxAlloc::new();
            // keys whose hashes all start in the last group (wrap-around)
            // or in group 0: unmix gives the key of a chosen hash
            let crowd_key = |r: &mut Rng, last: bool| -> u64 {
                loop {
                    let hh = (r.next() & !(mask as u64)) | if last { (mask & !3) as u64 | r.below(4) } else { r.below(4) };
                    let k = crate::keys::unmix(hh);
                    // packed-form keys only (low 4 bits 1..=10): the
                    // union-find orders keys, long / interned ones need tables
                    if (1..=10).contains(&(k & 0xF)) {
                        return k;
                    }
                }
            };
            let mut keys_in = Vec::new();
            for i in 0..crowd {
                let a = crowd_key(&mut r, i % 2 == 0);
                let b = crowd_key(&mut r, i % 3 == 0);
                t.link(a, b, &mut alloc);
                keys_in.push(a);
                keys_in.push(b);
            }
            let pk = |r: &mut Rng| (r.next() & !0xF) | (1 + r.below(10));
            for _ in 0..npairs {
                let a = pk(&mut r);
                let b = if r.chance(0.5) { pk(&mut r) } else { *r.pick(&keys_in) };
                t.link(a, b, &mut alloc);
                keys_in.push(a);
                keys_in.push(b);
            }
            for b in 0..t.root_blocks() {
                t.finalize_roots(b);
            }
            for b in 0..t.val_blocks() {
                t.finalize_vals(b);
            }
            let p = k5::Probe::new(&t);
            for i in 0..100_000 {
                let k = match i % 4 {
                    0 => *r.pick(&keys_in),
                    1 => {
                        let last = r.chance(0.5);
                        crowd_key(&mut r, last)
                    }
                    _ => pk(&mut r),
                };
                let h = crate::keys::mix(k);
                let got = unsafe { pv(&t, k, h) };
                assert_eq!(got.is_some(), t.person_of_h(k, h).is_some(), "key {:x}, {} pairs + {} crowded", k, npairs, crowd);
                assert_eq!(unsafe { ph(&p, k, h) }, t.person_hash(k, h), "key {:x}, {} pairs + {} crowded", k, npairs, crowd);
            }
        }
    }
}
