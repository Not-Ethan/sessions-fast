//! Adversarial tests of the k3 kernels (k3.rs) against densep's own kernels
//! (the oracle: `densep::ev_task_any` / `cl_task_any` with `k3::ON` false).
//!
//! Every test builds CSV text, places it in a buffer that ends exactly at a
//! PROT_NONE page (and, separately, one that starts right after a PROT_NONE
//! page), cuts it into tasks as `main::load_input` would at many chunk sizes
//! (one task per line up to one task for the whole buffer), runs both kernels
//! on every task and requires the same records: every bucket's DE and DP
//! sequence, the XE list and the interned exceptional user IDs (events), every
//! bucket's DC list and the interned IDs (clicks). A failure names the task,
//! the line(s) that differ on their own (exact bytes) and both record lists.
//!
//! k3 is x86-64 AVX2/BMI2 only: the tests exist on x86_64 and return early
//! (with a note) when the CPU lacks the features. `k3::ON` is never set: k3 is
//! called directly, the oracle runs because ON is false.
//!
//! Run (release, the x86 benchmark VM): `cargo test --release k3test`.

#[cfg(not(target_arch = "x86_64"))]
#[test]
fn k3test_skipped_not_x86() {
    eprintln!("k3test: k3 is x86-64 only; nothing to test on this architecture");
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use crate::dense::{unscramble, M1, M2, MASK};
    use crate::densep::{self, Dense, EvOut, Xmap, DC};
    use crate::k3::{self, Stats};
    use crate::lookup::IdTable;
    use crate::parts::Parts;
    use crate::{scan, Ctx, Task, DAY, LATE, WEEK};
    use std::ffi::c_void;

    const NPROD: usize = 3000;
    const NCAMP: usize = 100;

    // ------------------------------------------------------------ PRNG

    /// splitmix64 (deterministic, no crates).
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Rng {
            Rng(seed ^ 0x2545_F491_4F6C_DD1D)
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
        /// uniform in [lo, hi)
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

    // ------------------------------------------------------------ environment

    struct Env {
        ctx: Ctx,
        d: Dense,
    }

    fn ms_of(s: &str) -> i64 {
        crate::ts::parse_ts(s.as_bytes()).unwrap_or_else(|| panic!("bad test timestamp {}", s))
    }

    /// A Ctx as main::run builds it, for window [ws, we), NPROD products
    /// (p1.., 20 categories + unknown) and NCAMP campaigns (c_1..; with
    /// `hashed`, two non-canonical names too, so the table is a hash map).
    fn mk_ctx(ws: i64, we: i64, hashed: bool) -> Ctx {
        let first_day = ws.div_euclid(DAY);
        let ndays = ((we - 1).div_euclid(DAY) - first_day + 1) as usize;
        let mut cat_names: Vec<Vec<u8>> = (0..20).map(|i| format!("cat{:02}", i).into_bytes()).collect();
        cat_names.push(b"unknown".to_vec());
        let pentries: Vec<(Vec<u8>, u16)> = (1..=NPROD).map(|i| (format!("p{}", i).into_bytes(), (i % 20) as u16)).collect();
        let mut camp_ids: Vec<Vec<u8>> = (1..=NCAMP).map(|i| format!("c_{}", i).into_bytes()).collect();
        if hashed {
            camp_ids.push(b"organic".to_vec());
            camp_ids.push(b"x9".to_vec());
        }
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

    /// A Dense as densep::decide returns it (keys given).
    fn mk_dense(ke: u64, ku: u64, bound: u64, pmax: u64, bshift: u32) -> Dense {
        Dense { ke, ku, bound, pmax, n_est: bound, s_est: pmax * 8, lines_est: 1 << 16, pfrac: 0.03, bshift }
    }

    fn env_at(ws: &str, days: i64, bound: u64, pmax: u64, bshift: u32, hashed: bool) -> Env {
        let ws = ms_of(ws);
        Env { ctx: mk_ctx(ws, ws + days * DAY, hashed), d: mk_dense(0x5A_1234_C3E1 & MASK, 0x0F_1E2D_3C4B & MASK, bound, pmax, bshift) }
    }

    fn env_default() -> Env {
        env_at("2026-10-01T00:00:00Z", 7, 1 << 20, 1 << 14, 16, false)
    }

    /// Small buckets (bshift 10), a bound that is no power of two, other
    /// keys, a hashed campaign table.
    fn env_small_buckets() -> Env {
        let mut e = env_at("2026-10-01T00:00:00Z", 7, 777_777, 3000, 10, true);
        e.d.ke = 0xFF_FFFF_FFFF & MASK;
        e.d.ku = 1;
        e
    }

    /// The largest sizes densep::decide picks (bound 2^32, pmax 2^27): many
    /// more scrambled values are dense; bshift 20.
    fn env_big() -> Env {
        let mut e = env_at("2026-10-01T00:00:00Z", 7, 1 << 32, 1 << 27, 20, false);
        e.d.ke = 0x31_4159_2653 & MASK;
        e.d.ku = 0x27_1828_1828 & MASK;
        e
    }

    fn have_cpu() -> bool {
        let ok = crate::use_avx2();
        if !ok {
            eprintln!("k3test: no AVX2/BMI1/BMI2/LZCNT/POPCNT (or SS_NOAVX2 set): k3 tests skipped");
        }
        ok
    }

    // ------------------------------------------------------------ IDs, times

    /// The generator's scramble (densep::scramble).
    fn scramble(v: u64, key: u64) -> u64 {
        let mut m = (v & MASK).wrapping_mul(M1) & MASK;
        m = (m ^ key) & MASK;
        m ^= m >> 17;
        m.wrapping_mul(M2) & MASK
    }

    fn b36(mut v: u64) -> String {
        let mut s = Vec::new();
        loop {
            s.push(b"0123456789abcdefghijklmnopqrstuvwxyz"[(v % 36) as usize]);
            v /= 36;
            if v == 0 {
                break;
            }
        }
        s.reverse();
        String::from_utf8(s).unwrap()
    }

    fn eid(d: &Dense, n: u64) -> String {
        format!("e{}", b36(scramble(n, d.ke)))
    }

    fn uid(d: &Dense, u: u64) -> String {
        format!("u{}", b36(scramble(u, d.ku)))
    }

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

    /// `YYYY-MM-DDTHH:MM:SS` of the UTC time of ms (floored to the second).
    fn hms(ms: i64) -> String {
        let (y, m, d) = civil(ms.div_euclid(DAY));
        let r = ms.rem_euclid(DAY) / 1000;
        format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}", y, m, d, r / 3600, r / 60 % 60, r % 60)
    }

    fn f1(ms: i64) -> String {
        format!("{}Z", hms(ms))
    }

    /// F2 with offset `off` minutes (local = UTC + off).
    fn f2(ms: i64, off: i64) -> String {
        let s = if off < 0 { '-' } else { '+' };
        let a = off.abs();
        format!("{}{}{:02}:{:02}", hms(ms + off * 60_000), s, a / 60, a % 60)
    }

    fn f3(ms: i64) -> String {
        ms.to_string()
    }

    const OFFS: &[i64] = &[0, 45, 60, 120, 330, 345, 540, 600, 720, 1439, -60, -240, -300, -480, -570, -600, -1439];

    fn join(f: &[String], qmask: u32) -> Vec<u8> {
        let mut out = Vec::new();
        for (i, x) in f.iter().enumerate() {
            if i > 0 {
                out.push(b',');
            }
            let q = (qmask >> i) & 1 == 1;
            if q {
                out.push(b'"');
            }
            out.extend_from_slice(x.as_bytes());
            if q {
                out.push(b'"');
            }
        }
        out
    }

    fn join_raw(f: &[Vec<u8>]) -> Vec<u8> {
        let mut out = Vec::new();
        for (i, x) in f.iter().enumerate() {
            if i > 0 {
                out.push(b',');
            }
            out.extend_from_slice(x);
        }
        out
    }

    // ------------------------------------------------------------ generator

    struct Gen<'a> {
        e: &'a Env,
        r: Rng,
    }

    impl<'a> Gen<'a> {
        fn new(e: &'a Env, seed: u64) -> Gen<'a> {
            Gen { e, r: Rng::new(seed) }
        }

        fn user(&mut self) -> String {
            let d = self.e.d;
            let alias = if self.r.chance(0.6) { 0 } else { self.r.below(8) };
            let u = self.r.below(d.pmax) * 512 + alias;
            uid(&d, u)
        }

        fn ts_form(&mut self, t: i64) -> String {
            let x = self.r.below(100);
            if x < 55 {
                f1(t)
            } else if x < 67 {
                let o = *self.r.pick(OFFS);
                f2(t, o)
            } else {
                f3(t)
            }
        }

        /// A generator-shaped event line (valid or not by the usual rates).
        fn ev(&mut self) -> Vec<String> {
            let e = self.e;
            let d = e.d;
            let (ws, we) = (e.ctx.ws, e.ctx.we);
            let n = if self.r.chance(0.97) { self.r.below(d.bound) } else { self.r.below(MASK) };
            let t = self.r.range(ws, we);
            let ing = if self.r.chance(0.95) { t + self.r.range(-1500, 600_000) } else { t + self.r.range(LATE - 3000, LATE + 3000) };
            let ty = self.r.below(100);
            let (typ, price) = if ty < 10 {
                ("purchase", format!("{}", self.r.range(100, 50_001)))
            } else if ty < 30 {
                ("cart", String::new())
            } else {
                ("view", String::new())
            };
            let prod = if self.r.chance(0.97) { format!("p{}", self.r.range(1, NPROD as i64 + 1)) } else { format!("p{}", self.r.range(NPROD as i64 + 1, 99_999)) };
            let ts = self.ts_form(t);
            vec![eid(&d, n), self.user(), ts, f1(ing), typ.to_string(), prod, price]
        }

        /// A generator-shaped click line.
        fn cl(&mut self) -> Vec<String> {
            let e = self.e;
            let (ws, we) = (e.ctx.ws, e.ctx.we);
            let t = self.r.range(ws - WEEK, we);
            let mut ck = format!("k{}", b36(self.r.below(MASK)));
            if self.r.chance(0.03) {
                ck = ck.to_uppercase();
            }
            let camp = if self.r.chance(0.95) { format!("c_{}", self.r.range(1, NCAMP as i64 + 1)) } else { format!("c_{}", self.r.range(NCAMP as i64 + 1, 9999)) };
            let ts = self.ts_form(t);
            vec![ck, self.user(), camp, ts]
        }
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

    #[derive(PartialEq, Eq, Debug, Clone, Copy)]
    enum R {
        De { b: usize, n: u32, ingest: u32, ts: u32, us: u32 },
        Dp { b: usize, n: u32, ingest: u32, pay: u32 },
        Xe { key: u64, ingest: u32, valid: bool, us: u32, ts: u32, pay: u32 },
    }

    struct Outs {
        nb: usize,
        nbc: usize,
        a: EvOut,
        b: EvOut,
        ca: Parts<DC>,
        cb: Parts<DC>,
    }

    impl Outs {
        fn new(d: &Dense) -> Outs {
            let nb = (d.bound >> d.bshift) as usize + 1;
            let nbc = ((d.pmax * 8) >> 16) as usize + 1;
            let mk = || EvOut { de: Parts::with_wc(nb, 64, false), dp: Parts::with_wc(nb, 8, false), xe: Vec::new() };
            Outs { nb, nbc, a: mk(), b: mk(), ca: Parts::with_wc(nbc, 64, false), cb: Parts::with_wc(nbc, 64, false) }
        }
        fn reset(&mut self) {
            for o in [&mut self.a, &mut self.b] {
                o.de.reset();
                o.dp.reset();
                o.xe.clear();
            }
            self.ca.reset();
            self.cb.reset();
        }
    }

    /// Every bucket's DE sequence, every bucket's DP sequence, the XE list
    /// (exactly what k3::check_ev compares).
    fn ev_recs(o: &EvOut, nb: usize) -> Vec<R> {
        let mut v = Vec::new();
        for i in 0..nb {
            for r in o.de.get(i).iter().flat_map(|s| s.iter()) {
                v.push(R::De { b: i, n: r.n, ingest: r.ingest, ts: r.ts, us: r.us });
            }
        }
        for i in 0..nb {
            for r in o.dp.get(i).iter().flat_map(|s| s.iter()) {
                v.push(R::Dp { b: i, n: r.n, ingest: r.ingest, pay: r.pay });
            }
        }
        for x in &o.xe {
            v.push(R::Xe { key: x.key, ingest: x.ingest, valid: x.valid, us: x.us, ts: x.ts, pay: x.pay });
        }
        v
    }

    /// (bucket, us, camp, ts, ck) of every bucket in order (k3::dc_list).
    fn cl_recs(v: &Parts<DC>, nbc: usize) -> Vec<(usize, u32, u32, u32, u64)> {
        let mut out = Vec::new();
        for i in 0..nbc {
            for r in v.get(i).iter().flat_map(|s| s.iter()) {
                out.push((i, r.us, r.camp, r.ts, r.ck));
            }
        }
        out
    }

    fn add(a: &mut Stats, b: Stats) {
        a.fast += b.fast;
        a.line_fb += b.line_fb;
        a.range_fb += b.range_fb;
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
        if v.len() > 40 {
            format!("{:?} ... ({} records)", &v[..40], v.len())
        } else {
            format!("{:?}", v)
        }
    }

    #[derive(Clone, Copy, PartialEq)]
    enum Kind {
        Ev,
        Cl,
    }

    /// Both kernels on one task (fresh outputs) -> (old, k3) as debug text
    /// records and k3's stats.
    fn run_one(e: &Env, o: &mut Outs, kind: Kind, buf: &[u8], t: Task, xa: &Xmap, xb: &Xmap) -> (Vec<String>, Vec<String>, Stats) {
        o.reset();
        let safe_end = buf.as_ptr() as usize + buf.len();
        if kind == Kind::Ev {
            densep::ev_task_any(buf, t, &e.ctx, &e.d, xa, &mut o.a, safe_end);
            let st = k3::ev_task_k3(buf, t, &e.ctx, &e.d, xb, &mut o.b);
            let f = |v: Vec<R>| v.iter().map(|r| format!("{:?}", r)).collect::<Vec<_>>();
            (f(ev_recs(&o.a, o.nb)), f(ev_recs(&o.b, o.nb)), st)
        } else {
            densep::cl_task_any(buf, t, &e.ctx, &e.d, xa, &mut o.ca, o.nbc, safe_end);
            let st = k3::cl_task_k3(buf, t, &e.ctx, &e.d, xb, &mut o.cb, o.nbc);
            let f = |v: Vec<(usize, u32, u32, u32, u64)>| v.iter().map(|r| format!("{:?}", r)).collect::<Vec<_>>();
            (f(cl_recs(&o.ca, o.nbc)), f(cl_recs(&o.cb, o.nbc)), st)
        }
    }

    /// The lines of task t that differ on their own (fresh interning).
    fn culprits(e: &Env, o: &mut Outs, kind: Kind, buf: &[u8], t: Task) -> String {
        let mut s = String::new();
        let mut l = t.start;
        let mut shown = 0;
        while l < t.end && shown < 5 {
            let end = scan::next_line_start(buf, l).min(t.end);
            let one = Task { file: 0, start: l, end };
            let (xa, xb) = (Xmap::new(), Xmap::new());
            let (a, b, _) = run_one(e, o, kind, buf, one, &xa, &xb);
            if a != b || xa.key_table() != xb.key_table() {
                s += &format!(
                    "  line at {} ({} bytes before the buffer end): \"{}\"\n    old: {}\n    k3:  {}\n    old interned {:x?}, k3 interned {:x?}\n",
                    l,
                    buf.len() - l,
                    show(&buf[l..end]),
                    list(&a),
                    list(&b),
                    xa.key_table(),
                    xb.key_table()
                );
                shown += 1;
            }
            l = end;
        }
        if shown == 0 {
            s += "  (no single line differs on its own: the difference depends on the task context)\n";
        }
        s
    }

    /// Both kernels over `tasks` of buf (interning shared across the tasks,
    /// as in a run); Err with a report on the first difference.
    fn cmp(e: &Env, o: &mut Outs, kind: Kind, buf: &[u8], tasks: &[Task], what: &str) -> Result<Stats, String> {
        let (xa, xb) = (Xmap::new(), Xmap::new());
        let mut st = Stats::default();
        for &t in tasks {
            let (a, b, s) = run_one(e, o, kind, buf, t, &xa, &xb);
            add(&mut st, s);
            if a != b {
                let mut rep = format!(
                    "k3test {}: {} MISMATCH in task [{}, {}) of a {}-byte buffer\n  task records old: {}\n  task records k3:  {}\n",
                    what,
                    if kind == Kind::Ev { "EVENT" } else { "CLICK" },
                    t.start,
                    t.end,
                    buf.len(),
                    list(&a),
                    list(&b)
                );
                rep += &culprits(e, o, kind, buf, t);
                if t.end - t.start <= 2000 {
                    rep += &format!("  task text: \"{}\"\n", show(&buf[t.start..t.end]));
                }
                return Err(rep);
            }
        }
        if xa.key_table() != xb.key_table() {
            return Err(format!("k3test {}: MISMATCH of interned user IDs: old {:x?} k3 {:x?}", what, xa.key_table(), xb.key_table()));
        }
        Ok(st)
    }

    /// Tasks as main::load_input cuts them from `start` with this chunk.
    fn chunk_tasks(b: &[u8], start: usize, chunk: usize) -> Vec<Task> {
        let mut v = Vec::new();
        let mut s = start;
        while s < b.len() {
            let nominal = s + chunk;
            let e = if nominal >= b.len() { b.len() } else { scan::next_line_start(b, nominal - 1) };
            v.push(Task { file: 0, start: s, end: e });
            s = e;
        }
        v
    }

    /// chunk 1 = one task per line; the last = the whole buffer.
    const CHUNKS: &[usize] = &[1, 2, 7, 64, 100, 191, 192, 193, 300, 777, 4096, 1 << 40];
    const CHUNKS_FEW: &[usize] = &[1, 300, 1 << 40];

    /// Text in both guarded placements, every chunking; panics with the
    /// report on a difference. Returns k3's stats of the whole-buffer task
    /// (end-aligned placement).
    fn check_text(e: &Env, o: &mut Outs, kind: Kind, text: &[u8], chunks: &[usize], what: &str) -> Stats {
        let mut whole = Stats::default();
        for at_end in [true, false] {
            let g = GBuf::new(text, at_end);
            let b = g.bytes();
            for &c in chunks {
                let tasks = chunk_tasks(b, 0, c);
                match cmp(e, o, kind, b, &tasks, &format!("{} [{}, chunk {}]", what, if at_end { "end at guard" } else { "start after guard" }, c)) {
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

    /// Lines (no newline at their end) between a 40-byte front filler and a
    /// 256-byte back filler, so that every line can take the fast path.
    fn framed(lines: &[Vec<u8>]) -> Vec<u8> {
        let mut t = Vec::new();
        t.extend_from_slice(b"event_id,user_id,ts,ingest_ts,event_type,\n");
        for l in lines {
            t.extend_from_slice(l);
            t.push(b'\n');
        }
        for _ in 0..4 {
            t.extend_from_slice(&[b'x'; 63]);
            t.push(b'\n');
        }
        t
    }

    /// Lines in batches of `per` (each batch framed), every chunking.
    fn check_lines(e: &Env, kind: Kind, lines: &[Vec<u8>], per: usize, chunks: &[usize], what: &str) -> Stats {
        let mut o = Outs::new(&e.d);
        let mut tot = Stats::default();
        for (i, part) in lines.chunks(per.max(1)).enumerate() {
            let st = check_text(e, &mut o, kind, &framed(part), chunks, &format!("{} batch {}", what, i));
            add(&mut tot, st);
        }
        eprintln!(
            "k3test {}: {} lines identical; k3 whole-buffer stats: fast {} line fallback {} range fallback {}",
            what,
            lines.len(),
            tot.fast,
            tot.line_fb,
            tot.range_fb
        );
        tot
    }

    // ------------------------------------------------------------ case builders

    fn s(x: &str) -> String {
        x.to_string()
    }

    fn with(f: &[String], i: usize, v: &str) -> Vec<String> {
        let mut g = f.to_vec();
        g[i] = v.to_string();
        g
    }

    const PRICES: &[&str] = &[
        "", "1", "250", "0250", "10000000", "10000001", "0", "00", "-5", "+5", "19.99", "1e3", " 250", "250 ", "1000\r", "00000000000010000000",
        "4294967296", "99999999999999999999", "\"250\"", "25 0", "٢٥٠", "1,0",
    ];
    const TYPES: &[&str] = &["view", "cart", "purchase", "Purchase", "PURCHASE", "View", "cart ", " cart", "click", "", "views", "purchas", "purchasee", "vie", "car", "purchas\r", "view\r"];

    /// Quote anomalies and field-count changes of one line's fields.
    fn quote_variants(f: &[String], out: &mut Vec<Vec<u8>>, r: &mut Rng) {
        let nf = f.len();
        let j = r.below(nf as u64) as usize;
        let raw: Vec<Vec<u8>> = f.iter().map(|x| x.as_bytes().to_vec()).collect();
        let mut v = |g: &dyn Fn(&mut Vec<Vec<u8>>)| {
            let mut x = raw.clone();
            g(&mut x);
            out.push(join_raw(&x));
        };
        v(&|x| x[j].insert(0, b'"'));
        v(&|x| x[j].push(b'"'));
        v(&|x| {
            x[j].insert(0, b'"');
            x[j].extend_from_slice(b"\"\"");
        });
        v(&|x| {
            let k = x[j].len().min(1);
            x[j].insert(k, b'"');
        });
        v(&|x| x[j] = b"\"\"".to_vec());
        v(&|x| x[j] = b"\"".to_vec());
        v(&|x| x[j] = b"\"\"\"\"".to_vec());
        v(&|x| {
            let mut y = b"\"".to_vec();
            y.extend_from_slice(&x[j]);
            y.extend_from_slice(b",x\"");
            x[j] = y;
        });
        v(&|x| {
            let mut y = b"\"".to_vec();
            y.extend_from_slice(&x[j]);
            y.extend_from_slice(b"\nx\"");
            x[j] = y;
        });
        v(&|x| {
            let mut y = b"\"".to_vec();
            y.extend_from_slice(&x[j]);
            y.extend_from_slice(b"\"\"x\"");
            x[j] = y;
        });
        v(&|x| {
            let mut y = b"\"".to_vec();
            y.extend_from_slice(&x[j]);
            y.extend_from_slice(b"\"x");
            x[j] = y;
        });
        v(&|x| {
            let mut y = b" \"".to_vec();
            y.extend_from_slice(&x[j]);
            y.push(b'"');
            x[j] = y;
        });
        v(&|x| {
            x.pop();
        });
        v(&|x| x.push(b"x".to_vec()));
        v(&|x| x.push(Vec::new()));
        v(&|x| {
            x.push(Vec::new());
            x.push(Vec::new());
        });
        v(&|x| x.insert(j, Vec::new()));
        v(&|x| {
            x.remove(j);
        });
        v(&|x| x[j] = Vec::new());
        v(&|x| x[j].push(b'\r'));
        v(&|x| x.last_mut().unwrap().push(b'\r'));
        v(&|x| {
            let rep = x[j].repeat(1 + j * 3);
            x[j] = rep;
        });
        v(&|x| x[j] = vec![b'7'; 300]);
        v(&|x| x[j] = vec![b'a'; 5000]);
        v(&|x| x[j] = vec![b' '; 40]);
        v(&|x| x.truncate(1));
        v(&|x| {
            let mut y = join_raw(x);
            let k = y.len() / 2;
            y.truncate(k);
            *x = vec![y];
        });
    }

    fn ev_valid_cases(e: &Env, seed: u64, nbase: usize) -> Vec<Vec<u8>> {
        let mut g = Gen::new(e, seed);
        let d = e.d;
        let mut out: Vec<Vec<u8>> = Vec::new();
        for i in 0..nbase {
            let f = g.ev();
            out.push(join(&f, 0));
            // a random quoting pattern, and all 128 for the first lines
            out.push(join(&f, g.r.below(128) as u32));
            if i < 6 {
                for m in 0..128 {
                    out.push(join(&f, m));
                }
            }
            let t = e.ctx.ws + g.r.below((e.ctx.we - e.ctx.ws) as u64) as i64;
            // ts in every form
            let mut tsv = vec![f1(t), f3(t), format!("000{}", t), format!("{:016}", t), f3(t / 10), f3(t * 10), f3(t - t % 1000)];
            for &o in OFFS {
                tsv.push(f2(t, o));
            }
            tsv.push(format!("{}-00:00", hms(t)));
            for v in &tsv {
                out.push(join(&with(&f, 2, v), 0));
            }
            // ingest_ts in other forms
            let ing = t + g.r.range(0, 600_000);
            for v in [f2(ing, 330), f2(ing, -600), f3(ing), f3(ing - ing % 1000), format!("{:016}", ing)] {
                let mut h = with(&f, 2, &f1(t));
                h[3] = v;
                out.push(join(&h, 0));
            }
            // types and prices
            for &ty in TYPES {
                if g.r.chance(0.3) {
                    out.push(join(&with(&f, 4, ty), 0));
                }
            }
            for &pr in PRICES {
                let mut h = with(&f, 4, "purchase");
                h[6] = s(pr);
                out.push(join(&h, 0));
                if g.r.chance(0.2) {
                    h[4] = s("view");
                    out.push(join(&h, 0));
                }
            }
            for pv in ["", "p1", "p3000", "p3001", "P1", "p0", "p01", "p", "p 1", "p99999999", "p1 ", "c_1"] {
                let mut h = with(&f, 4, "purchase");
                h[5] = s(pv);
                h[6] = s("250");
                out.push(join(&h, 0));
            }
            // empty fields
            for j in 0..7 {
                out.push(join(&with(&f, j, ""), 0));
                out.push(join(&with(&f, j, "\"\""), 0));
            }
            // a different (exceptional) user for a dense event
            out.push(join(&with(&f, 1, &format!("{}x", f[1])), 0));
            out.push(join(&with(&f, 1, &uid(&d, g.r.below(d.pmax) * 512 + 8)), 0));
            quote_variants(&f, &mut out, &mut g.r);
            out.push(join(&f, 0).iter().chain(b"\r").copied().collect());
        }
        out.push(b"event_id,user_id,ts,ingest_ts,event_type,product_id,price_cents".to_vec());
        out.push(Vec::new());
        out.push(Vec::new());
        out.push(b"\r".to_vec());
        out.push(b",,,,,,".to_vec());
        out.push(b"\"\",\"\",\"\",\"\",\"\",\"\",\"\"".to_vec());
        out
    }

    fn id_variants(r: &mut Rng, prefix: u8, real: &str) -> Vec<String> {
        let p = (prefix as char).to_string();
        let rest = &real[1..];
        let mut v = vec![
            s(""),
            p.clone(),
            format!("{}0", p),
            format!("{}00", p),
            format!("{}0{}", p, rest),
            format!("{}00{}", p, rest),
            real.to_uppercase(),
            format!("{}{}", p.to_uppercase(), rest),
            format!("{}{}", p, rest.to_uppercase()),
            format!(" {}", real),
            format!("{} ", real),
            format!("  {}  ", real),
            real[..real.len() - 1].to_string(),
            format!("{}0", real),
            format!("{}z", real),
            format!("{}{}", p, "z".repeat(8)),
            format!("{}{}", p, "z".repeat(9)),
            format!("{}1{}", p, "0".repeat(7)),
            format!("{}1{}", p, "0".repeat(8)),
            format!("{}ffffffff", p),
            format!("{}fzzzzzzz", p),
            format!("{}g0000000", p),
            format!("{}-_AZaz09", p),
            format!("{}ab_cd", p),
            format!("{}ab-cd", p),
            format!("{}ab.cd", p),
            format!("{}ab cd", p),
            format!("{}ab:cd", p),
            format!("{}ab/cd", p),
            format!("{}{}", if prefix == b'e' { "u" } else { "e" }, rest),
            format!("x{}", rest),
            format!("{}{}", p, real),
            format!("{}{}", rest, p),
            b36(MASK),
            format!("{}{}", p, b36(MASK)),
            format!("{}{}", p, b36(MASK + 1)),
            format!("{}{}", p, b36(MASK - 1)),
            format!("{}{}", p, b36(1u64 << 39)),
            format!("{}{}", p, b36(36u64.pow(8) - 1)),
            format!("{}{}", p, b36(36u64.pow(7))),
            format!("{}{}", p, b36(36u64.pow(7) - 1)),
            format!("{}1", p),
            format!("{}z", p),
            format!("{}0{}", p, "0".repeat(10)),
        ];
        for n in 0..=12 {
            let digits: String = (0..n).map(|_| *r.pick(b"0123456789abcdefghijklmnopqrstuvwxyz") as char).collect();
            v.push(format!("{}{}", p, digits));
            let nz: String = (0..n).map(|i| if i == 0 { *r.pick(b"123456789abcdefghijklmnopqrstuvwxyz") as char } else { *r.pick(b"0123456789abcdefghijklmnopqrstuvwxyz") as char }).collect();
            v.push(format!("{}{}", p, nz));
        }
        v
    }

    /// Byte-level variants of an ID (bytes >= 0x80 and control bytes).
    fn id_byte_variants(real: &str) -> Vec<Vec<u8>> {
        let mut v = Vec::new();
        let b = real.as_bytes();
        for &c in &[0x00u8, 0x80, 0xE9, 0xFF, b'\t', b'"', b'\r', 0x7F, b'0' - 1, b'9' + 1, b'a' - 1, b'z' + 1, b'A', b'Z'] {
            for i in 1..b.len().min(4) {
                let mut x = b.to_vec();
                x[i] = c;
                v.push(x);
            }
            let mut x = b.to_vec();
            x.push(c);
            v.push(x);
        }
        v
    }

    fn ev_id_cases(e: &Env, seed: u64, nbase: usize) -> Vec<Vec<u8>> {
        let mut g = Gen::new(e, seed);
        let d = e.d;
        let mut out = Vec::new();
        // event numbers at and around the bound, users at the dense limits
        let ns = [0, 1, 35, 36, d.bound - 1, d.bound, d.bound + 1, d.bound * 2, MASK, MASK - 1, 1 << 39, (1 << 39) - 1];
        let us = [
            0,
            1,
            7,
            8,
            9,
            511,
            512,
            (d.pmax - 1) * 512,
            (d.pmax - 1) * 512 + 7,
            (d.pmax - 1) * 512 + 8,
            d.pmax * 512,
            d.pmax * 512 + 1,
            (1 << 39) - 1,
            1 << 39,
            MASK,
            (1 << 36) + 3,
        ];
        for i in 0..nbase {
            let f = g.ev();
            let valid = with(&with(&f, 4, "view"), 2, &f1(e.ctx.ws + 3600_000));
            let invalid = with(&valid, 2, &f1(e.ctx.we + 3600_000));
            for base in [&valid, &invalid, &f] {
                for v in id_variants(&mut g.r, b'e', &base[0]) {
                    out.push(join(&with(base, 0, &v), 0));
                }
                for x in id_byte_variants(&base[0]) {
                    let mut raw: Vec<Vec<u8>> = base.iter().map(|y| y.as_bytes().to_vec()).collect();
                    raw[0] = x;
                    out.push(join_raw(&raw));
                }
            }
            for v in id_variants(&mut g.r, b'u', &valid[1]) {
                out.push(join(&with(&valid, 1, &v), 0));
                if i % 4 == 0 {
                    out.push(join(&with(&with(&valid, 4, "purchase"), 1, &v), 0));
                }
            }
            for x in id_byte_variants(&valid[1]) {
                let mut raw: Vec<Vec<u8>> = valid.iter().map(|y| y.as_bytes().to_vec()).collect();
                raw[1] = x;
                out.push(join_raw(&raw));
            }
            if i < 8 {
                for &n in &ns {
                    out.push(join(&with(&valid, 0, &eid(&d, n)), 0));
                    out.push(join(&with(&invalid, 0, &eid(&d, n)), 0));
                }
                for &u in &us {
                    out.push(join(&with(&valid, 1, &uid(&d, u)), 0));
                    out.push(join(&with(&with(&valid, 1, &uid(&d, u)), 4, "purchase"), 0x2));
                }
            }
            // quoted IDs
            out.push(join(&valid, 0b11));
            out.push(join(&invalid, 0b01));
        }
        out
    }

    /// Timestamp values: window edges, late limits, out-of-table dates,
    /// every single-byte change and every invalid form of rules.md.
    fn ts_values(lo: i64, we: i64) -> Vec<String> {
        let mut out = Vec::new();
        let mid = (lo + we) / 2;
        let edges = [lo - 1000, lo - 1, lo, lo + 1, lo + 999, lo + 1000, we - 1000, we - 1, we, we + 1, we + 1000, mid, mid + 123, lo - 2 * DAY, lo - 3 * DAY, we + 20 * DAY, we + 30 * DAY];
        for &t in &edges {
            let sec = t - t.rem_euclid(1000);
            out.push(f3(t));
            out.push(format!("{:016}", t));
            out.push(format!("000{}", t));
            out.push(f3(t)[..f3(t).len() - 1].to_string());
            out.push(format!("{}0", t));
            out.push(f1(sec));
            for o in [0, 330, -600, 720, -570, 45, 1439, -1439] {
                out.push(f2(sec, o));
            }
        }
        let base = f1(mid - mid.rem_euclid(1000));
        let b19 = &base[..19];
        let b11 = &base[..11];
        out.extend([
            s(""),
            base.to_lowercase(),
            base.replace('T', " "),
            b19.to_string(),
            format!("{}z", b19),
            base.replace('T', "t"),
            format!("{}.000Z", b19),
            format!("{}.5Z", b19),
            format!("{}+0200", b19),
            format!("{}+24:00", b19),
            format!("{}-24:00", b19),
            format!("{}+05:60", b19),
            format!("{}+23:59", b19),
            format!("{}-23:59", b19),
            format!("{}-00:00", b19),
            format!("{}+00:00", b19),
            format!("{}+5:30", b19),
            format!("{}+05:3", b19),
            format!("{}+05:30x", b19),
            format!("{}+05:30Z", b19),
            format!("{}Z+05:30", b19),
            format!("{}24:00:00Z", b11),
            format!("{}23:60:00Z", b11),
            format!("{}23:59:60Z", b11),
            format!("{}23:59:59Z", b11),
            format!("{}00:00:00Z", b11),
            format!("{}99:99:99Z", b11),
            format!("{}13{}", &base[..5], &base[7..]),
            format!("{}00{}", &base[..5], &base[7..]),
            format!("{}32{}", &base[..8], &base[10..]),
            format!("{}00{}", &base[..8], &base[10..]),
            format!("{}31{}", &base[..8], &base[10..]),
            format!("{}02-30{}", &base[..5], &base[10..]),
            format!("{}02-29{}", &base[..5], &base[10..]),
            format!("{}04-31{}", &base[..5], &base[10..]),
            s("2028-02-29T12:00:00Z"),
            s("2027-02-29T12:00:00Z"),
            s("2100-02-29T12:00:00Z"),
            s("2000-02-29T12:00:00Z"),
            s("2028-02-29T12:00:00+01:00"),
            s("2026-12-31T23:59:59Z"),
            s("2027-01-01T00:00:00Z"),
            s("2026-12-31T23:30:00-01:00"),
            s("1970-01-01T00:00:00Z"),
            s("0000-01-01T00:00:00Z"),
            s("9999-12-31T23:59:59Z"),
            s("1.79e12"),
            s("-5"),
            s("-1791071890000"),
            s("+1791071890000"),
            s("1791071890000.0"),
            s("86400000"),
            s("0"),
            s("0000000000000"),
            s("9999999999999"),
            s("9223372036854775807"),
            s("9223372036854775808"),
            s("99999999999999999999"),
            s("18446744073709551616"),
            format!("{}\r", base),
            format!(" {}", base),
            format!("{} ", base),
            format!("\"{}\"", base),
            s("１７９１０７１８９００００"),
            base.replace('-', "/"),
            base.replace(':', "."),
        ]);
        // every single-byte change of an F1, an F2 and an F3 value
        let f2v = f2(mid - mid.rem_euclid(1000), 330);
        let f3v = f3(mid);
        for v in [base.clone(), f2v, f3v] {
            for i in 0..v.len() {
                for c in "0129:-+TZzt/ a".chars() {
                    let mut x: Vec<char> = v.chars().collect();
                    x[i] = c;
                    out.push(x.into_iter().collect());
                }
                let mut x = v.clone();
                x.remove(i);
                out.push(x);
            }
        }
        out
    }

    fn ev_ts_cases(e: &Env, seed: u64, nbase: usize) -> Vec<Vec<u8>> {
        let mut g = Gen::new(e, seed);
        let (ws, we) = (e.ctx.ws, e.ctx.we);
        let tsv = ts_values(ws, we);
        let mut out = Vec::new();
        for i in 0..nbase {
            let f = g.ev();
            let t0 = ws + 3 * DAY + 1000 * g.r.range(0, 86_400);
            let valid = with(&with(&with(&f, 2, &f1(t0)), 3, &f1(t0 + 5000)), 4, if i % 3 == 0 { "purchase" } else { "view" });
            let valid = with(&valid, 6, "250");
            // ts values, ingest a little later than ts where ts parses
            for v in &tsv {
                let mut h = with(&valid, 2, v);
                if let Some(t) = crate::ts::parse_ts(v.as_bytes()) {
                    if (ws - 30 * DAY..we + 30 * DAY).contains(&t) {
                        h[3] = f1(t + 2000);
                    }
                }
                out.push(join(&h, 0));
                if g.r.chance(0.1) {
                    out.push(join(&h, 1 << 2));
                }
            }
            // ingest_ts values
            for v in tsv.iter().step_by(if i < 4 { 1 } else { 7 }) {
                out.push(join(&with(&valid, 3, v), 0));
                if g.r.chance(0.1) {
                    out.push(join(&with(&valid, 3, v), 1 << 3));
                }
            }
            // lateness: ingest - ts at and around LATE, in every ts form
            for dl in [LATE - 1000, LATE - 1, LATE, LATE + 1, LATE + 999, LATE + 1000, LATE + 2000, -1000, -LATE, 0, 5 * DAY] {
                let ing = t0 + 1000 * g.r.range(0, 10);
                let t = ing - dl;
                for ts in [f1(t), f3(t), f2(t, 330), f2(t, -600)] {
                    let h = with(&with(&valid, 2, &ts), 3, &f1(ing));
                    out.push(join(&h, 0));
                    out.push(join(&with(&h, 4, "view"), 0));
                }
            }
            // ts at the window edges with ingest at the late limit
            for t in [ws, ws + 1, we - 1, we - 1000] {
                for ing in [t, t + LATE, t + LATE + 1000, t - t.rem_euclid(1000) + LATE] {
                    for ts in [f1(t), f3(t), f2(t, 720), f2(t, -570)] {
                        out.push(join(&with(&with(&valid, 2, &ts), 3, &f1(ing)), 0));
                    }
                }
            }
            // ingest dates around the 28-day table (from ws - 2 days)
            for dd in [-3 * DAY, -2 * DAY - 1000, -2 * DAY, 25 * DAY, 26 * DAY - 1000, 26 * DAY, 27 * DAY, 30 * DAY, 400 * DAY, -400 * DAY] {
                let ing = ws + dd;
                out.push(join(&with(&valid, 3, &f1(ing)), 0));
                out.push(join(&with(&with(&valid, 3, &f1(ing)), 2, &f1(ing - 3600_000)), 0));
            }
        }
        out
    }

    fn mutate(r: &mut Rng, l: &[u8]) -> Vec<u8> {
        const SET: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz,\"\n\r -+:.TZ";
        let mut x = l.to_vec();
        let k = 1 + r.below(3) as usize;
        for _ in 0..k {
            let c = if r.chance(0.75) { *r.pick(SET) } else { r.below(256) as u8 };
            let i = r.below(x.len() as u64 + 1) as usize;
            match r.below(10) {
                0..=5 => {
                    if i < x.len() {
                        x[i] = c;
                    } else {
                        x.push(c);
                    }
                }
                6..=8 => x.insert(i, c),
                _ => {
                    if i < x.len() {
                        x.remove(i);
                    }
                }
            }
        }
        x
    }

    /// 1..3 changes that keep the line's shape (a digit for a digit, a
    /// base-36 letter for a base-36 character, a sign for a sign), so that
    /// most lines stay on k3's fast path with other values (hour 29, day 00,
    /// IDs with a leading zero, offsets +19:77, ...).
    fn mutate_value(r: &mut Rng, l: &[u8]) -> Vec<u8> {
        let mut x = l.to_vec();
        let k = 1 + r.below(3) as usize;
        for _ in 0..k {
            let i = r.below(x.len() as u64) as usize;
            let c = x[i];
            x[i] = match c {
                b'0'..=b'9' => {
                    if r.chance(0.8) {
                        b'0' + r.below(10) as u8
                    } else {
                        *r.pick(b"abcxyz:-+ ")
                    }
                }
                b'a'..=b'z' => *r.pick(b"0123456789abcdefghijklmnopqrstuvwxyz"),
                b'+' | b'-' => *r.pick(b"+-:0Z"),
                b':' => *r.pick(b":-+0"),
                b'T' => *r.pick(b"Tt 0"),
                b'Z' => *r.pick(b"Zz0+"),
                b',' => *r.pick(b",,,;\""),
                b'"' => *r.pick(b"\"\"\"' "),
                _ => c,
            };
        }
        x
    }

    // ------------------------------------------------------------ tests: harness

    /// The guard page faults on a read one byte past the buffer end and one
    /// byte before the start (checked in a forked child).
    #[test]
    fn k3test_guard_pages_fault() {
        for at_end in [true, false] {
            let g = GBuf::new(b"abc\n", at_end);
            let p = if at_end { unsafe { g.bytes().as_ptr().add(4) } } else { unsafe { g.bytes().as_ptr().sub(1) } };
            let pid = unsafe { fork() };
            assert!(pid >= 0, "fork failed");
            if pid == 0 {
                unsafe {
                    let v = std::ptr::read_volatile(p);
                    _exit(v as i32 & 1);
                }
            }
            let mut st = 0i32;
            unsafe { waitpid(pid, &mut st, 0) };
            let sig = st & 0x7f;
            assert!(sig == 11 || sig == 10, "read outside the guarded buffer did not fault (status {:#x})", st);
        }
    }

    /// IDs and timestamps as the tests write them decode as intended, and
    /// generator-shaped lines take k3's fast path (the tests reach it).
    #[test]
    fn k3test_harness_sanity() {
        let e = env_default();
        let d = e.d;
        for n in [0u64, 1, 12345, d.bound - 1, MASK] {
            assert_eq!(unscramble(scramble(n, d.ke), d.ke), n);
            assert_eq!(crate::dense::decode36(eid(&d, n).as_bytes(), b'e').map(|v| unscramble(v, d.ke)), Some(n));
        }
        for t in [e.ctx.ws, e.ctx.ws + 123_456_000, e.ctx.we - 1000] {
            assert_eq!(crate::ts::parse_ts(f1(t).as_bytes()), Some(t));
            for &o in OFFS {
                assert_eq!(crate::ts::parse_ts(f2(t, o).as_bytes()), Some(t), "{}", f2(t, o));
            }
            assert_eq!(crate::ts::parse_ts(f3(t).as_bytes()), Some(t));
        }
        if !have_cpu() {
            return;
        }
        let mut g = Gen::new(&e, 7);
        let mut lines = Vec::new();
        while lines.len() < 2000 {
            let f = g.ev();
            if f[4] != "purchase" && !f[2].contains('+') {
                lines.push(join(&f, 0));
            }
        }
        let st = check_lines(&e, Kind::Ev, &lines, 2000, CHUNKS_FEW, "sanity ev");
        assert!(st.fast >= 1900, "generator-shaped event lines should take k3's fast path: {:?}", st);
        let mut cl = Vec::new();
        for _ in 0..2000 {
            cl.push(join(&g.cl(), 0));
        }
        let st = check_lines(&e, Kind::Cl, &cl, 2000, CHUNKS_FEW, "sanity cl");
        assert!(st.fast >= 1800, "generator-shaped click lines should take k3's fast path: {:?}", st);
    }

    // ------------------------------------------------------------ tests: events

    #[test]
    fn k3test_ev_valid_forms() {
        if !have_cpu() {
            return;
        }
        let e = env_default();
        let lines = ev_valid_cases(&e, 11, 150);
        check_lines(&e, Kind::Ev, &lines, 400, CHUNKS, "ev valid forms");
        let e2 = env_small_buckets();
        let lines = ev_valid_cases(&e2, 12, 60);
        check_lines(&e2, Kind::Ev, &lines, 400, CHUNKS_FEW, "ev valid forms (small buckets)");
    }

    #[test]
    fn k3test_ev_id_shapes() {
        if !have_cpu() {
            return;
        }
        let e = env_default();
        let lines = ev_id_cases(&e, 21, 40);
        check_lines(&e, Kind::Ev, &lines, 400, CHUNKS, "ev id shapes");
        let e2 = env_small_buckets();
        let lines = ev_id_cases(&e2, 22, 20);
        check_lines(&e2, Kind::Ev, &lines, 400, CHUNKS_FEW, "ev id shapes (small buckets)");
    }

    #[test]
    fn k3test_ev_timestamps() {
        if !have_cpu() {
            return;
        }
        // windows: plain, not at midnight, over Feb 29, over the year end,
        // 1 day, 23 days
        let envs = [
            env_default(),
            env_at("2026-10-01T05:30:17Z", 7, 1 << 20, 1 << 14, 16, false),
            env_at("2028-02-25T00:00:00Z", 7, 1 << 20, 1 << 14, 16, false),
            env_at("2026-12-28T00:00:00Z", 7, 1 << 20, 1 << 14, 16, false),
            env_at("2026-10-01T00:00:00Z", 1, 1 << 20, 1 << 14, 16, false),
            env_at("2026-10-01T00:00:00Z", 23, 1 << 20, 1 << 14, 16, false),
        ];
        for (k, e) in envs.iter().enumerate() {
            let lines = ev_ts_cases(e, 31 + k as u64, if k == 0 { 12 } else { 4 });
            check_lines(e, Kind::Ev, &lines, 400, if k == 0 { CHUNKS } else { CHUNKS_FEW }, &format!("ev timestamps (window {})", k));
        }
    }

    #[test]
    fn k3test_ev_mutations() {
        if !have_cpu() {
            return;
        }
        for (k, e) in [env_default(), env_small_buckets()].iter().enumerate() {
            let mut g = Gen::new(e, 41 + k as u64);
            let mut r = Rng::new(141 + k as u64);
            let n = if k == 0 { 600_000 } else { 150_000 };
            let mut lines = Vec::with_capacity(n);
            while lines.len() < n {
                let f = g.ev();
                let l = join(&f, if r.chance(0.2) { r.below(128) as u32 } else { 0 });
                lines.push(mutate(&mut r, &l));
            }
            check_lines(e, Kind::Ev, &lines, 1000, &[1, 193, 4096, 1 << 40], &format!("ev mutations (env {})", k));
        }
    }

    /// Shape-preserving value mutations (most lines stay on the fast path).
    #[test]
    fn k3test_ev_value_mutations() {
        if !have_cpu() {
            return;
        }
        for (k, e) in [env_default(), env_small_buckets(), env_at("2028-02-26T00:00:00Z", 5, 1 << 20, 1 << 14, 16, false)].iter().enumerate() {
            let mut g = Gen::new(e, 241 + k as u64);
            let mut r = Rng::new(341 + k as u64);
            let n = if k == 0 { 400_000 } else { 100_000 };
            let mut lines = Vec::with_capacity(n);
            while lines.len() < n {
                let mut f = g.ev();
                if r.chance(0.3) {
                    f[4] = s("purchase");
                    f[6] = format!("{}", r.range(1, 10_000_001));
                }
                let l = join(&f, if r.chance(0.1) { r.below(128) as u32 } else { 0 });
                lines.push(mutate_value(&mut r, &l));
            }
            check_lines(e, Kind::Ev, &lines, 1000, &[1, 4096, 1 << 40], &format!("ev value mutations (env {})", k));
        }
    }

    #[test]
    fn k3test_cl_value_mutations() {
        if !have_cpu() {
            return;
        }
        for (k, e) in [env_default(), env_small_buckets()].iter().enumerate() {
            let mut g = Gen::new(e, 251 + k as u64);
            let mut r = Rng::new(351 + k as u64);
            let n = if k == 0 { 300_000 } else { 100_000 };
            let mut lines = Vec::with_capacity(n);
            while lines.len() < n {
                let f = g.cl();
                let l = join(&f, if r.chance(0.1) { r.below(16) as u32 } else { 0 });
                lines.push(mutate_value(&mut r, &l));
            }
            check_lines(e, Kind::Cl, &lines, 1000, &[1, 4096, 1 << 40], &format!("cl value mutations (env {})", k));
        }
    }

    /// Lines whose every field is drawn from a pool of adversarial values
    /// (valid values most of the time), so anomalies combine.
    fn combo_lines(e: &Env, kind: Kind, seed: u64, n: usize) -> Vec<Vec<u8>> {
        let mut g = Gen::new(e, seed);
        let d = e.d;
        let (ws, we) = (e.ctx.ws, e.ctx.we);
        let lo = if kind == Kind::Ev { ws } else { ws - WEEK };
        let tsv = ts_values(lo, we);
        let mut eids = id_variants(&mut g.r, if kind == Kind::Ev { b'e' } else { b'k' }, &eid(&d, 5));
        for nn in [0, 1, d.bound - 1, d.bound, d.bound + 1, MASK] {
            eids.push(eid(&d, nn));
        }
        let mut uids = id_variants(&mut g.r, b'u', &uid(&d, 512 * 3 + 2));
        for u in [0, 7, 8, (d.pmax - 1) * 512 + 7, d.pmax * 512, (1 << 39) - 1, 1 << 39] {
            uids.push(uid(&d, u));
        }
        let prods = ["", "p1", "p3000", "p3001", "P1", "p0", "p01", "p", "p 1", "p99999999", "c_1"];
        let camps = ["c_1", "c_100", "c_101", "c_0", "c_01", "C_1", "c_", "", " c_1", "organic", "x9"];
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            let mut f = if kind == Kind::Ev { g.ev() } else { g.cl() };
            let nf = f.len();
            let tsi = if kind == Kind::Ev { 2 } else { 3 };
            if g.r.chance(0.25) {
                f[0] = g.r.pick(&eids).clone();
            }
            if g.r.chance(0.25) {
                f[1] = g.r.pick(&uids).clone();
            }
            if g.r.chance(0.3) {
                f[tsi] = g.r.pick(&tsv).clone();
                if kind == Kind::Ev && g.r.chance(0.7) {
                    if let Some(t) = crate::ts::parse_ts(f[2].as_bytes()) {
                        if (ws - 30 * DAY..we + 30 * DAY).contains(&t) {
                            f[3] = f1(t + g.r.range(-5000, LATE + 5000));
                        }
                    }
                }
            }
            if kind == Kind::Ev {
                if g.r.chance(0.15) {
                    f[3] = g.r.pick(&tsv).clone();
                }
                if g.r.chance(0.2) {
                    f[4] = s(*g.r.pick(TYPES));
                }
                if g.r.chance(0.3) {
                    f[6] = s(*g.r.pick(PRICES));
                }
                if g.r.chance(0.15) {
                    f[5] = s(*g.r.pick(&prods));
                }
            } else if g.r.chance(0.2) {
                f[2] = s(*g.r.pick(&camps));
            }
            let q = if g.r.chance(0.2) { g.r.below(1 << nf) as u32 } else { 0 };
            out.push(join(&f, q));
        }
        out
    }

    #[test]
    fn k3test_ev_combos() {
        if !have_cpu() {
            return;
        }
        for (k, e) in [env_default(), env_small_buckets(), env_big()].iter().enumerate() {
            let lines = combo_lines(e, Kind::Ev, 401 + k as u64, if k == 0 { 300_000 } else { 100_000 });
            let chunks: &[usize] = if k == 2 { &[4096, 1 << 40] } else { &[1, 4096, 1 << 40] };
            check_lines(e, Kind::Ev, &lines, 1000, chunks, &format!("ev combos (env {})", k));
        }
    }

    #[test]
    fn k3test_cl_combos() {
        if !have_cpu() {
            return;
        }
        for (k, e) in [env_default(), env_small_buckets(), env_big()].iter().enumerate() {
            let lines = combo_lines(e, Kind::Cl, 501 + k as u64, if k == 0 { 200_000 } else { 100_000 });
            let chunks: &[usize] = if k == 2 { &[4096, 1 << 40] } else { &[1, 4096, 1 << 40] };
            check_lines(e, Kind::Cl, &lines, 1000, chunks, &format!("cl combos (env {})", k));
        }
    }

    /// Value mutations with the largest dense sizes (more mutated IDs stay
    /// dense).
    #[test]
    fn k3test_big_dense_value_mutations() {
        if !have_cpu() {
            return;
        }
        let e = env_big();
        let mut g = Gen::new(&e, 601);
        let mut r = Rng::new(602);
        let mut ev = Vec::new();
        let mut cl = Vec::new();
        for _ in 0..150_000 {
            let l = join(&g.ev(), if r.chance(0.1) { r.below(128) as u32 } else { 0 });
            ev.push(mutate_value(&mut r, &l));
            let l = join(&g.cl(), if r.chance(0.1) { r.below(16) as u32 } else { 0 });
            cl.push(mutate_value(&mut r, &l));
        }
        check_lines(&e, Kind::Ev, &ev, 1000, &[4096, 1 << 40], "ev value mutations (big dense)");
        check_lines(&e, Kind::Cl, &cl, 1000, &[4096, 1 << 40], "cl value mutations (big dense)");
    }

    /// Field and line lengths across k3's split limits: field 3 starting
    /// at offsets around 32 and 64, fields 3..6 around 64 bytes, lines
    /// around 96 bytes (product, price and ID lengths, ts forms, quotes).
    #[test]
    fn k3test_ev_length_sweep() {
        if !have_cpu() {
            return;
        }
        let e = env_default();
        let d = e.d;
        let mut g = Gen::new(&e, 91);
        let mut lines = Vec::new();
        for b in 0..8 {
            let f = g.ev();
            let t = e.ctx.ws + 1000 * g.r.range(0, 6 * 86_400) + 7;
            // short and long canonical IDs (n with few digits after scrambling is rare: use raw values too)
            let ids = [f[0].clone(), eid(&d, b), format!("e{}", b36(g.r.below(36))), format!("e{}", b36(36 + g.r.below(1000)))];
            for ts in [f1(t), f2(t, 330), f2(t, -600), f3(t), format!("{:020}", t)] {
                for id in &ids {
                    for plen in 1..=90usize {
                        let prod = if plen == 1 { s("p") } else if plen <= 5 { format!("p{}", 10usize.pow(plen as u32 - 2)) } else { format!("p1{}", "0".repeat(plen - 2)) };
                        for (ty, pr) in [("view", s("")), ("purchase", s("250")), ("purchase", format!("{}250", "0".repeat(plen % 40))), ("cart", "9".repeat(plen % 30))] {
                            let h = vec![id.clone(), f[1].clone(), ts.clone(), f1(t + 4000), s(ty), prod.clone(), pr];
                            lines.push(join(&h, 0));
                            if plen % 7 == 0 {
                                lines.push(join(&h, g.r.below(128) as u32));
                            }
                        }
                    }
                }
            }
        }
        check_lines(&e, Kind::Ev, &lines, 500, &[1, 4096, 1 << 40], "ev length sweep");
    }

    #[test]
    fn k3test_cl_length_sweep() {
        if !have_cpu() {
            return;
        }
        let e = env_default();
        let mut g = Gen::new(&e, 92);
        let mut lines = Vec::new();
        for _ in 0..8 {
            let f = g.cl();
            let t = e.ctx.ws - WEEK + 1000 * g.r.range(0, 13 * 86_400) + 7;
            for ts in [f1(t), f2(t, 330), f2(t, -600), f3(t), format!("{:020}", t)] {
                for clen in 0..=24usize {
                    let ck: String = (0..clen).map(|_| *g.r.pick(b"0123456789abcdefghijklmnopqrstuvwxyz") as char).collect();
                    for camp in ["c_1", "c_77", "c_100", "c_101", "c_0000000000000000000000001", "c_99999999999999999999999999999999"] {
                        let h = vec![ck.clone(), f[1].clone(), s(camp), ts.clone()];
                        lines.push(join(&h, 0));
                        if clen % 3 == 0 {
                            lines.push(join(&h, g.r.below(16) as u32));
                        }
                    }
                }
            }
        }
        check_lines(&e, Kind::Cl, &lines, 500, &[1, 4096, 1 << 40], "cl length sweep");
    }

    /// The same text at every alignment: 0..64 bytes of leading padding
    /// lines, 0..260 bytes after the last line (with and without a final
    /// newline), so lines start and end at every offset from the buffer's
    /// start and end (the 16-byte front, the 192-byte margin).
    fn alignment(kind: Kind, e: &Env, seed: u64) {
        let mut g = Gen::new(e, seed);
        let mut lines: Vec<Vec<u8>> = Vec::new();
        for i in 0..6 {
            let mut f = if kind == Kind::Ev { g.ev() } else { g.cl() };
            if kind == Kind::Ev && i == 2 {
                f[4] = s("purchase");
                f[6] = s("4242");
            }
            lines.push(join(&f, if i == 4 { 0b101 } else { 0 }));
        }
        let mut o = Outs::new(&e.d);
        let body: Vec<u8> = lines.iter().flat_map(|l| l.iter().copied().chain([b'\n'])).collect();
        let name = if kind == Kind::Ev { "ev" } else { "cl" };
        for lead in 0..=64usize {
            let mut t = Vec::new();
            if lead > 0 {
                t.extend(std::iter::repeat(b'x').take(lead - 1));
                t.push(b'\n');
            }
            t.extend_from_slice(&body);
            t.extend_from_slice(&[b'y'; 255]);
            t.push(b'\n');
            check_text(e, &mut o, kind, &t, CHUNKS_FEW, &format!("{} alignment lead {}", name, lead));
        }
        for trail in 0..=260usize {
            for nl in [true, false] {
                let mut t = Vec::new();
                t.extend_from_slice(&[b'x'; 31]);
                t.push(b'\n');
                t.extend_from_slice(&body);
                if trail == 0 && !nl {
                    t.pop(); // the last line ends at the buffer end without a newline
                }
                if trail > 0 {
                    t.extend(std::iter::repeat(b'y').take(trail - 1));
                    if nl {
                        t.push(b'\n');
                    } else {
                        t.push(b'y');
                    }
                }
                check_text(e, &mut o, kind, &t, CHUNKS_FEW, &format!("{} alignment trail {} newline {}", name, trail, nl));
            }
        }
        // each line alone as the final line, at every distance from the end
        for l in &lines {
            for trail in 0..200usize {
                let mut t = Vec::new();
                t.extend_from_slice(&[b'x'; 31]);
                t.push(b'\n');
                t.extend_from_slice(l);
                t.push(b'\n');
                t.extend(std::iter::repeat(b'z').take(trail));
                check_text(e, &mut o, kind, &t, &[1, 1 << 40], &format!("{} single line, {} bytes after it", name, trail));
            }
        }
        // tiny buffers: a line alone (with and without its newline), after
        // 0..20 bytes of padding, and buffers around MARGIN + FRONT bytes
        for l in &lines {
            for lead in 0..=20usize {
                for nl in [true, false] {
                    let mut t = Vec::new();
                    if lead > 0 {
                        t.extend(std::iter::repeat(b'x').take(lead - 1));
                        t.push(b'\n');
                    }
                    t.extend_from_slice(l);
                    if nl {
                        t.push(b'\n');
                    }
                    check_text(e, &mut o, kind, &t, &[1, 1 << 40], &format!("{} tiny buffer, lead {}", name, lead));
                    for total in 200..=216usize {
                        let mut u = t.clone();
                        if !nl {
                            u.push(b'\n');
                        }
                        if u.len() < total {
                            u.extend(std::iter::repeat(b'w').take(total - u.len()));
                            check_text(e, &mut o, kind, &u, &[1, 1 << 40], &format!("{} buffer of {} bytes, lead {}", name, total, lead));
                        }
                    }
                }
            }
        }
        eprintln!("k3test {} alignment: OK", name);
    }

    #[test]
    fn k3test_ev_alignment() {
        if !have_cpu() {
            return;
        }
        alignment(Kind::Ev, &env_default(), 51);
    }

    // ------------------------------------------------------------ tests: clicks

    fn cl_cases(e: &Env, seed: u64, nbase: usize) -> Vec<Vec<u8>> {
        let mut g = Gen::new(e, seed);
        let d = e.d;
        let (ws, we) = (e.ctx.ws, e.ctx.we);
        let tsv = ts_values(ws - WEEK, we);
        let camps = ["c_1", "c_100", "c_101", "c_2000", "c_0", "c_01", "C_1", "c_", "c", "", "c_1 ", " c_1", "c_99999999", "c_1x", "organic", "x9", "\"c_5\"", "c_5\r"];
        let mut out = Vec::new();
        for i in 0..nbase {
            let f = g.cl();
            out.push(join(&f, 0));
            out.push(join(&f, g.r.below(16) as u32));
            if i < 4 {
                for m in 0..16 {
                    out.push(join(&f, m));
                }
            }
            let valid = with(&f, 3, &f1(ws + 1000 * g.r.range(0, 86_400)));
            for v in id_variants(&mut g.r, b'k', &valid[0]) {
                out.push(join(&with(&valid, 0, &v), 0));
            }
            for n in 1..=12 {
                let ck: String = (0..n).map(|_| *g.r.pick(b"0123456789abcdefghijklmnopqrstuvwxyz") as char).collect();
                out.push(join(&with(&valid, 0, &ck), 0));
            }
            for x in id_byte_variants(&valid[0]) {
                let mut raw: Vec<Vec<u8>> = valid.iter().map(|y| y.as_bytes().to_vec()).collect();
                raw[0] = x;
                out.push(join_raw(&raw));
            }
            for v in id_variants(&mut g.r, b'u', &valid[1]) {
                out.push(join(&with(&valid, 1, &v), 0));
            }
            for x in id_byte_variants(&valid[1]) {
                let mut raw: Vec<Vec<u8>> = valid.iter().map(|y| y.as_bytes().to_vec()).collect();
                raw[1] = x;
                out.push(join_raw(&raw));
            }
            if i < 6 {
                for u in [0, 7, 8, 511, (d.pmax - 1) * 512 + 7, d.pmax * 512, (1 << 39) - 1, 1 << 39, MASK] {
                    out.push(join(&with(&valid, 1, &uid(&d, u)), 0));
                }
            }
            for c in camps {
                out.push(join(&with(&valid, 2, c), 0));
            }
            let step = if i < 4 { 1 } else { 5 };
            for v in tsv.iter().step_by(step) {
                out.push(join(&with(&valid, 3, v), 0));
                if g.r.chance(0.1) {
                    out.push(join(&with(&valid, 3, v), 1 << 3));
                }
            }
            for j in 0..4 {
                out.push(join(&with(&valid, j, ""), 0));
            }
            quote_variants(&valid, &mut out, &mut g.r);
        }
        out.push(b"click_id,user_id,campaign_id,ts".to_vec());
        out.push(Vec::new());
        out.push(b",,,".to_vec());
        out
    }

    #[test]
    fn k3test_cl_cases() {
        if !have_cpu() {
            return;
        }
        let envs = [
            env_default(),
            env_small_buckets(),
            env_at("2028-02-28T00:00:00Z", 7, 1 << 20, 1 << 14, 16, false),
            env_at("2026-10-01T05:30:17Z", 3, 1 << 20, 1 << 14, 16, true),
        ];
        for (k, e) in envs.iter().enumerate() {
            let lines = cl_cases(e, 61 + k as u64, if k == 0 { 40 } else { 10 });
            check_lines(e, Kind::Cl, &lines, 400, if k == 0 { CHUNKS } else { CHUNKS_FEW }, &format!("cl cases (env {})", k));
        }
    }

    #[test]
    fn k3test_cl_mutations() {
        if !have_cpu() {
            return;
        }
        for (k, e) in [env_default(), env_small_buckets()].iter().enumerate() {
            let mut g = Gen::new(e, 71 + k as u64);
            let mut r = Rng::new(171 + k as u64);
            let n = if k == 0 { 400_000 } else { 100_000 };
            let mut lines = Vec::with_capacity(n);
            while lines.len() < n {
                let f = g.cl();
                let l = join(&f, if r.chance(0.2) { r.below(16) as u32 } else { 0 });
                lines.push(mutate(&mut r, &l));
            }
            check_lines(e, Kind::Cl, &lines, 1000, &[1, 193, 4096, 1 << 40], &format!("cl mutations (env {})", k));
        }
    }

    #[test]
    fn k3test_cl_alignment() {
        if !have_cpu() {
            return;
        }
        alignment(Kind::Cl, &env_default(), 81);
    }
}
