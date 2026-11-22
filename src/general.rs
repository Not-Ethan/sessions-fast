//! General pipeline: single-threaded, plain collections, written line by line
//! from rules.md (it follows reference/reference.py step by step). It has none
//! of the fast pipelines' representation limits (event offsets below 2^31 ms,
//! category ranks in 8 bits, campaign indices in 16 bits), so `run` uses it
//! for an input outside those limits instead of failing; SS_GENERAL=1 forces
//! it (tests). Slow, and it never runs on generated data.

use std::collections::HashMap;
use std::path::Path;

use crate::{fail, list_csv, parquet, read_file, scan, ts, DAY, WEEK};

const GAP: i64 = crate::GAP as i64;
const LATE: i64 = crate::LATE;

/// Calls `f` with the unquoted fields of every well-formed data line (header
/// skipped) of every `*.csv` file in `dir`, in file-name order.
fn csv_rows<const N: usize>(dir: &Path, f: &mut dyn FnMut(&[&[u8]; N])) {
    for path in list_csv(dir) {
        let text = read_file(&path);
        table_rows(&text, f);
    }
}

fn table_rows<const N: usize>(text: &[u8], f: &mut dyn FnMut(&[&[u8]; N])) {
    if text.is_empty() {
        return;
    }
    let h = scan::next_line_start(text, 0);
    scan::for_each_record::<N, _>(text, h, text.len(), |r| f(r));
}

fn all_digits(b: &[u8]) -> bool {
    !b.is_empty() && b.iter().all(|c| c.is_ascii_digit())
}

/// Union-find over IDs (links).
struct Persons {
    index: HashMap<Vec<u8>, u32>,
    parent: Vec<u32>,
    /// per root: the byte-wise smallest ID of its set (after `finish`)
    name: Vec<u32>,
    ids: Vec<Vec<u8>>,
}

impl Persons {
    fn id(&mut self, k: &[u8]) -> u32 {
        if let Some(&i) = self.index.get(k) {
            return i;
        }
        let i = self.ids.len() as u32;
        self.index.insert(k.to_vec(), i);
        self.ids.push(k.to_vec());
        self.parent.push(i);
        i
    }
    fn find(&mut self, x: u32) -> u32 {
        let mut r = x;
        while self.parent[r as usize] != r {
            r = self.parent[r as usize];
        }
        let mut x = x;
        while self.parent[x as usize] != r {
            let n = self.parent[x as usize];
            self.parent[x as usize] = r;
            x = n;
        }
        r
    }
    fn union(&mut self, a: &[u8], b: &[u8]) {
        let (a, b) = (self.id(a), self.id(b));
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.parent[ra as usize] = rb;
        }
    }
    fn finish(&mut self) {
        self.name = (0..self.ids.len() as u32).collect();
        for x in 0..self.ids.len() as u32 {
            let r = self.find(x) as usize;
            if self.ids[x as usize] < self.ids[self.name[r] as usize] {
                self.name[r] = x;
            }
        }
    }
    /// The person's name of user ID `uid`.
    fn person(&mut self, uid: &[u8]) -> Vec<u8> {
        match self.index.get(uid) {
            Some(&i) => {
                let r = self.find(i) as usize;
                self.ids[self.name[r] as usize].clone()
            }
            None => uid.to_vec(),
        }
    }
}

/// Interns byte strings as dense u32 indices.
#[derive(Default)]
struct Names {
    index: HashMap<Vec<u8>, u32>,
    names: Vec<Vec<u8>>,
}

impl Names {
    fn get(&mut self, k: &[u8]) -> u32 {
        if let Some(&i) = self.index.get(k) {
            return i;
        }
        let i = self.names.len() as u32;
        self.index.insert(k.to_vec(), i);
        self.names.push(k.to_vec());
        i
    }
}

/// No person: an invalid copy (it still takes part in dedup).
const INVALID: u32 = u32::MAX;

/// A kept event copy, validated when parsed (validity depends only on the
/// copy's own fields and on the links, which are read first).
#[derive(Clone, Copy)]
struct Ev {
    ingest: i64,
    ts: i64,
    /// person index, or INVALID
    person: u32,
    /// purchase: category index; else u32::MAX
    cat: u32,
    price: u64,
}

/// Runs the whole job on the general path; returns the verbose report line.
pub fn run(data_dir: &Path, out_dir: &Path, ws: i64, we: i64, why: &str) -> String {
    let t0 = std::time::Instant::now();
    // products (category indices) and campaigns
    let mut cats = Names::default();
    let mut products: HashMap<Vec<u8>, u32> = HashMap::new();
    table_rows::<2>(&read_file(&data_dir.join("products.csv")), &mut |f| {
        let c = cats.get(f[1]);
        products.insert(f[0].to_vec(), c);
    });
    let unknown = cats.get(b"unknown");
    let mut camp_index: HashMap<Vec<u8>, u32> = HashMap::new();
    let mut camp_rows: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    table_rows::<2>(&read_file(&data_dir.join("campaigns.csv")), &mut |f| {
        if !camp_index.contains_key(f[0]) {
            camp_index.insert(f[0].to_vec(), camp_rows.len() as u32);
            camp_rows.push((f[0].to_vec(), f[1].to_vec()));
        }
    });

    // links
    let mut ps = Persons { index: HashMap::new(), parent: Vec::new(), name: Vec::new(), ids: Vec::new() };
    csv_rows::<2>(&data_dir.join("links"), &mut |f| {
        if !f[0].is_empty() && !f[1].is_empty() {
            ps.union(f[0], f[1]);
        }
    });
    ps.finish();
    let mut persons = Names::default();

    // events: dedup on event_id (smallest ingest_ts; on a tie the first copy,
    // equal-ingest copies are identical); each copy validated as parsed
    let mut kept: HashMap<Box<[u8]>, Ev> = HashMap::new();
    csv_rows::<7>(&data_dir.join("events"), &mut |f| {
        if f[0].is_empty() {
            return;
        }
        // ingest_ts is always valid F1; a copy without one cannot be ordered
        let Some(ingest) = ts::parse_ts(f[3]) else { return };
        if let Some(prev) = kept.get(f[0]) {
            if ingest >= prev.ingest {
                return;
            }
        }
        let mut e = Ev { ingest, ts: 0, person: INVALID, cat: u32::MAX, price: 0 };
        let (uid, ts_s, etype, pid, price_s) = (f[1], f[2], f[4], f[5], f[6]);
        let valid = (|| {
            if uid.is_empty() || pid.is_empty() {
                return None;
            }
            let t = ts::parse_ts(ts_s)?;
            if !(ws <= t && t < we) {
                return None;
            }
            let purchase = match etype {
                b"purchase" => true,
                b"view" | b"cart" => false,
                _ => return None,
            };
            let mut price = 0u64;
            if purchase {
                if !all_digits(price_s) {
                    return None;
                }
                // digits only: at most 10,000,000 after leading zeros
                let digits = &price_s[price_s.iter().position(|&c| c != b'0').unwrap_or(price_s.len())..];
                if digits.len() > 8 {
                    return None;
                }
                price = digits.iter().fold(0u64, |v, &c| v * 10 + (c - b'0') as u64);
                if !(1..=10_000_000).contains(&price) {
                    return None;
                }
            }
            if ingest - t > LATE {
                return None;
            }
            Some((t, purchase, price))
        })();
        if let Some((t, purchase, price)) = valid {
            e.ts = t;
            e.person = persons.get(&ps.person(uid));
            if purchase {
                e.price = price;
                e.cat = products.get(pid).copied().unwrap_or(unknown);
            }
        }
        kept.insert(f[0].into(), e);
    });
    // person -> (ts, price, category or u32::MAX for view / cart)
    let mut by_person: Vec<Vec<(i64, u64, u32)>> = Vec::new();
    for (_, e) in kept.drain() {
        if e.person == INVALID {
            continue;
        }
        if by_person.len() <= e.person as usize {
            by_person.resize_with(e.person as usize + 1, Vec::new);
        }
        by_person[e.person as usize].push((e.ts, e.price, e.cat));
    }
    drop(kept);

    // clicks: dedup on click_id (copies are identical: keep the first), each
    // validated as parsed: (person or INVALID, ts, campaign)
    let mut seen: HashMap<Box<[u8]>, (u32, i64, u32)> = HashMap::new();
    csv_rows::<4>(&data_dir.join("clicks"), &mut |f| {
        if f[0].is_empty() || seen.contains_key(f[0]) {
            return;
        }
        let mut c = (INVALID, 0i64, 0u32);
        if !f[1].is_empty() {
            if let (Some(t), Some(&camp)) = (ts::parse_ts(f[3]), camp_index.get(f[2])) {
                if ws - WEEK <= t && t < we {
                    c = (persons.get(&ps.person(f[1])), t, camp);
                }
            }
        }
        seen.insert(f[0].into(), c);
    });
    // person -> (ts, click_id, campaign)
    let mut clicks: HashMap<u32, Vec<(i64, Box<[u8]>, u32)>> = HashMap::new();
    for (cid, (p, t, camp)) in seen.drain() {
        if p != INVALID {
            clicks.entry(p).or_default().push((t, cid, camp));
        }
    }

    // sessions, user_daily, attribution, campaign_daily
    // (person, day) -> (sessions, events, revenue, category sums)
    let mut user_daily: HashMap<(u32, i64), (u64, u64, u64, HashMap<u32, u64>)> = HashMap::new();
    // (campaign, day) -> (sessions, purchasing, revenue, assisted); u32::MAX = organic
    let mut camp_daily: HashMap<(u32, i64), (u64, u64, u64, u64)> = HashMap::new();
    let none = Vec::new();
    for (p, evs) in by_person.iter_mut().enumerate() {
        let p = p as u32;
        evs.sort_by_key(|e| e.0);
        let pclicks = clicks.get(&p).unwrap_or(&none);
        let mut i = 0;
        while i < evs.len() {
            let start = evs[i].0;
            let mut j = i + 1;
            while j < evs.len() && evs[j].0 - evs[j - 1].0 <= GAP {
                j += 1;
            }
            let sev = &evs[i..j];
            let day = start.div_euclid(DAY);
            let ud = user_daily.entry((p, day)).or_default();
            ud.0 += 1;
            ud.1 += sev.len() as u64;
            let mut rev = 0u64;
            let mut purch = false;
            for e in sev.iter().filter(|e| e.2 != u32::MAX) {
                purch = true;
                rev += e.1;
                *ud.3.entry(e.2).or_default() += e.1;
            }
            ud.2 += rev;
            // attribution: latest ts, ties to the smallest click_id
            let q: Vec<&(i64, Box<[u8]>, u32)> = pclicks.iter().filter(|c| c.0 <= start && start - c.0 <= WEEK).collect();
            let credit = q.iter().max_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.cmp(&a.1))).map_or(u32::MAX, |c| c.2);
            let cd = camp_daily.entry((credit, day)).or_default();
            cd.0 += 1;
            cd.1 += purch as u64;
            cd.2 += rev;
            let mut assisted: Vec<u32> = q.iter().map(|c| c.2).collect();
            assisted.sort_unstable();
            assisted.dedup();
            for camp in assisted {
                camp_daily.entry((camp, day)).or_default().3 += 1;
            }
            i = j;
        }
    }

    // outputs
    let plain = |b: &mut Vec<u8>, x: &[u8]| {
        b.extend_from_slice(&(x.len() as u32).to_le_bytes());
        b.extend_from_slice(x);
    };
    let ud = out_dir.join("user_daily");
    let cd = out_dir.join("campaign_daily");
    std::fs::create_dir_all(&ud).unwrap_or_else(|e| fail(&format!("{}: {}", ud.display(), e)));
    std::fs::create_dir_all(&cd).unwrap_or_else(|e| fail(&format!("{}: {}", cd.display(), e)));
    let (mut uid, mut days, mut sess, mut evs, mut revs, mut def, mut top) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let nrows = user_daily.len();
    for (i, ((p, day), (s, n, rev, sums))) in user_daily.into_iter().enumerate() {
        plain(&mut uid, &persons.names[p as usize]);
        days.push(day as i32);
        sess.push(s as i64);
        evs.push(n as i64);
        revs.push(rev as i64);
        if i % 8 == 0 {
            def.push(0u8);
        }
        // highest sum; a tie goes to the byte-wise smallest category
        let name = |c: &u32| &cats.names[*c as usize];
        if let Some((c, _)) = sums.iter().max_by(|a, b| a.1.cmp(b.1).then_with(|| name(b.0).cmp(name(a.0)))) {
            def[i / 8] |= 1 << (i % 8);
            plain(&mut top, &cats.names[*c as usize]);
        }
    }
    let cols = [
        parquet::Raw::Req { name: "user_id", ty: parquet::T_BYTE_ARRAY, conv: Some(parquet::CONV_UTF8), body: &uid },
        parquet::Raw::Req { name: "day", ty: parquet::T_INT32, conv: Some(parquet::CONV_DATE), body: parquet::le_bytes(&days) },
        parquet::Raw::Req { name: "sessions", ty: parquet::T_INT64, conv: None, body: parquet::le_bytes(&sess) },
        parquet::Raw::Req { name: "events", ty: parquet::T_INT64, conv: None, body: parquet::le_bytes(&evs) },
        parquet::Raw::Req { name: "revenue_cents", ty: parquet::T_INT64, conv: None, body: parquet::le_bytes(&revs) },
        parquet::Raw::OptStr { name: "top_category", def: &def, body: &top },
    ];
    let path = ud.join("part-0.parquet");
    parquet::write_raw(&path, nrows, &cols).unwrap_or_else(|e| fail(&format!("{}: {}", path.display(), e)));

    let (mut ids, mut chs, mut days, mut sv, mut pv, mut rv, mut av) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let nrows = camp_daily.len();
    for ((c, day), (s, p, r, a)) in camp_daily {
        if c == u32::MAX {
            plain(&mut ids, b"organic");
            plain(&mut chs, b"organic");
        } else {
            plain(&mut ids, &camp_rows[c as usize].0);
            plain(&mut chs, &camp_rows[c as usize].1);
        }
        days.push(day as i32);
        sv.push(s as i64);
        pv.push(p as i64);
        rv.push(r as i64);
        av.push(a as i64);
    }
    let cols = [
        parquet::Raw::Req { name: "campaign_id", ty: parquet::T_BYTE_ARRAY, conv: Some(parquet::CONV_UTF8), body: &ids },
        parquet::Raw::Req { name: "channel", ty: parquet::T_BYTE_ARRAY, conv: Some(parquet::CONV_UTF8), body: &chs },
        parquet::Raw::Req { name: "day", ty: parquet::T_INT32, conv: Some(parquet::CONV_DATE), body: parquet::le_bytes(&days) },
        parquet::Raw::Req { name: "sessions", ty: parquet::T_INT64, conv: None, body: parquet::le_bytes(&sv) },
        parquet::Raw::Req { name: "purchasing_sessions", ty: parquet::T_INT64, conv: None, body: parquet::le_bytes(&pv) },
        parquet::Raw::Req { name: "revenue_cents", ty: parquet::T_INT64, conv: None, body: parquet::le_bytes(&rv) },
        parquet::Raw::Req { name: "assisted_sessions", ty: parquet::T_INT64, conv: None, body: parquet::le_bytes(&av) },
    ];
    let path = cd.join("part-0.parquet");
    parquet::write_raw(&path, nrows, &cols).unwrap_or_else(|e| fail(&format!("{}: {}", path.display(), e)));
    format!("sessions: pipeline=general ({}) total={:.0}ms", why, t0.elapsed().as_secs_f64() * 1e3)
}
