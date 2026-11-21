#!/usr/bin/env python3
"""Adversarial event / click text for the k3 differential checker (SS_K3CHECK).

Builds a dataset directory from a real one (manifest, products, campaigns and
links copied): its events/ and clicks/ files hold real lines (so the dense keys
are recovered) mixed with
  * random single-byte mutations of real lines (hostile and random bytes),
  * every quoting pattern of real lines, broken quotes, wrong field counts,
    empty fields, blank lines, \\r, long and short lines,
  * IDs of length 0..12 (upper case, spaces, leading zeros, non-base-36 bytes),
  * ts in F1 / F2 / F3 at the window edges (and the click window's), leap days,
    every invalid form of rules.md, F3 with leading zeros and other lengths,
  * prices (leading zeros, 0, 10000000, 10000001, signs, decimals, empty),
    event types (case variants, padding), products,
and some files end without a newline. Then run the checker on it, e.g.
  SS_DENSE=force SS_K3CHECK=ev,cl SS_CHUNK=4096 DATA_DIR=out sessions

Usage: k3adv.py SRC_DIR OUT_DIR [--lines N] [--mut N] [--seed S]
"""
import argparse
import json
import os
import random
import shutil
from datetime import datetime, timedelta, timezone


def iso(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def f2(ms, off_min):
    t = datetime.fromtimestamp(ms / 1000, tz=timezone.utc) + timedelta(minutes=off_min)
    s = "+" if off_min >= 0 else "-"
    a = abs(off_min)
    return t.strftime("%Y-%m-%dT%H:%M:%S") + "%s%02d:%02d" % (s, a // 60, a % 60)


def parse_ms(s):
    return int(datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp() * 1000)


def split_line(line):
    """Fields of a well-formed line (quotes removed), or None."""
    out = []
    for f in line.split(","):
        if len(f) >= 2 and f[0] == '"' and f[-1] == '"':
            f = f[1:-1]
        if '"' in f:
            return None
        out.append(f)
    return out


def join(fields, quotes):
    return ",".join(('"%s"' % f) if q else f for f, q in zip(fields, quotes))


HOSTILE = [",", '"', "\n", "\r", " ", "\x00", "\xff", "Z", "z", "T", "t", ":", "-", "+", ".", "0", "9", "a", "A", "e", "u", "k", "/"]


def ts_variants(ws, we, click):
    lo = ws - 7 * 86400000 if click else ws
    edges = [lo - 1000, lo - 1, lo, lo + 1, lo + 999, lo + 1000, we - 1000, we - 1, we, we + 1, we + 1000,
             (lo + we) // 2, (lo + we) // 2 + 123]
    out = []
    for t in edges:
        sec = t - t % 1000
        out += [str(t), "%016d" % t, "000" + str(t), str(t)[:-1], str(t) + "0", iso(sec), f2(sec, 0), f2(sec, 330),
                f2(sec, -600), f2(sec, 720), f2(sec, -570), f2(sec, 45)]
    base = iso((lo + we) // 2 - ((lo + we) // 2) % 1000)
    out += ["", base.lower(), base.replace("T", " "), base[:-1], base[:-1] + "z", base.replace("T", "t"),
            base[:-1] + ".000Z", base[:-1] + "+0200", base[:-1] + "+24:00", base[:-1] + "+05:60", base[:-1] + "-00:00",
            base[:-1] + "+00:00", base[:-1] + "+5:30", base[:-1] + "+05:3", base[:-1] + "+05:30x",
            base[:11] + "24:00:00Z", base[:11] + "23:60:00Z", base[:11] + "23:59:60Z", base[:11] + "23:59:59Z",
            base[:5] + "13" + base[7:], base[:5] + "02-30" + base[10:], base[:5] + "02-29" + base[10:],
            "2028-02-29T12:00:00Z", "2027-02-29T12:00:00Z", "2028-02-29T12:00:00+01:00", "1.79e12", "-5",
            "1791071890000.0", "86400000", "0", "9223372036854775807", "9223372036854775808", "99999999999999999999",
            base + "\r", " " + base, base + " ", "１７９１０７１８９００００", base.replace("-", "/"), base[:19] + "Z"]
    # every single-byte change of an F1 and an F2 value
    f1 = iso((lo + we) // 2 - ((lo + we) // 2) % 1000)
    f2v = f2((lo + we) // 2 - ((lo + we) // 2) % 1000, 330)
    for v in (f1, f2v, str((lo + we) // 2)):
        for i in range(len(v)):
            for c in "0959:-+TZzt/ ":
                out.append(v[:i] + c + v[i + 1:])
    return out


def id_variants(rng, prefix, real):
    out = ["", prefix, prefix + "0", prefix + "00", prefix + "0" + real[1:], real.upper(), real[0].upper() + real[1:],
           " " + real, real + " ", real[:-1], real + "0", real + "z", prefix + "z" * 8, prefix + "zzzzzzzzz",
           prefix + "1" + "0" * 7, prefix + "1" + "0" * 8, prefix + "ffffffff", prefix + "fzzzzzzz", prefix + "g0000000",
           prefix + "-_AZaz09", prefix + "ab_cd", prefix + "ab-cd", real.replace(real[2], "\xe9", 1)]
    for n in range(0, 13):
        out.append(prefix + "".join(rng.choice("0123456789abcdefghijklmnopqrstuvwxyz") for _ in range(n)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--lines", type=int, default=60000, help="real lines per kind")
    ap.add_argument("--mut", type=int, default=120000, help="single-byte mutations per kind")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    os.makedirs(a.out)
    for f in ("manifest.json", "products.csv", "campaigns.csv"):
        shutil.copy(os.path.join(a.src, f), os.path.join(a.out, f))
    shutil.copytree(os.path.join(a.src, "links"), os.path.join(a.out, "links"))
    man = json.load(open(os.path.join(a.src, "manifest.json")))
    ws, we = parse_ms(man["window_start"]), parse_ms(man["window_end"])

    for kind, nf in (("events", 7), ("clicks", 4)):
        click = kind == "clicks"
        real = []
        srcdir = os.path.join(a.src, kind)
        for fn in sorted(os.listdir(srcdir)):
            if not fn.endswith(".csv"):
                continue
            with open(os.path.join(srcdir, fn), encoding="latin-1", newline="") as fh:
                header = fh.readline()
                for line in fh:
                    real.append(line.rstrip("\n"))
                    if len(real) >= a.lines:
                        break
            if len(real) >= a.lines:
                break
        good = [l for l in real if split_line(l) and len(split_line(l)) == nf]
        adv = []
        # mutations
        for _ in range(a.mut):
            l = rng.choice(good)
            i = rng.randrange(len(l) + 1)
            r = rng.random()
            c = rng.choice(HOSTILE) if r < 0.7 else chr(rng.randrange(256))
            if r < 0.85:
                l = l[:i] + c + l[i + 1:]
            elif r < 0.95:
                l = l[:i] + c + l[i:]
            else:
                l = l[:i] + l[i + 1:]
            adv.extend(l.split("\n"))
        # quoting patterns, broken quotes, field counts
        for l in rng.sample(good, min(len(good), 3000)):
            f = split_line(l)
            qp = rng.randrange(1 << nf)
            adv.append(join(f, [(qp >> j) & 1 for j in range(nf)]))
            j = rng.randrange(nf)
            adv.append(join(f[:j] + ['"' + f[j]] + f[j + 1:], [0] * nf))
            adv.append(join(f[:j] + [f[j] + '"'] + f[j + 1:], [0] * nf))
            adv.append(join(f[:j] + ['"' + f[j] + '""'] + f[j + 1:], [0] * nf))
            adv.append(join(f[:j] + [f[j][:1] + '"' + f[j][1:]] + f[j + 1:], [0] * nf))
            adv.append(",".join(f[:-1]))
            adv.append(",".join(f + ["x"]))
            adv.append(",".join(f[:j] + [""] + f[j + 1:]))
            adv.append(",".join(f[:j] + ['""'] + f[j + 1:]))
            adv.append(",".join(f) + "\r")
            adv.append(",".join(f[:j] + [f[j] + "\r"] + f[j + 1:]))
            adv.append(",".join(f[:j] + [f[j] * rng.randrange(2, 12)] + f[j + 1:]))
            adv.append("")
            adv.append(",".join(f)[: rng.randrange(1, max(2, len(l)))])
            adv.append(",".join(f) + "," * rng.randrange(1, 4))
        all_q = [l for l in good[:200]]
        for qp in range(1 << nf):
            for l in all_q[:8]:
                adv.append(join(split_line(l), [(qp >> j) & 1 for j in range(nf)]))
        # field value variants
        tsv = ts_variants(ws, we, click)
        for l in rng.sample(good, min(len(good), 400)):
            f = split_line(l)
            tsi = 3 if click else 2
            for v in id_variants(rng, f[0][:1] or ("k" if click else "e"), f[0]):
                adv.append(",".join([v] + f[1:]))
            for v in id_variants(rng, "u", f[1] or "uabcdefgh"):
                adv.append(",".join(f[:1] + [v] + f[2:]))
            for v in rng.sample(tsv, 60):
                g = list(f)
                g[tsi] = v
                adv.append(join(g, [rng.random() < 0.2 for _ in range(nf)]))
            if click:
                for v in ["c_1", "c_2000", "c_2001", "c_0", "c_01", "C_1", "c_", "c", "", "c_1 ", " c_1", "c_99999999", "c_1x"]:
                    adv.append(",".join(f[:2] + [v] + f[3:]))
            else:
                for v in tsv[:40] + [iso(ws), iso(we + 6 * 86400000), iso(ws - 3 * 86400000), "1791071890000", f2(ws, 60)]:
                    adv.append(",".join(f[:3] + [v] + f[4:]))
                for v in ["view", "cart", "purchase", "Purchase", "PURCHASE", "View", "cart ", " cart", "click", "",
                          "views", "purchas", "purchasee", "vie", "car"]:
                    for pr in ["", "250", "0250", "1", "0", "00", "10000000", "10000001", "250000000", "19.99", "-5",
                               "+500", "1e3", "1000\r", "00000000000010000000", " 250", "250 "]:
                        if rng.random() < 0.15:
                            adv.append(",".join(f[:4] + [v, f[5], pr]))
                for v in ["", "p1", "p50000", "p50001", "P1", "p0", "p01", "p", "p 1", "p99999999"]:
                    adv.append(",".join(f[:4] + ["purchase", v, "250"]))
        # long lines
        for _ in range(300):
            adv.append(",".join(rng.choice(good).split(",")[:-1] + ["x" * rng.randrange(50, 300)]))
        lines = real + adv
        rng.shuffle(lines)
        # keep the first lines real (key recovery samples the first lines of the first tasks)
        lines = real[:3000] + lines
        os.makedirs(os.path.join(a.out, kind))
        nfiles = 7
        per = len(lines) // nfiles + 1
        for k in range(nfiles):
            chunk = lines[k * per:(k + 1) * per]
            text = header + "\n".join(chunk) + ("\n" if k % 3 else "")
            with open(os.path.join(a.out, kind, "part-%05d.csv" % k), "w", encoding="utf-8", newline="") as fh:
                fh.write(text)
        print(kind, "real", len(real), "adversarial", len(adv))


if __name__ == "__main__":
    main()
