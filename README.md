# Shopping Sessions: native Rust pipeline

An entry for the Builders Buildathon "Shopping Sessions" track. One static Rust
binary (no dependencies, no runtime) in an empty `scratch` image (~2.5 MB). It
reads the click-stream CSVs from `/data`, applies every rule in `rules.md`, and
writes `/output/user_daily/*.parquet` and `/output/campaign_daily/*.parquet`.

## Layout

| Path | What |
| --- | --- |
| `src/main.rs` | Pipeline phases, records, sessionization, attribution, output |
| `src/scan.rs` | CSV dialect scanner (SIMD masks, branch-light line splitting) |
| `src/fastev.rs` | Line-local splitters for event and click lines (general scanner as fallback) |
| `src/ts.rs` | Timestamp forms F1/F2/F3, prices; fast SWAR path + general path |
| `src/keys.rs` | IDs as exact 64-bit keys (order-preserving packing, verified hash keys for 33-byte hex IDs, interned fallback) |
| `src/links.rs` | Lock-free concurrent union-find for linked IDs |
| `src/lookup.rs` | Product / campaign lookup (direct array when IDs are `prefix + number`) |
| `src/parts.rs` | Per-thread partitioned buffers in huge-page arenas |
| `src/parquet.rs` | Minimal Parquet writer (PLAIN, uncompressed, Thrift compact metadata) |
| `reference/reference.py` | Slow, literal Python reference used for cross-checking |
| `tools/` | Fixture runner, output comparer, cross-check drivers, gate and harness scripts (`tools/a3/`) |
| `fixtures/` | The 36 organizer edge-case fixtures |

## How it works

All work runs on `NCPU` threads with dynamic work distribution (atomic task
counters); phases are separated by joins.

1. **Setup.** Read `manifest.json`, `products.csv`, `campaigns.csv`. Product and
   campaign IDs of the form `prefix + canonical decimal` use a direct array;
   anything else falls back to an exact hash map. Every `*.csv` in `events/`,
   `clicks/`, `links/` is memory-mapped and cut into ~2 MiB line-aligned chunks.
2. **CSV scanning.** Per 4 KiB window: SIMD masks of `,` `\n` `"` (NEON, SSE2,
   or AVX2 selected at run time), delimiter and newline positions flattened by
   fixed-width unrolled loops, then one iteration per line with branch-free
   field slicing. A field is quoted iff it starts with `"`; the line is valid iff
   it has exactly N fields, every quoted field ends with `"` and is at least 2
   bytes, and the line holds exactly 2 quotes per quoted field.
3. **IDs.** IDs are opaque byte strings: equality and byte-wise order are
   exactly those of the strings, whatever their shape. Three exact encodings
   into a u64 key, told apart by the low 4 bits (1-10, 0xE, 0xF):
   * a non-empty ID of at most 10 bytes over `-0-9A-Z_a-z` is packed (6-bit
     codes in ASCII order, left-aligned, length in the low nibble), so numeric
     order equals byte order;
   * an ID of exactly one such byte followed by exactly 32 lowercase hex digits
     (one ID in eight in the current data) gets a 60-bit seeded hash of
     (first byte, 128-bit value), computed without any shared state. Each
     thread logs every (key, full value) it creates; after the link lines are
     parsed and again after all parsing, a parallel pass checks that no key
     stands for two different IDs and builds the table that decodes these keys
     (names, byte-wise order: long vs long by (first byte, value), anything
     else by the strings). If two IDs ever share a key, the whole job runs
     again with another seed, after three seeds with these IDs interned, so the
     result is exact for every input (`SS_LONGBITS=n` forces collisions for
     tests, `SS_LONGOFF=1` interns them);
   * every other string is interned in a global table (rare; exact equality,
     byte-wise order by comparing the strings).
4. **Links.** Parsed in parallel; a concurrent open-addressing table plus
   lock-free union-find that always hooks the root with the larger key under the
   smaller one, so each component's root is its byte-wise smallest ID (the
   person's name). After a finalize pass each slot maps user key -> person key.
5. **Events.** Each line: event_id key, `ingest_ts`, and a validity bit
   (user, ts in window, type, product, purchase price, not late) with the payload
   (user, ts offset, price, category rank). Records are scattered by a hash of
   event_id into partitions per thread. Event lines are split one at a time
   (`src/fastev.rs`): comma/newline/quote masks of a 64-byte window at the line
   start (fields 0-2) and one at field 3 (fields 3-6); a line whose first three
   fields do not fit 64 bytes (long IDs) is cut the same way with a 128-byte
   first window. The quote rule is checked with a carry trick. Any line of
   another shape goes through the general scanner on its own (`SS_FASTEV=0` uses
   the scanner for everything, `SS_WIDECUT=0` disables the 128-byte cut).
6. **Clicks.** Split from one 64-byte (else 128-byte) window per line, with the
   general scanner as fallback; parsed, validated (ts in the click window,
   campaign known),
   resolved to a person (link-table lookups prefetched through a small ring),
   scattered by a hash of the person. Click dedup is unnecessary: copies are
   identical, so they cannot change the credited click or the assist set.
7. **Dedup.** Per event_id partition (split in-cache into <= 24K-record
   sub-buckets): hash table keeping the copy with the smallest `ingest_ts`;
   kept copies that are valid are resolved to their person (prefetched) and
   scattered by person.
8. **Sessions.** Per person partition: group by person (hash map + counting
   sort), sort each person's events by ts, split on gaps > 30 min, aggregate per
   (person, day), credit each session with a sliding 7-day window over the
   person's clicks sorted by (ts, click_id), and count assisting campaigns from
   the window's active set.
9. **Output.** Each thread writes its own `user_daily` part file; one file for
   `campaign_daily`. `day` is INT32 annotated DATE, strings are BYTE_ARRAY
   annotated UTF8/STRING, counts are INT32/INT64.

## Build and run

```sh
docker buildx build --platform linux/amd64 --load -t shopping-sessions .
IMAGE=shopping-sessions tools/docker_run.sh dev out   # the scorer's container flags
```

Without Docker: `cargo build --release && target/release/sessions DATA_DIR OUTPUT_DIR`.
`SS_VERBOSE=1` prints per-phase wall time and thread CPU time to stderr.

## Testing

```sh
pip install polars
python3 tools/run_fixtures.py fixtures -- target/release/sessions      # 36 fixtures
python3 tools/run_fixtures.py fixtures -- python3 reference/reference.py
python3 tools/compare.py OUT_DIR path/to/dev-answer                     # exact diff
python3 tools/crosscheck.py GENERATE_BIN target/release/sessions WORKDIR 2:1000000:250000 ...
cargo test --release                                                    # unit + fuzz tests
```

`crosscheck.py` generates datasets with the published generator, runs the Python
reference and the binary under several thread / partition / chunk settings, and
requires identical tables.

## Verification done

Current data (scorer version 5: one ID in eight is a 33-byte hex form), with
`tools/a3/a3_v5gate.sh` and `tools/a3/v5_adv.py`:

- All 36 fixtures, native and in the linux/amd64 image with the scorer's
  container flags, on the AVX2 and the forced SSE path and with forced key
  collisions, long IDs interned, and the 128-byte cut off.
- 1M-event dev set: exact match with the published v5 `dev-answer` under every
  switch above, at 1 and 8 threads.
- 20M (three seeds) and 60M events: identical to the outputs of an earlier
  build that the organisers' scorer accepted on this data, and to the build
  before it.
- `v5_adv.py`: 56 generated and mutated datasets (near-twin hex IDs differing
  in one digit or one half, the same digits under other prefixes, almost-hex
  shapes, quoting, long IDs as person names and in link merges and splits,
  duplicate events, click ties between long and short IDs, lines of every
  length around the window sizes, CRLF, many small files, every ID long), each
  in 12 switch settings, all identical, and identical to the Python reference
  (`reference/reference.py`, written independently from `rules.md`) up to 300K
  events.
- `cargo test`: fuzz tests of the scanner against a naive splitter (including
  window crossings), of the line splitters against the scanner (with long IDs),
  of the fast timestamp parser against the general one, of ID packing (round
  trip and byte order), of long-ID keys (exact shape test, round trip, equality
  and order against the strings, collision detection through the interner
  fallback) and of the SSE hex decoder.

## Performance notes

On the organisers' scorer (8 vCPUs, Intel Xeon, 24 GB): about 1.06 s container
time for the 20M-event practice set at the time of writing. `SS_VERBOSE=1`
prints the phases (links, parse, dedup, sessions, write), thread CPU per phase,
and the long-ID statistics (entries logged, distinct, verification time).

## Limits (all far above anything the generator produces)

- Window length up to 24 days (event timestamps are stored as 31-bit ms offsets).
- At most 255 product categories and 65,534 campaigns.
- `ingest_ts` is ordered at whole-second precision (it is always F1); values
  more than 34 years from the window are ranked exactly in a second pass.
