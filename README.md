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
| `src/fastev.rs` | Line-local splitter for event lines (general scanner as fallback) |
| `src/ts.rs` | Timestamp forms F1/F2/F3, prices; fast SWAR path + general path |
| `src/keys.rs` | IDs as exact 64-bit keys (order-preserving packing + interned fallback) |
| `src/links.rs` | Lock-free concurrent union-find for linked IDs |
| `src/lookup.rs` | Product / campaign lookup (direct array when IDs are `prefix + number`) |
| `src/parts.rs` | Per-thread partitioned buffers in huge-page arenas |
| `src/parquet.rs` | Minimal Parquet writer (PLAIN, uncompressed, Thrift compact metadata) |
| `reference/reference.py` | Slow, literal Python reference used for cross-checking |
| `tools/` | Fixture runner, output comparer, generator-based cross-check driver |
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
3. **IDs.** A non-empty ID of at most 10 bytes over `-0-9A-Z_a-z` is packed into
   a u64 (6-bit codes in ASCII order, left-aligned, length in the low nibble), so
   numeric order equals byte order. Any other string is interned in a global
   table (rare; exact equality, byte-wise order by comparing the strings).
4. **Links.** Parsed in parallel; a concurrent open-addressing table plus
   lock-free union-find that always hooks the root with the larger key under the
   smaller one, so each component's root is its byte-wise smallest ID (the
   person's name). After a finalize pass each slot maps user key -> person key.
5. **Events.** Each line: event_id key, `ingest_ts`, and a validity bit
   (user, ts in window, type, product, purchase price, not late) with the payload
   (user, ts offset, price, category rank). Records are scattered by a hash of
   event_id into 256 partitions per thread. Event lines are split one at a time
   (`src/fastev.rs`): comma/newline/quote masks of a 64-byte window at the line
   start (fields 0-2) and one at field 3 (fields 3-6); the quote rule is checked
   with a carry trick, the next line start comes from the newline masks. Any line
   of another shape goes through the general scanner on its own
   (`SS_FASTEV=0` uses the scanner for everything).
6. **Clicks.** Parsed, validated (ts in the click window, campaign known),
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

- All 36 fixtures: native build, and the linux/amd64 image run with the scorer's
  container flags (read-only root, uid 1000, no network), on both the AVX2 path
  and the forced SSE2 path (`SS_NOAVX2=1`).
- 1M-event dev set: exact match with the published `dev-answer` (native arm64,
  and the amd64 image under emulation on both SIMD paths).
- The Python reference (`reference/reference.py`, written independently from
  `rules.md`) passes all fixtures and matches `dev-answer` exactly.
- Cross-checks against the reference on generated data (distinct seeds, 1K to
  3M events, clicks from 2.5% to 300% of events), each dataset run under 3-4
  settings (1 to 8 threads, 4 to 4096 partitions, tiny chunks, capacities forced
  into the overflow path), all identical to the reference: 64 datasets on the
  final build, 195 more on earlier builds of the same design.
- 20M events: identical tables across 3 different thread/partition/capacity
  settings, and the rules' self-check identities hold.
- `cargo test`: fuzz tests of the scanner against a naive splitter (including
  4 KiB window crossings), of the fast timestamp parser against the general one
  (3M inputs), and of ID packing (round trip and byte order).

## Performance notes

Measured on an Apple M1 (4 performance + 4 efficiency cores, 16 GB) that was
heavily shared with other workloads (load average 80-400, swap nearly full,
thermally throttled), so wall-clock numbers are pessimistic and noisy; the
scorer (8 vCPUs of an Intel Xeon, linux/amd64) could not be measured.

- 20M events, 8 threads, native arm64, best of 3: 3.6 s wall (input not fully
  page-cached on this machine); thread CPU: parse 4.6 s, dedup 1.6 s,
  sessions + output 1.4-2.1 s. Peak RSS 2.3-3.2 GB including mapped input.
- 4M events, 1 thread: about 2.7 G cycles in total (about 510 cycles per input
  line), split parse 56% / dedup 23% / sessions 20%.
- 1M-event dev set in the amd64 image: peak RSS 212 MB.

Expected peak memory at 60M events: roughly 10-12 GB including page-cached
input (extrapolated linearly from 20M, not measured), within the 24 GB limit.

## Limits (all far above anything the generator produces)

- Window length up to 24 days (event timestamps are stored as 31-bit ms offsets).
- At most 255 product categories and 65,534 campaigns.
- `ingest_ts` is ordered at whole-second precision (it is always F1), within
  about +-68 years of the window (values beyond are clamped).
