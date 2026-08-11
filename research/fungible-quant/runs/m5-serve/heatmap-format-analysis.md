# Why base64-in-JSON, and what the alternatives actually cost

Michel asked whether base64-in-JSON is the right wire format for the
activation matrix, or whether something standard — pandas, Parquet — would be
better. Measured on the real 75×256 payload rather than argued.

## The numbers

| format | bytes | gzipped | encode | decode | |
|---|---|---|---|---|---|
| **base64 bf16 in JSON** (current) | 78,342 | **35,647** | ~0 | ~0 | one body, metadata inline |
| plain nested JSON floats | 191,410 | 39,018 | — | — | 2.44× the wire |
| nested JSON, 1 decimal | 183,553 | 37,801 | — | — | still 2.34× |
| numpy `.npy` float32 | 76,928 | 34,543 | — | — | no browser reader |
| `.npz` (deflate inside) | 35,194 | 35,194 | — | — | no browser reader |
| **parquet LONG (zstd)** | **31,418** | 30,948 | 23.2 ms | 118.3 ms | smallest, slowest |
| parquet LONG (snappy) | 36,282 | 31,510 | 0.8 ms | 1.5 ms | good balance |
| parquet LONG (uncompressed) | 54,291 | 31,251 | 0.6 ms | 0.6 ms | |
| arrow IPC stream LONG | 173,384 | 36,502 | 0.8 ms | **0.1 ms** | zero-copy |

"LONG" = the tidy `(layer, expert, count, tier)` shape a dataframe user
expects — 19,200 rows. It is what makes Parquet compress so well and what
makes Arrow IPC *large*, since the layer and expert columns repeat per cell.

## The finding that matters

**Once gzip is on, every format lands between 31 and 39 KB.** The wire-format
choice is worth a few KB; the transport encoding is worth 2.4×. And the
endpoint already says so, in its own payload:

```
"the client did not offer gzip; this body is ~2.4x the size of the
 compressed one"
```

My capture scripts were not sending `Accept-Encoding: gzip`. Fixed — `curl
--compressed` and an explicit header plus decompress in the Python client.
Reading your own payload's warnings turns out to be cheaper than benchmarking
seven encodings.

## So is base64-in-JSON right?

**For the live endpoint: yes, and for reasons the benchmark does not show.**

- One HTTP body carries the matrix *and* its metadata — `step`, `policy_sha`,
  the collector `window`, `ranks.agree`, `warnings`. A Parquet response would
  need the metadata somewhere else, and the pairing is the point: a matrix
  without the step and policy it belongs to is not evidence.
- Browsers decode base64 to `Uint8Array` natively. Parquet and Arrow in a
  browser mean shipping a WASM reader to a page whose whole job is to draw a
  heatmap.
- No server-side dependency. Adding pyarrow to a vLLM worker to serve a
  monitoring endpoint is a poor trade.
- Encode cost is ~0. Parquet-zstd costs 23 ms to write and 118 ms to read —
  on an endpoint polled every few seconds, against a serve whose GPUs we are
  trying to keep busy.

**For the archive: no — Parquet, clearly.** Which is exactly the split the
Arrow project itself recommends: *store in Parquet, transport in Arrow*
([Arrow FAQ](https://arrow.apache.org/faq/)). Parquet's encoding and
compression "typically yields much smaller files, making Parquet a better
choice for archival storage", while Arrow "eliminates encoding and decoding
overheads" for interchange.

We are accumulating samples — 9 so far at 78 KB, and every BT run adds more.
As an archive that is the wrong shape: nine JSON blobs you must decode one at
a time to ask "which experts were hot in layer 40 across every run". In
Parquet LONG form that is one `duckdb` query over a directory, at 31 KB per
sample.

## What to change

1. **Done:** ask for gzip. 78 KB → 36 KB for one header, on every poll.
2. **Worth doing:** a `heatmap_to_parquet.py` that folds captured samples into
   one partitioned dataset (`run`, `step`, `layer`, `expert`, `count`,
   `tier`). That is the format for the analysis we keep hand-rolling — the
   EN/ZH contrast, the top-K stability traces, R10-CMP — and it makes them
   `SELECT` statements instead of bespoke scripts.
3. **Not worth doing:** changing the live endpoint. It is fit for purpose, and
   the one real inefficiency in the path was mine, not its.

## Sources

- [Apache Arrow FAQ — Arrow vs Parquet](https://arrow.apache.org/faq/)
- [Streaming, Serialization, and IPC — Arrow docs](https://arrow.apache.org/docs/python/ipc.html)
- [Our journey at F5 with Apache Arrow](https://arrow.apache.org/blog/2023/04/11/our-journey-at-f5-with-apache-arrow-part-1/)
- [Arrow IPC support in DuckDB](https://duckdb.org/2025/05/23/arrow-ipc-support-in-duckdb)
