# Progressive Loader: how segments actually get fetched

Written for the PR. Every claim here is either a code property or a measured
observation with the log line that produced it; where a number is not yet
measured on the real model it is marked **TBM** rather than estimated.

## The question this answers

> Are the segments (or full layers) downloaded in parallel or in sequence?
> Is downloading pipelined with tensor loading, or does it happen as a block?
> Is it using the HF client (and token) whenever possible?

## Four levels of concurrency, and where each one came from

`progressive_weights_iterator` is a generator consumed by
`model.load_weights()`. That makes fetching naturally streamed — fetch, yield,
GPU load — and every optimisation below has to preserve that property rather
than trade it away.

| level | before | now |
|---|---|---|
| chunks within one object | parallel (Xet) | parallel (Xet) |
| the two Ks a mixed layer needs | **sequential** | one task each, concurrent |
| across layers | **1 layer, `max_workers=1`** | `VLLM_FQ_PREFETCH_DEPTH=3` |
| across the 4 TP ranks | **4x duplicate downloads** | one fetch, shared |

### Chunks within an object

Whole-object pulls go through `hf_hub_download`. This is the only path that
gets connection reuse, resume, Xet dedup and parallel chunked transfer; the
ranged path cannot use it, because arbitrary byte ranges are outside its API.

**`hf_transfer` is not the knob.** `huggingface_hub` 1.27 deprecated it — the
constant `HF_HUB_ENABLE_HF_TRANSFER` no longer exists in `constants.py`, and
setting it produces only a `FutureWarning` while transfers run at default
speed. The live knob is `HF_XET_HIGH_PERFORMANCE`, and `hf_xet` is present in
the GG rootfs.

### Across the Ks of one layer

A mixed layer draws from **two** objects (e.g. 192 experts from the K3 object,
64 from the K4 object). Submitting them as one task ran them back to back,
leaving the link idle between files even with chunk parallelism inside each.
Now one task per `(layer, K)`.

### Across layers

One layer of lookahead only hides the download if the download is faster than
the GPU load. It usually is not, so the loader went back to blocking. Depth
defaults to 3 layers ahead, each up to 2 objects wide.

This replaced a **self-inflicted regression**: the first bulk-prefetch version
downloaded a whole ~2.5 GB segment *synchronously before yielding anything*,
stalling the GPU for minutes per layer. It traded pipelining for throughput.

### Across TP ranks — the largest single win

Every TP rank runs its own weight iterator over the same policy, so all four
want the same objects simultaneously. From a real TP4 boot:

```
(Worker_TP0) FQ progressive layer 3: tiers=((3, 206), (4, 50)) bits_digest=d704612a2fdb
(Worker_TP1) FQ progressive layer 3: tiers=((3, 206), (4, 50)) bits_digest=d704612a2fdb
(Worker_TP2) FQ progressive layer 3: tiers=((3, 206), (4, 50)) bits_digest=d704612a2fdb
(Worker_TP3) FQ progressive layer 3: tiers=((3, 206), (4, 50)) bits_digest=d704612a2fdb
```

Identical composition, identical digest — identical segment requirements. The
cache confirmed it: four `.part` files racing for one object.

```
2.1 GiB  segments/tmp3mhjg_1z.part
1.6 GiB  segments/tmpft9a4vg7.part
1.1 GiB  segments/tmpf8g1t_f_.part
0.8 GiB  segments/tmpno440lz5.part
```

That is 4x the bytes and 4x the transient disk. With depth x width it would
have become ~24 concurrent fetches of ~6 distinct objects. Now one rank takes
an `flock`, downloads, and the others find the file already present.

The lock **falls through on timeout** instead of failing the boot: a duplicate
download is wasteful, not incorrect, because the rename is atomic.

## Bounded footprint

Prefetch originally had **no eviction**. `_prefetched` grew monotonically and
nothing unlinked, so a 75-layer boot left every object it touched on disk —
bounded by model size, not by prefetch depth. On this box (3.0 T at 88%, a
180 G campaign floor) that is disk exhaustion, not a slow cleanup.

`release_layer()` drops a layer's objects once its tensors have reached the
GPU, making resident bytes O(depth) rather than O(model). Two rules keep it
safe:

- **Only our own cache dir.** `hf_hub_download` is given `local_dir` so the
  blob is ours to unlink; a blob in the *shared* HF cache may be in use by
  another process and is never touched. `FQ_PREFETCH_HF_SHARED=1` opts back
  into the shared cache (and out of eviction).
- **Refcounted across ranks.** Rank 0 finishing layer 3 must not delete a
  segment rank 2 is still reading — that race would silently demote rank 2 to
  ranged reads. Markers from killed ranks are pruned via `/proc`, so a
  preemption cannot pin disk forever.

## Authentication

Unauthenticated Hub requests are rate-limited. The Hub says so itself, once
per rank:

```
(Worker_TP1) Warning: You are sending unauthenticated requests to the HF Hub.
             Please set a HF_TOKEN to enable higher rate limits and faster downloads.
```

`serve-demo1.sh` now sources the token from `~/.fq_env` in a subshell that
prints only the token — no other secret enters the server environment — and
the boot banner reports presence, never the value.

## Why bulk fetch is a requirement, not an optimisation

Per-expert ranged fetch measured **18.2 s/expert** (TLS handshake, attestation
lookup and a ranged GET each). GLM-5.2 has 19,200 routed experts across 75
layers: **~97 hours**. Pulling each needed object once is the same bytes in
~2 requests per layer.

## Operator visibility

An operator cannot distinguish a slow download from a hung boot in a silent
log. `prefetch_whole` reports every 256 MiB:

```
FQ fetch L3 K3 layer-003.k3.safetensors: 1.2/3.7 GiB (33%) at 214 MiB/s
```

and a layer that has to block on its own prefetch says so explicitly:

```
FQ progressive L4: waiting on background prefetch (206 experts want K3, 50 experts want K4)
```

## Measured end-to-end

**TBM** — first clean boot with all four levels active. The comparison that
belongs here is wall-clock weight-load time against the recorded per-expert
baseline (45 MiB/min, 18.2 s/expert).

## Bugs this sequence surfaced

| symptom | cause |
|---|---|
| `NameError: name 'logger' is not defined`, all 4 workers dead mid-load | `progressive.py` has no module logger — it uses a local `_emit()` so CPU tests can load it standalone. Added verbosity used `logger.info` |
| boot killed at 600 s with the load progressing normally | `VLLM_ENGINE_READY_TIMEOUT_S` default; a progressive boot downloads *during* load |
| transfers at default speed despite "enabling" fast transfer | `HF_HUB_ENABLE_HF_TRANSFER` is a no-op on `huggingface_hub` 1.27 |
| unbounded cache growth | prefetch had no eviction |
| 4x download volume | no cross-rank sharing |
