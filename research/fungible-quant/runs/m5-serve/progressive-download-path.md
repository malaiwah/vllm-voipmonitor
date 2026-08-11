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

Sustained bulk-fetch throughput, read off a live TP4 boot (attempt 9,
2026-08-11 13:09–13:10, authenticated, Xet high-performance, depth 1):

```
FQ downloads: 3 in flight, 2.2 GiB this boot, 149 MiB/s
FQ downloads: 3 in flight, 2.8 GiB this boot, 148 MiB/s
FQ downloads: 3 in flight, 5.2 GiB this boot, 142 MiB/s
```

| path | rate | source |
|---|---:|---|
| per-expert ranged fetch | 0.75 MiB/s (45 MiB/min) | earlier measurement, 18.2 s/expert |
| bulk object fetch, HF client + Xet | **142–149 MiB/s** | above |

**~190x.** That is the quantitative case for bulk fetch: it is not an
optimisation of the per-expert path, it is a different order of magnitude.
GLM-5.2's 19,200 experts at 18.2 s each is ~97 hours; the same bytes as
whole objects at ~145 MiB/s is bounded by the model size, not the expert
count.

Caveats, so the number is not read wider than it is: this is *download*
throughput during weight load, not time-to-ready (JIT compilation dominates a
cold boot at ~9 min), and it was measured on one box against one Hub region.

### Not yet measured

Full 75-layer wall-clock weight-load time. Every boot so far has been cut
short by a defect rather than finishing -- see the table below.

## What a cold boot actually costs

Worth stating plainly, because it bounds the claim. Progressive boot fetches
what the policy asks for and does not already have. On this box that split is
lopsided:

| tier | local | consequence |
|---|---|---|
| K2 | 75/75 + index | never fetched |
| K4 | 56/75 + index | 56 layers skip the network entirely |
| K5 | 24/75 + index | never fetched for the layers it covers |
| **K3** | **no index, no segments** | **all 75 layers fetched, ~5 GB each** |

K3 is the seeded policy's BASE tier — 206 of 256 experts at layer 3 — so a
cold cache pays roughly 375 GB before the model is up, and per-layer wall
clock is dominated by that single object:

```
layer 3  13:21:46
layer 4  13:21:49   (+3 s     — both objects already cached)
layer 5  13:30:38   (+8m49s   — fresh K3 fetch)
layer 6  13:35:05   (+4m27s)
```

The 3-second layer is the honest upper bound on what the loader costs when
the bytes are already present; the 9-minute layer is the honest lower bound
on what a cold tier costs. Both are the same code. This is why local-first
resolution matters more than any transfer tuning: the fastest download is the
one that does not happen.

## Bugs this sequence surfaced

| symptom | cause |
|---|---|
| `NameError: name 'logger' is not defined`, all 4 workers dead mid-load | `progressive.py` has no module logger — it uses a local `_emit()` so CPU tests can load it standalone. Added verbosity used `logger.info` |
| boot killed at 600 s with the load progressing normally | `VLLM_ENGINE_READY_TIMEOUT_S` default; a progressive boot downloads *during* load |
| transfers at default speed despite "enabling" fast transfer | `HF_HUB_ENABLE_HF_TRANSFER` is a no-op on `huggingface_hub` 1.27 |
| unbounded cache growth | prefetch had no eviction |
| 4x download volume | no cross-rank sharing |
| 190 experts silently degraded to K2, boot wedged | `self.stats["bytes_from_prefetch"] += ...` incremented an UNDECLARED counter, on the SUCCESS branch of the prefetch fast path — so the KeyError fired exactly when prefetch worked, was caught as a source rejection, and dropped the expert down the K ladder. The better prefetch performed, the more experts degraded |
| 270 identical `REJECT error:KeyError` lines | rejection messages named the exception TYPE and discarded its argument. Fixed first; it then found the bug above in one traceback |
| whole-box DNS outage mid-run | environmental (an interruptible instance) — `git push` from an unrelated process failed identically. Recovered on its own |
