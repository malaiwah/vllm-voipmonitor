# Progressive Loader battle-test plan

Ten tests (BT-0 … BT-9) covering the full operational lifecycle, not just the
cold boot. Written after the encode campaign was paused, because the campaign
was consuming the disk headroom and the four GPUs that these tests need.

## Why this plan exists

Everything proven so far is about **arriving**: a checkpoint assembled from
segments boots, serves at 219 tok/s, scores 89.2% on GSM8K. None of it is about
**living** — what happens on the second boot, when the desired K is not
available, when the missing fragments arrive an hour later, or when the box is
preempted mid-swap. Those are the states an operator actually spends time in,
and they are the states with no evidence behind them.

A second reason: this pipeline has produced at least seven distinct failures
that looked exactly like success — processes alive, exit code 0, plausible
logs, nothing produced. Every test below therefore asserts on an **artifact**
(a byte counter, a segment count, a composition table, a sealed file), never on
liveness or wall-clock alone.

## The lifecycle under test

```
   BT-1 cold boot ──► BT-2 hot restart ──► BT-3 offline restart
                              │
   BT-4 degraded boot ────────┴──► BT-5 converge ──► BT-6 converge under load
                                        │                    │
                                   BT-7 quality delta    BT-8 restart converged
                                                              │
                                                         BT-9 kill -9 mid-swap
```

BT-0 gates all of them.

---

## BT-0 — Reclaim disk + define the cache-retention policy

**Prerequisite, not a test.** Two things are wrong right now.

*Disk.* The campaign holds encoder shards and window captures for work already
published to HF. Verify each K2/K4 layer is present on HF (`list_repo_files`,
compare per-layer segment counts), then drop the local shards and captures,
keeping `*.done.json` for idempotency. Target: >600 GB free.

*Retention.* The floor-triggered guard prunes the fragment cache as it fills.
Measured today: **191 GiB delivered, 74 GB retained, of which `segments/` holds
1.9 GB and `fragments/` 70 GB.** A restart would re-download almost everything.
That is not a slow warm restart — it is a cold restart wearing a warm label,
and it would silently invalidate BT-2.

Fix: make retention *tiered* rather than LRU-by-accident. Pin the base tier the
policy currently depends on; let speculative higher-K fragments be evictable.
Eviction must be a decision the policy makes, not a side effect of a disk
watermark.

**Exit criteria:** >600 GB free; base-tier working set pinned and measured;
`du` of the pinned set recorded so BT-2's zero-fetch assertion has a baseline.

---

## BT-1 — Cold boot from network, time-to-serve baseline

Finish the in-flight boot and record: wall time to `Application startup
complete`, GiB delivered, sustained MiB/s, the JIT share vs the fetch share,
and the startup composition table.

Known good: bulk layer fetch runs 142–149 MiB/s against 0.75 MiB/s for
per-expert ranged reads (~190×). Cold JIT is ~9 min against an empty private
cache and must be reported separately — it is a CUDA cost, not a loader cost,
and folding the two together flatters or damns the loader for the wrong reason.

**Exit criteria:** a serving endpoint and one row in the results table that
every later run is diffed against.

---

## BT-2 — Hot restart: same policy, warm cache

The headline operational claim. Restart with an unchanged policy against the
primed cache.

**Assert on bytes, not seconds.** The delivered counter must be ~0. A restart
that is merely *faster* proves nothing — it could be page cache, a warmer JIT,
or a quieter network. Only a zero fetch counter proves the cache was used.

Also assert the composition table is byte-identical to BT-1's: same experts, same
K, same mean bits. A restart that comes back at a *different* posture has lost
state, even if it comes back fast.

**Exit criteria:** ~0 GiB fetched; TTFS recorded against BT-1; identical table.

---

## BT-3 — Offline restart, `HF_HUB_OFFLINE=1`

Boot with the env var set and the network unreachable. This path was silently
ignored until it was fixed today — the loader used `hf_hub_url` for the URL
string and then went to raw urllib, which no HF env var can reach.

Assert: boot succeeds from local dirs and cache alone; the posture banner logs
OFFLINE; a genuinely missing fragment degrades down the ladder as a MISS rather
than raising. `OfflineError` must not be retried — retrying an offline error is
just a slower failure.

**Exit criteria:** served tokens with no network; one deliberate missing
fragment shown degrading cleanly.

---

## BT-4 — Deliberately degraded boot

Boot a policy demanding K4/K5 for experts that have no fragments anywhere.

Assert: no crash; each miss walks *down* the ladder; every fallback is recorded
as a deficit; `encode-queue.jsonl` receives the request. Then check the
composition table **reports what it actually loaded**, not what was asked for.
A table that echoes the request is worse than no table — it makes a degraded
serve indistinguishable from a healthy one.

**Exit criteria:** measured distance below desired posture; deficit ledger
matches the table; queue populated.

---

## BT-5 — Convergence: background fetch/encode repays the deficits

From BT-4's state, let `ConvergenceWorker` run: fetch what exists, encode on the
fly what does not, live-swap through `fq_converge_layers`.

Verify the swap **installed**, not that the RPC returned — this call has
previously reported success for work it never performed. Check the resident
tensor, not the return value.

**Exit criteria:** time-to-converge; per-layer progress trace; final
composition == desired posture; memory flat throughout (fixed cardinality means
promotion comes out of budget, never out of headroom that does not exist).

---

## BT-6 — Convergence under load *(the flagship demo)*

BT-5 while saturating the serve.

Record the swap timeline against decode throughput via `swap_evidence.py`,
capture composition-table diffs showing experts moving K3→K4 live, and assert:
zero request failures, no throughput collapse during swaps, constant memory.

This is the image the PR is built around — a live server upgrading its own
experts while answering. Everything else in this plan is scaffolding for it.

**Exit criteria:** annotated throughput-vs-swap chart; a table diff with a
timestamp; failure count 0.

---

## BT-7 — Quality delta: does the converged posture help?

GSM8K on the degraded arm vs the converged arm, **same 250-item subsample, same
seed, paired**. At 250 items the standard error is ≈2 points, so an unpaired
1–2 point delta is noise. Pairing is what makes a small delta readable.

Baseline to beat: 89.2% flexible-extract on flat K3.

This is the only test that asks whether re-tiering improves *output* rather than
just moving bits around. It is also the one most likely to return an
uncomfortable answer, which is exactly why it is in the plan and why the result
gets published either way.

**Exit criteria:** paired per-item comparison, delta with CI, published
regardless of sign.

---

## BT-8 — Restart after convergence: does the posture persist?

Restart once converged. Assert the serve returns at the *converged* posture
without re-deriving it — the policy store carries committed membership, the
cache carries the fragments.

Failure mode to catch: it reverts to the seed posture and silently re-does hours
of convergence, looking healthy the entire time. Supersedes M5-E.

**Exit criteria:** post-restart table == pre-restart table; ~0 GiB fetched.

---

## BT-9 — `kill -9` mid-swap: crash safety

Hard-kill during an active swap, restart, inspect.

Assert: no torn layer, no half-installed expert, no corrupt cache entry,
automatic recovery. Staging completes before quiesce, so this *should* be safe
by construction — the point is to prove it on the real model, on a box that gets
preempted for real, rather than in a unit test where the timing is polite.

**Exit criteria:** post-crash restart serves; assembled-tensor hashes unchanged
for the layer that was mid-swap.

---

## Ground rules

- One tmux window per run; on-disk resumable state; commit and push after each
  completed test.
- Never assert on liveness. Assert on artifacts: byte counters, segment counts,
  sealed files, table diffs.
- Any run that touches correctness (KLD), runtime memory, or throughput (PP and
  TG) is instrumented, and the instrumentation ships with it.
- Deploy to the rootfs before every boot (`deploy-fq.sh`) — the serve loads
  `exl3_fungible` from the extracted r33 rootfs, not the source tree. Three
  wasted boots so far trace to exactly this.
- Verify GPUs are free before launching. Pattern-based `pkill` does not match
  the TP workers (they exec through the rootfs `ld-linux` shim, so argv contains
  neither "vllm" nor "VLLM::"); reap by device via
  `nvidia-smi --query-compute-apps`.
