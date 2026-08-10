# FragmentResolver operator controls — multi-repo sources, trust filtering, lazy-encode fallback

Date: 2026-08-10. gg-vllm `fq/m1-stats-collector` commit `b69feebca` (on top
of the M4 swap engine `a16c87f73`); same files synced additively into the
gg-v20-r33 rootfs venv (`fragments.py`, `progressive.py`, new
`lazy_encode.py`). CPU-only change; package suite **77 passed** (was 60),
5 GPU tests skipped as before.

## 1. Configurable multi-repo sources

```
VLLM_FQ_SOURCES=repoA@main,org/repoB          # ordered, repo_id[@revision]
VLLM_FQ_SOURCES_MODE=prepend|replace|append   # default prepend
```

The env list combines with the manifest `sources` chain per the mode
(prepend = env repos are tried first). **Local segment dirs always resolve
first** regardless. Each source keeps its own `index-k{K}.json` fetch memo
and its own attestation disk cache
(`VLLM_FQ_CACHE/attestations/<source>/layer-LLL.kK.jsonl`), so mirrors with
*different* segment encodings of the same layer/K cannot poison each other.
Duplicate specs (env ∩ manifest) collapse to their first position.
Explicitly passed `FragmentResolver(sources=[...])` still bypasses all of
this (test seam).

## 2. Attestation-based trust filtering (10 §4)

```
VLLM_FQ_TRUST_SIGNERS=<hex ed25519 pubkey>[,...]   # default: manifest signer_pubkey
VLLM_FQ_TRUST_PREDICATES=repack-of,encode-of,derived-from   # default shown
```

Enforcement is armed **only when a trust anchor exists** (env signers set,
or the manifest carries `signer_pubkey` — fq_repack/fq_prime outputs do).
Without an anchor the legacy sha-only behavior is unchanged.

When armed, a fragment is accepted from a source only if some line of that
source's `attestations/layer-LLL.kK.jsonl` (fq-attestation/1 envelope:
b64 payload + ed25519 signature + keyid):

1. is signed by an allowed key (`keyid` ∈ trust list, signature verifies
   over the raw payload bytes — PyNaCl or `cryptography`, imported lazily;
   the gg rootfs has cryptography 50.0.0), and
2. carries a trusted `predicate`.

Countersignatures: files may hold several lines for the same fragment; ANY
allowed line passing the predicate filter accepts (a rogue line alongside a
trusted one does not block). The fetched bytes are then verified against
**that trusted line's** `expert_sha256` — integrity sha checks remain
unconditional and rejection of one mirror no longer aborts the chain (the
first verification error is re-raised only if nothing else accepts).

## 3. Lazy-encode fallback ladder (boot never blocks)

```
VLLM_FQ_K_FALLBACK=3            # ordered substitute Ks per miss ("" = off)
VLLM_FQ_ENCODE_QUEUE=<path>     # default VLLM_FQ_CACHE/encode-queue.jsonl
```

If the requested K is unavailable/untrusted everywhere, the resolver walks
the fallback ladder and returns the substitute **marked**: `Fragment.k` is
the actually-loaded K, `Fragment.requested_k` the ask, `.substituted` True.
`progressive_weights_iterator` consumes resolve()+materialize() so the
per-layer tier line and `bits_digest` are computed from **reality** (the
loaded Ks), substitutions are called out
(`substituted=e0:K4->K3,...`), and the new `actual_bits_out` parameter
returns the per-layer loaded bits — feed it to
`write_tier_bitmap(policy, path, actual_bits=...)` so the serve's
`hybrid_tr3_tail` bitmap records loaded Ks, not wishes. (For a GPU serve
with substitutions the bitmap must be synthesized from a pre-flighted
resolve pass, since the exl3 planner sizes slabs from it before weights
stream; CPU-validated here, serve pre-flight is follow-up.)

Every substitution AND hard miss appends
`{layer, expert, k, reason, requested_utc}` to the persisted JSONL queue
(dedup by (L,E,K); O(1) append; queue failures are warnings, never boot
errors). Worker CLI:

```
gg-run.sh python -m vllm.model_executor.layers.quantization.exl3_fungible.lazy_encode \
    --drain [--execute] [--queue PATH] [--bf16-dir D] [--capture-dir D] [--encoder-cmd T] [--limit N]
```

`--drain` alone is DRY-RUN: validates each entry against
`VLLM_FQ_BF16_DIR` (expert weights present in
`model.safetensors.index.json`) and `VLLM_FQ_CAPTURE_DIR`
(`layer_{L:03d}/` payload present), prints the command that would run,
exit 0 iff all entries are runnable, queue untouched. `--execute` formats
`VLLM_FQ_ENCODER_CMD` (placeholders `{layer} {layer03} {expert} {k}
{bf16_dir} {capture_dir}`) per entry, shells out, and removes completed
entries. Default template targets the sha-pinned driver
(**tested-by-dryrun only** — no GPU encode was run in this change):

```
python .../tools/fruit-encoder/fruit_encode_driver.py --encode --bits {k} \
    --layers {layer} --src {bf16_dir} --capture-dir {capture_dir} --workers 1 --gpus 0
```

Note the driver's unit of work is a layer (encodes all its experts — a
correct superset, amortized over every queued miss of that layer);
`{expert}` is reserved for future single-expert drivers.

## 4. Verbose decisions + stats

Every `resolve()` emits one structured chain line — DEBUG on plain
success, INFO on substitution, WARNING on failure. Real line from the test
run (`test_decision_line_full_chain`):

```
INFO 08-10 19:47:36 [fragments.py:840] FQ resolve L3/e0 K4: local(1 dirs) MISS; \
  hf:repoA@ab12 REJECT predicate=derived-from not-trusted; \
  hf:repoB@cd34 REJECT sha-mismatch; FALLBACK K3 local ACCEPT (encode queued #1)
```

Vocabulary: `MISS` / `REJECT <reason>` / `ACCEPT` / `FALLBACK K<k>` /
`UNAVAILABLE`; reasons: `predicate=<p> not-trusted`, `signer not-trusted`,
`bad-signature`, `no-attestation`, `no-expert-sha`, `sha-mismatch`,
`error:<Type>`. `resolver.stats` adds matching counters
(`reject_predicate`, `reject_signer`, `reject_signature`,
`reject_no_attestation`, `reject_no_expert_sha`, `reject_sha_mismatch`,
`source_miss`, `source_error`, `fallback_substituted`, `encode_queued`,
`unavailable`) alongside the existing local/cache/fetched/verified set;
the progressive stream-complete line now also reports
`substituted=` / `encode_queued=`. Perf: decision assembly is one list +
join per resolve on the boot/swap IO path — no per-token work.

## 5. Tests (CPU, fake sources, no GPU)

`tests/exl3_fungible/test_trust_lazy_cpu.py`, +17: source-mode ordering
(unit + behavioral prepend/append, dedup, invalid mode), predicate
rejection + default-accept, signer rejection, countersigner acceptance,
env-signers override, bad-signature (rejected before any body byte is
fetched), trust-off legacy behavior, fallback substitution surfacing the
actual K (resolver + progressive stream + tier-bitmap-from-reality),
queue dedup/persistence, DRY-RUN drain (OK + BLOCKED + rc), --execute
drain via injected runner, decision-line vocabulary at DEBUG/INFO/WARNING,
per-reason counters. Ed25519 signing in tests uses the rootfs
`cryptography`; attestation envelopes match fq_repack's Signer format
byte-for-byte (canonical-JSON payload).
