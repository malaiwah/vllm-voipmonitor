# demo-1 first boot: FragmentUnavailableError killed the engine

Recorded because it is exactly the failure mode the operator asked about —
"what happens if a weight is not available in the required K bpw? it needs
not to crash" — and the answer, on this path, was that it crashed.

## What happened

```
FQ resolve L3/e19 K3: local(1 dirs) MISS;
  hf:malaiwah/GLM-5.2-EXL3-FQ-segments@main REJECT error:IncompleteRead;
  UNAVAILABLE (encode queued)
-> FragmentUnavailableError -> WorkerProc init failed -> EngineCore dead
```

## Three causes, stacked

1. **No local K3 segments.** I deleted all 76 during disk reclaims — they were
   verified published to HF first, which made them "safe" to drop, and that
   reasoning was right for durability and wrong for this boot. There is not
   even an `index-k3.json` left. So every one of 19,200 routed experts had to
   be fetched over the network.
2. **A transient HF error was treated as terminal.** `IncompleteRead` is a
   retryable network condition; the resolver counted it as a REJECT for that
   source and moved on to fail.
3. **The missing-K hardening does not cover this call site.** `resolve_best()`
   was added to never raise, but `progressive.py:348` calls
   `resolver.resolve()` directly, which still raises. The hardening work
   traced callers of `resolve()` and hardened the swap path; the progressive
   BOOT path kept the raising variant.

Cause 3 is the one that matters beyond this run: boot is precisely when a
fragment is most likely to be missing, and it is the least acceptable place to
die. A missing expert at boot should degrade to the nearest available K and
log it, exactly as the swap path now does.

## Note on the queue

`encode-queue.jsonl` did receive an entry before the raise, so the enqueue side
of on-the-fly encoding works. The failure is in what happens *after* the
enqueue: it should continue with a lower K, not abort.
