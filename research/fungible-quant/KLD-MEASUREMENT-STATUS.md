# KLD Measurement Status

## SIQ Baseline: SUCCESS

The SIQ model loads and runs successfully on AIBoss RTX 5090 via the GG vLLM
turnkey container (`glm52-turnkey:r31-vllm258`). Logprobs were collected for
10 calibration prompts.

Container command:
```bash
podman run --rm --gpus all --ipc=host --entrypoint [] \
  -v /tmp/poc_residual:/tmp/poc_residual:Z \
  -v /home/mbelleau/models:/host_models:ro \
  -e TORCH_EXTENSIONS_DIR=/tmp/poc_residual/cache \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  localhost/glm52-turnkey:r31-vllm258 \
  /opt/venv/bin/python3 /tmp/poc_residual/measure_kld_vllm.py \
  --siq-path /host_models/GLM-5.2-SIQ-Fruit-Instruct \
  --k2-path /tmp/poc_residual/fruit_msrt_output/base --tp 1
```

## K2 Base Checkpoint: LOADING FAILED

The K2 base checkpoint loads shards but fails EXL3 tensor validation:
```
ValueError: Invalid EXL3 MoE tensors for expert=0, projection=w1
```

### Root Cause: suh/svh/mcg Format Mismatch

The `fq_assemble_lora` encoder produces tensors that don't match the EXL3
checkpoint format that vLLM's loader expects:

| Tensor | SIQ (correct) | K2 base (our output) | Issue |
|--------|---------------|---------------------|-------|
| mcg | `()` scalar int32 | `(1,)` int32 | Shape: scalar vs 1-element |
| suh | `(1024,)` float16 | `(512,)` float32 | Shape AND dtype wrong |
| svh | `(512,)` float16 | `(1024,)` float32 | Shape AND dtype wrong |
| trellis | `(32,64,48)` int16 (K3) | `(32,64,32)` int16 (K2) | Correct (K=2) |

The `compute_hadamard_vectors()` function in `fq_assemble_lora.py` produces
incorrect suh/svh shapes. The EXL3 format stores:
- `suh`: per-row sign/scale vector with shape = (input_size,), dtype = float16
- `svh`: per-column sign/scale vector with shape = (output_size,), dtype = float16
- `mcg`: scalar int32 sentinel (shape `()`, not `(1,)`)

### Fix Required

The `compute_hadamard_vectors` and trellis writing in `fq_assemble_lora.py`
need to match the exact EXL3 checkpoint format:
1. suh/svh must be float16, shaped as (rows,) and (cols,) respectively
2. mcg must be a scalar tensor, not 1-element
3. The suh/svh content must be the sign vectors used in regularization, not
   the combined scale+sign vectors

This is a format-level bug in the encoder, not a fundamental algorithm issue.
The weight-level MSE measurements (which use the PoC's independent quantization
directly, not the checkpoint format) are valid and show the key results:
- MSRT K2+K2trsc matches K4 at 4bpw
- MSRT K2+K1+K2trsc is 3.6× better than K4 at 5bpw

## KLD Results

KLD could not be computed because the K2 base checkpoint fails to load in
vLLM due to the suh/svh/mcg format mismatch. Once the format is fixed,
the KLD measurement script is ready to run.

The SIQ baseline logprobs were collected successfully, confirming the
vLLM GG + EXL3 + Fruit model serving path works.
