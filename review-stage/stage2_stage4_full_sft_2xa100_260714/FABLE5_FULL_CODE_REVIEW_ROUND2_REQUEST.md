# Fable5 Round-2 full executable review request

Date: 2026-07-14

## Review target

- Branch: `stage2-4-full-2xa100-review-260714`
- Frozen code candidate: `aeaf4e5`
- Round-1 baseline: `00c7493`
- Normative specification: `IMPLEMENTATION_SPEC.md`
- Round-1 report: `FABLE5_FULL_CODE_REVIEW_ROUND1.md`

Review the actual code and failure paths in the frozen candidate.  Do not
approve from this summary alone.  The only acceptable final verdict is
`APPROVE_TO_RUN` or `BLOCK` followed by exact, executable blockers.

No 2xA100 training, CUDA, NCCL, or real UVM allocation has been run from this
candidate.  This is a pre-run code review; do not treat local mocks or static
tests as GPU evidence.

## Fixed Round-1 findings

1. The instantiated model is bound before model construction to the exact
   approved DeepSeek-R1-Distill-Llama-8B revision and a committed seven-file
   size/SHA-256 manifest.  The runtime rejects extra top-level loadable files,
   wrong architecture/class/config/path, wrong parameter count/dtype, and any
   pause-token ID/resize drift.
2. The first real optimizer update now audits every allocated `state1` and
   `state2` on every rank, including dtype, shape, quantization metadata,
   distinct tensor identities, paged-buffer identity, exact
   `GlobalPageManager` membership, state step, and complete parameter
   coverage.
3. `eval_strategy` and `save_strategy` are pinned to `steps`.
4. Cold partial GC is conservative: it deletes only exact managed partials
   older than the threshold whose bound owner PID is proven dead twice;
   complete recoverable partials are recovered, and unknown partials remain.
5. Watchers use `python3` consistently.  `final/` is explicitly
   non-authoritative; all downstream consumers bind sealed checkpoint-1064.

## Mandatory optimizer decision

The formal optimizer remains `paged_adamw_8bit`, now pinned to
`bitsandbytes==0.46.1`.  The code preserves exactly the Transformers 4.52.4
automatic override for `model.embed_tokens.weight`: its two moments are FP32
but paged.  `lm_head.weight` and all other large parameters remain uint8.
This costs approximately 2.94 GiB per rank relative to quantizing the two
embedding moments.

The first-step gate now requires the process-global manager registries to be
an exact whitelist: both `index2config` and `pid2config` must contain only the
input-embedding entry and its value must be exactly `{"optim_bits": 32}`;
the sole module override must match it.  It also rechecks the effective
`get_config()` update rule after manager overlays (learning rate, betas,
epsilon, weight decay, bits, minimum size, max-unorm, and skip-zeros).

Give an explicit decision:

- `APPROVE_EMBEDDING_FP32_PAGED_OVERRIDE`, or
- `REJECT_EMBEDDING_OVERRIDE_AND_REQUIRE_ALL8BIT`, with a concrete reason.

Also confirm whether keeping paged AdamW8bit canonical is approved.  A
non-paged A/B performance benchmark is optional systems work and may not
change the optimizer mid-run.

## Resume, provenance, and storage paths to re-review

- Exact versions are captured for Python, PyTorch, Transformers 4.52.4, TRL
  0.8.1, Accelerate, bitsandbytes 0.46.1, tokenizers, safetensors, CUDA
  runtime/driver, NCCL, vLLM, and rclone.  The three algorithm-sensitive
  package versions are hard startup gates.
- Provenance contains the approved model-file manifest, resolved and semantic
  config hashes, code commit/diff, exact base/resized parameter and tensor
  counts, full gradient audit, optimizer audit, seed/world-size/batch, data
  manifests, checkpoint lineage, and every checkpoint file hash.
- Resume validates path-portable semantic lineage plus stable run-ID/R2-root
  bindings.  It rehashes the sealed parent, restores model/optimizer/scheduler
  and RNG, recreates all large bnb moments as UVM paged buffers in bounded
  chunks, verifies content digests and exact manager identities, then performs
  a second full sealed-checkpoint rehash after RNG restoration.
- Neither watcher starts on resume until both ranks finish restore and rank 0
  atomically writes a unique nonce-bound readiness record outside the managed
  checkpoint tree.  A stale or preplanted readiness path is rejected before
  training is spawned.
- The lifecycle is `/dev/shm` hot seal -> verified/atomic `/workspace` cold ->
  verified immutable R2 prefix -> local deletion.  Deletion occurs only after
  destination-bound receipts and download-side rehashes.
- Re-review DDP error symmetry, callback ordering, watcher liveness, crash
  paths, and the temporary GPU-memory peak while ordinary loaded optimizer
  tensors are replaced by UVM buffers.

## Scientific-contract checks

- True full-weight SFT; no LoRA, rows-only KL, PPC, pause port, forced pause,
  or early stopping.  Two A100 ranks, batch 1/rank, accumulation 16, global
  batch 32, 17,000 rows, two epochs, exact terminal step 1,064, seed 260615,
  BF16/TF32, maximum sequence length 4,096.
- The primary layer grid is the exact Stage-1 four-source grid with only
  terminal readout index 32 removed:
  `[4,6,7,8,10,12,14,16,17,18,20,21,22,24,25,26,28,30]`.
  Index 32 must remain diagnostic-only and must never reach Stage3 selection,
  its confirmatory gate, a direction artifact, or Stage4 steering.
- Stage3 uses 400 frozen candidates (30 TRAIN + 70 SEALED per source), each
  scheduled for exactly 100 natural draws: 40,000 generations.  `10 eligible
  TRAIN prompts/source` is only a liveness floor.  A paper-level confirmatory
  claim requires at least 30 eligible SEALED prompts/source and 120 total,
  with all ineligible prompts retained in coverage reporting.
- Layer selection is training-only nested four-source LOSO.  vLLM is limited
  to natural rollouts and judging; exact HF replay is used for hidden states
  and steering.  Stage4 remains the matched-relative A0--A5 design.

State explicitly whether the dense grid is scientifically consistent with
Stage1 and whether the sealed adequacy rule is sufficient for a confirmatory
paper claim.  If not, give the exact alternative before any rollout is run.

## Local verification evidence

- Formal targeted suite: `130 passed, 2 skipped`.
- Broad suite after excluding two known local-environment surfaces:
  `320 passed, 13 skipped`.
- With only the direct-Torch Stage4 collection file excluded, the broad suite
  is `326 passed, 13 skipped, 1 failed`; the sole failure is an existing macOS
  `/tmp` -> `/private/tmp` path-resolution assertion in a Stage1 test.
- The raw suite cannot collect `tests/test_stage4_targeting.py` because this
  local Python environment has no PyTorch.  Real Torch hook tests and the
  2xA100 preflight remain mandatory.
- `py_compile`, `bash -n`, JSON parsing, and `git diff --check` pass.  The shell
  reports only the local missing `C.UTF-8` locale warning.

Please independently run every test your environment permits and distinguish
approval-blocked commands from executed commands.

## Required report structure

1. Review scope and commands actually executed.
2. Findings ordered as blockers, major nonblockers, and minor nonblockers.
3. Explicit optimizer/embedding-override decision.
4. Explicit resume/storage/provenance verdict.
5. Explicit layer-grid and sample-adequacy verdict.
6. Status against all six professor questions.
7. Final verdict: exactly `APPROVE_TO_RUN` or `BLOCK` with exact blockers.
