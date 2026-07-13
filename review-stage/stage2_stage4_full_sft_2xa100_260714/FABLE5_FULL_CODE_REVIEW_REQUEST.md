# Fable5 full code review request: formal Stage 2--4, 8B, 2xA100

## Review mode

Review the executable code in this branch, not only the protocol prose.  This
is a pre-run review: no formal 8B results are being claimed yet.  Report every
finding with severity (`BLOCKER`, `MAJOR`, `MINOR`), exact file/line evidence,
the consequence, and a concrete fix.  End with one verdict:

- `APPROVE_TO_RUN`
- `APPROVE_AFTER_LISTED_FIXES`
- `REDESIGN_REQUIRED`

You may propose or patch genuine correctness/design defects.  Do not replace
the frozen research question with a different method or revive the archival
LoRA, rows-only KL, PPC, learned-delta, projection-clamp, or forced-runtime-pause
protocols.

## Pre-review integrity evidence

- The review branch was rebuilt from `feb45451aa7d13ba2642e55d64e8922349f54b88`
  with a formal-only patch.  Historical LoRA, PPC, pause-port, forced-pause,
  and expanded pause-KL executable paths were not copied into the three mixed
  training entrypoints; their canonical negative sentinels remain.
- Formal CPU/static suite: `172 passed, 4 skipped`.  The four skips are
  optional real Torch/Transformers model-hook tests unavailable in the local
  lightweight environment, not failed assertions.
- Fourteen formal CLI `--help` import checks, Python compilation, three shell
  syntax checks, and `git diff --check` pass.
- This evidence is not a substitute for the requested code-path review or the
  2xA100 preflight.

## Frozen scientific contract

- Base model: DeepSeek-R1-Distill-Llama-8B.
- Stage 2 is true full-weight SFT, not LoRA and not rows-only/KL-transparent
  training.  It inserts exactly three pause tokens after `cot_4` and before
  `cot_5` using a neutral, decontaminated reasoning corpus.
- Stage 2 uses 17,000 train / 500 validation / 500 test rows, two epochs,
  2 GPUs, micro-batch 1/GPU, gradient accumulation 16, global batch 32,
  seed 260615, epoch-driven `max_steps=-1`, no early stopping, no best-model
  loading, and exactly 1,064 optimizer steps under Transformers 4.52.4.
- Stage 3 freezes 30 direction-training and 70 sealed candidate prompts per
  source across four sources.  Each prompt has exactly 100 scheduled draws,
  retained whether generation/judging succeeds or fails.  Eligibility is
  >=90 valid exact-location primary labels with >=5 safe and >=5 unsafe.
- The 10 eligible training prompts/source threshold is a liveness floor, not
  the paper test sample.  The formal sealed adequacy gate is >=30/source and
  >=120 total, and every eligible sealed prompt is analyzed.
- Stage 3 uses training-only nested four-source LOSO to select a layer.  The
  primary grid is the Stage-1 grid minus the non-steerable terminal readout:
  `[4,6,7,8,10,12,14,16,17,18,20,21,22,24,25,26,28,30]`.
  Hidden-state index 32 is a readout diagnostic only because injection there
  cannot alter subsequent KV state.
- Stage 4 retains the reviewed matched-relative algorithm:
  `h' = h - rho * ||h|| * unit(unsafe - safe)`, with
  `rho = alpha * 0.10`.  Full-SFT changes require recomputing every
  checkpoint-dependent hidden artifact, direction, layer, random control,
  calibrated alpha, generation, and judgment; they do not authorize a new
  steering objective.
- Stage 4 arms are A0 base, A1 full-SFT unsteered, A2 pause steering, A3
  matched pre-pause ordinary-token steering, A4 matched post-pause steering,
  and A5 matched orthogonal-random pause perturbation.
- Alpha is selected only on the frozen A1/A2 calibration set using WildGuard,
  as the smallest viable nonzero point-estimate candidate.  Final harmful and
  all benign runs must bind the exact passed calibration-report SHA-256.

## Professor questions this implementation must answer

1. Demonstrate the pause is a clean intervention point using matched A2 vs A3
   and A2 vs A4 capability/coherence/semantic counterfactuals.
2. Add prompt-only / pre-CoT probes so trajectory signal is not silently prompt
   classification.
3. Replace the abandoned six-source statement with honest four-source LOSO and
   report variance/transfer across all retained sources.
4. Separate the SFT-data effect (A0 vs A1) from steering, and audit whether the
   hidden direction tracks refusal, length, lexical/style/source nuisances.
   These controls diagnose construct validity and do not replace the hidden
   signal gate.
5. Name and bridge the teacher-forced/vLLM/HF replay vs self-generated
   distribution/engine mismatch.
6. Always report absolute residual unsafe rates (including unknown/failure),
   not only relative reductions.

## Mandatory implementation audit

### Stage 2 optimizer and memory

The canonical configuration currently uses `paged_adamw_8bit`.  Determine
whether this should remain canonical on 2x80GB A100 or whether
`adamw_bnb_8bit` (same AdamW/8-bit semantics without UVM paging) has a credible
speed advantage at the exact worst-case 4,096-token workload.  Do not recommend
a change solely from static estimates.  If a preflight is necessary, specify
the smallest apples-to-apples clean-process protocol and hard abort thresholds
for allocated/reserved/NVML memory, CPU/UVM/PCIe traffic, finite loss/gradients,
parameter coverage, and step throughput.  Treat fused PyTorch AdamW,
Adafactor, Lion, GaLore, etc. as different algorithms unless proven otherwise.

Pre-review note: no such GPU A/B benchmark has been run yet, so this branch
deliberately remains on the paged canonical optimizer.  The code locks the
instantiated class to `bitsandbytes.optim.adamw.AdamW`, requires literal
boolean `is_paged=true`, `optim_bits=8`, exact group hyperparameters, full
parameter coverage, and instantiated `SFTTrainer.max_seq_length=4096` in the
provenance.  Decide whether inspection of the first allocated optimizer state
(`state1`/`state2` dtype and coverage, plus paged-manager registration) must
also be a hard pre-run gate, and whether the A/B benchmark must be implemented
before `APPROVE_TO_RUN` or can remain an optional performance optimization.

### Full-train and provenance

Verify that instantiated—not merely configured—objects prove:

- full 8B parameter trainability and exact optimizer coverage with no PEFT;
- nonzero finite first-step gradients across all decoder layers and pause-token
  embedding/output rows;
- the middle-layer checksum changes after the first optimizer step;
- exact optimizer class, paged/8-bit flags, betas, epsilon, weight decay;
- exact package/CUDA/driver/NCCL/vLLM/rclone versions;
- base model revision/content hash, tokenizer/chat-template/pause-token hash,
  dataset/freeze/config/code hashes, and terminal checkpoint-1064 provenance;
- no inherited positive `MAX_STEPS`, early stopping, best-checkpoint loading,
  stale resume, or prior SFT checkpoint as the starting model.

### Checkpoint storage lifecycle

Audit the failure and success paths for:

`/dev/shm hot -> sealed /workspace cold -> verified R2 -> local deletion`.

No local checkpoint, final model, or metadata may be deleted before a
destination-bound receipt and download-side hash check succeeds.  Training
must stop if either watcher exits early.  Final hot sync must finish before R2
finalization.  Restore must download into a same-parent partial directory,
verify checkpoint-1064 plus provenance/receipt, atomically rename, and clean a
failed partial without exposing an unverified final directory.

### Backend boundaries

- vLLM is appropriate for natural fixed-budget rollouts and open-model judges.
- HF exact-token replay/hooks are required for hidden extraction and steering
  before later-layer KV storage.
- Confirm no code incorrectly assumes a replay edit at hidden-state index 32
  reaches future KV state.
- Confirm terminal scheduled failures remain in all denominators and are never
  replaced by extra draws.

### Data freeze question requiring an explicit answer

The external cosine audit is exhaustive at >=0.90 and content-bound.  Current
validation requires manual decisions for every pair actually listed in
`top_neighbors union threshold_hits`.  Clarify the minimum scientifically
necessary interpretation of “top-neighbor manual audit”:

1. every candidate's nearest neighbor (~18,000+ manual pairs), or
2. every unique >=0.90 hit plus a preregistered global/stratified top-K or gray
   band `[0.85,0.90)` sample, while all per-candidate nearest neighbors remain
   automatically recorded and hash-bound.

Also state whether connected-component adjudication may cover redundant
>=0.90 edges.  Do not silently weaken the current fail-closed implementation;
make any approved rule explicit in code/config/tests.

### Stage 3/4 leakage and calibration

Confirm that sealed labels cannot choose the final Stage-4 layer/direction,
that diagnostics cannot enter the signal gate, and that the calibration report
mechanically proves A1/A2 alpha-zero token identity, the frozen candidate order,
target-norm integrity, and selection of the first passing alpha.  Verify final,
capability, compliance, semantic-task, and final-analysis inputs all carry the
same report SHA and checkpoint/artifact/ledger bindings.

## Primary files

- `review-stage/stage2_stage4_full_sft_2xa100_260714/IMPLEMENTATION_SPEC.md`
- `configs/experiment/stage2_intra_pause_sft_8b_2xa100.yaml`
- `src/cot_safety/training/` and `legacy/COTPauseToken/src/trl_train.py`
- `pipelines/runpod_watch_hot_checkpoints.sh`
- `pipelines/runpod_watch_cold_checkpoints_to_r2.sh`
- `src/cot_safety/data/stage2_formal_freeze.py`
- `configs/experiment/stage3_formal_8b_2xa100.yaml`
- `src/cot_safety/probes/stage3_*.py` and formal Stage-3 scripts
- `configs/experiment/stage4_full_sft_clean_8b_2xa100.yaml`
- `src/cot_safety/steering/stage4_formal.py`
- `src/cot_safety/steering/stage4_generation.py`
- `src/cot_safety/eval/stage4_calibration.py`
- `src/cot_safety/eval/stage4_formal_analysis.py`
- formal Stage-4 generation/calibration/judging/analysis scripts and tests

Please trace real call paths, run the available CPU/static tests, and inspect
resume/failure branches—not only happy-path functions.
