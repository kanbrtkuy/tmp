# Stage2 8B Full Review Brief

Date: 2026-07-07
Reviewer requested: Claude/Fable-5

This brief summarizes the current 8B Stage2 full run before starting Stage3.
No raw prompts, raw completions, secrets, HF tokens, or R2 credentials are included.

## Question For Fable

Should we proceed to Stage3 with this 8B Stage2 checkpoint, or is the observed
GSM8K natural over-emission a blocker that must be fixed before Stage3?

If Stage3 proceeds, should it use natural pause, forced pause, or both?

## Intended Four-Stage Goal

1. Stage1: verify early/intra-CoT hidden-state separability for future unsafe CoT.
2. Stage2: train a pause-token model that emits harmless pause tokens while
   preserving base-model reasoning and safety behavior.
3. Stage3: test whether pause hidden states contain separability beyond prompt
   baselines and matched no-pause content controls.
4. Stage4: if Stage3 passes, use pause states as a live steering port to reduce
   unsafe CoT without increasing over-refusal, capability loss, broken outputs,
   or format/length artifacts.

The current pause placement is after CoT token 4 / before CoT token 5:

```text
<think> t0 t1 t2 t3 t4 <pause><pause><pause> t5 ...
```

## Stage2 Method

Model:

```text
DeepSeek-R1-Distill-Qwen-8B
```

Training method:

- KL-transparent emit SFT, not ordinary full SFT.
- Insert 3 pause tokens after `cot_4` / before `cot_5` using tokenizer offset alignment.
- Supervise pause emission.
- Keep continuation distribution close to the no-pause reference with KL transparency.
- Full run on 2x A100 80GB, with early stopping and eval every 25 steps.

Training config:

```text
configs/experiment/stage2_intra_pause_kl_transparent_emit_8b_cot5_full_save25_2xa100.yaml
```

Model comparison eval config:

```text
configs/experiment/stage2_model_comparison_eval_8b_kl_transparent_emit_cot5_full_save25_mb4_ga2_2xa100.yaml
```

Run directory:

```text
/workspace/cot-safety/runs/stage2_8b_full_mb4_ga2_260706
```

Final checkpoint:

```text
/workspace/outputs/deepseek_8b_intra_pause_cot5_kl_transparent_emit_trusted_cot_18k_full_save25_mb4_ga2_2xa100/final
```

## Training Outcome

The run early-stopped at step 575, around epoch 0.54.

Final training/eval facts:

- final eval loss: 0.007503
- target pause argmax rate: 1.0
- continuation KL: about 0.0006 to 0.0008
- final model was moved from `/dev/shm` to `/workspace`
- final model was uploaded to R2 and verified by the watcher

R2 model/archive root:

```text
cloudflare_r2_cot_safety:cot-safety/stage2-stage3/20260706-2xa100-8b-full-stage2-cot5-mb4-ga2
```

## Stage2 Eval/Judge Completion

Eval root:

```text
/workspace/cot-safety/runs/eval/stage2_model_comparison_deepseek_8b_kl_transparent_emit_cot5_full_save25_mb4_ga2_2xa100
```

Eval data:

- capability: 800 rows
  - GSM8K: 500
  - MATH500: 300
- safety: 1150 rows
  - HarmBench standard: 200
  - JailbreakBench harmful: 100
  - StrongReject: 300
  - OR-Bench hard: 300
  - XSTest safe: 250

Generation completed for all 6 groups:

- base natural capability: 800
- base natural safety: 1150
- pause natural capability: 800
- pause natural safety: 1150
- pause forced capability: 800
- pause forced safety: 1150

Judge completed for all 9 safety outputs:

- WildGuard: base natural, pause natural, pause forced, all 1150/1150
- LlamaGuard: base natural, pause natural, pause forced, all 1150/1150
- HarmBench: base natural, pause natural, pause forced, all 1150/1150

Stage2 eval results were backed up to R2 and verified:

```text
cloudflare_r2_cot_safety:cot-safety/stage2-stage3/20260706-2xa100-8b-full-stage2-cot5-mb4-ga2/runs/eval/stage2_model_comparison_deepseek_8b_kl_transparent_emit_cot5_full_save25_mb4_ga2_2xa100
Total objects: 376
Total size: 385.209 MiB
rclone check: 0 differences found, 376 matching files
```

## Capability Results

| group | GSM8K accuracy | MATH500 accuracy | parse rate | notes |
|---|---:|---:|---:|---|
| base natural | 0.660 | 0.4467 | 1.0 | reference |
| pause natural | 0.646 | 0.4333 | 1.0 | small drop |
| pause forced | 0.652 | 0.4533 | 1.0 | close to base |

Executor interpretation:

- No broad capability collapse.
- MATH forced is slightly above base; this should not be claimed as an improvement.
- The intended Stage2 claim is behavior preservation, not safety or capability improvement.

## Safety Results

Unsafe-valid rates on unsafe reference prompts:

| judge | base natural | pause natural | pause forced |
|---|---:|---:|---:|
| HarmBench | 0.3600 | 0.3700 | 0.3567 |
| LlamaGuard | 0.5200 | 0.5267 | 0.5100 |
| WildGuard | 0.4233 | 0.3800 | 0.4133 |

Safe-reference aggregate behavior:

| judge | metric | base natural | pause natural | pause forced |
|---|---|---:|---:|---:|
| HarmBench | unsafe-valid on safe refs | 0.6382 | 0.6400 | 0.6273 |
| LlamaGuard | unsafe-valid on safe refs | 0.0782 | 0.0727 | 0.0855 |
| WildGuard | unsafe-valid on safe refs | 0.0491 | 0.0473 | 0.0564 |

Executor interpretation:

- Safety behavior is broadly close to base across all three judges.
- Any small reduction/increase should be treated as noisy unless later confidence intervals support it.
- This Stage2 run does not show an obvious unsafe-prompt safety drift or broken-output drift.

## Pause Emission Results

Natural pause emission:

| split | dataset | n | natural pause rate | exact single run of 3 | off-target pause rate | avg pause count | first pause index inside think |
|---|---|---:|---:|---:|---:|---:|---:|
| capability | GSM8K | 500 | 1.000 | 0.196 | 0.802 | 8.288 | 4.058 |
| capability | MATH500 | 300 | 1.000 | 0.9267 | 0.0033 | 3.377 | 5.840 |
| safety | HarmBench standard | 200 | 1.000 | 0.855 | 0.025 | 3.285 | 6.095 |
| safety | JailbreakBench | 100 | 1.000 | 0.920 | 0.010 | 3.170 | 6.070 |
| safety | OR-Bench hard | 300 | 1.000 | 0.7967 | 0.020 | 3.470 | 5.813 |
| safety | StrongReject | 300 | 1.000 | 0.890 | 0.0133 | 3.217 | 5.997 |
| safety | XSTest | 250 | 1.000 | 0.700 | 0.004 | 3.792 | 5.792 |

Forced pause evaluation:

- forced_or_full_pause_rate is 1.0 on all datasets by construction.
- avg inserted pause count is 3.0 on all datasets.
- capability and safety behavior under forced pause remain close to base.

Executor concern:

- Most safety datasets and MATH500 show near-target natural pause emission.
- GSM8K natural mode has serious pause over-emission: exact 3-pause only 19.6%,
  off-target pause rate 80.2%, and average pause count 8.288.
- This may be an emission-format problem specific to short arithmetic traces, not
  necessarily a blocker for Stage3 safety-source probing.
- However, because Stage2's clean goal is "insert pause without altering behavior",
  Fable should decide whether this is a blocker before Stage3.

## Recommended Decision Candidates

Conservative option:

- Do not start Stage3 yet.
- First inspect GSM8K over-emission examples and decide whether to adjust
  Stage2 training or decoding constraints.

Middle option:

- Proceed to Stage3 safety-source probing using both natural and forced pause.
- Treat forced pause as the aligned-position diagnostic, and natural pause as
  deployment realism.
- Add an explicit emission-quality filter/report per source.
- Do not make a global Stage2 clean-emission claim until GSM8K is fixed or
  explained.

Aggressive option:

- Proceed directly to full Stage3 with forced pause only.
- Risk: this may dodge the natural-emission defect and overstate Stage2 success.

Executor recommendation before Fable:

- Use the middle option if Fable agrees.
- For Stage3, run the newer Stage1 paired data and preserve Stage1 splits.
- Report natural and forced separately.
- Gate Stage4 on Stage3 evidence and a later liveness battery.

## Requested Fable Output

Please give a critical decision:

1. Is this Stage2 checkpoint good enough to proceed to Stage3?
2. Is GSM8K natural over-emission a blocker, a non-blocking limitation, or a
   sign that Stage2 should be rerun?
3. Should Stage3 use natural pause, forced pause, or both?
4. What minimal sanity checks should be done before spending more A100 time?
5. What claims are allowed and not allowed from these Stage2 results?
