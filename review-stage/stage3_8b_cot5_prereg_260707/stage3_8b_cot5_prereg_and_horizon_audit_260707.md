# Stage3 8B CoT5 Preregistration And Horizon Audit - 2026-07-07

This memo is the pre-extraction ruling for the 8B Stage3 run that uses the
KL-transparent Stage2 checkpoint with pauses inserted after `cot_4` / before
`cot_5`.

## Inputs

- Stage2 checkpoint:
  `/workspace/outputs/deepseek_8b_intra_pause_cot5_kl_transparent_emit_trusted_cot_18k_full_save25_mb4_ga2_2xa100/final`
- Stage1 paired freeze:
  `/workspace/cot-safety/runs/stage1_post_hb_260705_after_hb_n100_loso/loso_freeze_fixed_budget_samples_000_099/stage1_prepared/`
- Sources:
  `harmbench_standard`, `reasoningshield`, `strongreject_full`,
  `wildjailbreak_vanilla_harmful`
- Stage3 configs:
  `configs/experiment/stage3_intra_pause_probe_stage1_paired_*_8b_cot5_2xa100.yaml`

## Scope

This run is a teacher-forced/off-policy Stage3 readout on frozen Stage1 paired
data. It is supporting evidence for whether pause positions are readable
steering anchors.

It is not the on-policy confirmatory Stage3 result. It must not be used to
claim that pause states predict self-generated unsafe CoT trajectories until an
on-policy slice with per-generation CoT judge labels is run.

## Pause And Control Position Convention

The Stage2/Stage3 insertion convention is:

```text
<think> t0 t1 t2 t3 t4 <|pause|><|pause|><|pause|> t5 ...
```

`insert_cot_offset=5` means the three pause tokens are inserted immediately
before the original fifth content token, `t5`, after tokenizer-level alignment.

For a causal decoder hidden state at position `p`, the representation has seen
the token at `p` and predicts the next token. Therefore:

- `cot_4` / `control_cot_4` has seen original content through `t4`.
- `pause_0` has seen original content through `t4` plus one pause token.
- `pause_1` and `pause_2` have seen original content through `t4` plus two or
  three pause tokens.
- `control_cot_5` has seen original content through `t5`.
- `control_cot_6` has seen original content through `t6`.

To close the one-token horizon asymmetry, the 8B Stage3 configs now include
`control_cot_4` in addition to `control_cot_5` and `control_cot_6`.

The primary exact-horizon content control is `control_cot_4`. `control_cot_5`
and `control_cot_6` are lead/ordinary-content diagnostics and should not be
used as the only blocking comparator.

Dry-run verification:

```text
--insert_cot_offset 5
--positions ...,control_cot_4,control_cot_5,control_cot_6,...
--control_cot_offsets 4,5,6
```

## Primary Teacher-Forced Endpoint

For each source, report:

- best pause readout across `pause_0`, `pause_1`, `pause_2` and pause pooled
  variants, with layer and position fixed by validation only;
- prompt baselines: `last_prompt_token`, `pre_think`;
- exact-horizon content control: `control_cot_4`;
- lead content controls: `control_cot_5`, `control_cot_6`;
- pre/post diagnostics: `pre_pause_*`, `post_pause_*`;
- AUROC, balanced accuracy, recall/FPR, and paired/bootstrap CI where available.

## Success Rule For Teacher-Forced Supporting Evidence

This run supports a readable-anchor claim only if at least three of four sources
meet all of:

- pause AUROC is above prompt baselines;
- pause AUROC is at least `0.05` above `control_cot_4`;
- the pause-minus-`control_cot_4` CI excludes zero;
- no source shows a large data or position parsing failure.

If pause beats prompt baselines but does not beat `control_cot_4`, the result is
readable but not pause-specific. That is still useful for Stage4 liveness and
steering design, but it is not evidence of independent monitoring advantage.

## Explicit Non-Rescue Rules

- Do not promote `control_cot_5/6` failures or wins into the primary comparator
  after seeing results.
- Do not use forced insertion results to claim natural pause emission reliability.
- Do not claim on-policy trajectory monitoring from this teacher-forced run.
- Do not search additional layers, pooling recipes, or positions after seeing
  test metrics and report them as confirmatory.
- If the primary teacher-forced success rule fails, record the negative/limited
  result and move to on-policy/liveness only.

## On-Policy Confirmatory Gate

The decisive Stage3 gate remains an on-policy slice:

- use the 8B pause model to sample multiple CoTs per prompt;
- judge each generated CoT for unsafe content;
- extract pause hidden states from those generated trajectories;
- evaluate within-prompt separability and prompt/content controls.

Natural pause generation is the primary on-policy mode. Forced pause generation
is a control/diagnostic mode and cannot be promoted to primary after the fact.

On-policy success requires at least:

- mixed safe/unsafe generations for enough prompts;
- pause readout above prompt and content controls;
- `Delta AUROC >= 0.05`;
- CI excluding zero;
- at least three of four sources passing.

## Current Stage2 Sanity Gate

The Stage2 8B checkpoint passed behavior-preservation sanity:

- final eval loss: `0.007503`;
- pause target argmax rate: `1.0`;
- continuation KL: approximately `0.0006-0.0008`;
- no material capability/safety degradation in Stage2 model comparison;
- Stage3-source generation sanity: stripped pause bleed `0.0`;
- GSM8K over-emission did not reduce accuracy in the audited split.

This clears Stage2 for Stage3 investigation, not Stage3 claims.
