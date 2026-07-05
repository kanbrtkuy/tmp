# Stage2 Cot5 Position Update Review Brief

Date: 2026-07-06

## Scope

This packet asks for a focused code/design review of a Stage2 pause insertion
position change and its propagation into Stage3 and Stage4.

The desired pause layout is now:

```text
<think> t0 t1 t2 t3 t4 <|pause|><|pause|><|pause|> t5 ...
```

In this codebase, `cot_offset=k` means "insert pause tokens before CoT token
index k." Therefore the desired layout is implemented as:

```text
cot_offset: 5
insert_pause_after_cot_tokens: 5
```

## Reason For The Change

The Stage1-only Fable review concluded that if the Stage1 hidden@cot_4 signal
is used only as an exploratory engineering default, the most causally aligned
Stage2 insertion point is after `cot_4` / before `cot_5`, because hidden@cot_4
is a hidden state after the model has consumed `cot_4`.

Important claim boundary:

- This is not evidence that the position is optimal.
- This is not evidence that Stage1 validates a pause port.
- It is only a preregistered engineering default for the next Stage2 run.

## Main Code Changes To Review

Please inspect the full `cot-safety/` tree in this packet, especially:

- `configs/experiment/stage2_intra_pause_kl_transparent_emit_1p5b_cot5_save25_max400_4xa6000.yaml`
- `configs/experiment/stage2_intra_pause_kl_transparent_emit_8b_cot5_save50_max400_4xa100.yaml`
- `configs/experiment/stage2_model_comparison_eval_1p5b_kl_transparent_emit_cot5_4xa6000.yaml`
- `configs/experiment/stage2_model_comparison_eval_8b_kl_transparent_emit_cot5_4xa100.yaml`
- `configs/experiment/stage3_intra_pause_probe_kl_transparent_1p5b_cot5.yaml`
- `configs/experiment/stage3_intra_pause_probe_kl_transparent_8b_cot5_4xa100.yaml`
- `configs/experiment/stage4_pause_gprs.yaml`
- `configs/experiment/stage4_pause_gprs_8b_4xa100.yaml`
- `legacy/PauseProbe/scripts/probe/extract_hidden_states.py`
- `legacy/PauseProbe/scripts/probe/run_intra_pause_probe_full.py`
- `scripts/run_stage3_intra_pause_probe.py`
- `tests/test_pause_insertion.py`

## Intended Stage Flow After This Change

Stage2:

- Prepare trusted CoT SFT data with pause after cot_4 / before cot_5.
- Train KL-transparent emit model.
- Evaluate base vs natural pause model vs forced cot5 pause insertion.

Stage3:

- Generate/on-policy judge with the Stage2 cot5 model.
- Extract pause hidden states at the actual cot5 pause site.
- Include prompt baselines.
- Include true no-pause matched content controls at `control_cot_5` and
  `control_cot_6`, not the old hard-coded `control_cot_3/4`.
- Decide whether pause positions add signal beyond prompt baseline and content
  controls.

Stage4:

- Still gated behind liveness and Stage3 evidence.
- Forced pause injection should target the same after-cot4/before-cot5 site:
  `insert_pause_after_cot_tokens: 5`.
- GPRS steering should continue only if liveness and Stage3 gates pass.

## Local Verification Already Run

These checks passed locally:

```text
python3 -m py_compile scripts/run_stage2_sft.py scripts/run_stage3_intra_pause_probe.py scripts/run_stage3_evidence_report.py scripts/run_stage3_on_policy_confirmatory.py scripts/run_stage4_liveness.py scripts/run_stage4_steering.py legacy/PauseProbe/scripts/probe/extract_hidden_states.py legacy/PauseProbe/scripts/probe/run_intra_pause_probe_full.py src/cot_safety/formatting/pause_insertion.py

PYTHONPATH=src .venv-stage1-test/bin/python -m pytest tests/test_pause_insertion.py
PYTHONPATH=src .venv-stage1-test/bin/python -m pytest tests/test_stage3_evidence.py
PYTHONPATH=src .venv-stage1-test/bin/python -m pytest tests/test_stage3_on_policy_confirmatory.py
PYTHONPATH=src .venv-stage1-test/bin/python -m pytest tests/test_stage4_gprs_liveness.py
PYTHONPATH=src python3 scripts/smoke_test.py
```

Dry-runs also confirmed:

- Stage2 1.5B and 8B builders use `--cot_offset 5`.
- Stage2 model comparison forced conditions use
  `--insert_pause_after_cot_tokens 5`.
- Stage3 1.5B and 8B use `--insert_cot_offset 5`,
  `--cot_offsets 4,5,6,9,10`, and `--control_cot_offsets 5,6`.
- Stage4 GPRS 1.5B and 8B configs point to cot5 Stage3 evidence/probe paths.

## Questions For Fable

1. Is `cot_offset: 5` the correct implementation of the intended
   after-cot4/before-cot5 layout under this codebase's token semantics?
2. Did the cot5 change propagate consistently through Stage2 data prep,
   training, model-comparison eval, Stage3 extraction/probing, and Stage4
   liveness/GPRS configs?
3. Are the new Stage3 controls (`control_cot_5`, `control_cot_6`) the right
   matched pause-free controls for this insertion site?
4. Is there any remaining code path that could silently train/evaluate the old
   cot3/cot4 insertion while producing cot5-looking output names?
5. Any blocker before running the new Stage2 KL-transparent cot5 SFT?

