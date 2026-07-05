# R1 Fix Notes For Fable Re-Review

Date: 2026-07-06

Fable R1 verdict was `PASS_WITH_REQUIRED_FIXES`.

## Fixes Applied

### RF-1 fixed

`cot-safety/configs/model/deepseek_r1_distill_qwen_1_5b.yaml` now pins:

```yaml
pause:
  cot_offset: 5
```

Verification:

```text
PYTHONPATH=src python3 -c "from cot_safety.config import load_config; cfg=load_config('configs/experiment/stage2_intra_pause_kl_transparent_emit_1p5b_cot5_save25_max400_4xa6000.yaml'); print(cfg['pause']['cot_offset'], cfg['sft']['cot_offset'])"
=> 5 5
```

### RF-2 fixed

`cot-safety/scripts/run_stage4_steering.py` now passes:

```text
INSERT_PAUSE_AFTER_COT_TOKENS
N_INSERT_PAUSES
MODEL_LABEL
```

`cot-safety/legacy/PauseProbe/scripts/steering/run_intra_pause_full_steering_eval.sh`
and `run_intra_pause_multiseed_generation_judge.sh` now use those env vars
instead of hard-coded cot3 values.

Verification:

```text
rg "insert_pause_after_cot_tokens 3|deepseek_intra_pause_cot3_sft" legacy/PauseProbe/scripts/steering scripts/run_stage4_steering.py configs/experiment/stage4_pause_gprs.yaml configs/experiment/stage4_pause_gprs_8b_4xa100.yaml
=> no matches

Stage4 build_env for 1.5B and 8B:
=> 5 3 deepseek_intra_pause_cot5_sft
```

### Packet completeness fixed

The review packet was regenerated with root-only data exclusions, so
`cot-safety/legacy/PauseProbe/scripts/data/` is now included.

## Checks Re-Run

```text
python3 -m py_compile scripts/run_stage4_steering.py
bash -n legacy/PauseProbe/scripts/steering/run_intra_pause_full_steering_eval.sh
bash -n legacy/PauseProbe/scripts/steering/run_intra_pause_multiseed_generation_judge.sh
PYTHONPATH=src python3 scripts/smoke_test.py
PYTHONPATH=src .venv-stage1-test/bin/python -m pytest tests/test_pause_insertion.py tests/test_stage3_evidence.py tests/test_stage3_on_policy_confirmatory.py tests/test_stage4_gprs_liveness.py
PYTHONPATH=src python3 scripts/run_stage4_steering.py --config configs/experiment/stage4_pause_gprs.yaml --phase validate --dry_run
```

All passed.

## Re-Review Request

Please review only whether RF-1 and RF-2 are now fixed and whether the prior
verdict can be upgraded to `PASS_TO_RUN_STAGE2`.

If you still see a blocker, give exact file/line references and the minimal
patch.

