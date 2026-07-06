# Fable Review Request: Stage3 Patch Before Running Remaining Folds

Please review the current local repository code under:

`/Users/baby/Documents/SafeChain/cot-safety`

Context:

We have a four-stage project:

1. Stage1 checks whether early CoT hidden states have separability between safe vs unsafe CoT trajectories.
2. Stage2 trains a KL-transparent intra-CoT pause-token model. Pause is inserted after `cot_4` / before `cot_5`, and the goal is to make the model emit pause tokens without materially changing continuation behavior.
3. Stage3 checks whether hidden states at pause positions carry safe/unsafe trajectory signal. We do not require a very strong signal; the minimum useful claim is that pause hidden states contain a readable trajectory-safety signal beyond prompt-only baseline. A stronger claim is that pause/post-pause beats matched no-pause content controls.
4. Stage4 will only proceed after liveness checks confirm pause is a useful steering port.

Your previous review of the WJB Stage3 screen said:

- Patched eval sharding runner/configs were OK to run for remaining Stage1 paired folds.
- Stage3 adjudication tooling needed edits before treating fold status as final.
- Stage4 remains blocked until liveness/on-policy producer chain is implemented.
- The WJB result should be interpreted as "pause has signal, independent pause-specific advantage not established", not simply "pause has no signal".

I just made a new patch. Please review whether the patch is correct, whether it preserves the project logic, and whether it is OK to run the remaining Stage1 paired folds with it.

Files to inspect:

- `src/cot_safety/probes/stage3_evidence.py`
- `scripts/run_stage3_evidence_report.py`
- `legacy/PauseProbe/scripts/probe/merge_hidden_shards.py`
- `legacy/PauseProbe/scripts/probe/run_position_scan_batched.py`
- `legacy/PauseProbe/scripts/probe/run_intra_pause_probe_full.py`
- `legacy/PauseProbe/scripts/data/prepare_intra_pause_probe_data.py`
- `configs/experiment/stage3_intra_pause_probe.yaml`
- `configs/experiment/stage3_intra_pause_probe_stage1_paired_wjb_1p5b_cot5_2xa6000.yaml`
- `configs/experiment/stage3_intra_pause_probe_stage1_paired_harmbench_1p5b_cot5_2xa6000.yaml`
- `configs/experiment/stage3_intra_pause_probe_stage1_paired_reasoningshield_1p5b_cot5_2xa6000.yaml`
- `configs/experiment/stage3_intra_pause_probe_stage1_paired_strongreject_1p5b_cot5_2xa6000.yaml`
- `tests/test_stage3_evidence.py`

Patch intent:

1. Split Stage3 reporting into two levels:
   - `pause_signal`: basic question, whether pause hidden states carry a trajectory-safety signal beyond prompt-only baseline.
   - `independent_pause_signal`: stronger question, whether pause/post-pause beats prompt-only and matched no-pause content controls.
2. Fix leakage in `best_main`: choose pause/post-pause by validation AUROC, not test AUROC.
3. Add cluster bootstrap CI from per-example `predictions_test.jsonl`, defaulting `prediction_root` to the summary directory.
4. Add duplicate-position noise floor using `cot_4`/`pre_pause_1`, `cot_5`/`post_pause_1`, `cot_6`/`post_pause_2`.
5. Preserve optional row metadata (`source_families`, `risk_types`, `pair_ids`, `match_families`) during hidden shard merge.
6. Make batched linear probe training more deterministic/stable by zero initialization and higher epoch/patience defaults.
7. Make `run_intra_pause_probe_full.py` write config snapshots inside each run output rather than a shared parent.
8. Add preserve-splits row ID uniqueness checks.
9. Add configs for remaining Stage1 paired folds.
10. Set `probe.model_kinds` to `[linear]`, matching the actual batched scan implementation.

Local checks already run:

- `python3 -m py_compile` on edited Python files: passed.
- `PYTHONPATH=cot-safety/src cot-safety/.venv-stage1-test/bin/python -m pytest cot-safety/tests/test_stage3_evidence.py -q`: 4 passed.
- Dry-runs for harmbench/reasoningshield/strongreject configs show `preserve_input_splits`, `eval_shards=4`, and cuda:0/cuda:1 extraction assignment.

Please answer concretely:

1. Is this patch scientifically aligned with the four-stage goal?
2. Does the new `pause_signal` vs `independent_pause_signal` framing correctly reflect our Stage3 objective?
3. Are there any correctness bugs in the bootstrap, noise-floor, best-row selection, or hidden-shard metadata merge logic?
4. Is it OK to run remaining Stage1 paired folds with these configs now?
5. Should WJB be rerun under the zero-init/100-epoch probe settings before comparing across folds?
6. What must be fixed before Stage4, if anything?

Please prioritize blocking issues first, then non-blocking improvements.
