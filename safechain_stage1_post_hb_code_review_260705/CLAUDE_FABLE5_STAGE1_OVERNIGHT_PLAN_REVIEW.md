# Claude Fable 5 Review - Stage 1 Overnight Execution Plan

Date: 2026-07-05
Scope: content-quiet review of the Stage1 overnight execution plan, post-HB
orchestrator, frozen-fold launcher, exporter, configs, and exporter tests.

## Verdict

Safe to let the RunPod watcher continue after HB completes.

No hard blockers were found.

## Methodology Alignment

Fable found that the plan matches the intended methodology:

- post-HB raw snapshot runs before fixed-budget selection
- primary fixed-budget policy is first-N with `N=100`
- later HB-only samples are diagnostic-only and do not enter primary LOSO
- ReasoningShield extra pair file is explicit through `EXTRA_LOSO_PAIR_JSONL`
- source floors are checked both before freeze and after freeze via
  `keep_pairs_by_source`
- HT remains external/quarantine-only and does not enter LOSO
- formal LOSO still requires completed passing human QA

## Provisional Overnight Bypass

Fable judged `ALLOW_UNREVIEWED_GPU_STAGE1=1` acceptable for an overnight
debug-only run because:

- it writes `human_qa_gate_bypassed.json`
- the file explicitly marks the gate status as
  `bypassed_for_unreviewed_provisional_gpu_run`
- the warning says outputs are provisional/debug-only until human QA passes
- the bypass does not create a passing `stage1_human_qa_summary.json`

Fable emphasized that final claims must exclude provisional outputs until human
QA passes.

## Frozen-Fold Launcher

Fable found that `pipelines/run_stage1_sequence.sh` correctly:

- consumes `${STAGE1_FREEZE_DIR}/folds/<source>/normalized`
- exports each fold to `cotpause/{train,val,test}.json`
- generates fold-specific Stage1 and Stage1b configs
- sets `data.prepared_data_dir` per fold
- runs Stage1/Stage1b per frozen fold rather than using an old data directory

## Verification Items Before Final Claims

- Check `stage1_loso_freeze_summary.json`, especially
  `keep_pairs_by_source`, after freeze.
- Confirm `EXTRA_LOSO_PAIR_JSONL` is set and exists for four-source LOSO.
- Check `hb_raw_snapshot_summary.json` after the raw snapshot step.
- Complete human QA and require a passing summary before formal LOSO claims.
- Optionally add one more provisional marker into GPU run metadata/archive
  READMEs.
