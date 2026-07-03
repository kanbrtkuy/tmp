# Public Review Manifest

This directory is a curated public review packet for the SafeChain Stage 1
natural-pair experiments.

## Included

- `README.md`
  - high-level disclosure boundary and review goal.
- `fable_review_prompt.md`
  - aggregate-only prompt for external review.
- `plan/`
  - Stage 1 natural-pair experiment plan in English and Chinese.
- `res/`
  - aggregate Stage 1 natural-pair result summaries in English and Chinese.
- `analysis_reports/`
  - previous aggregate-only Fable5 review note.
- `scripts/`
  - selected Stage 1 data export, provenance, surface-baseline, and hidden-probe
    runner code.
- `configs/`
  - selected experiment/data/model configs needed to understand the reported
    runs.
- `tests/`
  - selected unit tests for Stage 1 export/provenance/surface-baseline helpers.

## Excluded

The packet intentionally excludes:

- raw prompts,
- raw chain-of-thought trajectories,
- raw unsafe trajectories,
- generated model outputs,
- JSONL run data,
- hidden-state arrays,
- model checkpoints,
- credentials,
- API keys/tokens,
- Cloudflare or RunPod secrets,
- large run artifacts.

## Intended Use

Use `fable_review_prompt.md` as the main review prompt. The copied code and
configs are included only so the reviewer can inspect whether the pipeline and
reported controls are conceptually adequate. They are not sufficient to
reproduce the experiments without the private datasets and run artifacts.
