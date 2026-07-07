# Fable Review: Stage2.1 Pause Chain Code Implementation

Date: 2026-07-07

Prompt:
`review-stage/stage2_1_clean_pause_design_260707/fable_prompt_stage2_1_code_review_260707.md`

Branch reviewed:
`https://github.com/kanbrtkuy/tmp/tree/stage21-pause-chain`

Commit reviewed:
`6a8dae3 Add Stage2.1 pause chain training framework`

## Raw Review

I couldn't delete `cot-safety/tmp_repro_location_metric.py` (rm blocked in this session) — please remove it, or keep it as the seed for the regression test in fix #1 below.

---

# Stage2.1 Pause Chain — Critical Review (fable-5 xhigh)

Reviewed on the local working tree at `cot-safety/` (matches the listed file set for `6a8dae3`). I read the trainer, insertion/relabel helpers, metrics, mining, mixer, builder, validator, diag, eval harness, both shell/orchestration layers, the hydra training entry, all four Stage2.1 configs, and all four test files. I also ran a live repro of one bug.

## Verdict: **almost** — Score: **7/10**

The core training objective is correctly implemented and the parts you were burned by before are right this time. But the metric you plan to select checkpoints with is broken for the *location* half of the paper target, and the DAgger loop has two unglued seams. Neither requires GPU time to fix.

## What I verified as correct

- Shift alignment in `pause_kl_trainer.py` is correct.
- Continuation KL pair alignment is correct.
- Rows-only training plus all-pause-id masking makes the self-teacher equivalent to a frozen non-pause teacher distribution.
- Stop-after-chain mask lands suppression on the first post-chain prediction.
- Suppression excludes pause-target positions.
- Multi-pause masking is consistent.
- Rows-only plumbing passes all three tokens into `format_only.trainable_tokens` and `special_tokens_to_add`.
- Stage2.1 configs disable eval-loss best-checkpoint selection / early stopping.
- Main path does not use decoding constraints and does not inject safety labels.

## Ranked Blockers

1. **High: `location_match` is wrong in real configurations.**
   It counts all tokens inside `<think>`, while formatter/builder skip leading whitespace tokens. `diag_stage2_checkpoint.py` also does not pass a tokenizer, falling back to word splitting. `run_model_comparison_generation.py` reuses `insert_pause_after_cot_tokens` as expected offset, which is wrong for natural mode.

2. **High: DAgger loop has unglued seams.**
   Mining drops task suffixes used during generation, especially hurting GSM8K/MATH. The mix output layout does not directly match `run_stage2_sft.py` expectations, and there is no iter1 config/driver.

3. **Medium: mixed dataset schema is not load-tested.**
   Static rows plus mined rows may fail pyarrow unification because of nested metadata, `sample_weight`, or non-string ids. Weighting is duplication-only; `sample_weight` itself is inert.

4. **Medium: emit margin does not enforce chain order.**
   Current margin compares against non-pause vocab only, so at target `p2`, rival pauses `p1/p3` are excluded. CE separates them, but margin should require the target pause to beat all rival pause tokens too.

5. **Medium: pause-head fallback is unsafe.**
   If enabled, training applies it but generation does not. It also breaks rows-only claims. Delete it or hard-fail when enabled.

6. **Low: eval harness natural mode is a trap.**
   Default `--insert_pause_after_cot_tokens 3` silently runs forced control. Add explicit `--generation_mode natural|forced` and decouple `--expected_cot_offset`.

7. **Low: `stage21_selection` config is decorative.**
   Plumb thresholds into diagnostics or users may accidentally use diag defaults.

8. **Low hygiene.**
   `--copy_val_test` cannot be disabled; mining lacks split-provenance assertion; `FORMAT_ONLY_TRAINABLE_TOKENS` missing from dry-run env print; hydra multi-token override should be validated at launch.

## Minimum Code Fixes Before GPU

1. Make location computation tokenize inside-think prefix and skip leading whitespace tokens; return `None` without tokenizer instead of word-count fallback; add `--tokenizer_path` to diagnostics; add independent `--expected_cot_offset` to eval harness.
2. Preserve exact conditioned prompt, including task suffix, in mined rows.
3. Change emit margin competitor to all vocab except the target token.
4. Make mixer write the layout `run_stage2_sft.py` expects, or add layout mode; add a DAgger iter1 config.
5. Delete or hard-fail `pause_head.enabled=true`.

## Minimum Tests Before GPU

1. Regression test for leading-whitespace tokenizer location convention.
2. `_pause_stripped_batch` and `_select_kl_pairs` with distinct ids.
3. End-to-end mini: static dir + mined jsonl -> mix -> dataset load/formatting.
4. Test that after `p1`, margin loss requires `p2` to beat `p1`.
5. Real `PauseKLSFTTrainer.__init__` test with a stub tokenizer.

## Recommended First Run Sequence

1. Data prep + validators only with the 1.5B Stage2.1 config.
2. 25-step smoke run.
3. Full 400-step 1.5B pilot; evaluate natural generation and diagnose with tokenizer-aware metrics.
4. Only if exact-chain trajectory beats the old repeated-token run: mine from the best train-side checkpoint, mix one DAgger round, then run 8B short400.

## Claims Before Results

Allowed: implementation-level claim that Stage2.1 is a distinct-chain,
rows-only pause objective with margin emit/suppression, stop-after-chain, and
continuation KL to a frozen non-pause teacher distribution.

Disallowed: claims that over-emission is fixed, boundary-location claims before
the location metric fix, DAgger robustness claims before a mined round trains,
or using 1.5B pilot as proof that the 8B cot_4/cot_5 port is valid.

Bottom line: trainer gradients are mostly trustworthy; fix measurement and the
DAgger seam before spending GPU.
