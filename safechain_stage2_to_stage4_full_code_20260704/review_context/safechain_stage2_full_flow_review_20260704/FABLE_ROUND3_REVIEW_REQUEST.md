# Fable Round 3 Review Request: Verify Remaining Stage2 Blockers Are Closed

Please do a focused third review of the current packet.

Read:

1. `README.md`
2. `CLAUDE_FABLE5_ROUND2_REVIEW.md`
3. The two files changed after Round 2:
   - `cot-safety/configs/experiment/stage2_intra_pause_kl_transparent_emit_1p5b_cot3_save25_max400_4xa6000.yaml`
   - `cot-safety/legacy/PauseProbe/scripts/eval/run_model_comparison_generation.py`

Round 2 found:

- NEW-B1: 1.5B config invalid because save/eval steps were incompatible with
  `load_best_model_at_end: true`.
- NEW-B2: vLLM default detokenization would strip `<|pause|>`, making natural
  self-emission metrics falsely zero.

Current fixes:

- NEW-B1: keep `save_steps: 25`, `eval_steps: 50`, but set
  `load_best_model_at_end: false` and `early_stopping.enabled: false`.
- NEW-B2: set vLLM `SamplingParams(skip_special_tokens=False,
  spaces_between_special_tokens=False)` and strip terminal EOS text before
  metrics/judging.

Please answer:

1. Are NEW-B1 and NEW-B2 fixed?
2. Did either fix introduce a new blocker?
3. Is the packet now acceptable for the next concrete step: running the
   pytest suite on the pod, then the 1.5B single-GPU smoke?
4. What remains claim-blocking but not code-blocking?

End with a concise final go/no-go table.
