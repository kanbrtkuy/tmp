# Fable-5 Review Request: Stage1 Results and Stage2 Pause Placement

Please act as a strict senior ML methods reviewer. We need a narrow decision:

> Based on the current Stage1 evidence and the four-stage SafeChain plan, where
> should Stage2 insert pause tokens?

Read `STAGE1_STAGE2_EVIDENCE_SUMMARY.md` first. The packet is sanitized and
contains only aggregate information.

## Core Decision To Review

We currently lean toward:

```text
DeepSeek-8B Stage2 mainline: insert 3 pause tokens before cot_4.
DeepSeek-8B ablation: before cot_3.
Do not make cot_120 the mainline pause position.
```

But the user correctly pointed out a tension:

```text
If Stage1's strongest hidden readout often appears around cot_120, why are we
still proposing cot_3/cot_4 for Stage2?
```

We need your high-rigor answer.

## Questions

1. Is it methodologically correct to choose an early intervention position
   (`before cot_4`) even if the strongest Stage1 readout is often much later
   (`cot_120` or similar)?

2. Please distinguish clearly between:
   - best readout / forecasting location,
   - best pause-port training location,
   - best steering intervention location.

3. Given the latest post-HB Stage1 negative/control result against
   full-trajectory surface baselines, should Stage2 proceed at all? If yes,
   what should Stage2 claim and not claim?

4. For DeepSeek-8B specifically, should the main Stage2 run use:
   - `before cot_3`,
   - `before cot_4`,
   - a later fixed point such as `cot_16`, `cot_64`, or `cot_120`,
   - multiple/periodic pause blocks,
   - prompt/pre-think pause,
   - or something else?

5. Should we run a late-pause ablation before committing the main Stage2 GPU
   budget? If yes, what is the minimum credible ablation? If no, why not?

6. How should existing Stage2 evidence affect the decision?
   - cot4 format-only ckpt250 looked healthier than cot3 format-only.
   - cot3 format-only made unsafe-prompt unsafe_valid_rate worse than base.
   - cot3 full-SFT diagnostic improved some safety metrics but is not a clean
     format-only intervention.

7. Does the current evidence justify saying "cot4 is the main 8B pause-port
   candidate" or should this be weakened to "cot4 is the least-bad engineering
   default pending matched-horizon Stage1 / Stage3 confirmation"?

8. If Stage2 uses `before cot_4`, what exact gates must pass before Stage3 and
   Stage4 claims?

9. Please give a recommended first-run order, including:
   - any CPU-only checks before GPU,
   - the first Stage2 config/placement,
   - required neutral-behavior checks,
   - whether to run cot3/cot4/cot120 ablations,
   - and the go/no-go criteria for moving to Stage3.

10. What is the most scientifically honest answer to the user's question:
    "We saw strongest signal around cot120; why not put pauses there?"

## Desired Output

Please produce:

- Executive verdict.
- A position recommendation table.
- A short explanation of cot120 readout vs cot4 intervention.
- Required gates before Stage2, Stage3, and Stage4 claims.
- First-run order.
- "Do not claim" list.
- Final decision label:
  - `COT4_MAINLINE`,
  - `LATE_PAUSE_FIRST`,
  - `MULTI_POSITION_ABLATION_FIRST`,
  - `DO_NOT_PROCEED_TO_STAGE2`,
  - or another explicit label.

