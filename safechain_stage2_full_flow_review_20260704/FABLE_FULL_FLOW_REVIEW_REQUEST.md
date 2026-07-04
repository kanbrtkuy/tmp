# Fable Full-Flow Review Request: Latest SafeChain Stage2

Please perform a precise, complete review of the latest SafeChain Stage2 flow.
Do not optimize for brevity. The goal is correctness before GPU spending.

Read `README.md` first, then inspect the included code/config files.

## Project Goal For Stage2

Stage2 should train a model to make pause tokens available at intra-CoT cot3/cot4
positions while preserving the base model's behavior as much as possible. The
pause checkpoint must remain usable by Stage3 probes and later Stage4 steering.

The latest proposed method is:

```text
kl_transparent_emit:
  CE only on pause-token positions
  + KL-to-base continuation matching on pause-stripped teacher inputs
  + suppression of pause-token probability at non-target positions
```

This should replace ordinary full-response SFT as the clean Stage2 candidate.
Old full-SFT and format-only paths should remain available for baselines and
diagnostics.

## What To Review

Please review the complete flow:

1. Stage2 data prep.
2. Stage2 training runner/config plumbing.
3. Custom `PauseKLSFTTrainer` loss and indexing.
4. Shell/Hydra integration.
5. New 1.5B and 8B configs.
6. Stage2 model-comparison evaluation.
7. Whether the produced checkpoint/tokenizer still works for Stage3 without
   changing Stage3 code.

## Specific Questions

Please answer all of these in detail:

1. Is the full Stage2 flow internally consistent?
   - raw data -> prepared variants -> training -> checkpoint -> model eval
   - are paths/config inheritance/defaults coherent?

2. Does the new `kl_transparent_emit` branch preserve the existing data format?
   - No Stage1/Stage3 downstream format breakage?

3. Does `PauseKLSFTTrainer` implement the intended objective correctly?
   - pause-slot CE
   - continuation KL after pause stripping
   - pause suppression outside target pause slots
   - teacher/student prediction alignment
   - padding and label masking
   - DDP and gradient behavior

4. Are there hidden issues with using the same model as teacher under rows-only
   training?
   - Does the rows-only assertion really protect this?
   - Is teacher eval mode enough?
   - Any effect of tied embeddings in Qwen 1.5B?

5. Does Stage2 evaluation still answer the right questions?
   - Does current model-comparison eval still force-insert pauses?
   - Does it measure natural self-emission at all?
   - What exact changes are needed in eval before we can claim the model learned
     to self-emit pauses?
   - What metrics are missing for transparency, KL drift, and behavior drift?

6. Are the new configs good first GPU jobs?
   - 1.5B cot3 max400 on 4xA6000
   - 8B cot4 max400 on 4xA100
   - batch sizes, LR, max KL tokens, save/eval cadence, early stopping

7. Does this implementation remain compatible with Stage3?
   - checkpoint layout
   - tokenizer special token
   - cot3/cot4 insertion convention
   - hidden position naming / pause offsets

8. What minimum tests should be added before running real training?
   Please distinguish:
   - local unit tests
   - single-GPU tiny smoke
   - 4-GPU smoke
   - post-training invariants

9. What minimum eval should be run after the first trained checkpoint?
   Include:
   - natural pause emission
   - uncapped KL drift
   - behavior equivalence vs base
   - Stage3 probe readiness

10. Is the code acceptable as an experimental branch?
    If not, list blockers with exact file/function fixes.

## Desired Output

Please produce:

- Executive verdict.
- Blockers / must-fix before GPU.
- Correctness review of each stage.
- Loss/indexing proof or counterexample.
- Required code changes, if any.
- Required tests.
- Recommended first run order.
- What not to claim yet.
- A final go/no-go table for:
  - code review packet
  - 1.5B smoke
  - 1.5B full Stage2 pilot
  - 8B pilot
  - Stage3 handoff

Please be blunt and concrete.
