# Stage1 Construct-Validity Prompt-Baseline Review Packet

Date: 2026-07-05

Reviewer target: `claude-fable-5`

This packet contains only aggregate statistics and document references. It
contains no raw prompts, raw CoTs, or row-level examples.

## Question

A professor added a construct-validity concern:

> Is the `token_3` signal about reasoning, or about the prompt? In a
> teacher-forced setup the prompt fully determines those early tokens, so a
> probe on the last pre-`<think>` position might already hit about 0.97. If it
> does, the result is mostly prompt classification and the trajectory-monitoring
> framing weakens. Add a prompt-only / pre-CoT probe baseline and show the
> trajectory adds signal beyond it.

We need Fable-5 to judge:

1. Which parts of this concern are already solved by existing evidence?
2. Which parts remain unsolved or only partially solved?
3. Does this prompt-baseline evidence change the previous `STOP_CURRENT_STAGE1`
   verdict for the frozen Stage1 test set?
4. What is the safest paper wording?

## Existing Stage1 Closure Boundary

Previous Fable-5 verdict:

- `FRESH_PREREG_ONLY`
- `STOP_CURRENT_STAGE1 for the frozen test set`
- No improvement-seeking code or runs on the current frozen Stage1 set.

Allowed work on current frozen Stage1:

- operating-point reporting;
- integrity/audit completion for the negative claim;
- sensitivity/power statement;
- row-audit coverage-gap documentation;
- write-up.

Forbidden work on current frozen Stage1:

- new probe families, layers, positions, classifier heads, pooling schemes;
- calibration rescues;
- subgroup or per-k slicing to find a positive cell;
- re-splitting or reweighting existing pairs;
- GPU regeneration for current Stage1 rescue.

## Prompt-Baseline Evidence

### Natural Same-Prompt Prompt-Only / Pre-CoT Controls

The strongest evidence for this professor concern is the later natural-pair
setting, because safe and unsafe arms share the same prompt.

Aggregate prompt baseline results from `res/stage1_natural_pair_experiment_results_260703_zh.md`:

| Setting | Baseline / Control | AUROC |
|---|---|---:|
| Natural 8B generated/generated | prompt-only TF-IDF | 0.5000 |
| Natural 8B generated-safe/original-unsafe | prompt-only TF-IDF | 0.5000 |
| Natural 8B generated/generated, R1-8B extractor | best prompt/pre-CoT hidden | 0.5000 |
| Natural 32B generated/generated, R1-8B extractor | best prompt/pre-CoT hidden | 0.5000 |
| Natural 32B generated/generated, R1-32B extractor | best prompt/pre-CoT hidden | about 0.51 |

CoT-position hidden probes in those natural-pair runs still show signal:

| Setting | CoT-position result | AUROC |
|---|---|---:|
| Natural 8B generated/generated Stage1b, R1-1.5B extractor | `cot_4 / layer18` test-max | 0.7328 |
| Natural 8B generated/generated Stage1b dense, R1-8B extractor | `cot_16 / layer14` test-max | 0.7632 |
| Natural 32B generated/generated Stage1b dense, R1-8B extractor | `cot_9 / layer8` test-max | 0.8044 |
| Natural 32B generated/generated Stage1b dense, R1-32B extractor | `cot_4 / layer52` test-max | 0.8148 |
| Natural 32B generated/generated Stage1b dense, R1-32B extractor | `cot_9 / layer32` validation-selected test | 0.7890 |

Executor's proposed interpretation:

- The specific failure mode "prompt-only / pre-CoT already reaches about 0.97
  AUROC" is not observed in the setting that directly controls prompt.
- Same-prompt natural-pair runs strongly reduce the pure prompt-classification
  concern.
- They do not eliminate CoT-internal surface/style/length/refusal-pattern
  confounds.

### Earlier Stage1b Prompt-Only / Pre-CoT Controls

Stage1b implemented prompt-only and pre-CoT positions as first-class scan
positions:

- `last_prompt_token`: strict prompt-only baseline.
- `pre_think`: boundary immediately before `<think>`.
- `think_last`: stronger pre-CoT control, not a strict prompt-only baseline.

Aggregate heldout mean AUROC from 2026-06-30 Stage1b:

| Model | Position | Heldout mean AUROC |
|---|---:|---:|
| 1.5B | `last_prompt_token` | 0.785 |
| 1.5B | `pre_think` | 0.830 |
| 1.5B | `think_last` | 0.833 |
| 1.5B | `cot_3` | 0.868 |
| 1.5B | `cot_4` | 0.891 |
| 8B | `last_prompt_token` | 0.801 |
| 8B | `pre_think` | 0.843 |
| 8B | `think_last` | 0.859 |
| 8B | `cot_3` | 0.912 |
| 8B | `cot_4` | 0.921 |

Secondary interpretation:

- In older source-family Stage1b, strict prompt-only does contain nontrivial
  signal, but it is still below early-CoT heldout signal.
- Early-CoT positions sharpen that signal:
  - 1.5B: `cot_4` improves over strict prompt-only by +0.106, and over stronger
    pre-CoT controls by about +0.058 to +0.061.
  - 8B: `cot_4` improves over strict prompt-only by +0.120, and over stronger
    pre-CoT controls by about +0.062 to +0.078.
- This is useful historical support, but the natural same-prompt runs should be
  the primary response to the professor's prompt-classification concern.

## Other Stage1 Context

Current post-HB Stage1 remains negative for the strong claim:

- Hidden probes are above chance on held-out sources, often around 0.7 to 0.84.
- Hidden does not beat full/matched surface text baselines or length controls.
- Equal-horizon and excluded-source lead-time follow-ups did not confirm a
  stable hidden advantage.

Therefore, the executor proposes this wording:

> In natural same-prompt pairs, prompt-only text and prompt/pre-CoT hidden
> controls are near random, while CoT-position hidden probes retain above-chance
> signal. Thus the early-CoT signal is not explained by prompt classification
> alone. However, Stage1 does not demonstrate hidden-state superiority over
> surface text or length baselines; the result should be framed as above-chance
> decodable signal, not clean latent safety semantics or a stable
> trajectory-monitoring advantage.

## Source Documents

Local authoritative documents:

- `cot-safety/res/professor_feedback_status_260705_zh.md`
- `cot-safety/res/stage1_stage1b_prompt_baseline_summary_20260630_zh.md`
- `cot-safety/res/stage1_natural_pair_experiment_results_260703_zh.md`
- `cot-safety/review-stage/stage1_auto_improve_260705/fable5_fresh_path_review_260705.md`
- `cot-safety/review-stage/stage1_auto_improve_260705/REVIEW_STATE.json`
