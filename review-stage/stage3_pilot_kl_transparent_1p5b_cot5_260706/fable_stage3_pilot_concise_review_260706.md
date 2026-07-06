Reviewed against the project record (equal-horizon thread closed 2026-07-05; 1.5B pause-position ruling 2026-07-06). Verdict and specifics below.

## Verdict

As a **framework sanity check**: yes, adequate. Probes train, splits preserved, pause states are readable (0.72-0.81), no crash in the pipeline. As **evidence for anything**: weak, and two of your three headline comparisons are not citable.

## Blockers (for the pause-specific independent-signal claim, not for Stage2 SFT)

1. **Independent margin is negative on 3/3 sources.** The 2026-07-06 ruling set the 1.5B position-claim gate at pause beats base-hidden@matched-position AND matched-horizon surface, Delta AUROC >= 0.05, CI excl 0, on >=3/4 sources. You're at -0.01 to -0.02. This gate is currently *failed*, not pending. No independent-signal claim.
2. **The vs-prompt margins are inert.** Prompt-only = 0.5000 exactly is by-construction in the same-prompt paired design (identical prefixes under teacher forcing) -- hygiene check only, per the 2026-07-03 ruling. The CI margins [0.27, 0.36] etc. are margins over a guaranteed-chance baseline. Never headline them.
3. **This is not the on-policy Stage3 read.** Stage3 on Stage1 *prepared* data is teacher-forced, off-policy text -- DPI framing applies: pause states are deterministic functions of the preceding text, so "pause reads out content" is expected, not a finding. The whole point of Stage3 in the plan was on-policy validation (unsafe-side on-policy was Stage1's known gap). This pilot doesn't touch that.

## Not a blocker

Failure to beat content control does **not** block full Stage2 SFT. The project motivation already pivoted (post-A2) to **causal steering at pause states**, not monitoring advantage. For Stage4 steering you need pause states to *carry* the signal (0.72-0.81 does), not to carry *extra* signal beyond content. Continue -- but with the claim scoped as "pause positions are convenient, readable steering anchors," nothing stronger.

## Risks

- **Missing CI on the independent margin.** -0.0105 without a paired bootstrap CI is uninterpretable (parity vs. deficit). Given Stage1 A2 CIs were ~+/-0.01 at n~600, yours may straddle 0 -- but that's parity at best, still no positive claim.
- **Horizon fairness of the content control.** The off-by-one audit (hidden@cot_4 <-> insertion before cot_5) is still open. If content control sees cot_5 tokens the pause never sees, -0.01 is a *lead-time* result -- but lead-time is ruled exploratory-only and recipe-sensitive. If horizons are matched, it's a clean null. Pin down which one this is before reporting.
- **WJB missing** = 3/4 sources, below the >=3/4-with-positive-Delta bar anyway, but also your training-mass family -- file-drawer optics if the clean batch ships without it.
- **1.5B S2 gates.** Position is unvalidated at 1.5B (8B evidence only). Did the pilot Stage2 pass its own S2-G1..G5, tokenizer/position-convention audit, and coverage curves? If not, this pilot sits on an unaudited insertion.
- **No matched-horizon surface baseline** in the comparison set -- the claim gate names it explicitly.

## Minimal checks before full Stage2/Stage3

1. Paired bootstrap CI on pause-content margin, per source.
2. Resolve the horizon question: does content control condition on tokens the pause position cannot see? Document one way or the other.
3. Clean WJB rerun (complete the 4-source batch).
4. Confirm/backfill 1.5B S2 gates + Qwen tokenizer/position audit + pause coverage curves.
5. Add the matched-horizon surface (char TF-IDF) arm.
6. Small **on-policy** Stage3 slice (~100-200 judge-filtered rollout pairs from the pause model, eval-only) -- this is the decisive experiment, and cheap.

## Next step

Proceed to full Stage2 SFT (steering-scoped), but pre-register the on-policy Stage3 confirmatory *before* running it, with the independent-margin gate stated up front and an accepted-negative branch -- same discipline as the A2 stop. Do not spend GPU expanding off-policy Stage3 evals; they can only re-confirm DPI.

Want me to draft the on-policy Stage3 prereg memo?
