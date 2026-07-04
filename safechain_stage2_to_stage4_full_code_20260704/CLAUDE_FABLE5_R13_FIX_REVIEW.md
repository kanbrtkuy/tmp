# Fable R13-H1 Fix Review (Round 13.1)

- **Scope:** targeted follow-up to the Round 13 four-stage alignment review (`CLAUDE_FABLE5_FOUR_STAGE_ALIGNMENT_REVIEW.md`). HEAD is `7dbc421`, which adds the R13 review doc **and** the R13-H1 fix to `tests/test_safe_rewrite.py` (13 lines changed in the test, no production code touched — verified via `git show HEAD --stat`; working tree clean). No code edited in this round.
- **Method:** full diff read; full read of `tests/test_safe_rewrite.py` and the relevant `safe_rewrite.py` paths (`word_count`, `length_target_for_unsafe`, `base_pair_record`, `_length_match_pass`, `update_pair_record_with_generated_safe`, `merge_generated_pair`); independent numeric re-derivation of the word counts and target band; test execution with the same R13 pytest shim (`/tmp/fable_r13_pytest/`, `pytorchenv` — numpy + torch 2.3.1, no transformers).

## Test run (this machine)

| Target | Result |
|---|---|
| `tests/test_safe_rewrite.py` | **11 passed, 0 failed** (incl. the fixed `test_merge_generated_pair_and_long_rows`) |
| Stage3/4: `test_stage3_evidence.py`, `test_stage3_on_policy_confirmatory.py`, `test_stage4_gprs_liveness.py`, `test_steering_scope.py` | **24 passed, 0 failed** (torch projection tests included) |
| Full suite, all 13 `tests/test_*.py` | **57 passed, 0 failed, 0 skipped** (R13: 56 passed, 1 failed — count consistent: the one red test is now green, none deleted or skipped) |

## 1. Is R13-H1 closed? — YES

The fix takes R13's option A (fix the test fixture; keep the floor hard) — the right call, since a config-driven floor would have weakened a quality gate to save a test.

Verified trace (re-derived numerically against the code, not just by test outcome):

- `unsafe_reasoning` = 40 whitespace words → `length_target_for_unsafe` (safe_rewrite.py:382-429) with min_ratio 0.75 / max_ratio 1.10 / abs bounds [20, 60] → band **[30, 44]** (pure ratio band: 30 = ⌊40·0.75⌋ > 20, 44 = ⌈40·1.1⌉ < 60).
- `safe_reasoning` = 22 whitespace words (`word_count` = `clean_text().split()`, safe_rewrite.py:865-866) → **22 ≥ 20** clears `MIN_GENERATED_SAFE_REASONING_WORDS` (:30, :726-729), no `ValueError`.
- **22 ∉ [30, 44]** → `_length_match_pass` (:703-712, fed via `base_pair_record`'s `length_target` copy at :691) → `length_match_pass: False`, `ok: True` — exactly the asserted outcome.

The R13 red (`ValueError: generated safe reasoning is empty or too short`) is gone; the test passed here and the full suite is green. **R12-F1's "no known red in main" precondition is now discharged** — the formal land ritual still means a pod run with real pytest, but there is no longer a known blocker to it.

## 2. Does the test still cover the intended behavior? — YES, and coverage improved

The intent was always the *soft* path: a generated rewrite that is accepted (`ok: True`) but flagged for the length-repair loop (`length_match_pass: False`), plus the merge/long-rows plumbing. That is preserved — and the fixture is now strictly better than the pre-R13 one:

- **Old fixture** (2-word unsafe, 3-word safe): band collapsed to the degenerate absolute-min clamp **[20, 20]**, so `length_match_pass=False` was driven by the clamp, never by the ratio logic.
- **New fixture** (40-word unsafe, 22-word safe): band **[30, 44]** comes from the ratio arms of `length_target_for_unsafe` — the test now exercises the real length-matching computation for the first time.
- Margins are robust: +2 words above the floor, −8 below the band — a small change in tokenization semantics can't silently flip either side.
- All other assertions (unsafe trajectory passthrough, `<think>` rendering, style-profile determinism/propagation, `pair_record_to_long_rows` variant/label ordering) are unchanged and still meaningful; the 22-word safe text is also a realistic high-level rewrite rather than filler.

## 3. New issues introduced? — NONE

- Diff touches only the one test function; the other 10 safe_rewrite tests still use their own fixtures (their short `bad_cot` strings never reach the generated-safe floor, which applies only in `update_pair_record_with_generated_safe`).
- Production `safe_rewrite.py` is byte-identical to the R13-reviewed tree; Stage2→4 gate path untouched (the R13 statement that safe-rewrite is off that path stands).
- One **pre-existing** (not introduced) gap, minor: the hard floor's rejection path is untested — no test anywhere asserts `pytest.raises(ValueError)` for a <20-word generated rewrite (grep: no test references `MIN_GENERATED_SAFE_REASONING_WORDS` or the "too short" message). The floor was only ever "tested" by accidentally breaking this test. If someone later deletes or loosens the floor, the suite stays green. → **R13.1-L1 (low): add a below-floor raises test.** Not a blocker for anything.

## 4. Headline verdict — UNCHANGED, with R13-H1 struck

Go/No-Go deltas vs R13:

| Action | R13 | Now |
|---|---|---|
| Run Stage2 1.5B `kl_transparent_emit` train + eval | GO | **GO** (unchanged) |
| Claim "full suite green" / land ritual (R12-F1) | NO-GO until R13-H1 fixed | **GO** — 57/57 green here via shim; confirm once with real pod pytest as the formal ritual |
| Stage3 screen / Stage3 confirmatory / Stage4 gate-GPRS | GO / NO-GO / NO-GO | unchanged — this fix touches none of it |

Remaining required follow-ups from R13, all still open and unaffected: **D1/D2** (stale plan text: UltraFeedback→trusted-CoT-18k, full-SFT→kl_transparent_emit, StrongReject/WildJailbreak→curated sources), **B1** (Stage3 on-policy producer chain — still the critical path), B2/B3/B4, F4/F5, R13-M2 — plus new low-priority **R13.1-L1** above.

## Headline verdict

**PASS to run Stage2 with required follow-ups.**

R13-H1 is cleanly closed — the test now clears the hard quality floor while genuinely exercising the ratio-derived length-match target it was always meant to cover, the full suite is green on this machine for the first time (57/0/0), and no production code moved. The headline stays where R13 left it because the remaining follow-ups were never about this test: update the stale plan text (D1/D2) alongside the Stage2 run, run pod pytest once to formalize the now-unblocked land ritual, and keep the Stage3 on-policy producer chain (B1) as the critical path to everything past the teacher-forced screen.
