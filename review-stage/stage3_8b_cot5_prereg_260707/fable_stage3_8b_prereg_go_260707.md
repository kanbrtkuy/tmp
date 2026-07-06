# Fable Review: Stage3 8B CoT5 Prereg GO

Date: 2026-07-07

## Verdict

**GO** for the formal 8B teacher-forced Stage3 run as supporting evidence only.

Fable confirmed that the preregistration and horizon audit resolve the immediate
pre-extraction blockers:

- `control_cot_4` closes the one-token horizon asymmetry and is the primary
  comparator.
- `control_cot_5` and `control_cot_6` remain lead/ordinary-content diagnostics.
- The success rule is fixed before extraction:
  pause above prompt baselines, pause minus `control_cot_4 >= 0.05`, CI excluding
  zero, and at least three of four sources passing.
- Non-rescue rules are explicit.
- The scope is limited to teacher-forced supporting evidence.
- On-policy Stage3 remains the confirmatory gate.

## Required Lines To Hold

1. No layer/position search after test metrics; selection must be validation-fixed.
2. Human QA was bypassed, so paper language is limited to automated hash/near-dup
   checks and must not claim human-verified data quality.
3. If pause beats prompt but not `control_cot_4`, report "readable but not
   pause-specific" without softening.
4. On-policy slice, with natural primary and forced control, remains the
   confirmatory gate. Teacher-forced results cannot upgrade themselves.
