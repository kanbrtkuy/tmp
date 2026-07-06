# Fable Follow-up: WJB Clean Rerun

Date: 2026-07-06

Prompt summary:
- WJB was clean-rerun under the same Stage3 code and paired Stage1 prepared split.
- WJB result: pause AUROC 0.7983, prompt baseline 0.5002, true content control AUROC 0.7552, independent pause/post-pause margin -0.0077, pause-only margin over content control 0.0431 with CI [0.0320, 0.0538].
- Four-source table now includes HarmBench, ReasoningShield, StrongReject, and WildJailbreak.

Fable response:

> Ruling recorded in memory. Bottom line: WJB closes the sanity gap but doesn't change the verdict -- count the +0.043 as exploratory, fix the horizon/content-control audit first, then run the on-policy Stage3 slice; full Stage2 SFT is gated on that slice passing.
