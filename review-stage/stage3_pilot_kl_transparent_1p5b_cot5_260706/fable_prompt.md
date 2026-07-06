Please review this Stage3 pilot result and the relevant Stage3 code objectively.

Context:
- We are studying whether inserted pause tokens can serve as early intervention/readout points for unsafe CoT trajectories.
- Stage1 found useful early CoT signal around `cot_4`, so Stage2 inserts pause tokens after `cot_4` / before `cot_5`.
- Stage2 model used here is only a pilot KL-transparent 1.5B checkpoint, not a full Stage2 model.
- This Stage3 run is intended as a framework sanity check, not final paper evidence.

Four-stage goal:
1. Stage1: verify latent separability on original model trajectories.
2. Stage2: train a pause-token model that emits pause tokens with minimal behavior/capability shift.
3. Stage3: verify whether pause positions carry safe/unsafe trajectory signal.
4. Stage4: only after Stage3 succeeds, use pause positions as steering ports to reduce unsafe CoT without over-refusal or capability loss.

Relevant local files to inspect:
- `src/cot_safety/probes/stage3_evidence.py`
- `scripts/run_stage3_evidence_report.py`
- `legacy/PauseProbe/scripts/data/prepare_intra_pause_probe_data.py`
- `legacy/PauseProbe/scripts/probe/extract_hidden_states.py`
- `legacy/PauseProbe/scripts/probe/merge_hidden_shards.py`
- `legacy/PauseProbe/scripts/probe/run_intra_pause_probe_full.py`
- `legacy/PauseProbe/scripts/probe/run_position_scan_batched.py`
- `configs/experiment/stage3_intra_pause_probe_stage1_paired_harmbench_1p5b_cot5_2xa6000.yaml`
- `configs/experiment/stage3_intra_pause_probe_stage1_paired_reasoningshield_1p5b_cot5_2xa6000.yaml`
- `configs/experiment/stage3_intra_pause_probe_stage1_paired_strongreject_1p5b_cot5_2xa6000.yaml`
- `review-stage/stage3_pilot_kl_transparent_1p5b_cot5_260706/pilot_summary.md`
- `review-stage/stage3_pilot_kl_transparent_1p5b_cot5_260706/evidence/*.json`

Pilot results:
- HarmBench: pause AUROC 0.8120 vs prompt baseline 0.5000, margin +0.3120, pair-cluster CI [0.2688, 0.3550], true content control AUROC 0.8225, independent margin -0.0105.
- ReasoningShield: pause AUROC 0.7196 vs prompt baseline 0.5000, margin +0.2196, pair-cluster CI [0.1833, 0.2557], true content control AUROC 0.7263, independent margin -0.0185.
- StrongReject: pause AUROC 0.7347 vs prompt baseline 0.5000, margin +0.2347, pair-cluster CI [0.1982, 0.2723], true content control AUROC 0.7218, independent margin -0.0193.

Questions:
1. For a pilot Stage2 checkpoint, is this enough to say the Stage3 framework is functioning and pause states carry a real signal beyond prompt-only baselines?
2. Is the failure to beat true content control a blocker for continuing Stage2/full-SFT development, or only a blocker for making the stronger "pause-specific independent signal" claim?
3. Do you see any implementation or evaluation bugs that make these pilot numbers unreliable?
4. Before we run full Stage2, what minimal Stage3 changes would you require?
5. Given the four-stage goal, what should we do next?

Please be critical and concise. Separate blockers from non-blocking suggestions.
