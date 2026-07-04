from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class PipelineStep:
    name: str
    stage: str
    action: str
    command: list[str]
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_for_config(config: dict[str, Any]) -> list[PipelineStep]:
    """Return an auditable command plan for a stage or full-pipeline config.

    The first refactor keeps the execution plan explicit.  Deeper migration can
    replace these compatibility commands with native Python calls while keeping
    this public planning interface stable.
    """

    run_name = str(config.get("run", {}).get("name") or config.get("pipeline", {}).get("name") or "run")
    probe = config.get("probe", {})
    sft = config.get("sft", {})
    steering = config.get("steering", {})
    eval_cfg = config.get("eval", {})
    if config.get("pipeline", {}).get("stages"):
        return [
            PipelineStep(
                name="resolve_stage_configs",
                stage="pipeline",
                action="resolve",
                command=["cot-safety", "config", "show", "--config", "<stage-config>"],
                notes="Resolve each listed stage config before launch.",
            )
        ]

    if probe.get("task") == "positionscan_trajprobe":
        return [
            PipelineStep(
                name="run_stage1_positionscan",
                stage="stage1",
                action="probe_scan",
                command=[
                    "python",
                    "scripts/run_stage1_positionscan.py",
                    "--config",
                    "<config>",
                ],
                notes=(
                    "Runs data prep, hidden extraction, single-layer scan, and "
                    "multilayer ablations using model/runtime/probe settings from config."
                ),
            ),
        ]

    if sft:
        return [
            PipelineStep(
                name="build_intra_pause_sft_splits",
                stage="stage2",
                action="data_prepare",
                command=[
                    "python",
                    "scripts/run_stage2_sft.py",
                    "--config",
                    "<config>",
                    "--skip_train",
                ],
                notes="Build and validate cot-offset-specific intra-pause SFT splits from config.",
            ),
            PipelineStep(
                name="train_intra_pause_sft",
                stage="stage2",
                action="sft_train",
                command=[
                    "python",
                    "scripts/run_stage2_sft.py",
                    "--config",
                    "<config>",
                    "--skip_data_prep",
                ],
                notes="Launch DDP SFT using runtime batch, accumulation, and dataloader settings.",
            ),
        ]

    if config.get("hidden", {}) and probe:
        return [
            PipelineStep(
                name="prepare_intra_pause_probe_data",
                stage="stage3",
                action="data_prepare",
                command=[
                    "python",
                    "scripts/run_stage3_intra_pause_probe.py",
                    "--config",
                    "<config>",
                    "--skip_hidden_extraction",
                    "--skip_single_scan",
                    "--skip_pooled",
                ],
                notes="Rewrite high-quality trajectory rows into intra-pause format.",
            ),
            PipelineStep(
                name="extract_hidden_states",
                stage="stage3",
                action="hidden_extract",
                command=[
                    "python",
                    "scripts/run_stage3_intra_pause_probe.py",
                    "--config",
                    "<config>",
                    "--skip_base_data_prep",
                    "--skip_intra_data_prep",
                    "--skip_single_scan",
                    "--skip_pooled",
                ],
                notes="Use configured model, layers, positions, and runtime extraction jobs.",
            ),
            PipelineStep(
                name="run_intra_pause_probe",
                stage="stage3",
                action="probe_scan",
                command=[
                    "python",
                    "scripts/run_stage3_intra_pause_probe.py",
                    "--config",
                    "<config>",
                    "--skip_base_data_prep",
                    "--skip_intra_data_prep",
                    "--skip_hidden_extraction",
                ],
                notes="Compatibility entrypoint while native orchestration is ported.",
            ),
            PipelineStep(
                name="stage3_pause_vs_baselines_report",
                stage="stage3",
                action="evidence_report",
                command=[
                    "python",
                    "scripts/run_stage3_evidence_report.py",
                    "--config",
                    "<config>",
                ],
                notes=(
                    "Report whether pause/post-pause probe signal exceeds both prompt-only "
                    "baselines and true no-pause content controls. This is still a "
                    "teacher-forced screen, not the within-prompt on-policy confirmation."
                ),
            ),
        ]

    if steering:
        method = str(steering.get("method", "learned_delta"))
        liveness = config.get("liveness", {})
        steps = [
            PipelineStep(
                name="validate_pause_only_scope",
                stage="stage4",
                action="validate",
                command=["cot-safety", "steer", "validate-scope", "--config", "<config>"],
                notes="Reject any pre/post/cot steering target, including grouped target_specs.",
            )
        ]
        if liveness.get("enabled", method != "learned_delta"):
            steps.append(
                PipelineStep(
                    name="run_pause_liveness_battery",
                    stage="stage4",
                    action="liveness",
                    command=[
                        "python",
                        "scripts/run_stage4_liveness.py",
                        "--config",
                        "<config>",
                    ],
                    notes=(
                        "Gate steering on pause-port liveness: green => Stage3/GPRS; "
                        "yellow => live layers only + queue Stage2.5-A; "
                        "red => stop Stage4 and branch to Stage2.5-A/B."
                    ),
                )
            )
        if method in {"gprs", "projection"}:
            steps.extend(
                [
                    PipelineStep(
                        name="build_gprs_artifacts",
                        stage="stage4",
                        action="direction_build",
                        command=[
                            "python",
                            "scripts/build_stage4_gprs_artifacts.py",
                            "--config",
                            "<config>",
                            "--hidden_npz",
                            "<stage3-train-hidden-npz>",
                            "--probe_checkpoint_source",
                            "<stage3-probe-checkpoint>",
                        ],
                        notes=(
                            "Build mean-diff direction and safe-centroid artifacts from Stage3 hidden states, "
                            "then copy the selected Stage3 probe checkpoint used as the GPRS gate."
                        ),
                    ),
                    PipelineStep(
                        name="generate_and_judge_gprs",
                        stage="stage4",
                        action="eval",
                        command=[
                            "python",
                            "scripts/run_stage4_steering.py",
                            "--config",
                            "<config>",
                            "--phase",
                            "eval",
                        ],
                    notes=(
                        "Future step; runner is fail-closed until the GPRS generation hook exists. "
                        "Run GPRS only after liveness green, fixed Stage3, direction QC, and random-direction control."
                    ),
                    ),
                ]
            )
            return steps
        return [
            *steps,
            PipelineStep(
                name="train_learned_delta",
                stage="stage4",
                action="steer_train",
                command=[
                    "python",
                    "legacy/PauseProbe/scripts/steering/run_intra_pause_learned_delta_pilot.py",
                    "--config-backed-run",
                    run_name,
                ],
                notes="Legacy baseline/control only; not the primary method for kl_transparent_emit Stage2.",
            ),
            PipelineStep(
                name="generate_and_judge",
                stage="stage4",
                action="eval",
                command=[
                    "python",
                    "scripts/run_stage4_steering.py",
                    "--config",
                    "<config>",
                    "--phase",
                    "eval",
                ],
                notes=(
                    "Archival learned-delta reproduction only. Primary kl_transparent_emit "
                    "Stage4 should use GPRS after liveness and fixed Stage3 evidence."
                ),
            ),
        ]

    if eval_cfg.get("model_conditions"):
        return [
            PipelineStep(
                name="prepare_model_comparison_eval_data",
                stage="eval",
                action="data_prepare",
                command=[
                    "python",
                    "scripts/run_model_comparison_eval.py",
                    "--config",
                    "<config>",
                    "--phase",
                    "prepare",
                ],
                notes="Normalize configured benchmark sources into prompt-only capability and safety JSONL files.",
            ),
            PipelineStep(
                name="generate_model_comparison_outputs",
                stage="eval",
                action="generate",
                command=[
                    "python",
                    "scripts/run_model_comparison_eval.py",
                    "--config",
                    "<config>",
                    "--phase",
                    "generate",
                ],
                notes="Generate outputs for every configured model condition with condition-specific pause placement.",
            ),
            PipelineStep(
                name="judge_model_comparison_outputs",
                stage="eval",
                action="judge",
                command=[
                    "python",
                    "scripts/run_model_comparison_eval.py",
                    "--config",
                    "<config>",
                    "--phase",
                    "judge",
                ],
                notes="Run configured open judges over safety generations and normalize labels.",
            ),
            PipelineStep(
                name="summarize_model_comparison_eval",
                stage="eval",
                action="summarize",
                command=[
                    "python",
                    "scripts/run_model_comparison_eval.py",
                    "--config",
                    "<config>",
                    "--phase",
                    "summary",
                ],
                notes="Write capability and safety CSV summaries for configured model conditions.",
            ),
        ]

    return [
        PipelineStep(
            name="unknown",
            stage="unknown",
            action="inspect",
            command=["cot-safety", "config", "show", "--config", "<config>"],
            notes="No recognized stage keys found.",
        )
    ]
