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
                    "legacy/COTPauseToken/scripts/data_generation/pause_sft/build_intra_think_pause_sft_splits.py",
                    "<raw-jsonl>",
                    "<tokenizer>",
                    "<output-root>",
                ],
                notes="Pause placement and split sizes come from config.",
            ),
            PipelineStep(
                name="train_intra_pause_sft",
                stage="stage2",
                action="sft_train",
                command=[
                    "bash",
                    "legacy/COTPauseToken/scripts/training/run_4gpu_intra_pause_sft.sh",
                    "<data-dir>",
                    "<output-dir>",
                    "<model-path>",
                    run_name,
                    "1",
                ],
                notes="Use runtime config to choose 4xA100 vs smaller launch settings.",
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
                    "legacy/PauseProbe/scripts/data/prepare_intra_pause_probe_data.py",
                    "--config-backed-run",
                    run_name,
                ],
                notes="Rewrite high-quality trajectory rows into intra-pause format.",
            ),
            PipelineStep(
                name="extract_hidden_states",
                stage="stage3",
                action="hidden_extract",
                command=[
                    "python",
                    "legacy/PauseProbe/scripts/probe/extract_hidden_states.py",
                    "--config-backed-run",
                    run_name,
                ],
                notes="Use configured model, layers, positions, and runtime extraction jobs.",
            ),
            PipelineStep(
                name="run_intra_pause_probe",
                stage="stage3",
                action="probe_scan",
                command=[
                    "python",
                    "legacy/PauseProbe/scripts/probe/run_intra_pause_probe_full.py",
                    "--config-backed-run",
                    run_name,
                ],
                notes="Compatibility entrypoint while native orchestration is ported.",
            ),
        ]

    if steering:
        return [
            PipelineStep(
                name="validate_pause_only_scope",
                stage="stage4",
                action="validate",
                command=["cot-safety", "steer", "validate-scope", "--config", "<config>"],
                notes="Reject any pre/post/cot steering target.",
            ),
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
                notes="Learn delta at pause_0/pause_1/pause_2 only.",
            ),
            PipelineStep(
                name="generate_and_judge",
                stage="stage4",
                action="eval",
                command=[
                    "bash",
                    "legacy/PauseProbe/scripts/steering/run_intra_pause_full_steering_eval.sh",
                ],
                notes="Evaluate base, SFT, and SFT+steering with configured judges and datasets.",
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
