#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def resolve_repo_path(value: str, *, base_dir: Path = REPO_ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def select_state_block(npz_path: Path, *, layer: int, positions: list[str]) -> tuple[Any, Any, dict[str, Any]]:
    import numpy as np

    with np.load(npz_path, allow_pickle=True) as data:
        features = np.asarray(data["features"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.int64)
        layer_ids = [int(item) for item in data["layer_ids"].tolist()]
        position_names = [str(item) for item in data["position_names"].tolist()]
        valid_mask = np.asarray(data["valid_mask"], dtype=bool) if "valid_mask" in data.files else None
    if layer not in layer_ids:
        raise ValueError(f"Layer {layer} not found in {npz_path}; available={layer_ids}")
    missing = [position for position in positions if position not in position_names]
    if missing:
        raise ValueError(f"Positions missing from {npz_path}: {missing}; available={position_names}")
    layer_idx = layer_ids.index(layer)
    pos_idx = [position_names.index(position) for position in positions]
    keep = labels >= 0
    if valid_mask is not None:
        keep &= valid_mask[:, pos_idx].all(axis=1)
    n_dropped_invalid = int((labels >= 0).sum() - keep.sum())
    states = features[:, layer_idx, pos_idx, :].mean(axis=1)
    states = states[keep]
    labels = labels[keep]
    if not (labels == 0).any() or not (labels == 1).any():
        raise ValueError(f"Need both safe(0) and unsafe(1) labels in {npz_path}")
    meta = {
        "hidden_npz": str(npz_path),
        "layer": layer,
        "positions": positions,
        "n_rows": int(labels.shape[0]),
        "n_safe": int((labels == 0).sum()),
        "n_unsafe": int((labels == 1).sum()),
        "n_dropped_invalid_positions": n_dropped_invalid,
    }
    return states, labels, meta


def build_artifacts(
    *,
    hidden_npz: Path,
    direction_path: Path,
    centroid_path: Path,
    probe_target: Path,
    probe_source: Path | None,
    stage3_evidence_report: Path,
    layer: int,
    positions: list[str],
    manifest_path: Path,
) -> dict[str, Any]:
    evidence = read_json(stage3_evidence_report)
    if evidence.get("status") != "pass" or evidence.get("pause_only_status") != "pass":
        raise SystemExit(
            "Refusing to build GPRS artifacts because Stage3 evidence did not pass: "
            f"status={evidence.get('status')} pause_only_status={evidence.get('pause_only_status')} "
            f"report={stage3_evidence_report}"
        )
    import torch

    states, labels, meta = select_state_block(hidden_npz, layer=layer, positions=positions)
    safe = states[labels == 0]
    unsafe = states[labels == 1]
    safe_centroid = safe.mean(axis=0)
    unsafe_centroid = unsafe.mean(axis=0)
    direction = unsafe_centroid - safe_centroid
    direction_norm = float((direction**2).sum() ** 0.5)
    if direction_norm <= 0.0:
        raise ValueError("Unsafe-safe mean-diff direction has zero norm.")
    direction = direction / direction_norm

    direction_path.parent.mkdir(parents=True, exist_ok=True)
    centroid_path.parent.mkdir(parents=True, exist_ok=True)
    probe_target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"direction": torch.as_tensor(direction), **meta}, direction_path)
    torch.save({"safe_centroid": torch.as_tensor(safe_centroid), **meta}, centroid_path)
    if probe_source is not None:
        if not probe_source.exists():
            raise FileNotFoundError(f"Probe checkpoint source is missing: {probe_source}")
        probe_payload = torch.load(probe_source, map_location="cpu")
        if isinstance(probe_payload, dict):
            probe_layers = [int(item) for item in probe_payload.get("layers", []) or []]
            probe_positions = [str(item) for item in probe_payload.get("positions", []) or []]
            if probe_layers and layer not in probe_layers:
                raise ValueError(f"Probe checkpoint layers {probe_layers} do not include steering layer {layer}.")
            if probe_positions and set(probe_positions) != set(positions):
                raise ValueError(
                    f"Probe checkpoint positions {probe_positions} do not match requested positions {positions}."
                )
        if probe_source.resolve() != probe_target.resolve():
            shutil.copy2(probe_source, probe_target)
    elif not probe_target.exists():
        raise FileNotFoundError(
            "Probe checkpoint is missing. Pass --probe_checkpoint_source or create "
            f"the configured checkpoint first: {probe_target}"
        )
    manifest = {
        "status": "ready",
        "direction_artifact": str(direction_path),
        "safe_centroid": str(centroid_path),
        "probe_checkpoint": str(probe_target),
        "direction_provenance": "teacher_forced_prompt_labels",
        "stage3_evidence": {
            "path": str(stage3_evidence_report),
            "status": evidence.get("status"),
            "pause_only_status": evidence.get("pause_only_status"),
            "pause_minus_best_baseline": evidence.get("pause_minus_best_baseline"),
            "pause_only_margin": evidence.get("pause_only_margin"),
            "selection_metric": evidence.get("selection_metric"),
            "metric": evidence.get("metric"),
        },
        "direction_norm_before_normalization": direction_norm,
        **meta,
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Stage4 GPRS mean-diff direction and safe-centroid artifacts from Stage3 hidden states."
    )
    parser.add_argument("--config", default="configs/experiment/stage4_pause_gprs.yaml")
    parser.add_argument("--hidden_npz", required=True, help="Stage3 hidden NPZ, usually the train split.")
    parser.add_argument("--stage3_evidence_report", default=None)
    parser.add_argument("--positions", default=None, help="Comma-separated positions. Defaults to steering.target_positions.")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--probe_checkpoint_source", default=None)
    parser.add_argument("--manifest_json", default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    from cot_safety.config import load_config
    from cot_safety.steering.gprs import validate_gprs_config

    config = load_config(REPO_ROOT / args.config)
    meta = validate_gprs_config(config)
    if meta["method"] == "learned_delta":
        raise SystemExit("GPRS artifact builder requires steering.method: gprs/projection.")
    steering = config.get("steering", {})
    layer = int(args.layer if args.layer is not None else steering.get("layer", 14))
    positions = (
        [piece.strip() for piece in args.positions.split(",") if piece.strip()]
        if args.positions
        else [str(item) for item in steering.get("target_positions", ["pause_0", "pause_1", "pause_2"])]
    )
    direction_path = resolve_repo_path(meta["direction_artifact"])
    centroid_path = resolve_repo_path(meta["safe_centroid"])
    probe_target = resolve_repo_path(meta["probe_checkpoint"])
    evidence_report_raw = args.stage3_evidence_report or meta.get("stage3_evidence_report") or ""
    if not str(evidence_report_raw).strip():
        raise SystemExit("GPRS artifact builder requires --stage3_evidence_report or steering.gprs.stage3_evidence_report.")
    evidence_report = resolve_repo_path(str(evidence_report_raw))
    probe_source = resolve_repo_path(args.probe_checkpoint_source) if args.probe_checkpoint_source else None
    manifest_path = (
        resolve_repo_path(args.manifest_json)
        if args.manifest_json
        else direction_path.parent / "gprs_artifact_manifest.json"
    )
    plan = {
        "hidden_npz": str(resolve_repo_path(args.hidden_npz)),
        "layer": layer,
        "positions": positions,
        "direction_artifact": str(direction_path),
        "safe_centroid": str(centroid_path),
        "probe_checkpoint": str(probe_target),
        "probe_checkpoint_source": str(probe_source) if probe_source else "",
        "stage3_evidence_report": str(evidence_report),
        "manifest_json": str(manifest_path),
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    manifest = build_artifacts(
        hidden_npz=resolve_repo_path(args.hidden_npz),
        direction_path=direction_path,
        centroid_path=centroid_path,
        probe_target=probe_target,
        probe_source=probe_source,
        stage3_evidence_report=evidence_report,
        layer=layer,
        positions=positions,
        manifest_path=manifest_path,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
