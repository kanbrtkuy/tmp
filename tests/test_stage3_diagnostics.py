from __future__ import annotations

import json
import numpy as np

from cot_safety.probes.stage3_diagnostics import (
    Stage3DiagnosticInputs,
    outer_fold_nuisance_diagnostic,
    prompt_only_propensity_diagnostic,
)
from cot_safety.probes.stage3_formal import FormalStage3Data


def make_diagnostic_data() -> tuple[FormalStage3Data, Stage3DiagnosticInputs]:
    sources = ("a", "b", "c", "d")
    labels = []
    states = []
    prompt_ids = []
    source_ids = []
    split_ids = []
    prompt_rows = []
    output_lengths = []
    refusal = []
    surface = []
    for split in ("stage3_train", "stage3_sealed"):
        for source_index, source in enumerate(sources):
            for prompt_index, unsafe_count in enumerate((10, 35, 65, 90)):
                prompt_id = f"{split}-{source}-{prompt_index}"
                propensity = unsafe_count / 100.0
                prompt_rows.append((split, source, prompt_id, propensity))
                local_labels = [0] * (100 - unsafe_count) + [1] * unsafe_count
                for draw_index, label in enumerate(local_labels):
                    labels.append(label)
                    # Main pause state carries a true within-prompt label signal.
                    states.append([[float(label) * 2.0 - 1.0, source_index * 0.01]])
                    prompt_ids.append(prompt_id)
                    source_ids.append(source)
                    split_ids.append(split)
                    output_lengths.append(20 + 10 * label + prompt_index)
                    refusal.append(1 - label)
                    surface.append([float(label), float(draw_index % 3), propensity])
    data = FormalStage3Data(
        states=np.asarray(states, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        prompt_ids=np.asarray(prompt_ids, dtype=object),
        source_ids=np.asarray(source_ids, dtype=object),
        split_ids=np.asarray(split_ids, dtype=object),
        valid_mask=np.ones(len(labels), dtype=bool),
        layer_ids=(4,),
    )
    prompt_states = np.zeros((len(prompt_rows), 1, 2, 2), dtype=np.float32)
    for index, (_, _, _, propensity) in enumerate(prompt_rows):
        prompt_states[index, 0, :, 0] = propensity
        prompt_states[index, 0, :, 1] = 1.0 - propensity
    inputs = Stage3DiagnosticInputs(
        prompt_states=prompt_states,
        prompt_state_valid=np.ones((len(prompt_rows), 2), dtype=bool),
        prompt_ids=np.asarray([row[2] for row in prompt_rows], dtype=object),
        prompt_source_ids=np.asarray([row[1] for row in prompt_rows], dtype=object),
        prompt_split_ids=np.asarray([row[0] for row in prompt_rows], dtype=object),
        prompt_state_cell_ids=np.asarray(
            [[f"{row[2]}:0", f"{row[2]}:0"] for row in prompt_rows], dtype=object
        ),
        row_prompt_lengths=np.asarray(
            [50 + int(prompt.rsplit("-", 1)[1]) for prompt in prompt_ids], dtype=np.int64
        ),
        row_output_lengths=np.asarray(output_lengths, dtype=np.int64),
        row_refusal_flags=np.asarray(refusal, dtype=np.int8),
        row_surface_features=np.asarray(surface, dtype=np.float32),
        layer_ids=(4,),
    )
    return data, inputs


def test_prompt_only_is_across_prompt_source_heldout_and_not_a_gate() -> None:
    data, inputs = make_diagnostic_data()
    result = prompt_only_propensity_diagnostic(
        data,
        inputs,
        sources=("a", "b", "c", "d"),
        candidate_layers=(4,),
        ridge_grid=(0.01, 0.1),
    )
    assert result["diagnostic_only"] is True
    assert result["changes_stage3_gate"] is False
    assert result["sample_unit"] == "prompt"
    for position in ("last_prompt_token", "pre_think"):
        assert result["positions"][position]["macro_prompt_level_spearman"] > 0.99
        for fold in result["positions"][position]["outer_folds"].values():
            assert fold["selection_scope"] == "stage3_train_inner_source_loso_only"
            assert fold["sealed_test_prompts"] == 4
    json.dumps(result, allow_nan=False)


def test_nuisance_uses_only_outer_fold_sealed_scores_and_prompt_clusters() -> None:
    data, inputs = make_diagnostic_data()
    result = outer_fold_nuisance_diagnostic(
        data,
        inputs,
        sources=("a", "b", "c", "d"),
        candidate_layers=(4,),
    )
    assert result["diagnostic_only"] is True
    assert result["hidden_scores"] == "sealed outer-heldout-source scores only"
    assert set(result["associations"]) == {
        "output_token_length",
        "prompt_token_length",
        "wildguard_response_refusal",
        "source_heldout_surface_text_score",
    }
    for fold in result["outer_folds"].values():
        assert fold["sealed_scored_rollouts"] == 400
        assert len(fold["training_sources"]) == 3
    assert (
        result["associations"]["output_token_length"][
            "macro_within_prompt_pearson"
        ]
        > 0.9
    )
    json.dumps(result, allow_nan=False)
