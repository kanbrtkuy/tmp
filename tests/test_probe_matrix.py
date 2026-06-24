from __future__ import annotations

import numpy as np

from cot_safety.probes.features import make_probe_matrix


def test_probe_matrix_concat_layers_and_pause_span():
    data = {
        "features": np.arange(2 * 2 * 3 * 4, dtype=np.float32).reshape(2, 2, 3, 4),
        "valid_mask": np.array([[True, True, True], [True, True, True]]),
        "labels": np.array([0, 1]),
        "position_names": np.array(["pause_0", "pause_1", "pause_2"], dtype=object),
        "layer_ids": np.array([7, 14]),
    }

    x, y, meta, kept = make_probe_matrix(
        data,
        position_names=["pause_0", "pause_1", "pause_2"],
        layer_ids=[7, 14],
        layer_combine="concat",
        position_pool="concat",
    )

    assert x.shape == (2, 24)
    assert y.tolist() == [0.0, 1.0]
    assert meta["positions"] == ["pause_0", "pause_1", "pause_2"]
    assert meta["layers"] == [7, 14]
    assert kept.tolist() == [0, 1]
