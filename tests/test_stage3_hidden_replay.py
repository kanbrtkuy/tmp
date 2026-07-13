from __future__ import annotations

import numpy as np
import pytest

from cot_safety.probes.stage3_hidden_replay import (
    ExactReplayItem,
    Stage3HiddenReplayError,
    build_exact_replay_batch,
    replay_with_oom_policy,
)


def test_exact_replay_batch_right_pads_without_changing_absolute_positions() -> None:
    batch = build_exact_replay_batch(
        [
            ExactReplayItem(
                token_ids=(11, 12, 13, 14),
                target_positions=(0, 3, 0),
                target_valid=(True, True, False),
            ),
            ExactReplayItem(
                token_ids=(21, 22),
                target_positions=(1, 0, 0),
                target_valid=(True, False, False),
            ),
        ],
        pad_token_id=99,
    )
    np.testing.assert_array_equal(batch.input_ids, [[11, 12, 13, 14], [21, 22, 99, 99]])
    np.testing.assert_array_equal(batch.attention_mask, [[1, 1, 1, 1], [1, 1, 0, 0]])
    np.testing.assert_array_equal(batch.position_ids, [[0, 1, 2, 3], [0, 1, 0, 0]])
    np.testing.assert_array_equal(batch.target_positions, [[0, 3, 0], [1, 0, 0]])
    np.testing.assert_array_equal(batch.target_valid, [[True, True, False], [True, False, False]])
    assert batch.sequence_lengths.tolist() == [4, 2]


def test_exact_replay_rejects_target_outside_unretokenized_ids() -> None:
    with pytest.raises(Stage3HiddenReplayError, match="outside sequence"):
        ExactReplayItem(
            token_ids=(1, 2),
            target_positions=(2,),
            target_valid=(True,),
        )


def test_cuda_oom_policy_halves_only_the_failing_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    import cot_safety.probes.stage3_hidden_replay as replay

    def fake_capture(_model, batch, *, layer_ids, device):
        del device
        if batch.input_ids.shape[0] > 2:
            raise RuntimeError("CUDA out of memory")
        rows = batch.input_ids.shape[0]
        targets = batch.target_positions.shape[1]
        values = np.zeros((rows, len(layer_ids), targets, 1), dtype=np.float16)
        values[:, :, :, 0] = batch.input_ids[:, :1, None]
        return values

    monkeypatch.setattr(replay, "capture_exact_hidden_batch", fake_capture)
    items = [ExactReplayItem((index + 1, 9), (0,), (True,)) for index in range(5)]
    values, runtime = replay_with_oom_policy(
        object(),
        items,
        layer_ids=(4,),
        pad_token_id=0,
        device="cuda",
        batch_size=4,
        min_batch_size=1,
        oom_policy="halve",
    )
    assert runtime["cuda_oom_retries"] == 1
    assert values[:, 0, 0, 0].tolist() == [1, 2, 3, 4, 5]


def test_cuda_oom_fail_policy_stops_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    import cot_safety.probes.stage3_hidden_replay as replay

    def always_oom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(replay, "capture_exact_hidden_batch", always_oom)
    with pytest.raises(Stage3HiddenReplayError, match="stopped fail-closed"):
        replay_with_oom_policy(
            object(),
            [ExactReplayItem((1,), (0,), (True,))],
            layer_ids=(4,),
            pad_token_id=0,
            device="cuda",
            batch_size=1,
            oom_policy="fail",
        )


def test_hook_capture_matches_output_hidden_states_on_tiny_llama() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from cot_safety.probes.stage3_hidden_replay import capture_exact_hidden_batch

    config = transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    model = transformers.LlamaForCausalLM(config).eval()
    batch = build_exact_replay_batch(
        [
            ExactReplayItem((1, 2, 3, 4), (1, 3), (True, True)),
            ExactReplayItem((5, 6, 7), (0, 2), (True, True)),
        ],
        pad_token_id=0,
    )
    captured = capture_exact_hidden_batch(model, batch, layer_ids=(1, 2), device="cpu")
    with torch.inference_mode():
        reference = model.model(
            input_ids=torch.as_tensor(batch.input_ids),
            attention_mask=torch.as_tensor(batch.attention_mask),
            position_ids=torch.as_tensor(batch.position_ids),
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        ).hidden_states
    expected = np.stack(
        [
            np.stack(
                [
                    reference[layer][row, batch.target_positions[row]].numpy()
                    for layer in (1, 2)
                ]
            )
            for row in range(2)
        ]
    )
    np.testing.assert_allclose(captured.astype(np.float32), expected, atol=2e-3, rtol=2e-3)
