from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F


def load_pause_kl_module():
    if "trl" not in sys.modules:
        trl_stub = types.ModuleType("trl")

        class SFTTrainer:  # pragma: no cover - only used when trl is absent
            def __init__(self, *args, **kwargs):
                del args
                self.model = kwargs.get("model")
                self.tokenizer = kwargs.get("tokenizer")
                self.processing_class = kwargs.get("processing_class")
                self.args = kwargs.get("args") or SimpleNamespace(weight_decay=0.0, logging_steps=100)
                self.state = SimpleNamespace(global_step=0)

            def add_callback(self, callback):
                self.callbacks = getattr(self, "callbacks", [])
                self.callbacks.append(callback)

        trl_stub.SFTTrainer = SFTTrainer
        sys.modules["trl"] = trl_stub

    if "transformers" not in sys.modules:
        transformers_stub = types.ModuleType("transformers")

        class TrainerCallback:  # pragma: no cover - only used when transformers is absent
            pass

        transformers_stub.TrainerCallback = TrainerCallback
        sys.modules["transformers"] = transformers_stub

    path = (
        Path(__file__).resolve().parents[1]
        / "legacy"
        / "COTPauseToken"
        / "src"
        / "utils"
        / "pause_kl_trainer.py"
    )
    spec = importlib.util.spec_from_file_location("pause_kl_trainer_for_tests", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pause_kl_module = load_pause_kl_module()
PauseKLSFTTrainer = pause_kl_module.PauseKLSFTTrainer
RowsOnlyInvariantCallback = pause_kl_module._RowsOnlyInvariantCallback


class TinyLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 8, hidden_size: int = 5) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.body = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, attention_mask=None, use_cache=False, output_hidden_states=False):
        del attention_mask, use_cache
        hidden = self.body(self.embed(input_ids))
        payload = {"logits": self.lm_head(hidden)}
        if output_hidden_states:
            payload["hidden_states"] = (hidden,)
        return SimpleNamespace(**payload)


class StubTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    unk_token_id = 99

    def __init__(self):
        self.ids = {"<|pause_1|>": 4, "<|pause_2|>": 5, "<|pause_3|>": 6}

    def convert_tokens_to_ids(self, token):
        return self.ids.get(token, self.unk_token_id)


def bare_trainer(pause_token_id: int = 6):
    trainer = PauseKLSFTTrainer.__new__(PauseKLSFTTrainer)
    trainer.pause_token_id = pause_token_id
    trainer.pause_token_ids = (pause_token_id,)
    trainer.pause_token_id_set = {pause_token_id}
    trainer.pad_token_id = 0
    trainer.max_kl_tokens_per_example = 256
    trainer.require_pause_before_continuation_kl = True
    trainer.temperature = 1.0
    trainer.suppression_chunk_size = 2
    trainer.emit_weight = 0.3
    trainer.continuation_weight = 1.0
    trainer.pre_weight = 0.0
    trainer.suppression_weight = 1.0
    trainer.emit_margin_weight = 0.0
    trainer.stop_weight = 0.0
    trainer.emit_margin = 3.0
    trainer.suppression_margin = 5.0
    trainer.suppression_loss_type = "unlikelihood"
    trainer.pause_head_enabled = False
    trainer.pause_head = None
    trainer.enabled = True
    trainer.teacher_eval_mode = True
    trainer.state = SimpleNamespace(global_step=1)
    trainer.args = SimpleNamespace(logging_steps=100, weight_decay=0.0)
    trainer._last_pause_kl_log_step = -1
    trainer._pause_row_initial = {}
    trainer.log = lambda payload: None
    return trainer


def test_init_resolves_distinct_pause_ids_with_rows_only_callbacks():
    model = TinyLM(vocab_size=8, hidden_size=6)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.get_input_embeddings().weight.requires_grad_(True)
    model.get_output_embeddings().weight.requires_grad_(True)

    trainer = PauseKLSFTTrainer(
        model=model,
        tokenizer=StubTokenizer(),
        args=SimpleNamespace(weight_decay=0.0, logging_steps=100),
        pause_kl={
            "pause_tokens": ["<|pause_1|>", "<|pause_2|>", "<|pause_3|>"],
            "post_step_invariant_check": True,
        },
    )

    assert trainer.pause_token_ids == (4, 5, 6)
    assert trainer.pause_token_id_set == {4, 5, 6}
    assert trainer.callbacks


def test_pause_stripped_batch_preserves_mapping_and_padding():
    trainer = bare_trainer()
    input_ids = torch.tensor(
        [
            [10, 6, 6, 11, 12, 0],
            [20, 21, 0, 0, 0, 0],
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0],
            [1, 1, 0, 0, 0, 0],
        ]
    )

    teacher_ids, teacher_mask, mappings = trainer._pause_stripped_batch(input_ids, attention_mask)

    assert teacher_ids.tolist() == [[10, 11, 12], [20, 21, 0]]
    assert teacher_mask.tolist() == [[1, 1, 1], [1, 1, 0]]
    assert mappings == [{0: 0, 3: 1, 4: 2}, {0: 0, 1: 1}]


def test_select_kl_pairs_aligns_predicted_tokens_after_pause():
    trainer = bare_trainer()
    input_ids = torch.tensor(
        [
            [10, 11, 6, 6, 12, 13],
            [20, 21, 22, 0, 0, 0],
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0],
        ]
    )
    labels = torch.tensor(
        [
            [-100, -100, 6, 6, 12, 13],
            [-100, 21, 22, -100, -100, -100],
        ]
    )

    teacher_ids, _, mappings = trainer._pause_stripped_batch(input_ids, attention_mask)
    post_pairs, pre_pairs = trainer._select_kl_pairs(input_ids, labels, attention_mask, mappings)

    assert (0, 3, 1) in post_pairs
    assert (0, 4, 2) in post_pairs
    assert all(pair[0] == 1 for pair in pre_pairs)
    for batch_idx, student_idx, teacher_idx in post_pairs + pre_pairs:
        assert input_ids[batch_idx, student_idx + 1].item() == teacher_ids[batch_idx, teacher_idx + 1].item()


def test_select_kl_pairs_aligns_after_distinct_pause_chain():
    trainer = bare_trainer(pause_token_id=4)
    trainer.pause_token_ids = (4, 5, 6)
    trainer.pause_token_id_set = {4, 5, 6}
    input_ids = torch.tensor([[10, 11, 4, 5, 6, 12, 13]])
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([[-100, -100, 4, 5, 6, 12, 13]])

    teacher_ids, _, mappings = trainer._pause_stripped_batch(input_ids, attention_mask)
    post_pairs, pre_pairs = trainer._select_kl_pairs(input_ids, labels, attention_mask, mappings)

    assert teacher_ids.tolist() == [[10, 11, 12, 13]]
    assert pre_pairs == []
    assert (0, 5, 2) in post_pairs
    for batch_idx, student_idx, teacher_idx in post_pairs:
        assert input_ids[batch_idx, student_idx + 1].item() == teacher_ids[batch_idx, teacher_idx + 1].item()


def test_kl_loss_masks_pause_slot_and_stays_finite():
    trainer = bare_trainer()
    torch.manual_seed(0)
    student_logits = torch.randn(1, 2, 8)
    teacher_logits = student_logits.clone()
    for pause_id in trainer.pause_token_ids:
        teacher_logits[:, :, pause_id] += 1000.0

    loss = trainer._kl_loss(student_logits, teacher_logits, [(0, 0, 0), (0, 1, 1)])

    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_pause_losses_match_manual_shifted_ce_and_suppression():
    trainer = bare_trainer(pause_token_id=4)
    trainer.suppression_chunk_size = 1
    logits = torch.tensor(
        [
            [
                [0.1, 0.2, 0.3, 0.4, 1.0],
                [0.5, 0.1, -0.2, 0.3, -0.4],
                [-0.1, 0.7, 0.2, 0.0, -0.3],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    input_ids = torch.tensor([[1, 4, 2, 3]])
    labels = torch.tensor([[-100, 4, 2, 3]])

    emit, suppression, emit_margin, stop_loss = trainer._pause_losses(logits, input_ids, labels)

    expected_emit = F.cross_entropy(logits[:, 0, :], torch.tensor([4]))
    non_pause_logits = torch.stack([logits[0, 1], logits[0, 2]])
    pause_log_probs = non_pause_logits[:, 4] - torch.logsumexp(non_pause_logits, dim=-1)
    expected_suppression = -torch.log1p(-pause_log_probs.exp()).mean()
    assert emit.item() == pytest.approx(expected_emit.item(), abs=1e-6)
    assert suppression.item() == pytest.approx(expected_suppression.item(), abs=1e-6)
    assert torch.isfinite(emit_margin)
    assert stop_loss.item() == pytest.approx(0.0, abs=1e-6)


def test_distinct_pause_chain_masks_stop_after_pause3():
    trainer = bare_trainer(pause_token_id=4)
    trainer.pause_token_ids = (4, 5, 6)
    trainer.pause_token_id_set = {4, 5, 6}
    trainer.suppression_loss_type = "margin"
    logits = torch.zeros(1, 5, 8)
    input_ids = torch.tensor([[1, 4, 5, 6, 2]])
    labels = torch.tensor([[-100, 4, 5, 6, 2]])

    emit, suppression, emit_margin, stop_loss = trainer._pause_losses(logits, input_ids, labels)
    stop_mask = trainer._stop_after_pause_chain_mask(input_ids[:, :-1], labels[:, 1:].ne(-100))

    assert stop_mask.tolist() == [[False, False, False, True]]
    assert torch.isfinite(emit)
    assert torch.isfinite(suppression)
    assert torch.isfinite(emit_margin)
    assert torch.isfinite(stop_loss)


def test_emit_margin_competes_against_rival_pause_tokens():
    trainer = bare_trainer(pause_token_id=4)
    trainer.pause_token_ids = (4, 5, 6)
    trainer.pause_token_id_set = {4, 5, 6}
    trainer.emit_margin = 0.0
    logits = torch.zeros(1, 8)
    logits[0, 4] = 3.0  # rival p1
    logits[0, 5] = 1.0  # target p2
    logits[0, 2] = 0.5  # best non-pause

    loss = trainer._emit_margin_loss(logits, torch.tensor([5]), trainer._pause_ids_tensor(logits.device))

    assert loss.item() == pytest.approx(F.softplus(torch.tensor(2.0)).item(), abs=1e-6)


def test_rows_only_guard_rejects_weight_decay_and_body_params():
    trainer = bare_trainer()
    model = TinyLM()
    trainer.model = model
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.get_input_embeddings().weight.requires_grad_(True)
    model.get_output_embeddings().weight.requires_grad_(True)

    trainer.args = SimpleNamespace(weight_decay=0.01)
    with pytest.raises(ValueError, match="weight_decay"):
        trainer._assert_rows_only_training()

    trainer.args = SimpleNamespace(weight_decay=0.0)
    trainer._assert_rows_only_training()
    model.body.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="Unexpected trainable parameters"):
        trainer._assert_rows_only_training()


def test_rows_only_invariant_detects_non_pause_row_mutation():
    state = SimpleNamespace(global_step=1)
    control = SimpleNamespace()
    model = TinyLM(vocab_size=7, hidden_size=3)
    callback = RowsOnlyInvariantCallback(pause_token_id=6, chunk_rows=2)
    callback.on_train_begin(None, state, control, model=model)
    with torch.no_grad():
        model.get_input_embeddings().weight[6].add_(1.0)
        model.get_output_embeddings().weight[6].add_(1.0)
    callback.on_step_end(None, state, control, model=model)

    callback = RowsOnlyInvariantCallback(pause_token_id=6, chunk_rows=2)
    callback.on_train_begin(None, state, control, model=model)
    with torch.no_grad():
        model.get_input_embeddings().weight[2].add_(1.0)
    with pytest.raises(ValueError, match="non-pause rows changed"):
        callback.on_step_end(None, state, control, model=model)


def test_rows_only_invariant_rechecks_after_interval_and_train_end():
    control = SimpleNamespace()
    model = TinyLM(vocab_size=7, hidden_size=3)
    callback = RowsOnlyInvariantCallback(pause_token_id=6, chunk_rows=2, check_interval_steps=2)
    callback.on_train_begin(None, SimpleNamespace(global_step=0), control, model=model)

    callback.on_step_end(None, SimpleNamespace(global_step=1), control, model=model)
    with torch.no_grad():
        model.get_input_embeddings().weight[2].add_(1.0)
    callback.on_step_end(None, SimpleNamespace(global_step=2), control, model=model)
    with pytest.raises(ValueError, match="non-pause rows changed"):
        callback.on_step_end(None, SimpleNamespace(global_step=3), control, model=model)

    callback = RowsOnlyInvariantCallback(pause_token_id=6, chunk_rows=2, check_interval_steps=100)
    model = TinyLM(vocab_size=7, hidden_size=3)
    callback.on_train_begin(None, SimpleNamespace(global_step=0), control, model=model)
    callback.on_step_end(None, SimpleNamespace(global_step=1), control, model=model)
    with torch.no_grad():
        model.get_input_embeddings().weight[3].add_(1.0)
    with pytest.raises(ValueError, match="train end"):
        callback.on_train_end(None, SimpleNamespace(global_step=2), control, model=model)


def test_compute_loss_is_finite_on_batch_with_and_without_pauses():
    trainer = bare_trainer()
    model = TinyLM(vocab_size=8, hidden_size=6)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.get_input_embeddings().weight.requires_grad_(True)
    model.get_output_embeddings().weight.requires_grad_(True)

    inputs = {
        "input_ids": torch.tensor(
            [
                [1, 2, 6, 6, 3, 4],
                [1, 2, 3, 4, 0, 0],
            ]
        ),
        "attention_mask": torch.tensor(
            [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 0],
            ]
        ),
        "labels": torch.tensor(
            [
                [-100, -100, 6, 6, 3, 4],
                [-100, -100, 3, 4, -100, -100],
            ]
        ),
    }

    loss = trainer.compute_loss(model, inputs)
    loss.backward()

    assert torch.isfinite(loss)
    assert model.body.weight.grad is None
    assert model.get_input_embeddings().weight.grad is not None
    assert model.get_output_embeddings().weight.grad is not None
