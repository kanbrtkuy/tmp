from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from cot_safety.steering.stage4_generation import (  # noqa: E402
    CounterKey,
    SamplingSpec,
    Stage4GenerationError,
    assert_rho_zero_bit_exact,
    binding_payload,
    content_sha256,
    counter_uniform,
    counterfactual_generate_batch,
    counterfactual_greedy_generate_batch,
    exact_matched_relative_hook,
    failure_content_sha256,
    later_kv_change_report,
    natural_greedy_generate_batch,
    prefix_kv_integrity_preflight,
    request_fingerprint,
    resolve_a1_target_plan,
    rho_zero_reference_alias,
    row_integrity_sha256,
    sample_top_p_from_uniform,
    sha256_text,
    terminal_checkpoint_binding_from_provenance,
    validate_resume_row,
)


class TinyTokenizer:
    pieces = {
        1: "<A>",
        2: "<think>",
        3: " ",
        4: "</think>",
        10: "a",
        11: "b",
        12: "c",
        13: "d",
        14: "e",
        15: "f",
        16: "g",
        17: "h",
        99: "<pause>",
    }
    all_special_ids = [1, 2, 4, 99]

    def decode(self, ids, skip_special_tokens=False):  # noqa: ARG002
        return "".join(self.pieces[int(item)] for item in ids)


def target_fixture():
    prompt = [1]
    # post_pause_2 is 16, not the intervening pure-whitespace token 3.
    output = [2, 10, 11, 12, 13, 14, 99, 99, 99, 15, 3, 16, 17, 4]
    plan = resolve_a1_target_plan(
        TinyTokenizer(),
        prompt_token_ids=prompt,
        output_token_ids=output,
        pause_token_id=99,
        assistant_ids=[1],
        think_ids=[2],
        end_think_ids=[4],
    )
    return prompt, output, plan


def test_counter_uniform_and_top_p_are_deterministic_and_position_keyed() -> None:
    first = counter_uniform(
        run_id="run", prompt_id="prompt", rollout_seed=17, absolute_output_position=3
    )
    assert first == counter_uniform(
        run_id="run", prompt_id="prompt", rollout_seed=17, absolute_output_position=3
    )
    assert first != counter_uniform(
        run_id="run", prompt_id="prompt", rollout_seed=17, absolute_output_position=4
    )
    assert 0.0 < first < 1.0
    logits = torch.tensor([20.0, 0.0, -5.0])
    assert sample_top_p_from_uniform(logits, uniform=0.999, temperature=0.6, top_p=0.95) == 0


def test_exact_a1_target_resolution_uses_first_three_ordinary_post_pause_tokens() -> None:
    _prompt, _output, plan = target_fixture()
    assert plan.structural_valid is True
    assert plan.missing == ()
    assert [plan.token_ids[f"post_pause_{index}"] for index in (1, 2, 3)] == [15, 16, 17]
    assert plan.positions["post_pause_2"] == plan.positions["post_pause_1"] + 2
    assert [plan.token_ids[f"pause_{index}"] for index in range(3)] == [99, 99, 99]


class FakeBlock(torch.nn.Module):
    def __init__(self, index: int):
        super().__init__()
        self.index = int(index)

    def forward(self, hidden):
        return (hidden + float(self.index + 1),)


class FakeModel(torch.nn.Module):
    """A hook-compatible decoder whose returned cache is easy to inspect."""

    def __init__(self, *, n_layers: int = 4, hidden: int = 6, vocab: int = 128):
        super().__init__()
        self.hidden = hidden
        self.vocab = vocab
        self.model = SimpleNamespace(layers=torch.nn.ModuleList([FakeBlock(i) for i in range(n_layers)]))

    def forward(
        self,
        input_ids,
        attention_mask=None,  # noqa: ARG002
        position_ids=None,  # noqa: ARG002
        past_key_values=None,  # noqa: ARG002
        use_cache=True,  # noqa: ARG002
        return_dict=True,  # noqa: ARG002
    ):
        hidden = torch.nn.functional.one_hot(
            input_ids.remainder(self.hidden), num_classes=self.hidden
        ).float()
        cache = []
        for block in self.model.layers:
            hidden = block(hidden)[0]
            cache.append((hidden.clone(), (hidden * 2.0).clone()))
        logits = torch.full(
            (input_ids.shape[0], input_ids.shape[1], self.vocab),
            -30.0,
            device=input_ids.device,
        )
        logits[..., 4] = 30.0  # immediately emit the EOS id
        return SimpleNamespace(logits=logits, past_key_values=tuple(cache))


def test_exact_hook_touches_three_and_changes_later_layer_kv() -> None:
    model = FakeModel()
    input_ids = torch.tensor([[10, 11, 12, 13, 14]])
    baseline = model(input_ids=input_ids)
    mask = torch.tensor([[False, True, True, True, False]])
    with exact_matched_relative_hook(
        model,
        hidden_state_index=1,
        padded_target_mask=mask,
        unit_direction=torch.tensor([1.0, 2.0, 0.0, 0.0, 0.0, 0.0]),
        rho=0.05,
        target_names_by_row=[["pause_0", "pause_1", "pause_2"]],
        target_token_ids_by_row=[[11, 12, 13]],
    ) as stats:
        steered = model(input_ids=input_ids)
    assert stats["num_applied_calls"] == 1
    assert stats["per_row"][0]["touched_token_ids"] == [11, 12, 13]
    assert stats["per_row"][0]["actual_relative_norms"] == pytest.approx([0.05] * 3)
    report = later_kv_change_report(
        baseline.past_key_values,
        steered.past_key_values,
        decoder_block_index=0,
    )
    assert report["pass"] is True
    assert any(row["changed"] for row in report["later_layers"])

    with pytest.raises(Stage4GenerationError, match="exactly_three_targets"):
        with exact_matched_relative_hook(
            model,
            hidden_state_index=1,
            padded_target_mask=torch.tensor([[False, True, True, False, False]]),
            unit_direction=torch.ones(6),
            rho=0.05,
            target_names_by_row=[["a", "b", "c"]],
            target_token_ids_by_row=[[1, 2, 3]],
        ):
            pass


def test_counterfactual_replays_only_through_target_and_uses_returned_cache() -> None:
    prompt, output, plan = target_fixture()
    generated, finish, audits = counterfactual_generate_batch(
        FakeModel(),
        prompt_token_ids=[prompt],
        a1_output_token_ids=[output],
        target_plans=[plan],
        target_names=("pause_0", "pause_1", "pause_2"),
        unit_direction=torch.ones(6),
        hidden_state_index=1,
        rho=0.025,
        counter_keys=[CounterKey("run", "prompt", 7)],
        sampling=SamplingSpec(),
        pad_token_id=0,
        eos_token_ids=4,
        device=torch.device("cpu"),
    )
    # The replay ends at pause_2 (9 generated tokens), then the cache-backed
    # continuation emits EOS.  Natural A1 post-pause tokens are not replayed.
    assert generated[0] == output[:9] + [4]
    assert finish == ["eos"]
    assert audits[0]["teacher_replay_output_tokens"] == 9
    assert audits[0]["hook_timing"]["cache_returned_after_application"] is True
    assert audits[0]["actual_relative_norms"] == pytest.approx([0.025] * 3)

    preflight = prefix_kv_integrity_preflight(
        FakeModel(),
        prompt_token_ids=prompt,
        a1_output_token_ids=output,
        target_plan=plan,
        target_names=("pause_0", "pause_1", "pause_2"),
        unit_direction=torch.ones(6),
        hidden_state_index=1,
        rho=0.025,
        pad_token_id=0,
        device=torch.device("cpu"),
    )
    assert preflight["status"] == "pass"
    assert preflight["num_applied_calls"] == 1
    assert preflight["cache_returned_after_hook"] is True

    natural_ids, natural_finish = natural_greedy_generate_batch(
        FakeModel(),
        prompt_token_ids=[prompt],
        pad_token_id=0,
        eos_token_ids=4,
        device=torch.device("cpu"),
    )
    assert natural_ids == [[4]]
    assert natural_finish == ["eos"]
    greedy_ids, greedy_finish, greedy_audit = counterfactual_greedy_generate_batch(
        FakeModel(),
        prompt_token_ids=[prompt],
        a1_output_token_ids=[output],
        target_plans=[plan],
        target_names=("cot_2", "cot_3", "cot_4"),
        unit_direction=torch.ones(6),
        hidden_state_index=1,
        rho=0.025,
        pad_token_id=0,
        eos_token_ids=4,
        device=torch.device("cpu"),
    )
    assert greedy_ids[0] == output[:6] + [4]
    assert greedy_finish == ["eos"]
    assert greedy_audit[0]["teacher_replay_output_tokens"] == 6


def valid_binding():
    return binding_payload(
        run_id="run",
        phase="final",
        model_condition="full_sft",
        model_sha256="1" * 64,
        tokenizer_sha256="2" * 64,
        artifact_manifest_sha256="3" * 64,
        config_file_sha256="4" * 64,
        config_resolved_sha256="5" * 64,
        ledger_sha256="6" * 64,
        ledger_manifest_sha256="7" * 64,
        layer=28,
        sampling=SamplingSpec(),
        norm_cap=0.10,
        stage2_provenance_sha256="8" * 64,
        terminal_checkpoint_completion_marker_sha256="9" * 64,
        calibration_report_sha256="a" * 64,
    )


def test_rho_zero_alias_is_bit_exact_and_resume_checks_every_hash() -> None:
    prompt = [1, 2]
    output = [10, 11, 4]
    content = content_sha256(prompt, output)
    alias = rho_zero_reference_alias(
        prompt_token_ids=prompt,
        a1_output_token_ids=output,
        a1_content_hash=content,
    )
    assert alias["output_token_ids"] == output
    assert_rho_zero_bit_exact(
        {"prompt_token_ids": prompt, "output_token_ids": output, "generated_content_sha256": content},
        {"prompt_token_ids": prompt, **alias},
    )
    binding = valid_binding()
    prompt_hash = sha256_text("hi")
    request = request_fingerprint(
        binding=binding,
        source="s",
        split="stage4_final",
        prompt_id="p",
        prompt_sha256=prompt_hash,
        rollout_seed=1,
        draw_index=0,
        arm="A1",
        alpha=0.0,
    )
    row = {
        "schema_version": "stage4_formal_minimal_prefix_generation_v1",
        "request_sha256": request,
        "binding": binding,
        "source": "s",
        "split": "stage4_final",
        "prompt_id": "p",
        "rollout_seed": 1,
        "draw_index": 0,
        "arm": "A1",
        "alpha": 0.0,
        "generation_status": "complete",
        "prompt": "hi",
        "prompt_sha256": prompt_hash,
        "prompt_token_ids": prompt,
        "output_token_ids": output,
        "generated_content_sha256": content,
        "generated_text": "answer",
        "generated_text_sha256": "0db52f4076c082518412afd3dd3576e2cb0c63703fd7fed5e23ade60efef31d9",
        "generated_for_judge": "answer",
        "generated_for_judge_sha256": "0db52f4076c082518412afd3dd3576e2cb0c63703fd7fed5e23ade60efef31d9",
        "failure": None,
        "failure_content_sha256": None,
    }
    row["row_integrity_sha256"] = row_integrity_sha256(row)
    validate_resume_row(row, expected_request_sha256=request)
    with pytest.raises(Stage4GenerationError, match="request_binding_mismatch"):
        validate_resume_row(row, expected_request_sha256="other")
    changed = dict(row)
    changed["output_token_ids"] = [10, 4]
    with pytest.raises(Stage4GenerationError, match="content_hash_mismatch"):
        validate_resume_row(changed, expected_request_sha256=request)

    failure = {"code": "missing_target", "detail": "pause_2"}
    failed = {
        "schema_version": row["schema_version"],
        "request_sha256": request,
        "binding": binding,
        "source": "s",
        "split": "stage4_final",
        "prompt_id": "p",
        "rollout_seed": 1,
        "draw_index": 0,
        "arm": "A1",
        "alpha": 0.0,
        "generation_status": "scheduled_failure",
        "prompt": "hi",
        "prompt_sha256": row["prompt_sha256"],
        "prompt_token_ids": prompt,
        "output_token_ids": None,
        "generated_content_sha256": None,
        "failure": failure,
        "failure_content_sha256": failure_content_sha256(request, failure),
    }
    failed["row_integrity_sha256"] = row_integrity_sha256(failed)
    validate_resume_row(failed, expected_request_sha256=request)


def test_terminal_sft_identity_is_checkpoint_manifest_not_base_model_hash() -> None:
    terminal_sha = "a" * 64
    provenance = {
        "model": {"sha256": "b" * 64},
        "checkpoints": [
            {
                "step": 1064,
                "manifest_sha256": terminal_sha,
                "completion_marker_sha256": "c" * 64,
                "files": [{"path": "model.safetensors"}],
            }
        ],
    }
    sealed = {
        "checkpoint_name": "checkpoint-1064",
        "global_step": 1064,
        "manifest_sha256": terminal_sha,
        "completion_marker_sha256": "c" * 64,
        "payload_bytes": 10,
    }
    result = terminal_checkpoint_binding_from_provenance(provenance, sealed)
    assert result["sha256"] == terminal_sha
    assert result["sha256"] != provenance["model"]["sha256"]
    assert result["binding_kind"] == "terminal_checkpoint_manifest_sha256"
    assert result["base_model_sha256"] == "b" * 64

    bad = dict(sealed)
    bad["manifest_sha256"] = "d" * 64
    with pytest.raises(Stage4GenerationError, match="does_not_match_checkpoint_seal"):
        terminal_checkpoint_binding_from_provenance(provenance, bad)
