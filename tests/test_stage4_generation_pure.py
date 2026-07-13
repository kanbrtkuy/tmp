from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from cot_safety.steering.stage4_generation import (
    SamplingSpec,
    Stage4GenerationError,
    assert_rho_zero_bit_exact,
    binding_payload,
    content_sha256,
    counter_uniform,
    failure_content_sha256,
    request_fingerprint,
    resolve_a1_target_plan,
    rho_zero_reference_alias,
    row_integrity_sha256,
    sha256_text,
    terminal_checkpoint_binding_from_provenance,
    tokenizer_content_fingerprint,
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


def test_counter_uniform_is_deterministic_and_position_keyed() -> None:
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


def test_a1_plan_skips_nonordinary_post_pause_tokens() -> None:
    prompt = [1]
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
    assert plan.structural_valid is True
    assert plan.missing == ()
    assert [plan.token_ids[f"post_pause_{index}"] for index in (1, 2, 3)] == [15, 16, 17]
    assert plan.positions["post_pause_2"] == plan.positions["post_pause_1"] + 2

    short = resolve_a1_target_plan(
        TinyTokenizer(),
        prompt_token_ids=prompt,
        output_token_ids=[2, 10, 11, 12, 13, 14, 99, 99, 99, 15, 4],
        pause_token_id=99,
        assistant_ids=[1],
        think_ids=[2],
        end_think_ids=[4],
    )
    assert short.structural_valid is True
    assert "post_pause_2" in short.missing
    assert all(name not in short.missing for name in ("pause_0", "pause_1", "pause_2"))


def test_rho_zero_alias_and_content_bound_resume() -> None:
    prompt = [1, 2]
    output = [10, 11, 4]
    content = content_sha256(prompt, output)
    alias = rho_zero_reference_alias(
        prompt_token_ids=prompt,
        a1_output_token_ids=output,
        a1_content_hash=content,
    )
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
        validate_resume_row(row, expected_request_sha256="wrong")
    changed_text = dict(row)
    changed_text["generated_for_judge"] = "tampered"
    changed_text["row_integrity_sha256"] = row_integrity_sha256(changed_text)
    with pytest.raises(Stage4GenerationError, match="judge_text_hash_mismatch"):
        validate_resume_row(changed_text, expected_request_sha256=request)
    changed_prompt = dict(row)
    changed_prompt["prompt"] = "different"
    changed_prompt["row_integrity_sha256"] = row_integrity_sha256(changed_prompt)
    with pytest.raises(Stage4GenerationError, match="prompt_text_hash_mismatch"):
        validate_resume_row(changed_prompt, expected_request_sha256=request)

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


def test_terminal_checkpoint_manifest_is_the_sft_identity() -> None:
    terminal_sha = "a" * 64
    provenance = {
        "model": {"sha256": "b" * 64},
        "checkpoints": [
            {
                "step": 1064,
                "manifest_sha256": terminal_sha,
                "completion_marker_sha256": "c" * 64,
                "files": ["weights"],
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

    bad = dict(sealed)
    bad["manifest_sha256"] = "d" * 64
    with pytest.raises(Stage4GenerationError, match="does_not_match_checkpoint_seal"):
        terminal_checkpoint_binding_from_provenance(provenance, bad)


def test_full_sft_request_refuses_base_hash_semantics() -> None:
    binding = valid_binding()
    binding["model_hash_kind"] = "base_model_content_sha256"
    with pytest.raises(Stage4GenerationError, match="terminal_checkpoint_manifest"):
        request_fingerprint(
            binding=binding,
            source="s",
            split="stage4_final",
            prompt_id="p",
            prompt_sha256="a" * 64,
            rollout_seed=1,
            draw_index=0,
            arm="A1",
            alpha=0.0,
        )

    with pytest.raises(Stage4GenerationError, match="64_lowercase_hex_sha256"):
        binding_payload(
            run_id="run",
            phase="final",
            model_condition="original_base",
            model_sha256="model-placeholder",
            tokenizer_sha256="2" * 64,
            artifact_manifest_sha256="3" * 64,
            config_file_sha256="4" * 64,
            config_resolved_sha256="5" * 64,
            ledger_sha256="6" * 64,
            ledger_manifest_sha256="7" * 64,
            layer=None,
            sampling=SamplingSpec(),
            norm_cap=0.10,
        )


def test_generic_tokenizer_fingerprint_binds_vocab_special_map_and_chat_template() -> None:
    class Tokenizer:
        special_tokens_map = {"eos_token": "</s>"}
        model_max_length = 4096
        padding_side = "right"
        truncation_side = "right"
        chat_template = "{{ messages }}"

        def get_vocab(self):
            return {"a": 0, "</s>": 1}

    first = tokenizer_content_fingerprint(Tokenizer())
    second = tokenizer_content_fingerprint(Tokenizer())
    assert first == second
    assert len(first["sha256"]) == 64
    changed = Tokenizer()
    changed.chat_template = "changed"
    assert tokenizer_content_fingerprint(changed)["stage2_core_sha256"] == first["stage2_core_sha256"]
    assert tokenizer_content_fingerprint(changed)["sha256"] != first["sha256"]
    assert tokenizer_content_fingerprint(changed)["chat_template_sha256"] != first["chat_template_sha256"]


def test_resume_exact_judge_text_must_decode_from_saved_ids() -> None:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_stage4_formal_generation_hf.py"
    )
    spec = importlib.util.spec_from_file_location("stage4_formal_generation_cli", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    class Decoder:
        def decode(self, ids, skip_special_tokens=False):  # noqa: ARG002
            return "decoded:" + ",".join(str(item) for item in ids)

    row = {
        "generation_status": "complete",
        "output_token_ids": [1, 2],
        "generated_text": "decoded:1,2",
        "generated_for_judge": "decoded:1,2",
    }
    module._validate_exact_decoded_resume_text({"cell": row}, Decoder())
    row["generated_for_judge"] = "tampered"
    with pytest.raises(Stage4GenerationError, match="judge_text_decode_mismatch"):
        module._validate_exact_decoded_resume_text({"cell": row}, Decoder())


def test_cli_schedule_shards_groups_not_arms_and_freezes_calibration_grid() -> None:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_stage4_formal_generation_hf.py"
    )
    spec = importlib.util.spec_from_file_location("stage4_formal_schedule_cli", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    sources = ["harmbench", "reasoningshield", "strongreject", "wildjailbreak"]
    ledger = [
        {
            "source": source,
            "split": "stage4_final",
            "prompt_id": f"{source}:p",
            "family_id": f"{source}:f",
            "prompt": f"prompt-{source}",
        }
        for source in sources
    ]
    shards = []
    for shard_index in range(2):
        shards.extend(
            module.build_groups(
                ledger,
                phase="final",
                split="stage4_final",
                expected_prompts_per_source=1,
                draws_per_prompt=2,
                global_seed=260713,
                run_id="run",
                sources=sources,
                shard_index=shard_index,
                num_shards=2,
            )
        )
    assert len(shards) == 8
    assert len({row["group_id"] for row in shards}) == 8
    assert module._arm_alphas("calibration", selected_alpha=None) == [
        ("A1", 0.0),
        ("A2", 0.0),
        ("A2", 0.10),
        ("A2", 0.25),
        ("A2", 0.50),
        ("A2", 1.00),
    ]
    assert [arm for arm, _ in module._arm_alphas("final", selected_alpha=0.25)] == [
        "A0",
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
    ]

    class BaseTokenizerWithoutPause:
        def convert_tokens_to_ids(self, _token):
            raise AssertionError("A0 must not resolve the SFT-only pause token")

    assert (
        module._special_ids_for_condition(
            BaseTokenizerWithoutPause(), "<|pause|>", "original_base"
        )
        is None
    )


def test_a0_verifies_actual_local_base_snapshot_content(tmp_path: Path) -> None:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_stage4_formal_generation_hf.py"
    )
    spec = importlib.util.spec_from_file_location("stage4_a0_binding_cli", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    snapshot = tmp_path / "base"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "weights.safetensors").write_bytes(b"weights")
    from cot_safety.training.full_sft_runtime import directory_content_manifest

    expected = directory_content_manifest(snapshot)["sha256"]
    assert module._verify_original_base_directory(snapshot, expected)["sha256"] == expected
    (snapshot / "weights.safetensors").write_bytes(b"changed")
    with pytest.raises(Stage4GenerationError, match="directory_content_hash"):
        module._verify_original_base_directory(snapshot, expected)


def test_formal_stage4_config_has_no_unexpanded_environment_expressions() -> None:
    from cot_safety.config import load_config

    config = load_config(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "experiment"
        / "stage4_full_sft_clean_8b_2xa100.yaml"
    )

    def strings(value):
        if isinstance(value, dict):
            for item in value.values():
                yield from strings(item)
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)
        elif isinstance(value, str):
            yield value

    assert not [value for value in strings(config) if "${" in value]
