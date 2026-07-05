from cot_safety.data.safe_rewrite import (
    build_safe_length_repair_prompt,
    build_safe_polish_prompt,
    build_safe_rewrite_prompt,
    configured_style_profiles,
    length_target_for_unsafe,
    make_tasks,
    merge_generated_pair,
    normalize_generated_rewrite,
    pair_record_to_long_rows,
    select_rows,
)


def sample_config():
    return {
        "prompt_version": "test_prompt_v1",
        "data": {
            "prompt_fields": ["query"],
            "unsafe_trajectory_fields": ["bad_cot"],
            "unsafe_final_answer_fields": ["bad_final"],
            "id_fields": ["id"],
            "include_unsafe_trajectory": True,
        },
        "selection": {
            "shuffle": False,
            "require_unsafe_label": True,
            "label_field": "label",
        },
        "generation": {
            "provider": "openai",
            "model": "gpt-4.1",
            "temperature": 0.2,
            "max_unsafe_excerpt_chars": 64,
            "length_matching": {
                "enabled": True,
                "min_ratio": 0.75,
                "max_ratio": 1.10,
                "min_reasoning_words": 20,
                "max_reasoning_words": 60,
            },
            "style_profiles": {
                "enabled": True,
                "seed": 123,
                "choices": ["defensive_technical", "risk_assessment"],
            },
        },
    }


def test_select_rows_can_require_unsafe_label():
    rows = [
        {"id": "a", "query": "Prompt A", "bad_cot": "Unsafe reasoning.", "label": "unsafe"},
        {"id": "b", "query": "Prompt B", "bad_cot": "Safe reasoning.", "label": "safe"},
    ]

    selected = select_rows(rows, sample_config())

    assert [row["id"] for row in selected] == ["a"]


def test_make_tasks_extracts_configured_fields_and_truncates_excerpt():
    rows = [
        {
            "id": "p001",
            "query": "User prompt",
            "bad_cot": "x" * 80,
            "bad_final": "Original final",
            "label": "unsafe",
            "source": "unit",
        }
    ]

    tasks = make_tasks(sample_config(), rows)

    assert len(tasks) == 1
    assert tasks[0]["pair_id"] == "p001::unsafe_to_safe"
    assert tasks[0]["prompt"] == "User prompt"
    assert tasks[0]["unsafe_trajectory_excerpt"] == "x" * 64
    assert tasks[0]["length_target"]["target_min_words"] >= 20
    assert tasks[0]["style_profile"] in {"defensive_technical", "risk_assessment"}
    assert "Unsafe trajectory to correct" in tasks[0]["user_prompt"]
    assert "Target safe reasoning length" in tasks[0]["user_prompt"]
    assert "Assigned style_profile" in tasks[0]["user_prompt"]


def test_style_profile_assignment_is_deterministic():
    rows = [
        {
            "id": "p001",
            "query": "User prompt",
            "bad_cot": "Unsafe reasoning.",
            "label": "unsafe",
        }
    ]

    first = make_tasks(sample_config(), rows)[0]
    second = make_tasks(sample_config(), rows)[0]

    assert configured_style_profiles(sample_config()) == ["defensive_technical", "risk_assessment"]
    assert first["style_profile"] == second["style_profile"]
    assert first["style_instructions"]


def test_normalize_generated_rewrite_accepts_aliases():
    generated = normalize_generated_rewrite({"reasoning": "safe text", "answer": "final"})

    assert generated["safe_reasoning"] == "safe text"
    assert generated["safe_trajectory"] == "<think>\nsafe text\n</think>\nfinal"
    assert generated["safe_final_answer"] == "final"
    assert generated["label"] == "safe"


def test_normalize_generated_rewrite_parses_legacy_think_trajectory():
    generated = normalize_generated_rewrite(
        {"safe_trajectory": "<think>Safe rationale.</think> Safe answer."}
    )

    assert generated["safe_reasoning"] == "Safe rationale."
    assert generated["safe_final_answer"] == "Safe answer."
    assert generated["safe_trajectory"] == "<think>\nSafe rationale.\n</think>\nSafe answer."


def test_merge_generated_pair_and_long_rows():
    unsafe_reasoning = " ".join(f"unsafe_step_{idx}" for idx in range(40))
    safe_reasoning = (
        "Assess the request at a high level, identify the safety risk, refuse operational details, "
        "and redirect toward defensive review and benign alternatives."
    )
    rows = [
        {
            "id": "p001",
            "query": "User prompt",
            "bad_cot": unsafe_reasoning,
            "bad_final": "Unsafe final.",
            "label": "unsafe",
        }
    ]
    task = make_tasks(sample_config(), rows)[0]
    record = merge_generated_pair(
        task,
        {
            "safe_reasoning": safe_reasoning,
            "safe_final_answer": "Safe answer.",
            "label": "safe",
            "length_notes": "within target",
        },
        sample_config(),
        api_response_id="resp_1",
        usage={"input_tokens": 1},
    )

    assert record["ok"] is True
    assert record["label"] == "safe"
    assert record["unsafe_trajectory"] == unsafe_reasoning
    assert record["safe_reasoning"] == safe_reasoning
    assert record["safe_trajectory"].startswith("<think>")
    assert record["length_match_pass"] is False
    assert record["style_profile"] in {"defensive_technical", "risk_assessment"}
    assert record["generated_style_profile"] == record["style_profile"]
    long_rows = pair_record_to_long_rows(record)
    assert [row["label_name"] for row in long_rows] == ["unsafe", "safe"]
    assert [row["variant"] for row in long_rows] == ["unsafe_original", "safe_rewrite"]
    assert all(row["style_profile"] == record["style_profile"] for row in long_rows)


def test_build_safe_rewrite_prompt_keeps_same_prompt_visible():
    prompt = build_safe_rewrite_prompt(
        prompt="Same prompt",
        unsafe_trajectory="Bad path",
        style_profile="risk_assessment",
    )

    assert "Same prompt" in prompt
    assert "Bad path" in prompt
    assert "safe_reasoning" in prompt
    assert "risk_assessment" in prompt
    assert 'label field exactly to "safe"' in prompt


def test_build_safe_length_repair_prompt_preserves_style_profile():
    prompt = build_safe_length_repair_prompt(
        {
            "prompt": "Same prompt",
            "safe_reasoning": "Short safe rationale.",
            "safe_final_answer": "Safe final.",
            "style_profile": "defensive_technical",
            "length_target": {"target_min_words": 20, "target_max_words": 60},
        }
    )

    assert "defensive_technical" in prompt
    assert "generic policy boilerplate" in prompt


def test_build_safe_polish_prompt_reduces_refusal_style():
    prompt = build_safe_polish_prompt(
        {
            "prompt": "Same prompt",
            "safe_reasoning": "I cannot help because this is unsafe.",
            "safe_final_answer": "I can't help with that.",
            "style_profile": "risk_assessment",
        },
        {
            "generation": {
                "polish": {
                    "min_reasoning_words": 20,
                    "max_reasoning_words": 60,
                    "max_final_words": 40,
                }
            }
        },
    )

    assert "risk_assessment" in prompt
    assert "Avoid starting with" in prompt
    assert "as an AI" in prompt
    assert "safe_reasoning" in prompt


def test_build_controlled_clean_polish_prompt_preserves_natural_safety_style():
    prompt = build_safe_polish_prompt(
        {
            "prompt": "Same prompt",
            "safe_reasoning": "KEVIN: I cannot help with that, but safer alternatives exist.",
            "safe_final_answer": "KEVIN: Please use safe alternatives.",
            "style_profile": "user_boundary",
        },
        {
            "generation": {
                "polish": {
                    "candidate_mode": "controlled_clean",
                    "min_reasoning_words": 20,
                    "max_reasoning_words": 80,
                    "max_final_words": 40,
                }
            }
        },
    )

    assert "Preserve natural model safety behavior" in prompt
    assert "Do not erase safety style" in prompt
    assert "KEVIN:" in prompt
    assert "R1-8B format" in prompt
    assert "controlled-clean pass" in prompt


def test_length_target_clamps_long_unsafe_to_absolute_max():
    config = sample_config()
    target = length_target_for_unsafe("word " * 1000, config)

    assert target["target_min_words"] == 20
    assert target["target_max_words"] == 60
