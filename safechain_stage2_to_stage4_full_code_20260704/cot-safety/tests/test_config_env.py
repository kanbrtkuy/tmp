import os

from cot_safety.config import expand_env_string


def test_expand_env_supports_default_for_missing(monkeypatch):
    monkeypatch.delenv("COT_SAFETY_OUTPUT_ROOT", raising=False)

    value = expand_env_string("${COT_SAFETY_OUTPUT_ROOT:-/workspace/outputs}/run")

    assert value == "/workspace/outputs/run"


def test_expand_env_supports_default_for_empty(monkeypatch):
    monkeypatch.setenv("COT_SAFETY_OUTPUT_ROOT", "")

    value = expand_env_string("${COT_SAFETY_OUTPUT_ROOT:-/workspace/outputs}/run")

    assert value == "/workspace/outputs/run"


def test_expand_env_uses_present_value(monkeypatch):
    monkeypatch.setenv("COT_SAFETY_OUTPUT_ROOT", "/dev/shm/cot-safety-hot/outputs")

    value = expand_env_string("${COT_SAFETY_OUTPUT_ROOT:-/workspace/outputs}/run")

    assert value == "/dev/shm/cot-safety-hot/outputs/run"


def test_expand_env_preserves_unknown_without_default(monkeypatch):
    monkeypatch.delenv("UNKNOWN_COT_SAFETY_ROOT", raising=False)

    value = expand_env_string("${UNKNOWN_COT_SAFETY_ROOT}/run")

    assert value == "${UNKNOWN_COT_SAFETY_ROOT}/run"
