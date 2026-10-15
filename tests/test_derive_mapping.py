"""derive_mapping — OpenRouter /models 메타 → 모델 규칙 6칸 초안의 결정론 회귀 가드.

fixture = OpenRouter /models 응답(2026-10-15 조회)에서 해당 항목의 `id`·`reasoning`·`supported_parameters` 를
그대로 복사한 것이다(네트워크 없음).
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
import yaml

from conftest import PKG_ROOT
from langchain_model_preset_gen import catalog
from langchain_model_preset_gen import derive_mapping as d

SIX = ["none", "low", "medium", "high", "xhigh", "max"]
VOCAB = {
    "effort_levels": SIX,
    "slot_actions": ["set", "disable", "omit"],
    "sampling_status": ["accepted", "rejected", "ignored"],
    "sampling_keys": ["temperature", "top_p", "top_k"],
}

# ── or_models.json 실항목 복사 (id·reasoning·supported_parameters) ──
OR_MODELS = {"data": [
    {"id": "deepseek/deepseek-v4.1-flash",
     "reasoning": {"mandatory": False, "default_enabled": True, "supported_efforts": ["max", "high", "low"],
                   "default_effort": "high"},
     "supported_parameters": ["frequency_penalty", "include_reasoning", "logit_bias", "logprobs", "max_tokens",
                              "min_p", "presence_penalty", "reasoning", "reasoning_effort",
                              "repetition_penalty", "response_format", "seed", "stop", "structured_outputs",
                              "temperature", "tool_choice", "tools", "top_k", "top_logprobs", "top_p"]},
    {"id": "deepseek/deepseek-v4-pro",
     "reasoning": {"mandatory": False, "supported_efforts": ["xhigh", "high"], "default_effort": "high"},
     "supported_parameters": ["frequency_penalty", "include_reasoning", "logit_bias", "logprobs",
                              "max_completion_tokens", "max_tokens", "min_p", "presence_penalty", "reasoning",
                              "reasoning_effort", "repetition_penalty", "response_format", "seed", "stop",
                              "structured_outputs", "temperature", "tool_choice", "tools", "top_k",
                              "top_logprobs", "top_p"]},
    {"id": "openai/gpt-oss-20b",
     "reasoning": {"mandatory": True, "supported_efforts": ["high", "medium", "low"], "default_effort": "medium"},
     "supported_parameters": ["frequency_penalty", "include_reasoning", "logit_bias", "logprobs", "max_tokens",
                              "min_p", "presence_penalty", "reasoning", "reasoning_effort",
                              "repetition_penalty", "response_format", "seed", "stop", "structured_outputs",
                              "temperature", "tool_choice", "tools", "top_k", "top_logprobs", "top_p"]},
    {"id": "moonshotai/kimi-k2.7-code",
     "reasoning": {"mandatory": True, "default_enabled": True},
     "supported_parameters": ["frequency_penalty", "include_reasoning", "logit_bias", "logprobs", "max_tokens",
                              "min_p", "parallel_tool_calls", "presence_penalty", "reasoning",
                              "repetition_penalty", "response_format", "seed", "stop", "structured_outputs",
                              "temperature", "tool_choice", "tools", "top_k", "top_logprobs", "top_p"]},
    {"id": "google/gemma-3-12b-it",
     "supported_parameters": ["frequency_penalty", "logit_bias", "max_tokens", "min_p", "presence_penalty",
                              "repetition_penalty", "response_format", "seed", "stop", "structured_outputs",
                              "temperature", "tool_choice", "tools", "top_k", "top_p"]},
    {"id": "xiaomi/mimo-v2.6-flash",
     "reasoning": {"mandatory": False},
     "supported_parameters": ["frequency_penalty", "include_reasoning", "logit_bias", "logprobs", "max_tokens",
                              "min_p", "presence_penalty", "reasoning", "repetition_penalty",
                              "response_format", "seed", "stop", "structured_outputs", "temperature",
                              "tool_choice", "tools", "top_k", "top_logprobs", "top_p"]},
    {"id": "qwen/qwen3.8-max-prime",
     "reasoning": {"mandatory": True, "default_enabled": True,
                   "supported_efforts": ["xhigh", "high", "medium", "low", "minimal"], "default_effort": "xhigh"},
     "supported_parameters": ["frequency_penalty", "include_reasoning", "logprobs", "max_tokens",
                              "presence_penalty", "reasoning", "reasoning_effort", "response_format", "seed",
                              "stop", "structured_outputs", "temperature", "tool_choice", "tools", "top_k",
                              "top_logprobs", "top_p"]},
    {"id": "mistralai/mistral-large-4-0",
     "reasoning": {"mandatory": False, "default_enabled": True, "supported_efforts": ["high", "none"],
                   "default_effort": "high"},
     "supported_parameters": ["frequency_penalty", "include_reasoning", "logprobs", "max_tokens",
                              "presence_penalty", "reasoning", "reasoning_effort", "response_format", "seed",
                              "stop", "structured_outputs", "temperature", "tool_choice", "tools",
                              "top_logprobs", "top_p"]},
    {"id": "openai/gpt-6-luna",
     "reasoning": {"mandatory": False, "default_enabled": True,
                   "supported_efforts": ["max", "xhigh", "high", "medium", "low", "none"],
                   "default_effort": "medium"},
     "supported_parameters": ["include_reasoning", "max_completion_tokens", "max_tokens", "reasoning",
                              "reasoning_effort", "response_format", "seed", "structured_outputs",
                              "tool_choice", "tools", "verbosity"]},
]}

SET = lambda v: {"action": "set", "value": v}  # noqa: E731
DISABLE = {"action": "disable"}
OMIT = {"action": "omit"}


def _model(mid: str) -> dict:
    return d.find_model(OR_MODELS, mid)


def _mapping(mid: str) -> dict:
    return d.derive_rule(_model(mid), VOCAB, source="fixture")["mapping"]


def test_effort_order_inserts_minimal_after_none():
    assert d.effort_order(VOCAB) == ["none", "minimal", "low", "medium", "high", "xhigh", "max"]


def test_deepseek_three_levels_round_up_and_max_reserved():
    """low/high/max 3단계 — medium→high(올림), xhigh→high(max 로 올리지 않음, 인터뷰 DeepSeek 예)."""
    assert _mapping("deepseek/deepseek-v4.1-flash") == {
        "none": DISABLE, "low": SET("low"), "medium": SET("high"), "high": SET("high"),
        "xhigh": SET("high"), "max": SET("max")}


def test_deepseek_pro_cap_at_highest():
    """xhigh/high 2단계 — low·medium 올림 → high, max 는 최고 지원(xhigh)으로 상한."""
    assert _mapping("deepseek/deepseek-v4-pro") == {
        "none": DISABLE, "low": SET("high"), "medium": SET("high"), "high": SET("high"),
        "xhigh": SET("xhigh"), "max": SET("xhigh")}


def test_gpt_oss_mandatory_none_lowest():
    assert _mapping("openai/gpt-oss-20b") == {
        "none": SET("low"), "low": SET("low"), "medium": SET("medium"), "high": SET("high"),
        "xhigh": SET("high"), "max": SET("high")}


def test_kimi_mandatory_no_efforts_all_omit():
    assert _mapping("moonshotai/kimi-k2.7-code") == {s: OMIT for s in SIX}


def test_gemma3_no_reasoning_metadata_all_omit():
    assert _mapping("google/gemma-3-12b-it") == {s: OMIT for s in SIX}


def test_mimo_toggle_disable_and_enable():
    assert _mapping("xiaomi/mimo-v2.6-flash") == {"none": DISABLE} | {s: SET("enable") for s in SIX[1:]}


def test_mandatory_with_minimal_none_maps_to_minimal():
    m = _mapping("qwen/qwen3.8-max-prime")
    assert m["none"] == SET("minimal")
    assert m["low"] == SET("low") and m["max"] == SET("xhigh")


def test_none_in_supported_efforts_disables():
    m = _mapping("mistralai/mistral-large-4-0")
    assert m["none"] == DISABLE
    assert all(m[s] == SET("high") for s in SIX[1:])


def test_full_six_identity():
    assert _mapping("openai/gpt-6-luna") == {"none": DISABLE} | {s: SET(s) for s in SIX[1:]}


def test_unknown_effort_level_raises():
    model = {"id": "x/y", "reasoning": {"mandatory": False, "supported_efforts": ["ultra"]},
             "supported_parameters": []}
    with pytest.raises(ValueError, match="ultra"):
        d.derive_rule(model, VOCAB, source="fixture")


def test_find_model_unknown_raises():
    with pytest.raises(ValueError, match="no/such"):
        d.find_model(OR_MODELS, "no/such")


def test_rule_shape_and_sampling():
    rule = d.derive_rule(_model("openai/gpt-6-luna"), VOCAB, source="fixture")
    assert list(rule) == ["match", "provider_class", "injection", "mapping", "end_marker", "sampling", "basis"]
    assert rule["match"] == "openai/gpt-6-luna"
    assert rule["provider_class"] == "openai_compatible" and rule["injection"] == "or_reasoning"
    assert rule["end_marker"] is None
    assert rule["sampling"] == {"temperature": "rejected", "top_p": "rejected", "top_k": "rejected"}
    assert list(rule["mapping"]) == SIX
    assert "fixture" in rule["basis"] and "supported_efforts" in rule["basis"]


def test_sampling_from_supported_parameters():
    rule = d.derive_rule(_model("mistralai/mistral-large-4-0"), VOCAB, source="fixture")
    assert rule["sampling"] == {"temperature": "accepted", "top_p": "accepted", "top_k": "rejected"}


def test_render_rule_is_model_rules_file_content(tmp_path):
    rule = d.derive_rule(_model("deepseek/deepseek-v4.1-flash"), VOCAB, source="fixture")
    loaded = yaml.safe_load(d.render_rule(rule))
    assert loaded == rule
    (tmp_path / "model_rules").mkdir()
    (tmp_path / "model_rules" / "draft.yaml").write_text(d.render_rule(rule), encoding="utf-8")
    assert catalog.load_model_rules(tmp_path) == [rule], "초안은 model_rules/ 파일 형식 그대로다"


def test_draft_rule_validates_against_real_vocabulary():
    data = catalog.load_data(PKG_ROOT)
    for mid in ("deepseek/deepseek-v4.1-flash", "openai/gpt-oss-20b", "moonshotai/kimi-k2.7-code"):
        catalog.validate_model_rule(d.derive_rule(_model(mid), data["vocabulary"], "fixture"),
                                    data["vocabulary"], data["injection"])


def test_cli_from_cached_json(tmp_path):
    models = tmp_path / "or_models.json"
    models.write_text(json.dumps(OR_MODELS), encoding="utf-8")
    r = subprocess.run([sys.executable, str(PKG_ROOT / "scripts" / "derive_mapping.py"), "--model-id",
                        "openai/gpt-oss-20b", "--models-json", str(models)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rule = yaml.safe_load(r.stdout)
    assert rule["mapping"]["none"] == SET("low")
    assert str(models) in rule["basis"]
