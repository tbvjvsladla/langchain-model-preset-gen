"""injection 순수 함수 — 6칸 한 칸 → SDK 인자 extras, 원문 HTTP 평탄화, 경로 병합."""

from __future__ import annotations

import pytest

from langchain_model_preset_gen import injection as inj

TABLE = {
    "ct_enable_thinking": {
        "disable": {"extra_body.chat_template_kwargs.enable_thinking": False},
        "values": {
            "enable": {"extra_body.chat_template_kwargs.enable_thinking": True},
            "low_effort": {"extra_body.chat_template_kwargs.enable_thinking": True,
                           "extra_body.chat_template_kwargs.low_effort": True},
        },
    },
    "or_reasoning": {
        "disable": {"extra_body.reasoning.enabled": False},
        "values": {"enable": {"extra_body.reasoning.enabled": True}},
        "value_path": "extra_body.reasoning.effort",
    },
    "openai_native": {"disable": {"reasoning_effort": "none"}, "value_path": "reasoning_effort"},
    "gemini_level": {"value_path": "thinking_level"},
    "none": {},
}


def test_normalize_slot_scalar_shorthand():
    assert inj.normalize_slot("low") == {"action": "set", "value": "low"}


def test_normalize_slot_dict_is_copy():
    slot = {"action": "omit"}
    out = inj.normalize_slot(slot)
    assert out == slot and out is not slot


def test_expand_slot_ct_disable_nested():
    assert inj.expand_slot(TABLE, "ct_enable_thinking", {"action": "disable"}) == \
        {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}


def test_expand_slot_ct_low_effort_merges_two_paths():
    assert inj.expand_slot(TABLE, "ct_enable_thinking", {"action": "set", "value": "low_effort"}) == \
        {"extra_body": {"chat_template_kwargs": {"enable_thinking": True, "low_effort": True}}}


def test_expand_slot_ct_scalar_shorthand_enable():
    assert inj.expand_slot(TABLE, "ct_enable_thinking", "enable") == \
        {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}


def test_expand_slot_ct_set_unknown_value_without_value_path_raises():
    with pytest.raises(KeyError):
        inj.expand_slot(TABLE, "ct_enable_thinking", {"action": "set", "value": "turbo"})


def test_expand_slot_or_reasoning_values_and_value_path():
    assert inj.expand_slot(TABLE, "or_reasoning", "enable") == {"extra_body": {"reasoning": {"enabled": True}}}
    assert inj.expand_slot(TABLE, "or_reasoning", "high") == {"extra_body": {"reasoning": {"effort": "high"}}}
    assert inj.expand_slot(TABLE, "or_reasoning", {"action": "disable"}) == \
        {"extra_body": {"reasoning": {"enabled": False}}}


def test_expand_slot_openai_native_top_level():
    assert inj.expand_slot(TABLE, "openai_native", {"action": "disable"}) == {"reasoning_effort": "none"}
    assert inj.expand_slot(TABLE, "openai_native", "low") == {"reasoning_effort": "low"}


def test_expand_slot_omit_is_empty_for_any_injection():
    for inj_id in TABLE:
        assert inj.expand_slot(TABLE, inj_id, {"action": "omit"}) == {}


def test_expand_slot_unknown_injection_raises_keyerror():
    with pytest.raises(KeyError):
        inj.expand_slot(TABLE, "xb_magic", "low")


def test_expand_slot_unknown_action_raises_valueerror():
    with pytest.raises(ValueError, match="action"):
        inj.expand_slot(TABLE, "or_reasoning", {"action": "boost"})


def test_expand_slot_disable_undeclared_raises_keyerror():
    with pytest.raises(KeyError):
        inj.expand_slot(TABLE, "gemini_level", {"action": "disable"})


def test_expand_slot_set_on_none_injection_raises_keyerror():
    with pytest.raises(KeyError):
        inj.expand_slot(TABLE, "none", "low")


def test_to_http_body_flattens_extra_body_and_keeps_top_level():
    extras = {"reasoning_effort": "low", "extra_body": {"reasoning": {"effort": "high"}, "top_k": 20}}
    assert inj.to_http_body(extras) == {"reasoning_effort": "low", "reasoning": {"effort": "high"}, "top_k": 20}
    assert inj.to_http_body({}) == {}


def test_to_http_body_collision_raises():
    with pytest.raises(ValueError, match="충돌"):
        inj.to_http_body({"top_k": 1, "extra_body": {"top_k": 2}})


def test_expand_paths_duplicate_leaf_raises():
    with pytest.raises(ValueError, match="충돌"):
        inj.expand_paths({"x.y": 1, "x": 2})


def test_deep_merge_nested_and_conflict():
    assert inj.deep_merge({"extra_body": {"a": 1}}, {"extra_body": {"b": 2}}) == {"extra_body": {"a": 1, "b": 2}}
    with pytest.raises(ValueError, match="충돌"):
        inj.deep_merge({"extra_body": {"a": 1}}, {"extra_body": {"a": 2}})
