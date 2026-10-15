"""catalog — 모델 규칙 최장 접두어 매칭·provider 병합·규칙 검증, 배포본 data/·model_rules/ 정합."""

from __future__ import annotations

import copy

import pytest

from conftest import PKG_ROOT
from langchain_model_preset_gen import catalog

SIX = ["none", "low", "medium", "high", "xhigh", "max"]
DATA = catalog.load_data(PKG_ROOT)
VOCAB = DATA["vocabulary"]
INJECTION = DATA["injection"]
REAL_RULES = catalog.load_model_rules(PKG_ROOT)


def rule(match: str, provider_class: str = "openai_compatible", basis: str = "t", **over) -> dict:
    body = {"match": match, "provider_class": provider_class, "injection": "ct_enable_thinking",
            "mapping": {"none": {"action": "disable"}, "low": {"action": "set", "value": "low_effort"},
                        "medium": "enable", "high": "enable", "xhigh": "enable", "max": "enable"},
            "end_marker": "</think>", "sampling": {"temperature": "accepted", "top_p": "accepted", "top_k": "accepted"},
            "basis": basis}
    body.update(over)
    return body


# ── 최장 접두어 매칭 ─────────────────────────────────────────────────────────

def test_longest_prefix_wins_regardless_of_order():
    broad, narrow = rule("granite", basis="broad"), rule("granite-4.2-8b", basis="narrow")
    for rules in ([broad, narrow], [narrow, broad]):
        assert catalog.find_model_rule("Granite-4.2-8B-instruct", "openai_compatible", rules)["basis"] == "narrow"
    assert catalog.find_model_rule("granite-3", "openai_compatible", [narrow, broad])["basis"] == "broad"


def test_longest_prefix_equal_length_ambiguous_raises():
    with pytest.raises(ValueError, match="모호"):
        catalog.find_model_rule("qwen3-4b", "openai_compatible", [rule("qwen3-4", basis="a"), rule("QWEN3-4", basis="b")])


def test_model_rules_provider_class_gate():
    with pytest.raises(KeyError, match="model_rules"):
        catalog.find_model_rule("granite-4.2-8b", "openai", [rule("granite-4.2-8b")])


def test_model_rules_unregistered_raises_keyerror():
    with pytest.raises(KeyError, match="model_rules 미등록"):
        catalog.find_model_rule("mystery-x", "openai_compatible", REAL_RULES)


def test_model_rules_real_flash_variant_prefers_longer_match():
    """z-ai/glm-5.3 와 z-ai/glm-5.3-flash 가 함께 있어도 flash 모델은 flash 규칙을 받는다(파일 순서 무관)."""
    assert catalog.find_model_rule("z-ai/glm-5.3-flash", "openai_compatible", REAL_RULES)["match"] == "z-ai/glm-5.3-flash"
    assert catalog.find_model_rule("z-ai/glm-5.3", "openai_compatible", REAL_RULES)["match"] == "z-ai/glm-5.3"


def test_model_rules_file_keys_exact(tmp_path):
    (tmp_path / "model_rules").mkdir()
    (tmp_path / "model_rules" / "x.yaml").write_text("match: a\nprovider_class: openai\n", encoding="utf-8")
    with pytest.raises(ValueError, match="키 불일치"):
        catalog.load_model_rules(tmp_path)


def test_model_rules_folder_one_rule_per_file_and_all_valid():
    files = sorted(p.name for p in (PKG_ROOT / "model_rules").glob("*.yaml"))
    assert len(files) == len(REAL_RULES) >= 2
    for r in REAL_RULES:
        catalog.validate_model_rule(r, VOCAB, INJECTION)
    keys = [(r["match"].lower(), r["provider_class"]) for r in REAL_RULES]
    assert len(keys) == len(set(keys)), "같은 match·provider_class 규칙이 둘 이상"


def test_embedding_capability_longest_prefix():
    cap = catalog.find_embedding_capability("gemini-embedding-2", "google", DATA)
    assert cap["query_mode_surface"] == "task_title_prefix" and cap["native_dimensions"] == 3072
    with pytest.raises(KeyError, match="embedding_capability"):
        catalog.find_embedding_capability("mystery-emb", "openai_compatible", DATA)


# ── provider 병합 ───────────────────────────────────────────────────────────

def test_merge_providers_site_name_clash_raises():
    with pytest.raises(ValueError, match="공식"):
        catalog.merge_providers(DATA["providers"], {"openai": {"provider_class": "openai", "base_url": None,
                                                                "api_key_env": "X"}})


def test_merge_providers_entry_keys_checked():
    with pytest.raises(ValueError, match="키 위반"):
        catalog.merge_providers({}, {"s": {"provider_class": "openai_compatible", "base_url": "http://h/v1"}})


def test_data_providers_are_official_only():
    """사이트 엔드포인트(사설 주소·localhost)는 배포본 data 에 없다 — configs 몫."""
    for name, entry in DATA["providers"].items():
        assert entry["base_url"] is None or entry["base_url"].startswith("https://"), name
        assert entry["provider_class"] in DATA["provider_classes"]


# ── 규칙 검증 (발행·해석·정적 검사 공용) ───────────────────────────────────────

@pytest.mark.parametrize("mutate,match", [
    (lambda r: r["mapping"].pop("xhigh"), "6칸 누락"),
    (lambda r: r["mapping"].update(minimal={"action": "omit"}), "minimal"),
    (lambda r: r["mapping"].update(low={"action": "boost", "value": "x"}), "action"),
    (lambda r: r["mapping"].update(low={"action": "set"}), "value"),
    (lambda r: r["mapping"].update(none={"action": "disable", "value": "x"}), "value"),
    (lambda r: r["mapping"].update(high={"action": "set", "value": "turbo"}), "turbo"),
    (lambda r: r.update(injection="xb_magic"), "xb_magic"),
    (lambda r: r["sampling"].update(top_k="maybe"), "maybe"),
    (lambda r: r["sampling"].update(temperature={"by_effort": {"none": "accepted", "default": "maybe"}}), "maybe"),
    (lambda r: r["sampling"].update(temperature={"by_effort": {"ultra": "accepted", "default": "ignored"}}), "ultra"),
    (lambda r: r["sampling"].update(temperature={"by_effort": {"none": "accepted"}}), "default"),
    (lambda r: r["sampling"].pop("top_k"), "top_k"),
])
def test_model_rules_validation_rejects(mutate, match):
    bad = copy.deepcopy(rule("granite-4.2-8b"))
    mutate(bad)
    with pytest.raises(ValueError, match=match):
        catalog.validate_model_rule(bad, VOCAB, INJECTION)


def test_model_rules_disable_on_injection_without_disable_raises():
    bad = rule("gpt-x", "openai", injection="gemini_level", mapping={lv: "low" for lv in SIX} | {"none": {"action": "disable"}})
    with pytest.raises(ValueError, match="disable"):
        catalog.validate_model_rule(bad, VOCAB, INJECTION)


def test_model_rules_shorthand_normalized_and_original_untouched():
    r = rule("deepseek/x", injection="or_reasoning",
             mapping={"none": {"action": "disable"}, "low": "low", "medium": "high", "high": "high", "xhigh": "high",
                      "max": "max"})
    out = catalog.validate_model_rule(r, VOCAB, INJECTION)
    assert out["mapping"]["low"] == {"action": "set", "value": "low"} and list(out["mapping"]) == SIX
    assert r["mapping"]["low"] == "low"


def test_sampling_status_by_effort():
    entry = {"by_effort": {"none": "accepted", "default": "ignored"}}
    assert catalog.sampling_status(entry, "none") == "accepted"
    assert catalog.sampling_status(entry, "high") == "ignored"
    assert catalog.sampling_status("rejected", "high") == "rejected"


def test_load_preset_name_must_match_stem(tmp_path):
    (tmp_path / "presets").mkdir()
    (tmp_path / "presets" / "RAG.yaml").write_text("name: OTHER\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stem"):
        catalog.load_preset("RAG", tmp_path)
    with pytest.raises(FileNotFoundError):
        catalog.load_preset("NOPE", tmp_path)
