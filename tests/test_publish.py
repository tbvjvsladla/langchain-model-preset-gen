"""publish — 발행 형식·preset 검증·configs 구동(profiles_dir·overridable 기본값 펼침)·provider 해소."""

from __future__ import annotations

import copy
import datetime as dt
import importlib.metadata as md
import subprocess
import sys

import pytest
import yaml

from conftest import FULL_OVERRIDABLE, PKG_ROOT, SITE_BASE_URL, make_skill
from langchain_model_preset_gen import catalog
from langchain_model_preset_gen import publish as pb

TODAY = dt.date(2026, 10, 15)
SIX = ["none", "low", "medium", "high", "xhigh", "max"]
LLM_OVERRIDABLE = list(FULL_OVERRIDABLE)
LLM_TOP_KEYS = ["kind", "model_name", "serving_provider", "provider_class", "base_url", "api_key_env",
                "preset", "overridable", "reasoning_effort", "sampling", "preset_values", "limits", "probe"]
EMB_TOP_KEYS = ["kind", "model_name", "serving_provider", "provider_class", "base_url", "api_key_env",
                "preset", "overridable", "dimensions", "preset_values", "limits", "supports", "probe"]

GRANITE_RULE = {
    "match": "granite-4.2-8b", "provider_class": "openai_compatible", "injection": "ct_enable_thinking",
    "mapping": {"none": {"action": "disable"}, "low": {"action": "set", "value": "low_effort"},
                "medium": {"action": "set", "value": "enable"}, "high": {"action": "set", "value": "enable"},
                "xhigh": {"action": "set", "value": "enable"}, "max": {"action": "set", "value": "enable"}},
    "end_marker": "</think>",
    "sampling": {"temperature": "accepted", "top_p": "accepted", "top_k": "accepted"},
    "basis": "모델카드",
}
JUDGE = {"name": "JUDGE", "intent": "balanced", "sampling": {"temperature": 0.0, "top_p": None, "top_k": None},
         "effort": "high", "limits": {"timeout_seconds": 300, "max_retries": 2, "response_deadline_seconds": None,
                                      "max_tokens": 32768}}


def gen(skill, model="granite-4.2-8b", preset="RAG", sp="site_vllm", kind="llm", **kw):
    return pb.generate(model, preset, sp, kind, skill_dir=skill, today=TODAY, **kw)


# ── 파일명 · 배타 생성 · profiles_dir ───────────────────────────────────────

def test_sanitize_and_filename():
    assert pb.sanitize("moonshotai/kimi-k2.6") == "moonshotai_kimi-k2.6"
    assert pb.derive_filename("deepseek/deepseek-v4.1-flash", "RAG", "openrouter") == \
        "deepseek_deepseek-v4.1-flash_RAG_openrouter.yaml"


def test_profiles_dir_from_config_is_publish_target(tmp_path):
    skill = make_skill(tmp_path / "proj", profiles_dir="out/profiles")
    path, _ = gen(skill)
    assert path == (tmp_path / "proj" / "out" / "profiles" / "granite-4.2-8b_RAG_site_vllm.yaml").resolve()
    assert path.exists()


def test_generate_duplicate_filename_refused(skill):
    path, _ = gen(skill)
    original = path.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        gen(skill)
    assert path.read_text(encoding="utf-8") == original


def test_generate_dry_run_writes_nothing(skill):
    path, _ = gen(skill, write=False)
    assert not path.exists()


# ── preset ───────────────────────────────────────────────────────────────

def test_llm_preset_effort_null_raises(tmp_path):
    skill = make_skill(tmp_path / "p", presets={"RAG": {**yaml.safe_load((PKG_ROOT / "presets/RAG.yaml").read_text()),
                                                         "effort": None}})
    with pytest.raises(ValueError, match="effort"):
        gen(skill)


@pytest.mark.parametrize("mutate,match", [
    (lambda p: p["limits"].pop("max_retries"), "limits"),
    (lambda p: p.update(intent="fun"), "intent"),
    (lambda p: p["sampling"].pop("top_k"), "sampling"),
])
def test_llm_preset_vocabulary_checked(tmp_path, mutate, match):
    bad = yaml.safe_load((PKG_ROOT / "presets/RAG.yaml").read_text(encoding="utf-8"))
    mutate(bad)
    with pytest.raises(ValueError, match=match):
        gen(make_skill(tmp_path / "p", presets={"RAG": bad}))


def test_llm_with_embedding_preset_raises(skill):
    with pytest.raises((ValueError, KeyError)):
        gen(skill, preset="RETRIEVAL")


def test_preset_values_and_limits_copied(tmp_path):
    _, prof = gen(make_skill(tmp_path / "p", presets={"JUDGE": JUDGE}), preset="JUDGE")
    assert prof["preset_values"] == {"temperature": 0.0, "top_p": None, "top_k": None, "effort": "high"}
    assert prof["limits"] == JUDGE["limits"]


# ── configs overridable: 기본값 펼침 · 키 목록 ────────────────────────────────

def test_overridable_defaults_overlay_preset_values(tmp_path):
    skill = make_skill(tmp_path / "p", overridable={**FULL_OVERRIDABLE, "temperature": 0.3, "effort": "low",
                                                    "timeout_seconds": 45})
    _, prof = gen(skill)
    assert prof["preset_values"] == {"temperature": 0.3, "top_p": 0.9, "top_k": 20, "effort": "low"}
    assert prof["limits"]["timeout_seconds"] == 45 and prof["limits"]["max_tokens"] == 8192


def test_overridable_key_list_locks_missing_keys(tmp_path):
    skill = make_skill(tmp_path / "p", overridable={"effort": None, "max_tokens": None})
    _, prof = gen(skill)
    assert prof["overridable"] == ["effort", "max_tokens"]
    _, emb = gen(skill, model="qwen/qwen3-embedding-4b", preset="RETRIEVAL", sp="openrouter", kind="embedding")
    assert emb["overridable"] == []


def test_overridable_effort_default_vocabulary_checked(tmp_path):
    skill = make_skill(tmp_path / "p", overridable={**FULL_OVERRIDABLE, "effort": "turbo"})
    with pytest.raises(ValueError, match="effort"):
        gen(skill)


def test_overridable_embedding_defaults_and_unsupported_removed(tmp_path):
    skill = make_skill(tmp_path / "p", overridable={**FULL_OVERRIDABLE, "timeout_seconds": 30})
    _, prof = gen(skill, model="gemini-embedding-2", preset="RETRIEVAL", sp="google", kind="embedding")
    assert prof["limits"] == {"timeout_seconds": 30, "max_retries": 2}
    assert prof["overridable"] == ["timeout_seconds"], "google 임베딩 SDK 에 max_retries 없음 → 목록에서 제외"


# ── 모델 규칙 · provider ─────────────────────────────────────────────────────

def test_model_rules_longest_prefix_used_by_publish(tmp_path):
    broad = {**copy.deepcopy(GRANITE_RULE), "match": "granite", "basis": "broad"}
    skill = make_skill(tmp_path / "p", rules=[broad, GRANITE_RULE])
    _, prof = gen(skill)
    assert prof["reasoning_effort"]["basis"] == "모델카드"


def test_model_rules_unregistered_raises_and_writes_nothing(skill):
    with pytest.raises(KeyError, match="model_rules"):
        gen(skill, model="mystery-x")
    assert not (skill.parents[2] / "profiles" / "mystery-x_RAG_site_vllm.yaml").exists()


def test_unknown_serving_provider_raises(skill):
    with pytest.raises(ValueError, match="serving_provider"):
        gen(skill, sp="no-such")


def test_site_provider_shadowing_official_name_raises(tmp_path):
    skill = make_skill(tmp_path / "p", providers={"openrouter": {"provider_class": "openai_compatible",
                                                                 "base_url": "http://x/v1", "api_key_env": None}})
    with pytest.raises(ValueError, match="공식"):
        gen(skill, model="deepseek/deepseek-v4.1-flash", sp="openrouter")


def test_rule_not_mutated_by_publish(tmp_path):
    shorthand = {**copy.deepcopy(GRANITE_RULE), "mapping": {**GRANITE_RULE["mapping"], "medium": "enable"}}
    skill = make_skill(tmp_path / "p", rules=[shorthand])
    _, prof = gen(skill)
    assert prof["reasoning_effort"]["mapping"]["medium"] == {"action": "set", "value": "enable"}
    assert catalog.load_model_rules(skill)[0]["mapping"]["medium"] == "enable"


# ── LLM 발행 형식 ───────────────────────────────────────────────────────────

def test_llm_profile_key_order_and_values(skill):
    path, prof = gen(skill)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert list(loaded) == LLM_TOP_KEYS and loaded == prof
    assert loaded["base_url"] == SITE_BASE_URL and loaded["api_key_env"] is None
    assert loaded["overridable"] == LLM_OVERRIDABLE
    re_ = loaded["reasoning_effort"]
    assert list(re_) == ["injection", "mapping", "end_marker", "basis"]
    assert re_["injection"] == "ct_enable_thinking" and re_["mapping"] == GRANITE_RULE["mapping"]
    assert loaded["preset_values"] == {"temperature": 0.1, "top_p": 0.9, "top_k": 20, "effort": "medium"}
    assert list(loaded["limits"]) == ["timeout_seconds", "max_retries", "response_deadline_seconds", "max_tokens"]


def test_llm_probe_static_initial(skill):
    _, prof = gen(skill)
    assert prof["probe"] == {"status": "probe_error", "method": "static", "measured_at": None, "terminal": None,
                             "runnable": {m: "probe_error" for m in
                                          ["invoke", "ainvoke", "stream", "astream", "batch", "abatch"]}}


def test_header_two_lines(skill):
    path, _ = gen(skill)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("# langchain-model-preset-gen 발행")
    assert lines[1] == "# preset 출처: RAG (스킬 presets/, 발행일 2026-10-15) · 실측: 미측정"
    assert lines[2] == "kind: llm"


# ── 임베딩 발행 ─────────────────────────────────────────────────────────────

def test_embedding_profile_shape(skill):
    path, prof = gen(skill, model="qwen/qwen3-embedding-4b", preset="RETRIEVAL", sp="openrouter", kind="embedding")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert list(loaded) == EMB_TOP_KEYS and loaded == prof
    assert loaded["overridable"] == ["timeout_seconds", "max_retries"] and loaded["dimensions"] is None
    assert loaded["preset_values"] == {"intent": "retrieval", "modality": "text"}
    sup = loaded["supports"]
    assert sup["query_mode_surface"] == "instruct_prefix" and sup["normalization"] is None
    assert sup["native_dimensions"] == 2560 and "document_template" not in sup
    probe = loaded["probe"]
    assert probe["status"] == "probe_error" and probe["method"] == "static_v5"
    assert probe["measured_with"] == f"langchain-openai=={md.version('langchain-openai')}"
    assert probe["requires"] == "langchain-openai>=1.0,<2.0"


def test_embedding_gemini_document_template_carried(skill):
    _, prof = gen(skill, model="gemini-embedding-2", preset="RETRIEVAL", sp="google", kind="embedding")
    assert prof["supports"]["document_template"] == "title: none | text: {text}"


@pytest.mark.parametrize("dims,match", [(2560, "inert"), (4096, "native")])
def test_embedding_dimensions_guard(skill, dims, match):
    with pytest.raises(ValueError, match=match):
        gen(skill, model="qwen/qwen3-embedding-4b", preset="RETRIEVAL", sp="openrouter", kind="embedding",
            dimensions=dims)


def test_embedding_dimensions_valid_and_llm_rejects_dimensions(skill):
    _, prof = gen(skill, model="qwen/qwen3-embedding-4b", preset="RETRIEVAL", sp="openrouter", kind="embedding",
                  dimensions=1024)
    assert prof["dimensions"] == 1024
    with pytest.raises(ValueError, match="dimensions"):
        gen(skill, dimensions=1024)


def test_embedding_unregistered_model_raises(skill):
    with pytest.raises(KeyError, match="embedding_capability"):
        gen(skill, model="mystery-emb", preset="RETRIEVAL", sp="openrouter", kind="embedding")


def test_requires_range_outside_installed_raises():
    classes = catalog.load_data(PKG_ROOT)["provider_classes"]
    stale = {"openai": {**classes["openai"], "requires_range": ">=0.1,<0.2"}}
    with pytest.raises(ValueError, match="requires_range"):
        pb.requires_range("openai", stale)


# ── CLI ────────────────────────────────────────────────────────────────

def test_cli_dry_run_uses_skill_dir_config(skill):
    r = subprocess.run([sys.executable, str(PKG_ROOT / "scripts" / "publish.py"), "--model-name", "granite-4.2-8b",
                        "--preset", "RAG", "--provider", "site_vllm", "--kind", "llm", "--skill-dir", str(skill),
                        "--dry-run"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "granite-4.2-8b_RAG_site_vllm.yaml" in r.stdout and "reasoning_effort:" in r.stdout
    assert not (skill.parents[2] / "profiles").exists() or not any((skill.parents[2] / "profiles").iterdir())


def test_cli_has_no_env_file_or_max_len_options():
    for script in ("publish.py", "preflight.py", "probe.py"):
        help_text = subprocess.run([sys.executable, str(PKG_ROOT / "scripts" / script), "--help"],
                                   capture_output=True, text=True, check=True).stdout
        assert "--env-file" not in help_text and "--max-len" not in help_text, script
