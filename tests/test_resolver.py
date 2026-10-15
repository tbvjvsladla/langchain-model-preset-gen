"""resolver — 공개 해석기: 잠금(locked)·override 우선순위·SDK 인자·ChatModel/Embeddings 생성·임베딩 입력 형식·정규화.

SDK 생성자는 MagicMock(sdk_mock fixture)으로만 흉내 낸다(네트워크 없음). 프로필은 배포본 정본 data/·presets/·model_rules/ 로 발행한다.
"""

from __future__ import annotations

import asyncio
import copy
import math

import pytest

from conftest import SITE_BASE_URL, make_skill
from langchain_model_preset_gen import LockedKeyError, load_chat_model, load_embeddings, resolve
from langchain_model_preset_gen import publish as pb
from langchain_model_preset_gen import resolver as rs
from langchain_model_preset_gen.embeddings import ProfileEmbeddings


@pytest.fixture
def mk(skill):
    def build(model="granite-4.2-8b", sp="site_vllm", preset="RAG"):
        return pb.build_llm_profile(model, preset, sp, skill)
    return build


@pytest.fixture
def mk_emb(skill):
    def build(model="gemini-embedding-2", sp="google", dims=None, measured=None):
        prof = pb.build_embedding_profile(model, "RETRIEVAL", sp, dims, skill)
        prof["probe"]["measured_at_dimensions"] = measured
        return prof
    return build


# ── 잠금 (locked) ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("key", ["model_name", "base_url", "provider_class", "api_key_env", "reasoning_effort",
                                 "sampling", "probe"])
def test_resolver_locked_llm_keys_rejected(mk, key):
    with pytest.raises(LockedKeyError, match=key):
        resolve(mk(), {key: "x"})


@pytest.mark.parametrize("key", ["supports", "dimensions", "model_name", "base_url", "probe"])
def test_resolver_locked_embedding_keys_rejected(mk_emb, key):
    with pytest.raises(LockedKeyError, match=key):
        resolve(mk_emb(), {key: 1})


def test_resolver_locked_error_is_valueerror_and_unknown_key_rejected(mk):
    with pytest.raises(ValueError):
        resolve(mk(), {"temprature": 0.3})


def test_resolver_locked_when_profile_overridable_shrinks(mk):
    prof = mk()
    prof["overridable"].remove("effort")
    with pytest.raises(LockedKeyError, match="effort"):
        resolve(prof, {"effort": "high"})


def test_resolver_override_overridable_keys_accepted(mk):
    spec = resolve(mk(), {"effort": "high", "timeout_seconds": 7, "max_tokens": 64, "temperature": 0.5})
    assert spec["effective"]["effort"] == "high" and spec["effective"]["timeout_seconds"] == 7
    assert spec["sdk_kwargs"]["timeout"] == 7 and spec["sdk_kwargs"]["max_tokens"] == 64
    assert spec["sdk_kwargs"]["temperature"] == 0.5


def test_resolver_override_precedence_three_levels(tmp_path):
    """preset < configs overridable 기본값(발행 시 펼침) < Agent override."""
    skill = make_skill(tmp_path / "p", overridable={**{k: None for k in rs.LLM_OVERRIDABLE}, "temperature": 0.3})
    prof = pb.build_llm_profile("granite-4.2-8b", "RAG", "site_vllm", skill)
    assert resolve(prof)["effective"]["temperature"] == 0.3                # config 기본값 > preset 0.1
    assert resolve(prof, {"temperature": 0.7})["effective"]["temperature"] == 0.7
    assert resolve(prof)["effective"]["top_p"] == 0.9                      # 기본값 null → preset 값


def test_resolver_sampling_override_rejected_when_not_accepted_at_effort(mk):
    prof = mk("gpt-5.4-mini", "openai")                                    # temperature by_effort none: accepted
    with pytest.raises(ValueError, match="sampling override"):
        resolve(prof, {"temperature": 0.2})
    assert resolve(prof, {"effort": "none", "temperature": 0.2})["sdk_kwargs"]["temperature"] == 0.2


def test_resolver_effort_override_vocabulary(mk):
    with pytest.raises(ValueError, match="어휘"):
        resolve(mk(), {"effort": "turbo"})


def test_resolver_sent_nulls_unaccepted_sampling(mk):
    spec = resolve(mk("gpt-5.4-mini", "openai"))
    assert spec["sent"]["temperature"] is None and spec["sent"]["top_k"] is None
    assert spec["effective"]["temperature"] == 0.1 and spec["sent"]["effort"] == "medium"
    assert set(spec["sent"]) == set(rs.LLM_OVERRIDABLE)


def test_resolver_reads_profile_path(skill, tmp_path):
    path, _ = pb.generate("granite-4.2-8b", "RAG", "site_vllm", "llm", skill_dir=skill)
    spec = resolve(path)
    assert spec["name"] == path.name and spec["model_name"] == "granite-4.2-8b"


# ── SDK 인자 (provider 분기) ────────────────────────────────────────────────

def test_resolver_kwargs_openai_compatible_granite(mk):
    spec = resolve(mk())
    assert spec["sdk_class"] == "langchain_openai:ChatOpenAI"
    assert spec["sdk_kwargs"] == {"model": "granite-4.2-8b", "base_url": SITE_BASE_URL, "max_tokens": 8192,
                                  "timeout": 120, "max_retries": 2, "temperature": 0.1, "top_p": 0.9,
                                  "extra_body": {"chat_template_kwargs": {"enable_thinking": True}, "top_k": 20}}
    assert "api_key" not in spec["sdk_kwargs"], "키는 spec 에 싣지 않는다"


def test_resolver_kwargs_override_effort_changes_slot(mk):
    kw = resolve(mk(), {"effort": "low"})["sdk_kwargs"]
    assert kw["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True, "low_effort": True}


def test_resolver_kwargs_openai_native(mk):
    assert resolve(mk("gpt-5.4-mini", "openai"))["sdk_kwargs"] == {
        "model": "gpt-5.4-mini", "reasoning_effort": "medium", "max_tokens": 8192, "timeout": 120, "max_retries": 2}
    kw = resolve(mk("gpt-5.4-mini", "openai"), {"effort": "none"})["sdk_kwargs"]
    assert kw["reasoning_effort"] == "none" and kw["temperature"] == 0.1 and "extra_body" not in kw


def test_resolver_kwargs_anthropic(mk):
    spec = resolve(mk("claude-haiku-4-5-20251001", "anthropic"))
    assert spec["sdk_class"] == "langchain_anthropic:ChatAnthropic"
    assert spec["sdk_kwargs"]["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert "temperature" not in spec["sdk_kwargs"] and "base_url" not in spec["sdk_kwargs"]
    assert resolve(mk("claude-sonnet-5", "anthropic"))["sdk_kwargs"]["effort"] == "medium"
    off = resolve(mk("claude-sonnet-5", "anthropic"), {"effort": "none"})["sdk_kwargs"]
    assert off["thinking"] == {"type": "disabled"} and "effort" not in off
    opus = resolve(mk("claude-opus-4-8", "anthropic"))["sdk_kwargs"]
    assert opus["thinking"] == {"type": "adaptive"} and opus["effort"] == "medium"


def test_resolver_kwargs_google(mk):
    spec = resolve(mk("gemini-3.6-flash", "google"))
    assert spec["sdk_class"] == "langchain_google_genai:ChatGoogleGenerativeAI"
    assert spec["sdk_kwargs"] == {"model": "gemini-3.6-flash", "thinking_level": "medium", "max_output_tokens": 8192,
                                  "timeout": 120, "max_retries": 2}


def test_resolver_kwargs_omit_slot_sends_no_reasoning(mk):
    prof = mk()
    prof["reasoning_effort"] = {**prof["reasoning_effort"], "injection": "none",
                                "mapping": {lv: {"action": "omit"} for lv in prof["reasoning_effort"]["mapping"]}}
    assert resolve(prof)["sdk_kwargs"]["extra_body"] == {"top_k": 20}


def test_resolver_rejects_invalid_profile_mapping(mk):
    prof = mk()
    del prof["reasoning_effort"]["mapping"]["xhigh"]
    with pytest.raises(ValueError, match="6칸"):
        resolve(prof)


def test_resolver_http_base_url_native_fallback(mk):
    assert resolve(mk("gpt-5.4-mini", "openai"))["http_base_url"] == "https://api.openai.com/v1"
    prof = mk()
    prof["base_url"] = None
    with pytest.raises(ValueError, match="native_base_url"):
        resolve(prof)


# ── load_chat_model / client_kwargs ───────────────────────────────────────────

def _write_profile(skill, model="granite-4.2-8b", sp="site_vllm", kind="llm", preset="RAG"):
    path, _ = pb.generate(model, preset, sp, kind, skill_dir=skill)
    return path


def test_resolver_load_chat_model_passes_kwargs_and_no_auth_placeholder(skill, sdk_mock):
    SDK = sdk_mock("langchain_openai.ChatOpenAI")
    load_chat_model(_write_profile(skill), {"effort": "high"}, skill_dir=skill, client_kwargs={"stream_usage": True})
    kw = SDK.call_args.kwargs
    assert kw["api_key"] == rs.NO_AUTH_API_KEY and kw["stream_usage"] is True
    assert kw["base_url"] == SITE_BASE_URL and kw["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}


def test_resolver_load_chat_model_key_from_env_file(skill, sdk_mock, no_env_keys):
    (skill.parents[2] / "keys" / "a.env").write_text("ANTHROPIC_API_KEY=sk-file\n", encoding="utf-8")
    SDK = sdk_mock("langchain_anthropic.ChatAnthropic")
    load_chat_model(_write_profile(skill, "claude-sonnet-5", "anthropic"), skill_dir=skill)
    assert SDK.call_args.kwargs["api_key"] == "sk-file" and SDK.call_args.kwargs["effort"] == "medium"


def test_resolver_load_chat_model_missing_key_raises(skill, no_env_keys):
    with pytest.raises(KeyError, match="OPENAI_API_KEY"):
        load_chat_model(_write_profile(skill, "gpt-5.4-mini", "openai"), skill_dir=skill)


@pytest.mark.parametrize("extra", [{"temperature": 1.0}, {"extra_body": {"x": 1}}, {"model": "other"},
                                   {"api_key": "k"}, {"reasoning_effort": "max"}, {"base_url": "http://x"}])
def test_resolver_locked_client_kwargs_rejected(skill, extra):
    with pytest.raises(LockedKeyError, match="client_kwargs"):
        load_chat_model(_write_profile(skill), skill_dir=skill, client_kwargs=extra)


def test_resolver_kind_guards(skill):
    emb = _write_profile(skill, "gemini-embedding-2", "google", "embedding", "RETRIEVAL")
    with pytest.raises(ValueError, match="LLM"):
        load_chat_model(emb, skill_dir=skill)
    with pytest.raises(ValueError, match="임베딩"):
        load_embeddings(_write_profile(skill), skill_dir=skill)


# ── 임베딩 해석 ─────────────────────────────────────────────────────────────

def test_resolver_embedding_kwargs_google(mk_emb):
    prof = mk_emb("gemini-embedding-001", dims=768, measured=768)
    prof["limits"]["timeout_seconds"] = 45
    spec = resolve(prof)
    assert spec["sdk_class"] == "langchain_google_genai:GoogleGenerativeAIEmbeddings"
    assert spec["sdk_kwargs"] == {"model": "gemini-embedding-001", "output_dimensionality": 768,
                                  "request_options": {"timeout": 45}}
    assert spec["mechanism"] == "sdk_kwarg"
    assert spec["sent"] == {"timeout_seconds": 45, "max_retries": None}, "SDK 로 가지 않는 max_retries 는 sent 에서 null"


def test_resolver_embedding_kwargs_openai_compatible(mk_emb):
    spec = resolve(mk_emb("perplexity/pplx-embed-v1-4b", "openrouter"))
    assert spec["sdk_kwargs"] == {"model": "perplexity/pplx-embed-v1-4b", "base_url": "https://openrouter.ai/api/v1",
                                  "dimensions": None, "timeout": 120, "max_retries": 2, "tiktoken_enabled": False,
                                  "check_embedding_ctx_length": False}


def test_resolver_embedding_override_limits_and_unsupported(mk_emb):
    assert resolve(mk_emb("perplexity/pplx-embed-v1-4b", "openrouter"), {"timeout_seconds": 9})["sdk_kwargs"]["timeout"] == 9
    prof = mk_emb()
    prof["overridable"] = ["timeout_seconds", "max_retries"]            # 손 편집으로 google max_retries 를 열어도
    with pytest.raises(ValueError, match="SDK"):
        resolve(prof, {"max_retries": 5})


@pytest.mark.parametrize("dims,measured,match", [(3072, None, "inert"), (4096, None, "native"), (None, 768, "낡았다")])
def test_resolver_embedding_dimension_guards(mk_emb, dims, measured, match):
    prof = mk_emb()
    prof["dimensions"], prof["probe"]["measured_at_dimensions"] = dims, measured
    with pytest.raises(ValueError, match=match):
        resolve(prof)


def test_resolver_embedding_modality_strict(mk_emb):
    prof = mk_emb()
    prof["preset_values"]["modality"] = "image"
    with pytest.raises(ValueError, match="modality"):
        resolve(prof)


def test_resolver_embedding_anthropic_has_no_embedding_class(mk_emb, mk):
    prof = mk_emb()
    prof["provider_class"] = "anthropic"
    with pytest.raises(KeyError):
        resolve(prof)


class FakeEmb:
    """inner Embeddings 대역 — 받은 입력을 기록하고 고정 벡터를 돌려준다."""

    def __init__(self, vec=(3.0, 4.0)):
        self.vec = list(vec)
        self.queries: list[str] = []
        self.docs: list[list[str]] = []

    def embed_query(self, text):
        self.queries.append(text)
        return list(self.vec)

    def embed_documents(self, texts):
        self.docs.append(list(texts))
        return [list(self.vec) for _ in texts]

    async def aembed_query(self, text):
        return self.embed_query(text)

    async def aembed_documents(self, texts):
        return self.embed_documents(texts)


SUPPORTS = {"prefix_template": "Instruct: {task}\nQuery:{query}", "document_template": "title: none | text: {text}"}


def test_embed_text_prefix_query_only_and_normalized():
    inner = FakeEmb()
    emb = ProfileEmbeddings(inner, SUPPORTS, "text_prefix", "find answers")
    assert emb.embed_query("q") == [0.6, 0.8]
    assert inner.queries == ["Instruct: find answers\nQuery:q"]
    assert emb.embed_documents(["d"]) == [[0.6, 0.8]] and inner.docs == [["d"]]


def test_embed_text_prefix_both_templates():
    inner = FakeEmb()
    emb = ProfileEmbeddings(inner, {"prefix_template": "task: question answering | query: {query}",
                                    "document_template": "title: none | text: {text}"}, "text_prefix_both", None)
    emb.embed_query("q")
    emb.embed_documents(["d1", "d2"])
    assert inner.queries == ["task: question answering | query: q"]
    assert inner.docs == [["title: none | text: d1", "title: none | text: d2"]]


@pytest.mark.parametrize("mechanism", ["sdk_kwarg", None])
def test_embed_symmetric_mechanisms_pass_text_through(mechanism):
    inner = FakeEmb()
    emb = ProfileEmbeddings(inner, SUPPORTS, mechanism, None)
    emb.embed_query("q")
    emb.embed_documents(["d"])
    assert inner.queries == ["q"] and inner.docs == [["d"]]


def test_embed_async_paths_normalized():
    emb = ProfileEmbeddings(FakeEmb(), SUPPORTS, None, None)
    assert asyncio.run(emb.aembed_query("q")) == [0.6, 0.8]
    assert asyncio.run(emb.aembed_documents(["d"])) == [[0.6, 0.8]]


def test_embed_builtin_unit_vector_unchanged():
    vec = [1 / math.sqrt(2), 1 / math.sqrt(2)]
    out = ProfileEmbeddings(FakeEmb(vec), SUPPORTS, None, None).embed_query("q")
    assert all(math.isclose(a, b) for a, b in zip(out, vec))


def test_embed_zero_vector_raises():
    with pytest.raises(ZeroDivisionError):
        ProfileEmbeddings(FakeEmb((0.0, 0.0)), SUPPORTS, None, None).embed_query("q")


def test_embed_text_prefix_requires_query_instruction():
    with pytest.raises(ValueError, match="query_instruction"):
        ProfileEmbeddings(FakeEmb(), SUPPORTS, "text_prefix", None)


def test_embed_text_prefix_both_missing_document_template_raises():
    emb = ProfileEmbeddings(FakeEmb(), {"prefix_template": "q: {query}"}, "text_prefix_both", None)
    with pytest.raises(KeyError):
        emb.embed_documents(["d"])


def test_resolver_load_embeddings_builds_wrapper(skill, sdk_mock, no_env_keys):
    (skill.parents[2] / "keys" / "a.env").write_text("OPENROUTER_API_KEY=sk-or\n", encoding="utf-8")
    SDK = sdk_mock("langchain_openai.OpenAIEmbeddings")
    path, _ = pb.generate("qwen/qwen3-embedding-4b", "RETRIEVAL", "openrouter", "embedding", skill_dir=skill)
    emb = load_embeddings(path, skill_dir=skill, query_instruction="find")
    assert isinstance(emb, ProfileEmbeddings) and emb.mechanism == "text_prefix" and emb.inner is SDK.return_value
    assert SDK.call_args.kwargs["api_key"] == "sk-or" and SDK.call_args.kwargs["tiktoken_enabled"] is False
    with pytest.raises(ValueError, match="query_instruction"):
        load_embeddings(path, skill_dir=skill)


def test_resolver_spec_is_deterministic(mk):
    prof = mk()
    assert resolve(copy.deepcopy(prof)) == resolve(copy.deepcopy(prof))
