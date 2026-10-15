"""probe_runnable — 6 호출 방식(LLM)·embed 4종 실측의 verdict 분류·재시도·상태 규칙·노름 측정.

실제 provider 대신 LangChain 의 가짜 모델(GenericFakeChatModel)과 대역 Embeddings 를 쓴다(네트워크 없음).
"""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from conftest import PKG_ROOT
from langchain_model_preset_gen import catalog
from langchain_model_preset_gen import preflight as p
from langchain_model_preset_gen import probe_runnable as pr
from langchain_model_preset_gen import publish as pb
from langchain_model_preset_gen import resolver as rs

LLM_METHODS = ["invoke", "ainvoke", "stream", "astream", "batch", "abatch"]
EMB_METHODS = ["embed_query", "embed_documents", "aembed_query", "aembed_documents"]
DATA = catalog.load_data(PKG_ROOT)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(pr.time, "sleep", lambda s: None)


def fake_chat(n: int = 12):
    return GenericFakeChatModel(messages=iter([AIMessage(content="12 입니다")] * n))


class RecordingEmb:
    def __init__(self, vec=(3.0, 4.0), fail: dict | None = None):
        self.vec, self.fail, self.calls = list(vec), fail or {}, []

    def _hit(self, name):
        self.calls.append(name)
        if name in self.fail:
            raise self.fail[name]

    def embed_query(self, text):
        self._hit("embed_query")
        return list(self.vec)

    def embed_documents(self, texts):
        self._hit("embed_documents")
        return [list(self.vec) for _ in texts]

    async def aembed_query(self, text):
        self._hit("aembed_query")
        return list(self.vec)

    async def aembed_documents(self, texts):
        self._hit("aembed_documents")
        return [list(self.vec) for _ in texts]


# ── verdict 분류 ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("exc,verdict", [
    (RuntimeError("HTTP 429 rate limit exceeded"), "probe_error"),
    (TimeoutError("request timed out"), "probe_error"),
    (ConnectionError("connection reset"), "probe_error"),
    (RuntimeError("무슨 일인지 모르겠음"), "probe_error"),
    (NotImplementedError(), "unsupported"),
    (ValueError("this model does not support streaming"), "unsupported"),
    (Exception("Error code: 400 - {'message': 'Could not finish the message because max_tokens or model output "
               "limit was reached.', 'type': 'invalid_request_error'}"), "probe_error"),
    (Exception("Error code: 400 - {'message': 'Unsupported parameter: top_p', 'type': 'invalid_request_error'}"),
     "unsupported"),
    (Exception("Error code: 404 - {'message': 'The model `x` does not exist', 'type': 'invalid_request_error', "
               "'code': 'model_not_found'}"), "probe_error"),
    (TypeError('No synchronous function provided to "retrieve".'), "unsupported"),   # async 전용 Runnable — 사실
])
def test_probe_runnable_verdict_classification(exc, verdict):
    assert pr.classify_exception(exc) == verdict


def test_probe_runnable_smoke_retries_transient_then_gives_up(monkeypatch):
    calls = []

    def boom(target, method, prompt):
        calls.append(method)
        raise RuntimeError("429 rate limit")
    monkeypatch.setattr(pr, "run_method", boom)
    assert pr.smoke_one(object(), method="invoke", prompt="hi", retries=2) == "probe_error"
    assert len(calls) == 3


def test_probe_runnable_unsupported_does_not_retry(monkeypatch):
    calls = []

    def nope(target, method, prompt):
        calls.append(method)
        raise NotImplementedError()
    monkeypatch.setattr(pr, "run_method", nope)
    assert pr.smoke_one(object(), method="batch", prompt="hi", retries=2) == "unsupported"
    assert calls == ["batch"]


# ── 6 호출 방식 (실제 Runnable 경로) ─────────────────────────────────────────

def test_probe_runnable_six_call_modes_on_fake_chat_model():
    out = pr.check_runnable(fake_chat(), LLM_METHODS, "hi", embedding=False)
    assert out == {"runnable": {m: "ok" for m in LLM_METHODS}, "status": "ok"}


def test_probe_runnable_llm_status_ok_only_when_all_six_ok(monkeypatch):
    monkeypatch.setattr(pr, "smoke_one", lambda t, method, prompt: "unsupported" if method == "batch" else "ok")
    assert pr.check_runnable(object(), LLM_METHODS, "hi", embedding=False)["status"] == "probe_error"
    monkeypatch.setattr(pr, "smoke_one", lambda t, method, prompt: "probe_error" if method == "stream" else "ok")
    out = pr.check_runnable(object(), LLM_METHODS, "hi", embedding=False)
    assert out["status"] == "probe_error" and out["runnable"]["invoke"] == "ok"


def test_probe_runnable_empty_stream_is_not_ok():
    class Empty(GenericFakeChatModel):
        def _stream(self, *a, **k):
            return iter(())
    model = Empty(messages=iter([AIMessage(content="x")] * 4))
    assert pr.smoke_one(model, method="stream", prompt="hi", retries=0) == "probe_error"


# ── 임베딩 4종 ───────────────────────────────────────────────────────────

def test_probe_runnable_embed_four_methods_and_status():
    emb = RecordingEmb()
    assert pr.check_runnable(emb, EMB_METHODS, "hi", embedding=True) == {
        "runnable": {m: "ok" for m in EMB_METHODS}, "status": "ok"}
    assert emb.calls == EMB_METHODS


def test_probe_runnable_embed_unsupported_still_ok_but_probe_error_fails():
    emb = RecordingEmb(fail={"aembed_documents": NotImplementedError()})
    assert pr.check_runnable(emb, EMB_METHODS, "hi", embedding=True)["status"] == "ok"
    emb = RecordingEmb(fail={"embed_query": RuntimeError("model_not_found 404")})
    assert pr.check_runnable(emb, EMB_METHODS, "hi", embedding=True)["status"] == "probe_error"


def test_probe_runnable_embed_norm_and_dimensions():
    assert pr.measure_embedding(RecordingEmb((3.0, 4.0)), "hi") == {"norm_measured": 5.0, "measured_at_dimensions": 2}
    assert pr.derive_normalization(1.009) == "builtin"
    assert pr.derive_normalization(1.02) == "required"


# ── preflight runnable: 해석기 객체를 그대로 쓴다 ──────────────────────────────

def test_probe_runnable_uses_resolver_model_object(skill, monkeypatch):
    seen = {}

    def spy_build(spec, key, client_kwargs=None):
        seen["kwargs"] = rs.constructor_kwargs(spec, key, client_kwargs)
        return fake_chat()
    monkeypatch.setattr(rs, "build_chat_model", spy_build)
    spec = rs.resolve(pb.build_llm_profile("granite-4.2-8b", "RAG", "site_vllm", skill), {"effort": "low"})
    out = p.check_runnable(spec, None, DATA)
    assert out["status"] == "ok" and out["target"] == "GenericFakeChatModel"
    assert seen["kwargs"]["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True, "low_effort": True}


def test_probe_runnable_embedding_measures_raw_inner_norm(skill, monkeypatch):
    inner = RecordingEmb((3.0, 4.0))
    monkeypatch.setattr(rs, "import_class", lambda dotted: (lambda **kw: inner))
    spec = rs.resolve(pb.build_embedding_profile("qwen/qwen3-embedding-4b", "RETRIEVAL", "openrouter", None, skill))
    out = p.check_runnable(spec, "k", DATA)
    assert out["status"] == "ok" and out["norm_measured"] == 5.0, "노름은 정규화 전 raw 벡터"
    assert out["target"] == "ProfileEmbeddings"


# ── 적대 리뷰 보강: 인증·컨텍스트 초과·status_code·단어 경계 / 임베딩 핵심 경로 ─────────────────

class _StatusError(Exception):
    def __init__(self, msg, status_code):
        super().__init__(msg)
        self.status_code = status_code


@pytest.mark.parametrize("exc,verdict", [
    # OpenAI 401 은 type 이 invalid_request_error 다 — 키 문제는 모델 사실(unsupported)이 아니다
    (Exception("Error code: 401 - {'error': {'message': 'Incorrect API key provided', 'type': 'invalid_request_error'}}"),
     "probe_error"),
    (_StatusError("Unsupported something", 401), "probe_error"),
    (_StatusError("forbidden", 403), "probe_error"),
    (_StatusError("slow down", 429), "probe_error"),
    (_StatusError("upstream", 503), "probe_error"),
    (Exception("Error code: 400 - {'message': \"This model's maximum context length is 8192 tokens\", "
               "'type': 'invalid_request_error', 'code': 'context_length_exceeded'}"), "probe_error"),
    # 숫자 표지는 단어 경계로만 — request id 안의 429 는 rate limit 이 아니다
    (Exception("Unsupported parameter: top_p (request id req_4290ab)"), "unsupported"),
    # LangChain RunnableLambda(afunc 만) 의 동기 호출 거부 — async 전용 대상의 사실
    (NotImplementedError("Cannot invoke a coroutine function synchronously.Use `ainvoke` instead."), "unsupported"),
    (TypeError("Cannot invoke a coroutine function synchronously.Use `ainvoke` instead."), "unsupported"),
])
def test_probe_runnable_verdict_classification_hardened(exc, verdict):
    assert pr.classify_exception(exc) == verdict


@pytest.mark.parametrize("fail", [
    {m: NotImplementedError() for m in EMB_METHODS},                      # 4종 전부 unsupported
    {"embed_documents": NotImplementedError()},                          # 인덱스 빌드 경로
    {"embed_query": NotImplementedError()},                              # 질의 경로 + 노름 실측 원자료
])
def test_probe_runnable_embed_core_sync_methods_must_be_ok(fail):
    emb = RecordingEmb(fail=fail)
    assert pr.check_runnable(emb, EMB_METHODS, "hi", embedding=True)["status"] == "probe_error"
