"""probe_graph — 실측 상태 그래프(끄기·켜기 두 끝점 판정·sampling 수용·verdict 전이)의 결정론 회귀 가드.

HTTP 는 httpx.MockTransport 로만 흉내 낸다(실호출 없음). 전이표·주입표는 테스트 전용 dict 를 쓰고
(배포본 data/ 변경과 독립), 어휘·provider_class 표는 배포본 정본을 쓴다. 요청 인자는 해석기 spec 에서 온다.
off/on/sampling 요청은 fixture 의 질문 문자열로 구분한다.
"""

from __future__ import annotations

import json

import httpx
import pytest

from conftest import PKG_ROOT
from langchain_model_preset_gen import catalog
from langchain_model_preset_gen import injection as inj
from langchain_model_preset_gen import probe_graph as pg
from langchain_model_preset_gen import publish as pb
from langchain_model_preset_gen import resolver as rs

PROBE_MAX_TOKENS = 2048
OFF_PROMPT = "OFF_CHECK"
ON_PROMPT = "ON_CHECK"
ON_DEEP_PROMPT = "ON_DEEP_CHECK"
SAMPLING_PROMPT = "SAMPLING_CHECK"
BASE_URL = "http://probe.test/v1"

PROBE_TABLE = {
    "probe_request_max_tokens": PROBE_MAX_TOKENS,
    "prompts": {"off_check": OFF_PROMPT, "on_check": ON_PROMPT, "on_check_deep": ON_DEEP_PROMPT,
                "sampling_check": SAMPLING_PROMPT},
    "evidence": {"openai_compatible": {
        "text": ["choices.0.message.reasoning", "choices.0.message.reasoning_content"],
        "tokens": ["usage.completion_tokens_details.reasoning_tokens"],
    }},
    "answer": {"openai_compatible": {"content": "choices.0.message.content",
                                     "finish_reason": "choices.0.finish_reason", "finish_length": ["length"]}},
    "end_markers_default": ["</think>", "</thinking>"],
    "records": {"R_off_not_available_declared": {"then": "on_path", "meaning": "test"}},
    "nodes": {
        "connect": {"outcomes": {"model_listed": "off_gate", "model_missing": "T_model_missing"}},
        "off_gate": {"outcomes": {"none_is_disable": "off_probe",
                                  "none_is_omit_or_level": "R_off_not_available_declared"}},
        "off_probe": {"outcomes": {"clean": "on_path", "leak": "T_harmful_off",
                                   "thinking": "T_off_ignored", "http_4xx": "T_rejected",
                                   "timeout": "T_no_response_in_timeout"}},
        "on_path": {"outcomes": {"healthy": "sampling", "suppressed": "on_path_deep",
                                 "budget_exhausted": "T_budget_exhausted",
                                 "timeout": "T_no_response_in_timeout", "leak": "T_reasoning_leak",
                                 "http_4xx": "T_rejected"}},
        "on_path_deep": {"outcomes": {"healthy": "sampling", "suppressed": "T_on_suppressed",
                                      "budget_exhausted": "T_budget_exhausted",
                                      "timeout": "T_no_response_in_timeout", "leak": "T_reasoning_leak",
                                      "http_4xx": "T_rejected"}},
        "sampling": {"outcomes": {"all_accepted_as_declared": "PASS",
                                  "declared_accepted_but_4xx": "T_sampling_rejected",
                                  "timeout": "T_no_response_in_timeout"}},
        "any": {"outcomes": {"http_429_5xx_or_transport": "U_undeterminable"}},
    },
    "diagnoses": {t: f"diag {t}" for t in [
        "PASS", "T_model_missing", "T_harmful_off", "T_off_ignored", "T_rejected", "T_on_suppressed",
        "T_budget_exhausted", "T_no_response_in_timeout", "T_reasoning_leak", "T_sampling_rejected",
        "U_undeterminable"]},
}

INJECTION_TABLE = {
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
    "none": {},
}

FULL_MAPPING = {
    "none": {"action": "disable"},
    "low": {"action": "set", "value": "low_effort"},
    "medium": {"action": "set", "value": "enable"},
    "high": {"action": "set", "value": "enable"},
    "xhigh": {"action": "set", "value": "enable"},
    "max": {"action": "set", "value": "enable"},
}


REAL_DATA = catalog.load_data(PKG_ROOT)


@pytest.fixture
def data_dir() -> dict:
    """테스트 전이표·주입표 + 정본 어휘·provider_class 표(원문 HTTP 형식 자체가 검증 대상)."""
    return {**REAL_DATA, "probe_graph": PROBE_TABLE, "injection": INJECTION_TABLE}


def make_profile(**over) -> dict:
    prof = {
        "kind": "llm",
        "model_name": "granite-4.2-8b",
        "serving_provider": "site_vllm",
        "provider_class": "openai_compatible",
        "base_url": BASE_URL,
        "api_key_env": None,
        "preset": "RAG",
        "overridable": ["temperature", "top_p", "top_k", "effort", "timeout_seconds", "max_retries",
                        "response_deadline_seconds", "max_tokens"],
        "reasoning_effort": {"injection": "ct_enable_thinking", "mapping": dict(FULL_MAPPING),
                             "end_marker": "</think>", "basis": "test"},
        "sampling": {"temperature": "accepted", "top_p": "accepted", "top_k": "accepted"},
        "preset_values": {"temperature": 0.1, "top_p": 0.9, "top_k": 20, "effort": "medium"},
        "limits": {"timeout_seconds": 120, "max_retries": 2, "response_deadline_seconds": None,
                   "max_tokens": 8192},
        "probe": {"status": "probe_error", "method": "static", "measured_at": None, "terminal": None,
                  "runnable": {}},
    }
    prof.update(over)
    return prof


def chat(content: str | None = "12", reasoning: str = "", rtokens: int | None = 0,
         finish: str = "stop", status: int = 200) -> httpx.Response:
    return httpx.Response(status, json={
        "choices": [{"message": {"role": "assistant", "content": content, "reasoning": reasoning},
                     "finish_reason": finish}],
        "usage": {"completion_tokens": 10, "completion_tokens_details": {"reasoning_tokens": rtokens}},
    })


OAC_EVIDENCE = PROBE_TABLE["evidence"]["openai_compatible"]
THINKING_ON = {"content": "12", "reasoning": "생각 중...", "rtokens": 40}
CLEAN_OFF = {"content": "12", "reasoning": "", "rtokens": 0}


class Recorder:
    """phase(models/off/on/sampling:<key>) → 응답 생성기. 받은 요청을 기록한다."""

    def __init__(self, responses: dict, models: list[str] | None = None):
        self.responses = responses
        self.models = ["granite-4.2-8b"] if models is None else models
        self.requests: list[tuple[str, httpx.Request, dict | None]] = []

    def phase_of(self, req: httpx.Request, body: dict | None) -> str:
        if req.method == "GET":
            return "models"
        prompt = body["messages"][0]["content"]
        if prompt == OFF_PROMPT:
            return "off"
        if prompt == ON_PROMPT:
            return "on"
        if prompt == ON_DEEP_PROMPT:
            return "on_deep"
        keys = [k for k in ("temperature", "top_p", "top_k") if k in body]
        return f"sampling:{keys[0]}"

    def __call__(self, req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content) if req.method == "POST" else None
        phase = self.phase_of(req, body)
        self.requests.append((phase, req, body))
        spec = self.responses.get(phase, self.responses.get("sampling" if phase.startswith("sampling") else phase))
        if phase == "models" and spec is None:
            return httpx.Response(200, json={"data": [{"id": m} for m in self.models]})
        if callable(spec):
            return spec(req)
        if isinstance(spec, httpx.Response):
            return spec
        return chat(**(spec or THINKING_ON))

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))

    def bodies(self, phase: str) -> list[dict]:
        return [b for p, _, b in self.requests if p == phase]


def run(profile, rec: Recorder, data: dict, effective=None, key=None) -> dict:
    spec = rs.resolve(profile, effective, data=data)
    return pg.run_probe(spec, key, probe_table=data["probe_graph"], provider_classes=data["provider_classes"],
                        client=rec.client())


# ════════════════ 순수 함수 ════════════════

def test_thinking_evidence_text_only():
    body = chat(reasoning="abc", rtokens=None).json()
    ev = pg.thinking_evidence(body, OAC_EVIDENCE)
    assert ev["thinking"] is True


def test_thinking_evidence_tokens_only():
    """텍스트 숨김 모델(gemini·gpt-6-luna) — reasoning 텍스트 0, 토큰만 > 0."""
    body = chat(reasoning="", rtokens=37).json()
    assert pg.thinking_evidence(body, OAC_EVIDENCE)["thinking"] is True


def test_thinking_evidence_none_when_all_zero_or_absent():
    body = {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}
    assert pg.thinking_evidence(body, OAC_EVIDENCE)["thinking"] is False


def test_resolve_transition_node_outcome():
    assert pg.resolve_transition(PROBE_TABLE, "on_path", "healthy") == "sampling"


def test_resolve_transition_any_fallback():
    assert pg.resolve_transition(PROBE_TABLE, "off_probe", "http_429_5xx_or_transport") == "U_undeterminable"


def test_resolve_transition_undefined_raises():
    with pytest.raises(pg.UndefinedTransition):
        pg.resolve_transition(PROBE_TABLE, "connect", "http_4xx")


def test_probe_effective_from_resolver_override(data_dir):
    eff = rs.resolve(make_profile(), {"effort": "high", "timeout_seconds": 5}, data=data_dir)["effective"]
    assert eff["effort"] == "high" and eff["timeout_seconds"] == 5 and eff["temperature"] == 0.1


def test_probe_unknown_override_key_raises(data_dir):
    with pytest.raises(rs.LockedKeyError):
        run(make_profile(), Recorder({}), data_dir, effective={"temprature": 0.3})


def test_probe_rejects_embedding_spec():
    spec = {"kind": "embedding", "name": "e"}
    with pytest.raises(ValueError, match="LLM 전용"):
        pg.run_probe(spec, None, probe_table=PROBE_TABLE, provider_classes={})


def test_build_graph_node_set_mismatch_raises():
    bad = json.loads(json.dumps(PROBE_TABLE))
    bad["nodes"]["extra_node"] = {"outcomes": {}}
    with pytest.raises(ValueError):
        pg.check_table_nodes(bad)


def test_mermaid_contains_all_nodes():
    text = pg.build_graph(PROBE_TABLE).get_graph().draw_mermaid()
    for n in ("connect", "off_gate", "off_probe", "on_path", "sampling"):
        assert n in text


# ════════════════ 전 경로 (MockTransport) ════════════════

def test_full_pass_path_and_requests(data_dir):
    rec = Recorder({"off": CLEAN_OFF, "on": THINKING_ON, "sampling": THINKING_ON})
    res = run(make_profile(), rec, data_dir)
    assert res["terminal"] == "PASS"
    assert res["status"] == "ok"
    assert res["diagnosis"] == "diag PASS"
    assert res["path"] == ["connect", "off_gate", "off_probe", "on_path", "sampling"]
    off = rec.bodies("off")[0]
    assert off["chat_template_kwargs"] == {"enable_thinking": False}   # extra_body. 평탄화
    assert off["max_tokens"] == PROBE_MAX_TOKENS
    on = rec.bodies("on")[0]
    assert on["chat_template_kwargs"] == {"enable_thinking": True}     # medium → enable
    assert "temperature" not in on
    assert rec.bodies("sampling:temperature")[0]["temperature"] == 0.1
    assert rec.bodies("sampling:top_p")[0]["top_p"] == 0.9
    assert rec.bodies("sampling:top_k")[0]["top_k"] == 20
    # sampling 요청에도 effort 칸 주입이 실린다
    assert rec.bodies("sampling:top_k")[0]["chat_template_kwargs"] == {"enable_thinking": True}


def test_on_path_uses_effective_timeout(data_dir):
    rec = Recorder({"off": CLEAN_OFF})
    run(make_profile(), rec, data_dir, effective={"timeout_seconds": 7})
    on_req = [r for p, r, _ in rec.requests if p == "on"][0]
    assert on_req.extensions["timeout"]["read"] == 7


def test_effective_max_tokens_below_cap_is_sent(data_dir):
    """실효 max_tokens 가 탐침 상한보다 작으면 그 값을 보낸다 — override max_tokens 가 게이트 실측에서 inert 이면 안 된다."""
    rec = Recorder({"off": CLEAN_OFF, "on": THINKING_ON, "sampling": THINKING_ON})
    run(make_profile(), rec, data_dir, effective={"max_tokens": 64})
    assert rec.bodies("off")[0]["max_tokens"] == 64
    assert rec.bodies("on")[0]["max_tokens"] == 64
    assert rec.bodies("sampling:temperature")[0]["max_tokens"] == 64


def test_effective_max_tokens_above_cap_is_capped(data_dir):
    """상한은 비용 상한이다 — 실효값(preset 8192)이 더 커도 상한을 넘지 않는다."""
    rec = Recorder({"off": CLEAN_OFF, "on": THINKING_ON, "sampling": THINKING_ON})
    run(make_profile(), rec, data_dir)
    assert rec.bodies("on")[0]["max_tokens"] == PROBE_MAX_TOKENS


def test_override_effort_changes_slot(data_dir):
    rec = Recorder({"off": CLEAN_OFF})
    run(make_profile(), rec, data_dir, effective={"effort": "low"})
    assert rec.bodies("on")[0]["chat_template_kwargs"] == {"enable_thinking": True, "low_effort": True}


def test_connect_model_missing(data_dir):
    rec = Recorder({}, models=["other-model"])
    res = run(make_profile(), rec, data_dir)
    assert res["terminal"] == "T_model_missing"
    assert res["status"] == "probe_error"
    assert res["diagnosis"] == "diag T_model_missing"
    assert res["path"] == ["connect"]


def test_connect_429_undeterminable(data_dir):
    rec = Recorder({"models": httpx.Response(429, text="rate")})
    res = run(make_profile(), rec, data_dir)
    assert res["terminal"] == "U_undeterminable"
    assert res["status"] == "probe_error"


def test_connect_401_is_undefined_transition(data_dir):
    rec = Recorder({"models": httpx.Response(401, text="no key")})
    with pytest.raises(pg.UndefinedTransition):
        run(make_profile(), rec, data_dir)


def test_off_gate_omit_skips_off_probe_and_records(data_dir):
    mapping = dict(FULL_MAPPING, none={"action": "omit"})
    prof = make_profile(reasoning_effort={"injection": "ct_enable_thinking", "mapping": mapping,
                                          "end_marker": "</think>", "basis": "t"})
    rec = Recorder({})
    res = run(prof, rec, data_dir)
    assert res["path"] == ["connect", "off_gate", "on_path", "sampling"]
    assert res["records"] == ["R_off_not_available_declared"]
    assert rec.bodies("off") == []
    assert res["terminal"] == "PASS"


def test_off_gate_level_slot_skips_off_probe(data_dir):
    """끌 수 없는 모델: none 칸 = 최저 단계 (스칼라 축약형)."""
    mapping = dict(FULL_MAPPING, none="low_effort")
    prof = make_profile(reasoning_effort={"injection": "ct_enable_thinking", "mapping": mapping,
                                          "end_marker": "</think>", "basis": "t"})
    res = run(prof, Recorder({}), data_dir)
    assert "off_probe" not in res["path"]
    assert res["records"] == ["R_off_not_available_declared"]


def test_off_probe_leak_harmful_off(data_dir):
    rec = Recorder({"off": {"content": "생각</think>12", "reasoning": "", "rtokens": 0}})
    assert run(make_profile(), rec, data_dir)["terminal"] == "T_harmful_off"


def test_off_probe_thinking_off_ignored(data_dir):
    rec = Recorder({"off": THINKING_ON})
    res = run(make_profile(), rec, data_dir)
    assert res["terminal"] == "T_off_ignored"
    assert res["diagnosis"] == "diag T_off_ignored"


def test_off_probe_tokens_only_counts_as_thinking(data_dir):
    rec = Recorder({"off": {"content": "12", "reasoning": "", "rtokens": 5}})
    assert run(make_profile(), rec, data_dir)["terminal"] == "T_off_ignored"


def test_off_probe_400_rejected(data_dir):
    rec = Recorder({"off": httpx.Response(400, text="bad kwarg")})
    assert run(make_profile(), rec, data_dir)["terminal"] == "T_rejected"


def test_off_probe_transport_error_undeterminable(data_dir):
    def boom(req):
        raise httpx.ConnectError("refused", request=req)
    rec = Recorder({"off": boom})
    assert run(make_profile(), rec, data_dir)["terminal"] == "U_undeterminable"


def test_on_path_suppressed_when_set_and_no_thinking(data_dir):
    """심화 질문에서도 사고 0이어야 T_on_suppressed 다."""
    rec = Recorder({"off": CLEAN_OFF, "on": CLEAN_OFF, "on_deep": CLEAN_OFF})
    res = run(make_profile(), rec, data_dir)
    assert res["terminal"] == "T_on_suppressed"
    assert res["path"] == ["connect", "off_gate", "off_probe", "on_path", "on_path_deep"]


def test_on_path_suppressed_then_deep_thinking_passes(data_dir):
    """적응형 모델이 쉬운 질문에서 사고를 건너뛰면 심화 질문 1회로 다시 본다 (gpt-5.6-sol high 0토큰 → 심화 52토큰 실측)."""
    rec = Recorder({"off": CLEAN_OFF, "on": CLEAN_OFF, "on_deep": THINKING_ON, "sampling": THINKING_ON})
    res = run(make_profile(), rec, data_dir)
    assert res["terminal"] == "PASS"
    assert res["path"] == ["connect", "off_gate", "off_probe", "on_path", "on_path_deep", "sampling"]
    deep = rec.bodies("on_deep")[0]
    assert deep["chat_template_kwargs"] == {"enable_thinking": True}     # 같은 실효 effort 칸
    assert len(rec.bodies("on_deep")) == 1                                # 1회만


def test_on_path_healthy_skips_deep(data_dir):
    """첫 질문에서 사고하면 심화 노드에 가지 않는다 — 소형 모델은 심화 질문(시간 초과 위험)을 받지 않는다."""
    rec = Recorder({"off": CLEAN_OFF, "on": THINKING_ON, "sampling": THINKING_ON})
    res = run(make_profile(), rec, data_dir)
    assert "on_path_deep" not in res["path"] and rec.bodies("on_deep") == []


def test_on_path_deep_timeout_is_no_response(data_dir):
    def slow(req):
        raise httpx.ReadTimeout("slow", request=req)
    rec = Recorder({"off": CLEAN_OFF, "on": CLEAN_OFF, "on_deep": slow})
    assert run(make_profile(), rec, data_dir)["terminal"] == "T_no_response_in_timeout"


def test_on_path_omit_slot_without_thinking_is_healthy(data_dir):
    """칸이 omit이면 사고 0이어도 suppressed가 아니다 (lfm형)."""
    omit_all = {k: {"action": "omit"} for k in FULL_MAPPING}
    prof = make_profile(reasoning_effort={"injection": "none", "mapping": omit_all,
                                          "end_marker": "</think>", "basis": "t"})
    res = run(prof, Recorder({"on": CLEAN_OFF, "sampling": CLEAN_OFF}), data_dir)
    assert res["terminal"] == "PASS"
    assert res["path"] == ["connect", "off_gate", "on_path", "sampling"]


def test_on_path_budget_exhausted(data_dir):
    rec = Recorder({"off": CLEAN_OFF, "on": {"content": "", "reasoning": "긴 생각", "rtokens": 2048,
                                             "finish": "length"}})
    assert run(make_profile(), rec, data_dir)["terminal"] == "T_budget_exhausted"


def test_on_path_budget_exhausted_null_content(data_dir):
    rec = Recorder({"off": CLEAN_OFF, "on": {"content": None, "reasoning": "x", "rtokens": 9,
                                             "finish": "length"}})
    assert run(make_profile(), rec, data_dir)["terminal"] == "T_budget_exhausted"


def test_on_path_leak_end_marker_in_content(data_dir):
    rec = Recorder({"off": CLEAN_OFF, "on": {"content": "생각</think>12", "reasoning": "", "rtokens": 0}})
    assert run(make_profile(), rec, data_dir)["terminal"] == "T_reasoning_leak"


def test_on_path_leak_uses_default_markers_when_profile_marker_null(data_dir):
    prof = make_profile(reasoning_effort={"injection": "ct_enable_thinking", "mapping": dict(FULL_MAPPING),
                                          "end_marker": None, "basis": "t"})
    rec = Recorder({"off": CLEAN_OFF, "on": {"content": "a</thinking>b", "reasoning": "x", "rtokens": 3}})
    assert run(prof, rec, data_dir)["terminal"] == "T_reasoning_leak"


def test_on_path_timeout(data_dir):
    def slow(req):
        raise httpx.ReadTimeout("slow", request=req)
    rec = Recorder({"off": CLEAN_OFF, "on": slow})
    res = run(make_profile(), rec, data_dir)
    assert res["terminal"] == "T_no_response_in_timeout"
    assert res["status"] == "probe_error"


def _raiser(exc_cls):
    def raise_(req):
        raise exc_cls("x", request=req)
    return raise_


def test_off_probe_timeout_no_response_in_timeout(data_dir):
    """[T3.4] off_probe 도 on_path 와 같은 실효 timeout — 응답 대기 초과는 모델 사실(timeout 상향 후보)."""
    res = run(make_profile(), Recorder({"off": _raiser(httpx.ReadTimeout)}), data_dir)
    assert res["terminal"] == "T_no_response_in_timeout"
    assert res["path"] == ["connect", "off_gate", "off_probe"]


def test_sampling_timeout_no_response_in_timeout(data_dir):
    rec = Recorder({"off": CLEAN_OFF, "on": THINKING_ON, "sampling:temperature": _raiser(httpx.ReadTimeout)})
    res = run(make_profile(), rec, data_dir)
    assert res["terminal"] == "T_no_response_in_timeout"
    assert [p for p, _, _ in rec.requests if p.startswith("sampling")] == ["sampling:temperature"]   # 첫 실패에서 멈춤


@pytest.mark.parametrize("exc_cls", [httpx.ConnectTimeout, httpx.PoolTimeout, httpx.WriteTimeout])
def test_non_read_timeouts_are_transport_undeterminable(data_dir, exc_cls):
    """ReadTimeout 만 '모델이 timeout 안에 답하지 않음'. 연결·풀·쓰기 timeout 은 인프라 → U_undeterminable."""
    rec = Recorder({"off": CLEAN_OFF, "on": _raiser(exc_cls)})
    res = run(make_profile(), rec, data_dir)
    assert res["terminal"] == "U_undeterminable"
    on_obs = [o for o in res["observations"] if o["node"] == "on_path"][0]
    assert on_obs["kind"] == "transport" and exc_cls.__name__ in on_obs["error"]


def test_classify_timeout_obs_is_timeout_outcome_on_every_request_node():
    obs = {"kind": "timeout", "status": None, "body": None}
    fmt = pg.response_format(PROBE_TABLE, "openai_compatible")
    assert pg.classify_off_probe(obs, ["</think>"], fmt) == pg.OUT_TIMEOUT
    assert pg.classify_on_path(obs, "set", ["</think>"], fmt) == pg.OUT_TIMEOUT
    assert pg.classify_sampling([obs]) == pg.OUT_TIMEOUT


def test_on_path_400_rejected(data_dir):
    rec = Recorder({"off": CLEAN_OFF, "on": httpx.Response(400, text="bad")})
    assert run(make_profile(), rec, data_dir)["terminal"] == "T_rejected"


def test_on_path_503_undeterminable(data_dir):
    rec = Recorder({"off": CLEAN_OFF, "on": httpx.Response(503, text="down")})
    assert run(make_profile(), rec, data_dir)["terminal"] == "U_undeterminable"


def test_on_path_empty_content_stop_is_undefined_transition(data_dir):
    """전이표에 없는 관측(content 비어 있음 + stop) → 즉시 멈춤."""
    rec = Recorder({"off": CLEAN_OFF, "on": {"content": "", "reasoning": "x", "rtokens": 3, "finish": "stop"}})
    with pytest.raises(pg.UndefinedTransition):
        run(make_profile(), rec, data_dir)


def test_sampling_4xx_rejected(data_dir):
    rec = Recorder({"off": CLEAN_OFF, "on": THINKING_ON, "sampling:top_k": httpx.Response(400, text="top_k")})
    res = run(make_profile(), rec, data_dir)
    assert res["terminal"] == "T_sampling_rejected"
    assert res["diagnosis"] == "diag T_sampling_rejected"


def test_sampling_429_undeterminable(data_dir):
    rec = Recorder({"off": CLEAN_OFF, "on": THINKING_ON, "sampling:temperature": httpx.Response(429, text="r")})
    assert run(make_profile(), rec, data_dir)["terminal"] == "U_undeterminable"


def test_sampling_sends_only_accepted_non_null_keys(data_dir):
    prof = make_profile(sampling={"temperature": "accepted", "top_p": "ignored", "top_k": "rejected"},
                        preset_values={"temperature": 0.0, "top_p": None, "top_k": None, "effort": "medium"})
    rec = Recorder({"off": CLEAN_OFF})
    assert run(prof, rec, data_dir)["terminal"] == "PASS"
    assert [p for p, _, _ in rec.requests if p.startswith("sampling")] == ["sampling:temperature"]
    assert rec.bodies("sampling:temperature")[0]["temperature"] == 0.0


def test_sampling_null_value_not_sent(data_dir):
    prof = make_profile(preset_values={"temperature": 0.0, "top_p": None, "top_k": None, "effort": "medium"})
    rec = Recorder({"off": CLEAN_OFF})
    run(prof, rec, data_dir)
    assert [p for p, _, _ in rec.requests if p.startswith("sampling")] == ["sampling:temperature"]


def test_sampling_by_effort_resolution(data_dir):
    """gpt-5.x형: temperature는 none 단계에서만 accepted."""
    samp = {"temperature": {"by_effort": {"none": "accepted", "default": "ignored"}},
            "top_p": "ignored", "top_k": "ignored"}
    rec = Recorder({"off": CLEAN_OFF})
    run(make_profile(sampling=samp), rec, data_dir)                       # effort medium
    assert [p for p, _, _ in rec.requests if p.startswith("sampling")] == []

    rec2 = Recorder({"off": CLEAN_OFF, "on": CLEAN_OFF, "sampling": CLEAN_OFF})
    omit_none = dict(FULL_MAPPING, none={"action": "omit"})
    prof = make_profile(sampling=samp, reasoning_effort={"injection": "ct_enable_thinking", "mapping": omit_none,
                                                         "end_marker": "</think>", "basis": "t"})
    res = run(prof, rec2, data_dir, effective={"effort": "none"})
    assert res["terminal"] == "PASS"
    assert [p for p, _, _ in rec2.requests if p.startswith("sampling")] == ["sampling:temperature"]


def test_observations_recorded_per_request(data_dir):
    rec = Recorder({"off": CLEAN_OFF, "on": THINKING_ON, "sampling": THINKING_ON})
    res = run(make_profile(), rec, data_dir)
    nodes = [o["node"] for o in res["observations"]]
    assert nodes == ["connect", "off_probe", "on_path", "sampling", "sampling", "sampling"]
    on_obs = [o for o in res["observations"] if o["node"] == "on_path"][0]
    assert on_obs["status"] == 200
    assert on_obs["thinking"] is True


def test_auth_header_from_api_key_env(data_dir):
    prof = make_profile(api_key_env="OPENROUTER_API_KEY")
    rec = Recorder({"off": CLEAN_OFF})
    run(prof, rec, data_dir, key="sk-test")
    assert all(r.headers["Authorization"] == "Bearer sk-test" for _, r, _ in rec.requests)


def test_no_auth_header_for_local(data_dir):
    rec = Recorder({"off": CLEAN_OFF})
    run(make_profile(), rec, data_dir)
    assert all("Authorization" not in r.headers for _, r, _ in rec.requests)


def test_probe_profile_missing_api_key_raises(skill, no_env_keys):
    path, _ = pb.generate("gpt-5.4-mini", "RAG", "openai", "llm", skill_dir=skill)
    with pytest.raises(KeyError, match="OPENAI_API_KEY"):
        pg.probe_profile(path, skill_dir=skill, client=Recorder({}).client())


def test_or_reasoning_injection_body(data_dir):
    mapping = dict(FULL_MAPPING, medium={"action": "set", "value": "high"})
    prof = make_profile(reasoning_effort={"injection": "or_reasoning", "mapping": mapping,
                                          "end_marker": None, "basis": "t"})
    rec = Recorder({"off": CLEAN_OFF})
    run(prof, rec, data_dir)
    assert rec.bodies("off")[0]["reasoning"] == {"enabled": False}
    assert rec.bodies("on")[0]["reasoning"] == {"effort": "high"}


# ════════════════ 스킬 data/ 정본과의 정합 ════════════════

def test_skill_data_tables_compile_and_cover_terminals():
    """정본 probe_graph.yaml: 노드 집합 일치 + 모든 종착에 진단 + 모든 R_*에 then."""
    table, injection_table = REAL_DATA["probe_graph"], REAL_DATA["injection"]
    pg.build_graph(table)                                   # check_table_nodes 포함
    for node in table["nodes"].values():
        for target in node["outcomes"].values():
            records, nxt, terminal = pg._next_hop(table, target)
            assert nxt in pg.NODE_NAMES or terminal in table["diagnoses"]
    for inj_id in ("ct_enable_thinking", "or_reasoning", "openai_native", "none"):
        assert inj_id in injection_table


def test_skill_table_timeout_outcome_on_every_timed_request_node():
    """[T3.4] 실효 timeout_seconds 로 생성 요청을 보내는 세 노드는 timeout 을 같은 종착으로 보낸다."""
    table = REAL_DATA["probe_graph"]
    for node in ("off_probe", "on_path", "sampling"):
        assert table["nodes"][node]["outcomes"][pg.OUT_TIMEOUT] == "T_no_response_in_timeout"


# ════════════════ CLI (종료코드) ════════════════

def _cli(monkeypatch, tmp_path, *extra, status="ok"):
    prof = tmp_path / "p.yaml"
    prof.write_text("kind: llm\n", encoding="utf-8")
    monkeypatch.setattr(pg, "probe_profile", lambda *a, **k: {"status": status, "terminal": "X"})
    return pg.main(["--profile", str(prof), *extra])


def test_cli_exit_codes_pass_and_probe_error(monkeypatch, tmp_path):
    assert _cli(monkeypatch, tmp_path, status="ok") == 0
    assert _cli(monkeypatch, tmp_path, status="probe_error") == pg.PROBE_ERROR_EXIT_CODE


def test_cli_probe_error_exit_code_distinct_from_crash_and_usage():
    """1 = 잡히지 않은 예외(파이썬), 2 = argparse 사용법 오류 — 결과 코드가 이 둘과 겹치면 호출자가 구분 못 한다."""
    assert pg.PROBE_ERROR_EXIT_CODE not in (0, 1, 2)


def test_cli_env_file_option_removed(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        _cli(monkeypatch, tmp_path, "--env-file", "x.env")


# ════════════════ native provider_class — 배포본 정본 data/ + 발행 프로필 ════════════════
# 정본 probe_graph.yaml 은 off_check·on_check 질문이 같아 질문으로 단계를 가를 수 없다 → 요청 순서로 응답한다.

REAL_PROBE_TABLE, REAL_INJECTION = REAL_DATA["probe_graph"], REAL_DATA["injection"]
REAL_PROVIDER_CLASSES = REAL_DATA["provider_classes"]
REAL_CAP = REAL_PROBE_TABLE["probe_request_max_tokens"]
NATIVE_KEYS = {"openai": "sk-o", "anthropic": "sk-a", "google": "g-key"}


class SeqRecorder:
    """GET → models 응답, POST → posts 목록을 순서대로. 받은 요청을 기록한다."""

    def __init__(self, models: dict, posts: list):
        self.models = models
        self.posts = list(posts)
        self.requests: list[httpx.Request] = []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.requests.append(req)
        if req.method == "GET":
            return httpx.Response(200, json=self.models)
        return httpx.Response(200, json=self.posts.pop(0)) if isinstance(self.posts[0], dict) else self.posts.pop(0)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))

    def post_bodies(self) -> list[dict]:
        return [json.loads(r.content) for r in self.requests if r.method == "POST"]


def _native_run(profile: dict, rec: SeqRecorder, effective=None) -> dict:
    spec = rs.resolve(profile, effective, data=REAL_DATA)
    return pg.run_probe(spec, NATIVE_KEYS[profile["provider_class"]], probe_table=REAL_PROBE_TABLE,
                        provider_classes=REAL_PROVIDER_CLASSES, client=rec.client())


@pytest.fixture
def gp(skill):
    """발행기 — 배포본 정본 규칙·preset 으로 프로필 dict 를 만든다."""
    class _P:
        @staticmethod
        def build_llm_profile(model, preset, sp):
            return pb.build_llm_profile(model, preset, sp, skill)
    return _P


def _openai_resp(content="12", rtokens=0, finish="stop") -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": finish}],
            "usage": {"completion_tokens": 5, "completion_tokens_details": {"reasoning_tokens": rtokens}}}


def _anthropic_resp(text="12", thinking=None, signature="", tokens=None, stop="end_turn") -> dict:
    content = []
    if thinking is not None:
        content.append({"type": "thinking", "thinking": thinking, "signature": signature})
    if text is not None:
        content.append({"type": "text", "text": text})
    usage = {"input_tokens": 9, "output_tokens": 5}
    if tokens is not None:
        usage["output_tokens_details"] = {"thinking_tokens": tokens}
    return {"type": "message", "role": "assistant", "content": content, "stop_reason": stop, "usage": usage}


def _gemini_resp(text="12", thoughts=0, finish="STOP", thought_text=None) -> dict:
    parts = [] if thought_text is None else [{"text": thought_text, "thought": True}]
    parts += [] if text is None else [{"text": text}]
    usage = {"promptTokenCount": 9, "candidatesTokenCount": 2}
    if thoughts:
        usage["thoughtsTokenCount"] = thoughts
    return {"candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": finish}],
            "usageMetadata": usage}


OPENAI_MODELS = {"data": [{"id": "gpt-5.4-mini"}]}
ANTHROPIC_MODELS = {"data": [{"id": "claude-haiku-4-5-20251001"}, {"id": "claude-sonnet-5"}], "has_more": False}
GEMINI_MODELS = {"models": [{"name": "models/gemini-3.6-flash"}]}


def test_values_filter_syntax():
    body = {"content": [{"type": "thinking", "thinking": "ab"}, {"type": "text", "text": "x"},
                        {"type": "text", "text": "y"}],
            "parts": [{"text": "t", "thought": True}, {"text": "a"}]}
    assert pg._values(body, "content[type=text].text") == ["x", "y"]
    assert pg._values(body, "content[type=thinking].thinking") == ["ab"]
    assert pg._values(body, "parts[thought=true].text") == ["t"]
    assert pg._values(body, "parts[thought!=true].text") == ["a"]       # 키 없는 원소는 != 에 맞는다
    assert pg._values(body, "content.5.text") == []
    with pytest.raises(ValueError):
        pg._values(body, "content[type=text")


def test_native_base_url_null_for_openai_compatible_raises():
    prof = make_profile(base_url=None)
    with pytest.raises(ValueError, match="native_base_url"):
        rs.native_base_url(prof, REAL_PROVIDER_CLASSES)


def test_native_body_mapped_unknown_sdk_kwarg_raises():
    """messages api 에 extra_body(openai 계열 인자)가 오면 body_paths 에 없어 KeyError — 주입 방식 불일치."""
    spec = REAL_PROVIDER_CLASSES["anthropic"]["http"]
    with pytest.raises(KeyError):
        pg.native_body(spec, "m", "q", REAL_CAP, {"extra_body": {"reasoning": {"enabled": True}}})


def test_openai_native_request_and_pass(gp):
    prof = gp.build_llm_profile("gpt-5.4-mini", "RAG", "openai")
    rec = SeqRecorder(OPENAI_MODELS, [_openai_resp(rtokens=0), _openai_resp(rtokens=32)])
    res = _native_run(prof, rec)
    assert res["terminal"] == "PASS"
    get = rec.requests[0]
    assert str(get.url) == "https://api.openai.com/v1/models"
    assert get.headers["Authorization"] == "Bearer sk-o"
    off, on = rec.post_bodies()
    assert str(rec.requests[1].url) == "https://api.openai.com/v1/chat/completions"
    assert off["reasoning_effort"] == "none" and on["reasoning_effort"] == "medium"
    assert off["max_completion_tokens"] == REAL_CAP and "max_tokens" not in off
    assert len(rec.post_bodies()) == 2       # sampling: by_effort medium → ignored → 전송 0건


def test_openai_native_tokens_only_evidence_off_ignored(gp):
    prof = gp.build_llm_profile("gpt-5.4-mini", "RAG", "openai")
    rec = SeqRecorder(OPENAI_MODELS, [_openai_resp(rtokens=9)])
    assert _native_run(prof, rec)["terminal"] == "T_off_ignored"


def test_openai_native_budget_exhausted(gp):
    prof = gp.build_llm_profile("gpt-5.4-mini", "RAG", "openai")
    rec = SeqRecorder(OPENAI_MODELS, [_openai_resp(), _openai_resp(content="", rtokens=REAL_CAP, finish="length")])
    assert _native_run(prof, rec)["terminal"] == "T_budget_exhausted"


def test_anthropic_thinking_budget_request_and_pass(gp):
    prof = gp.build_llm_profile("claude-haiku-4-5-20251001", "RAG", "anthropic")
    rec = SeqRecorder(ANTHROPIC_MODELS, [_anthropic_resp(), _anthropic_resp(thinking="생각", signature="s", tokens=40)])
    res = _native_run(prof, rec)
    assert res["terminal"] == "PASS"
    get = rec.requests[0]
    assert get.url.path == "/v1/models" and get.url.params["limit"] == "1000"
    assert get.headers["x-api-key"] == "sk-a" and get.headers["anthropic-version"] == "2023-06-01"
    assert "Authorization" not in get.headers
    assert str(rec.requests[1].url) == "https://api.anthropic.com/v1/messages"
    off, on = rec.post_bodies()
    assert off["thinking"] == {"type": "disabled"} and off["max_tokens"] == REAL_CAP
    assert on["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert on["messages"] == [{"role": "user", "content": REAL_PROBE_TABLE["prompts"]["on_check"]}]
    assert len(rec.post_bodies()) == 2       # sampling: medium → rejected → 전송 0건


def test_anthropic_effort_maps_to_output_config(gp):
    prof = gp.build_llm_profile("claude-sonnet-5", "RAG", "anthropic")
    rec = SeqRecorder(ANTHROPIC_MODELS, [_anthropic_resp(), _anthropic_resp(thinking="", signature="sig", tokens=None)])
    res = _native_run(prof, rec)
    assert res["terminal"] == "PASS"     # display=omitted: thinking 빈 문자열, signature 만으로 사고 판정
    on = rec.post_bodies()[1]
    assert on["output_config"] == {"effort": "medium"} and "effort" not in on
    assert "thinking" not in on


def test_anthropic_adaptive_effort_enables_thinking_for_opus(gp):
    """thinking 기본 꺼짐 모델(claude-opus-4-8)은 effort 만 보내면 사고 0 — set 칸이 thinking adaptive 도 실어야 한다.
    실측 2026-10-15: output_config.effort=high 단독 → thinking 블록 없음(3/3), + thinking adaptive → thinking 블록·375토큰."""
    prof = gp.build_llm_profile("claude-opus-4-8", "JUDGE", "anthropic")
    assert prof["reasoning_effort"]["injection"] == "anthropic_adaptive_effort"
    rec = SeqRecorder({"data": [{"id": "claude-opus-4-8"}], "has_more": False},
                      [_anthropic_resp(), _anthropic_resp(thinking="", signature="sig", tokens=None)])
    res = _native_run(prof, rec)
    assert res["terminal"] == "PASS"
    off, on = rec.post_bodies()[0], rec.post_bodies()[1]
    assert off["thinking"] == {"type": "disabled"} and "output_config" not in off
    assert on["thinking"] == {"type": "adaptive"} and on["output_config"] == {"effort": "high"}


def test_anthropic_stop_max_tokens_is_budget_exhausted(gp):
    prof = gp.build_llm_profile("claude-haiku-4-5-20251001", "RAG", "anthropic")
    rec = SeqRecorder(ANTHROPIC_MODELS, [_anthropic_resp(),
                                         _anthropic_resp(text=None, thinking="긴 생각", signature="s", stop="max_tokens")])
    assert _native_run(prof, rec)["terminal"] == "T_budget_exhausted"


def test_anthropic_off_probe_thinking_block_is_off_ignored(gp):
    prof = gp.build_llm_profile("claude-haiku-4-5-20251001", "RAG", "anthropic")
    rec = SeqRecorder(ANTHROPIC_MODELS, [_anthropic_resp(thinking="", signature="sig")])
    assert _native_run(prof, rec)["terminal"] == "T_off_ignored"


def test_anthropic_alias_not_listed_is_model_missing(gp):
    """Anthropic 목록은 dated id(claude-haiku-4-5-20251001)만 싣는다 — alias 는 정확 일치로 찾지 않는다(라이브 실측)."""
    prof = gp.build_llm_profile("claude-haiku-4-5", "RAG", "anthropic")
    assert _native_run(prof, SeqRecorder(ANTHROPIC_MODELS, []))["terminal"] == "T_model_missing"


def test_gemini_level_request_and_pass(gp):
    prof = gp.build_llm_profile("gemini-3.6-flash", "RAG", "google")
    rec = SeqRecorder(GEMINI_MODELS, [_gemini_resp(thoughts=284)])
    res = _native_run(prof, rec)
    assert res["terminal"] == "PASS"
    assert res["records"] == ["R_off_not_available_declared"]           # none = minimal (끌 수 없음 선언)
    get = rec.requests[0]
    assert get.url.path == "/v1beta/models" and get.url.params["pageSize"] == "1000"
    assert get.headers["x-goog-api-key"] == "g-key" and "key" not in get.url.params
    post = rec.requests[1]
    assert str(post.url) == "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent"
    (on,) = rec.post_bodies()                                            # sampling 전부 ignored → 0건
    assert on["generationConfig"] == {"maxOutputTokens": REAL_CAP, "thinkingConfig": {"thinkingLevel": "medium"}}
    assert on["contents"][0]["parts"] == [{"text": REAL_PROBE_TABLE["prompts"]["on_check"]}]
    assert "model" not in on


def test_gemini_thought_parts_excluded_from_content_and_counted(gp):
    prof = gp.build_llm_profile("gemini-3.6-flash", "RAG", "google")
    rec = SeqRecorder(GEMINI_MODELS, [_gemini_resp(text="12", thought_text="생각</think>", thoughts=0)])
    res = _native_run(prof, rec)
    assert res["terminal"] == "PASS"                                     # thought 파트 marker 는 content 누출 아님
    on_obs = [o for o in res["observations"] if o["node"] == "on_path"][0]
    assert on_obs["content_head"] == "12" and on_obs["thinking"] is True


def test_gemini_max_tokens_budget_exhausted(gp):
    prof = gp.build_llm_profile("gemini-3.6-flash", "RAG", "google")
    rec = SeqRecorder(GEMINI_MODELS, [_gemini_resp(text=None, thoughts=REAL_CAP, finish="MAX_TOKENS")])
    assert _native_run(prof, rec)["terminal"] == "T_budget_exhausted"


def test_gemini_effort_none_minimal_without_thinking_is_not_suppressed(gp):
    """effort=none → minimal: 라이브 실측 thoughtsTokenCount 0 (2026-10-15).
    none 은 '끔' 의도이므로 사고 0은 정상 — suppressed 로 판정하지 않는다(on_path 통과 후 sampling)."""
    prof = gp.build_llm_profile("gemini-3.6-flash", "RAG", "google")
    rec = SeqRecorder(GEMINI_MODELS, [_gemini_resp(thoughts=0)] * 4)
    res = _native_run(prof, rec, effective={"effort": "none"})
    assert res["terminal"] != "T_on_suppressed"
    assert "on_path" in res["path"]
    assert rec.post_bodies()[0]["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "minimal"}


NATIVE_CLASSES = ("openai", "anthropic", "google")


def test_every_native_model_rule_slot_builds_a_native_body():
    """정본 model_rules 의 native class 규칙: 6칸 전부 원문 본문으로 변환된다(주입 경로 ↔ body_paths 정합)."""
    vocab = REAL_DATA["vocabulary"]
    rules = [r for r in catalog.load_model_rules(PKG_ROOT) if r["provider_class"] in NATIVE_CLASSES]
    assert {r["provider_class"] for r in rules} == set(NATIVE_CLASSES)
    for rule in rules:
        spec = REAL_PROVIDER_CLASSES[rule["provider_class"]]["http"]
        for level in vocab["effort_levels"]:
            extras = inj.expand_slot(REAL_INJECTION, rule["injection"], rule["mapping"][level])
            body = pg.native_body(spec, rule["match"], "q", REAL_CAP, extras)
            assert pg._dig(body, spec["max_tokens_path"]) == REAL_CAP


def test_thinking_budget_below_probe_cap():
    """Anthropic: budget_tokens < max_tokens 필수 — 탐침 상한이 모든 예산 값보다 커야 거짓 T_rejected 가 없다."""
    budgets = [v for spec in REAL_INJECTION.values() if isinstance(spec, dict)
               for vals in spec.get("values", {}).values() for k, v in vals.items() if k.endswith("budget_tokens")]
    assert budgets and all(b < REAL_CAP for b in budgets)


def test_skill_evidence_and_answer_declared_for_every_provider_class():
    for pc in REAL_PROVIDER_CLASSES:
        fmt = pg.response_format(REAL_PROBE_TABLE, pc)
        assert set(fmt["evidence"]) == {"text", "tokens"}
        assert {"content", "finish_reason", "finish_length"} <= set(fmt)


# ════════════════ 실측 본문 = 해석기 인자 (같은 결과에서 나온다) ════════════════

def _subset(small: dict, big: dict) -> bool:
    return all(k in big and (_subset(v, big[k]) if isinstance(v, dict) else big[k] == v) for k, v in small.items())


@pytest.mark.parametrize("rule", catalog.load_model_rules(PKG_ROOT), ids=lambda r: f"{r['provider_class']}:{r['match']}")
def test_probe_body_equals_resolver_render_for_every_rule_and_level(skill, rule):
    """모든 정본 규칙 × 6단계: 실측 그래프 본문의 effort 인자 = 해석기 SDK 인자를 원문 HTTP 로 옮긴 것.

    (a) 그래프가 쓰는 칸 extras 는 해석기 sdk_kwargs 의 부분집합이다 — 같은 expand_slot 결과.
    (b) chat_completions 는 그 extras 를 extra_body 평탄화한 것이 본문에 그대로 있다.
    """
    sp = {"openai": "openai", "anthropic": "anthropic", "google": "google"}.get(rule["provider_class"], "site_vllm")
    prof = pb.build_llm_profile(rule["match"], "RAG", sp, skill)
    http = REAL_PROVIDER_CLASSES[rule["provider_class"]]["http"]
    for level in REAL_DATA["vocabulary"]["effort_levels"]:
        spec = rs.resolve(prof, {"effort": level}, data=REAL_DATA)
        extras = rs.slot_extras(spec, level)
        assert _subset(extras, spec["sdk_kwargs"]), (level, extras)
        body = pg.native_body(http, rule["match"], "q", REAL_CAP, extras)
        if http["api"] == pg.API_CHAT_COMPLETIONS:
            assert _subset(inj.to_http_body(extras), body), (level, body)
