"""preflight — v4-1 정적 스키마·v1/v2/v3 원문 HTTP·오케스트레이션·write-back 의 결정론 회귀 가드.

네트워크 없음: 원문 HTTP 는 httpx.MockTransport, 실측 그래프·runnable 은 결과 dict 로 대체한다.
프로필은 발행기(publish)가 배포본 정본으로 지금 발행하는 형식을 쓴다(발행 ↔ 검증 정합).
"""

from __future__ import annotations

import copy
import datetime as dt
import json

import httpx
import pytest
import yaml

from langchain_model_preset_gen import catalog
from langchain_model_preset_gen import preflight as p
from langchain_model_preset_gen import probe_graph as pg
from langchain_model_preset_gen import publish as pb
from langchain_model_preset_gen import resolver as rs

TODAY = dt.date(2026, 10, 15)
LLM_METHODS = ("invoke", "ainvoke", "stream", "astream", "batch", "abatch")
EMB_METHODS = ("embed_query", "embed_documents", "aembed_query", "aembed_documents")


@pytest.fixture
def llm(skill):
    return lambda model="granite-4.2-8b", sp="site_vllm", preset="RAG": pb.build_llm_profile(model, preset, sp, skill)


@pytest.fixture
def emb(skill):
    return lambda model="gemini-embedding-2", sp="google", dims=None: pb.build_embedding_profile(
        model, "RETRIEVAL", sp, dims, skill)


@pytest.fixture
def classes(skill):
    return catalog.load_data(skill)["provider_classes"]


def write_profile(skill, profile: dict):
    path = skill.parents[2] / "profiles" / pb.derive_filename(profile["model_name"], profile["preset"],
                                                               profile["serving_provider"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pb.render_yaml(profile, TODAY), encoding="utf-8")
    return path


def _mut(profile: dict, fn) -> dict:
    bad = copy.deepcopy(profile)
    fn(bad)
    return bad


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


# ════════════════ v4-1 정적 스키마 ════════════════

def test_v4_1_llm_published_profile_validates(llm, skill):
    out = p.check_v4_1_static(llm(), skill)
    assert out == {"kind": "llm", "injection": "ct_enable_thinking", "effort": "medium",
                   "slot": {"action": "set", "value": "enable"}, "probe_status": "probe_error"}


@pytest.mark.parametrize("model,sp", [("gpt-5.4-mini", "openai"), ("claude-sonnet-5", "anthropic"),
                                      ("gemini-3.6-flash", "google")])
def test_v4_1_native_profiles_validate(llm, skill, model, sp):
    assert p.check_v4_1_static(llm(model, sp), skill)["kind"] == "llm"


def test_v4_1_embedding_published_profile_validates(emb, skill):
    assert p.check_v4_1_static(emb(), skill) == {"kind": "embedding", "query_mode_surface": "task_title_prefix",
                                                 "native_dimensions": 3072, "probe_status": "probe_error"}


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d["reasoning_effort"]["mapping"].pop("xhigh"), "6칸"),
    (lambda d: d["reasoning_effort"].update(injection="nope"), "injection"),
    (lambda d: d["reasoning_effort"]["mapping"].update(low={"action": "boost"}), "action"),
    (lambda d: d["limits"].pop("max_retries"), "limits"),
    (lambda d: d["overridable"].append("model_name"), "overridable"),
    (lambda d: d["overridable"].append("top_k"), "overridable"),
    (lambda d: d["sampling"].update(top_k="maybe"), "sampling"),
    (lambda d: d["preset_values"].update(effort="turbo"), "effort"),
    (lambda d: d["preset_values"].pop("top_k"), "preset_values"),
    (lambda d: d["probe"].update(extra=1), "probe"),
    (lambda d: d["probe"].update(status="great"), "probe.status"),
    (lambda d: d["probe"].update(method="live_v4"), "probe.method"),
    (lambda d: d["probe"].update(terminal="T_nope"), "terminal"),
    (lambda d: d["probe"]["runnable"].pop("abatch"), "runnable"),
    (lambda d: d["probe"]["runnable"].update(invoke="maybe"), "runnable"),
    (lambda d: d.update(base_url="http://elsewhere/v1"), "잠금"),
])
def test_v4_1_llm_violations_raise(llm, skill, mutate, match):
    with pytest.raises(ValueError, match=match):
        p.check_v4_1_static(_mut(llm(), mutate), skill)


def test_v4_1_llm_key_order_violation_raises(llm, skill):
    prof = llm()
    reordered = {"kind": prof["kind"], "preset": prof["preset"],
                 **{k: v for k, v in prof.items() if k not in ("kind", "preset")}}
    with pytest.raises(ValueError, match="순서"):
        p.check_v4_1_static(reordered, skill)


def test_v4_1_llm_overridable_subset_allowed(llm, skill):
    assert p.check_v4_1_static(_mut(llm(), lambda d: d["overridable"].remove("effort")), skill)["kind"] == "llm"


def test_v4_1_llm_live_probe_record_validates(llm, skill):
    live = _mut(llm(), lambda d: d["probe"].update(status="ok", method="probe_graph_live", terminal="PASS",
                                                    measured_at="2026-10-15T00:00:00+00:00",
                                                    runnable={m: "ok" for m in LLM_METHODS}))
    assert p.check_v4_1_static(live, skill)["probe_status"] == "ok"


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d["supports"].update(query_mode_surface="none"), "supports"),
    (lambda d: d["supports"].update(normalization="weird"), "normalization"),
    (lambda d: d["limits"].update(response_deadline_seconds=None), "limits"),
    (lambda d: d["preset_values"].update(intent="similarity"), "preset_values"),
    (lambda d: d["probe"].update(method="probe_graph_live"), "probe.method"),
    (lambda d: d["overridable"].append("max_tokens"), "overridable"),
])
def test_v4_1_embedding_violations_raise(emb, skill, mutate, match):
    with pytest.raises(ValueError, match=match):
        p.check_v4_1_static(_mut(emb(), mutate), skill)


def test_v4_1_embedding_measured_normalization_allowed(emb, skill):
    assert p.check_v4_1_static(_mut(emb(), lambda d: d["supports"].update(normalization="builtin")), skill)


# ════════════════ v1 · v2 · v3 (MockTransport) ════════════════

def test_v1_google_strips_models_prefix_and_uses_header_key(llm, classes):
    seen = []

    def h(req):
        seen.append(req)
        return httpx.Response(200, json={"models": [{"name": "models/gemini-3.6-flash"}]})
    st, _ = p.check_v1_model_exists(rs.resolve(llm("gemini-3.6-flash", "google")), "gk", classes, _client(h))
    assert st == "PASS"
    assert seen[0].headers["x-goog-api-key"] == "gk" and seen[0].url.params["pageSize"] == "1000"


def test_v1_exact_match_only(llm, classes):
    def h(req):
        return httpx.Response(200, json={"data": [{"id": "claude-haiku-4-5-20251001"}]})
    st, _ = p.check_v1_model_exists(rs.resolve(llm("claude-haiku-4-5", "anthropic")), "a", classes, _client(h))
    assert st == "WARN"


def test_v1_http_error_and_transport_fail(llm, classes):
    st, msg = p.check_v1_model_exists(rs.resolve(llm("gpt-5.4-mini", "openai")), "k", classes,
                                      _client(lambda r: httpx.Response(401, text="bad key")))
    assert st == "FAIL" and "401" in msg

    def boom(req):
        raise httpx.ConnectError("refused", request=req)
    st, msg = p.check_v1_model_exists(rs.resolve(llm()), None, classes, _client(boom))
    assert st == "FAIL" and "ConnectError" in msg


def test_v1_embedding_models_path(emb, classes):
    seen = []

    def h(req):
        seen.append(req.url.path)
        return httpx.Response(200, json={"data": [{"id": "perplexity/pplx-embed-v1-4b"}]})
    spec = rs.resolve(emb("perplexity/pplx-embed-v1-4b", "openrouter"))
    assert p.check_v1_model_exists(spec, "k", classes, _client(h), "/embeddings/models")[0] == "PASS"
    assert seen == ["/api/v1/embeddings/models"]


def test_v2_local_loaded_reports_max_model_len(llm, classes):
    def h(req):
        assert "Authorization" not in req.headers
        return httpx.Response(200, json={"data": [{"id": "granite-4.2-8b", "max_model_len": 32768}]})
    st, msg = p.check_v2_local_loaded(rs.resolve(llm()), classes, _client(h))
    assert st == "PASS" and "32768" in msg


@pytest.mark.parametrize("status,verdict", [(200, "PASS"), (401, "FAIL"), (403, "FAIL"), (402, "FAIL"),
                                            (429, "FAIL"), (400, "WARN")])
def test_v3_status_mapping(llm, classes, status, verdict):
    def h(req):
        return httpx.Response(status, json={"ok": True}) if status == 200 else httpx.Response(status, text="x")
    assert p.check_v3_usable(rs.resolve(llm("gpt-5.4-mini", "openai")), "k", classes, _client(h))[0] == verdict


def test_v3_transport_failure_is_fail(llm, classes):
    def boom(req):
        raise httpx.ConnectError("refused", request=req)
    st, msg = p.check_v3_usable(rs.resolve(llm()), None, classes, _client(boom))
    assert st == "FAIL" and "ConnectError" in msg


def test_v3_native_request_shapes(llm, emb, classes):
    bodies = {}

    def h(req):
        bodies[req.url.host] = (req, json.loads(req.content))
        return httpx.Response(200, json={})
    p.check_v3_usable(rs.resolve(llm("gpt-5.4-mini", "openai")), "k", classes, _client(h))
    p.check_v3_usable(rs.resolve(llm("claude-sonnet-5", "anthropic")), "a", classes, _client(h))
    p.check_v3_usable(rs.resolve(emb()), "g", classes, _client(h))
    _, o = bodies["api.openai.com"]
    assert o["max_completion_tokens"] == p.V3_MAX_TOKENS and "max_tokens" not in o
    a_req, a = bodies["api.anthropic.com"]
    assert a_req.headers["x-api-key"] == "a" and a["max_tokens"] == p.V3_MAX_TOKENS
    g_req, gb = bodies["generativelanguage.googleapis.com"]
    assert g_req.url.path.endswith("/models/gemini-embedding-2:embedContent") and "content" in gb


# ════════════════ 오케스트레이션 · write-back ════════════════

def _pg_result(terminal="PASS") -> dict:
    return {"terminal": terminal, "status": "ok" if terminal == "PASS" else "probe_error",
            "diagnosis": f"diag {terminal}", "path": ["connect"], "records": [], "observations": [], "steps": []}


def _patch_live(monkeypatch, terminal="PASS", runnable_status="ok", kind="llm", extra=None):
    calls = {}

    def fake_probe(spec, key, **kw):
        calls["pg_spec"] = spec
        return _pg_result(terminal)

    def fake_runnable(spec, key, data):
        calls["run_spec"] = spec
        methods = EMB_METHODS if kind == "embedding" else LLM_METHODS
        status = "ok" if runnable_status == "ok" else "probe_error"
        return {"runnable": {m: runnable_status for m in methods}, "status": status, "target": "X", **(extra or {})}
    monkeypatch.setattr(pg, "run_probe", fake_probe)
    monkeypatch.setattr(p, "check_runnable", fake_runnable)
    return calls


def test_preflight_v4_1_only_makes_no_http(llm, skill, capsys):
    path = write_profile(skill, llm())

    def no_http(req):
        raise AssertionError(f"HTTP 호출 금지: {req.url}")
    assert p.preflight(path, ["v4-1"], skill_dir=skill, client=_client(no_http)) == 0
    assert "V4-1 PASS" in capsys.readouterr().out


def test_preflight_v1_auto_adds_v2_for_self_hosted(llm, skill, capsys):
    path = write_profile(skill, llm())
    h = lambda req: httpx.Response(200, json={"data": [{"id": "granite-4.2-8b"}]})  # noqa: E731
    assert p.preflight(path, ["v1"], skill_dir=skill, client=_client(h)) == 0
    out = capsys.readouterr().out
    assert "V1   PASS" in out and "V2   PASS" in out


def test_preflight_unknown_check_raises(llm, skill):
    with pytest.raises(ValueError, match="알 수 없는 검사"):
        p.preflight(write_profile(skill, llm()), ["v4-2"], skill_dir=skill)


def test_preflight_pg_reports_terminal_and_fails_rc(llm, skill, monkeypatch, capsys):
    path = write_profile(skill, llm())
    calls = _patch_live(monkeypatch, terminal="T_on_suppressed")
    assert p.preflight(path, ["pg"], override={"effort": "high"}, skill_dir=skill) == 1
    assert calls["pg_spec"]["effective"]["effort"] == "high", "override 는 해석기를 거쳐 실측 그래프로 간다"
    out = capsys.readouterr().out
    assert "PG   FAIL terminal=T_on_suppressed" in out and "diag T_on_suppressed" in out


def test_preflight_locked_override_rejected_before_probe(llm, skill, monkeypatch):
    path = write_profile(skill, llm())
    _patch_live(monkeypatch)
    with pytest.raises(rs.LockedKeyError):
        p.preflight(path, ["pg"], override={"base_url": "http://x/v1"}, skill_dir=skill)


def test_preflight_pg_on_embedding_raises(emb, skill):
    with pytest.raises(ValueError, match="LLM 전용"):
        p.preflight(write_profile(skill, emb()), ["pg"], skill_dir=skill)


def test_write_back_llm_pass_and_six_ok(llm, skill, monkeypatch):
    prof = llm()
    path = write_profile(skill, prof)
    before = path.read_text(encoding="utf-8")
    _patch_live(monkeypatch)
    assert p.preflight(path, ["pg", "runnable"], write_back_flag=True, skill_dir=skill, today=TODAY) == 0
    after = path.read_text(encoding="utf-8")
    probe = yaml.safe_load(after)["probe"]
    assert probe["status"] == "ok" and probe["method"] == "probe_graph_live" and probe["terminal"] == "PASS"
    assert probe["runnable"] == {m: "ok" for m in LLM_METHODS} and probe["measured_at"]
    lines_b, lines_a = before.splitlines(), after.splitlines()
    assert lines_a[0] == lines_b[0]
    assert lines_a[1] == "# preset 출처: RAG (스킬 presets/, 발행일 2026-10-15) · 실측: probe_graph live 2026-10-15"
    body = lambda text: text.split("\n", 2)[2].split("\nprobe:")[0]  # noqa: E731
    assert body(after) == body(before)
    assert list(yaml.safe_load(after)) == list(prof)


def test_write_back_preserves_user_comment(llm, skill, monkeypatch):
    path = write_profile(skill, llm())
    raw = path.read_text(encoding="utf-8").replace("limits:", "# 사용자 메모: timeout 상향 검토\nlimits:")
    path.write_text(raw, encoding="utf-8")
    _patch_live(monkeypatch)
    p.preflight(path, ["pg", "runnable"], write_back_flag=True, skill_dir=skill, today=TODAY)
    assert "# 사용자 메모: timeout 상향 검토\nlimits:" in path.read_text(encoding="utf-8")


def test_write_back_legacy_header_line1_kept(llm, skill, monkeypatch):
    """머리말 1줄째(구 스킬명)는 그대로 두고 2줄째 '실측' 칸만 바꾼다 — 기존 프로필 재발행 없이 실측 갱신."""
    path = write_profile(skill, llm())
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    lines[0] = "# model-preset-gen 발행 — 사용자 편집 가능.\n"
    path.write_text("".join(lines), encoding="utf-8")
    _patch_live(monkeypatch)
    p.preflight(path, ["pg", "runnable"], write_back_flag=True, skill_dir=skill, today=TODAY)
    assert path.read_text(encoding="utf-8").splitlines()[0] == "# model-preset-gen 발행 — 사용자 편집 가능."


@pytest.mark.parametrize("terminal,runnable_status", [("PASS", "probe_error"), ("PASS", "unsupported"),
                                                      ("T_budget_exhausted", "ok"), ("U_undeterminable", "ok")])
def test_write_back_llm_status_probe_error_unless_pass_and_all_ok(llm, skill, monkeypatch, terminal, runnable_status):
    path = write_profile(skill, llm())
    _patch_live(monkeypatch, terminal=terminal, runnable_status=runnable_status)
    assert p.preflight(path, ["pg", "runnable"], write_back_flag=True, skill_dir=skill, today=TODAY) == 1
    probe = yaml.safe_load(path.read_text(encoding="utf-8"))["probe"]
    assert probe["status"] == "probe_error" and probe["terminal"] == terminal


def test_write_back_requires_pg_and_runnable(llm, skill, monkeypatch):
    path = write_profile(skill, llm())
    before = path.read_bytes()
    _patch_live(monkeypatch)
    with pytest.raises(ValueError, match="write-back"):
        p.preflight(path, ["runnable"], write_back_flag=True, skill_dir=skill)
    assert path.read_bytes() == before


def test_write_back_with_override_refused(llm, skill, monkeypatch):
    path = write_profile(skill, llm())
    _patch_live(monkeypatch)
    with pytest.raises(ValueError, match="override"):
        p.preflight(path, ["pg", "runnable"], write_back_flag=True, override={"effort": "high"}, skill_dir=skill)


def test_write_back_not_written_when_check_errors_midway(llm, skill, monkeypatch):
    path = write_profile(skill, llm())
    before = path.read_bytes()
    _patch_live(monkeypatch)

    def boom(*a, **k):
        raise pg.UndefinedTransition("on_path", "empty_content")
    monkeypatch.setattr(pg, "run_probe", boom)
    with pytest.raises(pg.UndefinedTransition):
        p.preflight(path, ["pg", "runnable"], write_back_flag=True, skill_dir=skill)
    assert path.read_bytes() == before


def test_no_write_back_leaves_file(llm, skill, monkeypatch):
    path = write_profile(skill, llm())
    before = path.read_bytes()
    _patch_live(monkeypatch)
    assert p.preflight(path, ["pg", "runnable"], skill_dir=skill) == 0
    assert path.read_bytes() == before


def test_runnable_fail_rc_and_label(llm, skill, monkeypatch, capsys):
    path = write_profile(skill, llm())
    _patch_live(monkeypatch, runnable_status="probe_error")
    assert p.preflight(path, ["runnable"], skill_dir=skill) == 1
    assert "RUN  FAIL" in capsys.readouterr().out


def test_write_back_embedding_semantics(emb, skill, monkeypatch, no_env_keys):
    (skill.parents[2] / "keys" / "k.env").write_text("OPENROUTER_API_KEY=k\n", encoding="utf-8")
    prof = emb("perplexity/pplx-embed-v1-4b", "openrouter")
    path = write_profile(skill, prof)
    before = path.read_text(encoding="utf-8")
    _patch_live(monkeypatch, kind="embedding", extra={"norm_measured": 16.353, "measured_at_dimensions": 2560})
    assert p.preflight(path, ["runnable"], write_back_flag=True, skill_dir=skill, today=TODAY) == 0
    after = path.read_text(encoding="utf-8")
    out = yaml.safe_load(after)
    assert out["probe"]["status"] == "ok" and out["probe"]["method"] == "live_v5"
    assert out["probe"]["norm_measured"] == 16.353 and out["probe"]["measured_at_dimensions"] == 2560
    assert out["probe"]["requires"] == "langchain-openai>=1.0,<2.0"
    assert out["supports"]["normalization"] == "required"
    assert after.splitlines()[1].endswith("· 실측: live_v5 2026-10-15")
    body = lambda text: text.split("\n", 2)[2].split("\nsupports:")[0]  # noqa: E731
    assert body(after) == body(before)
    assert list(out) == list(prof)


def test_write_back_rejects_foreign_header(llm, skill, monkeypatch):
    path = write_profile(skill, llm())
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    lines[1] = "# 손으로 쓴 머리말\n"
    path.write_text("".join(lines), encoding="utf-8")
    before = path.read_bytes()
    _patch_live(monkeypatch)
    with pytest.raises(ValueError, match="머리말"):
        p.preflight(path, ["pg", "runnable"], write_back_flag=True, skill_dir=skill, today=TODAY)
    assert path.read_bytes() == before


def test_preflight_missing_api_key_raises(llm, skill, no_env_keys):
    with pytest.raises(KeyError, match="OPENAI_API_KEY"):
        p.preflight(write_profile(skill, llm("gpt-5.4-mini", "openai")), ["v4-1"], skill_dir=skill)
