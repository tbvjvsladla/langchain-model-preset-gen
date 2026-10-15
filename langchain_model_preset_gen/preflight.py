"""preflight — 프로필을 정적·실측 검증하고 (선택) 실측 결과를 프로필에 기록한다.

검사 (--checks, 콤마 구분, 실행 순서 고정):
  v1       모델 실재   : provider 모델 목록에 model_name 이 정확히 있는가.
  v2       로컬 적재   : 무인증 자체 호스팅이면 적재 여부 + vLLM max_model_len. v1 을 요청하면 자동 추가.
  v3       사용가능성  : 미니 호출 1회 (200 = OK / 401·403 = 키 불량 / 402·429 = 잔액·쿼터 / 전송 실패 = FAIL).
  v4-1     정적 스키마 : API 0회. 발행 형식·잠금 필드 = provider 표·6칸·sampling·probe 블록 형태.
  pg       실측 그래프 : 끄기·켜기 + sampling 수용 (LLM 전용, probe_graph).
  runnable 호출 방식   : 해석기 객체 그대로 LLM 6종 / 임베딩 4종 + 노름·차원 (프로필 probe 에 기록되는 대상).
  workflow 교정 후 실측: configs workflow_probe 가 감싼 workflow 로 같은 호출 방식을 돌린다. 결과는 출력만 하고
                       프로필에 기록하지 않는다 — 프로필 probe 는 모델·플랫폼의 사실이고, workflow 결과는 그 workflow 의 사실이다.
--write-back (검사가 중간에 예외로 멈추면 쓰지 않는다. --override 와 같이 쓸 수 없다):
  LLM    : pg 와 runnable 이 필요하다. probe = {status, method: probe_graph_live, measured_at, terminal, runnable}.
           status = ok ⇔ pg 종착 PASS AND runnable 6종 전부 ok. 그 밖은 probe_error.
  임베딩 : runnable 이 필요하다. probe.{status, method: live_v5, measured_at, measured_with, requires,
           measured_at_dimensions, norm_measured, runnable} + supports.normalization 파생(노름 ≈1 → builtin, 아니면 required).
  공통   : 머리말 2줄째 '실측: …' 을 방법·날짜로 바꾸고, LLM 은 `probe:`, 임베딩은 `supports:` 줄부터 끝까지만 다시 쓴다.
           나머지 내용·키 순서·사용자 주석은 바이트 그대로 둔다. 쓰기 전에 결과를 v4-1 로 다시 검증한다.
실측 하네스다 — HTTP 예외는 잡되 삼키지 않고 명시 결과(FAIL/WARN·probe_error)로 사상한다.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import re
import sys
from pathlib import Path

import httpx
import yaml

from . import catalog
from . import probe_graph as pg
from . import probe_runnable as pr
from . import publish as pb
from . import resolver as rs
from . import settings as st
from . import workflow_probe as wp

SKILL_DIR = catalog.SKILL_DIR
CHECK_ORDER = ("v1", "v2", "v3", "v4-1", "pg", "runnable", "workflow")
WORKFLOW_CHECK = "workflow"

LLM_LIVE_PROBE_METHOD = "probe_graph_live"
EMBEDDING_LIVE_PROBE_METHOD = "live_v5"
LLM_PROBE_KEYS = ("status", "method", "measured_at", "terminal", "runnable")
REASONING_EFFORT_KEYS = ("injection", "mapping", "end_marker", "basis")

HEADER_MEASURED_RE = re.compile(r"· 실측: .*$")
HEADER_MEASURED_LLM = "· 실측: probe_graph live {date}"
HEADER_MEASURED_EMBEDDING = "· 실측: live_v5 {date}"
LLM_REWRITE_FROM = "probe"
EMBEDDING_REWRITE_FROM = "supports"

HTTP_OK = 200
HTTP_AUTH_FAIL = (401, 403)
HTTP_BALANCE_FAIL = (402, 429)
HTTP_CHECK_TIMEOUT_SECONDS = 60     # v1·v3 상한 — 프로필 timeout 과 무관한 하네스 값
V3_MAX_TOKENS = 16                  # 접속·인증·잔액 확인용 최소 출력
V3_PROMPT = "hi"
MSG_HEAD_CHARS = 160


# ════════════════ v1 · v2 · v3 (원문 HTTP) ════════════════

def _http(client: httpx.Client, method: str, url: str, **kw) -> tuple[int | None, dict | str]:
    """요청 1회 → (status, json | 오류 문자열). 전송 실패는 status None 으로 표면화."""
    try:
        r = client.request(method, url, timeout=HTTP_CHECK_TIMEOUT_SECONDS, **kw)
    except httpx.HTTPError as e:
        return None, f"{type(e).__name__}: {e}"
    return (r.status_code, r.json()) if r.status_code == HTTP_OK else (r.status_code, r.text[:MSG_HEAD_CHARS])


def _http_spec(spec: dict, provider_classes: dict) -> dict:
    return provider_classes[spec["provider_class"]]["http"]


def check_v1_model_exists(spec: dict, key: str | None, provider_classes: dict, client: httpx.Client,
                          embedding_models_path: str | None = None) -> tuple[str, str]:
    """embedding_models_path: 임베딩 모델을 별도 목록으로 주는 provider 만 providers 에 선언한다(예: OpenRouter)."""
    http = _http_spec(spec, provider_classes)
    path = http["models_path"] if embedding_models_path is None else embedding_models_path
    status, body = _http(client, "GET", f"{spec['http_base_url']}{path}", params=http["models_query"] or None,
                         headers=pg.auth_headers(http, key))
    if status != HTTP_OK:
        return "FAIL", f"모델 목록 HTTP {status}: {str(body)[:MSG_HEAD_CHARS]}"
    ids = pg.listed_model_ids(body, http)
    if spec["model_name"] in ids:
        return "PASS", f"model 있음 ({len(ids)} ids)"
    return "WARN", f"model 이 목록에 없음 ({len(ids)} ids) — 서빙 슬러그 확인 (pg connect 는 T_model_missing)"


def check_v2_local_loaded(spec: dict, provider_classes: dict, client: httpx.Client) -> tuple[str, str]:
    http = _http_spec(spec, provider_classes)
    status, body = _http(client, "GET", f"{spec['http_base_url']}{http['models_path']}",
                         headers=pg.auth_headers(http, None))
    if status != HTTP_OK:
        return "FAIL", f"로컬 목록 HTTP {status}: {str(body)[:MSG_HEAD_CHARS]}"
    items = {item[http["models_id"]]: item for item in body[http["models_items"]]}
    if spec["model_name"] not in items:
        return "FAIL", f"로컬 엔드포인트에 미적재 — 적재 모델 {sorted(items)}"
    max_len = items[spec["model_name"]].get("max_model_len")   # vLLM 확장 필드 — 다른 서버엔 없을 수 있다(표시 전용)
    return "PASS", f"로컬 적재 (max_model_len={max_len})"


def check_v3_usable(spec: dict, key: str | None, provider_classes: dict, client: httpx.Client) -> tuple[str, str]:
    http = _http_spec(spec, provider_classes)
    model = spec["model_name"]
    if spec["kind"] == rs.KIND_EMBEDDING:
        path = http["embed_path"].format(model=model)
        body = ({"model": model, "input": V3_PROMPT} if http["api"] == pg.API_CHAT_COMPLETIONS
                else {"content": {"parts": [{"text": V3_PROMPT}]}})
    else:
        path = http["chat_path"].format(model=model)
        body = pg.native_body(http, model, V3_PROMPT, V3_MAX_TOKENS, {})
    status, text = _http(client, "POST", f"{spec['http_base_url']}{path}", json=body, headers=pg.auth_headers(http, key))
    if status == HTTP_OK:
        return "PASS", "미니 호출 OK (인증·잔액 OK)"
    if status is None:
        return "FAIL", f"전송 실패: {text}"
    if status in HTTP_AUTH_FAIL:
        return "FAIL", f"{status} 키 불량: {str(text)[:MSG_HEAD_CHARS]}"
    if status in HTTP_BALANCE_FAIL:
        return "FAIL", f"{status} 잔액/쿼터: {str(text)[:MSG_HEAD_CHARS]}"
    return "WARN", f"HTTP {status}: {str(text)[:MSG_HEAD_CHARS]}"


# ════════════════ v4-1 정적 스키마 ════════════════

def _require_subset_unique(values: list, allowed, what: str) -> None:
    unknown = [v for v in values if v not in allowed]
    if unknown or len(set(values)) != len(values):
        raise ValueError(f"{what} 위반 — 어휘 밖 {unknown}, 중복 {len(values) - len(set(values))}건 (허용: {list(allowed)})")


def _check_load_block(profile: dict, providers: dict) -> None:
    prov = catalog.resolve_provider(profile["serving_provider"], providers)
    drift = {k: (profile[k], prov[k]) for k in catalog.PROVIDER_KEYS if profile[k] != prov[k]}
    if drift:
        raise ValueError(f"잠금 필드가 provider 표와 다르다: {drift} — 재발행하거나 provider 표를 확인한다")


def _check_key_order(profile: dict, canonical: dict) -> None:
    if list(profile) != list(canonical):
        raise ValueError(f"최상위 키 순서·집합 위반 — 프로필 {list(profile)} ≠ 발행 형식 {list(canonical)}")


def _check_runnable(runnable: dict, methods: list, vocab: dict, what: str) -> None:
    catalog.require_exact_keys(runnable, methods, what)
    bad = {m: v for m, v in runnable.items() if v not in vocab["runnable_status_values"]}
    if bad:
        raise ValueError(f"{what} 값 어휘 위반: {bad} (허용: {vocab['runnable_status_values']})")


def check_v4_1_static(profile: dict, skill_dir: Path = SKILL_DIR) -> dict:
    """정적 검증 (API 0회). 위반은 ValueError/KeyError. 머리말 주석은 검증 대상이 아니다."""
    data = catalog.load_data(skill_dir)
    vocab = data["vocabulary"]
    providers = catalog.merge_providers(data["providers"], st.load_settings(skill_dir)["providers"])
    _check_load_block(profile, providers)
    if profile["kind"] == rs.KIND_EMBEDDING:
        return _check_v4_1_embedding(profile, vocab, skill_dir)
    if profile["kind"] != rs.KIND_LLM:
        raise ValueError(f"kind 는 llm|embedding: {profile['kind']!r}")
    canonical = pb.build_llm_profile(profile["model_name"], profile["preset"], profile["serving_provider"], skill_dir)
    _check_key_order(profile, canonical)
    _require_subset_unique(profile["overridable"], rs.LLM_OVERRIDABLE, "overridable")
    re_ = profile["reasoning_effort"]
    catalog.require_exact_keys(re_, REASONING_EFFORT_KEYS, "reasoning_effort")
    rule = catalog.validate_model_rule({"match": profile["model_name"], **re_, "sampling": profile["sampling"]},
                                       vocab, data["injection"])
    pv = profile["preset_values"]
    catalog.require_exact_keys(pv, [*vocab["sampling_keys"], "effort"], "preset_values")
    if pv["effort"] not in vocab["effort_levels"]:
        raise ValueError(f"preset_values.effort 어휘 위반: {pv['effort']!r} (허용: {vocab['effort_levels']})")
    catalog.require_exact_keys(profile["limits"], rs.LLM_LIMIT_KEYS, "limits")
    probe = profile["probe"]
    catalog.require_exact_keys(probe, LLM_PROBE_KEYS, "probe")
    if probe["status"] not in vocab["probe_status_values"]:
        raise ValueError(f"probe.status 어휘 위반: {probe['status']!r} (허용: {vocab['probe_status_values']})")
    if probe["method"] not in (pb.LLM_STATIC_PROBE_METHOD, LLM_LIVE_PROBE_METHOD):
        raise ValueError(f"probe.method 어휘 위반: {probe['method']!r}")
    if probe["terminal"] is not None and probe["terminal"] not in data["probe_graph"]["diagnoses"]:
        raise ValueError(f"probe.terminal 이 전이표 종착이 아니다: {probe['terminal']!r}")
    _check_runnable(probe["runnable"], vocab["runnable_methods_llm"], vocab, "probe.runnable")
    return {"kind": rs.KIND_LLM, "injection": rule["injection"], "effort": pv["effort"],
            "slot": rule["mapping"][pv["effort"]], "probe_status": probe["status"]}


def _check_v4_1_embedding(profile: dict, vocab: dict, skill_dir: Path) -> dict:
    canonical = pb.build_embedding_profile(profile["model_name"], profile["preset"], profile["serving_provider"],
                                           profile["dimensions"], skill_dir)
    _check_key_order(profile, canonical)
    _require_subset_unique(profile["overridable"], rs.EMBEDDING_OVERRIDABLE, "overridable")
    if profile["preset_values"] != canonical["preset_values"]:
        raise ValueError(f"preset_values {profile['preset_values']} ≠ preset {canonical['preset_values']}")
    catalog.require_exact_keys(profile["limits"], rs.EMBEDDING_LIMIT_KEYS, "limits")
    sup, want = profile["supports"], canonical["supports"]
    catalog.require_exact_keys(sup, list(want), "supports")
    drift = {k: (sup[k], want[k]) for k in want if k != "normalization" and sup[k] != want[k]}
    if drift:
        raise ValueError(f"supports 가 embedding_capability 와 다르다: {drift}")
    if sup["normalization"] is not None and sup["normalization"] not in vocab["embedding_normalization_values"]:
        raise ValueError(f"supports.normalization 어휘 위반: {sup['normalization']!r}")
    probe = profile["probe"]
    catalog.require_exact_keys(probe, list(canonical["probe"]), "probe")
    if probe["status"] not in vocab["probe_status_values"]:
        raise ValueError(f"probe.status 어휘 위반: {probe['status']!r}")
    if probe["method"] not in (pb.EMBEDDING_STATIC_PROBE_METHOD, EMBEDDING_LIVE_PROBE_METHOD):
        raise ValueError(f"probe.method 어휘 위반: {probe['method']!r}")
    _check_runnable(probe["runnable"], vocab["runnable_methods_embedding"], vocab, "probe.runnable")
    return {"kind": rs.KIND_EMBEDDING, "query_mode_surface": sup["query_mode_surface"],
            "native_dimensions": sup["native_dimensions"], "probe_status": probe["status"]}


# ════════════════ runnable ════════════════

def _build_model(spec: dict, key: str | None, data: dict):
    if spec["kind"] == rs.KIND_EMBEDDING:
        return rs.build_embeddings(spec, key, data["probe_graph"]["prompts"]["embedding_query_instruction"])
    return rs.build_chat_model(spec, key)


def _run_call_modes(target, spec: dict, data: dict) -> dict:
    is_emb = spec["kind"] == rs.KIND_EMBEDDING
    vocab = data["vocabulary"]
    methods = vocab["runnable_methods_embedding"] if is_emb else vocab["runnable_methods_llm"]
    report = pr.check_runnable(target, methods, data["probe_graph"]["prompts"]["runnable_check"], embedding=is_emb)
    report["target"] = type(target).__name__
    return report


def check_runnable(spec: dict, key: str | None, data: dict) -> dict:
    """해석기 객체 그대로 호출 방식 실측(프로필에 기록되는 결과). 임베딩은 raw 노름·차원도 잰다."""
    model = _build_model(spec, key, data)
    report = _run_call_modes(model, spec, data)
    if spec["kind"] == rs.KIND_EMBEDDING and report["runnable"]["embed_query"] == pr.RUNNABLE_OK:
        report.update(pr.measure_embedding(model.inner, data["probe_graph"]["prompts"]["runnable_check"]))
    return report


def check_workflow(spec: dict, key: str | None, data: dict, factory) -> dict:
    """factory(해석기 객체)가 돌려준 workflow 로 같은 호출 방식 실측 — 기록하지 않는 교정 후 확인."""
    target = wp.build_target(factory, _build_model(spec, key, data), embedding=spec["kind"] == rs.KIND_EMBEDDING)
    return _run_call_modes(target, spec, data)


# ════════════════ write-back ════════════════

def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def llm_probe_record(pg_result: dict, runnable: dict, measured_at: str) -> dict:
    all_ok = all(v == pr.RUNNABLE_OK for v in runnable["runnable"].values())
    status = pr.PROBE_OK if pg_result["terminal"] == pg.PASS_TERMINAL and all_ok else pr.PROBE_ERROR
    return {"status": status, "method": LLM_LIVE_PROBE_METHOD, "measured_at": measured_at,
            "terminal": pg_result["terminal"], "runnable": dict(runnable["runnable"])}


def embedding_written(profile: dict, runnable: dict, measured_at: str, provider_classes: dict) -> dict:
    pc = profile["provider_class"]
    supports = dict(profile["supports"])
    probe = dict(profile["probe"])
    probe["runnable"] = dict(runnable["runnable"])
    if "norm_measured" in runnable:
        probe["norm_measured"] = runnable["norm_measured"]
        probe["measured_at_dimensions"] = runnable["measured_at_dimensions"]
        supports["normalization"] = pr.derive_normalization(runnable["norm_measured"])
    probe.update({"status": runnable["status"], "method": EMBEDDING_LIVE_PROBE_METHOD, "measured_at": measured_at,
                  "measured_with": pb.measured_with(pc, provider_classes),
                  "requires": pb.requires_range(pc, provider_classes)})
    return {"supports": supports, "probe": probe}


def rewrite_profile_text(raw: str, tail: dict, rewrite_from: str, header_measured: str) -> str:
    lines = raw.splitlines(keepends=True)
    if HEADER_MEASURED_RE.search(lines[1]) is None:
        raise ValueError(f"머리말 2줄째에 '· 실측: …' 칸이 없다 — 이 스킬의 발행 형식이 아니다: {lines[1]!r}")
    lines[1] = HEADER_MEASURED_RE.sub(header_measured, lines[1].rstrip("\n")) + "\n"
    starts = [i for i, line in enumerate(lines) if line.startswith(f"{rewrite_from}:")]
    if len(starts) != 1:
        raise ValueError(f"최상위 `{rewrite_from}:` 줄이 정확히 1개가 아니다({len(starts)}개) — 다시 쓸 위치를 정할 수 없다")
    return "".join(lines[:starts[0]]) + pb.dump_yaml(tail)


def write_back(profile_path: str | Path, profile: dict, pg_result: dict | None, runnable: dict,
               today: datetime.date, measured_at: str, skill_dir: Path = SKILL_DIR) -> dict:
    path = Path(profile_path)
    raw = path.read_text(encoding="utf-8")
    if profile["kind"] == rs.KIND_EMBEDDING:
        tail = embedding_written(profile, runnable, measured_at, catalog.load_data(skill_dir)["provider_classes"])
        rewrite_from, header = EMBEDDING_REWRITE_FROM, HEADER_MEASURED_EMBEDDING
    else:
        tail = {"probe": llm_probe_record(pg_result, runnable, measured_at)}
        rewrite_from, header = LLM_REWRITE_FROM, HEADER_MEASURED_LLM
    expected = {**profile, **tail}
    check_v4_1_static(expected, skill_dir)
    text = rewrite_profile_text(raw, tail, rewrite_from, header.format(date=today.isoformat()))
    if yaml.safe_load(text) != expected:
        raise ValueError("write-back 결과를 다시 읽은 내용이 기대와 다르다 — 쓰지 않는다")
    path.write_text(text, encoding="utf-8")
    return expected


# ════════════════ 오케스트레이션 ════════════════

def _normalize_checks(checks: list[str], profile: dict) -> list[str]:
    unknown = [c for c in checks if c not in CHECK_ORDER]
    if unknown:
        raise ValueError(f"알 수 없는 검사: {unknown} (허용: {list(CHECK_ORDER)})")
    wanted = set(checks) | ({"v2"} if "v1" in checks and profile["api_key_env"] is None else set())
    return [c for c in CHECK_ORDER if c in wanted]


def preflight(profile_path: str | Path, checks: list[str], *, write_back_flag: bool = False,
              override: dict | None = None, skill_dir: Path = SKILL_DIR, client: httpx.Client | None = None,
              today: datetime.date | None = None) -> int:
    """검사 실행 → 종료코드(0 = FAIL 없음). 검사 중 예외는 그대로 올라간다(write-back 전에 멈춤)."""
    profile = catalog.load_yaml(Path(profile_path))
    order = _normalize_checks(checks, profile)
    is_emb = profile["kind"] == rs.KIND_EMBEDDING
    if write_back_flag:
        if WORKFLOW_CHECK in order:
            raise ValueError("--write-back 은 workflow 검사와 함께 쓸 수 없다 — workflow 결과는 프로필에 기록하지 않는다")
        if override:
            raise ValueError("--write-back 은 --override 와 함께 쓸 수 없다 — 실측값이 프로필 기본값이 아니다")
        need = {"runnable"} if is_emb else {"pg", "runnable"}
        if not need <= set(order):
            raise ValueError(f"--write-back 에는 검사 {sorted(need)} 가 필요하다 (LLM: 그래프 PASS AND 6종 ok)")
    if is_emb and "pg" in order:
        raise ValueError("pg(실측 그래프)는 LLM 전용이다")
    data = catalog.load_data(skill_dir)
    cfg = st.load_settings(skill_dir)
    factory = wp.configured_factory(skill_dir) if WORKFLOW_CHECK in order else None
    if WORKFLOW_CHECK in order and factory is None:
        raise ValueError("workflow 검사에는 configs/config.yaml workflow_probe 가 필요하다 (\"패키지.모듈:함수\")")
    providers = catalog.merge_providers(data["providers"], cfg["providers"])
    spec = rs.resolve(profile, override, data=data)
    key = st.resolve_api_key(spec["api_key_env"], cfg["env_file_path"])
    classes = data["provider_classes"]
    print(f"[preflight] {profile['model_name']} via {profile['serving_provider']} "
          f"({profile['provider_class']}, kind={profile['kind']})")
    rc = 0
    pg_result = None
    runnable = None
    with httpx.Client() if client is None else contextlib.nullcontext(client) as http:
        if "v1" in order:
            emb_path = providers[profile["serving_provider"]].get("embedding_models_path") if is_emb else None
            status, msg = check_v1_model_exists(spec, key, classes, http, emb_path)
            print(f"  V1   {status:4} {msg}")
            rc |= status == "FAIL"
        if "v2" in order:
            status, msg = check_v2_local_loaded(spec, classes, http)
            print(f"  V2   {status:4} {msg}")
            rc |= status == "FAIL"
        if "v3" in order:
            status, msg = check_v3_usable(spec, key, classes, http)
            print(f"  V3   {status:4} {msg}")
            rc |= status == "FAIL"
        if "v4-1" in order:
            print(f"  V4-1 PASS {check_v4_1_static(profile, skill_dir)}")
        if "pg" in order:
            pg_result = pg.run_probe(spec, key, probe_table=data["probe_graph"], provider_classes=classes, client=http)
            status = "PASS" if pg_result["status"] == pr.PROBE_OK else "FAIL"
            print(f"  PG   {status} terminal={pg_result['terminal']} path={pg_result['path']} "
                  f"records={pg_result['records']}\n       진단: {pg_result['diagnosis']}")
            rc |= status == "FAIL"
    if "runnable" in order:
        runnable = check_runnable(spec, key, data)
        status = "PASS" if runnable["status"] == pr.PROBE_OK else "FAIL"
        extra = {k: runnable[k] for k in ("norm_measured", "measured_at_dimensions") if k in runnable}
        print(f"  RUN  {status} target={runnable['target']} runnable={runnable['runnable']} {extra or ''}")
        rc |= status == "FAIL"
    if WORKFLOW_CHECK in order:
        workflow = check_workflow(spec, key, data, factory)
        status = "PASS" if workflow["status"] == pr.PROBE_OK else "FAIL"
        print(f"  WF   {status} target={workflow['target']} runnable={workflow['runnable']} (기록하지 않음)")
        rc |= status == "FAIL"
    if write_back_flag:
        written = write_back(profile_path, profile, pg_result, runnable, today or datetime.date.today(), _now_iso(),
                             skill_dir)
        print(f"  write-back → {profile_path} (probe.status={written['probe']['status']})")
    return int(rc)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="langchain-model-preset-gen preflight (v1/v2/v3/v4-1/pg/runnable + write-back)")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--checks", default="v1,v3,v4-1", help=f"콤마 구분 {','.join(CHECK_ORDER)}")
    ap.add_argument("--override", help="실효값 override JSON (pg·runnable 에 적용, write-back 과 함께 쓸 수 없음)")
    ap.add_argument("--write-back", action="store_true", help="실측 결과를 프로필 probe(+임베딩 normalization)에 기록")
    ap.add_argument("--skill-dir", default=str(SKILL_DIR))
    a = ap.parse_args(argv)
    pg.init_cli_logging()
    override = json.loads(a.override) if a.override else None
    return preflight(a.profile, [c.strip() for c in a.checks.split(",")], write_back_flag=a.write_back,
                     override=override, skill_dir=Path(a.skill_dir))


if __name__ == "__main__":
    sys.exit(main())
