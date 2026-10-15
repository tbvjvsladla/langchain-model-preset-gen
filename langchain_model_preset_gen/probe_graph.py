"""probe_graph — 프로필의 끄기·켜기와 sampling 수용표를 원문 HTTP 로 실측하는 결정론 상태 그래프.

노드: connect → off_gate → off_probe → on_path(→ suppressed 면 on_path_deep) → sampling (LangGraph StateGraph).
- 각 노드는 요청 1회(또는 판정만)를 하고 관측을 결과 이름(outcome)으로 분류한다(순수 분류 함수).
- 라우터는 data/probe_graph.yaml 전이표만 읽는다. 노드 전이표 → any 전이표 순으로 찾고 없으면 UndefinedTransition.
- 전이 대상: 노드 이름이면 그 노드, `R_*` 이면 기록 후 records.<R_*>.then 노드, 그 밖(T_*·U_*·PASS)은 종착.
- '사고함' = evidence.text 중 하나라도 길이 > 0 OR evidence.tokens 중 하나라도 > 0.
- 요청 본문의 effort·sampling 인자는 해석기 spec 에서 온다(resolver.slot_extras·sampling_status·effective) —
  Agent 가 받는 ChatModel 과 같은 결과에서 나온 인자를 원문 본문으로 옮긴다
  (chat_completions: extra_body 평탄화 / messages·generate_content: body_paths 표).
- 원문 HTTP 를 쓰는 이유: LangChain 은 vLLM 의 reasoning 필드를 버려 사고를 관측할 수 없다.

HTTP 예외는 잡되 삼키지 않고 명시 outcome(timeout·http_429_5xx_or_transport)으로 사상한다.
timeout = 응답 대기 초과(ReadTimeout)뿐이고 연결·풀·쓰기 timeout 은 전송 실패(인프라)다. write-back 은 preflight 몫이다.

CLI(scripts/probe.py): --profile <yaml> [--override JSON] [--mermaid]
  종료코드 0 = PASS, 10 = probe_error(결과 — stdout JSON), 그 밖(1 = 예외, 2 = 인자 오류) = 도구 실패.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import operator
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import httpx
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from . import catalog
from . import injection as inj
from . import resolver as rs
from . import settings as st

logger = logging.getLogger("langchain_model_preset_gen.probe_graph")

SKILL_DIR = catalog.SKILL_DIR

NODE_NAMES = ("connect", "off_gate", "off_probe", "on_path", "on_path_deep", "sampling")
ENTRY_NODE = "connect"
ANY_NODE = "any"
RECORD_PREFIX = "R_"
PASS_TERMINAL = "PASS"
NONE_LEVEL = "none"
PROBE_OK = "ok"
PROBE_ERROR = "probe_error"

OUT_INFRA = "http_429_5xx_or_transport"
OUT_4XX = "http_4xx"
OUT_TIMEOUT = "timeout"

HTTP_OK = 200
HTTP_CLIENT_ERROR_MIN = 400
HTTP_TOO_MANY_REQUESTS = 429
HTTP_SERVER_ERROR_MIN = 500

CONTENT_HEAD_CHARS = 200
ERROR_HEAD_CHARS = 300
ELAPSED_ROUND_DIGITS = 2
JSON_INDENT = 2                     # CLI stdout JSON 들여쓰기

# probe_error 는 실측 결과(stdout JSON)다. 파이썬 예외(1)·argparse 오류(2)와 겹치면 호출자(기동 게이트)가
# 결과와 도구 고장을 가르지 못한다 → 둘과 겹치지 않는 값.
PROBE_ERROR_EXIT_CODE = 10

API_CHAT_COMPLETIONS = "chat_completions"
API_MESSAGES = "messages"
API_GENERATE_CONTENT = "generate_content"
AUTH_BEARER = "bearer"
AUTH_HEADER_NAMES = {"x_api_key": "x-api-key", "x_goog_api_key": "x-goog-api-key"}

_SEGMENT_RE = re.compile(r"^(?P<key>[^\[\]]+)(\[(?P<fk>[^=!\]]+)(?P<op>!?=)(?P<fv>[^\]]*)\])?$")


class UndefinedTransition(Exception):
    """전이표에 없는 (노드, outcome) — 그래프를 즉시 멈춘다."""

    def __init__(self, node: str, outcome: str):
        super().__init__(f"전이표에 없는 전이: node={node!r} outcome={outcome!r}")
        self.node = node
        self.outcome = outcome


def check_table_nodes(probe_table: dict) -> None:
    declared = set(probe_table["nodes"]) - {ANY_NODE}
    if declared != set(NODE_NAMES):
        raise ValueError(f"전이표 노드 {sorted(declared)} ≠ 코드 노드 {sorted(NODE_NAMES)}")


# ════════════════ 증거 추출 · 분류 (순수 함수) ════════════════

def _match(item: Any, fk: str, op: str, fv: str) -> bool:
    actual = item.get(fk) if isinstance(item, dict) else None
    equal = actual is not None and str(actual).lower() == fv.lower()
    return equal if op == "=" else not equal


def _values(obj: Any, dotted: str) -> list:
    """경로 조회 → 값 목록(0개 이상). 경로 중간이 없으면 빈 목록(= 증거 없음)."""
    nodes = [obj]
    for segment in dotted.split(inj.PATH_SEP):
        m = _SEGMENT_RE.match(segment)
        if m is None:
            raise ValueError(f"경로 문법 위반: {segment!r} (in {dotted!r})")
        key = m["key"]
        nxt: list = []
        for node in nodes:
            if isinstance(node, list) and key.isdigit():
                idx = int(key)
                child = node[idx] if idx < len(node) else None
            elif isinstance(node, dict):
                child = node.get(key)
            else:
                child = None
            if child is None:
                continue
            if m["fk"] is None:
                nxt.append(child)
            elif isinstance(child, list):
                nxt.extend(item for item in child if _match(item, m["fk"], m["op"], m["fv"]))
        nodes = nxt
    return nodes


def _dig(obj: Any, dotted: str) -> Any:
    found = _values(obj, dotted)
    return found[0] if found else None


def thinking_evidence(body: dict, evidence_spec: dict) -> dict:
    """{text: {field: chars}, tokens: {field: n}, thinking: bool} — 텍스트 OR 토큰."""
    text = {field: sum(len(v) for v in _values(body, field) if isinstance(v, str)) for field in evidence_spec["text"]}
    tokens = {field: sum(v for v in _values(body, field) if isinstance(v, int) and not isinstance(v, bool))
              for field in evidence_spec["tokens"]}
    thinking = any(n > 0 for n in text.values()) or any(n > 0 for n in tokens.values())
    return {"text": text, "tokens": tokens, "thinking": thinking}


def response_format(probe_table: dict, provider_class: str) -> dict:
    return {"evidence": probe_table["evidence"][provider_class], **probe_table["answer"][provider_class]}


def _content(body: dict, fmt: dict) -> str:
    return "".join(v for v in _values(body, fmt["content"]) if isinstance(v, str))


def _hit_length(body: dict, fmt: dict) -> bool:
    return _dig(body, fmt["finish_reason"]) in fmt["finish_length"]


def _http_failure(obs: dict) -> str | None:
    """200 이면 None. 429·5xx·전송 → OUT_INFRA, 그 밖 4xx → OUT_4XX, 기타 → http_<code>(전이표 밖)."""
    if obs["kind"] != "http":
        return OUT_INFRA
    status = obs["status"]
    if status == HTTP_OK:
        return None
    if status == HTTP_TOO_MANY_REQUESTS or status >= HTTP_SERVER_ERROR_MIN:
        return OUT_INFRA
    if status >= HTTP_CLIENT_ERROR_MIN:
        return OUT_4XX
    return f"http_{status}"


def listed_model_ids(body: dict, http_spec: dict) -> set[str]:
    prefix = http_spec["models_id_prefix"]
    ids = (item[http_spec["models_id"]] for item in body[http_spec["models_items"]])
    return {i[len(prefix):] if prefix and i.startswith(prefix) else i for i in ids}


def classify_connect(obs: dict, model_name: str, http_spec: dict) -> str:
    failure = _http_failure(obs)
    if failure:
        return failure
    return "model_listed" if model_name in listed_model_ids(obs["body"], http_spec) else "model_missing"


def classify_off_gate(none_slot: str | dict) -> str:
    action = inj.normalize_slot(none_slot)["action"]
    return "none_is_disable" if action == inj.SLOT_DISABLE else "none_is_omit_or_level"


def classify_off_probe(obs: dict, end_markers: list[str], fmt: dict) -> str:
    if obs["kind"] == "timeout":
        return OUT_TIMEOUT
    failure = _http_failure(obs)
    if failure:
        return failure
    body = obs["body"]
    if thinking_evidence(body, fmt["evidence"])["thinking"]:
        return "thinking"
    content = _content(body, fmt)
    if any(m in content for m in end_markers):
        return "leak"
    return "clean" if content else "empty_content"


def classify_on_path(obs: dict, slot_action: str, end_markers: list[str], fmt: dict) -> str:
    if obs["kind"] == "timeout":
        return OUT_TIMEOUT
    failure = _http_failure(obs)
    if failure:
        return failure
    body = obs["body"]
    content = _content(body, fmt)
    if any(m in content for m in end_markers):
        return "leak"
    if _hit_length(body, fmt) and not content:
        return "budget_exhausted"
    if slot_action == inj.SLOT_SET and not thinking_evidence(body, fmt["evidence"])["thinking"]:
        return "suppressed"
    return "healthy" if content else "empty_content"


def classify_sampling(observations: list[dict]) -> str:
    for obs in observations:
        if obs["kind"] == "timeout":
            return OUT_TIMEOUT
        failure = _http_failure(obs)
        if failure == OUT_4XX:
            return "declared_accepted_but_4xx"
        if failure:
            return failure
    return "all_accepted_as_declared"


def resolve_transition(probe_table: dict, node: str, outcome: str) -> str:
    node_outs = probe_table["nodes"][node]["outcomes"]
    if outcome in node_outs:
        return node_outs[outcome]
    any_outs = probe_table["nodes"][ANY_NODE]["outcomes"]
    if outcome in any_outs:
        return any_outs[outcome]
    raise UndefinedTransition(node, outcome)


def _next_hop(probe_table: dict, target: str) -> tuple[list[str], str | None, str | None]:
    """전이 대상 → (records, 다음 노드 | None, 종착 | None)."""
    if target in NODE_NAMES:
        return [], target, None
    if target.startswith(RECORD_PREFIX):
        records, node, terminal = _next_hop(probe_table, probe_table["records"][target]["then"])
        return [target, *records], node, terminal
    return [], None, target


# ════════════════ 요청 (provider_class 별 원문 HTTP) ════════════════

def _mapped(sdk_kwargs: dict, body_paths: dict) -> dict:
    """SDK 인자 → 본문 (body_paths 표). 표에 없는 인자는 KeyError — 주입 방식과 class 가 맞지 않다."""
    return inj.expand_paths({body_paths[k]: v for k, v in sdk_kwargs.items()})


def auth_headers(http_spec: dict, key: str | None) -> dict:
    headers = dict(http_spec["headers"])
    if key is None:
        return headers
    auth = http_spec["auth"]
    if auth == AUTH_BEARER:
        headers["Authorization"] = f"Bearer {key}"
    else:
        headers[AUTH_HEADER_NAMES[auth]] = key
    return headers


def native_body(http_spec: dict, model: str, prompt: str, max_tokens: int, sdk_kwargs: dict) -> dict:
    """생성 요청 본문 1건. sdk_kwargs = 주입표 칸 펼침(SDK 인자 형식) + sampling 키(프로필 키 이름)."""
    api = http_spec["api"]
    cap = inj.expand_paths({http_spec["max_tokens_path"]: max_tokens})
    if api == API_CHAT_COMPLETIONS:
        base = {"model": model, "messages": [{"role": "user", "content": prompt}], **cap}
        return inj.deep_merge(base, inj.to_http_body(sdk_kwargs))
    if api == API_MESSAGES:
        base = {"model": model, "messages": [{"role": "user", "content": prompt}], **cap}
        return inj.deep_merge(base, _mapped(sdk_kwargs, http_spec["body_paths"]))
    if api == API_GENERATE_CONTENT:
        base = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], **cap}
        return inj.deep_merge(base, _mapped(sdk_kwargs, http_spec["body_paths"]))
    raise ValueError(f"http.api 어휘 밖: {api!r} — {API_CHAT_COMPLETIONS}|{API_MESSAGES}|{API_GENERATE_CONTENT}")


@dataclass
class ProbeContext:
    spec: dict
    api_key: str | None
    client: httpx.Client
    probe_table: dict
    provider_classes: dict

    @property
    def effective(self) -> dict:
        return self.spec["effective"]

    @property
    def http_spec(self) -> dict:
        return self.provider_classes[self.spec["provider_class"]]["http"]

    @property
    def fmt(self) -> dict:
        return response_format(self.probe_table, self.spec["provider_class"])

    @property
    def headers(self) -> dict:
        return auth_headers(self.http_spec, self.api_key)

    @property
    def end_markers(self) -> list[str]:
        marker = self.spec["end_marker"]
        return self.probe_table["end_markers_default"] if marker is None else [marker]

    def send(self, method: str, path: str, body: dict | None = None, params: dict | None = None) -> dict:
        """요청 1회 → 관측 {kind: http|timeout|transport, status, body, error, elapsed_seconds}."""
        url = f"{self.spec['http_base_url']}{path}"
        t0 = time.monotonic()
        try:   # 전송 실패를 삼키지 않고 kind 로 표면화 → 분류기가 명시 outcome 으로 사상
            r = self.client.request(method, url, json=body, params=params, headers=self.headers,
                                    timeout=self.effective["timeout_seconds"])
        except httpx.ReadTimeout as e:
            return {"kind": "timeout", "status": None, "body": None, "error": type(e).__name__,
                    "elapsed_seconds": round(time.monotonic() - t0, ELAPSED_ROUND_DIGITS)}
        except httpx.TransportError as e:
            return {"kind": "transport", "status": None, "body": None, "error": f"{type(e).__name__}: {e}",
                    "elapsed_seconds": round(time.monotonic() - t0, ELAPSED_ROUND_DIGITS)}
        ok = r.status_code == HTTP_OK
        return {"kind": "http", "status": r.status_code, "body": r.json() if ok else None,
                "error": None if ok else r.text[:ERROR_HEAD_CHARS],
                "elapsed_seconds": round(time.monotonic() - t0, ELAPSED_ROUND_DIGITS)}

    def list_models(self) -> dict:
        spec = self.http_spec
        return self.send("GET", spec["models_path"], params=spec["models_query"] or None)

    def chat(self, prompt_key: str, sdk_kwargs: dict) -> dict:
        spec = self.http_spec
        # 상한은 비용 상한이다. 실효 max_tokens(override 포함)가 더 작으면 그 값 — 아니면 max_tokens override 가 inert.
        max_tokens = min(self.effective["max_tokens"], self.probe_table["probe_request_max_tokens"])
        model = self.spec["model_name"]
        body = native_body(spec, model, self.probe_table["prompts"][prompt_key], max_tokens, sdk_kwargs)
        obs = self.send("POST", spec["chat_path"].format(model=model), body)
        obs["request_extras"] = {k: v for k, v in body.items() if k not in ("model", "messages", "contents")}
        return obs

    def summarize(self, node: str, obs: dict) -> dict:
        rec = {"node": node, "kind": obs["kind"], "status": obs["status"], "error": obs["error"],
               "elapsed_seconds": obs["elapsed_seconds"], "request_extras": obs.get("request_extras")}
        body = obs["body"]
        if body is not None and node != "connect":
            fmt = self.fmt
            ev = thinking_evidence(body, fmt["evidence"])
            content = _content(body, fmt)
            rec.update({"thinking": ev["thinking"], "evidence": {**ev["text"], **ev["tokens"]},
                        "content_head": content[:CONTENT_HEAD_CHARS], "finish_reason": _dig(body, fmt["finish_reason"])})
        return rec


# ════════════════ 그래프 ════════════════

class ProbeState(TypedDict):
    steps: Annotated[list[dict], operator.add]
    observations: Annotated[list[dict], operator.add]


def _step(node: str, outcome: str, observations: list[dict]) -> dict:
    logger.info("[probe:%s] outcome=%s", node, outcome)
    return {"steps": [{"node": node, "outcome": outcome}], "observations": observations}


def _expected_action(spec: dict) -> str:
    """effort=none 은 '끔' 의도 — none 칸이 최저 단계(set)여도 사고를 기대하지 않는다."""
    effort = spec["effective"]["effort"]
    return spec["mapping"][effort]["action"] if effort != NONE_LEVEL else NONE_LEVEL


def build_graph(probe_table: dict, ctx: ProbeContext | None = None):
    """전이표 기반 StateGraph 컴파일. ctx=None 은 그림(--mermaid) 전용."""
    check_table_nodes(probe_table)

    def connect(state: ProbeState) -> dict:
        obs = ctx.list_models()
        return _step("connect", classify_connect(obs, ctx.spec["model_name"], ctx.http_spec),
                     [ctx.summarize("connect", obs)])

    def off_gate(state: ProbeState) -> dict:
        return _step("off_gate", classify_off_gate(ctx.spec["mapping"][NONE_LEVEL]), [])

    def off_probe(state: ProbeState) -> dict:
        obs = ctx.chat("off_check", rs.slot_extras(ctx.spec, NONE_LEVEL))
        return _step("off_probe", classify_off_probe(obs, ctx.end_markers, ctx.fmt), [ctx.summarize("off_probe", obs)])

    def on_path(state: ProbeState) -> dict:
        obs = ctx.chat("on_check", rs.slot_extras(ctx.spec, ctx.effective["effort"]))
        outcome = classify_on_path(obs, _expected_action(ctx.spec), ctx.end_markers, ctx.fmt)
        return _step("on_path", outcome, [ctx.summarize("on_path", obs)])

    def on_path_deep(state: ProbeState) -> dict:
        # on_path 가 suppressed 일 때만 온다. 같은 판정·같은 실효 칸, 질문만 심화 1회.
        obs = ctx.chat("on_check_deep", rs.slot_extras(ctx.spec, ctx.effective["effort"]))
        outcome = classify_on_path(obs, _expected_action(ctx.spec), ctx.end_markers, ctx.fmt)
        return _step("on_path_deep", outcome, [ctx.summarize("on_path_deep", obs)])

    def sampling(state: ProbeState) -> dict:
        slot = rs.slot_extras(ctx.spec, ctx.effective["effort"])
        sent: list[dict] = []
        records: list[dict] = []
        for key, status in ctx.spec["sampling_status"].items():
            value = ctx.effective[key]
            if status != rs.ACCEPTED or value is None:
                continue                                   # accepted·non-null 만 전송
            obs = ctx.chat("sampling_check", {**slot, key: value})
            sent.append(obs)
            records.append({**ctx.summarize("sampling", obs), "sampling_key": key})
            if _http_failure(obs):
                break                                      # 첫 비200 에서 판정 확정
        return _step("sampling", classify_sampling(sent), records)

    def route(state: ProbeState) -> str:
        last = state["steps"][-1]
        _, node, _ = _next_hop(probe_table, resolve_transition(probe_table, last["node"], last["outcome"]))
        return END if node is None else node

    funcs = {"connect": connect, "off_gate": off_gate, "off_probe": off_probe, "on_path": on_path,
             "on_path_deep": on_path_deep, "sampling": sampling}
    builder = StateGraph(ProbeState)
    for name in NODE_NAMES:
        builder.add_node(name, funcs[name])
    builder.add_edge(START, ENTRY_NODE)
    for name in NODE_NAMES:
        targets = {_next_hop(probe_table, t)[1] for t in probe_table["nodes"][name]["outcomes"].values()}
        targets |= {_next_hop(probe_table, t)[1] for t in probe_table["nodes"][ANY_NODE]["outcomes"].values()}
        path_map = {t: t for t in targets if t is not None}
        if None in targets:
            path_map[END] = END
        builder.add_conditional_edges(name, route, path_map)
    return builder.compile()


def run_probe(spec: dict, api_key: str | None, *, probe_table: dict, provider_classes: dict,
              client: httpx.Client | None = None) -> dict:
    """LLM spec(resolver.resolve 결과) 실측 1회 → {terminal, status, diagnosis, path, steps, records, observations, effective}."""
    if spec["kind"] != rs.KIND_LLM:
        raise ValueError(f"실측 그래프는 LLM 전용이다 — {spec['name']} kind={spec['kind']}")
    with httpx.Client() if client is None else contextlib.nullcontext(client) as http:
        ctx = ProbeContext(spec, api_key, http, probe_table, provider_classes)
        final = build_graph(probe_table, ctx).invoke({"steps": [], "observations": []})
    records: list[str] = []
    terminal = None
    for step in final["steps"]:
        recs, _, terminal = _next_hop(probe_table, resolve_transition(probe_table, step["node"], step["outcome"]))
        records.extend(recs)
    return {
        "terminal": terminal,
        "status": PROBE_OK if terminal == PASS_TERMINAL else PROBE_ERROR,
        "diagnosis": probe_table["diagnoses"][terminal],
        "path": [s["node"] for s in final["steps"]],
        "steps": final["steps"],
        "records": records,
        "observations": final["observations"],
        "effective": spec["effective"],
    }


def probe_profile(profile, override: dict | None = None, *, skill_dir: Path = SKILL_DIR,
                  client: httpx.Client | None = None) -> dict:
    """프로필 경로 + override → 해석 → 키 해소 → 실측."""
    data = catalog.load_data(skill_dir)
    spec = rs.resolve(profile, override, data=data)
    key = st.resolve_api_key(spec["api_key_env"], st.load_settings(skill_dir)["env_file_path"])
    return run_probe(spec, key, probe_table=data["probe_graph"], provider_classes=data["provider_classes"], client=client)


def init_cli_logging() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="langchain-model-preset-gen 실측 상태 그래프 (원문 HTTP, write-back 없음)")
    ap.add_argument("--profile", help="프로필 yaml 경로")
    ap.add_argument("--override", help="실효값 override JSON (예: '{\"effort\": \"high\"}')")
    ap.add_argument("--skill-dir", default=str(SKILL_DIR))
    ap.add_argument("--mermaid", action="store_true", help="그래프 mermaid 출력")
    args = ap.parse_args(argv)
    init_cli_logging()
    skill_dir = Path(args.skill_dir)
    if args.mermaid:
        print(build_graph(catalog.load_data(skill_dir)["probe_graph"]).get_graph().draw_mermaid())
    if not args.profile:
        return 0
    override = json.loads(args.override) if args.override else None
    result = probe_profile(args.profile, override, skill_dir=skill_dir)
    print(json.dumps(result, ensure_ascii=False, indent=JSON_INDENT))
    return 0 if result["status"] == PROBE_OK else PROBE_ERROR_EXIT_CODE
