"""probe_runnable — 해석기가 만든 객체(또는 workflow_probe 가 감싼 Runnable)로 LangChain 호출 방식을 실제로 돌린다.

LLM   : invoke · ainvoke · stream · astream · batch · abatch      → status ok ⇔ 6종 전부 ok (unsupported 도 ok 가 아니다)
임베딩 : embed_query · embed_documents · aembed_query · aembed_documents → probe_error 가 하나라도 있으면 probe_error
         + embed_query 가 ok 면 정규화 전 raw 벡터(`inner`)의 노름·차원을 잰다(normalization 파생 원자료).

결과 어휘: ok(실제로 돌았다) / unsupported(provider 가 명시적으로 거부 — 모델 사실) / probe_error(하네스가 확증 못 함).
실측 하네스다 — 호출 예외는 잡아서 결과 어휘로 사상한다(삼키지 않고 기록한다). 일시적 실패는 지수 백오프로 재시도한다.
"""

from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger("langchain_model_preset_gen.probe_runnable")

RUNNABLE_OK = "ok"
RUNNABLE_UNSUPPORTED = "unsupported"
PROBE_OK = "ok"
PROBE_ERROR = "probe_error"

SMOKE_RETRIES = 2                   # 일시적 실패(429·네트워크) 재시도 횟수
SMOKE_BACKOFF_START_SECONDS = 1.0
SMOKE_BACKOFF_FACTOR = 2
MSG_HEAD_CHARS = 160

# 일시적 실패(하네스·네트워크·쿼터) — 모델 사실이 아니다 → probe_error.
_TRANSIENT_MARKERS = (
    "429", "rate limit", "ratelimit", "quota", "overloaded", "timeout", "timed out",
    "connection", "temporarily", "500", "502", "503", "504", "unavailable",
)
# 출력 상한 소진 — 사고 모델이 상한을 사고에 다 쓰면 400 invalid_request_error 가 온다 → 예산 문제(probe_error).
_BUDGET_MARKERS = ("max_tokens or model output limit was reached",)
# 모델·엔드포인트 부재(404) — 호출 방식 미지원이 아니다 → probe_error. unsupported 표지보다 먼저 본다.
_MISSING_MARKERS = ("model_not_found", "does not exist", "notfounderror", "not_found", "404")
# provider 가 그 호출·인자를 명시적으로 거부 — 이것만 모델 사실(unsupported).
_UNSUPPORTED_MARKERS = (
    "not support", "not supported", "unsupported", "notimplemented", "not implemented",
    "does not support", "unrecognized", "unknown parameter", "invalid_request",
)

# raw 노름 → normalization 파생 허용오차. builtin 실측치 1.0000, required 실측치 4.9~17.4 를 넉넉히 가른다.
NORM_BUILTIN_TOLERANCE = 1e-2
NORM_ROUND_DIGITS = 4
NORMALIZATION_BUILTIN = "builtin"
NORMALIZATION_REQUIRED = "required"


def classify_exception(exc: BaseException) -> str:
    """예외 → unsupported | probe_error. 모르는 실패는 probe_error(모델 사실로 단정하지 않는다)."""
    if isinstance(exc, NotImplementedError):
        return RUNNABLE_UNSUPPORTED
    msg = f"{type(exc).__name__}: {exc}".lower()
    for markers in (_TRANSIENT_MARKERS, _BUDGET_MARKERS, _MISSING_MARKERS):
        if any(m in msg for m in markers):
            return PROBE_ERROR
    if any(m in msg for m in _UNSUPPORTED_MARKERS):
        return RUNNABLE_UNSUPPORTED
    return PROBE_ERROR


def run_method(target, method: str, prompt: str) -> None:
    """호출 1종을 실제로 실행한다. 스트림이 0 청크면 RuntimeError(확증 불가)."""
    if method == "invoke":
        target.invoke(prompt)
    elif method == "ainvoke":
        asyncio.run(target.ainvoke(prompt))
    elif method == "stream":
        if not list(target.stream(prompt)):
            raise RuntimeError("stream 이 0 청크 — 확증 불가")
    elif method == "astream":
        async def _collect():
            return [c async for c in target.astream(prompt)]
        if not asyncio.run(_collect()):
            raise RuntimeError("astream 이 0 청크 — 확증 불가")
    elif method == "batch":
        target.batch([prompt])
    elif method == "abatch":
        asyncio.run(target.abatch([prompt]))
    elif method == "embed_query":
        target.embed_query(prompt)
    elif method == "embed_documents":
        target.embed_documents([prompt])
    elif method == "aembed_query":
        asyncio.run(target.aembed_query(prompt))
    elif method == "aembed_documents":
        asyncio.run(target.aembed_documents([prompt]))
    else:
        raise ValueError(f"모르는 호출 방식: {method!r}")


def smoke_one(target, *, method: str, prompt: str, retries: int = SMOKE_RETRIES) -> str:
    """호출 1종 실측 → ok | unsupported | probe_error. 일시적 실패는 백오프 후 재시도, 끝까지 실패면 probe_error."""
    delay = SMOKE_BACKOFF_START_SECONDS
    for attempt in range(retries + 1):
        try:
            run_method(target, method, prompt)
            return RUNNABLE_OK
        except Exception as exc:  # noqa: BLE001 — 실측 하네스: 예외를 결과 어휘로 사상해 기록한다
            verdict = classify_exception(exc)
            logger.info("[runnable] %s 시도 %d 실패 → %s: %s", method, attempt + 1, verdict, str(exc)[:MSG_HEAD_CHARS])
            if verdict == RUNNABLE_UNSUPPORTED or attempt == retries:
                return verdict
            time.sleep(delay)
            delay *= SMOKE_BACKOFF_FACTOR
    raise AssertionError("unreachable")


def check_runnable(target, methods: list[str], prompt: str, *, embedding: bool) -> dict:
    """호출 방식 전부 실측 → {runnable, status}."""
    runnable = {m: smoke_one(target, method=m, prompt=prompt) for m in methods}
    if embedding:
        status = PROBE_ERROR if PROBE_ERROR in runnable.values() else PROBE_OK
    else:
        status = PROBE_OK if all(v == RUNNABLE_OK for v in runnable.values()) else PROBE_ERROR
    return {"runnable": runnable, "status": status}


def measure_embedding(raw_embeddings, prompt: str) -> dict:
    """정규화 전 임베딩 객체로 노름·차원 실측."""
    vec = raw_embeddings.embed_query(prompt)
    return {"norm_measured": round(sum(x * x for x in vec) ** 0.5, NORM_ROUND_DIGITS),
            "measured_at_dimensions": len(vec)}


def derive_normalization(norm_measured: float) -> str:
    return NORMALIZATION_BUILTIN if abs(norm_measured - 1.0) <= NORM_BUILTIN_TOLERANCE else NORMALIZATION_REQUIRED
