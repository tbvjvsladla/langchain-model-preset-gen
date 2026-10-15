"""injection — 6칸 effort 매핑의 한 칸(slot)을 SDK 인자 extras 로 펼친다. 순수 함수만 둔다(파일 I/O 없음).

- slot 형식: `{action: set, value}` / `{action: disable}` / `{action: omit}`. 스칼라 축약형 `low` = `{action: set, value: low}`.
- 경로 표기 `a.b.c` 는 중첩 dict 로 펼친다. `extra_body.*` 는 OpenAI 스키마 밖 필드다.
  SDK 경로는 그대로 `extra_body=` 로 넘기고, 원문 HTTP 경로는 `to_http_body` 로 본문 최상위에 평탄화한다
  (OpenAI SDK 가 extra_body 를 본문 최상위에 합치는 것과 같은 의미).
- 해석기와 실측 그래프가 모두 `expand_slot` 하나를 쓴다 — 실측한 본문과 Agent 가 보내는 인자가 같은 결과에서 나온다.
- 미등록 injection·미정의 값/disable → 자연 KeyError, 어휘 밖 action → ValueError.
"""

from __future__ import annotations

from typing import Any

EXTRA_BODY_KEY = "extra_body"
PATH_SEP = "."
SLOT_SET = "set"
SLOT_DISABLE = "disable"
SLOT_OMIT = "omit"
SLOT_ACTIONS = (SLOT_SET, SLOT_DISABLE, SLOT_OMIT)


def normalize_slot(slot: str | dict) -> dict:
    """스칼라 축약형을 정규형 dict 로 바꾼다. dict 는 사본을 돌려준다."""
    if isinstance(slot, str):
        return {"action": SLOT_SET, "value": slot}
    return dict(slot)


def expand_paths(flat: dict[str, Any]) -> dict:
    """{'a.b.c': v} → {'a': {'b': {'c': v}}}. 같은 접두 경로는 하나의 dict 로 합친다. 같은 경로 두 번은 ValueError."""
    out: dict = {}
    for dotted, value in flat.items():
        *parents, leaf = dotted.split(PATH_SEP)
        node = out
        for key in parents:
            node = node.setdefault(key, {})
        if leaf in node:
            raise ValueError(f"인자 경로 충돌: {dotted!r} 가 두 번 지정됐다")
        node[leaf] = value
    return out


def deep_merge(base: dict, extra: dict) -> dict:
    """중첩 dict 병합 사본. 같은 leaf 경로에 두 값이 오면 ValueError — 조용한 덮어쓰기 금지."""
    out = dict(base)
    for key, value in extra.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        elif key in out:
            raise ValueError(f"인자 병합 충돌: {key!r} 에 {out[key]!r} 와 {value!r} 가 함께 온다")
        else:
            out[key] = value
    return out


def expand_slot(injection_table: dict, injection_id: str, slot: str | dict) -> dict:
    """한 칸 → SDK 인자 extras(중첩 dict). omit 은 빈 dict."""
    spec = injection_table[injection_id]
    norm = normalize_slot(slot)
    action = norm["action"]
    if action == SLOT_OMIT:
        return {}
    if action == SLOT_DISABLE:
        return expand_paths(spec["disable"])
    if action == SLOT_SET:
        value = norm["value"]
        values = spec.get("values", {})          # values 는 선택 키 — 없으면 value_path 로만 표현
        if value in values:
            return expand_paths(values[value])
        return expand_paths({spec["value_path"]: value})
    raise ValueError(f"slot action {action!r} 은 어휘 밖이다 — 허용: {SLOT_ACTIONS}")


def to_http_body(extras: dict) -> dict:
    """원문 HTTP 본문용: `extra_body` 하위를 최상위로 올린다. 키 충돌은 ValueError."""
    top = {k: v for k, v in extras.items() if k != EXTRA_BODY_KEY}
    lifted = extras.get(EXTRA_BODY_KEY, {})
    clash = set(top) & set(lifted)
    if clash:
        raise ValueError(f"extra_body 평탄화 키 충돌: {sorted(clash)}")
    return {**top, **lifted}
