"""derive_mapping — OpenRouter /models 메타데이터 → 모델 규칙 초안 1건. Agent 보조 도구.

결정론 규칙만 적용해 **초안**을 출력한다. Agent 가 모델카드와 대조해 확인한 뒤
model_rules/ 에 파일로 등록한다(이 도구는 파일을 쓰지 않는다).

6칸 규칙 (reasoning = 모델 항목의 `reasoning` 객체):
  · reasoning 메타 없음/빈 객체                         → 6칸 전부 omit
  · supported_efforts 없음 + mandatory                  → 6칸 전부 omit (끌 수도 조절할 수도 없음)
  · supported_efforts 없음 + not mandatory (토글형)     → none: disable, 나머지: {set: enable}
  · supported_efforts 있음
      none 칸 : 'none' ∈ supported_efforts 이거나 not mandatory → disable / 아니면 지원 최저 단계
      그 외 칸: 같은 단계 지원 → 그 단계 / 아니면 지원 단계 중 바로 위(단 max 로는 올리지 않음)
                / 위가 없으면 바로 아래(= 최고 지원 단계로 상한) / 그래도 없으면 위(max 포함)
  · sampling: sampling_keys 각각 supported_parameters 에 있으면 accepted, 없으면 rejected

CLI(scripts/derive_mapping.py): --model-id deepseek/deepseek-v4.1-flash --models-json <캐시한 /models JSON>
                                 --model-id openai/gpt-oss-20b --fetch
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from . import catalog
from .publish import dump_yaml

SKILL_DIR = catalog.SKILL_DIR

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
FETCH_TIMEOUT_SECONDS = 30            # /models 1회 조회 상한 — 응답 크기(~460종)에 비해 넉넉한 값

# 초안 규칙의 고정 필드 — OpenRouter 라우트는 전부 OpenAI 호환 + 통합 reasoning 객체 주입.
PROVIDER_CLASS = "openai_compatible"
INJECTION_ID = "or_reasoning"
TOGGLE_ENABLE_VALUE = "enable"        # data/injection.yaml or_reasoning.values 의 토글 키

ACTION_SET, ACTION_DISABLE, ACTION_OMIT = "set", "disable", "omit"
STATUS_ACCEPTED, STATUS_REJECTED = "accepted", "rejected"
NONE_LEVEL = "none"

# OpenRouter supported_efforts 에는 표준 6단계 밖의 'minimal' 이 등장한다. 순서상 none 과 low 사이다
# (OpenRouter 문서의 effort 서열). 6칸 슬롯은 아니지만 매핑 **대상**(값)이 될 수 있다.
OPENROUTER_EXTRA_LEVEL = "minimal"
OPENROUTER_EXTRA_LEVEL_AFTER = NONE_LEVEL

# 하위 칸의 올림으로는 도달시키지 않는 단계. max 는 비용 상한이 없는 극단 단계라, 표준 xhigh 를
# 모델의 max 로 올리면 조용히 비용이 커진다. 3단계 모델이면 medium·high·xhigh → high 로 두는 배치와
# DeepSeek 예(`xhigh: high, max: max`)가 이 규칙의 근거다.
ROUND_UP_EXCLUDED = ("max",)


def load_vocabulary(skill_dir: Path = SKILL_DIR) -> dict:
    return catalog.load_data(skill_dir)["vocabulary"]


def effort_order(vocab: dict) -> list[str]:
    """vocabulary effort_levels 에 OpenRouter 'minimal' 을 none 바로 뒤에 끼운 서열."""
    order = list(vocab["effort_levels"])
    order.insert(order.index(OPENROUTER_EXTRA_LEVEL_AFTER) + 1, OPENROUTER_EXTRA_LEVEL)
    return order


def _check_vocab(vocab: dict) -> None:
    """초안이 쓰는 고정 어휘가 vocabulary 정본 안에 있는지 — 어휘 drift 시 즉시 raise."""
    for action in (ACTION_SET, ACTION_DISABLE, ACTION_OMIT):
        if action not in vocab["slot_actions"]:
            raise ValueError(f"slot action {action!r} 가 vocabulary slot_actions 에 없다: {vocab['slot_actions']}")
    for status in (STATUS_ACCEPTED, STATUS_REJECTED):
        if status not in vocab["sampling_status"]:
            raise ValueError(f"sampling status {status!r} 가 vocabulary sampling_status 에 없다.")


def _pick_level(slot: str, targets: list[str], order: list[str]) -> str:
    if slot in targets:
        return slot
    rank = order.index(slot)
    up = [t for t in targets if order.index(t) > rank]
    up_allowed = [t for t in up if t not in ROUND_UP_EXCLUDED]
    down = [t for t in targets if order.index(t) < rank]
    if up_allowed:
        return up_allowed[0]
    if down:
        return down[-1]
    return up[0]


def derive_mapping(reasoning: dict, vocab: dict) -> dict:
    """reasoning 메타 → 정규화된 6칸 매핑."""
    levels = vocab["effort_levels"]
    omit_all = {lv: {"action": ACTION_OMIT} for lv in levels}
    if not reasoning:
        return omit_all
    mandatory = reasoning["mandatory"]
    efforts = reasoning.get("supported_efforts", [])   # 선택 필드 — 토글형·고정형 모델에는 없다
    if not efforts:
        if mandatory:
            return omit_all
        return {lv: ({"action": ACTION_DISABLE} if lv == NONE_LEVEL
                     else {"action": ACTION_SET, "value": TOGGLE_ENABLE_VALUE}) for lv in levels}
    order = effort_order(vocab)
    unknown = [e for e in efforts if e not in order]
    if unknown:
        raise ValueError(f"supported_efforts 에 서열 밖 단계 {unknown} (known: {order}) — 서열 정의 갱신 필요.")
    targets = sorted((e for e in efforts if e != NONE_LEVEL), key=order.index)
    if not targets:
        raise ValueError(f"supported_efforts={efforts} 에 none 외 단계가 없다 — 결정론 초안 불가, Agent 판단.")
    mapping: dict = {}
    for lv in levels:
        if lv == NONE_LEVEL:
            mapping[lv] = ({"action": ACTION_DISABLE} if NONE_LEVEL in efforts or not mandatory
                           else {"action": ACTION_SET, "value": targets[0]})
        else:
            mapping[lv] = {"action": ACTION_SET, "value": _pick_level(lv, targets, order)}
    return mapping


def find_model(models_json: dict, model_id: str) -> dict:
    for model in models_json["data"]:
        if model["id"] == model_id:
            return model
    raise ValueError(f"OpenRouter /models 에 {model_id!r} 가 없다.")


def derive_rule(model: dict, vocab: dict, source: str) -> dict:
    """모델 규칙 초안 1건 — 키 순서 = model_rules 파일 형식."""
    _check_vocab(vocab)
    reasoning = model.get("reasoning", {})   # 비추론 모델(gemma-3 등)은 키 자체가 없다 → 빈 메타
    params = model["supported_parameters"]
    sampling = {k: STATUS_ACCEPTED if k in params else STATUS_REJECTED for k in vocab["sampling_keys"]}
    accepted = [k for k in vocab["sampling_keys"] if k in params]
    basis = (f"OpenRouter /models 메타({source}): reasoning="
             f"{json.dumps(reasoning, ensure_ascii=False, sort_keys=True)}; "
             f"supported_parameters∩sampling=[{', '.join(accepted)}] — derive_mapping 결정론 초안, Agent 확인 필요")
    return {
        "match": model["id"],
        "provider_class": PROVIDER_CLASS,
        "injection": INJECTION_ID,
        "mapping": derive_mapping(reasoning, vocab),
        "end_marker": None,
        "sampling": sampling,
        "basis": basis,
    }


def render_rule(rule: dict) -> str:
    """model_rules/<파일>.yaml 내용 그대로 (규칙 1건)."""
    return dump_yaml(rule)


def _load_models(models_json: str | None, fetch: bool) -> tuple[dict, str]:
    if fetch:
        with urllib.request.urlopen(OPENROUTER_MODELS_URL, timeout=FETCH_TIMEOUT_SECONDS) as resp:
            return json.load(resp), OPENROUTER_MODELS_URL
    with open(models_json, encoding="utf-8") as f:
        return json.load(f), models_json


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="OpenRouter /models 메타 → 모델 규칙 6칸 초안 (Agent 확인용)")
    ap.add_argument("--model-id", required=True, help="OpenRouter model id (예: deepseek/deepseek-v4.1-flash)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--models-json", help="캐시한 /models JSON 경로")
    src.add_argument("--fetch", action="store_true", help=f"{OPENROUTER_MODELS_URL} 를 직접 조회")
    ap.add_argument("--skill-dir", default=str(SKILL_DIR))
    a = ap.parse_args(argv)
    models, source = _load_models(a.models_json, a.fetch)
    rule = derive_rule(find_model(models, a.model_id), load_vocabulary(Path(a.skill_dir)), source)
    print(render_rule(rule), end="")
    return 0

