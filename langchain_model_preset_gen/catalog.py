"""catalog — 스킬 안의 지식 파일을 읽고 검증한다.

  data/          배포자 소유 공통 지식 (어휘·주입표·provider_class 표·공식 provider·임베딩 사실표·입력 형식·실측 전이표)
  presets/       설치자 소유 preset (파일 1개 = preset 1개, name = 파일명 stem)
  model_rules/   설치자 소유 모델 규칙 (파일 1개 = 규칙 1건)

모델 규칙·임베딩 사실표 매칭은 같은 규칙이다: provider_class 가 같고 `model_name.lower().startswith(match.lower())`
인 항목 중 match 가 **가장 긴** 것. 같은 길이로 둘 이상이면 모호(ValueError), 하나도 없으면 미등록(KeyError).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from . import injection as inj

SKILL_DIR = Path(__file__).resolve().parents[1]

DATA_FILES = {
    "vocabulary": "vocabulary.yaml",
    "injection": "injection.yaml",
    "provider_classes": "provider_classes.yaml",
    "providers": "providers.yaml",
    "embedding_capability": "embedding_capability.yaml",
    "query_mode_map": "query_mode_map.yaml",
    "probe_graph": "probe_graph.yaml",
}
RULE_KEYS = ("match", "provider_class", "injection", "mapping", "end_marker", "sampling", "basis")
PROVIDER_KEYS = ("provider_class", "base_url", "api_key_env")
PROVIDER_OPTIONAL_KEYS = ("embedding_models_path",)
YAML_SUFFIX = ".yaml"
README_NAME = "README.md"

# sampling 수용표의 effort 단계별 분기 형식: {by_effort: {<단계>: 상태, default: 상태}}
BY_EFFORT_KEY = "by_effort"
BY_EFFORT_DEFAULT = "default"
SLOT_KEYS_SET = {"action", "value"}
SLOT_KEYS_BARE = {"action"}


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data(skill_dir: Path = SKILL_DIR) -> dict:
    """data/ 7개 파일 → {이름: 내용}. 파일이 없으면 FileNotFoundError."""
    data_dir = Path(skill_dir) / "data"
    return {key: load_yaml(data_dir / name) for key, name in DATA_FILES.items()}


def load_preset(name: str, skill_dir: Path = SKILL_DIR) -> dict:
    """presets/<NAME>.yaml — 없으면 FileNotFoundError. name 필드는 파일명 stem 과 같아야 한다."""
    body = load_yaml(Path(skill_dir) / "presets" / f"{name}{YAML_SUFFIX}")
    if body["name"] != name:
        raise ValueError(f"preset 파일 presets/{name}.yaml 의 name={body['name']!r} 가 파일명 stem 과 다르다")
    return body


def load_model_rules(skill_dir: Path = SKILL_DIR) -> list[dict]:
    """model_rules/*.yaml (README 제외) → 규칙 목록. 규칙 키 집합이 RULE_KEYS 와 다르면 ValueError."""
    rules = []
    for path in sorted((Path(skill_dir) / "model_rules").glob(f"*{YAML_SUFFIX}")):
        rule = load_yaml(path)
        require_exact_keys(rule, RULE_KEYS, f"model_rules/{path.name}")
        rules.append(rule)
    return rules


def longest_prefix_match(model_name: str, provider_class: str, entries: list[dict], what: str) -> dict:
    """provider_class 일치 + 대소문자 무시 접두어 매칭 중 match 가 가장 긴 항목."""
    name = model_name.lower()
    hits = [e for e in entries if e["provider_class"] == provider_class and name.startswith(e["match"].lower())]
    if not hits:
        raise KeyError(f"{what} 미등록: model_name={model_name!r}, provider_class={provider_class!r}")
    longest = max(len(e["match"]) for e in hits)
    top = [e for e in hits if len(e["match"]) == longest]
    if len(top) > 1:
        raise ValueError(f"{what} 모호: {model_name!r} 에 같은 길이 match {[e['match'] for e in top]} 가 함께 맞는다")
    return top[0]


def find_model_rule(model_name: str, provider_class: str, rules: list[dict]) -> dict:
    return longest_prefix_match(model_name, provider_class, rules, "model_rules")


def find_embedding_capability(model_name: str, provider_class: str, data: dict) -> dict:
    """임베딩 사실표 매칭 → match·provider_class 를 뺀 사실 dict."""
    entry = longest_prefix_match(model_name, provider_class, data["embedding_capability"]["capabilities"],
                                 "embedding_capability")
    return {k: v for k, v in entry.items() if k not in ("match", "provider_class")}


def merge_providers(official: dict, site: dict) -> dict:
    """공식(data) + 사이트(configs) serving_provider. 같은 이름이 둘 다 있으면 ValueError."""
    clash = set(official) & set(site)
    if clash:
        raise ValueError(f"configs providers 가 공식 serving_provider 이름을 다시 쓴다: {sorted(clash)} — 다른 이름을 쓴다")
    merged = {**official, **site}
    for name, entry in merged.items():
        extra = set(entry) - set(PROVIDER_KEYS) - set(PROVIDER_OPTIONAL_KEYS)
        missing = [k for k in PROVIDER_KEYS if k not in entry]
        if extra or missing:
            raise ValueError(f"serving_provider {name!r} 키 위반 — 누락 {missing}, 어휘 밖 {sorted(extra)}")
    return merged


def resolve_provider(serving_provider: str, providers: dict) -> dict:
    if serving_provider not in providers:
        raise ValueError(f"unknown serving_provider: {serving_provider!r} (known: {sorted(providers)}) "
                         f"— 사이트 엔드포인트는 configs/config.yaml providers 에 등록한다")
    return providers[serving_provider]


# ── 검증 (발행·해석·정적 검사가 같은 함수를 쓴다) ─────────────────────────────

def require_exact_keys(actual: dict, expected, what: str) -> None:
    missing = [k for k in expected if k not in actual]
    extra = [k for k in actual if k not in expected]
    if missing or extra:
        raise ValueError(f"{what} 키 불일치 — 누락 {missing}, 어휘 밖 {extra} (필수: {list(expected)})")


def _validate_slot(level: str, slot: dict, inj_id: str, spec: dict, vocab: dict) -> None:
    action = slot["action"]
    if action not in vocab["slot_actions"]:
        raise ValueError(f"mapping.{level} action 어휘 위반: {action!r} (allowed: {vocab['slot_actions']})")
    if action == inj.SLOT_SET:
        if set(slot) != SLOT_KEYS_SET or slot["value"] is None:
            raise ValueError(f"mapping.{level}: action=set 은 value 가 필요하다 — {slot!r}")
        value = slot["value"]
        in_table = "values" in spec and value in spec["values"]
        if not in_table and "value_path" not in spec:
            raise ValueError(f"mapping.{level}: value={value!r} 를 injection {inj_id!r} 로 주입할 수 없다 "
                             f"(values 표에 없고 value_path 도 없음)")
        return
    if set(slot) != SLOT_KEYS_BARE:
        raise ValueError(f"mapping.{level}: action={action} 은 value 를 갖지 않는다 — {slot!r}")
    if action == inj.SLOT_DISABLE and "disable" not in spec:
        raise ValueError(f"mapping.{level}: injection {inj_id!r} 에 disable 이 없다 — omit 또는 단계값으로 매핑")


def validate_effort_axis(inj_id: str, mapping: dict, vocab: dict, injection_table: dict, what: str) -> dict:
    """6칸 매핑 검증 → 정규화 사본. 칸 누락·어휘 밖 칸·미등록 injection·주입 불가 값은 ValueError."""
    levels = vocab["effort_levels"]
    missing = [lv for lv in levels if lv not in mapping]
    if missing:
        raise ValueError(f"{what}: 6칸 누락 {missing} — 6칸 전부 필수")
    extra = [lv for lv in mapping if lv not in levels]
    if extra:
        raise ValueError(f"{what}: effort_levels 밖 칸 {extra} (allowed: {levels})")
    if inj_id not in injection_table:
        raise ValueError(f"{what}: injection {inj_id!r} 미등록 (data/injection.yaml: {sorted(injection_table)})")
    normalized = {lv: inj.normalize_slot(mapping[lv]) for lv in levels}
    for lv, slot in normalized.items():
        _validate_slot(lv, slot, inj_id, injection_table[inj_id], vocab)
    return normalized


def validate_sampling(sampling: dict, vocab: dict, what: str) -> None:
    """키별 accepted|rejected|ignored 또는 {by_effort: {<단계>|default: 상태}}."""
    require_exact_keys(sampling, vocab["sampling_keys"], f"{what} sampling")
    statuses = vocab["sampling_status"]
    levels = vocab["effort_levels"]
    for key, status in sampling.items():
        if not isinstance(status, dict):
            if status not in statuses:
                raise ValueError(f"sampling.{key} 상태 어휘 위반: {status!r} (allowed: {statuses})")
            continue
        require_exact_keys(status, (BY_EFFORT_KEY,), f"sampling.{key} 분기형")
        table = status[BY_EFFORT_KEY]
        for level, st in table.items():
            if level not in levels and level != BY_EFFORT_DEFAULT:
                raise ValueError(f"sampling.{key}.by_effort 단계 어휘 위반: {level!r} "
                                 f"(allowed: {levels} + {BY_EFFORT_DEFAULT})")
            if st not in statuses:
                raise ValueError(f"sampling.{key}.by_effort.{level} 상태 어휘 위반: {st!r} (allowed: {statuses})")
        uncovered = [lv for lv in levels if lv not in table]
        if BY_EFFORT_DEFAULT not in table and uncovered:
            raise ValueError(f"sampling.{key}.by_effort: default 가 없고 미기재 단계 {uncovered} 가 있다")


def sampling_status(entry: str | dict, effort: str) -> str:
    """수용표 한 키 → 그 effort 단계의 상태. `{by_effort: ...}` 는 단계 → 없으면 default."""
    if isinstance(entry, str):
        return entry
    by_effort = entry[BY_EFFORT_KEY]
    return by_effort[effort] if effort in by_effort else by_effort[BY_EFFORT_DEFAULT]


def validate_model_rule(rule: dict, vocab: dict, injection_table: dict) -> dict:
    """규칙 검증 + 정규화 사본 {injection, mapping, end_marker, basis, sampling}. 원본은 바꾸지 않는다."""
    what = f"model rule {rule['match']!r}"
    mapping = validate_effort_axis(rule["injection"], rule["mapping"], vocab, injection_table, what)
    validate_sampling(rule["sampling"], vocab, what)
    return {
        "injection": rule["injection"],
        "mapping": mapping,
        "end_marker": rule["end_marker"],
        "basis": rule["basis"],
        "sampling": rule["sampling"],
    }
