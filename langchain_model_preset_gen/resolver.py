"""resolver — 프로필 + override → LangChain ChatModel / Embeddings. 이 패키지의 유일한 해석 지점이다.

  spec  = resolve(profile, override)            # 실효값·잠금 검사·주입 extras·SDK 인자 (I/O = 파일 읽기만)
  model = load_chat_model(profile, override)    # spec → ChatOpenAI / ChatAnthropic / ChatGoogleGenerativeAI
  emb   = load_embeddings(profile, override)    # spec → ProfileEmbeddings(OpenAIEmbeddings / GoogleGenerativeAIEmbeddings)

실효값 = 프로필 값(preset_values·limits — 발행 시 configs overridable 기본값이 이미 반영됨) ← override.
잠금: override 키 ⊆ 프로필 overridable. 밖이면 LockedKeyError. sampling override 는 그 실효 effort 에서 수용표가
accepted 일 때만 받는다(효과 없는 값을 받지 않는다). 잠긴 키 = model_name·base_url·provider_class·api_key_env·
reasoning_effort·sampling·probe(·임베딩 supports·dimensions) — 프로필 overridable 에 올 수 없다.

SDK 인자 = (실효 effort 칸 주입 extras) + (limits: max_tokens·timeout·max_retries) + (그 단계 accepted 이고 null 이
아닌 sampling). 대응표는 data/provider_classes.yaml sdk. response_deadline_seconds 는 SDK 인자가 아니다(서버 처리부 knob).
API 키는 spec 에 넣지 않는다 — load_* 가 settings.resolve_api_key 로 해소해 생성자에만 넘긴다.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from pathlib import Path

from pydantic import AliasChoices

from . import catalog
from . import injection as inj
from . import settings
from .embeddings import ProfileEmbeddings

SKILL_DIR = catalog.SKILL_DIR

KIND_LLM = "llm"
KIND_EMBEDDING = "embedding"
ACCEPTED = "accepted"
EFFORT_KEY = "effort"
# 프로필 계약 키 — 발행 형식과 같다(조정 대상이 아니다).
LLM_LIMIT_KEYS = ("timeout_seconds", "max_retries", "response_deadline_seconds", "max_tokens")
EMBEDDING_LIMIT_KEYS = ("timeout_seconds", "max_retries")
LLM_OVERRIDABLE = settings.OVERRIDABLE_VOCABULARY
EMBEDDING_OVERRIDABLE = EMBEDDING_LIMIT_KEYS
# SDK 로 보내는 limits 키 (response_deadline_seconds 제외)
SDK_LIMIT_KEYS = ("max_tokens", "timeout_seconds", "max_retries")
EMBEDDING_DIMENSIONS_KEY = "dimensions"
# api_key_env null = 무인증 자체 호스팅의 선언. LangChain 생성자는 키 문자열을 요구해 자리표시 문자열을 넣는다.
NO_AUTH_API_KEY = "not-needed"
MODEL_KWARG = "model"
# SDK 가 생성자 밖 요청 본문 인자를 받는 통로 — 프로필이 관리하는 인자를 우회할 수 있어 client_kwargs 로 받지 않는다.
MODEL_KWARGS_FIELD = "model_kwargs"
URL_PATH_SEP = "/"


class LockedKeyError(ValueError):
    """프로필 overridable 밖의 키를 override 하려 했다."""


def read_profile(profile: str | Path | Mapping) -> tuple[dict, str]:
    """경로 또는 dict → (프로필 dict, 이름). 이름은 오류 메시지용(파일명 또는 model_name)."""
    if isinstance(profile, Mapping):
        return dict(profile), str(profile["model_name"])
    path = Path(profile)
    return catalog.load_yaml(path), path.name


def check_overridable(overridable: list, allowed, name: str) -> None:
    """프로필 overridable ⊆ 그 kind 의 어휘. 밖의 키는 해석기가 SDK 로 보내지 않아 `sent` 에만 남는 무효 knob 이 된다."""
    outside = sorted(set(overridable) - set(allowed))
    if outside:
        raise ValueError(f"프로필 {name} 의 overridable 에 어휘 밖 키 {outside} — 허용: {list(allowed)}")


def check_override(override: Mapping, overridable: list, name: str) -> None:
    locked = sorted(set(override) - set(overridable))
    if locked:
        raise LockedKeyError(f"잠긴 키 override 거부: {locked} — 프로필 {name} 의 overridable={list(overridable)}")


def native_base_url(profile: Mapping, provider_classes: dict) -> str:
    """원문 HTTP 용 endpoint: 프로필 base_url, null 이면 provider_class native_base_url. 둘 다 null 이면 ValueError."""
    base = profile["base_url"]
    if base is None:
        base = provider_classes[profile["provider_class"]]["native_base_url"]
    if base is None:
        raise ValueError(f"base_url 이 null 인데 provider_class {profile['provider_class']!r} 에 native_base_url 이 없다 "
                         f"— openai_compatible 프로필은 base_url 이 필요하다")
    return base.rstrip(URL_PATH_SEP)


def _top_segments(paths) -> set[str]:
    return {p.split(inj.PATH_SEP)[0] for p in paths}


def _injection_paths(injection_table: dict) -> set[str]:
    paths: set[str] = set()
    for spec in injection_table.values():
        paths |= set(spec.get("disable", {}))
        for value in spec.get("values", {}).values():
            paths |= set(value)
        if "value_path" in spec:
            paths.add(spec["value_path"])
    return paths


def resolve(profile: str | Path | Mapping, override: Mapping | None = None, *,
            skill_dir: Path = SKILL_DIR, data: dict | None = None) -> dict:
    """프로필 + override → spec. data = catalog.load_data() 결과(테스트 주입용, 기본은 skill_dir/data)."""
    body, name = read_profile(profile)
    data = catalog.load_data(skill_dir) if data is None else data
    override = dict(override or {})
    kind = body["kind"]
    if kind == KIND_LLM:
        return _resolve_llm(body, override, data, name)
    if kind == KIND_EMBEDDING:
        return _resolve_embedding(body, override, data, name)
    raise ValueError(f"kind 는 llm|embedding: {kind!r} — 프로필 {name}")


def _load_block(body: dict) -> dict:
    return {k: body[k] for k in ("model_name", "serving_provider", "provider_class", "base_url", "api_key_env")}


def _resolve_llm(body: dict, override: dict, data: dict, name: str) -> dict:
    vocab = data["vocabulary"]
    re_ = body["reasoning_effort"]
    mapping = catalog.validate_effort_axis(re_["injection"], re_["mapping"], vocab, data["injection"],
                                           f"프로필 {name} reasoning_effort")
    catalog.validate_sampling(body["sampling"], vocab, f"프로필 {name}")
    overridable = list(body["overridable"])
    check_overridable(overridable, LLM_OVERRIDABLE, name)
    check_override(override, overridable, name)

    preset_values, limits = body["preset_values"], body["limits"]
    defaults = {key: preset_values[key] for key in vocab["sampling_keys"]}
    defaults[EFFORT_KEY] = preset_values[EFFORT_KEY]
    defaults.update({key: limits[key] for key in LLM_LIMIT_KEYS})
    effective = {**defaults, **override}
    effort = effective[EFFORT_KEY]
    if effort not in vocab["effort_levels"]:
        raise ValueError(f"effort={effort!r} 는 어휘 밖이다 (허용: {vocab['effort_levels']}) — 프로필 {name}")
    status = {key: catalog.sampling_status(body["sampling"][key], effort) for key in vocab["sampling_keys"]}
    not_accepted = {k: status[k] for k in override if k in status and status[k] != ACCEPTED}
    if not_accepted:
        raise ValueError(f"sampling override 거부 — 실효 effort={effort!r} 에서 {not_accepted} "
                         f"(accepted 인 키만 override 할 수 있다) — 프로필 {name}")
    sent = {k: (None if k in status and status[k] != ACCEPTED else v) for k, v in effective.items()}

    pc = body["provider_class"]
    sdk = data["provider_classes"][pc]["sdk"]
    slot = mapping[effort]
    extras = inj.expand_slot(data["injection"], re_["injection"], slot)
    flat = {sdk["kwarg_paths"][key]: effective[key] for key in SDK_LIMIT_KEYS}
    for key in vocab["sampling_keys"]:
        if status[key] == ACCEPTED and effective[key] is not None:
            flat[sdk["kwarg_paths"][key]] = effective[key]
    kwargs = inj.deep_merge({MODEL_KWARG: body["model_name"]}, extras)
    kwargs = inj.deep_merge(kwargs, inj.expand_paths(flat))
    if sdk["base_url_kwarg"] is not None:
        kwargs[sdk["base_url_kwarg"]] = body["base_url"]
    reserved = (_top_segments(sdk["kwarg_paths"].values()) | _top_segments(_injection_paths(data["injection"]))
                | set(sdk["chat_locked_kwargs"])
                | {MODEL_KWARG, MODEL_KWARGS_FIELD, sdk["api_key_kwarg"]} | ({sdk["base_url_kwarg"]} - {None}))
    return {
        "kind": KIND_LLM,
        "name": name,
        **_load_block(body),
        "http_base_url": native_base_url(body, data["provider_classes"]),
        "overridable": overridable,
        "override": override,
        "injection": re_["injection"],
        "injection_rules": data["injection"][re_["injection"]],
        "mapping": mapping,
        "end_marker": re_["end_marker"],
        "sampling": body["sampling"],
        "sampling_status": status,
        "effective": effective,
        "sent": sent,
        "effort_slot": slot,
        "probe": body["probe"],
        "sdk_class": sdk["chat_class"],
        "sdk_kwargs": kwargs,
        "api_key_kwarg": sdk["api_key_kwarg"],
        "reserved_kwargs": sorted(reserved),
    }


def _resolve_embedding(body: dict, override: dict, data: dict, name: str) -> dict:
    supports, probe = body["supports"], body["probe"]
    modality = body["preset_values"]["modality"]
    if modality not in supports["input_modalities"]:
        raise ValueError(f"preset_values.modality={modality!r} ∉ supports.input_modalities="
                         f"{supports['input_modalities']!r} — 프로필 {name}")
    native = supports["native_dimensions"]
    requested = body[EMBEDDING_DIMENSIONS_KEY]
    if requested is not None and requested == native:
        raise ValueError(f"dimensions={requested} == native({native}) — null 과 결과가 같은 inert knob. "
                         f"축소하지 않으면 dimensions: null 로 재발행 — 프로필 {name}")
    if requested is not None and requested > native:
        raise ValueError(f"dimensions={requested} > native({native}) — MRL 절단은 native 이하만 — 프로필 {name}")
    effective_dim = native if requested is None else requested
    measured = probe["measured_at_dimensions"]          # null = 미측정 (측정 여부 판단은 probe.status 의 몫)
    if measured is not None and measured != effective_dim:
        raise ValueError(f"실효 차원 {effective_dim} ≠ probe.measured_at_dimensions {measured} — 실측이 낡았다. "
                         f"preflight runnable 재실측 필요 — 프로필 {name}")
    overridable = list(body["overridable"])
    check_overridable(overridable, EMBEDDING_OVERRIDABLE, name)
    check_override(override, overridable, name)
    pc = body["provider_class"]
    sdk = data["provider_classes"][pc]["sdk"]
    unsupported = list(sdk["embedding_unsupported_kwargs"])
    inert = sorted(set(override) & set(unsupported))
    if inert:
        raise ValueError(f"임베딩 override {inert} 는 provider_class={pc} SDK 가 받지 않는 인자다 — 프로필 {name}")
    limits = body["limits"]
    effective = {**{key: limits[key] for key in EMBEDDING_LIMIT_KEYS}, **override}
    paths = sdk["embedding_kwarg_paths"]
    flat = {paths[EMBEDDING_DIMENSIONS_KEY]: requested}
    flat.update({paths[key]: effective[key] for key in EMBEDDING_LIMIT_KEYS if key not in unsupported})
    kwargs = inj.deep_merge({MODEL_KWARG: body["model_name"]}, inj.expand_paths(flat))
    kwargs = inj.deep_merge(kwargs, dict(sdk["embedding_fixed_kwargs"]))
    if sdk["base_url_kwarg"] is not None:
        kwargs[sdk["base_url_kwarg"]] = body["base_url"]
    reserved = (_top_segments(paths.values()) | set(sdk["embedding_fixed_kwargs"])
                | {MODEL_KWARG, MODEL_KWARGS_FIELD, sdk["api_key_kwarg"]} | ({sdk["base_url_kwarg"]} - {None}))
    surface = supports["query_mode_surface"]
    return {
        "kind": KIND_EMBEDDING,
        "name": name,
        **_load_block(body),
        "http_base_url": native_base_url(body, data["provider_classes"]),
        "overridable": overridable,
        "override": override,
        "dimensions": requested,
        "effective_dimensions": effective_dim,
        "supports": supports,
        "mechanism": data["query_mode_map"][surface]["mechanism"],     # 미등록 surface → 자연 KeyError
        "unsupported_kwargs": unsupported,
        "effective": effective,
        "sent": {k: (None if k in unsupported else v) for k, v in effective.items()},   # SDK 로 가지 않는 키는 null
        "probe": probe,
        "sdk_class": sdk["embedding_class"],                           # 임베딩 없는 class(anthropic) → 자연 KeyError
        "sdk_kwargs": kwargs,
        "api_key_kwarg": sdk["api_key_kwarg"],
        "reserved_kwargs": sorted(reserved),
    }


def slot_extras(spec: dict, level: str) -> dict:
    """spec 의 6칸 중 한 칸 → SDK 인자 extras (실측 그래프의 off·on 요청이 쓴다)."""
    return inj.expand_slot({spec["injection"]: spec["injection_rules"]}, spec["injection"], spec["mapping"][level])


def import_class(dotted: str):
    module, _, attr = dotted.partition(":")
    return getattr(importlib.import_module(module), attr)


def constructor_kwargs(spec: dict, api_key: str | None, client_kwargs: Mapping | None = None) -> dict:
    """spec SDK 인자 + 호출자 추가 인자 + 키. 추가 인자가 프로필이 관리하는 인자 이름이면 LockedKeyError."""
    extra = dict(client_kwargs or {})
    clash = sorted(set(extra) & set(spec["reserved_kwargs"]))
    if clash:
        raise LockedKeyError(f"client_kwargs {clash} 는 프로필이 관리하는 인자다 — override 로 바꾼다 (프로필 {spec['name']})")
    kwargs = {**spec["sdk_kwargs"], **extra}
    kwargs[spec["api_key_kwarg"]] = NO_AUTH_API_KEY if api_key is None else api_key
    return kwargs


def _field_of(sdk_class) -> dict[str, str]:
    """pydantic SDK 클래스의 생성자 인자 이름(필드명·alias·validation_alias) → 필드명.
    LangChain 생성자는 필드명과 alias 를 둘 다 받는다(예: ChatOpenAI max_completion_tokens → max_tokens)."""
    names: dict[str, str] = {}
    for field, info in sdk_class.model_fields.items():
        names[field] = field
        for alias in (info.alias, info.validation_alias):
            choices = alias.choices if isinstance(alias, AliasChoices) else [alias]
            names.update({choice: field for choice in choices if isinstance(choice, str)})
    return names


def check_client_fields(sdk_class, spec: dict, client_kwargs: Mapping | None) -> None:
    """client_kwargs 를 SDK 필드 단위로 검사한다. 이름 단위 잠금(constructor_kwargs)은 alias 로 우회할 수 있다.
      · 프로필이 관리하는 필드와 같은 필드 → LockedKeyError
      · SDK 필드가 아닌 이름 → ValueError (LangChain 은 모르는 이름을 model_kwargs 로 옮겨 요청 본문에 싣는다)"""
    extra = dict(client_kwargs or {})
    if not extra:
        return
    field_of = _field_of(sdk_class)
    unknown = sorted(set(extra) - set(field_of))
    if unknown:
        raise ValueError(f"client_kwargs {unknown} 는 {spec['sdk_class']} 의 SDK 필드가 아니다 (프로필 {spec['name']})")
    locked = {field_of.get(name, name) for name in spec["reserved_kwargs"]}
    clash = sorted(name for name in extra if field_of[name] in locked)
    if clash:
        raise LockedKeyError(f"client_kwargs {clash} 는 프로필이 관리하는 인자(alias 포함)다 — override 로 바꾼다 "
                             f"(프로필 {spec['name']})")


def build_chat_model(spec: dict, api_key: str | None, client_kwargs: Mapping | None = None):
    kwargs = constructor_kwargs(spec, api_key, client_kwargs)      # 이름 단위 잠금 검사가 SDK import 보다 먼저다
    sdk_class = import_class(spec["sdk_class"])
    check_client_fields(sdk_class, spec, client_kwargs)
    return sdk_class(**kwargs)


def build_embeddings(spec: dict, api_key: str | None, query_instruction: str | None = None,
                     client_kwargs: Mapping | None = None) -> ProfileEmbeddings:
    kwargs = constructor_kwargs(spec, api_key, client_kwargs)
    sdk_class = import_class(spec["sdk_class"])
    check_client_fields(sdk_class, spec, client_kwargs)
    return ProfileEmbeddings(sdk_class(**kwargs), spec["supports"], spec["mechanism"], query_instruction)


def _api_key(spec: dict, skill_dir: Path) -> str | None:
    return settings.resolve_api_key(spec["api_key_env"], settings.load_settings(skill_dir)["env_file_path"])


def load_chat_model(profile: str | Path | Mapping, override: Mapping | None = None, *,
                    skill_dir: Path = SKILL_DIR, client_kwargs: Mapping | None = None):
    """LLM 프로필 → LangChain ChatModel. client_kwargs = 프로필이 관리하지 않는 생성자 인자(예: stream_usage)."""
    spec = resolve(profile, override, skill_dir=skill_dir)
    if spec["kind"] != KIND_LLM:
        raise ValueError(f"load_chat_model 은 LLM 프로필만 받는다 — {spec['name']} kind={spec['kind']}")
    return build_chat_model(spec, _api_key(spec, skill_dir), client_kwargs)


def load_embeddings(profile: str | Path | Mapping, override: Mapping | None = None, *,
                    skill_dir: Path = SKILL_DIR, query_instruction: str | None = None,
                    client_kwargs: Mapping | None = None) -> ProfileEmbeddings:
    """임베딩 프로필 → ProfileEmbeddings. instruct_prefix 모델은 query_instruction 이 필요하다."""
    spec = resolve(profile, override, skill_dir=skill_dir)
    if spec["kind"] != KIND_EMBEDDING:
        raise ValueError(f"load_embeddings 는 임베딩 프로필만 받는다 — {spec['name']} kind={spec['kind']}")
    return build_embeddings(spec, _api_key(spec, skill_dir), query_instruction, client_kwargs)
