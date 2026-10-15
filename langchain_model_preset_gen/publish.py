"""publish — 구조화 필드 → 프로필 yaml 발행 (결정론).

순서: serving_provider 해소(공식 data + 사이트 configs, 같은 이름은 거부) → preset 로드·어휘 검증 →
모델 규칙 최장 접두어 매칭 → 6칸·sampling 검증 → 값 펼침 → 파일명 → 배타 생성(같은 이름이 있으면 FileExistsError).

값 펼침: 프로필 preset_values·limits = preset 값 위에 configs overridable 의 null 이 아닌 기본값을 얹은 결과.
프로필 overridable = configs overridable 의 키 목록(LLM 어휘 순서) — 임베딩은 그중 timeout_seconds·max_retries 에서
SDK 미지원 인자를 뺀 것. 발행 직후 probe 는 미측정 상태(status probe_error, method static)다.

파일명 = {sanitize(model_name)}_{preset}_{serving_provider}.yaml. sanitize = [A-Za-z0-9._-] 밖 문자 → '_'.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import re
import sys
from datetime import date
from pathlib import Path

import yaml
from packaging.specifiers import SpecifierSet

from . import catalog
from . import resolver as rs
from . import settings as st

SKILL_DIR = catalog.SKILL_DIR
_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")

# 발행 직후 probe 초기값 — 실측 write-back 이 갱신한다.
PROBE_INITIAL_STATUS = "probe_error"
LLM_STATIC_PROBE_METHOD = "static"
EMBEDDING_STATIC_PROBE_METHOD = "static_v5"

# yaml 줄바꿈 폭 — 기본 80 은 흐름 목록을 중간에서 접어 사람이 읽기 어렵다.
YAML_LINE_WIDTH = 1000

HEADER_LINE_1 = ("# langchain-model-preset-gen 발행 — 사용자 편집 가능. "
                 "재발행하려면 이 파일을 지운 뒤 다시 생성한다(같은 이름은 생성 거부).")
HEADER_LINE_2 = "# preset 출처: {preset} (스킬 presets/, 발행일 {published}) · 실측: 미측정"


def sanitize(s: object) -> str:
    """[A-Za-z0-9._-] 밖 문자를 '_' 로 치환 — 서빙 model_id 문자셋과 무손실 호환."""
    return _SAFE_RE.sub("_", str(s))


def derive_filename(model_name: str, preset: str, serving_provider: str) -> str:
    return f"{sanitize(model_name)}_{sanitize(preset)}_{sanitize(serving_provider)}.yaml"


# ── preset 검증 ─────────────────────────────────────────────────────────────

def validate_llm_preset(preset: dict, vocab: dict) -> None:
    name = preset["name"]
    if preset["intent"] not in vocab["intent"]["values"]:
        raise ValueError(f"LLM preset {name!r} intent 어휘 위반: {preset['intent']!r} (allowed: {vocab['intent']['values']})")
    if preset["effort"] not in vocab["effort_levels"]:
        raise ValueError(f"LLM preset {name!r} effort 어휘 위반: {preset['effort']!r} — 6단계 중 하나, null 금지 "
                         f"(allowed: {vocab['effort_levels']})")
    catalog.require_exact_keys(preset["sampling"], vocab["sampling_keys"], f"preset {name!r} sampling")
    catalog.require_exact_keys(preset["limits"], rs.LLM_LIMIT_KEYS, f"preset {name!r} limits")


def validate_embedding_preset(preset: dict, vocab: dict) -> None:
    name = preset["name"]
    if preset["intent"] not in vocab["embedding_intent"]["values"]:
        raise ValueError(f"embedding preset {name!r} intent 어휘 위반: {preset['intent']!r} "
                         f"(allowed: {vocab['embedding_intent']['values']})")
    if preset["modality"] not in vocab["embedding_modality"]["values"]:
        raise ValueError(f"embedding preset {name!r} modality 어휘 위반: {preset['modality']!r} "
                         f"(allowed: {vocab['embedding_modality']['values']})")
    catalog.require_exact_keys(preset["limits"], rs.EMBEDDING_LIMIT_KEYS, f"preset {name!r} limits")


def overlay_defaults(values: dict, overridable: dict) -> dict:
    """values 의 키 중 configs overridable 기본값이 null 이 아닌 것을 그 값으로 바꾼 사본."""
    return {k: (overridable[k] if overridable.get(k) is not None else v) for k, v in values.items()}


# ── dist 버전 (임베딩 probe.measured_with·requires) ──────────────────────────

def measured_with(pc: str, provider_classes: dict) -> str:
    """'<dist>==<설치본>'. 미설치 dist 는 PackageNotFoundError."""
    dist = provider_classes[pc]["langchain_dist"]
    return f"{dist}=={md.version(dist)}"


def requires_range(pc: str, provider_classes: dict) -> str:
    """'<dist><requires_range>'. 설치본이 검증 구간 밖이면 ValueError(재검증 후 구간 갱신)."""
    entry = provider_classes[pc]
    dist, spec = entry["langchain_dist"], entry["requires_range"]
    installed = md.version(dist)
    if installed not in SpecifierSet(spec):
        raise ValueError(f"{dist}=={installed} 가 data/provider_classes.yaml requires_range {spec!r} 밖 — "
                         f"재검증 후 requires_range 를 갱신한다")
    return f"{dist}{spec}"


# ── 프로필 조립 ─────────────────────────────────────────────────────────────

def _context(skill_dir: Path) -> dict:
    cfg = st.load_settings(skill_dir)
    data = catalog.load_data(skill_dir)
    return {"settings": cfg, "data": data, "providers": catalog.merge_providers(data["providers"], cfg["providers"])}


def _load_block(model_name: str, serving_provider: str, providers: dict) -> dict:
    if not sanitize(model_name):
        raise ValueError(f"model_name 이 sanitize 후 빈 문자열: {model_name!r}")
    prov = catalog.resolve_provider(serving_provider, providers)
    return {"model_name": model_name, "serving_provider": serving_provider, "provider_class": prov["provider_class"],
            "base_url": prov["base_url"], "api_key_env": prov["api_key_env"]}


def build_llm_profile(model_name: str, preset_name: str, serving_provider: str, skill_dir: Path = SKILL_DIR) -> dict:
    ctx = _context(skill_dir)
    vocab = ctx["data"]["vocabulary"]
    overridable = ctx["settings"]["overridable"]
    load = _load_block(model_name, serving_provider, ctx["providers"])
    preset = catalog.load_preset(preset_name, skill_dir)
    validate_llm_preset(preset, vocab)
    rule = catalog.validate_model_rule(
        catalog.find_model_rule(model_name, load["provider_class"], catalog.load_model_rules(skill_dir)),
        vocab, ctx["data"]["injection"])
    preset_values = overlay_defaults({**{k: preset["sampling"][k] for k in vocab["sampling_keys"]},
                                      "effort": preset["effort"]}, overridable)
    if preset_values["effort"] not in vocab["effort_levels"]:
        raise ValueError(f"configs overridable.effort 기본값 어휘 위반: {preset_values['effort']!r} "
                         f"(allowed: {vocab['effort_levels']})")
    return {
        "kind": rs.KIND_LLM,
        **load,
        "preset": preset_name,
        "overridable": [k for k in rs.LLM_OVERRIDABLE if k in overridable],
        "reasoning_effort": {k: rule[k] for k in ("injection", "mapping", "end_marker", "basis")},
        "sampling": rule["sampling"],
        "preset_values": preset_values,
        "limits": overlay_defaults({k: preset["limits"][k] for k in rs.LLM_LIMIT_KEYS}, overridable),
        "probe": {
            "status": PROBE_INITIAL_STATUS,
            "method": LLM_STATIC_PROBE_METHOD,
            "measured_at": None,
            "terminal": None,
            "runnable": {m: PROBE_INITIAL_STATUS for m in vocab["runnable_methods_llm"]},
        },
    }


def build_embedding_profile(model_name: str, preset_name: str, serving_provider: str,
                            dimensions: int | None = None, skill_dir: Path = SKILL_DIR) -> dict:
    ctx = _context(skill_dir)
    data = ctx["data"]
    vocab = data["vocabulary"]
    overridable = ctx["settings"]["overridable"]
    load = _load_block(model_name, serving_provider, ctx["providers"])
    preset = catalog.load_preset(preset_name, skill_dir)
    validate_embedding_preset(preset, vocab)
    pc = load["provider_class"]
    cap = catalog.find_embedding_capability(model_name, pc, data)
    native = cap["native_dimensions"]
    # dimensions(MRL 절단 요청): null = 미주입. native 와 같으면 null 과 결과가 같은 inert knob.
    if dimensions is not None:
        if int(dimensions) == native:
            raise ValueError(f"dimensions={dimensions} 가 native({native})와 같다 — null 과 결과가 같은 inert knob. "
                             f"축소하지 않으면 --dimensions 를 생략한다")
        if int(dimensions) > native:
            raise ValueError(f"dimensions={dimensions} > native({native}) — MRL 절단은 native 이하만 가능")
    classes = data["provider_classes"]
    unsupported = classes[pc]["sdk"]["embedding_unsupported_kwargs"]
    supports = {
        "query_mode_surface": cap["query_mode_surface"],
        "prefix_template": cap["prefix_template"],
        "task_text_injectable": cap["task_text_injectable"],
        "normalization": None,                      # 실측 raw 노름에서 파생 — 정적 층은 모른다
        "native_dimensions": native,
        "input_modalities": list(cap["input_modalities"]),
    }
    if "document_template" in cap:
        supports["document_template"] = cap["document_template"]
    return {
        "kind": rs.KIND_EMBEDDING,
        **load,
        "preset": preset_name,
        "overridable": [k for k in rs.EMBEDDING_OVERRIDABLE if k in overridable and k not in unsupported],
        "dimensions": None if dimensions is None else int(dimensions),
        "preset_values": {"intent": preset["intent"], "modality": preset["modality"]},
        "limits": overlay_defaults({k: preset["limits"][k] for k in rs.EMBEDDING_LIMIT_KEYS}, overridable),
        "supports": supports,
        "probe": {
            "status": PROBE_INITIAL_STATUS,
            "method": EMBEDDING_STATIC_PROBE_METHOD,
            "measured_at": "",
            "measured_with": measured_with(pc, classes),
            "requires": requires_range(pc, classes),
            "measured_at_dimensions": None,
            "norm_measured": None,
            "runnable": {m: PROBE_INITIAL_STATUS for m in vocab["runnable_methods_embedding"]},
        },
    }


def dump_yaml(block: dict) -> str:
    return yaml.safe_dump(block, sort_keys=False, allow_unicode=True, default_flow_style=None, width=YAML_LINE_WIDTH)


def render_yaml(profile: dict, published: date) -> str:
    header = HEADER_LINE_1 + "\n" + HEADER_LINE_2.format(preset=profile["preset"], published=published.isoformat()) + "\n"
    return header + dump_yaml(profile)


def generate(model_name: str, preset: str, serving_provider: str, kind: str, *, out_dir: Path | None = None,
             write: bool = True, dimensions: int | None = None, skill_dir: Path = SKILL_DIR,
             today: date | None = None) -> tuple[Path, dict]:
    """프로필 발행. out_dir=None 이면 configs profiles_dir."""
    if kind == rs.KIND_LLM:
        if dimensions is not None:
            raise ValueError("dimensions 는 kind=embedding 전용이다")
        profile = build_llm_profile(model_name, preset, serving_provider, skill_dir)
    elif kind == rs.KIND_EMBEDDING:
        profile = build_embedding_profile(model_name, preset, serving_provider, dimensions, skill_dir)
    else:
        raise ValueError(f"kind 는 llm|embedding: {kind!r}")
    published = date.today() if today is None else today
    target = st.load_settings(skill_dir)["profiles_dir"] if out_dir is None else Path(out_dir)
    path = target / derive_filename(model_name, preset, serving_provider)
    if write:
        target.mkdir(parents=True, exist_ok=True)
        with open(path, "x", encoding="utf-8") as f:     # 'x' = 배타 생성 — 같은 이름이 있으면 FileExistsError
            f.write(render_yaml(profile, published))
    return path, profile


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="langchain-model-preset-gen 프로필 발행 (결정론)")
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--preset", required=True, help="스킬 presets/<NAME>.yaml 의 NAME")
    ap.add_argument("--provider", required=True, help="serving_provider (data/providers.yaml 또는 configs providers)")
    ap.add_argument("--kind", required=True, choices=[rs.KIND_LLM, rs.KIND_EMBEDDING])
    ap.add_argument("--dimensions", type=int, default=None,
                    help="임베딩 MRL 절단 요청값 (생략 = null 미주입). native 와 같은 값은 inert → 거부")
    ap.add_argument("--out-dir", default=None, help="발행 위치 (기본 = configs/config.yaml profiles_dir)")
    ap.add_argument("--skill-dir", default=str(SKILL_DIR))
    ap.add_argument("--dry-run", action="store_true", help="파일을 만들지 않고 미리보기만 출력")
    a = ap.parse_args(argv)
    published = date.today()
    path, profile = generate(a.model_name, a.preset, a.provider, a.kind, out_dir=a.out_dir, write=not a.dry_run,
                             dimensions=a.dimensions, skill_dir=Path(a.skill_dir), today=published)
    print(f"{'[dry-run] ' if a.dry_run else ''}{path}")
    print(render_yaml(profile, published))
    return 0


if __name__ == "__main__":
    sys.exit(main())
