"""배포본 구조·데이터 불변식 — 어휘 분리·주입표↔provider_class 정합·preset 유효성·하드코딩 경로 0건."""

from __future__ import annotations

import re

import yaml

from conftest import PKG_ROOT
from langchain_model_preset_gen import catalog
from langchain_model_preset_gen import publish as pb

DATA = catalog.load_data(PKG_ROOT)
PY_FILES = sorted(p for p in PKG_ROOT.rglob("*.py") if ".git" not in p.parts)


def test_model_rules_not_in_data_or_presets():
    for folder in ("data", "presets"):
        for path in (PKG_ROOT / folder).glob("*.yaml"):
            assert not re.search(r"^model_rules:", path.read_text(encoding="utf-8"), re.M), path


def test_every_provider_class_has_sdk_and_http():
    for name, entry in DATA["providers"].items():
        pc = DATA["provider_classes"][entry["provider_class"]]
        assert {"chat_class", "api_key_kwarg", "base_url_kwarg", "kwarg_paths"} <= set(pc["sdk"]), name
        assert {"api", "chat_path", "models_path", "max_tokens_path"} <= set(pc["http"]), name


def test_embedding_capable_classes_declare_embedding_sdk():
    for cap in DATA["embedding_capability"]["capabilities"]:
        sdk = DATA["provider_classes"][cap["provider_class"]]["sdk"]
        assert {"embedding_class", "embedding_kwarg_paths", "embedding_fixed_kwargs",
                "embedding_unsupported_kwargs"} <= set(sdk), cap["match"]
        assert cap["query_mode_surface"] in DATA["query_mode_map"]


def test_query_mode_map_keys_equal_vocabulary_surfaces():
    assert set(DATA["query_mode_map"]) == set(DATA["vocabulary"]["embedding_query_mode_surfaces"]["values"])


def test_thinking_budget_below_probe_cap():
    cap = DATA["probe_graph"]["probe_request_max_tokens"]
    budgets = [v for spec in DATA["injection"].values() for vals in spec.get("values", {}).values()
               for k, v in vals.items() if k.endswith("budget_tokens")]
    assert budgets and all(b < cap for b in budgets)


def test_presets_all_valid_and_readme_present():
    vocab = DATA["vocabulary"]
    names = sorted(p.stem for p in (PKG_ROOT / "presets").glob("*.yaml"))
    assert {"RAG", "SQLGEN", "JUDGE", "RETRIEVAL"} <= set(names)
    for name in names:
        preset = catalog.load_preset(name, PKG_ROOT)
        (pb.validate_embedding_preset if "modality" in preset else pb.validate_llm_preset)(preset, vocab)
    assert (PKG_ROOT / "presets" / "README.md").exists() and (PKG_ROOT / "model_rules" / "README.md").exists()


def test_no_profiles_dir_hardcoding_in_code():
    """발행 위치는 configs profiles_dir 하나 — 코드에 프로젝트별 경로를 박지 않는다."""
    for path in PY_FILES:
        text = path.read_text(encoding="utf-8")
        assert "envs" + "/agents" not in text, path


def test_no_cross_module_copies_of_slot_expansion():
    """칸 펼침·경로 펼침은 injection 한 곳 — 다른 모듈에 같은 해석 함수가 복제되지 않는다."""
    pkg = PKG_ROOT / "langchain_model_preset_gen"
    pattern = re.compile(r"^def (expand_slot|expand_paths|normalize_slot|deep_merge|to_http_body)\(", re.M)
    owners = sorted({p.name for p in pkg.glob("*.py") if pattern.search(p.read_text(encoding="utf-8"))})
    assert owners == ["injection.py"], owners


def test_profile_skeleton_reference_parses():
    text = (PKG_ROOT / "references" / "profile-skeleton.yaml").read_text(encoding="utf-8")
    body = yaml.safe_load(text)
    assert body is None or isinstance(body, dict)
