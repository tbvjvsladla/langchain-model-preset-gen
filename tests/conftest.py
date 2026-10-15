"""공용 fixture — 임시 설치 트리 `<proj>/.claude/skills/langchain-model-preset-gen` 를 만든다.

네트워크·LLM 호출은 없다. data/·presets/·model_rules/ 는 배포본 정본을 복사하고, configs/config.yaml 과
키 폴더만 테스트가 정한다(사이트 provider 는 자리표시 host). 개별 테스트가 data·presets·rules 를 바꿔 쓸 수 있다.
provider 패키지(langchain-openai 등)가 설치되지 않은 환경(requirements*.txt 만 설치)에서도 전체가 돈다:
SDK 클래스는 가짜 모듈로, 발행 기록용 dist 버전은 고정 값으로 대체한다.
"""

from __future__ import annotations

import importlib.metadata as md
import shutil
import sys
import types
from pathlib import Path

import pytest
import yaml

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))

SKILL_NAME = "langchain-model-preset-gen"
SITE_BASE_URL = "http://vllm.test:8001/v1"
SITE_PROVIDERS = {
    "site_vllm": {"provider_class": "openai_compatible", "base_url": SITE_BASE_URL, "api_key_env": None},
}
FULL_OVERRIDABLE = {k: None for k in ("temperature", "top_p", "top_k", "effort", "timeout_seconds", "max_retries",
                                      "response_deadline_seconds", "max_tokens")}


def write_yaml(path: Path, body) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(body, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def make_skill(root: Path, *, providers: dict | None = None, overridable: dict | None = None,
               workflow_probe: str | None = None, profiles_dir: str = "profiles", env_file_path: str = "keys",
               rules: list[dict] | None = None, presets: dict | None = None, data: dict | None = None,
               config_extra: dict | None = None) -> Path:
    """root/.claude/skills/<name> 설치 트리. rules·presets·data(파일 키 → 내용)는 주면 그것으로 바꾼다."""
    skill = root / ".claude" / "skills" / SKILL_NAME
    for folder in ("data", "presets", "model_rules"):
        shutil.copytree(PKG_ROOT / folder, skill / folder)
    if rules is not None:
        shutil.rmtree(skill / "model_rules")
        (skill / "model_rules").mkdir()
        for i, rule in enumerate(rules):
            write_yaml(skill / "model_rules" / f"rule_{i:02d}.yaml", rule)
    if presets is not None:
        for name, body in presets.items():
            write_yaml(skill / "presets" / f"{name}.yaml", body)
    for key, body in (data or {}).items():
        write_yaml(skill / "data" / f"{key}.yaml", body)
    config = {"profiles_dir": profiles_dir, "env_file_path": env_file_path,
              "providers": SITE_PROVIDERS if providers is None else providers,
              "overridable": FULL_OVERRIDABLE if overridable is None else overridable,
              "workflow_probe": workflow_probe, **(config_extra or {})}
    write_yaml(skill / "configs" / "config.yaml", config)
    (root / env_file_path).mkdir(parents=True, exist_ok=True)
    return skill


@pytest.fixture
def skill(tmp_path) -> Path:
    return make_skill(tmp_path / "proj")


@pytest.fixture
def no_env_keys(monkeypatch):
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(name, raising=False)


# 발행 기록(probe.measured_with)용 provider dist 버전 — 테스트는 설치 여부와 무관하게 이 값을 본다.
PROVIDER_DIST_VERSIONS = {"langchain-openai": "1.3.4", "langchain-anthropic": "1.4.8", "langchain-google-genai": "4.2.7"}


@pytest.fixture(autouse=True)
def provider_dist_versions(monkeypatch):
    real = md.version
    monkeypatch.setattr(md, "version", lambda dist: PROVIDER_DIST_VERSIONS.get(dist) or real(dist))


@pytest.fixture
def sdk_mock(monkeypatch):
    """'모듈.클래스' 를 MagicMock 을 담은 가짜 모듈로 바꾸고 그 mock 을 돌려준다(provider 패키지 미설치에서도 동작)."""
    from unittest.mock import MagicMock

    def patch(target: str):
        module_name, attr = target.rsplit(".", 1)
        mock = MagicMock(name=target)
        fake = types.ModuleType(module_name)
        setattr(fake, attr, mock)
        monkeypatch.setitem(sys.modules, module_name, fake)
        return mock
    return patch
