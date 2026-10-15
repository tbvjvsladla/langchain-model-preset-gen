"""settings — 설치자 설정 `configs/config.yaml` 로드와 API 키 색인.

config.yaml 다섯 키는 모두 필수(누락 = 자연 KeyError)이고 모르는 키는 ValueError 로 거부한다(오타가 조용히 무시되지 않게).
상대경로는 '설치 프로젝트 루트' 기준이다. 프로젝트 루트 = 스킬 폴더(<root>/.claude/skills/<name>)에서 세 단계 위.
다른 위치에 설치했다면 절대경로를 쓴다 — 설치 위치를 추측하거나 마커 파일을 찾지 않는다.

API 키 (`resolve_api_key`):
  1. 프로세스 환경변수에 비어 있지 않은 값이 있으면 그것을 쓴다(docker --env-file·conda env vars 와 공존).
  2. 없으면 env_file_path 바로 아래 `*.env` 를 파일명 순서로 읽는다. 하위 폴더·`.env.example`·`*.bak` 는 대상이 아니고
     심볼릭 링크는 ValueError. 파싱만 한다(python-dotenv `dotenv_values`) — os.environ·셸은 건드리지 않는다.
  3. 요청한 키 하나만 채택한다. 빈 값은 무시하고, 같은 값의 중복은 허용하고, 파일끼리 값이 다르면 ValueError.
  4. 어디에도 없으면 KeyError. 오류 메시지에는 키 이름과 파일 경로만 쓰고 값은 쓰지 않는다.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from dotenv import dotenv_values

from .catalog import SKILL_DIR, load_yaml

CONFIG_RELPATH = Path("configs") / "config.yaml"
CONFIG_KEYS = ("profiles_dir", "env_file_path", "providers", "overridable", "workflow_probe")
# override 가능 키의 전체 어휘 (LLM). 임베딩은 이 중 timeout_seconds·max_retries 만 쓴다.
OVERRIDABLE_VOCABULARY = ("temperature", "top_p", "top_k", "effort", "timeout_seconds", "max_retries",
                          "response_deadline_seconds", "max_tokens")
# <root>/.claude/skills/<name> → parents[2] = <root>
PROJECT_ROOT_PARENT_INDEX = 2
ENV_FILE_SUFFIX = ".env"


def project_root(skill_dir: Path = SKILL_DIR) -> Path:
    return Path(skill_dir).resolve().parents[PROJECT_ROOT_PARENT_INDEX]


def resolve_config_path(value: str, skill_dir: Path = SKILL_DIR) -> Path:
    """config 의 경로 값 → 절대경로. 절대경로는 그대로, 상대경로는 프로젝트 루트 기준."""
    path = Path(value)
    return path if path.is_absolute() else project_root(skill_dir) / path


def load_settings(skill_dir: Path = SKILL_DIR) -> dict:
    """configs/config.yaml → {profiles_dir: Path, env_file_path: Path, providers, overridable, workflow_probe}."""
    raw = load_yaml(Path(skill_dir) / CONFIG_RELPATH)
    unknown = sorted(set(raw) - set(CONFIG_KEYS))
    if unknown:
        raise ValueError(f"configs/config.yaml 에 모르는 키 {unknown} — 허용: {list(CONFIG_KEYS)}")
    overridable = raw["overridable"]
    bad = sorted(set(overridable) - set(OVERRIDABLE_VOCABULARY))
    if bad:
        raise ValueError(f"configs/config.yaml overridable 에 모르는 키 {bad} — 허용: {list(OVERRIDABLE_VOCABULARY)}")
    providers = raw["providers"]
    workflow_probe = raw["workflow_probe"]
    return {
        "profiles_dir": resolve_config_path(raw["profiles_dir"], skill_dir),
        "env_file_path": resolve_config_path(raw["env_file_path"], skill_dir),
        "providers": {} if providers is None else dict(providers),
        "overridable": dict(overridable),
        "workflow_probe": workflow_probe,
    }


def env_files(env_dir: Path) -> list[Path]:
    """env_dir 바로 아래 `*.env` 파일(파일명 순). 심볼릭 링크는 ValueError. 폴더가 없으면 FileNotFoundError."""
    candidates = sorted(p for p in Path(env_dir).iterdir() if p.name.endswith(ENV_FILE_SUFFIX))
    links = [str(p) for p in candidates if p.is_symlink()]
    if links:
        raise ValueError(f"키 파일 심볼릭 링크는 읽지 않는다: {links}")
    return [p for p in candidates if p.is_file()]


def _checked_key(name: str, value: str, source: str) -> str:
    """키 값에 공백·제어문자가 있으면 거부한다(CRLF 키 파일·escape 된 개행 등). 값은 메시지에 싣지 않는다 —
    그런 값은 HTTP 헤더 예외 메시지에 원문으로 실려 출력에 새기 때문이다."""
    if any(ch.isspace() or not ch.isprintable() for ch in value):
        raise ValueError(f"API 키 {name} 값에 공백·제어문자가 있다 ({source}) — 키 값을 고친다")
    return value


def resolve_api_key(name: str | None, env_dir: Path, environ: Mapping[str, str] | None = None) -> str | None:
    """api_key_env 이름 → 키 값. name=None(무인증 자체 호스팅) → None."""
    if name is None:
        return None
    environ = os.environ if environ is None else environ
    if environ.get(name):
        return _checked_key(name, environ[name], "프로세스 환경변수")
    found: dict[Path, str] = {}
    for path in env_files(env_dir):
        value = dotenv_values(path).get(name)
        if value:
            found[path] = _checked_key(name, value, str(path))
    if len(set(found.values())) > 1:
        raise ValueError(f"{name} 값이 키 파일마다 다르다: {[str(p) for p in found]} — 하나로 맞춘다")
    if not found:
        raise KeyError(f"API 키 {name} 를 환경변수와 {env_dir}/*.env 어디에서도 찾지 못했다")
    return next(iter(found.values()))
