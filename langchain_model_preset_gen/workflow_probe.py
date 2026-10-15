"""workflow_probe — 설치자의 workflow 진입점을 실측 대상으로 쓴다 (configs/config.yaml `workflow_probe`).

형식: "패키지.모듈:함수". 함수는 해석기가 만든 객체 하나를 받아 실측 대상을 돌려준다.
  LLM 프로필      : func(chat_model) -> Runnable        (문자열 입력을 받아야 한다 — 입력 = probe_graph.yaml prompts.runnable_check)
  임베딩 프로필   : func(embeddings) -> embed_query·embed_documents·aembed_query·aembed_documents 를 가진 객체
모듈은 프로젝트 루트(settings.project_root) 기준으로 import 한다 — 로더가 그 경로를 sys.path 맨 앞에 넣는다.

검사(결정론, fail-fast): 형식 위반(':' 가 정확히 1개가 아님) → ValueError, import 실패 → 자연 raise,
반환값 규격 위반 → TypeError. 끄기·켜기 판정은 여기서 하지 않는다(Runnable 출력으로는 사고를 관측할 수 없다 —
실측 그래프가 같은 해석기 본문으로 원문 HTTP 판정을 한다). 이 실측이 증명하는 것은 "그 workflow 를 통해 호출 방식이 동작한다"이다.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from pathlib import Path

from langchain_core.runnables import Runnable

from . import settings as st

SPEC_SEP = ":"
EMBEDDING_METHODS = ("embed_query", "embed_documents", "aembed_query", "aembed_documents")


def load_factory(spec: str, project_root: Path) -> Callable:
    if spec.count(SPEC_SEP) != 1 or not all(spec.split(SPEC_SEP)):
        raise ValueError(f"workflow_probe 형식 위반: {spec!r} — \"패키지.모듈:함수\"")
    module_name, func_name = spec.split(SPEC_SEP)
    root = str(project_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    return getattr(importlib.import_module(module_name), func_name)


def build_target(factory: Callable, model, *, embedding: bool):
    target = factory(model)
    if embedding:
        missing = [m for m in EMBEDDING_METHODS if not callable(getattr(target, m, None))]
        if missing:
            raise TypeError(f"workflow_probe 반환값에 임베딩 메서드 {missing} 가 없다 — {type(target).__name__}")
        return target
    if not isinstance(target, Runnable):
        raise TypeError(f"workflow_probe 반환값이 Runnable 이 아니다 — {type(target).__name__}")
    return target


def configured_factory(skill_dir: Path) -> Callable | None:
    """configs workflow_probe → 함수. null 이면 None(해석기 객체를 직접 실측)."""
    spec = st.load_settings(skill_dir)["workflow_probe"]
    return None if spec is None else load_factory(spec, st.project_root(skill_dir))
