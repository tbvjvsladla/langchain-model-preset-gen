"""workflow_probe — configs `workflow_probe` ("모듈:함수") 로더의 fail-fast 규격 검사와 preflight 연결."""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from conftest import PKG_ROOT, make_skill
from langchain_model_preset_gen import catalog
from langchain_model_preset_gen import preflight as p
from langchain_model_preset_gen import publish as pb
from langchain_model_preset_gen import resolver as rs
from langchain_model_preset_gen import workflow_probe as wp

DATA = catalog.load_data(PKG_ROOT)


def _module(root, name, body):
    (root / f"{name}.py").write_text(body, encoding="utf-8")


@pytest.mark.parametrize("spec", ["no_colon", "a:b:c", ":func", "mod:"])
def test_workflow_probe_format_violation_raises(tmp_path, spec):
    with pytest.raises(ValueError, match="형식"):
        wp.load_factory(spec, tmp_path)


def test_workflow_probe_import_failure_raises_naturally(tmp_path):
    with pytest.raises(ModuleNotFoundError):
        wp.load_factory("wp_no_such_module_x:make", tmp_path)


def test_workflow_probe_missing_function_raises(tmp_path):
    _module(tmp_path, "wp_mod_nofunc", "x = 1\n")
    with pytest.raises(AttributeError):
        wp.load_factory("wp_mod_nofunc:make", tmp_path)


def test_workflow_probe_non_runnable_return_raises():
    with pytest.raises(TypeError, match="Runnable"):
        wp.build_target(lambda model: "not a runnable", object(), embedding=False)


def test_workflow_probe_embedding_missing_methods_raises():
    class Half:
        def embed_query(self, t):
            return [1.0]
    with pytest.raises(TypeError, match="aembed_query"):
        wp.build_target(lambda emb: Half(), object(), embedding=True)


def test_workflow_probe_loads_from_project_root(tmp_path):
    _module(tmp_path, "wp_mod_ok", "from langchain_core.runnables import RunnableLambda\n"
                                   "def make(model):\n    return RunnableLambda(lambda q: model.invoke(q))\n")
    factory = wp.load_factory("wp_mod_ok:make", tmp_path)
    assert isinstance(wp.build_target(factory, object(), embedding=False), RunnableLambda)


def test_workflow_probe_configured_null_is_none(skill):
    assert wp.configured_factory(skill) is None


def test_workflow_probe_preflight_runs_six_modes_through_workflow(tmp_path, monkeypatch):
    proj = tmp_path / "proj"
    skill = make_skill(proj, workflow_probe="wp_mod_chain:make")
    _module(proj, "wp_mod_chain", "from langchain_core.runnables import RunnableLambda\n"
                                  "def make(model):\n    return RunnableLambda(lambda q: q) | model\n")
    fake = GenericFakeChatModel(messages=iter([AIMessage(content="ok")] * 12))
    monkeypatch.setattr(rs, "build_chat_model", lambda spec, key, client_kwargs=None: fake)
    spec = rs.resolve(pb.build_llm_profile("granite-4.2-8b", "RAG", "site_vllm", skill))
    out = p.check_runnable(spec, None, DATA, wp.configured_factory(skill))
    assert out["status"] == "ok" and out["target"] == "RunnableSequence"
    assert list(out["runnable"]) == ["invoke", "ainvoke", "stream", "astream", "batch", "abatch"]
