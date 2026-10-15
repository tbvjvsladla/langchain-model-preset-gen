"""settings — configs/config.yaml 계약(profiles_dir·env_file_path·overridable …)과 *.env 색인."""

from __future__ import annotations

import os

import pytest
import yaml

from conftest import make_skill
from langchain_model_preset_gen import settings as st

KEY = "SITE_TEST_API_KEY"


def _config(skill):
    return yaml.safe_load((skill / "configs" / "config.yaml").read_text(encoding="utf-8"))


def _rewrite(skill, body):
    (skill / "configs" / "config.yaml").write_text(yaml.safe_dump(body), encoding="utf-8")


# ── config.yaml ──────────────────────────────────────────────────────────

def test_profiles_dir_and_env_file_path_relative_to_project_root(tmp_path):
    skill = make_skill(tmp_path / "proj", profiles_dir="conf/profiles_x", env_file_path="keys_x")
    cfg = st.load_settings(skill)
    assert cfg["profiles_dir"] == (tmp_path / "proj" / "conf" / "profiles_x").resolve()
    assert cfg["env_file_path"] == (tmp_path / "proj" / "keys_x").resolve()
    assert st.project_root(skill) == (tmp_path / "proj").resolve()


def test_profiles_dir_absolute_path_kept(tmp_path):
    skill = make_skill(tmp_path / "proj", profiles_dir=str(tmp_path / "abs_profiles"))
    assert st.load_settings(skill)["profiles_dir"] == tmp_path / "abs_profiles"


@pytest.mark.parametrize("key", ["profiles_dir", "env_file_path", "providers", "overridable", "workflow_probe"])
def test_config_missing_key_raises_keyerror(skill, key):
    body = _config(skill)
    del body[key]
    _rewrite(skill, body)
    with pytest.raises(KeyError):
        st.load_settings(skill)


def test_config_unknown_key_raises(skill):
    _rewrite(skill, {**_config(skill), "profile_dir": "typo"})
    with pytest.raises(ValueError, match="모르는 키"):
        st.load_settings(skill)


def test_overridable_unknown_key_raises(skill):
    body = _config(skill)
    body["overridable"]["temprature"] = None
    _rewrite(skill, body)
    with pytest.raises(ValueError, match="overridable"):
        st.load_settings(skill)


def test_overridable_defaults_and_null_providers_accepted(skill):
    body = _config(skill)
    body["providers"] = None
    body["overridable"] = {"effort": "high", "timeout_seconds": None}
    _rewrite(skill, body)
    cfg = st.load_settings(skill)
    assert cfg["providers"] == {} and cfg["overridable"] == {"effort": "high", "timeout_seconds": None}


def test_config_example_matches_contract():
    from conftest import PKG_ROOT
    body = yaml.safe_load((PKG_ROOT / "configs" / "config.example.yaml").read_text(encoding="utf-8"))
    assert list(body) == list(st.CONFIG_KEYS)
    assert set(body["overridable"]) == set(st.OVERRIDABLE_VOCABULARY)
    # 배포본에 config.yaml 이 없다는 것은 배포 검사(레이아웃) 몫이다 — 설치본에는 config.yaml 이 정당하게 있다.


# ── *.env 색인 ───────────────────────────────────────────────────────────

@pytest.fixture
def env_dir(tmp_path, monkeypatch):
    monkeypatch.delenv(KEY, raising=False)
    d = tmp_path / "envs"
    d.mkdir()
    return d


def test_env_file_index_reads_only_direct_star_env(env_dir):
    (env_dir / "b.env").write_text(f"{KEY}=from-b\n", encoding="utf-8")
    (env_dir / ".env.example").write_text(f"{KEY}=example\n", encoding="utf-8")
    (env_dir / "a.env.bak").write_text(f"{KEY}=bak\n", encoding="utf-8")
    (env_dir / "sub").mkdir()
    (env_dir / "sub" / "c.env").write_text(f"{KEY}=sub\n", encoding="utf-8")
    assert [p.name for p in st.env_files(env_dir)] == ["b.env"]
    assert st.resolve_api_key(KEY, env_dir) == "from-b"


def test_env_file_dotenv_named_file_included_in_name_order(env_dir):
    (env_dir / "z.env").write_text("OTHER=1\n", encoding="utf-8")
    (env_dir / ".env").write_text(f"{KEY}=dot\n", encoding="utf-8")
    assert [p.name for p in st.env_files(env_dir)] == [".env", "z.env"]
    assert st.resolve_api_key(KEY, env_dir) == "dot"


def test_env_file_symlink_rejected(env_dir, tmp_path):
    target = tmp_path / "real.env"
    target.write_text(f"{KEY}=x\n", encoding="utf-8")
    os.symlink(target, env_dir / "link.env")
    with pytest.raises(ValueError, match="심볼릭"):
        st.resolve_api_key(KEY, env_dir)


def test_env_file_only_requested_key_adopted_and_environ_untouched(env_dir):
    (env_dir / "k.env").write_text(f"{KEY}=v\nUNRELATED_SECRET=s\n", encoding="utf-8")
    assert st.resolve_api_key(KEY, env_dir) == "v"
    assert KEY not in os.environ and "UNRELATED_SECRET" not in os.environ


def test_env_file_process_environment_wins(env_dir, monkeypatch):
    (env_dir / "k.env").write_text(f"{KEY}=file\n", encoding="utf-8")
    monkeypatch.setenv(KEY, "process")
    assert st.resolve_api_key(KEY, env_dir) == "process"
    monkeypatch.setenv(KEY, "")
    assert st.resolve_api_key(KEY, env_dir) == "file", "빈 환경변수는 값이 아니다"


def test_env_file_conflicting_values_raise_without_leaking_value(env_dir):
    (env_dir / "a.env").write_text(f"{KEY}=secret-one\n", encoding="utf-8")
    (env_dir / "b.env").write_text(f"{KEY}=secret-two\n", encoding="utf-8")
    with pytest.raises(ValueError) as e:
        st.resolve_api_key(KEY, env_dir)
    assert "secret" not in str(e.value) and "a.env" in str(e.value)


def test_env_file_same_value_duplicates_and_empty_values(env_dir):
    (env_dir / "a.env").write_text(f"{KEY}=same\n", encoding="utf-8")
    (env_dir / "b.env").write_text(f"{KEY}=same\n", encoding="utf-8")
    (env_dir / "c.env").write_text(f"{KEY}=\n", encoding="utf-8")
    assert st.resolve_api_key(KEY, env_dir) == "same"


def test_env_file_missing_key_raises_keyerror(env_dir):
    (env_dir / "a.env").write_text("OTHER=1\n", encoding="utf-8")
    with pytest.raises(KeyError, match=KEY):
        st.resolve_api_key(KEY, env_dir)


def test_env_file_none_name_is_no_auth(env_dir):
    assert st.resolve_api_key(None, env_dir) is None


def test_env_file_export_syntax_parsed_not_executed(env_dir, tmp_path):
    marker = tmp_path / "executed"
    (env_dir / "a.env").write_text(f"export {KEY}=exported\nX=$(touch {marker})\n", encoding="utf-8")
    assert st.resolve_api_key(KEY, env_dir) == "exported"
    assert not marker.exists()


# ── 적대 리뷰 보강: 공백·제어문자가 섞인 키는 거부(값을 메시지에 싣지 않는다) ─────────────

@pytest.mark.parametrize("value", ["sk-SECRET\n", "sk-SECRET\r", " ", "sk SECRET", "\tsk-SECRET"])
def test_env_key_with_whitespace_or_control_rejected_from_environ(env_dir, monkeypatch, value):
    monkeypatch.setenv(KEY, value)
    with pytest.raises(ValueError) as e:
        st.resolve_api_key(KEY, env_dir)
    assert "SECRET" not in str(e.value) and KEY in str(e.value)


def test_env_key_with_escaped_newline_in_file_rejected(env_dir):
    (env_dir / "a.env").write_text(f'{KEY}="sk-SECRET\\n"\n', encoding="utf-8")
    with pytest.raises(ValueError) as e:
        st.resolve_api_key(KEY, env_dir)
    assert "SECRET" not in str(e.value) and "a.env" in str(e.value)
