# host별 설치 경로

폴더 전체를 아래 경로에 복사한다. `SKILL.md`만 떼어 쓰지 않는다.

| host | project-local 설치 경로 | 상태 |
|---|---|---|
| Claude Code | `<root>/.claude/skills/langchain-model-preset-gen/` | 검증 |
| Codex | `<root>/.agents/skills/langchain-model-preset-gen/` | 미검증 |
| Hermes | `<root>/.agents/skills/langchain-model-preset-gen/` | 미검증 |

`<root>` = 프로젝트 루트. 미검증 경로는 각 host의 공식 문서 기준이며 이 패키지로 실행해 보지 않았다.

## 상대경로가 풀리는 조건
`configs/config.yaml`의 상대경로(`profiles_dir`·`env_file_path`)는 스킬 폴더에서 **세 단계 위**를 프로젝트 루트로 보고 푼다.
- `<root>/<x>/skills/langchain-model-preset-gen` 형태(위 표의 세 경로)면 `<root>`가 되어 상대경로가 그대로 맞는다.
- 그 밖의 깊이에 설치하면 세 단계 위가 프로젝트 루트가 아니다. 이때는 `configs/config.yaml`에 **절대경로**를 쓴다. 스킬은 설치 위치를 추측하거나 마커 파일을 찾지 않는다.
- `workflow_probe` 모듈도 같은 "세 단계 위"를 import 기준으로 쓴다.

## 사용자 단위(user-level) 설치
홈 폴더 아래 공용 스킬 폴더(예: `~/.claude/skills/langchain-model-preset-gen`)에 설치하면 세 단계 위가 홈 폴더다. 또 `configs/config.yaml`·`presets/`·`model_rules/`가 스킬 폴더 안에 하나뿐이라 모든 프로젝트가 같은 설정을 공유한다. 프로젝트마다 다른 `profiles_dir`·`env_file_path`를 쓰려면 project-local 설치를 권한다. user-level로 쓴다면 config 경로를 모두 절대경로로 적는다.
