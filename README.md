# langchain-model-preset-gen

> **LangChain/LangGraph 에이전트가 쓸 LLM·임베딩 프로필을 "용도 preset + 6단계 추론 강도"로 만들고, 그 프로필이 실제 서빙 환경에서 LangChain으로 동작하는지 실측하는 스킬입니다. 에이전트는 같은 패키지의 해석기로 프로필을 LangChain 모델 객체로 바꿉니다.**
>
> 배포 버전 0.1.0 · Python 3.12 (검증) · Claude Code 공식 지원 · MIT

## 해결하려는 문제 — 모델을 바꿀 때마다 설정을 다시 손으로 맞춘다

LangGraph 노드 하나가 LLM을 부를 때 정해야 하는 값은 생각보다 많습니다.

- **샘플링:** temperature·top_p·top_k를 받는 모델, 거절(4xx)하는 모델, 받는 척하고 무시하는 모델이 있습니다. **추론을 끌 때만 샘플링을 받는 모델**도 있습니다.
- **추론 강도:** 같은 "강하게 생각해"가 제조사마다 다른 인자입니다(OpenAI `reasoning_effort`, Anthropic `thinking`·`effort`, Google `thinking_level`, OpenRouter `reasoning`, 로컬 vLLM `chat_template_kwargs.enable_thinking`). 단계 수도 2단계부터 6단계까지 다릅니다.
- **provider:** 같은 모델이라도 벤더 직결·OpenRouter·로컬 서버마다 주소·키·SDK 클래스·인자 이름이 다릅니다.
- **용도:** SQL 생성, RAG 답변, 평가(judge)마다 알맞은 기본값이 다릅니다.

이 조합을 모델마다 손으로 맞추는 일은 노동집약적이고, 잘못 맞춰도 에러 없이 "조용히 무시"되는 경우가 많습니다.

## 큰 그림

| 축 | 하는 일 |
|---|---|
| **① 용도 preset** | 용도별 기본값(샘플링·추론 강도·시간·출력 상한)을 작은 YAML 하나로 둡니다. 설치자마다 고쳐 쓰는 것이 정상입니다. |
| **② 6단계 추론 강도** | 사용자가 보는 추론 강도를 GPT-6 Luna의 6단계 `none·low·medium·high·xhigh·max`로 통일합니다. 6칸이 모델마다 실제 어떤 인자가 되는지는 모델 규칙에 적습니다(예: DeepSeek의 high·max 2단계 → 6칸에 다대일 배치). 샘플링 수용 여부도 같은 규칙에 effort 단계별로 적습니다. |
| **③ 실측** | 프로필이 서빙 엔드포인트에서 동작하는지 실제 호출로 확인합니다. LLM은 **끄기·켜기 두 끝점**(끄면 사고 0, 켜면 사고 있음)과 **LangChain 6가지 호출 방식**(`invoke·ainvoke·stream·astream·batch·abatch`), 임베딩은 `embed_query·embed_documents·aembed_query·aembed_documents`와 차원·노름입니다. |
| **④ 해석기와 override 경계** | 프로필 + override → LangChain `ChatModel`/`Embeddings`. 실행 중 바꿔도 되는 값(`overridable`)과 잠긴 값(모델명·주소·6칸 매핑·샘플링 수용표)을 나누고, 잠긴 키를 바꾸려 하면 거부합니다. **실측과 에이전트가 같은 해석기를 씁니다.** |

```text
presets/ (용도)  +  model_rules/ (모델별 6칸·샘플링 수용표)  +  configs/config.yaml (설치처 설정)
                                │
                                ▼  scripts/publish.py
                      프로필 yaml 발행  (probe.status = 미측정)
                                │
                                ▼  scripts/preflight.py --write-back
                      실측 → probe.status = ok | probe_error 기록
                                │
                                ▼  load_chat_model / load_embeddings
              에이전트가 프로필 + override로 LangChain 객체 생성
```

강도의 "깊이"(low가 max보다 덜 생각하는지)와 샘플링의 "효과"는 실측하지 않습니다. 실측은 인자가 받아지는지, 끄기·켜기가 작동하는지, LangChain 호출이 도는지만 봅니다.

### 지원 대상

| provider_class | LLM 클래스 | 임베딩 클래스 | 대표 경로 |
|---|---|---|---|
| `openai_compatible` | `ChatOpenAI` (base_url 지정) | `OpenAIEmbeddings` | OpenRouter · DeepSeek · Z.ai · MiniMax · Kimi 직결, 로컬 vLLM 등 OpenAI 호환 서버 |
| `openai` | `ChatOpenAI` | `OpenAIEmbeddings` | OpenAI 직결 |
| `anthropic` | `ChatAnthropic` | **없음** (Anthropic에는 임베딩 API가 없습니다) | Anthropic 직결 |
| `google` | `ChatGoogleGenerativeAI` | `GoogleGenerativeAIEmbeddings` | Gemini API 직결 |

- 공식 벤더 엔드포인트(openai·anthropic·google·deepseek·openrouter·zai·minimax·kimi)는 `data/providers.yaml`에 내장돼 있습니다. 사이트 전용 주소(로컬 vLLM 등)만 `configs/config.yaml`에 둡니다.
- 임베딩은 `data/embedding_capability.yaml`에 등록된 모델(입력 형식·native 차원)만 발행할 수 있습니다.
- 등록된 LLM 모델 규칙은 `model_rules/`에 있습니다. 없는 모델은 규칙 파일 1개를 추가하면 됩니다.

## 폴더 구성 — 누가 무엇을 고치나

| 경로 | 내용 | 누가 고치나 |
|---|---|---|
| `SKILL.md` | 에이전트가 따르는 절차 | 배포자 |
| `langchain_model_preset_gen/` | 해석기·발행·실측 코드 (import 가능한 패키지) | 배포자 |
| `scripts/` | 얇은 CLI 진입점 (`publish.py`·`preflight.py`·`probe.py`·`derive_mapping.py`) | 배포자 |
| `data/` | 어휘·주입 방식 표·provider_class 표·공식 엔드포인트·임베딩 사실표·실측 전이표 | 배포자 |
| `references/` | 교정 표준·override 가이드·환경별 키 설정·host별 설치 경로·프로필 골격 | 배포자 |
| `tests/` | 결정론 회귀 테스트 (LLM·네트워크 없음) | 배포자 |
| `presets/` | 용도 preset (파일 1개 = preset 1개) + README | **설치자** |
| `model_rules/` | 모델 규칙 (파일 1개 = 규칙 1건) + README | **설치자** |
| `configs/config.yaml` | 설치처 설정 (`config.example.yaml`을 복사해 만듦) | **설치자** |

패키지를 새 버전으로 올릴 때는 `presets/`·`model_rules/`·`configs/config.yaml`을 덮어쓰지 않습니다. 새 배포본의 preset·규칙은 비교해 필요한 파일만 옮깁니다. `data/`를 고치면 다음 업그레이드에서 덮인다는 점에 유의하세요.

## 설치

기본 설치 위치는 **Claude Code project-local** `.claude/skills/langchain-model-preset-gen/`입니다. Codex·Hermes용 경로는 [references/hosts.md](references/hosts.md)에 있습니다(미검증).

1. **폴더 전체**를 복사합니다. `SKILL.md`만 떼어 쓰지 않습니다.
   ```bash
   mkdir -p .claude/skills
   cp -R /path/to/langchain-model-preset-gen .claude/skills/langchain-model-preset-gen
   ```
2. 설치처 설정을 만듭니다.
   ```bash
   cd .claude/skills/langchain-model-preset-gen
   cp configs/config.example.yaml configs/config.yaml
   ```
3. 의존성을 설치합니다. uv를 전제하지 않습니다(venv·conda·pip·docker 모두 가능 — [references/env-setup.md](references/env-setup.md)).
   ```bash
   pip install -r requirements.txt                    # langchain-core · langgraph · python-dotenv
   pip install "langchain-openai>=1.0,<2.0"           # openai · openai_compatible 을 쓸 때
   pip install "langchain-anthropic>=1.0,<2.0"        # anthropic 을 쓸 때
   pip install "langchain-google-genai>=4.0,<5.0"     # google 을 쓸 때
   pip install -r requirements-dev.txt                # 스킬 자체 테스트(pytest)를 돌릴 때
   ```
   provider 패키지는 쓰는 것만 설치합니다. 버전 구간은 `data/provider_classes.yaml`의 `requires_range`와 같습니다. 설치본이 구간 밖이면 임베딩 발행·기록이 거부됩니다. `python-dotenv`는 LangChain 외의 유일한 의존성으로, `*.env` 키 파일을 읽는 데만 씁니다.
4. API 키 파일을 `env_file_path` 폴더에 둡니다(아래 키 읽기 규칙).

### `configs/config.yaml` — 다섯 키 모두 필수

```yaml
profiles_dir: envs/agents          # 프로필 발행 위치
env_file_path: envs                # 이 폴더 바로 아래 *.env 에서 API 키를 찾는다
providers:                         # 사이트 전용 serving_provider. 없으면 {}
  my_vllm_8001:
    provider_class: openai_compatible
    base_url: "http://<host>:8001/v1"
    api_key_env: null              # null = 무인증 자체 호스팅
overridable:                       # Agent가 override할 수 있는 키와 키별 기본값
  temperature: null                # null = preset 값 사용
  effort: null
  timeout_seconds: null
  # ... top_p · top_k · max_retries · response_deadline_seconds · max_tokens
workflow_probe: null               # 선택 — "패키지.모듈:함수"
```

| 키 | 뜻 |
|---|---|
| `profiles_dir` | `publish.py`가 프로필을 쓰는 폴더입니다. |
| `env_file_path` | API 키 `*.env` 파일이 있는 폴더입니다. |
| `providers` | 사이트 전용 엔드포인트입니다. 각 항목은 `provider_class`·`base_url`·`api_key_env` 세 키를 가집니다. 공식 이름(`openai` 등)을 다시 쓰면 거부합니다. |
| `overridable` | 여기 적힌 키만 Agent가 override할 수 있고, 목록에서 뺀 키는 잠깁니다. 값이 null이 아니면 발행 때 preset 값 대신 그 값이 프로필에 들어갑니다. 쓸 수 있는 키는 `temperature·top_p·top_k·effort·timeout_seconds·max_retries·response_deadline_seconds·max_tokens`입니다. 자세한 운영은 [references/override-guide.md](references/override-guide.md). |
| `workflow_probe` | 설치자의 workflow를 실측 대상으로 씁니다. 형식은 `"패키지.모듈:함수"`이고, 함수는 해석기가 만든 객체 하나(LLM이면 ChatModel, 임베딩이면 Embeddings)를 받아 실측 대상을 돌려줍니다. LLM은 문자열 입력을 받는 `Runnable`, 임베딩은 embed 4종 메서드를 가진 객체여야 합니다. null이면 해석기 객체를 직접 실측합니다. |

- 상대경로는 **프로젝트 루트** 기준입니다. 프로젝트 루트 = 스킬 폴더에서 세 단계 위(`<root>/.claude/skills/langchain-model-preset-gen` → `<root>`). 다른 깊이에 설치했다면 절대경로를 씁니다.
- 키가 빠지면 `KeyError`, 모르는 키가 있으면 `ValueError`로 멈춥니다(오타가 조용히 무시되지 않게).

### 키 읽기 규칙

1. 프로세스 환경변수에 비어 있지 않은 값이 있으면 그 값을 씁니다(docker `--env-file`·conda env vars와 공존).
2. 없으면 `env_file_path` **바로 아래** `*.env` 파일(`.env` 포함)을 파일명 순서로 읽습니다. 하위 폴더·`.env.example`·`*.bak`는 읽지 않고, 심볼릭 링크는 거부합니다.
3. 프로필의 `api_key_env` 키 **하나만** 채택합니다. 빈 값은 무시하고, 같은 값의 중복은 괜찮고, 파일끼리 값이 다르면 오류입니다(메시지에는 파일 경로만 나오고 값은 나오지 않습니다).
4. `python-dotenv`로 파싱만 합니다. 셸로 실행하지 않고 `os.environ`도 바꾸지 않습니다.
5. 어디에도 없으면 `KeyError`입니다. `api_key_env: null`이면 키 없이 호출합니다.

환경별 작성 예시는 [references/env-setup.md](references/env-setup.md)에 있습니다. 키 파일은 `chmod 600`으로 둡니다.

## 사용법

에이전트에게 "granite-4.2-8b를 로컬 vLLM에서 RAG용으로 프로필 만들어 줘"처럼 지시하면 `SKILL.md` 절차를 따릅니다. 직접 실행하는 명령은 다음과 같습니다(프로젝트 루트에서, `S=.claude/skills/langchain-model-preset-gen`).

```bash
# 1) 발행 — 미리보기(--dry-run) 후 실제 발행
python $S/scripts/publish.py --model-name granite-4.2-8b --preset RAG --provider my_vllm_8001 --kind llm --dry-run
python $S/scripts/publish.py --model-name granite-4.2-8b --preset RAG --provider my_vllm_8001 --kind llm
python $S/scripts/publish.py --model-name gemini-embedding-2 --preset RETRIEVAL --provider google --kind embedding

# 2) 실측 + 기록 (LLM / 임베딩)
python $S/scripts/preflight.py --profile envs/agents/granite-4.2-8b_RAG_my_vllm_8001.yaml \
  --checks v1,v3,v4-1,pg,runnable --write-back
python $S/scripts/preflight.py --profile envs/agents/gemini-embedding-2_RETRIEVAL_google.yaml \
  --checks v1,v3,v4-1,runnable --write-back

# 3) override 값으로 끄기·켜기만 다시 보기 (파일은 바꾸지 않음)
python $S/scripts/probe.py --profile envs/agents/granite-4.2-8b_RAG_my_vllm_8001.yaml --override '{"effort": "high"}'

# 4) OpenRouter 모델 규칙 초안 (파일을 쓰지 않음 — 확인 후 model_rules/ 에 저장)
python $S/scripts/derive_mapping.py --model-id deepseek/deepseek-v4.1-flash --fetch
```

- 파일명 = `{model_name 정리값}_{preset}_{serving_provider}.yaml`(`[A-Za-z0-9._-]` 밖 문자는 `_`). 같은 이름이 있으면 발행을 거부합니다. 재발행은 옛 파일을 지운 뒤 합니다.
- `preflight.py --checks` 기본값은 `v1,v3,v4-1`입니다. `--override`는 `--write-back`과 함께 쓸 수 없습니다.

### Python에서 쓰기

```python
import sys
sys.path.insert(0, "/abs/path/to/project/.claude/skills/langchain-model-preset-gen")   # 스킬 폴더

from langchain_model_preset_gen import LockedKeyError, load_chat_model, load_embeddings, resolve

P = "envs/agents/granite-4.2-8b_RAG_my_vllm_8001.yaml"
llm = load_chat_model(P, {"effort": "high"}, client_kwargs={"stream_usage": True})
spec = resolve(P, {"effort": "high"})
spec["probe"]["status"]     # 기동 게이트: ok | forced 여야 쓴다
spec["sent"]                # 실제로 보낸 값 (그 effort에서 받지 않는 sampling은 None)

emb = load_embeddings("envs/agents/qwen_qwen3-embedding-8b_RETRIEVAL_openrouter.yaml",
                      query_instruction="Given a question, retrieve passages that answer the question")
```

- `resolve(profile, override)` → 실효값(`effective`)·전송값(`sent`)·SDK 인자(`sdk_kwargs`)·`probe` 등을 담은 dict. API 키는 들어가지 않습니다.
- `load_chat_model` / `load_embeddings` → 키를 해소해 LangChain 객체를 만듭니다. `client_kwargs`에는 프로필이 관리하지 않는 생성자 인자만 넣습니다(프로필이 관리하는 이름이면 `LockedKeyError`).
- 임베딩 객체는 query·document 입력 형식(접두어 등)을 모델에 맞게 붙이고, 출력은 **항상 L2 정규화**합니다(이미 단위벡터인 모델에는 결과가 같습니다). `query_mode_surface: instruct_prefix` 모델은 `query_instruction`이 필수입니다.
- docker에서는 스킬 폴더를 이미지에 COPY하거나 볼륨으로 마운트해 같은 경로를 `sys.path`에 넣습니다.
- 기존 workflow를 이 해석기로 바꾸는 절차는 [references/correction-standard.md](references/correction-standard.md)에 있습니다.

## 실측이 보는 것

| 검사 | 내용 |
|---|---|
| `v1` | provider 모델 목록에 `model_name`이 정확히 있는가 (없으면 WARN) |
| `v2` | 무인증 자체 호스팅이면 모델 적재 여부 (`v1`을 요청하면 자동 추가) |
| `v3` | 미니 호출 1회 — 200 OK / 401·403 키 불량 / 402·429 잔액·쿼터 |
| `v4-1` | 정적 스키마 (API 0회) — 키 순서, provider 표와의 일치, 6칸·sampling·probe 블록 형태 |
| `pg` | 끄기·켜기 상태 그래프 (LLM 전용, 원문 HTTP) |
| `runnable` | 해석기 객체(또는 `workflow_probe` 대상)로 LLM 6종 / 임베딩 4종 + 노름·차원 |

**실측 그래프(`pg`)** 는 LangGraph로 만든 결정론 상태 그래프입니다.

```text
connect(모델 목록) → off_gate → off_probe(none 칸 주입) → on_path(실효 effort 칸 주입) ─→ sampling → PASS
                         │                                       └ 사고 0 → on_path_deep(심화 질문 1회)
                         └ none 칸이 omit·단계값이면 off_probe 를 건너뜀(끄기 불가가 모델 사실로 선언됨)
```

- 원문 HTTP로 보는 이유: LangChain은 vLLM 응답의 `reasoning` 필드를 버려 사고 여부를 관측할 수 없습니다. 요청 본문은 해석기가 만든 인자와 같은 결과에서 나오므로, **실측한 본문 = 에이전트가 보내는 인자**입니다.
- "사고함" = provider별 사고 텍스트 길이 또는 사고 토큰 수가 0보다 큼.
- 샘플링은 그 effort 단계에서 `accepted`로 선언된 키만 보내고 4xx 여부만 판정합니다.
- 노드당 1회 호출로 판정하고, 전이표(`data/probe_graph.yaml`)에 없는 결과는 즉시 멈춥니다. 429·5xx·전송 실패는 `U_undeterminable`(모델 사실 아님)입니다.
- `--write-back`은 LLM에서 `pg`와 `runnable`이 모두 필요하고, **pg PASS 그리고 6종 전부 ok**일 때만 `probe.status: ok`를 씁니다. 임베딩은 `runnable`만 필요하며 노름에서 `supports.normalization`(≈1 → builtin, 아니면 required)을 함께 기록합니다. 프로필의 나머지 내용·주석은 바이트 그대로 둡니다.
- `probe.status: forced`는 실측이 쓰지 않습니다. 사용자가 강행을 결정했을 때만 손으로 적는 값입니다.
- `probe.py` 종료코드: 0 = PASS, 10 = probe_error(결과 — stdout JSON), 그 밖 = 도구 실패.

## 안전 경계와 한계

- 규칙 없는 모델·어휘 밖 값·6칸 누락·잠긴 키 override·받지 않는 sampling의 override는 거부합니다. 가장 가까운 값으로 대체하지 않습니다.
- 실측은 실제 API를 호출합니다. 유료 provider는 비용이 듭니다(요청 1건 출력 상한 4096토큰).
- 켜기 판정은 확률적입니다. 적응형 모델이 질문을 쉽게 보고 사고를 건너뛰면 `T_on_suppressed`가 나올 수 있고, 재실측하면 다시 판정합니다.
- 실측 `ok`는 "설정이 전달되고 호출이 동작한다"는 뜻이며 답변 품질을 보증하지 않습니다.
- `langchain-anthropic`은 모델에 따라 `effort`와 함께 `thinking: adaptive`를 스스로 더할 수 있습니다. 실측 그래프(원문 HTTP)는 그 인자를 보내지 않으므로, SDK가 더한 부분은 `pg`의 검증 범위 밖입니다(`runnable`은 호출 성공만 확인합니다).
- 모델 규칙을 고쳐도 이미 발행된 프로필은 바뀌지 않습니다. 재발행해야 반영됩니다.
- 자체 테스트(pytest)는 LLM·네트워크 없는 결정론 회귀입니다. 통과해도 실측 성공은 아닙니다.
  ```bash
  python -m pytest -q tests/
  ```

## 이 배포본의 완료 판정 기준

- 기준 세트(로컬 vLLM LLM 4종 + `gemini-embedding-2`)가 **uv · venv+pip · docker** 세 환경에서 내장 실측을 통과한다.
- 참조 workflow 두 곳(RAG·NL2SQL)에서 이전 로더를 지우고 이 배포본을 설치 → 교정 표준대로 로더 교정 → `workflow_probe` 실측을 통과한다.
- 잠긴 키 override 거부 테스트가 통과하고, 금지 문자열 검사가 0건이다.

## 더 읽을 문서

| 파일 | 읽는 이유 |
|---|---|
| [SKILL.md](SKILL.md) | 에이전트 절차 |
| [references/correction-standard.md](references/correction-standard.md) | 기존 workflow의 LLM 로드 코드를 해석기로 바꾸는 표준과 교정 후 실측 |
| [references/override-guide.md](references/override-guide.md) | override 값 우선순위·잠금 |
| [references/env-setup.md](references/env-setup.md) | venv·conda·pip·docker별 키 설정 |
| [references/hosts.md](references/hosts.md) | Claude Code·Codex·Hermes 설치 경로 |
| [references/profile-skeleton.yaml](references/profile-skeleton.yaml) | 프로필 필드 설명 |
| [presets/README.md](presets/README.md) | preset 작성 |
| [model_rules/README.md](model_rules/README.md) | 모델 규칙 작성·6칸 배치 |

배포 무결성: `PACKAGE_MANIFEST.json`에 파일별 sha256·bytes·mode가 있다. 확인 `python scripts/package_manifest.py --check`.

라이선스: [MIT](LICENSE) · Copyright (c) AhnSangHun
