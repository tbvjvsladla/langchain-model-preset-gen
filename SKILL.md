---
name: langchain-model-preset-gen
description: >
  LangChain/LangGraph Agent가 쓸 LLM·임베딩 프로필 yaml을 용도 preset과 6단계 추론 강도(none·low·medium·high·xhigh·max)로
  발행하고, 실제 서빙 엔드포인트에서 실측(끄기·켜기 + LangChain 6 호출 방식 / 임베딩 embed 4종)해 probe.status를 기록한다.
  Agent는 공개 해석기(load_chat_model·load_embeddings)로 프로필 + override를 LangChain 객체로 바꾼다.
  Use when: 새 LLM·임베딩 프로필 추가·재발행, preset 적용·작성, serving provider 추가·변경, 모델 규칙(model_rules) 등록
  (OpenRouter 모델은 derive_mapping 초안), "probe 돌려줘"·실측 재판정, Agent 쪽 override 점검, 기존 workflow의 LLM 로드 코드를
  해석기 호출로 교정할 때.
---

# langchain-model-preset-gen

`S` = 이 스킬 폴더(기본 `.claude/skills/langchain-model-preset-gen`). 명령은 **프로젝트 루트**에서 `python $S/scripts/<x>.py`로 실행한다.
스크립트가 결정론 부분(검증·발행·실측·기록)을 맡고, 너는 지시 해석·모델카드 조사·사용자 질문·진단 해석을 맡는다.
스크립트가 거부한 값을 가까운 값이나 추측으로 우회하지 않는다.

## 0. 시작 조건
- `$S/configs/config.yaml`이 없으면 멈추고 사용자에게 `configs/config.example.yaml`을 복사해 채우도록 요청한다. 값을 대신 지어내지 않는다.

## 1. 지시 파싱
자연어 지시에서 다음을 뽑는다: `model_name` · `preset` · `serving_provider` · `kind`(llm | embedding) · (임베딩만) `dimensions`.
- `model_name`은 엔드포인트 모델 목록의 id **그대로**다. 실측 connect·v1은 정확 일치로 본다.
  - OpenRouter = `vendor/slug`(예 `moonshotai/kimi-k2.6`). Anthropic = 목록에 실린 dated id. Z.ai = 소문자. MiniMax = 대소문자 그대로.
- `serving_provider` = `data/providers.yaml`의 공식 이름(openai·anthropic·google·deepseek·openrouter·zai·minimax·kimi) 또는 `configs/config.yaml` `providers`의 사이트 이름.
- `preset` = `presets/`의 파일명 stem. 임베딩은 임베딩 preset(예 `RETRIEVAL`)만 쓴다.
- `dimensions`는 축소(MRL 절단)할 때만 준다. native와 같은 값은 발행이 거부한다.

## 2. 빠진 필드는 묻는다
- 빠지거나 모호한 필드는 **plain text로 한 번에 하나씩** 묻는다. 구조화 선택 팝업을 쓰지 않는다. 기본값을 추측해 채우지 않는다.
- 새 사이트 엔드포인트(로컬 vLLM 등): 사용자에게 `base_url`·`api_key_env`(무인증이면 null)를 확인받아 `configs/config.yaml` `providers`에 추가한다. 공식 이름을 다시 쓰면 발행이 거부한다.
- 새 preset이 필요하면 `presets/README.md` 형식으로 만들되, 값은 사용자 확인 후 쓴다.
- 임베딩 모델이 `data/embedding_capability.yaml`에 없으면 발행이 `embedding_capability 미등록`으로 거부한다. `data/`는 배포자 소유(업그레이드 시 덮임)이므로 사용자에게 보고하고 멈춘다.

## 3. 모델 규칙 확인 (LLM)
먼저 `--dry-run`으로 규칙 매칭을 확인한다.
```bash
python $S/scripts/publish.py --model-name <id> --preset <NAME> --provider <serving_provider> --kind llm --dry-run
```
- `model_rules 미등록`(KeyError)이면 규칙 파일을 만든다. 같은 길이 match 두 개가 맞으면 `모호`(ValueError) — 한쪽 match를 좁힌다.
- **OpenRouter 모델**: 초안을 만든 뒤 모델카드와 대조한다(끌 수 있는가 · 실제 단계 수 · 샘플링 수용 · 사고 종료 표지).
  ```bash
  python $S/scripts/derive_mapping.py --model-id <vendor/slug> --fetch          # 또는 --models-json <캐시한 /models JSON>
  ```
  초안은 파일을 쓰지 않는다. 확인·수정한 내용을 `model_rules/`에 파일 1개로 저장하고 `basis`의 "Agent 확인 필요" 문구를 실제 근거로 바꾼다.
- **그 밖(로컬·벤더 직결)**: 벤더 모델카드·API 문서를 조사해 같은 형식으로 쓴다. 작성법·배치 규칙은 `model_rules/README.md`.
- 같은 모델도 경로마다 규칙이 다르다: OpenRouter `vendor/slug` = `or_reasoning`, 로컬 vLLM = `ct_enable_thinking`, 벤더 직결 = 벤더 방식.
- 규칙을 고쳐도 이미 발행된 프로필은 바뀌지 않는다. 고친 규칙을 반영하려면 재발행한다(§4).

## 4. 발행
```bash
python $S/scripts/publish.py --model-name <id> --preset <NAME> --provider <serving_provider> --kind llm|embedding [--dimensions N]
```
- 위치 = `configs/config.yaml` `profiles_dir`(또는 `--out-dir`). 파일명 = `{model_name 정리값}_{preset}_{serving_provider}.yaml`.
- 같은 이름이 있으면 `FileExistsError`로 거부한다. 재발행은 사용자에게 확인받은 뒤 옛 파일을 지우고 다시 발행한다(손 편집은 사라진다).
- 발행 직후 `probe.status: probe_error` + `method: static`(임베딩 `static_v5`)은 **미측정**이지 실패가 아니다.

## 5. 실측과 기록
실측은 실제 API를 호출한다. 유료 provider면 실행 전에 사용자에게 알린다.
```bash
python $S/scripts/preflight.py --profile <프로필> --checks v1,v3,v4-1,pg,runnable --write-back   # LLM
python $S/scripts/preflight.py --profile <프로필> --checks v1,v3,v4-1,runnable --write-back      # 임베딩 (pg는 LLM 전용)
```
- 종료코드 0 = FAIL 없음. v1 WARN(목록에 없음)은 FAIL이 아니지만 pg connect가 `T_model_missing`으로 끝난다 — model_name부터 확인한다.
- `configs/config.yaml` `workflow_probe`가 설정돼 있으면 runnable은 그 workflow를 통해 돈다(LLM·임베딩 같은 함수).
- 검사가 예외로 멈추면 파일에 쓰지 않는다. 예외 메시지를 그대로 사용자에게 보고한다.

| probe.status | 의미 | 할 일 |
|---|---|---|
| `ok` | LLM: pg 종착 PASS **그리고** runnable 6종 전부 `ok`(`unsupported`도 ok가 아니다). 임베딩: runnable에 `probe_error` 없음 | 등록 가능 |
| `probe_error` + method `static`·`static_v5` | 미측정 | 위 명령으로 실측 |
| `probe_error` + method `probe_graph_live`·`live_v5` | 실측 실패 | 종착 진단을 사용자에게 보고 |
| `forced` | 사용자가 "강행"이라고 **명시**한 경우에만 네가 손으로 `probe.status: forced`로 고친다 | 강행 사유·시점을 보고에 남긴다. 다음 write-back이 덮어쓴다 |

pg 종착(`probe.terminal`) → 교정 후보 (자동 수정 없음 — 교정은 사용자 승인 후):

| 종착 | 교정 후보 |
|---|---|
| `T_model_missing` | model_name·base_url 확인 |
| `T_harmful_off` · `T_off_ignored` | none 칸을 omit 또는 최저 단계로 재매핑 (끄기가 사고를 본문으로 흘리거나 무시됨) |
| `T_rejected` | injection id·값 확인 |
| `T_on_suppressed` | 그 칸을 omit(기본 사고)으로 재매핑하거나 다른 단계 값 검토 |
| `T_budget_exhausted` | `limits.max_tokens` 상향 또는 effort 하향 |
| `T_no_response_in_timeout` | `limits.timeout_seconds` 상향 |
| `T_reasoning_leak` | 서버의 reasoning parser(vLLM `--reasoning-parser`) 확인 |
| `T_sampling_rejected` | 규칙 sampling 수용표에서 그 키를 `rejected`로 정정 |
| `U_undeterminable` | 429·5xx·전송 실패 — 모델 사실이 아니다. 다시 실측 |

- 6칸·sampling 같은 **모델 사실**은 `model_rules/` 파일을 고치고 재발행한다. `limits` 같은 **값**은 preset 또는 프로필 값을 고친다.
- 켜기(사고 관측)는 확률적이다. 재실측 결과가 최신 판정이다.

## 6. 소비 Agent에 등록
- 등록 조건 = `probe.status ∈ {ok, forced}`.
- 프로필 경로와 override를 소비 Agent의 config에 적는다. 저장 형식은 그 Agent의 몫이다. override 규칙은 `references/override-guide.md`.
- override가 있는 LLM 행은 그 실효값으로 실측 그래프를 통과해야 한다(종료코드 0 = PASS, 10 = probe_error 결과, 그 밖 = 도구 실패):
  ```bash
  python $S/scripts/probe.py --profile <프로필> --override '{"effort": "high"}'
  ```
- Agent 코드는 `load_chat_model(profile, override)` / `load_embeddings(profile, override, query_instruction=...)`로 객체를 만든다.
  `query_mode_surface: instruct_prefix` 임베딩은 `query_instruction`이 없으면 로드가 거부된다.

## 7. (선택) 기존 workflow 교정
프로젝트에 ChatOpenAI·ChatAnthropic·ChatGoogleGenerativeAI·Embeddings를 직접 생성하는 코드가 있으면 `references/correction-standard.md`대로 교정을 **제안**한다. 사용자 승인 후에만 수정하고, 교정 후 실측까지 마친다.

## 참고
| 파일 | 읽는 때 |
|---|---|
| `model_rules/README.md` | 규칙 작성·6칸 배치 |
| `presets/README.md` | preset 작성 |
| `references/override-guide.md` | override 값 우선순위·잠금 |
| `references/correction-standard.md` | 기존 workflow 교정 |
| `references/env-setup.md` | 키 파일·venv·conda·docker |
| `references/profile-skeleton.yaml` | 프로필 필드 의미 |
