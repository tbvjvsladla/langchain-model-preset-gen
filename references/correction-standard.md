# 기존 workflow 교정 표준 — LLM·임베딩 로드부를 해석기 호출로 바꾸기

이미 LangChain/LangGraph workflow가 있고, 그 코드가 `ChatOpenAI`·`ChatAnthropic`·`ChatGoogleGenerativeAI`·`OpenAIEmbeddings`·`GoogleGenerativeAIEmbeddings`를 **직접** 생성한다면 이 절차로 교정한다. 목표는 해석 로직을 한 벌(이 스킬의 해석기)로 줄이는 것이다. 실측이 검증한 인자와 workflow가 보내는 인자가 같은 함수에서 나와야 실측 결과가 workflow에 대해서도 참이 된다.

교정은 사용자 승인 후에만 한다. 브랜치·프로젝트마다 따로 교정하고 다른 프로젝트의 교정 코드를 복사하지 않는다(공통점은 이 표준뿐이다).

## 1. 모델 생성 지점을 전부 찾는다
```bash
grep -rnE "Chat(OpenAI|Anthropic|GoogleGenerativeAI)\(|(OpenAI|GoogleGenerativeAI)Embeddings\(|init_chat_model\(|init_embeddings\(" <workflow 코드 폴더>
```
- 생성자 호출뿐 아니라 그 인자를 조립하는 코드(effort·thinking·`extra_body`·sampling 분기, provider별 if문)도 목록에 넣는다.
- 노드마다 다른 모델을 쓰면 노드별로 어느 프로필이 대응하는지 표로 적는다.

## 2. 생성 지점을 해석기 호출로 바꾼다
```python
from langchain_model_preset_gen import load_chat_model, load_embeddings, resolve

llm = load_chat_model(profile_path, override)                     # override = {} 또는 행별 값
emb = load_embeddings(profile_path, override, query_instruction=TASK_TEXT)
```
- **행별(요청별·설정 행별) override는 `override` 인자로만** 넘긴다. 생성 뒤에 `.bind(temperature=...)` 등으로 덮어쓰지 않는다 — 실측이 본 값과 달라진다.
- `query_instruction`은 workflow가 소유하는 검색 지시문이다. `query_mode_surface: instruct_prefix` 임베딩은 이 값이 없으면 로드가 거부된다. 다른 형식의 모델에는 넘겨도 쓰이지 않는다.
- **소비자 전용 생성자 인자**(예: `stream_usage`, `stream_chunk_timeout` — 그 SDK 클래스가 실제로 받는 인자여야 한다)는 `client_kwargs`로 넘긴다.
  ```python
  llm = load_chat_model(profile_path, override, client_kwargs={"stream_usage": True})
  ```
  프로필이 관리하는 이름(`model`·키·`base_url`·`max_tokens`·`timeout`·`max_retries`·`temperature`·`top_p`·`extra_body`·`reasoning_effort`·`effort`·`thinking`·`thinking_level` 등, 정확한 목록은 `resolve(...)["reserved_kwargs"]`)을 넣으면 `LockedKeyError`가 난다. 그 값은 override로 바꾼다.
- 로그·응답에 "적용한 값"을 남기는 코드는 `resolve(profile_path, override)["sent"]`를 쓴다. 그 effort에서 받지 않는 sampling은 `None`으로 나온다. 요청한 값 그대로는 `["effective"]`다.
- 기동 시 확인하는 `probe.status`도 `resolve(...)["probe"]["status"]`로 읽는다.

## 3. 스킬 폴더를 import 경로에 넣는다
- 스킬 폴더(`.../.claude/skills/langchain-model-preset-gen`)를 `sys.path`에 넣는다. 경로는 workflow config에 한 곳만 두고, 코드에 박지 않는다.
- docker: 스킬 폴더를 이미지에 `COPY`하거나 볼륨으로 마운트하고, 컨테이너 안의 같은 경로를 `sys.path`에 넣는다. `configs/config.yaml`의 상대경로가 풀리도록 프로젝트 루트 구조를 유지한다([env-setup.md](env-setup.md)).
- 스킬을 vendoring(코드 복사)하거나 pip 패키지로 따로 빌드하지 않는다.

## 4. 남은 사본을 지운다
교정 후 workflow 안에 다음이 남아 있으면 삭제한다.
- effort 칸 → 인자 변환(주입) 로직, `extra_body` 조립, provider별 sampling 분기.
- 임베딩 query/document 접두어 처리, L2 정규화 래퍼(해석기의 임베딩 객체가 항상 정규화한다).
- 스킬 `data/`·`model_rules/`·`presets/`를 직접 읽는 코드. 소비자는 프로필과 공개 API(`resolve`·`load_chat_model`·`load_embeddings`·`LockedKeyError`)만 쓴다.

**소비자 고유 규칙은 소비자에 남긴다.** 예: 응답 상한 `response_deadline_seconds`는 SDK 인자가 아니라 서버 처리부(예: 그래프 전체를 시간 제한으로 감싸는 코드)가 쓰는 값이다. 이 값을 지킬 수 없는 소비자라면, 값이 설정된 프로필을 거부하는 규칙은 그 소비자 코드에 둔다.

## 5. 기동 게이트
- 프로필마다 `probe.status ∈ {ok, forced}`가 아니면 기동을 거부한다.
- override가 있는 LLM 행은 그 실효값으로 실측 그래프를 한 번 더 통과해야 한다.
  ```bash
  python <스킬>/scripts/probe.py --profile <프로필> --override '<override JSON>'
  ```
  종료코드 0 = PASS, 10 = probe_error(실측 결과 — 기동 거부), 그 밖(1 = 예외, 2 = 인자 오류) = 도구 실패(결과로 취급하지 말고 원인 보고). 실측 그래프는 LLM 전용이다.

## 6. 교정 전후 동등성을 테스트로 고정한다
교정이 보내는 인자를 바꾸지 않았음을 결정론 테스트(LLM·네트워크 없음)로 핀한다.
- 교정 전 코드가 생성자에 넘기던 인자와 교정 후 인자를 같은 프로필·override로 비교한다.
  - 교정 후 인자 = `resolve(profile, override)["sdk_kwargs"]` + `client_kwargs`(API 키 제외).
  - 또는 SDK 클래스를 mock으로 바꾸고 `load_chat_model`이 넘긴 인자를 잡아 비교한다.
- override가 있는 행, effort가 `none`인 행, sampling이 effort 단계별로 갈리는 모델을 반드시 포함한다.
- 차이가 나면 어느 쪽이 맞는지 사용자에게 보고한다. 교정 전 코드가 틀렸던 것이면 그 사실을 기록하고 해석기 쪽을 따른다.

## 7. 교정 후 실측
1. `configs/config.yaml`의 `workflow_probe`에 workflow 진입점 함수를 지정한다.
   ```yaml
   workflow_probe: "app.probe_entry:make_runnable"     # 프로젝트 루트 기준 import 경로
   ```
   - LLM 프로필: `func(chat_model) -> Runnable`(문자열 입력을 받아야 한다).
   - 임베딩 프로필: `func(embeddings) -> embed_query·embed_documents·aembed_query·aembed_documents를 가진 객체`.
   - 한 설정을 LLM·임베딩 실측이 같이 쓴다. 두 종류를 함께 실측한다면 함수가 받은 객체 종류에 맞는 대상을 돌려줘야 한다.
2. 실측한다.
   ```bash
   python <스킬>/scripts/preflight.py --profile <프로필> --checks pg,runnable
   ```
   - `runnable`은 workflow를 통해 6 호출 방식(임베딩은 embed 4종)을 돌린다. 끄기·켜기 판정은 Runnable 출력으로 관측할 수 없어 같은 해석기 인자로 원문 HTTP 그래프(`pg`)가 판정한다.
   - 이 실측이 추가로 증명하는 것은 "그 workflow를 통해 호출 방식이 동작한다"이다.
3. 결과를 프로필에 기록하려면 `--write-back`을 붙인다(override 없이). 기록 후 `probe.status`가 `ok`인지 확인한다.
4. workflow 자체 테스트와 서버 기동 확인을 함께 실행한다.
