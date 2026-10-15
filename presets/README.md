# presets/ — 용도 preset

preset은 용도별 기본값 묶음이다. **파일 1개 = preset 1개**이고, `name` 필드는 파일명 stem과 같아야 한다(`RAG.yaml` → `name: RAG`). 다르면 발행이 거부한다.
발행할 때 `--preset <NAME>`으로 고르고, 그 값이 프로필 `preset_values`·`limits`에 펼쳐진다.

## 이 폴더는 설치자 소유다
- 들어 있는 preset(`BALANCED`·`CREATIVE`·`JUDGE`·`RAG`·`SQLGEN`·`RETRIEVAL`)은 출발점 예시다. 설치처의 용도에 맞게 고치거나 새로 만든다.
- 설치처마다 preset이 달라지는 것이 정상이다. 같은 이름의 preset이 프로젝트마다 다른 값을 가져도 된다.
- 패키지를 새 버전으로 올릴 때 이 폴더를 덮어쓰지 않는다. 새 배포본의 preset은 비교해 필요한 것만 옮긴다.
- preset을 고쳐도 이미 발행된 프로필은 바뀌지 않는다. 반영하려면 재발행한다.

## LLM preset
```yaml
# preset — RAG (근거 문서 기반 답변). 한 줄 설명을 주석으로 둔다.
name: RAG
intent: grounded
sampling: {temperature: 0.1, top_p: 0.9, top_k: 20}
effort: medium
limits: {timeout_seconds: 120, max_retries: 2, response_deadline_seconds: null, max_tokens: 8192}
```
| 키 | 값 |
|---|---|
| `name` | 파일명 stem |
| `intent` | `deterministic · grounded · balanced · creative` 중 하나 |
| `sampling` | `temperature`·`top_p`·`top_k` 세 키 모두. 값이 null이면 그 키를 보내지 않는다 |
| `effort` | `none · low · medium · high · xhigh · max` 중 하나. null은 안 된다 |
| `limits` | `timeout_seconds`·`max_retries`·`response_deadline_seconds`·`max_tokens` 네 키 모두 |

- sampling 값은 모델 규칙의 수용표가 그 effort에서 `accepted`일 때만 실제로 보낸다. 받지 않는 모델에서는 오류 없이 빠진다.
- `max_tokens`는 사고 모델에서 사고와 답을 합친 상한이다. 너무 작으면 실측이 `T_budget_exhausted`로 끝난다. Anthropic thinking 예산 방식 모델은 `max_tokens`가 `budget_tokens`보다 커야 한다.
- `response_deadline_seconds`는 SDK 인자가 아니라 소비자 처리부가 쓰는 응답 상한이다. 쓰지 않으면 null.

## 임베딩 preset
```yaml
# preset — RETRIEVAL (임베딩 검색).
name: RETRIEVAL
intent: retrieval
modality: text
limits: {timeout_seconds: 120, max_retries: 2}
```
| 키 | 값 |
|---|---|
| `name` | 파일명 stem |
| `intent` | `retrieval · similarity · classification · clustering` 중 하나 |
| `modality` | `text · image` 중 하나. 모델의 `input_modalities`에 있어야 로드된다 |
| `limits` | `timeout_seconds`·`max_retries` 두 키 모두 |

임베딩 preset에는 `sampling`·`effort`·`max_tokens`·`response_deadline_seconds`가 없다.

## 어휘
위 어휘의 정본은 `data/vocabulary.yaml`(`intent`·`effort_levels`·`sampling_keys`·`embedding_intent`·`embedding_modality`)이다. 발행 때 검사하며, 어휘 밖 값이나 키 누락은 거부한다.

## 값 우선순위
preset 값 위에 `configs/config.yaml` `overridable`의 null이 아닌 기본값이 발행 때 얹히고, 실행 때 Agent override가 그 위에 얹힌다. 자세한 규칙은 `references/override-guide.md`.

## 지울 때 주의
이미 발행한 프로필의 `preflight` v4-1·write-back은 지금의 `presets/`·`model_rules/`로 발행 형식을 다시 만들어 비교한다. 그 프로필이 쓴 preset이나 규칙 파일을 지우면 그 프로필의 정적 검사·실측 기록이 실패한다. 해석기(`resolve`·`load_*`)는 영향이 없다.
