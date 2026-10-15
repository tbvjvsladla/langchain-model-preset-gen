# model_rules/ — 모델 규칙

모델 규칙은 "이 모델에서 6단계 추론 강도가 실제로 어떤 인자가 되는가"와 "샘플링 인자를 받는가"를 적은 파일이다. **파일 1개 = 규칙 1건**이고 이 폴더의 모든 `*.yaml`을 읽는다(README 제외). 발행할 때 모델에 맞는 규칙이 없으면 거부한다 — 추측으로 채우지 않는다.

이 폴더는 설치자 소유다. 패키지를 올릴 때 덮어쓰지 않는다. 규칙을 고쳐도 이미 발행된 프로필은 바뀌지 않으므로, 반영하려면 그 프로필을 지우고 재발행한 뒤 다시 실측한다.

## 파일 형식
```yaml
match: deepseek-v4-flash               # 모델명 접두어 (대소문자 무시)
provider_class: openai_compatible      # openai_compatible | openai | anthropic | google
injection: xb_thinking_effort          # data/injection.yaml 키
mapping:                               # 6칸 전부 필수
  none: {action: disable}
  low: {action: set, value: high}
  medium: {action: set, value: high}
  high: {action: set, value: high}
  xhigh: {action: set, value: max}
  max: {action: set, value: max}
end_marker: null                       # 사고 종료 표지 (예 "</think>"). null = 기본 표지 목록
sampling:
  temperature:
    by_effort: {none: accepted, default: ignored}
  top_p:
    by_effort: {none: accepted, default: ignored}
  top_k: rejected
basis: '근거 한 줄 — 모델카드·벤더 API 문서·OpenRouter 메타·실측'
```
- 키 집합은 정확히 `match · provider_class · injection · mapping · end_marker · sampling · basis`다. 빠지거나 더 있으면 거부한다.
- 파일명은 사람이 찾기 위한 것이고 매칭에 쓰이지 않는다. 관례: `<match의 '/'를 '_'로>__<provider_class>.yaml`.
- 이 폴더의 `*.yaml`은 전부 규칙으로 읽힌다. 백업·초안은 다른 확장자로 두거나 폴더 밖에 둔다.

## 칸(slot) 형식
| 형식 | 뜻 |
|---|---|
| `{action: set, value: v}` | `v`를 주입한다. `v`는 그 injection의 `values` 표 키이거나, `value_path`에 그대로 들어갈 값이다 |
| `{action: disable}` | 그 injection의 끄기 인자를 넣는다. 끄기 인자가 없는 방식(예 `gemini_level`)에는 쓸 수 없다 |
| `{action: omit}` | 아무것도 넣지 않는다(벤더 기본 동작) |
| `low: low` (축약형) | `{action: set, value: low}`와 같다 |

## sampling 수용표
키는 `temperature`·`top_p`·`top_k` 세 개 모두. 값은 상태 하나 또는 effort 단계별 분기다.

| 상태 | 뜻 |
|---|---|
| `accepted` | 받고 반영한다 — 이 상태이고 값이 null이 아닐 때만 요청에 싣는다 |
| `rejected` | 보내면 4xx이거나 인자가 없다 |
| `ignored` | 받지만 효과가 없다 |

분기형 `{by_effort: {<단계>: 상태, default: 상태}}` — 키는 6단계 일부와 `default`(나머지 전부). `default`가 없으면 6단계를 모두 적어야 한다. 예: 추론을 끌 때만 샘플링을 받는 모델은 `{by_effort: {none: accepted, default: ignored}}`.

## 매칭 — 가장 긴 접두어
1. 프로필의 `provider_class`와 같은 규칙만 후보다.
2. 그중 `model_name.lower()`가 `match.lower()`로 시작하는 규칙을 고른다.
3. `match`가 **가장 긴** 규칙 하나를 쓴다. 파일 순서는 상관없다.
4. 같은 길이로 둘 이상 맞으면 `모호`(ValueError), 하나도 없으면 `model_rules 미등록`(KeyError)으로 거부한다.

`match`는 모델명 단위로 좁게 쓴다(예: `gemini-3.6-flash`가 `gemini-3`보다 낫다). 변형 이름(양자화·날짜 접미)은 접두어로 함께 잡힌다.

## 6칸 배치 규칙
- **단계가 적은 모델은 다대일로 배치한다.** 예: 단계가 `high`·`max`뿐인 모델은 `low·medium·high → high`, `xhigh·max → max`. 단계 없이 켜고 끄기만 되는 모델은 `none: disable`, 나머지 5칸 `enable`.
- **none 칸**: 끌 수 있으면 `disable`. 끌 수 없으면 지원하는 최저 단계 값(예: Gemini `minimal`), 그것도 없으면 `omit`.
- **끄기가 해로우면 omit**: 끄기 인자를 무시하고 사고를 본문으로 흘리는 모델은 none을 `omit`으로 둔다(실측 `T_harmful_off`·`T_off_ignored`).
- **켜기가 사고를 끄면 omit**: 특정 단계 값을 주면 오히려 사고가 사라지는 모델은 그 칸을 `omit`으로 둔다(실측 `T_on_suppressed`).
- **손잡이가 없는 모델**(비추론 모델, 끌 수도 단계도 없는 모델)은 `injection: none`과 6칸 전부 `omit`.
- 강도의 "깊이"는 실측하지 않는다. 6칸 배치는 모델카드를 근거로 한 판단이며, 그 근거를 `basis`에 적는다.

## 경로마다 규칙이 다르다
같은 모델도 어떤 경로로 부르느냐에 따라 주입 방식이 다르다. 셋 다 `provider_class`가 `openai_compatible`일 수 있으므로 `match`로 구분한다.

| 경로 | match 예 | injection |
|---|---|---|
| OpenRouter | `z-ai/glm-5.2` (`vendor/slug`) | `or_reasoning` |
| 로컬 vLLM | `qwen3-4b` (서빙 모델명) | `ct_enable_thinking` |
| 벤더 직결 | `glm-5.2` (벤더 API 모델명) | 벤더 방식(예 `xb_thinking_effort`) |

벤더 직결 provider_class별 방식: `openai` = `openai_native`, `anthropic` = `anthropic_effort`·`anthropic_adaptive_effort`·`anthropic_thinking_budget` 중 모델에 맞는 것, `google` = `gemini_level`. 각 방식의 인자는 `data/injection.yaml` 주석에 있다.

## 초안 만들기와 검증
- **OpenRouter 모델**은 메타데이터로 초안을 만든다. 파일을 쓰지 않고 내용만 출력한다.
  ```bash
  python <스킬>/scripts/derive_mapping.py --model-id <vendor/slug> --fetch           # 또는 --models-json <캐시 JSON>
  ```
  초안 규칙: reasoning 메타 없음 → 6칸 omit / 끌 수 없고 단계 없음 → 6칸 omit / 토글형 → none disable + 나머지 `enable` / 단계형 → 같은 단계, 없으면 바로 위 단계(단 `max`로는 올리지 않음), 위가 없으면 바로 아래. sampling은 `supported_parameters`에 있으면 `accepted`, 없으면 `rejected`.
  초안을 모델카드와 대조해 고친 뒤 이 폴더에 저장하고, `basis`를 실제 근거로 바꾼다.
- **그 밖의 모델**은 벤더 모델카드·API 문서를 조사해 직접 쓴다.
- 쓴 뒤에는 발행(`publish.py --dry-run`)으로 매칭과 형식을 확인하고, 발행한 프로필을 `preflight.py --checks v4-1,pg`로 실측한다. 종착이 `PASS`가 아니면 진단(`data/probe_graph.yaml` `diagnoses`)대로 규칙을 고친다.

## 지울 때 주의
이미 발행한 프로필의 `preflight` v4-1·write-back은 지금의 `presets/`·`model_rules/`로 발행 형식을 다시 만들어 비교한다. 그 프로필이 쓴 preset이나 규칙 파일을 지우면 그 프로필의 정적 검사·실측 기록이 실패한다. 해석기(`resolve`·`load_*`)는 영향이 없다.
