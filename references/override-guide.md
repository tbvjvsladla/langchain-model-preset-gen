# override 운영 가이드 (Agent 쪽)

override는 Agent가 실행 시점에 프로필 값 일부를 바꾸는 dict다(예: `{"effort": "high", "temperature": 0.3}`).
해석기(`resolve`·`load_chat_model`·`load_embeddings`)가 override를 적용하고 잠금을 검사한다.
**override를 어디에 어떤 형식으로 저장할지(Agent config의 행, 요청 헤더 등)는 Agent의 몫이며 이 스킬의 범위 밖이다.** 해석기는 dict 하나만 받는다.

## 1. 값 우선순위
```text
preset 값  <  configs/config.yaml overridable 기본값  <  Agent override
              (발행 때 프로필에 구워짐)                  (해석기가 실행 때 적용)
```
- 발행 때 `overridable`의 값이 null이 아닌 키는 preset 값 대신 그 값이 프로필 `preset_values`·`limits`에 들어간다. 이후 config 기본값을 바꿔도 **이미 발행된 프로필은 바뀌지 않는다** — 반영하려면 재발행한다.
- override는 프로필 값 위에 얹힌다. override에 없는 키는 프로필 값 그대로다.

## 2. override할 수 있는 키
- 프로필의 `overridable` 목록에 있는 키만 받는다. 밖의 키는 `LockedKeyError`(`ValueError` 하위)로 거부된다.
- LLM 어휘: `temperature · top_p · top_k · effort · timeout_seconds · max_retries · response_deadline_seconds · max_tokens`.
- 임베딩 어휘: `timeout_seconds · max_retries`. 그 provider의 임베딩 SDK가 받지 않는 키는 발행 때 목록에서 빠진다(예: google 임베딩은 `max_retries` 없음).
- `effort`는 `none·low·medium·high·xhigh·max` 중 하나여야 한다.
- **sampling override는 실효 effort에서 수용표가 `accepted`일 때만 받는다.** 실효 effort = override의 `effort`가 있으면 그 값, 없으면 프로필 값.
  - 예: 수용표가 `temperature: {by_effort: {none: accepted, default: ignored}}`인 모델에 `{"effort": "high", "temperature": 0.3}`을 주면 거부된다. `{"effort": "none", "temperature": 0.3}`은 받는다.
  - 효과 없는 값을 받은 척하지 않기 위한 규칙이다. preset에 적힌 sampling 값이 그 effort에서 `accepted`가 아니면 오류가 아니라 **보내지 않을** 뿐이다.
- `response_deadline_seconds`는 SDK 인자가 아니다. 해석기는 값만 전달(`effective`·`sent`)하고, 지키는 일은 소비자 처리부가 맡는다.

## 3. 잠그기
특정 키를 Agent가 못 바꾸게 하려면 둘 중 하나를 한다.
1. `configs/config.yaml`의 `overridable`에서 그 키를 빼고 재발행한다(그 뒤 발행되는 모든 프로필에 적용).
2. 특정 프로필 파일의 `overridable` 목록에서 손으로 지운다. 목록은 어휘의 부분집합이면 되고(중복 금지), 정적 검사(`v4-1`)를 통과한다.

**항상 잠긴 키** — 어떤 설정으로도 override할 수 없다: `model_name` · `serving_provider` · `base_url` · `provider_class` · `api_key_env` · `reasoning_effort`(6칸 매핑) · `sampling`(수용표) · `probe`, 임베딩의 `supports` · `dimensions`.
- 모델·주소를 바꾸려면 다른 프로필을 쓴다.
- 6칸 매핑·수용표를 바꾸려면 `model_rules/`를 고치고 재발행한다.
- 임베딩 차원을 바꾸려면 `--dimensions`로 재발행하고 다시 실측한다. 실효 차원과 `probe.measured_at_dimensions`가 다르면 해석기가 로드를 거부한다.

생성자 인자를 `client_kwargs`로 넘길 때도 프로필이 관리하는 이름이면 `LockedKeyError`다. 그 값은 override로 바꾼다.

## 4. `effective`와 `sent`
`resolve(profile, override)`가 둘 다 돌려준다.

| 필드 | 뜻 |
|---|---|
| `effective` | 우선순위를 적용한 실효값 (요청한 값 그대로) |
| `sent` | 실제로 보낸 값. 실효 effort에서 `accepted`가 아닌 sampling 키는 `None` |

"적용한 값"을 로그나 응답에 남길 때는 `sent`를 쓴다. 실제 SDK 생성자 인자는 `sdk_kwargs`에 있다.

## 5. override 행의 실측
override는 실측된 프로필 기본값과 다른 인자를 보낸다. LLM override 행은 쓰기 전에 그 실효값으로 실측 그래프를 통과시킨다.
```bash
python <스킬>/scripts/probe.py --profile <프로필> --override '{"effort": "high"}'
```
종료코드 0 = PASS, 10 = probe_error(결과), 그 밖 = 도구 실패. 호출 방식까지 보려면 `preflight.py --checks pg,runnable --override '<JSON>'`(이때는 `--write-back` 불가 — 기록되는 값은 프로필 기본값의 실측이어야 한다).
