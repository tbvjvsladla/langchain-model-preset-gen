# 실행 환경별 키 설정 — venv · conda · pip · docker

스킬은 uv를 전제하지 않는다. 어느 환경이든 `python <스킬>/scripts/<x>.py`로 실행한다.
API 키는 `configs/config.yaml`의 `env_file_path` 폴더에 `*.env` 파일로 두거나 프로세스 환경변수로 넣는다.

## 1. 키를 찾는 순서 (`settings.resolve_api_key`)
프로필의 `api_key_env`가 null이면 키 없이 호출한다(무인증 자체 호스팅). 이름이 있으면:

1. **프로세스 환경변수**에 그 이름으로 비어 있지 않은 값이 있으면 그것을 쓴다. 파일은 보지 않는다.
2. 없으면 `env_file_path` 폴더 **바로 아래**의 `*.env` 파일을 파일명 순서로 읽는다.
   - `.env`라는 이름의 파일도 대상이다(`*.env`에 맞는다).
   - 하위 폴더, `.env.example`, `*.bak`처럼 `.env`로 끝나지 않는 파일은 읽지 않는다.
   - `*.env` 중 심볼릭 링크가 하나라도 있으면 `ValueError`로 멈춘다. 실제 파일을 둔다.
   - 폴더가 없으면 `FileNotFoundError`다.
3. **요청한 키 하나만** 채택한다. 파일의 다른 키는 쓰지 않는다.
   - 빈 값(`KEY=`)은 없는 것으로 본다.
   - 여러 파일에 같은 값이 있으면 괜찮다.
   - 파일끼리 값이 다르면 `ValueError`다. 메시지에는 파일 경로만 나오고 값은 나오지 않는다. 하나로 맞춘다.
4. 어디에도 없으면 `KeyError`다.

파일은 `python-dotenv`의 `dotenv_values`로 **파싱만** 한다. 셸로 실행하지 않으므로 `$(...)` 같은 치환은 일어나지 않고, `os.environ`도 바뀌지 않는다. `export KEY=...` 형식도 파싱된다.

키 값은 `resolve()` 결과·발행 프로필·실측 출력에 나오지 않는다.

## 2. 키 파일 만들기 (공통)
```bash
mkdir -p envs                                  # env_file_path 와 같은 폴더
cat > envs/providers.env <<'EOF'
OPENAI_API_KEY=<키>
OPENROUTER_API_KEY=<키>
GOOGLE_API_KEY=<키>
EOF
chmod 600 envs/providers.env
```
- 키 이름은 프로필의 `api_key_env`와 같아야 한다. 공식 provider의 이름은 `data/providers.yaml`에, 사이트 provider는 `configs/config.yaml` `providers`에 있다.
- 키 파일을 git에 올리지 않는다(`.gitignore`에 `envs/*.env` 등).

## 3. venv
```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r .claude/skills/langchain-model-preset-gen/requirements.txt
pip install "langchain-openai>=1.0,<2.0"       # 쓰는 provider 패키지만
python .claude/skills/langchain-model-preset-gen/scripts/preflight.py --profile envs/agents/<프로필>.yaml --checks v1,v3,v4-1
```
키는 2절의 파일로 둔다.

## 4. conda (미검증)
```bash
conda create -n lmpg python=3.12
conda activate lmpg
pip install -r .claude/skills/langchain-model-preset-gen/requirements.txt
conda env config vars set OPENAI_API_KEY=<키>  # 선택 — 환경변수로 두면 파일보다 우선한다
conda activate lmpg                             # env vars 를 반영하려면 다시 활성화
```
conda env vars를 쓰지 않으면 2절의 파일로 둔다. 이 경로는 실행 검증을 하지 않았다.

## 5. pip (가상환경 없이)
```bash
python3 -m pip install --user -r .claude/skills/langchain-model-preset-gen/requirements.txt
```
배포판에 따라 시스템 Python이 pip 설치를 막는다(externally-managed). 그때는 venv를 쓴다.

## 6. docker
기본 이미지에 의존성을 설치하고 **프로젝트 루트**를 마운트한다. `configs/config.yaml`의 상대경로는 스킬 폴더에서 세 단계 위(= 마운트한 프로젝트 루트) 기준으로 풀리므로, 스킬 폴더·`profiles_dir`·`env_file_path`가 같은 루트 아래 원래 구조대로 있어야 한다.

```bash
docker run --rm -it --network host \
  -v "$PWD":/work -w /work \
  python:3.12-slim bash -c '
    S=.claude/skills/langchain-model-preset-gen
    pip install -q -r $S/requirements.txt "langchain-openai>=1.0,<2.0" &&
    python $S/scripts/preflight.py --profile envs/agents/<프로필>.yaml --checks v1,v3,v4-1,pg,runnable'
```
- `--network host`: 같은 LAN의 자체 호스팅 엔드포인트(`http://<host>:8001/v1` 등)에 닿게 한다. 환경에 따라 host 네트워크가 LAN에 닿지 않을 수 있으니 v1·v3로 먼저 확인한다.
- 키 주입은 둘 중 하나다.
  - 마운트한 `env_file_path` 폴더의 `*.env` 파일을 그대로 쓴다(위 예시).
  - `--env-file envs/providers.env`로 컨테이너 환경변수에 넣는다. 환경변수가 파일보다 우선한다.
- 이미지에 COPY해서 쓸 때도 프로젝트 루트 아래 같은 구조를 유지하거나, `configs/config.yaml`에 절대경로를 쓴다.
- `pip install -r $S/requirements-dev.txt`를 더하면 컨테이너 안에서 `python -m pytest -q $S/tests`도 돌릴 수 있다.
