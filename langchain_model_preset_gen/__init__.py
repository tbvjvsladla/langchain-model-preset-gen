"""langchain-model-preset-gen — LangChain/LangGraph Agent 용 LLM·임베딩 프로필 발행·실측·해석.

공개 API (해석기 — 실측과 소비자가 같은 함수를 쓴다):
    resolve(profile, override)          → spec dict (실효값 effective · 실제 전송값 sent · SDK 인자)
    load_chat_model(profile, override)  → LangChain ChatModel
    load_embeddings(profile, override)  → ProfileEmbeddings (입력 형식·L2 정규화 포함)
    LockedKeyError                      → 프로필 overridable 밖의 키를 override 하면 난다

사용: 스킬 폴더를 sys.path 에 넣고 `from langchain_model_preset_gen import load_chat_model`.
"""

from .resolver import LockedKeyError, load_chat_model, load_embeddings, resolve

__all__ = ["LockedKeyError", "load_chat_model", "load_embeddings", "resolve"]
__version__ = "0.1.0"
