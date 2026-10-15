"""embeddings — 임베딩 프로필의 입력 형식과 정규화를 맡는 Embeddings 래퍼.

LangChain `Embeddings` 는 경로를 이미 나눈다: 적재 = embed_documents, 질의 = embed_query.
그래서 이 래퍼 한 곳이 query/document 입력 형식과 정규화를 맡으면 벡터스토어·리트리버 코드는 바뀌지 않는다.

1. 입력 형식 — 메커니즘(data/query_mode_map.yaml)별:
   text_prefix      : query = prefix_template.format(task=query_instruction, query=원문), document = 원문
   text_prefix_both : query = prefix_template.format(query=원문), document = document_template.format(text=원문)
   sdk_kwarg·null   : 원문 그대로 (sdk_kwarg 는 어댑터가 경로별 task_type 을 스스로 보낸다)
2. 정규화 — 양 경로 출력을 항상 L2 단위벡터로 만든다. 이미 단위벡터인 모델(normalization: builtin)에는 멱등이다.
   단위벡터에서는 L2 최근접 = 코사인 최근접이라 L2 인덱스가 그대로 코사인 검색기가 된다.
   영벡터는 0 나눗셈 ZeroDivisionError 로 자연 raise 한다.
정규화 전 raw 벡터가 필요하면(실측의 노름 측정) `inner` 를 직접 호출한다.
"""

from __future__ import annotations

import math

from langchain_core.embeddings import Embeddings

MECHANISM_TEXT_PREFIX = "text_prefix"
MECHANISM_TEXT_PREFIX_BOTH = "text_prefix_both"


def l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


class ProfileEmbeddings(Embeddings):
    """프로필 supports + 메커니즘 + workflow 지시 텍스트를 결합한 Embeddings."""

    def __init__(self, inner: Embeddings, supports: dict, mechanism: str | None, query_instruction: str | None) -> None:
        if mechanism == MECHANISM_TEXT_PREFIX and query_instruction is None:
            raise ValueError("instruct_prefix 임베딩은 query_instruction 이 필요하다 — "
                             "프리픽스 {task} 에 'None' 문자열이 들어가는 것을 막는다")
        self._inner = inner
        self._supports = supports
        self._mechanism = mechanism
        self._query_instruction = query_instruction

    @property
    def inner(self) -> Embeddings:
        return self._inner

    @property
    def mechanism(self) -> str | None:
        return self._mechanism

    def render_query(self, text: str) -> str:
        if self._mechanism == MECHANISM_TEXT_PREFIX:
            return self._supports["prefix_template"].format(task=self._query_instruction, query=text)
        if self._mechanism == MECHANISM_TEXT_PREFIX_BOTH:
            return self._supports["prefix_template"].format(query=text)
        return text

    def render_documents(self, texts: list[str]) -> list[str]:
        if self._mechanism == MECHANISM_TEXT_PREFIX_BOTH:
            template = self._supports["document_template"]          # 없으면 자연 KeyError — 양측 형식이 어긋난 인덱스 방지
            return [template.format(text=t) for t in texts]
        return texts

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [l2_normalize(v) for v in self._inner.embed_documents(self.render_documents(texts))]

    def embed_query(self, text: str) -> list[float]:
        return l2_normalize(self._inner.embed_query(self.render_query(text)))

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = await self._inner.aembed_documents(self.render_documents(texts))
        return [l2_normalize(v) for v in vecs]

    async def aembed_query(self, text: str) -> list[float]:
        return l2_normalize(await self._inner.aembed_query(self.render_query(text)))
