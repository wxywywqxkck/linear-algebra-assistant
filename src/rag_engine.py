"""
RAG 核心引擎

整合检索器、生成器、引用追踪器和会话管理器，
对外提供统一的 chat(query, session_id) 接口。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

from loguru import logger

from .citation import CitationTracker
from .generator import GeneratedAnswer, ResponseGenerator
from .retriever import RetrievalResult, VectorRetriever
from .session_manager import SessionManager

# LangSmith @traceable 装饰器（条件启用）
try:
    if os.getenv("LANGSMITH_TRACING_V2", "").lower() in ("true", "1", "yes"):
        from langsmith import traceable
    else:
        traceable = lambda **kw: (lambda f: f)  # no-op
except Exception:
    traceable = lambda **kw: (lambda f: f)  # no-op


@dataclass
class ChatResponse:
    """Chat API 统一响应"""
    answer: str
    sources: List[dict]
    session_id: str
    latency_ms: int
    retrieval_count: int
    llm_model: str


class RAGEngine:
    """
    RAG 核心引擎

    使用方式:
        engine = RAGEngine()
        resp = engine.chat("什么是矩阵的秩？")
    """

    def __init__(
        self,
        retriever: Optional[VectorRetriever] = None,
        generator: Optional[ResponseGenerator] = None,
        session_manager: Optional[SessionManager] = None,
    ):
        self.retriever = retriever or VectorRetriever()
        self.generator = generator or ResponseGenerator()
        self.session_manager = session_manager or SessionManager()
        self.citation_tracker = CitationTracker()

        # 可选：直接用 LlamaIndex 的 query_engine，效果更好
        self._llama_query_engine = None

    def _get_llama_query_engine(self):
        """使用 LlamaIndex 原生 query_engine（效果更好）"""
        if self._llama_query_engine is not None:
            return self._llama_query_engine

        from .config import get_config
        cfg = get_config()

        self.retriever._load_index()  # noqa: SLF001
        llm = self.generator._get_llm()  # noqa: SLF001

        self._llama_query_engine = self.retriever._index.as_query_engine(  # noqa: SLF001
            llm=llm,
            similarity_top_k=cfg.retrieval_top_k,
            response_mode="compact",
        )
        return self._llama_query_engine

    @traceable(run_type="chain")
    def chat(
        self,
        query: str,
        session_id: Optional[str] = None,
        use_llama_query_engine: bool = False,
        api_key: Optional[str] = None,
    ) -> ChatResponse:
        """
        执行一次完整的 RAG 问答。

        Args:
            query: 用户问题
            session_id: 会话 ID（可选，用于多轮对话）
            use_llama_query_engine: 是否使用 LlamaIndex 原生 query_engine
            api_key: 可选的临时 API Key（覆盖 .env 默认值，不写盘）

        Returns:
            ChatResponse
        """
        start = time.time()

        # 1. 获取或创建会话
        session = None
        conversation_history: List[dict] = []
        if session_id:
            session = self.session_manager.get_session(session_id)
            if session is None:
                session = self.session_manager.create_session()
                logger.info(f"🆕 创建新会话: {session.session_id}")
            conversation_history = session.history
            session_id = session.session_id
        else:
            session = self.session_manager.create_session()
            session_id = session.session_id

        # 2. 检索
        retrieval_results: List[RetrievalResult] = []
        sources_dicts: List[dict] = []
        if not use_llama_query_engine:
            retrieval_results = self.retriever.retrieve(query)
            sources_dicts = [
                {"text": r.text, "title": r.title, "source": r.source, "score": r.score}
                for r in retrieval_results
            ]

        # 3. 生成回答
        if use_llama_query_engine:
            answer_text, sources_dicts = self._chat_via_llama(query)
        else:
            generated = self.generator.generate(
                query, sources_dicts, conversation_history, api_key=api_key
            )
            answer_text = generated.answer

        # 4. 附加引用
        answer_text = self.citation_tracker.add_citations(answer_text, sources_dicts)

        # 5. 保存会话
        self.session_manager.append_message(session_id, "user", query)
        self.session_manager.append_message(session_id, "assistant", answer_text)

        latency_ms = int((time.time() - start) * 1000)

        return ChatResponse(
            answer=answer_text,
            sources=sources_dicts,
            session_id=session_id,
            latency_ms=latency_ms,
            retrieval_count=len(sources_dicts),
            llm_model=self.generator.model,
        )

    def _chat_via_llama(self, query: str) -> tuple[str, List[dict]]:
        """使用 LlamaIndex 原生 query_engine"""
        query_engine = self._get_llama_query_engine()
        response = query_engine.query(query)

        answer_text = str(response)

        # 提取 source nodes
        sources: List[dict] = []
        source_nodes = getattr(response, "source_nodes", None) or []
        for node in source_nodes:
            text = getattr(node, "node", node).get_text() if hasattr(node, "node") else getattr(node, "text", str(node))
            metadata = getattr(getattr(node, "node", node), "metadata", {}) or {}
            sources.append({
                "text": text,
                "title": metadata.get("title", metadata.get("source", "未知")),
                "source": metadata.get("source", ""),
            })

        return answer_text, sources

    def health_check(self) -> dict:
        """健康检查"""
        return {
            "status": "ok",
            "retriever_initialized": self.retriever.is_initialized(),
            "document_count": self.retriever.count_documents(),
            "llm_model": self.generator.model,
        }
