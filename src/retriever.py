"""
向量检索器

封装 LlamaIndex + Chroma 的向量检索功能。
硬编码知识库名称为 "linear_algebra"，不可动态切换。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from loguru import logger

# LangSmith @traceable 装饰器（条件启用）
try:
    if os.getenv("LANGSMITH_TRACING_V2", "").lower() in ("true", "1", "yes"):
        from langsmith import traceable
    else:
        traceable = lambda **kw: (lambda f: f)
except Exception:
    traceable = lambda **kw: (lambda f: f)


# ---------------------------------------------------------------------------
# 硬编码常量
# ---------------------------------------------------------------------------

COLLECTION_NAME = "linear_algebra"


@dataclass
class RetrievalResult:
    """单条检索结果"""
    text: str
    score: float
    metadata: dict = field(default_factory=dict)

    @property
    def title(self) -> str:
        return self.metadata.get("title", "未知")

    @property
    def source(self) -> str:
        return self.metadata.get("source", self.metadata.get("file_path", ""))


class VectorRetriever:
    """向量检索器 - 基于 LlamaIndex + Chroma（单库 fixed）"""

    def __init__(self, persist_dir: str | None = None):
        from .config import get_config
        cfg = get_config()

        self.persist_dir = persist_dir or cfg.absolute_chroma_dir
        self.top_k = cfg.retrieval_top_k
        self._index = None
        self._query_engine = None

    def _load_index(self):
        """从 Chroma 加载已有索引"""
        if self._index is not None:
            return

        from .config import init_settings
        init_settings()

        from llama_index.core import StorageContext, load_index_from_storage
        from llama_index.vector_stores.chroma import ChromaVectorStore
        import chromadb

        logger.info(f"📂 加载向量索引: {self.persist_dir} (collection={COLLECTION_NAME})")

        chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        chroma_collection = chroma_client.get_collection(COLLECTION_NAME)

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=self.persist_dir,
        )

        self._index = load_index_from_storage(storage_context)
        logger.info("✅ 向量索引加载完成")

    @traceable(run_type="retriever")
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """根据查询检索相关文档（固定 linear_algebra 库）"""
        self._load_index()

        retriever = self._index.as_retriever(similarity_top_k=top_k or self.top_k)
        nodes = retriever.retrieve(query)

        results = []
        for node in nodes:
            score = getattr(node, "score", None) or 0.0
            results.append(RetrievalResult(
                text=node.node.text,
                score=float(score),
                metadata=node.node.metadata or {},
            ))

        logger.info(f"🔍 检索完成: query='{query[:30]}...', results={len(results)}")
        return results

    # ------------------------------------------------------------------
    # 健康检查辅助方法
    # ------------------------------------------------------------------

    def is_initialized(self) -> bool:
        """向量索引是否已加载"""
        try:
            self._load_index()
            return self._index is not None
        except Exception:
            return False

    def count_documents(self) -> int:
        """库中的文档块数量"""
        try:
            self._load_index()
            return len(self._index.docstore.docs)
        except Exception:
            return 0
