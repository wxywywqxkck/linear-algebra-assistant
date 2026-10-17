"""
FastAPI 后端

提供 REST API，支持聊天、会话管理。
知识库固定为 data/textbooks/linear-algebra.txt，前端不可操作。
"""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from loguru import logger

from .config import get_config
from .rag_engine import RAGEngine
from .session_manager import SessionManager


# ---------------------------------------------------------------------------
# 硬编码常量（不可变）
# ---------------------------------------------------------------------------

COLLECTION_NAME = "linear_algebra"
TEXTBOOKS_DIR = "data/textbooks"


# ---------------------------------------------------------------------------
# App 初始化
# ---------------------------------------------------------------------------

app = FastAPI(
    title="线性代数 RAG 智能助手 API",
    description="基于 RAG 的线性代数智能问答系统（单库 fixed collection）",
    version="0.2.0",
)

# CORS（Streamlit 在不同端口需要允许跨域）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局单例
cfg = get_config()
rag_engine = RAGEngine()
session_mgr = rag_engine.session_manager


@app.on_event("startup")
def on_startup():
    """应用启动时：显式初始化 LangSmith 等可选组件"""
    # LangSmith 显式初始化
    if os.getenv("LANGSMITH_TRACING_V2", "").lower() in ("true", "1", "yes"):
        try:
            import langsmith
            api_key = os.getenv("LANGSMITH_API_KEY", "")
            project = os.getenv("LANGSMITH_PROJECT", "default")
            if api_key:
                logger.info(f"🟢 LangSmith enabled (project={project})")
            else:
                logger.warning("⚠️ LANGSMITH_TRACING_V2=true 但缺少 LANGSMITH_API_KEY")
        except Exception as e:
            logger.warning(f"LangSmith 初始化失败（不影响主功能）: {e}")


# ---------------------------------------------------------------------------
# 请求/响应模型
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(default=None, description="会话 ID，为空则创建新会话")
    query: str = Field(..., description="用户问题", min_length=1, max_length=2000)
    api_key: Optional[str] = Field(
        default=None,
        description="可选：用户临时 API Key（覆盖后端默认值，不写盘）",
    )


class SourceItem(BaseModel):
    title: str = ""
    source: str = ""


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    session_id: str
    latency_ms: int
    retrieval_count: int
    llm_model: str


class SessionInfo(BaseModel):
    session_id: str
    created_at: float
    updated_at: float


class HealthResponse(BaseModel):
    status: str
    retriever_initialized: bool
    document_count: int
    llm_model: str


# ---------------------------------------------------------------------------
# 路由（固定单库）
# ---------------------------------------------------------------------------

@app.get("/", tags=["系统"])
def root():
    return {
        "name": "线性代数 RAG 智能助手 API",
        "version": "0.2.0",
        "docs": "/docs",
        "knowledge_base": COLLECTION_NAME,
    }


@app.get("/api/health", response_model=HealthResponse, tags=["系统"])
def health():
    """健康检查"""
    return rag_engine.health_check()


@app.post("/api/chat", response_model=ChatResponse, tags=["对话"])
async def chat(req: ChatRequest):
    """
    执行 RAG 问答（固定知识库 linear_algebra）。

    - 如果 `session_id` 为空，会自动创建新会话
    - 系统会从固定知识库检索相关上下文，再调用 LLM 生成回答
    """
    try:
        resp = await asyncio.to_thread(
            rag_engine.chat,
            query=req.query,
            session_id=req.session_id,
            api_key=req.api_key,
        )
        return ChatResponse(
            answer=resp.answer,
            sources=[SourceItem(**s) for s in resp.sources],
            session_id=resp.session_id,
            latency_ms=resp.latency_ms,
            retrieval_count=resp.retrieval_count,
            llm_model=resp.llm_model,
        )
    except Exception as e:
        logger.exception("Chat 接口出错")
        raise HTTPException(status_code=500, detail=f"内部错误: {e}")


@app.post("/api/sessions", response_model=SessionInfo, tags=["会话"])
def create_session():
    """创建新会话"""
    session = session_mgr.create_session()
    return SessionInfo(
        session_id=session.session_id,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


@app.get("/api/sessions", response_model=List[SessionInfo], tags=["会话"])
def list_sessions():
    """列出所有会话"""
    sessions = session_mgr.list_sessions()
    return [SessionInfo(**s) for s in sessions]


@app.delete("/api/sessions/{session_id}", tags=["会话"])
def delete_session(session_id: str):
    """删除会话"""
    ok = session_mgr.delete_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"status": "success", "session_id": session_id}
