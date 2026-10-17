"""
配置管理模块

从 .env 文件加载配置，提供统一的配置访问接口。
支持：本地 HuggingFace Embedding + OpenAI 兼容 LLM（如 DeepSeek）
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# 加载 .env（从项目根目录）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


class AppConfig(BaseModel):
    """应用全局配置"""

    # ---------- LLM（OpenAI 兼容，支持 DeepSeek / 通义等） ----------
    llm_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    llm_base_url: str = Field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    )
    llm_model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "deepseek-chat"))
    llm_temperature: float = Field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.3")))
    llm_max_tokens: int = Field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "2048")))

    # ---------- Embedding ----------
    # provider: huggingface(本地) | openai(网络)
    embedding_provider: str = Field(default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "huggingface"))
    embedding_model: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    )
    embedding_dim: int = Field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "512")))
    # 如果 embedding_provider=openai，需要这两个
    embedding_api_key: str = Field(default_factory=lambda: os.getenv("EMBEDDING_API_KEY", ""))
    embedding_base_url: str = Field(default_factory=lambda: os.getenv("EMBEDDING_BASE_URL", ""))

    # ---------- Chroma ----------
    chroma_persist_dir: str = Field(default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db"))

    # ---------- 应用服务 ----------
    fastapi_host: str = Field(default_factory=lambda: os.getenv("FASTAPI_HOST", "127.0.0.1"))
    fastapi_port: int = Field(default_factory=lambda: int(os.getenv("FASTAPI_PORT", "8000")))
    streamlit_port: int = Field(default_factory=lambda: int(os.getenv("STREAMLIT_PORT", "8501")))

    # ---------- RAG ----------
    retrieval_top_k: int = Field(default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K", "5")))
    chunk_size: int = Field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "500")))
    chunk_overlap: int = Field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "50")))

    # ---------- API 安全 ----------
    api_access_key: str | None = Field(default_factory=lambda: os.getenv("API_ACCESS_KEY"))

    # ---------- 属性 ----------
    @property
    def absolute_chroma_dir(self) -> str:
        path = Path(self.chroma_persist_dir)
        if not path.is_absolute():
            path = _PROJECT_ROOT / path
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    @property
    def has_llm_key(self) -> bool:
        return bool(self.llm_api_key)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """获取应用配置（单例）"""
    return AppConfig()


# ===========================================================================
# LlamaIndex 全局 Settings 初始化
# ===========================================================================

_settings_initialized = False


def init_settings() -> None:
    """
    初始化 LlamaIndex 全局 Settings。

    只执行一次（幂等），在任何需要用 LlamaIndex 的地方先调用它。
    会根据 config 创建：
      - embed_model: HuggingFace 本地模型 或 OpenAI 兼容接口
      - llm: OpenAI 兼容 LLM（DeepSeek 等）
    """
    global _settings_initialized
    if _settings_initialized:
        return

    # HuggingFace 国内镜像（加速模型下载）
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    cfg = get_config()

    from llama_index.core import Settings

    # ---------- Monkey-patch LlamaIndex ----------
    # LlamaIndex 的 openai_modelname_to_contextsize() 只认识 OpenAI 自家模型，
    # 遇到 DeepSeek / 通义等兼容模型会抛 ValueError。
    # 这里补一个兜底：未知模型默认返回 8192。
    import llama_index.llms.openai.utils as _oi_utils
    _original = _oi_utils.openai_modelname_to_contextsize

    def _patched(model_name: str) -> int:
        try:
            return _original(model_name)
        except ValueError:
            return 8192  # DeepSeek / 通义 等常见窗口大小

    _oi_utils.openai_modelname_to_contextsize = _patched
    # 有些版本也把它挂在 llama_index.llms.openai.base 上，一起补
    try:
        import llama_index.llms.openai.base as _oi_base
        _oi_base.openai_modelname_to_contextsize = _patched
    except Exception:
        pass

    # ---------- Embedding ----------
    if cfg.embedding_provider == "huggingface":
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        Settings.embed_model = HuggingFaceEmbedding(
            model_name=cfg.embedding_model,
            device="cpu",  # 无 GPU 也能跑
        )
    else:
        # OpenAI 兼容 embedding
        from llama_index.embeddings.openai import OpenAIEmbedding

        Settings.embed_model = OpenAIEmbedding(
            model=cfg.embedding_model,
            api_key=cfg.embedding_api_key or cfg.llm_api_key,
            api_base=cfg.embedding_base_url or cfg.llm_base_url,
        )

    # ---------- LLM ----------
    # 不设置 Settings.llm —— 我们已经在 generator.py 里直接用
    # 原生 openai.OpenAI SDK 调用 Chat Completions，完全绕过 LlamaIndex
    # 的 LLM 层（它的 OpenAI 包装器硬编码了模型名校验）。
    # Settings.llm = None 时，LlamaIndex 检索/索引逻辑不受影响。
    Settings.llm = None

    _settings_initialized = True


def _reset_settings():
    """仅用于测试"""
    global _settings_initialized
    _settings_initialized = False
