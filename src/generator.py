"""
响应生成器

调用 LLM，基于检索到的上下文生成回答。
直接使用 OpenAI 兼容 SDK（支持 DeepSeek / 通义等），绕过 LlamaIndex 的 OpenAI 包装。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from loguru import logger

# LangSmith @traceable 装饰器（条件启用，用于追踪 LLM 生成链路）
try:
    if os.getenv("LANGSMITH_TRACING_V2", "").lower() in ("true", "1", "yes"):
        from langsmith import traceable
        logger.info("🟢 LangSmith tracing enabled")
    else:
        traceable = lambda **kw: (lambda f: f)  # no-op
except Exception:
    traceable = lambda **kw: (lambda f: f)  # no-op


# 系统 Prompt
SYSTEM_PROMPT = """你是一位专业的线性代数学习助手，擅长用清晰、易懂的方式讲解线性代数知识。

请遵循以下规则：
1. 严格基于提供的上下文（参考资料）回答问题，不要编造知识。
2. 如果上下文不足以回答问题，请明确说明"根据现有参考资料，我暂时无法回答这个问题"，不要猜测。
3. 回答时请引用来源，使用 [来源标题] 的格式标注。
4. 对于数学公式，使用 LaTeX 格式（如 $\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}$）。
5. 解释概念时尽量通俗易懂，可配合例子说明。
6. 解答习题时请逐步展示解题步骤。

回答结构建议：
- 先用一句话直接回答问题
- 再展开详细解释/证明/解题过程
- 最后总结关键点
"""


@dataclass
class GeneratedAnswer:
    """生成的回答"""
    answer: str
    sources: List[dict]
    total_tokens: int = 0


class ResponseGenerator:
    """响应生成器 - 基于 OpenAI 兼容 SDK"""

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        from .config import get_config
        cfg = get_config()

        self.model = model or cfg.llm_model
        self.temperature = temperature if temperature is not None else cfg.llm_temperature
        self.max_tokens = max_tokens or cfg.llm_max_tokens

        # 延迟初始化原生 OpenAI 客户端
        self._client = None
        self._client_key = None  # 记录当前 client 用的 key，用于切换检测

    def _get_client(self, api_key: Optional[str] = None):
        """
        获取 OpenAI 客户端（带 api_key 切换缓存）。

        - 传了 api_key 且和当前 client 的 key 不同 → 重建 client
        - 没传 → 用 config 里的默认 key
        """
        if api_key is None:
            from .config import get_config
            api_key = get_config().llm_api_key

        if self._client is None or self._client_key != api_key:
            from openai import OpenAI
            from .config import get_config
            cfg = get_config()

            self._client = OpenAI(
                api_key=api_key,
                base_url=cfg.llm_base_url,
            )
            self._client_key = api_key
        return self._client

    def build_context(self, sources: List[dict]) -> str:
        """将检索到的文档块拼接为上下文字符串"""
        context_parts = []
        for i, src in enumerate(sources, 1):
            title = src.get("title", src.get("source", "未知"))
            text = src.get("text", "")
            context_parts.append(f"--- 参考资料 {i} (来源: {title}) ---\n{text}")
        return "\n\n".join(context_parts)

    @traceable(run_type="llm")
    def generate(
        self,
        query: str,
        sources: List[dict],
        conversation_history: Optional[List[dict]] = None,
        api_key: Optional[str] = None,
    ) -> GeneratedAnswer:
        """
        基于检索结果生成回答。

        Args:
            query: 用户问题
            sources: 检索到的文档块，每个 dict 包含 text / title / source
            conversation_history: 多轮对话历史（可选）
            api_key: 可选的临时 API Key（覆盖 .env 里的默认值，不写盘）

        Returns:
            GeneratedAnswer
        """
        context = self.build_context(sources)

        # 构建用户消息
        user_message = f"""参考资料：
{context}

---

用户问题：{query}

请基于上述参考资料给出回答。"""

        # 构建消息列表（带历史）
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

        if conversation_history:
            # 只取最近 6 轮（避免过长）
            for msg in conversation_history[-6:]:
                messages.append(msg)

        messages.append({"role": "user", "content": user_message})

        logger.info(f"[LLM] Calling: model={self.model}, query='{query[:40]}...'")

        client = self._get_client(api_key=api_key)
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        answer_text = resp.choices[0].message.content or ""
        total_tokens = resp.usage.total_tokens if resp.usage else 0

        # 收集来源元数据
        source_meta = []
        for src in sources:
            source_meta.append({
                "title": src.get("title", src.get("source", "未知")),
                "source": src.get("source", ""),
            })

        return GeneratedAnswer(
            answer=answer_text,
            sources=source_meta,
            total_tokens=total_tokens,
        )
