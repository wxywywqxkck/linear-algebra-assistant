"""
引用追踪器

为回答附加引用来源标记，支持解析和渲染。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class Citation:
    """单条引用"""
    index: int
    title: str
    source: str = ""
    matched_text: str = ""


class CitationTracker:
    """引用追踪器"""

    # 匹配 [来源: XXX] 或 [XXX] 形式的引用标记
    CITATION_PATTERN = re.compile(
        r'\[来源[:：]\s*([^\]]+)\]|\[([^\]]{2,50})\]'
    )

    def __init__(self):
        self.citations: List[Citation] = []

    def add_citations(self, answer: str, sources: List[dict]) -> str:
        """
        确保回答中包含引用来源。如果 LLM 没有加引用标记，自动附加。

        Args:
            answer: LLM 生成的回答
            sources: 检索来源列表

        Returns:
            带引用标记的回答
        """
        if not sources:
            return answer

        # 检查是否已有引用标记
        has_citations = bool(self.CITATION_PATTERN.search(answer))

        if not has_citations:
            # 自动追加来源列表
            source_lines = ["\n\n--- **参考来源** ---"]
            for i, src in enumerate(sources, 1):
                title = src.get("title", "未知")
                source_lines.append(f"{i}. {title}")
            answer += "\n".join(source_lines)

        # 解析引用
        self._parse_citations(answer, sources)
        return answer

    def _parse_citations(self, answer: str, sources: List[dict]):
        """从回答中提取引用"""
        self.citations.clear()

        # 先收集所有可用标题
        available_titles = [(src.get("title", ""), src.get("source", "")) for src in sources]

        matches = self.CITATION_PATTERN.finditer(answer)
        for idx, match in enumerate(matches, 1):
            title = match.group(1) or match.group(2) or "未知"
            # 尝试匹配到原始来源
            matched_source = ""
            for t, s in available_titles:
                if title in t or t in title:
                    matched_source = s
                    break
            self.citations.append(Citation(
                index=idx,
                title=title,
                source=matched_source,
                matched_text=match.group(0),
            ))

    def get_citations(self) -> List[Citation]:
        return self.citations
