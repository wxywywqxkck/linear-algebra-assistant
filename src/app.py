"""
Streamlit 前端

线性代数智能助手的 Web 交互界面。
固定知识库 linear_algebra，前端不可操作知识库，只走 FastAPI。
"""

from __future__ import annotations

import streamlit as st
from loguru import logger

# ---------------------------------------------------------------------------
# 硬编码常量（不可变）
# ---------------------------------------------------------------------------

API_URL = "http://127.0.0.1:8000"
COLLECTION_NAME = "linear_algebra"


# ---------------------------------------------------------------------------
# 页面配置
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="线性代数智能助手",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# 会话初始化
# ---------------------------------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------------------------
# 工具函数（固定走 FastAPI）
# ---------------------------------------------------------------------------

def call_rag_api(query: str, session_id: str | None, api_key: str | None = None):
    """调用 FastAPI 后端（唯一入口）"""
    import requests
    body = {"query": query, "session_id": session_id}
    if api_key:
        body["api_key"] = api_key
    resp = requests.post(f"{API_URL}/api/chat", json=body, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    class _Resp:
        pass
    r = _Resp()
    r.answer = data["answer"]
    r.sources = data.get("sources", [])
    r.session_id = data.get("session_id", session_id)
    r.latency_ms = data.get("latency_ms", 0)
    return r


# ---------------------------------------------------------------------------
# 侧边栏（只读状态展示，无操作入口）
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("📐 线性代数智能助手")
    st.caption("基于 RAG 的本地智能问答")

    st.divider()

    # 固定知识库信息（只读）
    st.subheader("📚 知识库")
    st.info(f"固定库: `{COLLECTION_NAME}`")
    try:
        import requests as _req
        hc = _req.get(f"{API_URL}/api/health", timeout=3).json()
        st.metric("文档块数", hc.get("document_count", 0))
        st.metric("LLM 模型", hc.get("llm_model", "-"))
        if hc.get("retriever_initialized"):
            st.success("✅ 知识库已就绪")
        else:
            st.warning("⚠️ 知识库为空")
    except Exception:
        st.warning("⚠️ 后端未启动")

    st.divider()

    # LLM API Key（用户自填，前端控制）
    st.subheader("🔑 LLM 配置")
    saved_key = st.session_state.get("user_api_key", "")
    api_key_input = st.text_input(
        "DeepSeek API Key",
        value=saved_key,
        type="password",
        key="api_key_input",
        placeholder="sk-... （留空则使用后端 .env 默认值）",
        help="API Key 仅在当前浏览器会话有效，不会写入磁盘。留空则使用后端 .env 里的默认 Key。",
    )
    if api_key_input != saved_key:
        st.session_state.user_api_key = api_key_input

    st.divider()

    # 新会话
    if st.button("🆕 新建会话", use_container_width=True):
        st.session_state.session_id = None
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("线性代数 RAG 智能助手 v0.2.0")


# ---------------------------------------------------------------------------
# 主界面
# ---------------------------------------------------------------------------

st.title("📐 线性代数智能助手")
st.caption("用自然语言提问，获得基于知识库的精准解答")

# 建议问题
with st.expander("💡 试试这些问题"):
    suggestions = [
        "什么是矩阵的秩？",
        "如何求 3x3 矩阵的行列式？",
        "特征值和特征向量的几何意义是什么？",
        "请解释线性变换的概念",
        "高斯消元法的步骤是什么？",
    ]
    cols = st.columns(len(suggestions))
    for i, q in enumerate(suggestions):
        if cols[i].button(q, key=f"suggest_{i}"):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 参考来源"):
                for src in msg["sources"]:
                    title = src.get("title", "未知") if isinstance(src, dict) else getattr(src, "title", "未知")
                    source = src.get("source", "") if isinstance(src, dict) else getattr(src, "source", "")
                    st.markdown(f"- **{title}**  \n  `{source}`")
        if msg.get("latency_ms"):
            st.caption(f"⏱️  响应耗时: {msg['latency_ms']}ms")

# 输入框
if prompt := st.chat_input("输入你的线性代数问题..."):
    # 前端预检：必须填 API Key（后端 .env 已清空默认值）
    user_key = st.session_state.get("user_api_key") or ""
    if not user_key.strip():
        st.error("🔑 请先在左侧 **🔑 LLM 配置** 区域填写 DeepSeek API Key")
        st.info("DeepSeek API Key 获取地址：https://platform.deepseek.com/api_keys")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("思考中...", expanded=True) as status:
            try:
                resp = call_rag_api(
                    prompt,
                    st.session_state.session_id,
                    api_key=st.session_state.get("user_api_key") or None,
                )
                status.update(label="✅ 回答生成完成", state="complete", expanded=False)

            except Exception as e:
                status.update(label="❌ 出错了", state="error", expanded=True)
                st.error(f"生成失败: {e}")
                st.stop()

        st.markdown(resp.answer)

        if getattr(resp, "sources", None):
            with st.expander("📚 参考来源"):
                for src in resp.sources:
                    title = src.get("title", "未知") if isinstance(src, dict) else getattr(src, "title", "未知")
                    source = src.get("source", "") if isinstance(src, dict) else getattr(src, "source", "")
                    st.markdown(f"- **{title}**  \n  `{source}`")

        latency = getattr(resp, "latency_ms", 0)
        st.caption(f"⏱️  响应耗时: {latency}ms | 🔗 会话: {resp.session_id[:8]}...")

        st.session_state.session_id = resp.session_id
        st.session_state.messages.append({
            "role": "assistant",
            "content": resp.answer,
            "sources": getattr(resp, "sources", []),
            "latency_ms": latency,
        })
