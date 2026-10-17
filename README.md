# 基于 RAG 的线性代数智能助手

一个本地运行的线性代数学习助手，基于 RAG（Retrieval-Augmented Generation）技术构建。支持知识问答、概念解释、习题解答，并提供引用溯源。

## 技术栈

- **RAG 框架**: LlamaIndex
- **向量数据库**: Chroma
- **Embedding / LLM**: OpenAI API（text-embedding-3-small + GPT-4o）
- **后端**: FastAPI
- **前端**: Streamlit

## 快速开始

```bash
# 1. 创建虚拟环境并安装依赖
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 然后编辑 .env，填入你的 OPENAI_API_KEY

# 3. 构建知识库
# 将线性代数文档（PDF / Markdown）放入 data/textbooks/
python -m src.document_loader --input data/textbooks --output data/chroma_db

# 4. 启动服务
# Windows
scripts\start.bat
# Linux/Mac
./scripts/start.sh

# 5. 访问
# 前端: http://localhost:8501
# API:  http://localhost:8000/docs
```

## 项目结构

```
linear-algebra-assistant/
├── data/
│   ├── textbooks/      # 知识库文档
│   ├── problems/       # 习题集
│   └── chroma_db/      # Chroma 向量数据库（自动生成）
├── src/
│   ├── config.py           # 配置管理
│   ├── document_loader.py  # 文档加载器
│   ├── text_chunker.py     # 文本分块器
│   ├── embedding.py        # 向量化模块
│   ├── retriever.py        # 向量检索器
│   ├── generator.py        # 响应生成器
│   ├── citation.py         # 引用追踪器
│   ├── session_manager.py  # 会话管理器
│   ├── rag_engine.py       # RAG 核心引擎
│   ├── api.py              # FastAPI 后端
│   └── app.py              # Streamlit 前端
├── tests/
├── scripts/
├── docs/
├── .env
├── .env.example
├── requirements.txt
└── main.py
```

## 功能特性

- ✅ 基于知识库的精准问答
- ✅ 引用溯源（回答附带来源章节）
- ✅ 多轮对话，支持上下文理解
- ✅ 数学公式 LaTeX 渲染
- ✅ 文档增量导入

## 常见问题

**Q: 如何添加新文档？**
A: 将文档放入 `data/textbooks/`，然后运行 `python -m src.document_loader --input data/textbooks --output data/chroma_db`

**Q: 如何切换模型？**
A: 修改 `.env` 中的 `LLM_MODEL` 和 `EMBEDDING_MODEL`。

**Q: 如何完全重建知识库？**
A: 删除 `data/chroma_db/` 目录后重新运行文档导入命令。
