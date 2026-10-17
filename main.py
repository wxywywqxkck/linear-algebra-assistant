"""
入口文件 - 基于 RAG 的线性代数智能助手

使用方式:
    python main.py                 # 启动 Web 应用（后端 + 前端）
    python main.py --api-only      # 仅启动 FastAPI 后端
    python main.py --ui-only       # 仅启动 Streamlit 前端
    python main.py --rebuild       # 重建知识库
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def find_project_root() -> Path:
    """找到项目根目录（包含 .env.example 的目录）"""
    current = Path(__file__).resolve().parent
    return current


def run_backend():
    """启动 FastAPI 后端"""
    from src.config import get_config
    cfg = get_config()
    cmd = [
        sys.executable, "-m", "uvicorn",
        "src.api:app",
        "--host", cfg.fastapi_host,
        "--port", str(cfg.fastapi_port),
        "--reload",
    ]
    print(f"📡 启动后端: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(find_project_root()))


def run_frontend():
    """启动 Streamlit 前端"""
    from src.config import get_config
    cfg = get_config()
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        "src/app.py",
        "--server.port", str(cfg.streamlit_port),
    ]
    print(f"🎨 启动前端: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(find_project_root()))


def rebuild_knowledge_base():
    """重建知识库"""
    from src.document_loader import load_documents_to_chroma
    textbooks_dir = find_project_root() / "data" / "textbooks"
    chroma_dir = find_project_root() / "data" / "chroma_db"

    if not textbooks_dir.exists() or not any(textbooks_dir.iterdir()):
        print("⚠️  请先将文档放入 data/textbooks/ 目录")
        return

    print("🔄 正在重建知识库...")
    load_documents_to_chroma(
        input_dir=str(textbooks_dir),
        persist_dir=str(chroma_dir),
    )
    print("✅ 知识库重建完成！")


def main():
    parser = argparse.ArgumentParser(description="线性代数 RAG 智能助手")
    parser.add_argument("--api-only", action="store_true", help="仅启动后端 API")
    parser.add_argument("--ui-only", action="store_true", help="仅启动前端 UI")
    parser.add_argument("--rebuild", action="store_true", help="重建知识库")
    args = parser.parse_args()

    project_root = find_project_root()

    # 检查 .env
    env_path = project_root / ".env"
    if not env_path.exists():
        print("⚠️  未找到 .env 文件，请先复制 .env.example 为 .env 并填写配置")
        print("    cp .env.example .env")
        sys.exit(1)

    if args.rebuild:
        rebuild_knowledge_base()
        return

    if args.api_only:
        run_backend()
    elif args.ui_only:
        run_frontend()
    else:
        # 默认同时启动后端和前端
        from src.config import get_config
        cfg = get_config()

        import threading

        t_api = threading.Thread(target=run_backend, daemon=True)
        t_ui = threading.Thread(target=run_frontend, daemon=True)

        t_api.start()
        import time
        time.sleep(2)
        t_ui.start()

        print(f"\n✅ 服务已启动:")
        print(f"   前端界面: http://localhost:{cfg.streamlit_port}")
        print(f"   API 文档: http://localhost:{cfg.fastapi_port}/docs")
        print(f"   按 Ctrl+C 停止\n")

        try:
            t_api.join()
            t_ui.join()
        except KeyboardInterrupt:
            print("\n🛑 正在停止服务...")
            sys.exit(0)


if __name__ == "__main__":
    main()
