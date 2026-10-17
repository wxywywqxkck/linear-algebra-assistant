"""
会话管理器

使用 SQLite 持久化存储多轮对话历史。
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Session:
    """会话"""
    session_id: str
    history: List[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class SessionManager:
    """基于 SQLite 的会话管理器"""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            # 放在项目根目录 data/ 下
            project_root = Path(__file__).resolve().parent.parent
            db_path = str(project_root / "data" / "sessions.db")

        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    history TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            conn.commit()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create_session(self) -> Session:
        """创建新会话"""
        session_id = str(uuid.uuid4())
        now = time.time()
        session = Session(
            session_id=session_id,
            history=[],
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO sessions (session_id, history, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (session_id, json.dumps([], ensure_ascii=False), now, now),
                )
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """获取会话"""
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()

        if row is None:
            return None

        return Session(
            session_id=row["session_id"],
            history=json.loads(row["history"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> Optional[Session]:
        """向会话追加消息"""
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()

                if row is None:
                    return None

                history = json.loads(row["history"])
                history.append({"role": role, "content": content})
                now = time.time()

                # 限制历史长度（最近 20 条）
                if len(history) > 20:
                    history = history[-20:]

                conn.execute(
                    "UPDATE sessions SET history = ?, updated_at = ? WHERE session_id = ?",
                    (json.dumps(history, ensure_ascii=False), now, session_id),
                )

        return self.get_session(session_id)

    def list_sessions(self) -> List[dict]:
        """列出所有会话"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT session_id, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
            ).fetchall()

        return [
            {
                "session_id": r["session_id"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]

    def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    "DELETE FROM sessions WHERE session_id = ?",
                    (session_id,),
                )
                return cursor.rowcount > 0

    def cleanup_expired(self, days: int = 30) -> int:
        """清理过期会话"""
        cutoff = time.time() - days * 86400
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    "DELETE FROM sessions WHERE updated_at < ?",
                    (cutoff,),
                )
                return cursor.rowcount
