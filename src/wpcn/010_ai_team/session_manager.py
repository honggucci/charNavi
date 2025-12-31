"""
Session Manager - 프로젝트/태스크/에이전트별 세션 관리
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime


@dataclass
class Message:
    """대화 메시지"""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """프로젝트/태스크별 세션"""
    project: str           # "wpcn-backtest"
    task: str              # "v8-optimization"
    agent: str             # "claude", "gpt", "gemini"
    context: List[Dict] = field(default_factory=list)  # 대화 히스토리
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str, **kwargs):
        """메시지 추가"""
        msg = Message(role=role, content=content, metadata=kwargs)
        self.context.append(asdict(msg))
        self.updated_at = datetime.now().isoformat()

    def get_messages(self, last_n: Optional[int] = None) -> List[Dict]:
        """메시지 조회 (최근 n개)"""
        if last_n:
            return self.context[-last_n:]
        return self.context

    def clear(self):
        """세션 초기화"""
        self.context = []
        self.updated_at = datetime.now().isoformat()


class AITeamManager:
    """AI 팀 세션 관리자"""

    def __init__(self, project: str, session_dir: Optional[Path] = None):
        self.project = project
        self.session_dir = session_dir or Path(f"./sessions/{project}")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.sessions: Dict[str, Session] = {}

    def _get_session_key(self, task: str, agent: str) -> str:
        """세션 키 생성"""
        return f"{task}_{agent}"

    def _get_session_path(self, task: str, agent: str) -> Path:
        """세션 파일 경로"""
        return self.session_dir / f"{self._get_session_key(task, agent)}.json"

    def get_session(self, task: str, agent: str) -> Session:
        """태스크+에이전트별 세션 가져오기 (없으면 생성)"""
        key = self._get_session_key(task, agent)

        if key not in self.sessions:
            session_file = self._get_session_path(task, agent)

            if session_file.exists():
                # 기존 세션 로드
                data = json.loads(session_file.read_text(encoding="utf-8"))
                self.sessions[key] = Session(
                    project=data.get("project", self.project),
                    task=data.get("task", task),
                    agent=data.get("agent", agent),
                    context=data.get("context", []),
                    created_at=data.get("created_at", datetime.now().isoformat()),
                    updated_at=data.get("updated_at", datetime.now().isoformat()),
                    metadata=data.get("metadata", {})
                )
            else:
                # 새 세션 생성
                self.sessions[key] = Session(
                    project=self.project,
                    task=task,
                    agent=agent
                )

        return self.sessions[key]

    def save_session(self, task: str, agent: str):
        """세션 저장"""
        key = self._get_session_key(task, agent)
        session = self.sessions.get(key)

        if session:
            session_file = self._get_session_path(task, agent)
            session_file.write_text(
                json.dumps(asdict(session), ensure_ascii=False, indent=2),
                encoding="utf-8"
            )

    def save_all(self):
        """모든 세션 저장"""
        for key, session in self.sessions.items():
            task, agent = key.rsplit("_", 1)
            self.save_session(task, agent)

    def list_sessions(self) -> List[Dict]:
        """프로젝트 내 모든 세션 목록"""
        sessions = []
        for session_file in self.session_dir.glob("*.json"):
            try:
                data = json.loads(session_file.read_text(encoding="utf-8"))
                sessions.append({
                    "file": session_file.name,
                    "task": data.get("task"),
                    "agent": data.get("agent"),
                    "messages": len(data.get("context", [])),
                    "updated_at": data.get("updated_at")
                })
            except Exception:
                continue
        return sessions

    def delete_session(self, task: str, agent: str) -> bool:
        """세션 삭제"""
        key = self._get_session_key(task, agent)
        session_file = self._get_session_path(task, agent)

        if key in self.sessions:
            del self.sessions[key]

        if session_file.exists():
            session_file.unlink()
            return True
        return False


class WorkflowContext:
    """워크플로우 컨텍스트 - 여러 에이전트 간 공유 데이터"""

    def __init__(self, workflow_name: str, manager: AITeamManager):
        self.workflow_name = workflow_name
        self.manager = manager
        self.shared_data: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}  # 각 에이전트 결과

    def set(self, key: str, value: Any):
        """공유 데이터 설정"""
        self.shared_data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """공유 데이터 조회"""
        return self.shared_data.get(key, default)

    def add_result(self, agent: str, result: Any):
        """에이전트 결과 추가"""
        self.results[agent] = {
            "result": result,
            "timestamp": datetime.now().isoformat()
        }

    def get_result(self, agent: str) -> Optional[Any]:
        """에이전트 결과 조회"""
        return self.results.get(agent, {}).get("result")
