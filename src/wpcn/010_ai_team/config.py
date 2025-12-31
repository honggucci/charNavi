"""
AI Team Configuration
API 키 및 모델 설정
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional
from pathlib import Path


@dataclass
class ModelConfig:
    """개별 모델 설정"""
    model_id: str
    api_key: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 60


@dataclass
class AITeamConfig:
    """AI 팀 전체 설정"""

    # Claude 설정 (PM/코딩)
    claude: ModelConfig = field(default_factory=lambda: ModelConfig(
        model_id="claude-sonnet-4-20250514",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=8192,
        temperature=0.3  # 코딩용 낮은 temperature
    ))

    # GPT 설정 (전략/검증)
    gpt: ModelConfig = field(default_factory=lambda: ModelConfig(
        model_id="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=4096,
        temperature=0.5
    ))

    # Gemini 설정 (리서치/문서화)
    gemini: ModelConfig = field(default_factory=lambda: ModelConfig(
        model_id="gemini-2.0-flash",
        api_key=os.getenv("GOOGLE_API_KEY"),
        max_tokens=8192,
        temperature=0.7
    ))

    # 세션 저장 경로
    session_dir: Path = field(default_factory=lambda: Path("./sessions"))

    # 역할별 기본 에이전트 매핑
    role_mapping: Dict[str, str] = field(default_factory=lambda: {
        "pm": "claude",
        "orchestrator": "claude",
        "coding": "claude",
        "strategy": "gpt",
        "verification": "gpt",
        "research": "gemini",
        "documentation": "gemini"
    })

    def get_model_for_role(self, role: str) -> ModelConfig:
        """역할에 맞는 모델 설정 반환"""
        agent_name = self.role_mapping.get(role.lower(), "claude")
        return getattr(self, agent_name)

    @classmethod
    def from_env(cls) -> "AITeamConfig":
        """환경변수에서 설정 로드"""
        return cls()

    def validate(self) -> Dict[str, bool]:
        """API 키 유효성 검사"""
        return {
            "claude": bool(self.claude.api_key),
            "gpt": bool(self.gpt.api_key),
            "gemini": bool(self.gemini.api_key)
        }
