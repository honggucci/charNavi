"""
AI Team Module - Multi-LLM Orchestration System

LLM별 역할:
- Claude: PM/오케스트레이터, 코딩
- GPT: 전략 수립, 검증
- Gemini: 리서치/데이터 수집, 문서화
"""

from .config import AITeamConfig
from .session_manager import AITeamManager, Session

__all__ = ["AITeamConfig", "AITeamManager", "Session"]
