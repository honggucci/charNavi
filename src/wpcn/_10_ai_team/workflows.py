"""
AI Team Workflows - 자동화된 워크플로우
"""

import asyncio
from typing import Dict, Any, Optional
from .config import AITeamConfig
from .session_manager import AITeamManager, WorkflowContext
from .agents import ClaudeAgent, GPTAgent, GeminiAgent


class AITeam:
    """AI 팀 통합 인터페이스"""

    def __init__(self, project: str, config: Optional[AITeamConfig] = None):
        self.project = project
        self.config = config or AITeamConfig.from_env()
        self.manager = AITeamManager(project)

        # 에이전트 초기화
        self.claude = ClaudeAgent(self.config.claude)
        self.gpt = GPTAgent(self.config.gpt)
        self.gemini = GeminiAgent(self.config.gemini)

    def get_agent(self, name: str):
        """에이전트 이름으로 가져오기"""
        agents = {
            "claude": self.claude,
            "gpt": self.gpt,
            "gemini": self.gemini
        }
        return agents.get(name.lower())

    async def strategy_development(
        self,
        problem: str,
        task_name: str = "strategy"
    ) -> Dict[str, Any]:
        """
        전략 수립 워크플로우

        1. Claude: PM으로서 문제 분석
        2. GPT: 여러 전략 제안
        3. Claude + GPT: 크로스체크
        4. Gemini: 리서치 보완
        5. Claude: 최종 결정
        """
        ctx = WorkflowContext(f"{task_name}_strategy", self.manager)

        # 1. Claude PM이 문제 분석
        pm_session = self.manager.get_session(task_name, "claude_pm")
        analysis = await self.claude.chat(
            f"다음 문제를 분석하고 해결 방향을 제시해주세요:\n\n{problem}",
            session=pm_session,
            system_prompt="당신은 AI 팀의 PM입니다. 문제를 분석하고 팀원들에게 지시를 내립니다."
        )
        ctx.add_result("claude_analysis", analysis)
        self.manager.save_session(task_name, "claude_pm")

        # 2. GPT가 전략 제안
        strategy_session = self.manager.get_session(task_name, "gpt_strategy")
        strategies = await self.gpt.develop_strategy(
            problem,
            session=strategy_session
        )
        ctx.add_result("gpt_strategies", strategies)
        self.manager.save_session(task_name, "gpt_strategy")

        # 3. Claude가 GPT 전략 크로스체크
        review_session = self.manager.get_session(task_name, "claude_review")
        review = await self.claude.chat(
            f"GPT가 제안한 전략을 검토해주세요:\n\n{strategies}",
            session=review_session,
            system_prompt="당신은 전략 검토 전문가입니다. 비판적으로 분석하세요."
        )
        ctx.add_result("claude_review", review)
        self.manager.save_session(task_name, "claude_review")

        # 4. 최종 정리
        final_session = self.manager.get_session(task_name, "claude_final")
        final = await self.claude.chat(
            f"""원본 문제: {problem}

분석 결과: {analysis}

GPT 전략: {strategies}

검토 결과: {review}

위 내용을 종합하여 최종 전략을 결정해주세요.""",
            session=final_session,
            system_prompt="당신은 최종 의사결정자입니다."
        )
        ctx.add_result("final_strategy", final)
        self.manager.save_session(task_name, "claude_final")

        return {
            "analysis": analysis,
            "strategies": strategies,
            "review": review,
            "final_strategy": final
        }

    async def code_development(
        self,
        requirement: str,
        task_name: str = "coding"
    ) -> Dict[str, Any]:
        """
        코드 개발 워크플로우

        1. Claude: 요구사항 분석 및 설계
        2. Claude: 코드 작성
        3. GPT: 코드 검증
        4. Claude: 피드백 반영
        """
        ctx = WorkflowContext(f"{task_name}_coding", self.manager)

        # 1. 요구사항 분석
        design_session = self.manager.get_session(task_name, "claude_design")
        design = await self.claude.chat(
            f"다음 요구사항을 분석하고 설계해주세요:\n\n{requirement}",
            session=design_session,
            system_prompt="당신은 소프트웨어 아키텍트입니다. 요구사항을 분석하고 설계를 제시하세요."
        )
        ctx.add_result("design", design)
        self.manager.save_session(task_name, "claude_design")

        # 2. 코드 작성
        code_session = self.manager.get_session(task_name, "claude_code")
        code = await self.claude.chat(
            f"""설계 기반으로 코드를 작성해주세요:

설계:
{design}

요구사항:
{requirement}""",
            session=code_session,
            system_prompt="당신은 시니어 개발자입니다. 깔끔하고 효율적인 코드를 작성하세요."
        )
        ctx.add_result("initial_code", code)
        self.manager.save_session(task_name, "claude_code")

        # 3. GPT 검증
        verify_session = self.manager.get_session(task_name, "gpt_verify")
        verification = await self.gpt.verify_solution(
            code,
            requirement,
            session=verify_session
        )
        ctx.add_result("verification", verification)
        self.manager.save_session(task_name, "gpt_verify")

        # 4. 피드백 반영 (필요시)
        if isinstance(verification, dict) and verification.get("issues"):
            refine_session = self.manager.get_session(task_name, "claude_refine")
            refined = await self.claude.chat(
                f"""GPT 검증 결과를 반영하여 코드를 개선해주세요:

원본 코드:
{code}

검증 결과:
{verification}""",
                session=refine_session,
                system_prompt="피드백을 반영하여 코드를 개선하세요."
            )
            ctx.add_result("refined_code", refined)
            self.manager.save_session(task_name, "claude_refine")
            code = refined

        return {
            "design": design,
            "code": code,
            "verification": verification
        }

    async def research_and_document(
        self,
        topic: str,
        task_name: str = "research"
    ) -> Dict[str, Any]:
        """
        리서치 및 문서화 워크플로우

        1. Gemini: 주제 리서치
        2. Gemini: 문서화
        3. Claude: 검토 및 보완
        """
        ctx = WorkflowContext(f"{task_name}_research", self.manager)

        # 1. Gemini 리서치
        research_session = self.manager.get_session(task_name, "gemini_research")
        research = await self.gemini.research(
            topic,
            depth="comprehensive",
            session=research_session
        )
        ctx.add_result("research", research)
        self.manager.save_session(task_name, "gemini_research")

        # 2. Gemini 문서화
        doc_session = self.manager.get_session(task_name, "gemini_doc")
        documentation = await self.gemini.create_documentation(
            research,
            doc_type="readme",
            session=doc_session
        )
        ctx.add_result("documentation", documentation)
        self.manager.save_session(task_name, "gemini_doc")

        # 3. Claude 검토
        review_session = self.manager.get_session(task_name, "claude_review")
        review = await self.claude.chat(
            f"""Gemini가 작성한 문서를 검토해주세요:

{documentation}

개선이 필요한 부분을 지적해주세요.""",
            session=review_session,
            system_prompt="문서 검토자로서 품질을 확인하세요."
        )
        ctx.add_result("review", review)
        self.manager.save_session(task_name, "claude_review")

        return {
            "research": research,
            "documentation": documentation,
            "review": review
        }


# 사용 예시
async def example_usage():
    """사용 예시"""
    # AI 팀 초기화
    team = AITeam(project="wpcn-backtest")

    # 전략 수립 워크플로우
    result = await team.strategy_development(
        problem="15분봉 스캘핑 전략의 승률을 60% 이상으로 개선하고 싶습니다.",
        task_name="scalping_v2"
    )

    print("=== 최종 전략 ===")
    print(result["final_strategy"])


if __name__ == "__main__":
    asyncio.run(example_usage())
