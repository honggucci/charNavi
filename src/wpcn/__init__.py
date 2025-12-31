"""
WPCN Backtester - Wyckoff Phase Chart Navigator

폴더 구조:
    000_config/     - 설정 파일
    001_crypto/     - 암호화폐 거래소별 모듈
    002_data/       - 데이터 로딩/저장
    003_common/     - 공통 모듈 (core, features, wyckoff, navigation, metrics 등)
    004_execution/  - 백테스트 실행 엔진
    005_probability/ - 확률 계산 모듈
    006_engine/     - 엔진 레거시
    007_reporting/  - 리포팅
    008_tuning/     - 파라미터 튜닝
    009_cli/        - CLI
    010_ai_team/    - AI 팀 워크플로우
    999_legacy/     - 레거시 코드
"""
__all__ = ["__version__"]
__version__ = "0.1.0"
