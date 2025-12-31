"""
WPCN Backtester - Wyckoff Phase Chart Navigator

폴더 구조:
    _00_config/     - 설정 파일
    _01_crypto/     - 암호화폐 거래소별 모듈
    _02_data/       - 데이터 로딩/저장
    _03_common/     - 공통 모듈
    _04_execution/  - 백테스트 실행 엔진
    _05_probability/ - 확률 계산
    _06_engine/     - 엔진 레거시
    _07_reporting/  - 리포팅
    _08_tuning/     - 파라미터 튜닝
    _09_cli/        - CLI
    _10_ai_team/    - AI 팀
    _99_legacy/     - 레거시
"""
__all__ = ["__version__"]
__version__ = "0.1.0"
