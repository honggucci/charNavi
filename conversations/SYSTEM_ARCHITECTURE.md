# WPCN 시스템 아키텍처

> 마지막 업데이트: 2026-01-13 (v2.6.14)

## 프로젝트 개요

**WPCN (Wyckoff Probabilistic Crypto Navigator)**
- Wyckoff 분석 + 확률론 기반 암호화폐 백테스트/거래 엔진
- 멀티 타임프레임(MTF) 분석: 15분 ~ 1주
- 설계 철학: **"안 들어가는 날이 많아져야 계좌가 산다"**

---

## 프로젝트 구조 (v3.0)

```
wpcn-backtester-cli-noflask/
├── src/wpcn/
│   ├── _00_config/              # 설정
│   │   ├── config.py            # PATHS, setup_logging
│   │   ├── backtest.py          # 백테스트 기본 설정
│   │   ├── strategy.py          # 전략 설정
│   │   └── symbols.py           # 심볼별 설정
│   │
│   ├── _01_crypto/              # 거래소별 백테스트
│   │   └── 001_binance/
│   │       ├── 001_spot/        # 현물 (롱 only)
│   │       │   ├── 001_backtest/
│   │       │   ├── 002_sub/     # run_spot_backtest_mtf.py
│   │       │   └── 003_legacy/
│   │       └── 002_futures/     # 선물 (롱/숏)
│   │           ├── 001_backtest/
│   │           └── 003_legacy/
│   │
│   ├── _02_data/                # 데이터 처리
│   │   ├── ccxt_fetch.py        # CCXT 데이터 다운로드
│   │   ├── loader.py            # 데이터 로더
│   │   ├── resample.py          # 리샘플링
│   │   └── validation.py        # 데이터 검증
│   │
│   ├── _03_common/              # 공통 모듈
│   │   ├── _01_core/
│   │   │   └── types.py         # Theta, BacktestCosts, BacktestConfig
│   │   ├── _02_features/
│   │   │   ├── indicators.py    # ATR, RSI, Stoch RSI, ADX
│   │   │   ├── dynamic_params.py # Maxwell-Boltzmann, FFT, Hilbert
│   │   │   └── targets.py       # TP/SL 계산
│   │   ├── _03_wyckoff/
│   │   │   ├── box.py           # 박스 감지 (box_engine_freeze)
│   │   │   ├── events.py        # Spring/UTAD 감지
│   │   │   └── phases.py        # Wyckoff Phase A/B/C/D/E
│   │   └── _04_navigation/
│   │       ├── mtf_scoring.py   # MTF 점수 시스템
│   │       └── chart_navigator_v3.py
│   │
│   ├── _04_execution/           # 시뮬레이션 엔진
│   │   ├── broker_sim.py        # 메인 브로커 시뮬레이터
│   │   ├── broker_sim_mtf.py    # MTF 백테스트 (simulate_mtf)
│   │   ├── futures_backtest_15m.py  # 15분봉 지정가 백테스터 (v2.6.14)
│   │   ├── futures_backtest_v3.py   # 5분봉 시장가 백테스터
│   │   ├── hmm_entry_gate.py    # HMM 기반 진입 게이트
│   │   ├── invariants.py        # PnL 불변식 검증기 (v2.6.14)
│   │   ├── cost.py              # 비용 계산
│   │   └── state_persistence.py # 상태 저장
│   │
│   ├── _05_probability/         # 확률 모델
│   │   └── barrier.py           # Barrier Probability
│   │
│   ├── _06_engine/              # Navigation Engine
│   │   └── navigation.py        # compute_navigation
│   │
│   ├── _07_reporting/           # 리포트/차트
│   │
│   ├── _08_tuning/              # 파라미터 최적화 (v2.2)
│   │   ├── param_schema.py      # 파라미터 스키마 (분/봉 변환)
│   │   ├── param_store.py       # 파라미터 저장소
│   │   ├── param_optimizer.py   # Grid/Random/Bayesian/Optuna
│   │   ├── param_reducer.py     # 차원 축소 (v3)
│   │   ├── param_versioning.py  # Champion/Challenger (v2.2)
│   │   ├── walk_forward.py      # Walk-Forward 최적화
│   │   ├── sensitivity_analyzer.py # 민감도 분석
│   │   ├── oos_tracker.py       # OOS 실시간 추적
│   │   ├── scheduler.py         # 주간 배치 스케줄러
│   │   └── adaptive_space.py    # 적응형 탐색 공간
│   │
│   ├── _09_cli/                 # CLI 인터페이스
│   ├── _10_ai_team/             # AI 팀 연동 (hattz_empire)
│   ├── _11_flask/               # Flask 웹 (deprecated)
│   └── _99_legacy/              # 레거시 코드
│
├── data/
│   └── bronze/binance/futures/  # 원본 OHLCV 데이터
│       ├── BTC-USDT/15m/
│       ├── ETH-USDT/15m/
│       └── ...
│
├── results/                     # 최적화 결과 저장
├── runs/                        # 백테스트 결과 저장
├── tests/                       # 테스트 코드
└── docs/                        # 문서
```

---

## 핵심 타입 정의

### Theta (Wyckoff 박스 파라미터)
```python
@dataclass(frozen=True)
class Theta:
    pivot_lr: int      # 피벗 좌우 봉 수 (스윙 감지)
    box_L: int         # 박스 길이 (15m*96 = 24시간)
    m_freeze: int      # 박스 프리즈 기간
    atr_len: int       # ATR 계산 기간
    x_atr: float       # 돌파 임계값 (ATR 배수)
    m_bw: float        # 박스폭 비율 (reclaim 레벨)
    N_reclaim: int     # reclaim 허용 기간 (봉 수)
    N_fill: int        # 주문 체결 허용 기간
    F_min: float       # Fill Probability Minimum (0.0~1.0)
```

### BacktestCosts (거래 비용)
```python
@dataclass(frozen=True)
class BacktestCosts:
    fee_bps: float       # 수수료 (bps)
    slippage_bps: float  # 슬리피지 (bps)

# 프리셋
COSTS_SPOT_BTC = BacktestCosts(fee_bps=7.5, slippage_bps=3.0)   # 0.105% 편도
COSTS_SPOT_ALT = BacktestCosts(fee_bps=10.0, slippage_bps=10.0) # 0.20% 편도
COSTS_FUTURES = BacktestCosts(fee_bps=4.0, slippage_bps=5.0)    # 0.09% 편도
```

### BacktestConfig (백테스트 설정)
```python
@dataclass(frozen=True)
class BacktestConfig:
    initial_equity: float = 1.0
    max_hold_bars: int = 192       # 48시간 (15m*192)
    tp1_frac: float = 0.5          # TP1에서 50% 청산
    use_tp2: bool = True

    # Navigation Gate (v2.1 상향)
    conf_min: float = 0.65         # 최소 신뢰도
    edge_min: float = 0.70         # 최소 엣지
    confirm_bars: int = 2          # 휩쏘 방지
    reclaim_hold_bars: int = 2     # reclaim 후 박스 내부 유지

    # Regime Safety
    chaos_ret_atr: float = 3.0     # CHAOS 레짐 판정
    adx_len: int = 14
    adx_trend: float = 20.0
```

---

## 3개 프로젝트 구분

| 프로젝트 | 설명 | 진입점 |
|---------|------|--------|
| **현물 백테스트** | BTC 현물 롱 only, MTF 점수 시스템 | `run_spot_backtest_mtf.py` |
| **선물 백테스트** | 롱/숏 양방향, 레버리지 지원 | `run_futures_backtest_v2.py` |
| **파라미터 튜닝** | Walk-Forward + A/B Testing | `scheduler.py`, `run_tuning.py` |

---

## 프로젝트 1: 현물 백테스트

### 진입점
```
src/wpcn/_01_crypto/001_binance/001_spot/002_sub/run_spot_backtest_mtf.py
```

### 실행 명령
```bash
python -m wpcn._01_crypto.001_binance.001_spot.002_sub.run_spot_backtest_mtf
```

### 실행 흐름

```
main()
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 0: Walk-Forward 튜닝 (선택)                            │
│ 조건: USE_TUNING=True                                       │
│ 함수: run_walk_forward_tuning()                             │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 1: 데이터 로드                                         │
│ 함수: load_recent_data()                                    │
│ 경로: data/bronze/binance/futures/{symbol}/{timeframe}/     │
│ 출력: pd.DataFrame (timestamp, open, high, low, close, vol) │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: MTF 점수 계산                                       │
│ 함수: compute_mtf_scores()                                  │
│ 위치: _03_common/_04_navigation/mtf_scoring.py              │
│                                                             │
│ 타임프레임: 15m, 1h, 4h (+ 1d, 1w 선택)                     │
│ 출력: context_score, trigger_score (봉별)                   │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: 신호 생성                                           │
│ 함수: generate_mtf_signals()                                │
│                                                             │
│ 필터:                                                       │
│ - min_score >= 3.5~4.0                                      │
│ - min_tf_alignment >= 2 (2개 이상 TF 정렬)                  │
│ - min_rr_ratio >= 1.2~1.5                                   │
│ - min_context_score >= 1.5~2.0                              │
│ - min_trigger_score >= 1.0~1.5                              │
│                                                             │
│ 출력: LONG/SHORT 신호 DataFrame                             │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: 시뮬레이션                                          │
│ 함수: simulate_mtf()                                        │
│ 위치: _04_execution/broker_sim_mtf.py                       │
│                                                             │
│ 4.1 Navigation Gate                                         │
│     └─ conf >= conf_min, edge >= edge_min                   │
│                                                             │
│ 4.2 가격 체크                                               │
│     └─ low <= entry_price <= high (당일 체결 가능?)         │
│                                                             │
│ 4.3 포지션 진입                                             │
│     └─ Phase별 리스크% × confidence → 포지션 크기           │
│                                                             │
│ 4.4 청산 관리                                               │
│     └─ TP1(50%) → 손절 → 진입가로 이동                      │
│     └─ TP2(나머지)                                          │
│     └─ STOP (ATR × 배수)                                    │
│     └─ TIME_EXIT (max_hold_bars)                            │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 5: 결과 출력                                           │
│                                                             │
│ 반환값:                                                     │
│ - equity_df: 시간별 자산 곡선                               │
│ - trades_df: 모든 거래 기록                                 │
│ - signals_df: 생성된 신호 목록                              │
│ - nav_df: Navigation Gate 상태                              │
│                                                             │
│ 저장: runs/{timestamp}_mtf_backtest/ (parquet)              │
└─────────────────────────────────────────────────────────────┘
```

### 핵심 파일

| 파일 | 위치 | 역할 |
|-----|------|------|
| `run_spot_backtest_mtf.py` | `_01_crypto/.../001_spot/002_sub/` | 진입점, main() |
| `broker_sim_mtf.py` | `_04_execution/` | simulate_mtf() 시뮬레이션 |
| `mtf_scoring.py` | `_03_common/_04_navigation/` | MTF 점수 계산/신호 생성 |
| `box.py` | `_03_common/_03_wyckoff/` | box_engine_freeze() |
| `events.py` | `_03_common/_03_wyckoff/` | Spring/UTAD 감지 |
| `phases.py` | `_03_common/_03_wyckoff/` | Wyckoff Phase 판정 |
| `indicators.py` | `_03_common/_02_features/` | ATR, RSI, Stoch RSI |

### 환경변수

```env
# 필수
SYMBOL=BTC-USDT
TIMEFRAME=15m
BACKTEST_DAYS=90

# Theta (Wyckoff Box)
BOX_LOOKBACK=50
M_FREEZE=16
PIVOT_LR=3
N_FILL=5

# MTF 필터 (권장 범위)
MIN_SCORE=3.5              # 3.5~4.0
MIN_TF_ALIGNMENT=2
MIN_RR_RATIO=1.2           # 1.2~1.5
SL_ATR_MULT=1.5
TP_ATR_MULT=2.5
MIN_CONTEXT_SCORE=1.5      # 1.5~2.0
MIN_TRIGGER_SCORE=1.0      # 1.0~1.5

# 선택 기능
USE_TUNING=False
USE_PROBABILITY_MODEL=False
```

---

## 프로젝트 2: 선물 백테스트

### 진입점
```
src/wpcn/_01_crypto/001_binance/002_futures/001_backtest/run_futures_backtest_v2.py
```

### 실행 명령
```bash
python -m wpcn._01_crypto.001_binance.002_futures.001_backtest.run_futures_backtest_v2
```

### 선물 특징

| 항목 | 현물 | 선물 |
|-----|------|------|
| 방향 | 롱 only | 롱/숏 양방향 |
| 레버리지 | 1x | BTC 15x, ETH 5x |
| 펀딩비 | 없음 | 8시간마다 0.01% |
| 청산 | 없음 | 유지증거금 0.5% |

### 선물 설정

```python
FuturesConfigV2(
    leverage=15.0,              # BTC 15x
    margin_mode='isolated',
    maintenance_margin_rate=0.005,

    # 펀딩비
    funding_rate=0.0001,        # 0.01%
    funding_interval_bars=96,   # 8시간 (5분봉)

    # 15분봉 축적
    accumulation_pct_15m=0.03,  # 3%씩
    max_accumulation_pct=0.15,  # 최대 15%
    tp_pct_15m=0.015,           # +1.5%
    sl_pct_15m=0.010,           # -1.0%

    # 5분봉 단타
    scalping_pct_5m=0.02,       # 2%씩
    tp_pct_5m=0.008,            # +0.8%
    sl_pct_5m=0.005,            # -0.5%
)
```

---

## 프로젝트 3: 파라미터 튜닝 (v2.2)

### 모듈 구성

```
_08_tuning/
├── param_schema.py       # 파라미터 스키마 (분/봉 변환)
├── param_store.py        # 저장소 + 신뢰도 관리
├── param_versioning.py   # Champion/Challenger A/B Testing
├── param_optimizer.py    # Grid/Random/Bayesian/Optuna
├── param_reducer.py      # 차원 축소 (민감도 기반)
├── walk_forward.py       # Walk-Forward 최적화
├── sensitivity_analyzer.py # 민감도 분석
├── oos_tracker.py        # OOS 실시간 추적
├── scheduler.py          # 주간 배치 스케줄러
└── adaptive_space.py     # 적응형 탐색 공간
```

### 최적화 방식 (v3)

| 방식 | 클래스 | 설명 |
|-----|--------|------|
| Grid Search | `GridSearchOptimizer` | 모든 조합 탐색 |
| Random Search | `RandomSearchOptimizer` | 랜덤 샘플링 |
| Bayesian | `BayesianOptimizer` | GP 기반 (scikit-optimize) |
| **Optuna** | `OptunaOptimizer` | TPE 알고리즘 (v3 추가) |

### A/B Testing (v2.2)

```python
# Champion/Challenger 버전 관리
from wpcn._08_tuning import get_version_manager

manager = get_version_manager("BTC-USDT", "futures", "15m")

# 상태 확인
status = manager.get_status()
print(f"Champion: {status['champion']}")
print(f"Challenger: {status['challenger']}")

# 자동 승격/롤백 결정
action, comparison = manager.auto_decide()
```

**승격 조건**: Challenger OOS >= Champion OOS × 0.95
**롤백 조건**: Challenger OOS < Champion OOS × 0.80

### 주간 배치 실행 순서 (v2.2)

```
1. finalize_last_week_oos()     # 현재 Challenger의 OOS 확정
2. auto_promote_or_rollback()   # Champion vs Challenger 비교
3. run_adaptive_tuning()        # 새 파라미터 튜닝
4. register_challenger()        # 새 파라미터를 Challenger로 등록
```

### Walk-Forward 설정

```python
WalkForwardConfig(
    train_days=60,              # 훈련 기간
    test_days=30,               # 테스트 기간
    embargo_days=1,             # Embargo 기간
    n_candidates=50,            # 후보 수
    objective="ret_minus_mdd"   # 목적함수: ret% - MDD%
)
```

### V8 파라미터 공간

```python
V8_PARAM_SPACE = {
    # Wyckoff
    "pivot_lr": (2, 8),
    "box_L": (48, 192),
    "m_freeze": (16, 64),
    "atr_len": (10, 20),
    "x_atr": (0.2, 0.5),
    "m_bw": (0.05, 0.15),

    # 진입/청산
    "tp_pct": (0.01, 0.03),
    "sl_pct": (0.005, 0.015),
    "min_score": (3.0, 5.0),

    # RSI
    "rsi_oversold": (25, 40),
    "rsi_overbought": (60, 75),
}
```

---

## 기술 지표

### indicators.py

| 함수 | 설명 |
|-----|------|
| `atr(df, period)` | Average True Range |
| `rsi(df, period)` | Relative Strength Index |
| `stoch_rsi(df, period)` | Stochastic RSI (K, D) |
| `adx(df, period)` | Average Directional Index |
| `find_pivot_points(df, lr)` | 피벗 고점/저점 |
| `detect_rsi_divergence(df)` | RSI 다이버전스 |

### dynamic_params.py (물리학 기반)

| 클래스 | 설명 |
|--------|------|
| `MaxwellBoltzmannVolatility` | 변동성 레짐 분류 (low/medium/high/extreme) |
| `FFTCycleDetector` | 지배적 주기 감지 → 최적 보유 기간 |
| `HilbertPhaseAnalyzer` | 현재 위상 (bottom/rising/top/falling) |

---

## Wyckoff 모듈

### box.py
```python
def box_engine_freeze(df, theta) -> DataFrame:
    """
    박스 감지 및 프리즈
    - 고점/저점 기준 박스 범위 설정
    - m_freeze 기간 동안 박스 고정
    """
```

### events.py
```python
def detect_spring_utad(df, theta) -> DataFrame:
    """
    Spring/UTAD 감지
    - Spring: 박스 저점 이탈 → 빠른 복귀 (롱 신호)
    - UTAD: 박스 고점 돌파 → 빠른 복귀 (숏 신호)
    """
```

### phases.py
```python
def detect_wyckoff_phase(df, theta) -> DataFrame:
    """
    Wyckoff Phase 판정
    - ACCUMULATION: 박스권 + Spring 감지
    - RE_ACCUMULATION: 상승추세 + Higher Lows
    - DISTRIBUTION: 박스권 + UTAD 감지
    - RE_DISTRIBUTION: 하락추세 + Lower Highs
    - MARKUP/MARKDOWN: 박스 돌파
    """
```

---

## 데이터 흐름

```
┌─────────────────────────────────────────────────────────────┐
│                        데이터 흐름                           │
└─────────────────────────────────────────────────────────────┘

1. 데이터 준비
   ├─ OHLCV 로드 (parquet)
   │   └─ data/bronze/binance/futures/{symbol}/{tf}/{year}/{month}.parquet
   └─ 리샘플링 (15M → 1H, 4H)

2. 기술 지표 계산
   ├─ ATR (변동성)
   ├─ RSI, Stoch RSI (모멘텀)
   ├─ ADX (추세 강도)
   ├─ Wyckoff Phase (박스/스프링)
   └─ 다이버전스 (RSI)

3. 신호 생성
   ├─ MTF 점수 계산 (Context + Trigger)
   ├─ Navigation Gate 필터링
   └─ 최종 LONG/SHORT 신호

4. 백테스트 시뮬레이션
   ├─ 포지션 진입
   ├─ TP1/TP2/STOP 관리
   ├─ 시간 청산
   └─ 통계 계산 (수익률, MDD, 승률)

5. 파라미터 최적화 (선택)
   ├─ Walk-Forward 분할
   ├─ Optuna TPE 탐색
   ├─ OOS 검증
   └─ A/B Testing (Champion/Challenger)
```

---

## 백테스트 결과 예시 (2025-10-01 ~ 2025-12-30)

```
=== 결과 ===
초기 자본: $10,000
최종 자본: $10,142.59
수익률: +1.43%
MDD: 1.89%
진입: 30회 (모두 LONG - 현물 모드)
청산: 42회
승률: 57.1%

Exit types:
- STOP: 19회
- TP1: 12회
- TP2: 11회
```

---

## 빠른 참조

### 현물 백테스트
```bash
cd wpcn-backtester-cli-noflask
pip install -e .
python -m wpcn._01_crypto.001_binance.001_spot.002_sub.run_spot_backtest_mtf
```

### 선물 백테스트
```bash
python -m wpcn._01_crypto.001_binance.002_futures.001_backtest.run_futures_backtest_v2
```

### 파라미터 튜닝
```bash
# 단일 심볼 튜닝
python -m wpcn._08_tuning.run_tuning --symbol BTC-USDT --timeframe 15m

# 주간 배치 (A/B Testing 포함)
python -m wpcn._08_tuning.scheduler --run-now --market spot --timeframe 15m
```

### Python API
```python
from wpcn._03_common._01_core.types import Theta, BacktestCosts, BacktestConfig
from wpcn._04_execution.broker_sim_mtf import simulate_mtf

# 설정
theta = Theta(pivot_lr=3, box_L=50, m_freeze=16, atr_len=14,
              x_atr=2.0, m_bw=0.02, N_reclaim=8, N_fill=5, F_min=0.3)
costs = BacktestCosts(fee_bps=7.5, slippage_bps=5.0)
cfg = BacktestConfig(initial_equity=10000.0, max_hold_bars=288)

# 백테스트 실행
equity_df, trades_df, signals_df, nav_df = simulate_mtf(
    df=df,
    theta=theta,
    costs=costs,
    cfg=cfg,
    mtf=['15m', '1h', '4h'],
    spot_mode=True,
    min_score=3.5,
    min_tf_alignment=2,
    min_rr_ratio=1.2
)
```

---

## 설계 철학

> **"안 들어가는 날이 많아져야 계좌가 산다"**

### Gate 전략 (v2.1)

| 필터 | 조건 | 목적 |
|-----|------|------|
| conf_min | >= 0.65 | 쓰레기 신호 필터링 |
| edge_min | >= 0.70 | 엣지 없으면 진입 금지 |
| confirm_bars | >= 2 | 휩쏘 방지 |
| reclaim_hold_bars | >= 2 | 박스 내부 유지 확인 |

### 목표
- No-trade 비율: **30%+**
- 수익률 - MDD > 0
- 승률보다 손익비 중시

---

## v2.6.14 업데이트 (2026-01-13)

### 15분봉 지정가 백테스터

5분봉 ATR 스파이크로 인한 94% SL 히트 문제 해결을 위해 15분봉 전용 백테스터 추가.

**핵심 변경사항**:
- 타임프레임: 5분봉 → 15분봉 전용
- 진입: 시장가 → 지정가 (Pending Order)
- 체결 대기: 4봉(1시간) 미체결 시 취소
- 최대 보유: 32봉(8시간) 펀딩비 회피

**새 파일**:
- `src/wpcn/_04_execution/futures_backtest_15m.py`
- `src/wpcn/_04_execution/invariants.py` - PnL 불변식 검증기
- `policy_v1_2_relaxed_cooldown.py` - Relaxed Cooldown policy

### Policy v1.2 Relaxed Cooldown

Transition cooldown 병목(99% 차단) 해결:

```python
# v1.1 (기존)
transition_delta: 0.20
cooldown_bars: 2

# v1.2 (완화)
transition_delta: 0.40  # 더 큰 변화만 감지
cooldown_bars: 1        # 대기 시간 단축
```

**결과**:
| 버전 | 총 Test PnL | 2023 Trades | 2024 Trades |
|------|-------------|-------------|-------------|
| v1 (5분봉) | -$31,482 | - | - |
| v1.2 (15분봉) | **+$991** | **72** | **143** |

---

## 변경 이력

| 버전 | 날짜 | 변경사항 |
|-----|------|---------|
| **v2.6.14** | **2026-01-13** | **15분봉 지정가 백테스터 + Policy v1.2 Relaxed Cooldown** |
| v2.6.11 | 2026-01-12 | HMM Risk Filter v3 - apply_hmm_filter() 버그 수정 |
| v3.0 | 2026-01-04 | 문서 전면 리뉴얼, 현재 코드 구조 반영 |
| v2.2 | 2026-01-02 | A/B Testing (Champion/Challenger) 추가 |
| v2.1 | 2026-01-01 | OOS 추적, 민감도 분석, Gate 상향 |
| v2.0 | 2024-12-28 | MTF V3 설계, Re-Accum/Re-Distrib 추가 |
| v1.0 | 2024-12-26 | 초기 버전 |
