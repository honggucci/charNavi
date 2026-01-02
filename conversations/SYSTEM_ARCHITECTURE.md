# WPCN 시스템 아키텍처

> 마지막 업데이트: 2026-01-01

## 프로젝트 개요

**WPCN (Wyckoff Probabilistic Crypto Navigator)**
- Wyckoff 분석 + 확률론 기반 암호화폐 백테스트/거래 엔진
- 멀티 타임프레임(MTF) 분석: 5분 ~ 1주
- 설계 철학: "안 들어가는 날이 많아져야 계좌가 산다"

---

## 3개 프로젝트 구분

| 프로젝트 | 설명 | 진입점 |
|---------|------|--------|
| **현물 백테스트** | BTC 현물 롱 only, MTF 점수 시스템 | `run_spot_backtest_mtf.py` |
| **선물 백테스트** | 롱/숏 양방향, 15분 축적 + 5분 단타 | `run_futures_backtest_v2.py` |
| **파라미터 튜닝** | Walk-Forward 방식 Theta 최적화 | `USE_TUNING=True` |

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
main() (라인 251)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 0: Walk-Forward 튜닝 (선택)                            │
│ 조건: USE_TUNING=True                                       │
│ 함수: run_walk_forward_tuning() (라인 87)                   │
│                                                             │
│ 1. ThetaSpace 초기화 (파라미터 범위 정의)                   │
│ 2. 데이터 분할: Train(60일) + Embargo(1일) + Test(30일)     │
│ 3. 50개 후보 랜덤 샘플링 → simulate_mtf() 실행              │
│ 4. 목적함수: ret% - MDD% → 최고 점수 선택                   │
│ 5. Test 기간에서 검증 → 최종 Theta 반환                     │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 1: 데이터 로드                                         │
│ 함수: load_recent_data() (라인 47)                          │
│ 입력: symbol, timeframe, days                               │
│ 출력: pd.DataFrame (open, high, low, close, volume)         │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: MTF 점수 계산                                       │
│ 함수: compute_mtf_scores()                                  │
│ 위치: _03_common/_04_navigation/mtf_scoring.py              │
│                                                             │
│ 타임프레임: 15m, 1h, 4h, 1d, 1w                             │
│ 출력: context_score, trigger_score (봉별)                   │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: 신호 생성                                           │
│ 함수: generate_mtf_signals()                                │
│ 위치: _03_common/_04_navigation/mtf_scoring.py              │
│                                                             │
│ 필터:                                                       │
│ - min_score >= 4.0                                          │
│ - min_tf_alignment >= 2 (2개 이상 TF 정렬)                  │
│ - min_rr_ratio >= 1.5                                       │
│                                                             │
│ 출력: LONG/SHORT 신호 DataFrame                             │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: 시뮬레이션                                          │
│ 함수: simulate_mtf() (라인 102)                             │
│ 위치: _04_execution/broker_sim_mtf.py                       │
│                                                             │
│ 4.1 Navigation Gate (USE_NAVIGATION_GATE=True 시)           │
│     └─ compute_navigation()                                 │
│     └─ conf >= CONF_MIN, edge >= EDGE_MIN 체크              │
│                                                             │
│ 4.2 가격 체크                                               │
│     └─ l <= entry_price <= h (당일 범위 내 진입 가능?)      │
│                                                             │
│ 4.3 확률 필터 (USE_PROBABILITY_MODEL=True 시)               │
│     └─ calculate_barrier_probability()                      │
│     └─ P(TP) >= 0.50, EV_R >= 0.05 체크                     │
│                                                             │
│ 4.4 포지션 사이징                                           │
│     └─ calc_qty_risk_based() (라인 45)                      │
│     └─ Phase별 리스크% × confidence → 포지션 크기           │
│                                                             │
│ 4.5 거래 실행                                               │
│     └─ 진입 체결                                            │
│     └─ TP1/TP2 관리                                         │
│     └─ SL 청산                                              │
│     └─ 시간 청산 (MAX_HOLD_BARS)                            │
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
│ 저장: runs/ 디렉토리 (parquet)                              │
└─────────────────────────────────────────────────────────────┘
```

### 핵심 파일

| 파일 | 위치 | 역할 |
|-----|------|------|
| `run_spot_backtest_mtf.py` | `_01_crypto/.../001_spot/002_sub/` | 진입점, main() |
| `broker_sim_mtf.py` | `_04_execution/` | simulate_mtf() 시뮬레이션 |
| `mtf_scoring.py` | `_03_common/_04_navigation/` | MTF 점수 계산/신호 생성 |
| `box.py` | `_03_common/_03_wyckoff/` | box_engine_freeze() |
| `events.py` | `_03_common/_03_wyckoff/` | detect_spring_utad() |
| `barrier.py` | `_05_probability/` | calculate_barrier_probability() |
| `indicators.py` | `_03_common/_02_features/` | atr, rsi, stoch_rsi |

### 환경변수

```env
# 필수
SYMBOL=BTC-USDT
TIMEFRAME=15m
BACKTEST_DAYS=365

# Theta (Wyckoff Box)
BOX_LOOKBACK=50
M_FREEZE=16
PIVOT_LR=3
N_FILL=5

# MTF 필터
MIN_SCORE=4.0
MIN_TF_ALIGNMENT=2
MIN_RR_RATIO=1.5
SL_ATR_MULT=1.5
TP_ATR_MULT=2.5

# 선택 기능
USE_TUNING=False
USE_PROBABILITY_MODEL=False
USE_NAVIGATION_GATE=False
```

### 모듈 의존성

```
run_spot_backtest_mtf.py
    ├── wpcn._00_config.config (PATHS, setup_logging)
    ├── wpcn._03_common._01_core.types (Theta, BacktestCosts, BacktestConfig)
    ├── wpcn._04_execution.broker_sim_mtf (simulate_mtf)
    └── wpcn._08_tuning.theta_space (ThetaSpace) [조건부]

broker_sim_mtf.py
    ├── wpcn._03_common._04_navigation.mtf_scoring (compute_mtf_scores, generate_mtf_signals)
    ├── wpcn._06_engine.navigation (compute_navigation)
    ├── wpcn._05_probability.barrier (calculate_barrier_probability) [조건부]
    └── wpcn._04_execution.cost (apply_costs)
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

### 실행 흐름

```
main() (라인 163)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 반복: BTC/USDT (15x), ETH/USDT (5x) × 연도별               │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 1: 데이터 다운로드                                     │
│ 함수: fetch_ohlcv_to_parquet() (라인 64)                    │
│ 위치: wpcn.data.ccxt_fetch                                  │
│                                                             │
│ - CCXT로 Binance에서 5분봉 데이터 다운로드                  │
│ - 저장: data/raw/{exchange}_{symbol}_{tf}_{year}.parquet    │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: 데이터 로드                                         │
│ 함수: load_parquet() (라인 78)                              │
│ 위치: wpcn.data.loaders                                     │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: 선물 시뮬레이션                                     │
│ 함수: simulate_futures_v2() (라인 136)                      │
│ 위치: 002_futures/001_backtest/engine.py                    │
│                                                             │
│ 3.1 전처리                                                  │
│     ├─ 5분봉 → 15분봉 리샘플링                              │
│     ├─ Wyckoff Phase 감지 (15분봉)                          │
│     ├─ 박스 패턴 감지 (15분봉)                              │
│     ├─ RSI 다이버전스 감지 (5분봉)                          │
│     └─ RSI, Stochastic RSI 계산 (5분봉)                     │
│                                                             │
│ 3.2 15분봉 축적 포지션                                      │
│     ├─ Phase 기반 축적 신호 감지                            │
│     ├─ 3%씩 축적, 최대 15%                                  │
│     ├─ TP: +1.5%, SL: -1.0%                                 │
│     └─ 시간 청산: 6시간 (72봉)                              │
│                                                             │
│ 3.3 5분봉 단타 진입                                         │
│     ├─ RSI 다이버전스 + RSI 극단값 조건                     │
│     ├─ 2%씩 진입                                            │
│     ├─ TP: +0.8%, SL: -0.5%                                 │
│     └─ 시간 청산: 3시간 (36봉)                              │
│                                                             │
│ 3.4 펀딩비 처리                                             │
│     └─ 8시간마다 (96봉), 0.01%                              │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: 결과 출력                                           │
│                                                             │
│ 반환값:                                                     │
│ - equity_df: 시간별 자산 곡선                               │
│ - trades_df: 모든 거래 기록                                 │
│ - stats: 수익률, MDD, 거래 수, 축적 횟수, 청산 횟수, 펀딩비 │
└─────────────────────────────────────────────────────────────┘
```

### 선물 설정 (FuturesConfigV2)

```python
FuturesConfigV2(
    # 레버리지
    leverage=15.0 (BTC) / 5.0 (ETH),
    margin_mode='isolated',
    maintenance_margin_rate=0.005,  # 0.5%

    # 펀딩비
    funding_rate=0.0001,            # 0.01%
    funding_interval_bars=96,       # 8시간 (5분봉 기준)

    # 15분봉 축적
    accumulation_pct_15m=0.03,      # 3%씩
    max_accumulation_pct=0.15,      # 최대 15%
    tp_pct_15m=0.015,               # +1.5%
    sl_pct_15m=0.010,               # -1.0%
    max_hold_bars_15m=72,           # 6시간

    # 5분봉 단타
    scalping_pct_5m=0.02,           # 2%씩
    tp_pct_5m=0.008,                # +0.8%
    sl_pct_5m=0.005,                # -0.5%
    max_hold_bars_5m=36,            # 3시간

    # 다이버전스 필터
    rsi_oversold=45,
    rsi_overbought=55,
    stoch_oversold=40,
    stoch_overbought=60,
    accumulation_cooldown=6         # 30분 쿨다운
)
```

### 핵심 파일

| 파일 | 위치 | 역할 |
|-----|------|------|
| `run_futures_backtest_v2.py` | `002_futures/001_backtest/` | 진입점, main() |
| `engine.py` | `002_futures/001_backtest/` | simulate_futures_v2(), FuturesConfigV2 |
| `ccxt_fetch.py` | `_02_data/` | fetch_ohlcv_to_parquet() |
| `loaders.py` | `_02_data/` | load_parquet() |

---

## 프로젝트 3: 파라미터 튜닝

### 진입점
```
1. 통합: run_spot_backtest_mtf.py (USE_TUNING=True)
2. 독립: src/wpcn/_08_tuning/walk_forward.py
```

### 실행 흐름

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: ThetaSpace 초기화                                   │
│ 클래스: ThetaSpace                                          │
│ 위치: _08_tuning/theta_space.py:9                           │
│                                                             │
│ 파라미터 범위:                                              │
│ ┌─────────────┬────────────┬──────────────────────┐         │
│ │ 파라미터    │ 범위       │ 설명                 │         │
│ ├─────────────┼────────────┼──────────────────────┤         │
│ │ pivot_lr    │ (2, 5)     │ 피봇 좌우 거리       │         │
│ │ box_L       │ (30, 100)  │ 박스 길이            │         │
│ │ m_freeze    │ (8, 32)    │ 프리즈 기간          │         │
│ │ atr_len     │ (10, 20)   │ ATR 기간             │         │
│ │ x_atr       │ (1.5, 3.0) │ ATR 배수             │         │
│ │ m_bw        │ (0.01,0.05)│ 박스폭 비율          │         │
│ │ N_reclaim   │ (4, 16)    │ Reclaim 기간         │         │
│ │ N_fill      │ (3, 10)    │ 채움 확인 기간       │         │
│ │ F_min       │ (0.2, 0.5) │ 최소 채움 확률       │         │
│ └─────────────┴────────────┴──────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: 데이터 분할                                         │
│                                                             │
│ Walk-Forward 분할:                                          │
│ ┌──────────────────────────────────────────────────────┐    │
│ │ Train (60일)  │ Embargo (1일) │ Test (30일)         │    │
│ │ 17,280봉      │ 288봉         │ 8,640봉             │    │
│ └──────────────────────────────────────────────────────┘    │
│                                                             │
│ 5분봉 기준: 1일 = 288봉                                     │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: 후보 탐색                                           │
│ 함수: theta_space.sample()                                  │
│ 반복: n_candidates (기본 50회)                              │
│                                                             │
│ 각 반복:                                                    │
│ 1. 랜덤 Theta 샘플링                                        │
│ 2. simulate_mtf() 실행 (Train 데이터)                       │
│ 3. 목적함수 계산: ret% - MDD%                               │
│ 4. 최고 점수면 best_theta 업데이트                          │
│                                                             │
│ 진행 출력:                                                  │
│ [10/50] 탐색 중...                                          │
│ [15/50] 새 최고 점수: 12.34 (수익률: 18.5%, MDD: 6.2%)      │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: 테스트 검증                                         │
│                                                             │
│ - 최적 Theta로 Test 기간 백테스트                           │
│ - 테스트 수익률, 테스트 MDD 출력                            │
│ - 최종 Theta 파라미터 반환                                  │
│                                                             │
│ 출력 예시:                                                  │
│ === 테스트 검증 ===                                         │
│ 테스트 수익률: 8.45%                                        │
│ 테스트 MDD: 4.21%                                           │
│                                                             │
│ 최적 Theta:                                                 │
│   pivot_lr: 3                                               │
│   box_L: 72                                                 │
│   m_freeze: 24                                              │
│   N_fill: 6                                                 │
└─────────────────────────────────────────────────────────────┘
```

### 최적화 방식

| 방식 | 클래스 | 위치 | 설명 |
|-----|--------|------|------|
| Random Search | `RandomSearchOptimizer` | `param_optimizer.py:98` | 랜덤 샘플링 |
| Grid Search | `GridSearchOptimizer` | `param_optimizer.py:38` | 모든 조합 탐색 |
| Bayesian | `BayesianOptimizer` | scikit-optimize | GP 기반 |

### 목적함수

```python
# 기본 (현재 사용)
score = ret_pct - mdd

# 대안
score = sharpe_ratio
score = calmar_ratio  # ret / mdd
score = sortino_ratio
```

### 환경변수

```env
USE_TUNING=True
TUNING_TRAIN_DAYS=60
TUNING_TEST_DAYS=30
TUNING_N_CANDIDATES=50
```

### V8 파라미터 공간 (확장)

```python
# walk_forward.py:416-433
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

## 공유 컴포넌트

### 핵심 타입 (`_03_common/_01_core/types.py`)

```python
@dataclass
class Theta:  # 라인 91
    """Wyckoff 박스 파라미터"""
    pivot_lr: int = 4        # 피봇 좌우 봉 수
    box_L: int = 96          # 박스 길이 (24시간)
    m_freeze: int = 32       # 박스 고정 기간
    atr_len: int = 14        # ATR 길이
    x_atr: float = 0.35      # ATR 배수
    m_bw: float = 0.10       # 박스 폭 임계값
    N_reclaim: int = 3       # 리클레임 봉 수
    N_fill: int = 5          # 필 봉 수
    F_min: float = 0.70      # 최소 필 율

@dataclass
class BacktestCosts:  # 라인 16
    """거래 비용"""
    fee_bps: float = 7.5     # 수수료 (bps)
    slippage_bps: float = 5.0  # 슬리피지 (bps)

@dataclass
class BacktestConfig:  # 라인 47
    """백테스트 설정"""
    conf_min: float = 0.65   # 최소 신뢰도
    edge_min: float = 0.70   # 최소 엣지
    max_hold_bars: int = 192 # 최대 보유 기간
    initial_equity: float = 10000.0
```

### 기술 지표 (`_03_common/_02_features/indicators.py`)

| 함수 | 설명 |
|-----|------|
| `atr(df, period)` | Average True Range |
| `rsi(df, period)` | Relative Strength Index |
| `stoch_rsi(df, period)` | Stochastic RSI (K, D) |
| `adx(df, period)` | Average Directional Index |
| `find_pivot_points(df, lr)` | 피벗 고점/저점 |
| `detect_rsi_divergence(df)` | RSI 다이버전스 |

### Wyckoff 모듈 (`_03_common/_03_wyckoff/`)

| 파일 | 함수 | 설명 |
|-----|------|------|
| `box.py` | `box_engine_freeze()` | 박스 감지 및 고정 |
| `events.py` | `detect_spring_utad()` | Spring/UTAD 신호 |
| `phases.py` | `detect_wyckoff_phase()` | A/B/C/D/E 페이즈 판정 |

### 확률 모듈 (`_05_probability/barrier.py`)

```python
def calculate_barrier_probability(
    current_price, tp_price, sl_price,
    atr, max_hold_bars, is_long,
    entry_fee_pct, exit_fee_pct, slippage_pct,
    use_monte_carlo=True, n_simulations=1000
) -> BarrierProbability:
    """
    Returns:
        p_tp: TP 도달 확률
        p_sl: SL 도달 확률
        p_timeout: 시간초과 확률
        ev_r: 기대값 (R-unit)
    """
```

---

## 데이터 흐름 요약

```
┌─────────────────────────────────────────────────────────────┐
│                        공통 흐름                            │
└─────────────────────────────────────────────────────────────┘

1. 데이터 준비
   ├─ OHLCV 로드 (parquet / CCXT)
   └─ 리샘플링 (5M → 15M, 1H, 4H, 1D, 1W)

2. 기술 지표 계산
   ├─ ATR (변동성)
   ├─ RSI (모멘텀)
   ├─ ADX (추세)
   ├─ Wyckoff Phase (박스/스프링/스러스트)
   └─ 다이버전스 (RSI/Stochastic)

3. 신호 생성
   ├─ MTF 점수 계산 (Context + Trigger)
   ├─ Navigation Gate 필터링 (선택)
   └─ 최종 LONG/SHORT 신호

4. 백테스트 시뮬레이션
   ├─ 포지션 진입 (리스크 기반 사이징)
   ├─ TP/SL 관리
   ├─ 시간 청산
   └─ 통계 계산

5. 파라미터 최적화 (선택)
   ├─ Walk-Forward 분할
   ├─ 후보 탐색
   └─ Out-of-Sample 검증
```

---

## 설계 철학

> **"안 들어가는 날이 많아져야 계좌가 산다"**

### 필터링 전략

| 필터 | 조건 | 상태 |
|-----|------|------|
| MTF Score | Context >= 0.5, Trigger >= 0.5 | 활성화 |
| Navigation Gate | conf >= 0.55, edge >= 0.0 | 비활성화 |
| Probability Filter | P(TP) >= 0.50, EV_R >= 0.05 | 선택 |

### 목표

- No-trade 비율: **30%+**
- 수익률 - MDD > 0
- 승률보다 손익비 중시

---

## 빠른 참조

### 현물 백테스트 시작
```bash
cd wpcn-backtester-cli-noflask
python -m wpcn._01_crypto.001_binance.001_spot.002_sub.run_spot_backtest_mtf
```

### 선물 백테스트 시작
```bash
python -m wpcn._01_crypto.001_binance.002_futures.001_backtest.run_futures_backtest_v2
```

### 튜닝 활성화
```env
# .env
USE_TUNING=True
TUNING_TRAIN_DAYS=60
TUNING_TEST_DAYS=30
TUNING_N_CANDIDATES=50
```

### 확률 필터 활성화
```env
# .env
USE_PROBABILITY_MODEL=True
MIN_TP_PROBABILITY=0.50
MIN_EV_R=0.05
```
