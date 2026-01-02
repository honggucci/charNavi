# WPCN Spot Trading Logic Documentation
> 작성일: 2026-01-01
> 목적: 현물 매매 로직 상세 분석 및 참조용 문서

---

## 1. 전체 트레이딩 플로우

```
[OHLCV Data]
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  1. NAVIGATION (navigation.py)                              │
│  ├─ Regime Detection (RANGE/TREND_UP/TREND_DOWN/CHAOS)     │
│  ├─ 6-Page Scoring (accum/reaccum/distrib/redistr/spring/utad) │
│  ├─ Softmax → Confidence                                    │
│  └─ trade_gate = regime≠CHAOS & conf≥0.65 & edge≥0.70      │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  2. SIGNAL DETECTION (events.py)                            │
│  ├─ Spring: box_low break → reclaim → hold N bars          │
│  └─ UTAD: box_high break → reclaim → hold N bars           │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  3. DYNAMIC PARAMETERS (dynamic_params.py)                  │
│  ├─ Maxwell-Boltzmann → Volatility Regime                  │
│  ├─ FFT → Dominant Cycle (optimal hold)                    │
│  ├─ Hilbert → Phase Position (bottom/rising/top/falling)   │
│  └─ Output: stop_ratio, tp1_ratio, tp2_ratio               │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  4. POSITION MANAGEMENT (broker_sim.py)                     │
│  ├─ Phase A/B: Spider-web Accumulation (3% per signal)     │
│  ├─ Phase C: Main Position (20% allocation)                │
│  ├─ Exit: STOP / TP1 / TP2 / TIME_EXIT                     │
│  └─ Equity Tracking                                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Navigation Gate 계산 (navigation.py)

### 2.1 6-Page 시스템

| Page | 설명 | 주요 조건 |
|------|------|----------|
| `accum` | 축적(매집) | RANGE + z≤-2.0 + absorption + HTF oversold |
| `reaccum` | 재축적 | TREND_UP + (RANGE) + HTF oversold |
| `distrib` | 분산(배분) | RANGE + z≥+2.0 + absorption + HTF overbought |
| `redistr` | 재분산 | TREND_DOWN + (RANGE) + HTF overbought |
| `spring` | 스프링 신호 | edge_event=="spring" → +4.0 boost |
| `utad` | UTAD 신호 | edge_event=="utad" → +4.0 boost |

### 2.2 점수 계산 공식 (Lines 88-129)

```python
# 기본 점수 (regime + z-score + absorption + HTF)
scores["accum"] = 1.2*is_range + 1.0*z_ext_low + 0.7*absorption + 0.6*htf_oversold
scores["distrib"] = 1.2*is_range + 1.0*z_ext_high + 0.7*absorption + 0.6*htf_overbought
scores["reaccum"] = 1.0*is_up + 0.4*is_range + 0.3*htf_oversold
scores["redistr"] = 1.0*is_dn + 0.4*is_range + 0.3*htf_overbought

# 이벤트 점수 (Spring/UTAD 감지 시 +4.0 부스트)
if edge_event == "spring":
    scores["spring"] = 4.0 + 1.0*absorption + 0.6*htf_oversold
else:
    scores["spring"] = -0.5 + 0.3*z_ext_low

if edge_event == "utad":
    scores["utad"] = 4.0 + 1.0*absorption + 0.6*htf_overbought
else:
    scores["utad"] = -0.5 + 0.3*z_ext_high
```

### 2.3 Softmax 정규화 (Lines 131-140)

```python
def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.nanmax(x)  # 수치 안정성
    e = np.exp(x)
    return e / (np.nansum(e) + 1e-12)

# 각 행에 대해 softmax 적용 → 확률 분포
probs[i] = _softmax(scores[i])
```

### 2.4 Trade Gate 조건 (Lines 159-164)

```python
trade_gate = (
    (regime != "CHAOS") &           # 혼돈 시장 제외
    (confidence >= bt.conf_min) &   # 0.65 이상 (2026-01-01 상향)
    (edge_score >= bt.edge_min) &   # 0.70 이상 (2026-01-01 상향)
    (top1_runlen >= bt.confirm_bars) # 2 이상 (연속 확인)
).astype(int)
```

**핵심 철학**: "안 들어가는 날이 많아져야 계좌가 산다"

---

## 3. Dynamic Parameters (dynamic_params.py)

### 3.1 Maxwell-Boltzmann Volatility

변동성 분포를 물리학의 맥스웰-볼츠만 분포로 모델링:

```python
class MaxwellBoltzmannVolatility:
    def calculate_regime(self, atr_normalized):
        """
        Returns: 'low' | 'medium' | 'high' | 'extreme'

        - low: ATR < 25th percentile
        - medium: 25th ≤ ATR < 75th percentile
        - high: 75th ≤ ATR < 95th percentile
        - extreme: ATR ≥ 95th percentile
        """
```

### 3.2 FFT Cycle Detection

지배적 주기를 FFT로 감지하여 최적 보유 기간 결정:

```python
class FFTCycleDetector:
    def detect_dominant_cycle(self, prices, min_period=5, max_period=100):
        """
        - FFT 스펙트럼에서 최대 진폭 주기 탐지
        - 주기의 1/4 ~ 1/2을 최적 보유 기간으로 제안
        """
```

### 3.3 Hilbert Phase Analysis

힐버트 변환으로 현재 위상 판단:

```python
class HilbertPhaseAnalyzer:
    def analyze_phase(self, prices):
        """
        Returns: 'bottom' | 'rising' | 'top' | 'falling'

        - bottom: 매수 최적 (Spring 확인)
        - rising: 홀딩 유지
        - top: 매도 고려 (UTAD 확인)
        - falling: 현금 보유
        """
```

### 3.4 동적 ATR 배수 계산

```python
def calculate_dynamic_atr_multipliers(df, theta, bt):
    """
    Returns:
        stop_ratio: 0.15 ~ 0.35 (낮은 변동성 = 타이트, 높은 변동성 = 넓게)
        tp1_ratio:  0.50 ~ 0.80 (1차 익절)
        tp2_ratio:  0.70 ~ 1.20 (2차 익절)

    조건:
        - TP1 > round_trip_cost (비용 커버 필수)
        - TP2 > TP1 (항상)
        - STOP < TP1 (손익비 > 1.0)
    """
```

---

## 4. Position Management (broker_sim.py)

### 4.1 포지션 유형

| 유형 | 배분 | 설명 |
|------|------|------|
| **Main Position** | 20% | Phase C 신호 (Spring/UTAD 확정) |
| **Accumulation** | 3% × N | Phase A/B 스파이더웹 (분할 매집) |
| **Short Accumulation** | 마진 기반 | 헤지용 (Futures only) |

### 4.2 메인 루프 구조 (Lines 95-412)

```python
for i, bar in enumerate(nav.itertuples()):

    # ═══════════════════════════════════════════════════════
    # STEP 1: Phase A/B Accumulation (Lines 99-154)
    # ═══════════════════════════════════════════════════════
    if is_phase_ab and trade_gate:
        # 3% 배분으로 분할 매집
        entry_price = bar.close
        accumulation_positions.append({
            'entry': entry_price,
            'size': equity * 0.03,
            'stop': entry_price * (1 - stop_ratio),
            'tp': entry_price * (1 + 0.015)  # +1.5%
        })

    # ═══════════════════════════════════════════════════════
    # STEP 2: Accumulation TP/SL Check (Lines 156-241)
    # ═══════════════════════════════════════════════════════
    for pos in accumulation_positions:
        if bar.low <= pos['stop']:
            # 손절: -1.0% 손실
            realize_loss(pos)
        elif bar.high >= pos['tp']:
            # 익절: +1.5% 이익
            realize_profit(pos)

    # ═══════════════════════════════════════════════════════
    # STEP 3: Main Signal Processing (Lines 243-334)
    # ═══════════════════════════════════════════════════════
    if edge_score >= edge_min and trade_gate:
        if pending_order is None:
            # 주문 생성 (다음 봉 시가 체결)
            pending_order = {
                'direction': 'long' if edge_event == 'spring' else 'short',
                'size': equity * 0.20,
                'stop': dynamic_stop,
                'tp1': dynamic_tp1,
                'tp2': dynamic_tp2
            }

    # 펜딩 주문 체결
    if pending_order and bar_index > order_bar:
        fill_order(pending_order, bar.open)
        main_position = pending_order
        pending_order = None

    # ═══════════════════════════════════════════════════════
    # STEP 4: Position Exit Management (Lines 336-381)
    # ═══════════════════════════════════════════════════════
    if main_position:
        # STOP 체크
        if bar.low <= main_position['stop']:
            close_position('STOP', main_position['stop'])

        # TP1 체크 (50% 청산)
        elif bar.high >= main_position['tp1'] and not tp1_hit:
            partial_close(0.5, 'TP1', main_position['tp1'])
            tp1_hit = True
            # 손절 → 진입가로 이동 (본전)
            main_position['stop'] = main_position['entry']

        # TP2 체크 (나머지 청산)
        elif bar.high >= main_position['tp2']:
            close_position('TP2', main_position['tp2'])

        # TIME EXIT (최대 보유 기간 초과)
        elif bars_held >= max_hold_bars:
            close_position('TIME_EXIT', bar.close)

    # ═══════════════════════════════════════════════════════
    # STEP 5: Equity Calculation (Lines 383-412)
    # ═══════════════════════════════════════════════════════
    equity = cash + sum(open_position_values)
```

### 4.3 Exit 유형별 비용

| Exit Type | 비용 적용 | 설명 |
|-----------|----------|------|
| STOP | 손실 + 수수료 | 강제 청산, 슬리피지 추가 |
| TP1 | 이익 - 수수료 | 50% 부분 청산 |
| TP2 | 이익 - 수수료 | 나머지 청산 |
| TIME_EXIT | 현재가 - 수수료 | 시간 만료 청산 |

---

## 5. 비용 구조 (2026-01-01 수정)

### 5.1 비용 프리셋

```python
COSTS_SPOT_BTC = BacktestCosts(fee_bps=7.5, slippage_bps=3.0)   # 총 0.105%
COSTS_SPOT_ALT = BacktestCosts(fee_bps=10.0, slippage_bps=10.0) # 총 0.20%
COSTS_FUTURES = BacktestCosts(fee_bps=4.0, slippage_bps=5.0)    # 총 0.09%
```

### 5.2 비용 검증 로직

```python
def total_one_way_pct(self) -> float:
    return (self.fee_bps + self.slippage_bps) / 100

def total_round_trip_pct(self) -> float:
    return self.total_one_way_pct() * 2

# TP 검증: TP > round_trip_cost 필수
assert tp1_ratio > costs.total_round_trip_pct()
```

---

## 6. Gate Thresholds (2026-01-01 상향)

| Parameter | 이전 | 수정 | 이유 |
|-----------|-----|------|------|
| `conf_min` | 0.50 | **0.65** | 쓰레기 신호 필터링 |
| `edge_min` | 0.60 | **0.70** | 엣지 없으면 진입 금지 |
| `confirm_bars` | 1 | **2** | 휩쏘 방지 |
| `reclaim_hold_bars` | (없음) | **2** | 박스 내부 유지 확인 |

---

## 7. Spot Direction Ratios (현물 전용)

현물에서는 숏이 불가능하므로 `cash_exposure`로 대체:

```python
DIRECTION_RATIOS_SPOT = {
    "bullish": {"long_exposure": 0.80, "cash_exposure": 0.20},
    "neutral": {"long_exposure": 0.50, "cash_exposure": 0.50},
    "bearish": {"long_exposure": 0.20, "cash_exposure": 0.80}
}
```

**vs 선물**:
```python
DIRECTION_RATIOS_FUTURES = {
    "bullish": {"long": 0.70, "short": 0.30},
    "neutral": {"long": 0.50, "short": 0.50},
    "bearish": {"long": 0.30, "short": 0.70}
}
```

---

## 8. Spring/UTAD 감지 로직 (events.py)

### 8.1 Spring 감지 조건

```python
def detect_spring(df, theta):
    """
    Spring = 축적 구간에서 박스 저점 이탈 후 빠른 복귀

    조건:
    1. 가격이 box_low 아래로 이탈 (shakeout)
    2. 같은 봉 또는 다음 봉에서 box_low 위로 복귀 (reclaim)
    3. reclaim_hold_bars 동안 박스 내부 유지 (휩쏘 필터)
    4. 볼륨 급증 확인 (선택적)
    """
```

### 8.2 UTAD 감지 조건

```python
def detect_utad(df, theta):
    """
    UTAD = 분산 구간에서 박스 고점 돌파 후 빠른 하락

    조건:
    1. 가격이 box_high 위로 돌파 (upthrust)
    2. 같은 봉 또는 다음 봉에서 box_high 아래로 복귀 (reclaim)
    3. reclaim_hold_bars 동안 박스 내부 유지 (휩쏘 필터)
    4. 볼륨 급증 확인 (선택적)
    """
```

### 8.3 reclaim_hold_bars 로직

```python
# 2단계 검증 (휩쏘 방지)
def validate_reclaim(df, reclaim_bar, reclaim_hold_bars):
    for i in range(reclaim_hold_bars):
        bar = df.iloc[reclaim_bar + i]
        if bar.low < box_low or bar.high > box_high:
            return False  # 다시 이탈 → 무효화
    return True  # N봉 동안 박스 내부 유지 → 신호 확정
```

---

## 9. 요약: 트레이딩 플로우 체크리스트

1. **Regime 확인**: CHAOS면 진입 금지
2. **Navigation Gate**: conf ≥ 0.65, edge ≥ 0.70
3. **Signal 확인**: Spring 또는 UTAD 감지
4. **Reclaim 확인**: reclaim_hold_bars 동안 박스 내부 유지
5. **Dynamic Params**: 변동성에 맞는 TP/SL 계산
6. **비용 검증**: TP > round_trip_cost
7. **Position Entry**: 20% 배분 (Phase C) 또는 3% 분할 (Phase A/B)
8. **Exit 관리**: STOP → TP1(50%) → TP2(나머지) → TIME_EXIT

---

## 10. 파일 참조

| 파일 | 경로 | 핵심 기능 |
|------|------|----------|
| `broker_sim.py` | `_01_crypto/001_binance/001_spot/001_backtest/` | 메인 시뮬레이션 루프 |
| `navigation.py` | `_06_engine/` | Gate/Confidence 계산 |
| `dynamic_params.py` | `_03_common/_02_features/` | 동적 TP/SL 계산 |
| `events.py` | `_03_common/_03_wyckoff/` | Spring/UTAD 감지 |
| `phases.py` | `_03_common/_03_wyckoff/` | Phase A/B/C/D/E 감지 |
| `box.py` | `_03_common/_03_wyckoff/` | 박스 범위 계산 |
| `types.py` | `_03_common/_01_core/` | 타입 정의 (Theta, BacktestConfig) |
| `backtest.py` | `_00_config/` | 백테스트 설정 |

---

*문서 끝*
