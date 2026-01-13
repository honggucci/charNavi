# WPCN Session Backup - 2026-01-13

## Session Info
- **Date**: 2026-01-13
- **Topic**: HMM Risk Filter + D-lite ATR SL/TP 실험
- **Status**: 버그 발견 및 수정 완료

---

## 1. 세션 목표

BTC 선물 15분봉 평균회귀 트레이딩 시스템에서:
1. CHAMPION v1.0 (HMM 필터) 재현성 확인
2. D-lite (ATR 기반 SL/TP) 효과 검증
3. 수익 루프 달성 여부 확인

---

## 2. 핵심 발견사항

### 2.1 치명적 버그 발견

**`apply_hmm_filter()` 함수 버그**:
- 기존: 각 entry마다 "이후 첫 exit pnl" 참조 → **중복 계산**
- 누적 포지션(10 entries → 1 exit)에서 같은 exit pnl을 10번 계산
- 결과: 완전히 틀린 PnL 수치

**버그 수정**:
```python
# 기존 (버그)
for idx, row in entries.iterrows():
    later_exits = exits[(exits["time"] > t) & (exits["side"] == side)]
    original_pnl = later_exits.iloc[0].get("pnl", 0)  # 항상 같은 exit 참조!

# 수정
for idx, row in exits.iterrows():  # Exit 기준으로 순회
    original_pnl = row.get("pnl", 0)  # 각 exit의 실제 pnl
```

### 2.2 결과 변화

| 연도 | 버그 있을 때 | 버그 수정 후 |
|------|-------------|-------------|
| 2023 | -$167 | **+$1,934** |
| 2024 | -$7,975 | **-$5,332** |
| Trades 2023 | 159 | **936** |
| Trades 2024 | 423 | **673** |

**이전 "수익 루프" 선언은 완전히 틀렸다!**

### 2.3 청산 수수료 누락 확인

| 항목 | 포함 여부 |
|------|----------|
| 진입 수수료 (4bps) | ✅ 포함 |
| 진입 슬리피지 (3bps+) | ✅ 포함 |
| **청산 수수료 (4bps)** | ❌ **미포함** |
| 청산 슬리피지 (5bps) | ✅ 포함 |
| 펀딩비 | ✅ 포함 |

### 2.4 Same-bar TP/SL 충돌

- SL 우선 처리 (보수적) → 문제 없음
- OHLC 순서 고려 안 함 (단순화)

---

## 3. D-lite 실험 결과

### 3.1 ATR 설정 효과 측정 불가

버그 수정 후 테스트:
```
CHAMPION: 2023 +$1,934, 2024 -$5,332
ATR_2.0x3.0: 2023 $0 (0 trades), 2024 $0 (0 trades)
```

원인: HMM 필터가 너무 엄격
- `long_permit_states = ('markup',)` → markup 상태에서만 롱 허용
- ATR 트레이드 대부분이 markdown/distribution 상태에서 발생 → 전부 필터링

### 3.2 이전 +$22k "개선"은 버그의 산물

- Entry 기준 중복 계산 버그로 인해 발생
- 실제 ATR 효과 아님

---

## 4. 수정된 파일

### 4.1 hmm_integrated_backtest.py
- `apply_hmm_filter()`: Entry 기준 → Exit 기준으로 변경
- 버전: v2.6.11

### 4.2 futures_backtest_v3.py
- `FuturesConfigV3`: ATR SL/TP 파라미터 추가
  - `use_atr_sltp: bool = False`
  - `atr_sl_mult: float = 1.5`
  - `atr_tp_mult: float = 2.5`
- `AccumulatedPositionV3.add_entry()`: ATR 기반 SL/TP 로직 추가

### 4.3 생성된 실험 스크립트
- `verify_champion_baseline.py` - CHAMPION 재현성 검증
- `run_dlite_quick.py` - D-lite 빠른 테스트
- `analyze_dlite_winner.py` - 승자 상세 분석
- `verify_dlite_skeptic.py` - 비판적 검증
- `debug_2023_zero_loss.py` - 2023 손실=0 디버깅
- `debug_champion_pnl.py` - CHAMPION PnL 디버깅
- `debug_atr_trades.py` - ATR 트레이드 디버깅
- `debug_hmm_filter.py` - HMM 필터 디버깅
- `inspect_trades_df.py` - trades_df 구조 검사
- `policy_v1_1_challenger.py` - CHALLENGER v1.1 설정 (무효)

---

## 5. 다음 단계

1. **청산 수수료 추가** - LiquidationModel 또는 simulate_futures_v3에 fee_bps 적용
2. **HMM 필터 완화 테스트** - ATR 효과 측정 가능하게
3. **Entry-Exit 매칭 로직 검증** - 다른 버그 가능성 확인

---

## 6. 핵심 교훈

1. **"너무 좋은 결과"는 의심 대상** - 버그일 확률 높음
2. **백테스트 엔진 검증 필수** - 비용 처리, 체결 로직 확인
3. **비판적 페르소나** 덕분에 치명적 버그 발견
4. 버그 안 잡았으면 실거래에서 큰 손실 발생했을 수 있음

---

## 7. Red Flags (이전 결과에서)

버그 수정 전 의심 포인트:
- [!] 2024 승률 급등: 35.9% → 72.3% (+36.4%p)
- [!] 2023 트레이드 63% 감소: 159 → 58
- [!] 2023 Worst20 = $0 (손실 트레이드 없음?)
- [!] 2023 Avg R = 1984.5 (비정상적으로 높음)

모두 **버그 증상**이었음.

---

## 8. 현재 상태 (Session 1 종료)

```
CHAMPION v1.0 (버그 수정 후):
  2023: +$1,934 (손실 아니라 수익!)
  2024: -$5,332
  총합: -$3,398

D-lite ATR 효과:
  측정 불가 (HMM 필터가 모든 트레이드 차단)
```

**결론**: Exit 최적화 실험을 위해 HMM 필터 완화 또는 별도 테스트 환경 필요.

---

## 9. Session 2: v2.6.14 15분봉 지정가 백테스터 (PM 세션)

### 9.1 문제점 진단

이전 세션에서 발견된 핵심 문제:
- **5분봉 ATR 스파이크**: 5분봉 ATR로 SL/TP 설정 → 15분봉 체결 시 SL 94% 히트
- **시장가 진입**: 스파이크에 취약
- **Time-stop 미작동**: 0% (8시간 보유 제한 무의미)

### 9.2 v2.6.14 아키텍처 변경

**핵심 변경사항**:
```
- 타임프레임: 5분봉 → 15분봉 전용
- 진입: 시장가 → 지정가 (Pending Order)
- 체결 대기: 4봉(1시간) 미체결 시 취소
- SL/TP: 시장가 → 지정가 (체결 즉시 설정)
- Max Hold: 32봉(8시간) 펀딩비 회피
```

**새 파일**: `src/wpcn/_04_execution/futures_backtest_15m.py`
- `FuturesConfig15m` dataclass
- `PendingOrder` dataclass (지정가 주문)
- `Position15m` dataclass
- `simulate_futures_15m()` 함수
- `run_15m_backtest_from_5m()` 헬퍼 함수

### 9.3 PNL 계산 버그 수정

**버그**: Short 포지션 qty가 음수로 저장되어 PnL 부호 반전
```python
# 수정 전
return (self.entry_price - exit_price) * self.qty  # qty 음수 → PnL 반전

# 수정 후
return (self.entry_price - exit_price) * abs(self.qty)  # abs() 사용
```

### 9.4 Transition Cooldown 병목 발견

**상세 통계 추가 후 발견**:
```
signals_generated:        22,536
hmm_gate_blocked:         22,530 (99.97%)
blocked_by_transition:    22,324 (98.9%)  ← 주범!
blocked_by_short_permit:      49 (0.2%)
blocked_by_long_permit:      157 (0.7%)
```

**원인**: `transition_delta=0.20`이 너무 민감
- delta = posterior 최대 변화량
- 거의 매 봉마다 delta > 0.20 → cooldown 연속 발생

### 9.5 Policy v1.2 Relaxed Cooldown

**새 policy 파일**: `policy_v1_2_relaxed_cooldown.py`
```python
# v1.1 (기존)
transition_delta: 0.20
cooldown_bars: 2

# v1.2 (완화)
transition_delta: 0.40  # 더 큰 변화만 감지
cooldown_bars: 1        # 대기 시간 단축
```

### 9.6 최종 WFO 결과 (v1.2)

| 버전 | 총 Test PnL | 2023 Trades | 2024 Trades | Selected |
|------|-------------|-------------|-------------|----------|
| v1 (5분봉) | -$31,482 | - | - | - |
| v1.1 (delta=0.20) | -$173 | 3 | 18 | D_2.5x3.5 |
| **v1.2 (delta=0.40)** | **+$991** | **72** | **143** | **B_2.0x3.0** |

**연도별 상세**:
```
2023: +$71.45, 72 trades, TP 27.8%, SL 50.0%, Time-stop 22.2%
2024: +$919.58, 143 trades, TP 33.6%, SL 46.9%, Time-stop 19.6%
```

### 9.7 v1 대비 개선

| 지표 | v1 | v1.2 | 변화 |
|------|-----|------|------|
| Total PnL | -$31,482 | +$991 | **+$32,473** |
| SL Hit Rate | 94% | 46-50% | **-44~48%p** |
| Time-stop Rate | 0% | 20-22% | **+20~22%p** |
| Invariant | - | OK | 검증 통과 |

### 9.8 핵심 교훈

1. **분모 확인 필수**: block_rate 99%가 "진짜 문제"인지 상세 통계로 확인
2. **단일 병목 해결 효과**: transition cooldown 완화 하나로 손실 → 수익 전환
3. **표본 수 중요**: 3-18개 → 72-143개로 증가해야 통계적 의미

### 9.9 생성/수정 파일

**새 파일**:
- `src/wpcn/_04_execution/futures_backtest_15m.py` - 15분봉 지정가 백테스터
- `policy_v1_2_relaxed_cooldown.py` - Relaxed Cooldown policy

**수정 파일**:
- `run_dlite_wfo_v2.py` - 상세 통계 출력 추가, v1.2 policy 적용

---

## 10. 현재 상태 (Session 2 종료)

```
v1.2 RELAXED COOLDOWN (15분봉 지정가):
  2023: +$71 (72 trades)
  2024: +$920 (143 trades)
  총합: +$991

Selected Config: B_2.0x3.0 (ATR SL 2.0배, TP 3.0배)
Invariant: OK (PnL 검증 통과)
```

**다음 단계**:
1. Short permit 완화 테스트 (P(markdown) 0.60 → 0.55)
2. Long permit 추가 완화 테스트
3. 다른 D-lite 설정 탐색 (A, C, D)