# 현물 vs 선물 백테스트 비교 대화 - 2025-12-31

## 세션 요약

폴더 구조 정리 이후 현물(broker_sim)과 선물(V8) 백테스트 엔진의 차이점을 분석했습니다.

---

## 1. 주요 질문 및 답변

### Q1: 현물/선물 소스코드 차이가 뭐야?

| 구분 | 현물 (broker_sim) | 선물 (V8) |
|------|-------------------|-----------|
| **파일** | `_04_execution/broker_sim.py` | `_04_execution/futures_backtest_v8.py` |
| **레버리지** | 1x 고정 (spot_mode로 시뮬레이션 가능) | 1x ~ 10x 설정 가능 |
| **마진 시스템** | 없음 | 격리 마진, 청산가 계산 |
| **포지션 방향** | 롱/숏 모두 가능 | 롱/숏 모두 가능 |

### Q2: "전액 매수"는 뭐야? 현물도 롱숏 둘다 있는거 아니야?

**전액 매수**: 잘못된 설명이었음. 실제로는 둘 다 **피보나치 그리드 분할 진입** 사용.

**현물 롱/숏**: 맞음. `broker_sim.py`에도 `ACCUM_SHORT` 로직이 구현되어 있음.

### Q3: 진입 방식에서 현물/선물 정확한 차이점?

**공통점**:
- 둘 다 `detect_phase_signals()` 사용
- 피보나치 그리드 분할 진입 동일

```python
FIB_LEVELS = {
    'fib_236': 0.236,
    'fib_382': 0.382,
    'fib_500': 0.500,
    'fib_618': 0.618,
    'fib_786': 0.786
}
```

**차이점**:

| 항목 | broker_sim | V8 |
|------|------------|-----|
| **진입 소스** | Wyckoff Phase만 | Wyckoff Phase + MTF 신호 분석 |
| **추가 진입** | Phase C 진입 시 20%만 추가 | MTF 스코어 기반 별도 진입 |
| **레버리지 적용** | `position_pct` 그대로 | `position_pct * leverage` |
| **신호 필터** | Wyckoff Phase | RSI 다이버전스 + ZigZag + Stoch RSI + 확률 모델 |

### Q4: 선물로직엔 와이코프가 없어?

**틀림. V8에도 Wyckoff가 있음!**

V8의 Wyckoff 사용:

1. **피보나치 그리드**: `detect_phase_signals()` → Wyckoff Phase 기반
2. **MTF Wyckoff 분석**:
   - 4H Wyckoff: `_analyze_wyckoff(df_4h, ...)`
   - 1H Wyckoff: Spring/Upthrust 감지
3. **확률 모델 반영**:
   ```python
   if wyckoff_phase in ['accumulation', 're_accumulation']:
       drift += 0.02  # 상승 바이어스
   elif wyckoff_phase in ['distribution', 're_distribution']:
       drift -= 0.02  # 하락 바이어스
   ```

**최종 비교**:

| 항목 | broker_sim | V8 |
|------|------------|-----|
| **Wyckoff Phase 감지** | O | O |
| **피보나치 그리드** | O | O |
| **MTF Wyckoff (4H/1H)** | X | O |
| **Spring/Upthrust 감지** | O | O |
| **확률 모델 + Wyckoff** | X | O |

---

## 2. 사용자 의도

> "환경변수 값이 spot일때는 현물로직을, future일때는 선물로직을 실행하게끔 코드설정을 할꺼니까 상관없음."

→ 향후 환경변수 기반 모드 전환 구현 예정

---

## 3. 파일 참조

- [broker_sim.py](src/wpcn/_04_execution/broker_sim.py) - 현물 백테스트
- [futures_backtest_v8.py](src/wpcn/_04_execution/futures_backtest_v8.py) - 선물 백테스트 (V8)
- [phases.py](src/wpcn/_03_common/_03_wyckoff/phases.py) - Wyckoff Phase 감지

---

*생성일: 2025-12-31*
*브랜치: QWAS-SYS-15M-V1*
