# WPCN 세션 백업: HMM Risk Filter v3 배포
**날짜**: 2026-01-11
**태그**: v2026.01.11-hmmrisk-v3
**커밋**: 9829c08

---

## 1. 세션 개요

### 목표
악마의 변호인 점검표(Devil's Advocate Checklist) 기반으로 HMM Risk Filter v3 완성 및 배포

### 완료 항목
1. 수학 불변식 단위 테스트 (29개 테스트 통과)
2. Kupiec VaR 백테스트 구현
3. Adaptive Clip 메커니즘 구현
4. Shadow 모드 로깅 필드 추가
5. markdown 페널티 버그 수정
6. Git 태그 및 커밋

---

## 2. 핵심 구현 내용

### 2.1 수학 불변식 테스트 (TestMathInvariants)

```python
# 테스트 항목
- test_expected_var_never_zero: expected_var > ε 강제
- test_expected_var_with_near_zero_probabilities: 극단 확률에서도 유한값
- test_size_mult_monotonicity: expected_var ↑ → size_mult ↓
- test_probability_sum_equals_one: prob.sum() ≈ 1.0
- test_normalized_entropy_bounds: H ∈ [0, 1]
- test_var_values_are_negative: VaR < 0
```

### 2.2 Kupiec VaR 백테스트

```python
@dataclass
class KupiecResult:
    n: int              # 총 관측 수
    breaches: int       # VaR 초과 수
    alpha: float        # 예측 신뢰수준
    p_hat: float        # 실제 breach 비율
    lr_uc: float        # Likelihood Ratio 통계량
    p_value: float      # p-value
    pass_test: bool     # 테스트 통과 여부

def kupiec_uc(pnl, var, alpha, side="both") -> KupiecResult:
    """
    Kupiec Unconditional Coverage Test
    LR_uc = -2[log L(α) - log L(p̂)] ~ χ²(1)

    합격 기준:
    - p_value > 0.05 (5% 유의수준)
    - |p_hat - α| ≤ 0.5*α
    """
```

### 2.3 Adaptive Clip

```python
@dataclass
class AdaptiveClipConfig:
    sigma_ref: float = 1.0      # 참조 변동성
    kappa: float = 0.5          # 엔트로피 계수
    beta: float = 0.75          # 변동성 계수
    floor_cap: float = 1.0      # 하한
    ceil_cap: float = 1.5       # 상한

def adaptive_upper_cap(sigma_t, sigma_ref, entropy_t, kappa=0.5, beta=0.75,
                       floor_cap=1.0, ceil_cap=1.5) -> float:
    """
    Dual-axis 동적 상한:
    vol_cap = 1.25 * (σ_ref / σ_t)^β
    ent_cap = 1.5 - κ * H
    u_t = min(vol_cap, ent_cap)
    """

def sized_position_adaptive(...) -> Tuple[float, float, str]:
    """Returns: (size_mult, upper_cap, clip_hit)"""
```

### 2.4 Shadow 로깅 필드

```python
@dataclass
class TradeDecision:
    action: str              # ALLOW, NO_TRADE, COOLDOWN, SHORT_ONLY, LONG_ONLY
    size_mult: float
    hmm_state: str
    expected_var: float
    # Shadow fields
    hmm_probs: Optional[List[float]] = None
    delta_at_entry: Optional[float] = None
    clip_hit: str = "none"   # none, min, max
    k_transition_applied: float = 1.0

    def to_log_dict(self) -> dict:
        """JSON 직렬화 가능한 딕셔너리 반환"""
```

### 2.5 버그 수정: markdown 페널티

```python
# 수정 전 (버그)
size_mult *= cfg.markdown_size_penalty  # 하한 보장 깨짐

# 수정 후
size_mult = np.clip(
    size_mult * cfg.markdown_size_penalty,
    cfg.size_min,
    cfg.size_max
)
```

---

## 3. 배포 설정

### 3.1 Ship Config

```yaml
engine:
  policy: PolicyA  # State Machine
  transition:
    k_transition: 1.5
    cooldown:
      delta_threshold: 0.20
      bars: 2
  sizing:
    mode: adaptive_clip
    base: 5.0
    lower_cap: 0.25
    upper_cap_bounds: [1.0, 1.5]
    beta_vol: 0.75
    kappa_entropy: 0.5
  var:
    alpha: 0.01
    acceptance_band: 0.5
  logging:
    shadow_fields: true
```

### 3.2 하드 스톱

| 조건 | 액션 |
|------|------|
| Kupiec α=1% 3일 연속 실패 | upper_cap = 1.0 강제 |
| clip_hit_upper > 40% 3일 | β/κ += 0.1 |
| ΔPnL < 0 2주 연속 | PolicyB로 롤백 |

### 3.3 롤아웃 단계

| 단계 | 기간 | 사이징 | 중단 조건 |
|------|------|--------|-----------|
| Shadow | 3일 | 0% (시뮬) | Kupiec FAIL 또는 clip_hit > 40% |
| Canary | 7일 | 10-20% | ΔPnL < 0 (5일 EMA) |
| Ramp | 14일 | 50% | Calmar 하락 |
| Full | 지속 | 100% | 일반 모니터링 |

---

## 4. 테스트 결과

```
============================= test session starts =============================
collected 29 items

tests/test_hmm_risk_filter_invariants.py::TestMathInvariants::test_expected_var_never_zero PASSED
tests/test_hmm_risk_filter_invariants.py::TestMathInvariants::test_expected_var_with_near_zero_probabilities PASSED
tests/test_hmm_risk_filter_invariants.py::TestMathInvariants::test_size_mult_monotonicity PASSED
tests/test_hmm_risk_filter_invariants.py::TestMathInvariants::test_probability_sum_equals_one PASSED
tests/test_hmm_risk_filter_invariants.py::TestMathInvariants::test_normalized_entropy_bounds PASSED
tests/test_hmm_risk_filter_invariants.py::TestMathInvariants::test_var_values_are_negative PASSED
tests/test_hmm_risk_filter_invariants.py::TestCooldownLogic::test_cooldown_triggers_on_transition PASSED
tests/test_hmm_risk_filter_invariants.py::TestCooldownLogic::test_cooldown_counter_blocks_trade PASSED
tests/test_hmm_risk_filter_invariants.py::TestCooldownLogic::test_no_cooldown_on_small_transition PASSED
tests/test_hmm_risk_filter_invariants.py::TestSizingLogic::test_clip_bounds PASSED
tests/test_hmm_risk_filter_invariants.py::TestSizingLogic::test_markdown_size_penalty PASSED
tests/test_hmm_risk_filter_invariants.py::TestEdgeCases::test_first_bar_no_prev_posterior PASSED
tests/test_hmm_risk_filter_invariants.py::TestEdgeCases::test_all_states_produce_valid_decision PASSED
tests/test_hmm_risk_filter_invariants.py::TestShadowLogging::test_shadow_fields_included_when_enabled PASSED
tests/test_hmm_risk_filter_invariants.py::TestShadowLogging::test_shadow_fields_none_when_disabled PASSED
tests/test_hmm_risk_filter_invariants.py::TestShadowLogging::test_to_log_dict_json_serializable PASSED
tests/test_hmm_risk_filter_invariants.py::TestShadowLogging::test_transition_k_applied_on_cooldown PASSED
tests/test_hmm_risk_filter_invariants.py::TestShadowLogging::test_clip_hit_detection PASSED
tests/test_hmm_risk_filter_invariants.py::TestKupiecVaR::test_kupiec_nominal_rate PASSED
tests/test_hmm_risk_filter_invariants.py::TestKupiecVaR::test_kupiec_excess_breaches PASSED
tests/test_hmm_risk_filter_invariants.py::TestKupiecVaR::test_kupiec_too_few_breaches PASSED
tests/test_hmm_risk_filter_invariants.py::TestKupiecVaR::test_kupiec_result_structure PASSED
tests/test_hmm_risk_filter_invariants.py::TestAdaptiveClip::test_adaptive_cap_monotonic_volatility PASSED
tests/test_hmm_risk_filter_invariants.py::TestAdaptiveClip::test_adaptive_cap_monotonic_entropy PASSED
tests/test_hmm_risk_filter_invariants.py::TestAdaptiveClip::test_adaptive_cap_bounds PASSED
tests/test_hmm_risk_filter_invariants.py::TestAdaptiveClip::test_sized_position_adaptive_basic PASSED
tests/test_hmm_risk_filter_invariants.py::TestAdaptiveClip::test_sized_position_clip_detection PASSED
tests/test_hmm_risk_filter_invariants.py::TestAdaptiveClip::test_should_trade_filter_adaptive_integration PASSED
tests/test_hmm_risk_filter_invariants.py::TestAdaptiveClip::test_adaptive_vs_static_comparison PASSED

============================= 29 passed in 2.05s ==============================
```

---

## 5. 성과 지표 요약

### Walk-Forward 검증 결과

| Train | Test | Baseline | Filtered | Improvement |
|-------|------|----------|----------|-------------|
| 2021-2022 | 2023 | -X | -Y | +20.8% |
| 2022-2023 | 2024 | -X | -Y | +40.2% |
| 2021-2022-2023 | 2024 | -X | -Y | +30.5% |

**평균 개선율**: +30.5%
**모든 기간 양수**: ✅

### Policy A vs B 비교

- Policy A (State Machine): -17.76%
- Policy B (Baseline): -20.50%
- **차이**: +2.74%p

---

## 6. 다음 세션 프롬프트

### 6.1 Shadow 모드 모니터링 시작

```
WPCN HMM Risk Filter v3가 v2026.01.11-hmmrisk-v3 태그로 배포되었습니다.

현재 상태:
- 테스트: 29/29 통과
- 설정: PolicyA, K_TRANSITION=1.5, delta>0.20→2봉, Adaptive Clip (β=0.75, κ=0.5)

Shadow 모드 3일차입니다. 다음을 확인해주세요:
1. Kupiec VaR 백테스트 결과 (α=1% 기준)
2. clip_hit_upper 비율
3. 일별 ΔPnL (A vs B)

문제 발생 시 하드 스톱 적용:
- Kupiec FAIL 3일 → cap=1.0
- clip_hit>40% 3일 → β/κ +=0.1
- ΔPnL<0 2주 → rollback
```

### 6.2 다음 개발 단계

```
HMM Risk Filter v3 배포 완료 후 다음 우선순위:

1. Range 이벤트 전용 진입 (Spring/UTAD 테스트만) + ATR 트레일링 청산
2. Rally run 모듈 (ret_30d 게이트 + 지속성 피처)
3. 딜레이 최적화 (dwell×confirm 히트맵으로 중앙값 94→≤72봉)
4. 일반화 (ETH + 2개 알트, 2018-2020 및 2025 데이터)

현재 코드 위치:
- run_step9_hmm_risk_filter.py
- tests/test_hmm_risk_filter_invariants.py

시작하려면: python -m pytest tests/test_hmm_risk_filter_invariants.py -v
```

### 6.3 긴급 롤백 명령

```bash
# PolicyB + 정적 클립으로 롤백
cd /c/Users/hahonggu/Desktop/coin_master/projects/wpcn-backtester-cli-noflask

# 설정 변경 (수동)
# engine.policy = "PolicyB"
# engine.sizing.mode = "static_clip"
# engine.sizing.upper_cap_bounds = [1.0, 1.25]

# 확인
git log --oneline -3
git tag -l "v2026*"
```

---

## 7. 파일 목록

### 신규 생성
- `run_step9_hmm_risk_filter.py` - HMM Risk Filter v3 메인 코드
- `tests/test_hmm_risk_filter_invariants.py` - 29개 단위 테스트
- `docs/wpcn_session_20260111_hmm_risk_filter_v3.md` - 이 문서

### 수정됨
- (이전 세션에서 수정된 파일들)

---

## 8. 연락처 및 참조

- **프로젝트**: WPCN Backtester CLI
- **경로**: `C:\Users\hahonggu\Desktop\coin_master\projects\wpcn-backtester-cli-noflask`
- **Git 태그**: `v2026.01.11-hmmrisk-v3`
- **커밋**: `9829c08`

---

*이 문서는 2026-01-11 Claude Code CLI 세션에서 자동 생성되었습니다.*
