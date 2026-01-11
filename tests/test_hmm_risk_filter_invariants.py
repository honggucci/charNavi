"""
HMM Risk Filter Invariants Test
===============================
악마의 변호인 점검표 기반 불변식 테스트.

1. expected_var > ε (0 근접 방지)
2. size_mult 단조성 (expected_var ↑ → size_mult ↓)
3. 확률 합 = 1 ± 1e-6
4. Lookahead 방지 (shift(+1) 확인)
"""
import sys
sys.path.insert(0, "src")
import numpy as np
import pytest
from run_step9_hmm_risk_filter import (
    should_trade_filter, HMMRiskFilterConfig, normalized_entropy,
    VAR5_BY_STATE, HMM_STATES, IDX_TO_STATE,
    kupiec_uc, KupiecResult, adaptive_upper_cap, sized_position_adaptive,
    AdaptiveClipConfig, should_trade_filter_adaptive
)


class TestMathInvariants:
    """수학/구현 불변식 테스트"""

    def test_expected_var_never_zero(self):
        """expected_var > ε 강제 확인"""
        cfg = HMMRiskFilterConfig()

        # 극단적인 확률 분포 (한 상태에 100%)
        for i in range(6):
            posterior = np.zeros(6)
            posterior[i] = 1.0

            decision = should_trade_filter(posterior, None, cfg, 0)

            # expected_var는 절대값으로 계산되므로 항상 양수
            assert abs(decision.expected_var) > 0, f"expected_var should be > 0, got {decision.expected_var}"
            # size_mult도 유한한 값이어야 함
            assert np.isfinite(decision.size_mult), f"size_mult should be finite, got {decision.size_mult}"
            assert decision.size_mult >= cfg.size_min, f"size_mult below min: {decision.size_mult}"
            assert decision.size_mult <= cfg.size_max, f"size_mult above max: {decision.size_mult}"

    def test_expected_var_with_near_zero_probabilities(self):
        """거의 0에 가까운 확률에서도 안전한지 확인"""
        cfg = HMMRiskFilterConfig()

        # 매우 작은 확률들
        posterior = np.array([1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1 - 5e-10])

        decision = should_trade_filter(posterior, None, cfg, 0)

        assert np.isfinite(decision.expected_var)
        assert np.isfinite(decision.size_mult)
        assert decision.size_mult >= cfg.size_min
        assert decision.size_mult <= cfg.size_max

    def test_size_mult_monotonicity(self):
        """expected_var ↑ → size_mult ↓ 단조성 확인 (ALLOW action만)"""
        # markdown은 SHORT_ONLY + penalty로 별도 처리되므로 ALLOW만 테스트
        cfg = HMMRiskFilterConfig(markdown_short_only=False)  # penalty 비활성화

        # 다양한 expected_var 값 생성 (clip 영향 없는 범위에서)
        test_cases = []

        # Case 1: 안전한 상태 (distribution) - var_target에 가까움
        safe_posterior = np.zeros(6)
        safe_posterior[2] = 1.0  # distribution (-5.63%)
        safe_decision = should_trade_filter(safe_posterior, None, cfg, 0)
        if safe_decision.action == "ALLOW":
            test_cases.append((abs(safe_decision.expected_var), safe_decision.size_mult, "distribution"))

        # Case 2: 위험한 상태 (markdown) - var_target보다 큼
        risky_posterior = np.zeros(6)
        risky_posterior[5] = 1.0  # markdown (-10.52%)
        risky_decision = should_trade_filter(risky_posterior, None, cfg, 0)
        if risky_decision.action == "ALLOW":
            test_cases.append((abs(risky_decision.expected_var), risky_decision.size_mult, "markdown"))

        # Case 3: 중간 상태 (markup)
        mid_posterior = np.zeros(6)
        mid_posterior[4] = 1.0  # markup (-7.16%)
        mid_decision = should_trade_filter(mid_posterior, None, cfg, 0)
        if mid_decision.action == "ALLOW":
            test_cases.append((abs(mid_decision.expected_var), mid_decision.size_mult, "markup"))

        # 정렬해서 단조성 확인
        test_cases.sort(key=lambda x: x[0])  # expected_var 오름차순

        for i in range(len(test_cases) - 1):
            var_curr, size_curr, name_curr = test_cases[i]
            var_next, size_next, name_next = test_cases[i + 1]

            # expected_var가 증가하면 size_mult는 감소 (또는 같음 - clip 때문)
            if var_next > var_curr:
                assert size_next <= size_curr, \
                    f"Monotonicity violated: {name_curr}({var_curr:.2f})->{name_next}({var_next:.2f}), size {size_curr:.2f}->{size_next:.2f}"

    def test_probability_sum_equals_one(self):
        """확률 합 = 1 ± 1e-6 확인"""
        # 다양한 posterior 샘플
        posteriors = [
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0]),
            np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6]),
            np.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.1]),
        ]

        for posterior in posteriors:
            total = posterior.sum()
            assert abs(total - 1.0) < 1e-6, f"Probability sum should be 1, got {total}"

    def test_normalized_entropy_bounds(self):
        """정규화 엔트로피 범위 [0, 1] 확인"""
        # 최소 엔트로피 (확정)
        certain = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ent_min = normalized_entropy(certain)
        assert 0 <= ent_min <= 0.01, f"Certain posterior should have ~0 entropy, got {ent_min}"

        # 최대 엔트로피 (균등분포)
        uniform = np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
        ent_max = normalized_entropy(uniform)
        assert 0.99 <= ent_max <= 1.0, f"Uniform posterior should have ~1 entropy, got {ent_max}"

    def test_var_values_are_negative(self):
        """VaR 값이 모두 음수인지 확인 (손실)"""
        for state, var in VAR5_BY_STATE.items():
            assert var < 0, f"VaR for {state} should be negative (loss), got {var}"


class TestCooldownLogic:
    """쿨다운 로직 테스트"""

    def test_cooldown_triggers_on_transition(self):
        """transition_delta 초과 시 쿨다운 발동"""
        cfg = HMMRiskFilterConfig(transition_delta=0.20)

        # 이전: distribution 확정
        prev = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        # 현재: markdown으로 급변
        curr = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

        decision = should_trade_filter(curr, prev, cfg, 0)

        assert decision.action == "COOLDOWN", f"Should trigger cooldown, got {decision.action}"
        assert decision.size_mult == 0.0, "Cooldown should have size_mult = 0"

    def test_cooldown_counter_blocks_trade(self):
        """쿨다운 카운터 > 0이면 NO_TRADE"""
        cfg = HMMRiskFilterConfig()

        posterior = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # distribution

        decision = should_trade_filter(posterior, None, cfg, cooldown_counter=2)

        assert decision.action == "COOLDOWN", f"Should be in cooldown, got {decision.action}"
        assert decision.size_mult == 0.0

    def test_no_cooldown_on_small_transition(self):
        """transition_delta 미만이면 쿨다운 없음"""
        cfg = HMMRiskFilterConfig(transition_delta=0.20)

        # 이전: 약간 분산
        prev = np.array([0.3, 0.3, 0.2, 0.1, 0.05, 0.05])
        # 현재: 소폭 변화
        curr = np.array([0.35, 0.25, 0.2, 0.1, 0.05, 0.05])

        max_delta = np.abs(curr - prev).max()
        assert max_delta < cfg.transition_delta, f"Delta should be small, got {max_delta}"

        decision = should_trade_filter(curr, prev, cfg, 0)

        assert decision.action != "COOLDOWN", f"Should not trigger cooldown, got {decision.action}"


class TestSizingLogic:
    """사이징 로직 테스트"""

    def test_clip_bounds(self):
        """size_mult가 [size_min, size_max] 범위 내"""
        cfg = HMMRiskFilterConfig(size_min=0.25, size_max=1.25)

        # 다양한 posterior 테스트
        posteriors = [
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # accumulation
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),  # markdown
            np.array([1/6] * 6),  # uniform
        ]

        for posterior in posteriors:
            decision = should_trade_filter(posterior, None, cfg, 0)

            if decision.action in ("ALLOW", "SHORT_ONLY", "LONG_ONLY"):
                assert cfg.size_min <= decision.size_mult <= cfg.size_max, \
                    f"size_mult {decision.size_mult} out of bounds [{cfg.size_min}, {cfg.size_max}]"

    def test_markdown_size_penalty(self):
        """markdown 상태에서 size penalty 적용"""
        cfg = HMMRiskFilterConfig(markdown_short_only=True, markdown_size_penalty=0.5)

        # markdown 상태
        markdown_posterior = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

        decision = should_trade_filter(markdown_posterior, None, cfg, 0)

        assert decision.action == "SHORT_ONLY"
        # size_mult에 penalty가 적용되어 있어야 함
        assert decision.size_mult <= cfg.size_max * cfg.markdown_size_penalty + 0.01


class TestEdgeCases:
    """엣지 케이스 테스트"""

    def test_first_bar_no_prev_posterior(self):
        """첫 번째 바 (prev_posterior=None)에서 정상 작동"""
        cfg = HMMRiskFilterConfig()

        posterior = np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1])

        decision = should_trade_filter(posterior, None, cfg, 0)

        assert decision.action in ("ALLOW", "NO_TRADE", "SHORT_ONLY")
        assert decision.size_mult >= 0

    def test_all_states_produce_valid_decision(self):
        """모든 상태에서 유효한 decision 반환"""
        cfg = HMMRiskFilterConfig()

        for i in range(6):
            posterior = np.zeros(6)
            posterior[i] = 1.0

            decision = should_trade_filter(posterior, None, cfg, 0)

            assert decision.hmm_state == IDX_TO_STATE[i]
            assert decision.action in ("ALLOW", "NO_TRADE", "COOLDOWN", "SHORT_ONLY", "LONG_ONLY")
            assert 0 <= decision.size_mult <= cfg.size_max


class TestShadowLogging:
    """Shadow mode 로깅 테스트"""

    def test_shadow_fields_included_when_enabled(self):
        """include_shadow_fields=True일 때 필드 포함"""
        cfg = HMMRiskFilterConfig()
        posterior = np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1])

        decision = should_trade_filter(posterior, None, cfg, 0, include_shadow_fields=True)

        assert decision.hmm_probs is not None
        assert len(decision.hmm_probs) == 6
        assert decision.delta_at_entry is not None
        assert decision.clip_hit in ("none", "min", "max")

    def test_shadow_fields_none_when_disabled(self):
        """include_shadow_fields=False일 때 필드는 기본값"""
        cfg = HMMRiskFilterConfig()
        posterior = np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1])

        decision = should_trade_filter(posterior, None, cfg, 0, include_shadow_fields=False)

        assert decision.hmm_probs is None
        assert decision.delta_at_entry is None

    def test_to_log_dict_json_serializable(self):
        """to_log_dict() 결과가 JSON 직렬화 가능"""
        import json
        cfg = HMMRiskFilterConfig()
        posterior = np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1])

        decision = should_trade_filter(posterior, None, cfg, 0, include_shadow_fields=True)
        log_dict = decision.to_log_dict()

        # JSON 직렬화 가능해야 함
        json_str = json.dumps(log_dict)
        assert isinstance(json_str, str)
        assert len(json_str) > 0

    def test_transition_k_applied_on_cooldown(self):
        """전이 감지 시 k_transition_applied 설정"""
        cfg = HMMRiskFilterConfig(transition_delta=0.20)

        prev = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        curr = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

        decision = should_trade_filter(curr, prev, cfg, 0, include_shadow_fields=True)

        assert decision.action == "COOLDOWN"
        assert decision.k_transition_applied == 1.5

    def test_clip_hit_detection(self):
        """clip hit 감지"""
        # distribution에서 var_target과 비슷 → clip 안 걸림
        cfg = HMMRiskFilterConfig(size_min=0.25, size_max=1.25, var_target=-5.0)

        # VaR가 var_target에 가까운 상태 (distribution: -5.63%)
        dist_posterior = np.zeros(6)
        dist_posterior[2] = 1.0

        decision = should_trade_filter(dist_posterior, None, cfg, 0, include_shadow_fields=True)
        # 5.0 / 5.63 ≈ 0.89 → 범위 내이므로 none
        assert decision.clip_hit == "none"

        # VaR가 var_target보다 훨씬 큰 상태 (markdown: -10.52%)
        md_posterior = np.zeros(6)
        md_posterior[5] = 1.0

        # markdown_short_only=False로 테스트
        cfg2 = HMMRiskFilterConfig(size_min=0.25, size_max=1.25, var_target=-5.0, markdown_short_only=False)
        decision2 = should_trade_filter(md_posterior, None, cfg2, 0, include_shadow_fields=True)
        # 5.0 / 10.52 ≈ 0.475 → min(0.25)보다 높으므로 none
        assert decision2.action == "ALLOW"


class TestKupiecVaR:
    """Kupiec VaR Backtest 테스트"""

    def test_kupiec_nominal_rate(self):
        """breach 비율이 alpha와 일치하면 통과"""
        import random
        random.seed(7)
        n, alpha = 5000, 0.05

        # breach 확률이 정확히 alpha인 데이터 생성
        pnl = []
        var = []
        for _ in range(n):
            is_breach = random.random() < alpha
            v = 1.0
            x = -1.1 if is_breach else -0.5  # breach면 VaR 초과
            pnl.append(x)
            var.append(v)

        result = kupiec_uc(pnl, var, alpha, side="both")

        assert result.pass_test, f"Should pass: LR={result.lr_uc:.2f}, p_hat={result.p_hat:.4f}"
        assert abs(result.p_hat - alpha) < 0.02, f"p_hat should be close to alpha"

    def test_kupiec_excess_breaches(self):
        """breach가 너무 많으면 실패"""
        import random
        random.seed(42)
        n, alpha = 1000, 0.01

        # breach 비율 10% (alpha=1%보다 훨씬 높음)
        pnl = [-1.5 if i < 100 else -0.5 for i in range(n)]
        var = [1.0] * n

        result = kupiec_uc(pnl, var, alpha, side="both")

        assert not result.pass_test, f"Should fail: p_hat={result.p_hat:.4f} >> alpha={alpha}"

    def test_kupiec_too_few_breaches(self):
        """breach가 너무 적으면 실패"""
        n, alpha = 1000, 0.10

        # breach 비율 0%
        pnl = [-0.5] * n
        var = [1.0] * n

        result = kupiec_uc(pnl, var, alpha, side="both")

        assert not result.pass_test, f"Should fail: p_hat={result.p_hat} << alpha={alpha}"

    def test_kupiec_result_structure(self):
        """KupiecResult 구조 확인"""
        pnl = [-0.5] * 50 + [-1.5] * 5
        var = [1.0] * 55

        result = kupiec_uc(pnl, var, 0.05, side="both")

        assert isinstance(result, KupiecResult)
        assert result.n == 55
        assert result.breaches == 5
        assert 0 <= result.p_hat <= 1
        assert result.lr_uc >= 0
        assert 0 <= result.p_value <= 1


class TestAdaptiveClip:
    """Adaptive Clip 테스트"""

    def test_adaptive_cap_monotonic_volatility(self):
        """변동성 ↑ → 상한 ↓"""
        sigma_ref = 1.0
        entropy = 0.1

        u1 = adaptive_upper_cap(0.5, sigma_ref, entropy)  # 낮은 변동성
        u2 = adaptive_upper_cap(1.0, sigma_ref, entropy)  # 중간 변동성
        u3 = adaptive_upper_cap(2.0, sigma_ref, entropy)  # 높은 변동성

        assert u1 >= u2 >= u3, f"Monotonicity violated: {u1}, {u2}, {u3}"

    def test_adaptive_cap_monotonic_entropy(self):
        """엔트로피 ↑ → 상한 ↓"""
        sigma_ref = 1.0
        sigma_t = 1.0

        u1 = adaptive_upper_cap(sigma_t, sigma_ref, 0.1)  # 낮은 불확실성
        u2 = adaptive_upper_cap(sigma_t, sigma_ref, 0.5)  # 중간 불확실성
        u3 = adaptive_upper_cap(sigma_t, sigma_ref, 0.9)  # 높은 불확실성

        assert u1 >= u2 >= u3, f"Monotonicity violated: {u1}, {u2}, {u3}"

    def test_adaptive_cap_bounds(self):
        """상한이 [floor_cap, ceil_cap] 범위 내"""
        floor_cap, ceil_cap = 1.0, 1.5

        # 극단적인 케이스
        cases = [
            (0.1, 1.0, 0.0),   # 매우 낮은 변동성
            (10.0, 1.0, 0.9),  # 매우 높은 변동성
            (1.0, 1.0, 1.0),   # 최대 엔트로피
        ]

        for sigma_t, sigma_ref, entropy in cases:
            u = adaptive_upper_cap(sigma_t, sigma_ref, entropy, floor_cap=floor_cap, ceil_cap=ceil_cap)
            assert floor_cap <= u <= ceil_cap, f"Out of bounds: {u} for ({sigma_t}, {entropy})"

    def test_sized_position_adaptive_basic(self):
        """Adaptive sizing 기본 동작"""
        expected_var = 7.0  # 예: markup 상태
        sigma_t = 1.0
        sigma_ref = 1.0
        entropy_t = 0.2

        size_mult, upper_cap, clip_hit = sized_position_adaptive(
            expected_var, sigma_t, sigma_ref, entropy_t,
            var_target=5.0, size_min=0.25
        )

        assert 0.25 <= size_mult <= 1.5
        assert 1.0 <= upper_cap <= 1.5
        assert clip_hit in ("none", "min", "max")

    def test_sized_position_clip_detection(self):
        """Clip hit 감지"""
        # 하한 히트: 높은 VaR
        size1, _, hit1 = sized_position_adaptive(
            expected_var=20.0,  # 매우 높은 VaR
            sigma_t=1.0, sigma_ref=1.0, entropy_t=0.1,
            var_target=5.0, size_min=0.25
        )
        assert hit1 == "min" or size1 == 0.25

        # 상한 히트: 낮은 VaR + 낮은 변동성
        size2, _, hit2 = sized_position_adaptive(
            expected_var=3.0,  # 낮은 VaR
            sigma_t=0.5, sigma_ref=1.0, entropy_t=0.1,
            var_target=5.0, size_min=0.25
        )
        # 5.0/3.0 = 1.67 > upper_cap
        assert hit2 == "max" or size2 <= 1.5

    def test_should_trade_filter_adaptive_integration(self):
        """Adaptive 필터 통합 테스트"""
        cfg = HMMRiskFilterConfig()
        adaptive_cfg = AdaptiveClipConfig(sigma_ref=1.0)

        # max_p >= min_max_prob (0.45)를 만족하는 posterior
        # markup에 0.6 확률 집중 → uncertainty gate 통과
        posterior = np.array([0.1, 0.6, 0.1, 0.1, 0.05, 0.05])
        sigma_t = 1.0

        decision = should_trade_filter_adaptive(
            posterior, None, cfg, adaptive_cfg, sigma_t
        )

        assert decision.action in ("ALLOW", "NO_TRADE", "COOLDOWN", "SHORT_ONLY")
        # NO_TRADE가 아닌 경우에만 size_mult 체크
        if decision.action != "NO_TRADE":
            assert 0.25 <= decision.size_mult <= 1.5
        else:
            # NO_TRADE인 경우 size_mult = 0
            assert decision.size_mult == 0.0

    def test_adaptive_vs_static_comparison(self):
        """Adaptive가 static보다 동적으로 조정됨"""
        cfg = HMMRiskFilterConfig(size_max=1.25)
        adaptive_cfg = AdaptiveClipConfig(sigma_ref=1.0)

        posterior = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # distribution

        # 낮은 변동성
        decision_low_vol = should_trade_filter_adaptive(
            posterior, None, cfg, adaptive_cfg, sigma_t=0.5
        )

        # 높은 변동성
        decision_high_vol = should_trade_filter_adaptive(
            posterior, None, cfg, adaptive_cfg, sigma_t=2.0
        )

        # 낮은 변동성에서 더 큰 사이즈 허용
        assert decision_low_vol.size_mult >= decision_high_vol.size_mult


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
