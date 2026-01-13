"""
HMM Risk Filter Policy v1.1 (CHALLENGER - Relaxed)
=================================================

목적: 트레이드 수 확보 (연 50개 이상)
변경사항: long_permit_states 확장

주의: CHAMPION과 비교용. 실거래 적용 전 추가 검증 필요.
"""

from dataclasses import dataclass
from typing import Tuple
import hashlib
import json


@dataclass(frozen=True)
class PolicyV11RelaxedConfig:
    """
    Policy v1.1 - 완화된 롱 허용 (CHALLENGER)

    CHAMPION 대비 변경점:
    - long_permit_states: ('markup',) → ('markup', 'accumulation', 're_accumulation')
    - 롱 허용 범위 확대 → 트레이드 수 증가 예상
    """
    # A. Soft Sizing (CHAMPION 동일)
    var_target: float = -5.0
    size_min: float = 0.25
    size_max: float = 1.25

    # Transition Cooldown (CHAMPION 동일)
    transition_delta: float = 0.20
    cooldown_bars: int = 2

    # B. Short Permit (CHAMPION 동일)
    short_permit_enabled: bool = True
    short_permit_min_markdown_prob: float = 0.60
    short_permit_min_trend_strength: float = -0.10

    # C. Long Permit (완화됨!)
    long_permit_enabled: bool = True
    long_permit_states: Tuple[str, ...] = ('markup', 'accumulation', 're_accumulation')

    # 레거시 (비활성화)
    long_filter_enabled: bool = False
    long_filter_max_markdown_prob: float = 0.40

    def to_dict(self) -> dict:
        return {
            'var_target': self.var_target,
            'size_min': self.size_min,
            'size_max': self.size_max,
            'transition_delta': self.transition_delta,
            'cooldown_bars': self.cooldown_bars,
            'short_permit_enabled': self.short_permit_enabled,
            'short_permit_min_markdown_prob': self.short_permit_min_markdown_prob,
            'short_permit_min_trend_strength': self.short_permit_min_trend_strength,
            'long_permit_enabled': self.long_permit_enabled,
            'long_permit_states': list(self.long_permit_states),
            'long_filter_enabled': self.long_filter_enabled,
        }

    def config_hash(self) -> str:
        """설정 해시 (변경 감지용)"""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:12]


# CHALLENGER - Relaxed Config
POLICY_V11_RELAXED = PolicyV11RelaxedConfig()


# 변형 설정들 (A/B 테스트용)
@dataclass(frozen=True)
class PolicyV11NoLongFilterConfig:
    """
    Policy v1.1 - 롱 필터 비활성화 (더 공격적)

    모든 롱 허용, 숏만 필터링
    """
    # A. Soft Sizing
    var_target: float = -5.0
    size_min: float = 0.25
    size_max: float = 1.25

    # Transition Cooldown
    transition_delta: float = 0.20
    cooldown_bars: int = 2

    # B. Short Permit (유지)
    short_permit_enabled: bool = True
    short_permit_min_markdown_prob: float = 0.60
    short_permit_min_trend_strength: float = -0.10

    # C. Long Permit (비활성화!)
    long_permit_enabled: bool = False
    long_permit_states: Tuple[str, ...] = ()

    # 레거시
    long_filter_enabled: bool = False
    long_filter_max_markdown_prob: float = 0.40

    def to_dict(self) -> dict:
        return {
            'var_target': self.var_target,
            'size_min': self.size_min,
            'size_max': self.size_max,
            'transition_delta': self.transition_delta,
            'cooldown_bars': self.cooldown_bars,
            'short_permit_enabled': self.short_permit_enabled,
            'short_permit_min_markdown_prob': self.short_permit_min_markdown_prob,
            'short_permit_min_trend_strength': self.short_permit_min_trend_strength,
            'long_permit_enabled': self.long_permit_enabled,
            'long_permit_states': list(self.long_permit_states),
            'long_filter_enabled': self.long_filter_enabled,
        }

    def config_hash(self) -> str:
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:12]


POLICY_V11_NO_LONG_FILTER = PolicyV11NoLongFilterConfig()


def get_relaxed_configs():
    """완화된 설정 목록"""
    return {
        'v1.1_relaxed': POLICY_V11_RELAXED,
        'v1.1_no_long_filter': POLICY_V11_NO_LONG_FILTER,
    }


def print_comparison():
    """CHAMPION vs CHALLENGER 비교"""
    from policy_v1_config import POLICY_V1

    print("=" * 60)
    print("CHAMPION vs CHALLENGER 비교")
    print("=" * 60)
    print()
    print("[C] Long Permit States:")
    print(f"  CHAMPION (v1.0):       {POLICY_V1.long_permit_states}")
    print(f"  CHALLENGER (v1.1):     {POLICY_V11_RELAXED.long_permit_states}")
    print(f"  NO_LONG_FILTER (v1.1): long_permit_enabled={POLICY_V11_NO_LONG_FILTER.long_permit_enabled}")
    print()
    print("예상 효과:")
    print("  - CHAMPION: markup에서만 롱 → 롱 차단 많음")
    print("  - CHALLENGER: accumulation 포함 → 롱 허용 증가")
    print("  - NO_LONG_FILTER: 모든 롱 허용 → 최대 트레이드 수")
    print("=" * 60)


if __name__ == "__main__":
    print_comparison()