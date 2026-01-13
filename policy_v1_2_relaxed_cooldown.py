"""
HMM Risk Filter Policy v1.2 (CHALLENGER - Relaxed Cooldown)
==========================================================

목적: Transition cooldown 완화로 트레이드 수 대폭 확보

v1.1 대비 변경사항:
- transition_delta: 0.20 → 0.40 (더 큰 변화만 cooldown 유발)
- cooldown_bars: 2 → 1 (대기 시간 단축)

근거:
- v1.1에서 blocked_by_transition이 99% 차지
- delta > 0.20이 너무 민감함 (거의 모든 봉에서 발생)
"""

from dataclasses import dataclass
from typing import Tuple
import hashlib
import json


@dataclass(frozen=True)
class PolicyV12RelaxedCooldownConfig:
    """
    Policy v1.2 - Transition Cooldown 완화

    v1.1 대비 변경점:
    - transition_delta: 0.20 → 0.40
    - cooldown_bars: 2 → 1
    """
    # A. Soft Sizing (동일)
    var_target: float = -5.0
    size_min: float = 0.25
    size_max: float = 1.25

    # Transition Cooldown (완화됨!)
    transition_delta: float = 0.40  # 0.20 → 0.40
    cooldown_bars: int = 1          # 2 → 1

    # B. Short Permit (동일)
    short_permit_enabled: bool = True
    short_permit_min_markdown_prob: float = 0.60
    short_permit_min_trend_strength: float = -0.10

    # C. Long Permit (v1.1과 동일)
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


# CHALLENGER v1.2 - Relaxed Cooldown
POLICY_V12_RELAXED_COOLDOWN = PolicyV12RelaxedCooldownConfig()


@dataclass(frozen=True)
class PolicyV12NoCooldownConfig:
    """
    Policy v1.2 - Transition Cooldown 비활성화 (극단적 완화)

    transition_delta=1.0으로 사실상 cooldown off
    """
    # A. Soft Sizing
    var_target: float = -5.0
    size_min: float = 0.25
    size_max: float = 1.25

    # Transition Cooldown (비활성화!)
    transition_delta: float = 1.0   # 사실상 off (delta가 1.0 초과하려면 불가능)
    cooldown_bars: int = 0

    # B. Short Permit
    short_permit_enabled: bool = True
    short_permit_min_markdown_prob: float = 0.60
    short_permit_min_trend_strength: float = -0.10

    # C. Long Permit (v1.1과 동일)
    long_permit_enabled: bool = True
    long_permit_states: Tuple[str, ...] = ('markup', 'accumulation', 're_accumulation')

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


POLICY_V12_NO_COOLDOWN = PolicyV12NoCooldownConfig()


if __name__ == "__main__":
    from policy_v1_1_relaxed import POLICY_V11_RELAXED

    print("=" * 60)
    print("v1.1 vs v1.2 비교 (Cooldown 완화)")
    print("=" * 60)
    print()
    print("[Transition Cooldown]")
    print(f"  v1.1: transition_delta={POLICY_V11_RELAXED.transition_delta}, cooldown_bars={POLICY_V11_RELAXED.cooldown_bars}")
    print(f"  v1.2: transition_delta={POLICY_V12_RELAXED_COOLDOWN.transition_delta}, cooldown_bars={POLICY_V12_RELAXED_COOLDOWN.cooldown_bars}")
    print(f"  v1.2 No CD: transition_delta={POLICY_V12_NO_COOLDOWN.transition_delta} (사실상 off)")
    print()
    print("예상 효과:")
    print("  - v1.1: blocked_by_transition ~99%")
    print("  - v1.2: 큰 상태 전이만 차단 → 트레이드 수 증가")
    print("  - v1.2 No CD: 상태 전이 무시 → 최대 트레이드 수 (위험)")
    print("=" * 60)