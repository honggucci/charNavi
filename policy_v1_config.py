"""
HMM Risk Filter Policy v1.0 (Champion)
======================================
확정일: 2026-01-12
WFO 검증: 2023 (-$167, +99.9%), 2024 (-$7,975, +94.6%)

이 설정은 "계좌 생존" 단계 달성.
SL/TP 튜닝(D)은 이 위에서 실험으로만 진행.
"""

from dataclasses import dataclass
from typing import Tuple
import hashlib
import json


@dataclass(frozen=True)
class PolicyV1Config:
    """
    Policy v1.0 - 확정 설정 (변경 금지)

    구성:
    - A. Soft Sizing: VaR 기반 포지션 사이징
    - B. Short Permit: markdown + 하락추세에서만 숏 허용
    - C. Long Permit: markup에서만 롱 허용
    """
    # A. Soft Sizing
    var_target: float = -5.0
    size_min: float = 0.25
    size_max: float = 1.25

    # Transition Cooldown
    transition_delta: float = 0.20
    cooldown_bars: int = 2

    # B. Short Permit
    short_permit_enabled: bool = True
    short_permit_min_markdown_prob: float = 0.60
    short_permit_min_trend_strength: float = -0.10

    # C. Long Permit (핵심 변경)
    long_permit_enabled: bool = True
    long_permit_states: Tuple[str, ...] = ('markup',)

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


# Champion Config (절대 변경 금지)
POLICY_V1 = PolicyV1Config()

# WFO 검증 결과 기록
WFO_RESULTS = {
    'version': 'v1.0',
    'confirmed_date': '2026-01-12',
    'config_hash': POLICY_V1.config_hash(),
    'tests': [
        {
            'train': ['2021', '2022'],
            'test': '2023',
            'baseline_pnl': -140829,
            'filtered_pnl': -167,
            'improvement_pct': 99.9,
        },
        {
            'train': ['2021', '2022', '2023'],
            'test': '2024',
            'baseline_pnl': -148149,
            'filtered_pnl': -7975,
            'improvement_pct': 94.6,
        },
    ],
    'status': 'CHAMPION',
    'notes': [
        '계좌 생존 단계 달성',
        '2023: 거의 손익분기',
        '2024: 손실 94.6% 감소',
        'markup에서만 롱 허용 (독성 롱 제거)',
    ]
}


def get_champion_config():
    """Champion 설정 반환"""
    return POLICY_V1


def print_policy_summary():
    """정책 요약 출력"""
    print("=" * 60)
    print("HMM RISK FILTER POLICY v1.0 (CHAMPION)")
    print("=" * 60)
    print(f"Config Hash: {POLICY_V1.config_hash()}")
    print(f"Confirmed: {WFO_RESULTS['confirmed_date']}")
    print()
    print("[A] Soft Sizing")
    print(f"    var_target: {POLICY_V1.var_target}")
    print(f"    size_range: [{POLICY_V1.size_min}, {POLICY_V1.size_max}]")
    print()
    print("[B] Short Permit")
    print(f"    P(markdown) > {POLICY_V1.short_permit_min_markdown_prob}")
    print(f"    trend_strength < {POLICY_V1.short_permit_min_trend_strength}")
    print()
    print("[C] Long Permit")
    print(f"    allowed_states: {POLICY_V1.long_permit_states}")
    print()
    print("[WFO Results]")
    for test in WFO_RESULTS['tests']:
        print(f"    {test['train']} -> {test['test']}: "
              f"${test['filtered_pnl']:,} ({test['improvement_pct']:+.1f}%)")
    print()
    print(f"Status: {WFO_RESULTS['status']}")
    print("=" * 60)


if __name__ == "__main__":
    print_policy_summary()