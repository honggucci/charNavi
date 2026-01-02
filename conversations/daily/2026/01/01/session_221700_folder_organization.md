# 2026-01-01 튜닝 폴더 정리 및 구조 분석

## 개요
`_08_tuning/` 폴더 정리 및 튜닝 모듈 사용 시점 정리

---

## 수행한 작업

### 1. GPT 피드백 반영 (adaptive_space.py)
GPT가 지적한 5가지 "지뢰" 수정:

- **(A)** `_hints`를 param_space에서 분리 → `ParamHints` 별도 클래스 생성
- **(B)** `m_freeze`가 `box_L`에 의존하지만 독립 샘플링 → `validate_params()` 추가
- **(C)** `max_hold_bars`가 Theta/RunConfig 분리 필요 → `build_theta_and_config()` 어댑터 생성
- **(D)** Bayesian hints 미반영 → `create_constrained_sampler()` 추가
- **(E)** Percentile이 불완전 사이클 포함 → quality cycle 필터링 추가

### 2. 폴더 정리

#### 이동된 파일들:
| 원본 위치 | 새 위치 | 설명 |
|-----------|---------|------|
| `analyze_wyckoff_cycles.py` | `_08_tuning/legacy/` | 원본 (deprecated) |
| `analyze_wyckoff_cycles_gpt_patched.py` | `_08_tuning/legacy/` | GPT 패치 (deprecated) |
| `analyze_wyckoff_cycles_v2.py` | `_08_tuning/cycle_analyzer.py` | 현재 사용 버전 |
| `run_adaptive_tuning.py` | `_08_tuning/run_tuning.py` | 통합 파이프라인 |

#### 복원된 파일:
- `weekly_optimizer.py`: 실수로 legacy로 이동했으나 `scheduler.py`에서 사용 중이어서 복원

---

## 📁 `_08_tuning/` 튜닝 모듈 정리

### 실제로 사용되는 핵심 파일들

| 파일 | 역할 | 언제 사용? |
|------|------|-----------|
| `scheduler.py` | 매주 일요일 자동 최적화 스케줄러 | cron/Task Scheduler로 주간 실행 |
| `weekly_optimizer.py` | 주간 최적화 엔진 (Train 4주 / Val 1주) | scheduler.py에서 호출 |
| `walk_forward.py` | Walk-Forward 최적화 (과적합 방지) | 수동 또는 배치 최적화 시 |
| `param_optimizer.py` | Bayesian/Random Search 엔진 | walk_forward.py에서 호출 |
| `adaptive_space.py` | 사이클 분석 기반 탐색 공간 생성 | cycle_analyzer 결과로 공간 축소 |
| `cycle_analyzer.py` | Wyckoff 사이클 FFT 분석 | 적응형 파라미터 공간 생성 전 |
| `param_store.py` | 최적화 결과 저장/로드 | 최적화 완료 후 저장, 트레이딩 시 로드 |
| `run_tuning.py` | 통합 튜닝 파이프라인 | CLI 직접 실행 |

---

### 튜닝 실행 시점

```
┌──────────────────────────────────────────────────────────────────┐
│                    매주 일요일 00:00                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. scheduler.py 실행 (cron/Task Scheduler)                     │
│         │                                                        │
│         ▼                                                        │
│   2. weekly_optimizer.py                                         │
│      • 최근 4주 데이터로 Train                                    │
│      • 최근 1주 데이터로 Validation                               │
│      • 50개 후보 중 상위 10개 검증                                 │
│         │                                                        │
│         ▼                                                        │
│   3. param_store.py → results/params/에 JSON 저장                │
│         │                                                        │
│         ▼                                                        │
│   4. 다음 주 트레이딩 시 main.py에서 로드 (미구현)                  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

### 실행 방법

```bash
# 1. 즉시 실행 (강제)
python -m wpcn._08_tuning.scheduler --run-now --symbols BTC-USDT

# 2. 일요일 체크 후 실행
python -m wpcn._08_tuning.scheduler --check-sunday

# 3. cron 설정 (Linux)
0 0 * * 0 python -m wpcn._08_tuning.scheduler --run-now

# 4. 통합 파이프라인 (사이클 분석 + 최적화)
python -m wpcn._08_tuning.run_tuning --symbol BTC-USDT --timeframe 5m
```

---

### legacy/ 폴더 (사용 안 함)

| 파일 | 설명 |
|------|------|
| `analyze_wyckoff_cycles.py` | 원본 사이클 분석기 (v2로 대체됨) |
| `analyze_wyckoff_cycles_gpt_patched.py` | GPT 패치 버전 (v2로 통합됨) |

---

### 파이프라인 흐름도

```
[cycle_analyzer.py]
       │ FFT로 시장 사이클 분석
       ▼
[adaptive_space.py]
       │ 사이클 기반 파라미터 공간 생성
       ▼
[walk_forward.py / weekly_optimizer.py]
       │ Train/Test 분할 최적화
       ▼
[param_optimizer.py]
       │ Bayesian 또는 Random Search
       ▼
[param_store.py]
       │ JSON 저장
       ▼
[main.py] ← 로드 (TODO: 아직 미연동)
```

---

### 현재 상태

- **scheduler.py** → `weekly_optimizer.py` 연동 ✅ 완료
- **param_store.py** → 저장/로드 기능 ✅ 완료
- **main.py** → ParamStore에서 동적 로드 ❌ **미구현** (다음 작업)

---

## adaptive_space.py 주요 변경사항

### 새로 추가된 상수
```python
THETA_KEYS = {"pivot_lr", "box_L", "m_freeze", "atr_len", "x_atr", "m_bw", "N_reclaim", "N_fill", "F_min"}
CONFIG_KEYS = {"tp_pct", "sl_pct", "min_score", "max_hold_bars", "rsi_oversold", "rsi_overbought", "cooldown_bars"}
```

### 새로 추가된 함수들
```python
def build_theta_and_config(params: Dict[str, Any]) -> Tuple[Dict, Dict]:
    """최적화 결과를 Theta와 RunConfig로 분리"""
    theta_dict = {k: v for k, v in params.items() if k in THETA_KEYS}
    config_dict = {k: v for k, v in params.items() if k in CONFIG_KEYS}
    return theta_dict, config_dict

def validate_params(params: Dict[str, Any]) -> Tuple[bool, str]:
    """파라미터 제약조건 검증 (m_freeze < box_L 등)"""
    if m_freeze >= box_L:
        return False, f"m_freeze({m_freeze}) >= box_L({box_L})"
    # ...more constraints

def create_constrained_sampler(space: AdaptiveParamSpace, hints: ParamHints) -> Callable:
    """의존성 있는 파라미터를 올바르게 샘플링하는 함수 반환"""
    # box_L 먼저 샘플링 후 m_freeze 범위 제한

@dataclass
class ParamHints:  # param_space와 분리!
    recommended_box_L: Optional[int] = None
    recommended_atr_len: Optional[int] = None
    # ...

def generate_adaptive_space(...) -> Tuple[AdaptiveParamSpace, ParamHints]:
    # Returns (param_space, hints) tuple - hints 분리!
    quality_cycles = [c for c in cycles_detail
                     if c.get("complete", False) and c.get("direction_purity", 0) >= purity_threshold]
```

---

## 다음 작업
- `main.py`에서 `ParamStore`를 연동하여 매주 최적화된 파라미터를 자동으로 로드하도록 수정
