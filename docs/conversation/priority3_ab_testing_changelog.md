# Priority 3: A/B Testing / Version Management 수정 이력

## 버전: v2.2.1
## 작업일: 2026-01-02

---

## 0. 버그 수정 (GPT 리뷰 반영)

### v2.2 → v2.2.1 수정사항

| 문제 | 원인 | 수정 |
|------|------|------|
| `get_param_store(market)` 잘못 호출 | market이 store_dir로 전달됨 | `get_param_store(market=market)` |
| `store.load(symbol, tf, run_id)` TypeError | ParamStore.load()는 run_id 미지원 | `store.get_champion()`, `store.get_active()` 사용 |
| OOS 비교에 dict 사용 | `performance.oos`는 dict | `confidence.oos_performance` (float, 0~1) 사용 |
| `confidence.score` 없는 필드 | 존재하지 않는 필드 참조 | `confidence.confidence_score` (property) 사용 |
| run_id fallback 형식 오류 | `{symbol}_{tf}_{ts}` → 파일명 불일치 | 타임스탬프만 사용: `{ts}` |
| `register_challenger()` 히스토리 오류 | set_active 후 get_challenger → 새 버전 반환 | set_active 전에 old_challenger 저장 |
| `confidence None` 비교 TypeError | `None / float` 연산 | 모든 비교에 `is not None` 체크 추가 |
| `auto_decide()` 롤백 체크 truthy 버그 | `if champion_oos and challenger_oos:` → 0.0 스킵 | `is not None and > 0` 체크로 수정 |

### 헬퍼 함수 추가
```python
def _get_oos_performance(param) -> Optional[float]:
    """confidence.oos_performance (0~1 스케일) 안전 추출"""

def _get_confidence_score(param) -> Optional[float]:
    """confidence.confidence_score (0~100 스케일) 안전 추출"""

def _is_oos_pending(param) -> bool:
    """oos_pending 플래그 안전 체크"""
```

---

## 1. 개요

Champion/Challenger 기반 A/B Testing 시스템을 도입하여 파라미터 버전을 안전하게 관리합니다.

### 핵심 개념
- **Champion**: 현재 라이브 트레이딩에 사용 중인 검증된 파라미터
- **Challenger**: 새로 튜닝된 파라미터 (1주일간 OOS 테스트 후 승격 가능)
- **ACTIVE**: 현재 OOS 측정 대상 (Challenger와 동일)

---

## 2. 신규 파일

### `param_versioning.py` (신규 생성)

Champion/Challenger 버전 관리 모듈입니다.

**핵심 클래스:**
```python
@dataclass
class VersionInfo:
    """버전 정보"""
    run_id: str
    role: str  # "champion" | "challenger" | "retired"
    promoted_at: Optional[datetime] = None
    demoted_at: Optional[datetime] = None
    oos_performance: Optional[float] = None
    confidence_score: Optional[float] = None

@dataclass
class ComparisonResult:
    """Champion vs Challenger 비교 결과"""
    champion_run_id: Optional[str]
    challenger_run_id: Optional[str]
    champion_oos: Optional[float]
    challenger_oos: Optional[float]
    oos_improvement: float
    should_promote: bool
    reason: str

class ParamVersionManager:
    """Champion/Challenger 버전 관리자"""

    # 승격/롤백 임계값
    PROMOTE_OOS_THRESHOLD = 0.95      # OOS 5% 이내 하락까지 허용
    PROMOTE_CONFIDENCE_THRESHOLD = 0.90  # confidence 10% 이내 하락까지 허용
    ROLLBACK_OOS_THRESHOLD = 0.80     # OOS 20% 이상 하락 시 롤백
```

**주요 메서드:**
- `register_challenger(run_id)`: 새 Challenger 등록
- `compare_performance()`: Champion vs Challenger 비교
- `promote_challenger()`: Challenger를 Champion으로 승격
- `rollback_to_champion()`: Challenger 폐기, Champion 유지
- `auto_decide()`: 자동 승격/롤백 결정 및 실행
- `get_version_history()`: 버전 히스토리 조회

**팩토리 함수:**
```python
def get_version_manager(
    symbol: str = "BTCUSDT",
    market: str = "futures",
    timeframe: str = "15m"
) -> ParamVersionManager
```

---

## 3. 수정 파일

### `param_store.py` (수정)

Champion 관련 메서드 추가:

```python
# 새로 추가된 메서드들
def mark_as_champion(self, symbol: str, timeframe: str, run_id: str) -> bool:
    """파라미터를 Champion으로 마킹"""

def get_champion(self, symbol: str, timeframe: str) -> Optional[OptimizedParams]:
    """Champion 파라미터 조회"""

def demote_champion(self, symbol: str, timeframe: str) -> bool:
    """Champion 지위 해제"""

def is_champion(self, symbol: str, timeframe: str, run_id: str) -> bool:
    """특정 run_id가 Champion인지 확인"""

@property
def _base_path(self) -> Path:
    """버전 관리에서 사용하는 기본 경로"""
```

**인덱스 구조 변경:**
```json
{
  "BTCUSDT_15m": {
    "latest_file": "...",
    "updated_at": "...",
    "active": {
      "file": "...",
      "run_id": "...",
      "set_at": "..."
    },
    "champion": {
      "file": "...",
      "run_id": "...",
      "promoted_at": "..."
    }
  }
}
```

---

### `scheduler.py` (수정)

v2.1 → v2.2 업그레이드:

**주간 배치 실행 순서 변경:**
```
v2.1:
1. finalize_last_week_oos()
2. run_adaptive_tuning()
3. set_active(new_run_id)

v2.2:
1. finalize_last_week_oos()  # 현재 Challenger의 OOS 확정
2. auto_promote_or_rollback() # Champion vs Challenger 비교
3. run_adaptive_tuning()      # 새 파라미터 튜닝
4. register_challenger()      # 새 파라미터를 Challenger로 등록
```

**새로 추가된 메서드:**
```python
def _auto_promote_or_rollback(self, symbol: str) -> Dict[str, Any]:
    """
    Step 2: A/B 비교 후 자동 승격/롤백

    Champion vs Challenger OOS 성능을 비교하여:
    - Challenger가 우수하면 Champion으로 승격
    - Challenger가 현저히 나쁘면 롤백
    - 그 외에는 Challenger 유지 (추가 관찰)
    """

def _register_challenger(self, symbol: str, run_id: str) -> bool:
    """
    Step 4: 새 파라미터를 Challenger로 등록
    """
```

---

### `__init__.py` (수정)

v2.2 export 추가:

```python
# 버전 관리 (v2.2 A/B Testing)
from .param_versioning import (
    ParamVersionManager,
    VersionInfo,
    ComparisonResult,
    get_version_manager,
)

__all__ = [
    ...
    # === 버전 관리 (v2.2 A/B Testing) ===
    "ParamVersionManager",
    "VersionInfo",
    "ComparisonResult",
    "get_version_manager",
]
```

---

## 4. 승격/롤백 로직

### 승격 조건 (Challenger → Champion)
```python
if oos_ratio >= 0.95 and conf_ratio >= 0.90:
    promote_challenger()
```
- Challenger OOS >= Champion OOS * 0.95 (5% 이내 하락 허용)
- Challenger confidence >= Champion confidence * 0.90 (10% 이내 하락 허용)
- Challenger OOS가 측정 완료됨 (oos_pending=False)

### 롤백 조건 (Challenger 폐기)
```python
if oos_ratio < 0.80:
    rollback_to_champion()
```
- Challenger OOS < Champion OOS * 0.80 (20% 이상 하락)
- 또는 수동 롤백 요청

### 유지 조건
- 0.80 <= oos_ratio < 0.95: Challenger 유지, 추가 관찰 필요

---

## 5. 버전 히스토리 저장

파일: `{symbol}_{timeframe}_version_history.json`

```json
{
  "champion_run_id": "20260102_120000",
  "challenger_run_id": "20260109_120000",
  "history": [
    {
      "run_id": "20251225_120000",
      "role": "retired",
      "promoted_at": null,
      "demoted_at": "2026-01-02T12:00:00",
      "oos_performance": 0.75,
      "confidence_score": 68.5
    }
  ],
  "updated_at": "2026-01-02T12:00:00"
}
```

---

## 6. 사용 예시

### 버전 관리 기본 사용
```python
from wpcn._08_tuning import get_version_manager

manager = get_version_manager("BTCUSDT", "futures", "15m")

# 현재 상태 확인
status = manager.get_status()
print(f"Champion: {status['champion']}")
print(f"Challenger: {status['challenger']}")

# A/B 비교
comparison = manager.compare_performance()
print(f"Should promote: {comparison.should_promote}")
print(f"Reason: {comparison.reason}")

# 자동 결정
action, comparison = manager.auto_decide()
print(f"Action taken: {action}")
```

### 스케줄러와 연동
```bash
# 주간 배치 실행 (A/B Testing 포함)
python -m wpcn._08_tuning.scheduler --run-now --market spot --timeframe 15m
```

---

## 7. 수정 파일 요약

| 파일 | 작업 | 설명 |
|------|------|------|
| `param_versioning.py` | 신규 생성 | Champion/Challenger 버전 관리 모듈 |
| `param_store.py` | 수정 | mark_as_champion, get_champion, demote_champion, is_champion 추가 |
| `scheduler.py` | 수정 | v2.2: _auto_promote_or_rollback, _register_challenger 추가 |
| `__init__.py` | 수정 | ParamVersionManager, get_version_manager export |

---

## 8. 다음 단계 (Priority 4~6)

- **Priority 4**: Unit Tests - param_versioning, scheduler A/B 로직 테스트
- **Priority 5**: Live Trading 연동 - Champion 파라미터 자동 적용
- **Priority 6**: Alerting & Monitoring - 승격/롤백 시 알림

