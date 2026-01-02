# Priority 2: OOS 실시간 추적 수정 이력

## 버전: v2.1.1
## 작업일: 2026-01-02

---

## 1. GPT 1차 리뷰 (4건)

### CRITICAL-1: run_id가 파일경로 반환 문제
- **문제**: `ParamStore.save()`가 `str(filepath)` 반환 → `set_active(run_id)`에서 파일명 재구성 실패
- **수정**: `return timestamp` (YYYYmmdd_HHMMSS 형식)
- **파일**: `param_store.py:373`

### CRITICAL-2: OOS 측정이 ACTIVE가 아닌 다른 파라미터 사용
- **문제**: `load_reliable_as_bars()` 호출 → 신뢰도 기반으로 다른 파라미터 선택 가능
- **수정**: `convert_params_to_bars(active.params, timeframe)` 직접 변환
- **파일**: `oos_tracker.py:345`

### MAJOR-3: OOS 가중치 불일치
- **문제**: 문서상 30점인데 코드는 10점
- **수정**:
  - OOS 성능: 30점 (`oos_performance * 30`)
  - 기본 점수: 55점
  - OOS 미측정 패널티: 15점
- **파일**: `param_store.py:100-121`

### MAJOR-4: get_param_store 싱글톤 market 미분리
- **문제**: spot/futures가 같은 싱글톤 공유 → 데이터 오염
- **수정**: `_stores: Dict[str, ParamStore]` market별 분리
- **파일**: `param_store.py:786-811`

---

## 2. GPT 2차 리뷰 (2건)

### index 덮어쓰기로 active 정보 유실
- **문제**: `self._index[key] = {...}` 전체 덮어쓰기 → 기존 active 정보 삭제됨
- **수정**: `self._index[key].update({...})` 방식으로 변경
- **파일**: `param_store.py:361-368`

### confidence_score 음수 가능
- **문제**: OOS 미측정 시 패널티 15점 → 기본 점수 낮으면 음수 가능
- **수정**: `max(0.0, base_score + oos_score - missing_oos_penalty)`
- **파일**: `param_store.py:121`

---

## 3. GPT 3차 리뷰 - 운영 리스크 (2건)

### A: missing_data를 OOS 완료로 처리
- **문제**: 데이터 부족 시 `update_oos({"status":"missing_data"})` 호출
  → `oos_performance = 0.0` 저장 → 패널티 미적용 (OOS 완료 취급)
- **수정**:
  - `oos_performance = None` 유지 → 15점 패널티 적용됨
  - `oos_pending = True` 플래그 추가
  - `oos_skip_reason = "missing_data"` 기록
- **파일**: `param_store.py:737-754`

### B: missing_data 시 재시도 불가
- **문제**: `active.performance.oos is not None` 체크로 무조건 스킵
- **수정**: `oos_pending=True`면 재시도 허용
- **파일**: `oos_tracker.py:299-313`

### 새 필드 추가 (ParamConfidence)
- `oos_pending: bool` - 재시도 필요 여부
- `oos_skip_reason: str` - 스킵 사유
- **파일**: `param_store.py:61-63, 136-137, 152-153`

---

## 4. GPT 4차 리뷰 - 런타임 버그 (2건)

### oos_sharpe 미정의 → UnboundLocalError
- **문제**: missing_data 분기에서 `oos_sharpe` 정의 안됨 → print문에서 터짐
- **수정**: if/else 전에 `oos_sharpe = oos_metrics.get("sharpe_ratio", 0)` 선언
- **파일**: `param_store.py:747`

### 성공 시 oos_skip_reason 클리어 안됨
- **문제**: missing_data 후 성공해도 `oos_skip_reason = "missing_data"` 잔존
- **수정**: else 분기에 `oos_skip_reason = ""` 추가
- **파일**: `param_store.py:761`

---

## 수정 파일 요약

| 파일 | 수정 내용 |
|------|----------|
| `param_store.py` | run_id 반환, OOS 30점 체계, market별 싱글톤, index update 방식, 음수 clamp, missing_data 처리, oos_pending/oos_skip_reason 필드 |
| `oos_tracker.py` | ACTIVE params 직접 변환, oos_pending 재시도 로직 |
| `scheduler.py` | 3단계 주간 배치 (OOS확정→튜닝→ACTIVE지정) |
| `run_tuning.py` | run_id 반환, sensitivity 통합 |
| `__init__.py` | OOS exports 추가 |

---

## 최종 주간 배치 실행 순서

```
1. finalize_last_week_oos()  # 현재 ACTIVE의 지난주 OOS 확정
2. run_adaptive_tuning()      # 새 파라미터 튜닝 (sensitivity 포함)
3. set_active(new_run_id)     # 새 파라미터를 ACTIVE로 지정
```

---

## confidence_score 계산식 (v2.1.1)

```python
# 기본 점수 (55점 만점)
base_score = (
    overfit_score * 10 +      # 과적합 비율
    cv_consistency * 5 +       # CV 일관성
    temporal_stability * 5 +   # 시간 안정성
    cycle_quality * 10 +       # 사이클 품질
    fold_consistency * 5 +     # 폴드 일관성
    fft_reliability * 5 +      # FFT 신뢰도
    stability_score * 15       # 민감도 안정성
)

# OOS 점수 (30점 만점)
if oos_performance is not None:
    oos_score = oos_performance * 30
    missing_oos_penalty = 0
else:
    oos_score = 0
    missing_oos_penalty = 15

# 최종 (0~85점, 음수 방지)
confidence = max(0.0, base_score + oos_score - missing_oos_penalty)
```
