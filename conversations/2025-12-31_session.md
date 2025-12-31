# Claude Code 대화 기록 - 2025-12-31

## 세션 요약

이전 세션에서 이어진 대화로, 박스 락(Lock) 로직 롤백 및 주간 일요일 최적화 시스템 구현을 완료했습니다.

---

## 1. 이전 세션 요약 (컨텍스트)

### 완료된 작업
- P0-1: Stoch RSI shift(1) 룩어헤드 방지 완료
- P0-2: RSI 다이버전스 룩어헤드 방지 완료
- spot_mode 파라미터 추가 완료
- 90일 백테스트 실행: +26.27% (3x: +78.80%), MDD -0.64%

### 핵심 문제
- Wyckoff 박스가 매 봉마다 동적으로 변경됨
- 박스 락(Lock) 로직 구현 시도 → 사용자 피드백으로 롤백 결정

---

## 2. 이번 세션 작업 내용

### 2.1 박스 락(Lock) 로직 롤백

**사용자 피드백:**
> "박스를 고정하면... 그 고정한 값이 틀렸을 수도 있잖아? Phase A가 맞는지 틀린지 정확히 알 수 없음"

**롤백 내용:**
- `src/wpcn/wyckoff/box.py`: `LockedBox`, `BoxLockManager` 클래스 제거
- `src/wpcn/wyckoff/phases.py`:
  - `use_box_lock` 파라미터 제거
  - SC/BC/AR 감지 시 박스 락 로직 제거
  - TP/SL: 진입가 기준 % 방식으로 통일
    - 롱: +1.5% 익절, -1.0% 손절
    - 숏: -1.5% 익절, +1.0% 손절

### 2.2 주간 일요일 최적화 시스템 구현

**새로 생성된 파일:**

#### `src/wpcn/tuning/weekly_optimizer.py`
- `WeeklyOptimizer`: 주간 최적화 엔진
  - 학습 기간: 4주
  - 검증 기간: 1주
  - 랜덤 샘플 50개 후보 평가
- `OptimizableParams`: 최적화 가능한 파라미터
  - tp_pct, sl_pct (TP/SL %)
  - sl_atr_mult, tp_atr_mult (ATR 배수)
  - box_L, m_freeze (박스 파라미터)
  - min_score, cooldown_bars 등
- `ParamSearchSpace`: 탐색 공간 정의
- 목적 함수: 수익률 - MDD + 승률/PF 보너스

#### `src/wpcn/tuning/scheduler.py`
- `OptimizationScheduler`: 스케줄러 클래스
- CLI 지원:
  - `--run-now`: 즉시 실행
  - `--check-sunday`: 일요일 체크 후 실행
- cron/Windows Task Scheduler 연동 가능

#### `src/wpcn/tuning/__init__.py`
- 모듈 export 정리

---

## 3. Git 커밋 기록

```
55dab4e refactor: 박스 락 롤백 및 주간 일요일 최적화 시스템 구현
```

**변경 파일:**
- `src/wpcn/wyckoff/box.py` (롤백)
- `src/wpcn/wyckoff/phases.py` (롤백)
- `src/wpcn/tuning/weekly_optimizer.py` (신규)
- `src/wpcn/tuning/scheduler.py` (신규)
- `src/wpcn/tuning/__init__.py` (업데이트)

**GitHub 푸시 완료:**
```
e1523a0..55dab4e  QWAS-SYS-15M-V1 -> QWAS-SYS-15M-V1
```

---

## 4. 주요 결정 사항

| 항목 | 결정 | 이유 |
|------|------|------|
| 박스 락 | 롤백 | Phase A 감지의 불확실성 |
| TP/SL 방식 | 진입가 기준 % | 박스 경계보다 안정적 |
| 최적화 주기 | 주간 (일요일) | 일/월/년 대비 균형 |
| 학습 기간 | 4주 | 충분한 데이터 확보 |
| 검증 기간 | 1주 | 과적합 방지 |

---

## 5. 다른 컴퓨터에서 코드 받기

```bash
# 처음이라면 (클론)
git clone https://github.com/honggucci/charNavi.git
cd charNavi
git checkout QWAS-SYS-15M-V1

# 이미 클론되어 있다면 (풀)
cd charNavi
git pull origin QWAS-SYS-15M-V1
```

---

## 6. 다음 단계 (TODO)

- [ ] 주간 최적화 실제 데이터로 테스트
- [ ] V8BacktestEngine과 연동
- [ ] 최적화 결과 자동 적용 파이프라인
- [ ] 백테스트 성능 검증

---

*생성일: 2025-12-31*
*브랜치: QWAS-SYS-15M-V1*