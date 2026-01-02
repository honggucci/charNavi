"""
Param Versioning - Champion/Challenger 버전 관리
================================================

A/B 테스팅 기반 파라미터 버전 관리 시스템입니다.

핵심 기능:
1. Champion (현재 운영 버전) / Challenger (테스트 버전) 관리
2. OOS 성능 기반 자동 승격/롤백
3. 버전 히스토리 추적

용어:
- Champion: 현재 라이브 트레이딩에 사용 중인 검증된 파라미터
- Challenger: 새로 튜닝된 파라미터 (1주일간 OOS 테스트 후 승격 가능)
- ACTIVE: 현재 OOS 측정 대상 (Challenger와 동일)

사용법:
    from wpcn._08_tuning.param_versioning import ParamVersionManager

    manager = ParamVersionManager(store)

    # 새 Challenger 등록
    manager.register_challenger(new_run_id)

    # 1주일 후 OOS 비교
    result = manager.compare_performance()
    if result.should_promote:
        manager.promote_challenger()

    # 문제 발생 시 롤백
    manager.rollback_to_champion()
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json

from .param_store import ParamStore, get_param_store


def _get_oos_performance(param) -> Optional[float]:
    """
    파라미터에서 OOS 성능 추출 (0~1 스케일)

    confidence.oos_performance를 사용 (dict가 아닌 float)
    """
    if param is None:
        return None

    # confidence.oos_performance (float, 0~1 스케일)
    conf = getattr(param, 'confidence', None)
    if conf is None:
        return None

    # to_dict()로 접근 (안전)
    if hasattr(conf, 'to_dict'):
        conf_dict = conf.to_dict()
        return conf_dict.get('oos_performance')

    # 직접 접근
    return getattr(conf, 'oos_performance', None)


def _get_confidence_score(param) -> Optional[float]:
    """
    파라미터에서 confidence_score 추출 (0~100 스케일)

    confidence.confidence_score (property) 사용
    """
    if param is None:
        return None

    conf = getattr(param, 'confidence', None)
    if conf is None:
        return None

    # confidence_score는 property (to_dict에도 포함됨)
    if hasattr(conf, 'to_dict'):
        conf_dict = conf.to_dict()
        return conf_dict.get('confidence_score')

    # 직접 접근 (property)
    if hasattr(conf, 'confidence_score'):
        return conf.confidence_score

    return None


def _is_oos_pending(param) -> bool:
    """OOS 재시도 필요 여부 체크"""
    if param is None:
        return True

    conf = getattr(param, 'confidence', None)
    if conf is None:
        return True

    if hasattr(conf, 'to_dict'):
        conf_dict = conf.to_dict()
        return conf_dict.get('oos_pending', True)

    return getattr(conf, 'oos_pending', True)


@dataclass
class VersionInfo:
    """버전 정보"""
    run_id: str
    role: str  # "champion" | "challenger" | "retired"
    promoted_at: Optional[datetime] = None
    demoted_at: Optional[datetime] = None
    oos_performance: Optional[float] = None
    confidence_score: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "run_id": self.run_id,
            "role": self.role,
            "promoted_at": self.promoted_at.isoformat() if self.promoted_at else None,
            "demoted_at": self.demoted_at.isoformat() if self.demoted_at else None,
            "oos_performance": self.oos_performance,
            "confidence_score": self.confidence_score,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "VersionInfo":
        return cls(
            run_id=data["run_id"],
            role=data["role"],
            promoted_at=datetime.fromisoformat(data["promoted_at"]) if data.get("promoted_at") else None,
            demoted_at=datetime.fromisoformat(data["demoted_at"]) if data.get("demoted_at") else None,
            oos_performance=data.get("oos_performance"),
            confidence_score=data.get("confidence_score"),
        )


@dataclass
class ComparisonResult:
    """Champion vs Challenger 비교 결과"""
    champion_run_id: Optional[str]
    challenger_run_id: Optional[str]
    champion_oos: Optional[float]
    challenger_oos: Optional[float]
    champion_confidence: Optional[float]
    challenger_confidence: Optional[float]
    oos_improvement: float  # (challenger - champion) / champion
    confidence_improvement: float
    should_promote: bool
    reason: str

    def to_dict(self) -> Dict:
        return {
            "champion_run_id": self.champion_run_id,
            "challenger_run_id": self.challenger_run_id,
            "champion_oos": self.champion_oos,
            "challenger_oos": self.challenger_oos,
            "champion_confidence": self.champion_confidence,
            "challenger_confidence": self.challenger_confidence,
            "oos_improvement": self.oos_improvement,
            "confidence_improvement": self.confidence_improvement,
            "should_promote": self.should_promote,
            "reason": self.reason,
        }


class ParamVersionManager:
    """
    Champion/Challenger 버전 관리자

    승격 조건:
    1. Challenger OOS >= Champion OOS * 0.95 (5% 이내 하락 허용)
    2. Challenger confidence >= Champion confidence * 0.90
    3. Challenger OOS가 측정 완료됨 (oos_pending=False)

    롤백 조건:
    1. Challenger OOS < Champion OOS * 0.80 (20% 이상 하락)
    2. 또는 수동 롤백 요청
    """

    # 승격/롤백 임계값
    PROMOTE_OOS_THRESHOLD = 0.95      # OOS 5% 이내 하락까지 허용
    PROMOTE_CONFIDENCE_THRESHOLD = 0.90  # confidence 10% 이내 하락까지 허용
    ROLLBACK_OOS_THRESHOLD = 0.80     # OOS 20% 이상 하락 시 롤백

    # 최소 테스트 기간 (일)
    MIN_CHALLENGER_DAYS = 7

    def __init__(
        self,
        store: ParamStore,
        symbol: str = "BTCUSDT",
        market: str = "futures",
        timeframe: str = "15m"
    ):
        """
        Args:
            store: ParamStore 인스턴스
            symbol: 심볼
            market: 마켓 타입
            timeframe: 타임프레임
        """
        self.store = store
        self.symbol = symbol
        self.market = market
        self.timeframe = timeframe
        self._key = f"{symbol}_{timeframe}"

        # 버전 히스토리 파일 경로
        self._history_path = self.store._base_path / f"{self._key}_version_history.json"

        # 현재 champion/challenger 상태
        self._champion_run_id: Optional[str] = None
        self._challenger_run_id: Optional[str] = None
        self._history: List[VersionInfo] = []

        self._load_state()

    def _load_state(self):
        """버전 상태 로드"""
        if self._history_path.exists():
            try:
                with open(self._history_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                self._champion_run_id = data.get("champion_run_id")
                self._challenger_run_id = data.get("challenger_run_id")
                self._history = [
                    VersionInfo.from_dict(h) for h in data.get("history", [])
                ]
            except Exception as e:
                print(f"[VersionManager] Failed to load state: {e}")
                self._champion_run_id = None
                self._challenger_run_id = None
                self._history = []

        # store에서 ACTIVE 동기화 (Challenger = ACTIVE)
        active = self.store.get_active(self.symbol, self.timeframe)
        if active:
            active_run_id = getattr(active, 'run_id', None)
            if active_run_id and self._challenger_run_id != active_run_id:
                self._challenger_run_id = active_run_id
                self._save_state()

        # store에서 Champion 동기화
        champion = self.store.get_champion(self.symbol, self.timeframe)
        if champion:
            champion_run_id = getattr(champion, 'run_id', None)
            if champion_run_id and self._champion_run_id != champion_run_id:
                self._champion_run_id = champion_run_id
                self._save_state()

    def _save_state(self):
        """버전 상태 저장"""
        data = {
            "champion_run_id": self._champion_run_id,
            "challenger_run_id": self._challenger_run_id,
            "history": [h.to_dict() for h in self._history[-50:]],  # 최근 50개만 유지
            "updated_at": datetime.now().isoformat(),
        }

        try:
            with open(self._history_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[VersionManager] Failed to save state: {e}")

    @property
    def champion_run_id(self) -> Optional[str]:
        """현재 Champion run_id"""
        return self._champion_run_id

    @property
    def challenger_run_id(self) -> Optional[str]:
        """현재 Challenger run_id"""
        return self._challenger_run_id

    def get_champion(self) -> Optional[Any]:
        """
        Champion 파라미터 조회

        store.get_champion() 사용 (ParamStore API)
        """
        return self.store.get_champion(self.symbol, self.timeframe)

    def get_challenger(self) -> Optional[Any]:
        """
        Challenger 파라미터 조회

        Challenger = ACTIVE이므로 store.get_active() 사용
        """
        return self.store.get_active(self.symbol, self.timeframe)

    def register_challenger(self, run_id: str) -> bool:
        """
        새 Challenger 등록

        Args:
            run_id: 새 파라미터의 run_id

        Returns:
            등록 성공 여부
        """
        # v2.2.1 BUG FIX: set_active() 전에 기존 challenger 정보를 먼저 저장
        # (set_active 후에는 get_challenger()가 새 버전을 반환하므로)
        old_challenger = None
        old_run_id = self._challenger_run_id
        if old_run_id and old_run_id != run_id:
            old_challenger = self.get_challenger()  # 아직 이전 ACTIVE

        # 파라미터 존재 확인: set_active가 성공하면 존재함
        success = self.store.set_active(self.symbol, self.timeframe, run_id)
        if not success:
            print(f"[VersionManager] run_id={run_id} not found or set_active failed")
            return False

        # 기존 challenger를 히스토리에 retired로 기록 (미리 저장해둔 정보 사용)
        if old_run_id and old_run_id != run_id and old_challenger:
            self._history.append(VersionInfo(
                run_id=old_run_id,
                role="retired",
                demoted_at=datetime.now(),
                oos_performance=_get_oos_performance(old_challenger),
                confidence_score=_get_confidence_score(old_challenger),
            ))

        # 새 challenger 등록
        self._challenger_run_id = run_id

        self._save_state()
        print(f"[VersionManager] Registered challenger: {run_id}")

        return True

    def compare_performance(self) -> ComparisonResult:
        """
        Champion vs Challenger 성능 비교

        비교 기준:
        - OOS: confidence.oos_performance (0~1 스케일)
        - Confidence: confidence.confidence_score (0~100 스케일)

        Returns:
            ComparisonResult with should_promote flag
        """
        champion = self.get_champion()
        challenger = self.get_challenger()

        # OOS 성능 추출 (confidence.oos_performance, 0~1)
        champion_oos = _get_oos_performance(champion)
        challenger_oos = _get_oos_performance(challenger)

        # confidence_score 추출 (0~100)
        champion_conf = _get_confidence_score(champion)
        challenger_conf = _get_confidence_score(challenger)

        # 비교 불가 케이스: Challenger 없음
        if not challenger:
            return ComparisonResult(
                champion_run_id=self._champion_run_id,
                challenger_run_id=self._challenger_run_id,
                champion_oos=champion_oos,
                challenger_oos=None,
                champion_confidence=champion_conf,
                challenger_confidence=None,
                oos_improvement=0,
                confidence_improvement=0,
                should_promote=False,
                reason="No challenger registered"
            )

        # Challenger OOS 미측정
        if challenger_oos is None:
            oos_pending = _is_oos_pending(challenger)

            return ComparisonResult(
                champion_run_id=self._champion_run_id,
                challenger_run_id=self._challenger_run_id,
                champion_oos=champion_oos,
                challenger_oos=None,
                champion_confidence=champion_conf,
                challenger_confidence=challenger_conf,
                oos_improvement=0,
                confidence_improvement=0,
                should_promote=False,
                reason="Challenger OOS not measured yet" if oos_pending else "Challenger OOS pending retry"
            )

        # Champion이 없는 경우 (첫 번째 버전) → 자동 승격
        if not champion or champion_oos is None:
            return ComparisonResult(
                champion_run_id=self._champion_run_id,
                challenger_run_id=self._challenger_run_id,
                champion_oos=champion_oos,
                challenger_oos=challenger_oos,
                champion_confidence=champion_conf,
                challenger_confidence=challenger_conf,
                oos_improvement=1.0,  # 첫 번째는 무조건 개선
                confidence_improvement=1.0,
                should_promote=True,
                reason="First version - auto promote"
            )

        # 개선율 계산
        # v2.2.1 BUG FIX: None 체크 추가 (TypeError 방지)
        if champion_oos is not None and champion_oos > 0 and challenger_oos is not None:
            oos_improvement = (challenger_oos - champion_oos) / champion_oos
        elif challenger_oos is not None and challenger_oos > 0:
            oos_improvement = 1.0
        else:
            oos_improvement = 0.0

        if champion_conf is not None and champion_conf > 0 and challenger_conf is not None:
            conf_improvement = (challenger_conf - champion_conf) / champion_conf
        else:
            conf_improvement = 0.0

        # 승격 조건 판단
        # v2.2.1 BUG FIX: None 방어 (TypeError 방지)
        if champion_oos is not None and champion_oos > 0 and challenger_oos is not None:
            oos_ratio = challenger_oos / champion_oos
        else:
            oos_ratio = 1.0 if challenger_oos is not None and challenger_oos > 0 else 0.0

        if champion_conf is not None and champion_conf > 0 and challenger_conf is not None:
            conf_ratio = challenger_conf / champion_conf
        else:
            conf_ratio = 1.0 if challenger_conf is not None else 0.0

        should_promote = False
        reason = ""

        if oos_ratio >= self.PROMOTE_OOS_THRESHOLD and conf_ratio >= self.PROMOTE_CONFIDENCE_THRESHOLD:
            should_promote = True
            reason = f"OOS ratio={oos_ratio:.2f} >= {self.PROMOTE_OOS_THRESHOLD}, Conf ratio={conf_ratio:.2f} >= {self.PROMOTE_CONFIDENCE_THRESHOLD}"
        elif oos_ratio < self.ROLLBACK_OOS_THRESHOLD:
            reason = f"OOS significantly worse: ratio={oos_ratio:.2f} < {self.ROLLBACK_OOS_THRESHOLD} (recommend rollback)"
        else:
            reason = f"OOS ratio={oos_ratio:.2f}, Conf ratio={conf_ratio:.2f} - not enough improvement"

        return ComparisonResult(
            champion_run_id=self._champion_run_id,
            challenger_run_id=self._challenger_run_id,
            champion_oos=champion_oos,
            challenger_oos=challenger_oos,
            champion_confidence=champion_conf,
            challenger_confidence=challenger_conf,
            oos_improvement=oos_improvement,
            confidence_improvement=conf_improvement,
            should_promote=should_promote,
            reason=reason
        )

    def promote_challenger(self) -> bool:
        """
        Challenger를 Champion으로 승격

        Returns:
            승격 성공 여부
        """
        if not self._challenger_run_id:
            print("[VersionManager] No challenger to promote")
            return False

        challenger = self.get_challenger()
        if not challenger:
            print("[VersionManager] Challenger not found")
            return False

        # 기존 champion을 retired로 이동
        if self._champion_run_id:
            old_champion = self.get_champion()
            if old_champion:
                self._history.append(VersionInfo(
                    run_id=self._champion_run_id,
                    role="retired",
                    demoted_at=datetime.now(),
                    oos_performance=_get_oos_performance(old_champion),
                    confidence_score=_get_confidence_score(old_champion),
                ))

        # Challenger -> Champion
        self._champion_run_id = self._challenger_run_id
        self._challenger_run_id = None

        # 히스토리에 승격 기록
        self._history.append(VersionInfo(
            run_id=self._champion_run_id,
            role="champion",
            promoted_at=datetime.now(),
            oos_performance=_get_oos_performance(challenger),
            confidence_score=_get_confidence_score(challenger),
        ))

        # param_store에 is_champion 마킹
        self.store.mark_as_champion(self.symbol, self.timeframe, self._champion_run_id)

        self._save_state()
        print(f"[VersionManager] Promoted {self._champion_run_id} to Champion")

        return True

    def rollback_to_champion(self) -> bool:
        """
        Challenger 폐기하고 Champion 유지

        Returns:
            롤백 성공 여부
        """
        if not self._challenger_run_id:
            print("[VersionManager] No challenger to rollback")
            return False

        # Challenger를 retired로 기록
        challenger = self.get_challenger()
        if challenger:
            self._history.append(VersionInfo(
                run_id=self._challenger_run_id,
                role="retired",
                demoted_at=datetime.now(),
                oos_performance=_get_oos_performance(challenger),
                confidence_score=_get_confidence_score(challenger),
            ))

        rolled_back = self._challenger_run_id
        self._challenger_run_id = None

        # Champion이 있으면 ACTIVE로 복원
        if self._champion_run_id:
            self.store.set_active(self.symbol, self.timeframe, self._champion_run_id)

        self._save_state()
        print(f"[VersionManager] Rolled back challenger {rolled_back}, Champion {self._champion_run_id} remains active")

        return True

    def auto_decide(self) -> Tuple[str, ComparisonResult]:
        """
        자동 승격/롤백 결정 및 실행

        Returns:
            (action, comparison_result)
            action: "promoted" | "rolled_back" | "kept" | "waiting" | "no_challenger"
        """
        comparison = self.compare_performance()

        if not self._challenger_run_id:
            return ("no_challenger", comparison)

        if comparison.challenger_oos is None:
            return ("waiting", comparison)

        if comparison.should_promote:
            self.promote_challenger()
            return ("promoted", comparison)

        # 롤백 조건 체크
        # v2.2.1 BUG FIX: truthy 대신 is not None 사용 (0.0도 유효한 값)
        if comparison.champion_oos is not None and comparison.champion_oos > 0 and comparison.challenger_oos is not None:
            oos_ratio = comparison.challenger_oos / comparison.champion_oos
            if oos_ratio < self.ROLLBACK_OOS_THRESHOLD:
                self.rollback_to_champion()
                return ("rolled_back", comparison)

        return ("kept", comparison)

    def get_version_history(self, limit: int = 20) -> List[Dict]:
        """
        버전 히스토리 조회

        Args:
            limit: 반환할 최대 개수

        Returns:
            최근 버전 히스토리
        """
        return [h.to_dict() for h in self._history[-limit:]]

    def get_status(self) -> Dict:
        """현재 상태 요약"""
        champion = self.get_champion()
        challenger = self.get_challenger()

        return {
            "champion": {
                "run_id": self._champion_run_id,
                "oos_performance": _get_oos_performance(champion),
                "confidence_score": _get_confidence_score(champion),
            } if self._champion_run_id else None,
            "challenger": {
                "run_id": self._challenger_run_id,
                "oos_performance": _get_oos_performance(challenger),
                "confidence_score": _get_confidence_score(challenger),
            } if self._challenger_run_id else None,
            "history_count": len(self._history),
        }


def get_version_manager(
    symbol: str = "BTCUSDT",
    market: str = "futures",
    timeframe: str = "15m"
) -> ParamVersionManager:
    """
    ParamVersionManager 팩토리 함수

    Args:
        symbol: 심볼
        market: 마켓 타입
        timeframe: 타임프레임

    Returns:
        ParamVersionManager 인스턴스
    """
    # v2.1.1: market 키워드 인자로 전달 (시그니처 일치)
    store = get_param_store(market=market)
    return ParamVersionManager(store, symbol, market, timeframe)
