"""
파라미터 저장소 모듈 (v2)
=========================

최적화된 파라미터 저장/로드
- JSON 파일 기반 저장
- 심볼/타임프레임/마켓별 관리
- 히스토리 보관
- 신뢰도 점수 지원

v2 변경사항:
- schema_version 필드 추가
- market 필드 추가 (spot/futures)
- unit 필드 추가 (minutes)
- confidence 필드 추가 (신뢰도)
- performance 필드 구조화
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, List, Any
from pathlib import Path
import json

# 현재 스키마 버전
SCHEMA_VERSION = 2


@dataclass
class ParamConfidence:
    """
    파라미터 신뢰도 지표 (v2.1 - sensitivity 추가)

    GPT 지적 GAP-7 반영:
    - n_complete_cycles: 분석에 사용된 완전한 사이클 수
    - n_high_purity_cycles: 고순도 사이클 수
    - fold_consistency: 폴드별 파라미터 일관성 (표준편차 역수)
    - fft_reliability: FFT 분석 신뢰도 (peak_ratio)

    v2.1 추가:
    - sensitivity_score: 파라미터 민감도 (0~1, 낮을수록 안정적)
    - stability_grade: 안정성 등급 (A~F)
    """
    # 기존 지표
    overfit_ratio: float = 0.0       # test_score / train_score (1.0이 이상적)
    cv_consistency: float = 0.0      # Cross-validation 일관성 (0~1)
    temporal_stability: float = 0.0  # 시간 안정성 (0~1)
    oos_performance: Optional[float] = None  # Out-of-Sample 성능 (다음 주에 업데이트)

    # v2 추가 지표 (GAP-7 보강)
    n_complete_cycles: int = 0       # 분석에 사용된 완전한 사이클 수
    n_high_purity_cycles: int = 0    # 고순도 사이클 수 (purity >= 0.6)
    fold_consistency: float = 0.0    # 폴드별 robust_params 일관성 (0~1)
    fft_reliability: float = 0.0     # FFT peak_ratio (0~1)

    # v2.1 추가: 민감도 분석
    sensitivity_score: float = 0.0   # 파라미터 민감도 (0~1, 낮을수록 좋음)
    stability_grade: str = ""        # 안정성 등급 (A~F)
    unstable_params: List[str] = field(default_factory=list)  # 불안정 파라미터 목록

    # v2.1.1 추가: OOS 재시도 지원
    oos_pending: bool = False        # True면 OOS 재시도 필요 (missing_data 등)
    oos_skip_reason: str = ""        # OOS 스킵 사유 ("missing_data", "error" 등)

    @property
    def confidence_score(self) -> float:
        """
        종합 신뢰도 점수 (0~100)

        v2.1.1 계산 (OOS 가중치 30점으로 상향):
        - overfit_score * 10           # 과적합 비율 (15→10)
        - cv_consistency * 5           # CV 일관성 (10→5)
        - temporal_stability * 5       # 시간 안정성 (10→5)
        - cycle_quality * 10           # 사이클 품질 (15→10)
        - fold_consistency * 5         # 폴드 일관성 (10→5)
        - fft_reliability * 5          # FFT 신뢰도 (10→5)
        - stability_score * 15         # 민감도 안정성 (20→15)
        - oos_score * 30               # OOS 성능 (10→30) ★핵심★
        - missing_oos_penalty * 15     # OOS 미측정 시 패널티

        OOS가 핵심:
        - OOS 있으면: 최대 30점 획득 가능
        - OOS 없으면: 15점 패널티 (70점 만점)
        """
        # 과적합 점수: overfit_ratio가 1에 가까울수록 좋음
        overfit_score = max(0, 1 - abs(1 - self.overfit_ratio)) if self.overfit_ratio > 0 else 0

        # 사이클 품질 점수: 최소 10개 이상이면 만점, 5개 미만이면 0점
        if self.n_complete_cycles >= 10:
            cycle_quality = 1.0
        elif self.n_complete_cycles >= 5:
            cycle_quality = (self.n_complete_cycles - 5) / 5  # 5~10 선형
        else:
            cycle_quality = 0.0

        # 고순도 비율 보너스
        if self.n_complete_cycles > 0:
            purity_ratio = self.n_high_purity_cycles / self.n_complete_cycles
            cycle_quality = cycle_quality * 0.7 + purity_ratio * 0.3

        # 민감도 → 안정성 점수 (sensitivity가 낮을수록 안정적 = 높은 점수)
        stability_score = max(0, 1 - self.sensitivity_score)

        # OOS 점수 계산 (30점 만점)
        if self.oos_performance is not None:
            oos_score = self.oos_performance * 30  # 0~1 → 0~30
            missing_oos_penalty = 0
        else:
            oos_score = 0
            missing_oos_penalty = 15  # OOS 없으면 15점 패널티

        # 기본 점수 (OOS 제외 55점 만점)
        base_score = (
            overfit_score * 10 +
            self.cv_consistency * 5 +
            self.temporal_stability * 5 +
            cycle_quality * 10 +
            self.fold_consistency * 5 +
            self.fft_reliability * 5 +
            stability_score * 15
        )

        # 최종 점수 = 기본(55) + OOS(30) - 패널티(15)
        # v2.1.1: 음수 방지 clamp
        return max(0.0, base_score + oos_score - missing_oos_penalty)

    def to_dict(self) -> Dict:
        return {
            "overfit_ratio": self.overfit_ratio,
            "cv_consistency": self.cv_consistency,
            "temporal_stability": self.temporal_stability,
            "oos_performance": self.oos_performance,
            "n_complete_cycles": self.n_complete_cycles,
            "n_high_purity_cycles": self.n_high_purity_cycles,
            "fold_consistency": self.fold_consistency,
            "fft_reliability": self.fft_reliability,
            "sensitivity_score": self.sensitivity_score,
            "stability_grade": self.stability_grade,
            "unstable_params": self.unstable_params,
            "oos_pending": self.oos_pending,
            "oos_skip_reason": self.oos_skip_reason,
            "confidence_score": self.confidence_score
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ParamConfidence":
        return cls(
            overfit_ratio=data.get("overfit_ratio", 0.0),
            cv_consistency=data.get("cv_consistency", 0.0),
            temporal_stability=data.get("temporal_stability", 0.0),
            oos_performance=data.get("oos_performance"),
            n_complete_cycles=data.get("n_complete_cycles", 0),
            n_high_purity_cycles=data.get("n_high_purity_cycles", 0),
            fold_consistency=data.get("fold_consistency", 0.0),
            fft_reliability=data.get("fft_reliability", 0.0),
            sensitivity_score=data.get("sensitivity_score", 0.0),
            stability_grade=data.get("stability_grade", ""),
            unstable_params=data.get("unstable_params", []),
            oos_pending=data.get("oos_pending", False),
            oos_skip_reason=data.get("oos_skip_reason", ""),
        )


@dataclass
class ParamPerformance:
    """파라미터 성능 지표"""
    train: Dict[str, float] = field(default_factory=dict)   # {"return": 12.5, "sharpe": 1.8, "mdd": -5.2}
    val: Dict[str, float] = field(default_factory=dict)     # {"return": 8.3, "sharpe": 1.4, "mdd": -4.1}
    oos: Optional[Dict[str, float]] = None                  # 다음 주에 업데이트

    def to_dict(self) -> Dict:
        return {
            "train": self.train,
            "val": self.val,
            "oos": self.oos
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ParamPerformance":
        return cls(
            train=data.get("train", {}),
            val=data.get("val", {}),
            oos=data.get("oos")
        )


@dataclass
class OptimizedParams:
    """
    최적화된 파라미터 (v2)

    v2 필드:
    - schema_version: 스키마 버전
    - market: spot 또는 futures
    - unit: 시간 파라미터 단위 (minutes)
    - confidence: 신뢰도 지표
    - performance: 성능 지표 (구조화)
    """
    symbol: str
    timeframe: str
    params: Dict[str, Any]

    # v2 필수 필드
    schema_version: int = SCHEMA_VERSION
    market: str = "spot"
    unit: str = "minutes"

    # 메타데이터
    created_at: datetime = field(default_factory=datetime.now)
    valid_until: Optional[datetime] = None
    optimization_method: str = "walk_forward"

    # v2: 신뢰도 및 성능
    confidence: ParamConfidence = field(default_factory=ParamConfidence)
    performance: ParamPerformance = field(default_factory=ParamPerformance)

    # 레거시 호환 (deprecated)
    train_score: float = 0.0
    test_score: float = 0.0
    overfit_ratio: float = 0.0

    # 추가 정보
    n_folds: int = 0
    data_period: str = ""

    def to_dict(self) -> Dict:
        return {
            "schema_version": self.schema_version,
            "market": self.market,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "unit": self.unit,
            "params": self.params,
            "created_at": self.created_at.isoformat(),
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "optimization_method": self.optimization_method,
            "confidence": self.confidence.to_dict(),
            "performance": self.performance.to_dict(),
            "n_folds": self.n_folds,
            "data_period": self.data_period
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "OptimizedParams":
        """
        딕셔너리에서 OptimizedParams 생성
        v1 레거시 포맷도 지원
        """
        schema_version = data.get("schema_version", 1)

        # v2 필드
        if schema_version >= 2:
            confidence = ParamConfidence.from_dict(data.get("confidence", {}))
            performance = ParamPerformance.from_dict(data.get("performance", {}))
        else:
            # v1 레거시: train_score/test_score → confidence/performance 변환
            train_score = data.get("train_score", 0.0)
            test_score = data.get("test_score", 0.0)
            overfit = test_score / train_score if train_score > 0 else 0.0

            confidence = ParamConfidence(overfit_ratio=overfit)
            performance = ParamPerformance(
                train={"score": train_score},
                val={"score": test_score}
            )

        return cls(
            schema_version=schema_version,
            market=data.get("market", "spot"),
            symbol=data["symbol"],
            timeframe=data["timeframe"],
            unit=data.get("unit", "bars"),  # v1은 bars 단위였음
            params=data["params"],
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            valid_until=datetime.fromisoformat(data["valid_until"]) if data.get("valid_until") else None,
            optimization_method=data.get("optimization_method", "walk_forward"),
            confidence=confidence,
            performance=performance,
            train_score=data.get("train_score", 0.0),
            test_score=data.get("test_score", 0.0),
            overfit_ratio=data.get("overfit_ratio", 0.0),
            n_folds=data.get("n_folds", 0),
            data_period=data.get("data_period", "")
        )

    @property
    def confidence_score(self) -> float:
        """신뢰도 점수 (0~100)"""
        return self.confidence.confidence_score

    def is_reliable(self, min_score: float = 60.0) -> bool:
        """신뢰도 기준 충족 여부"""
        return self.confidence_score >= min_score


class ParamStore:
    """
    파라미터 저장소 (v2)

    최적화된 파라미터를 저장하고 로드하는 인터페이스
    - JSON 파일 기반 저장
    - 심볼/타임프레임/마켓별 관리
    - 히스토리 보관
    - 신뢰도 기반 파라미터 선택

    v2 변경:
    - market별 저장 디렉토리 분리
    - 신뢰도 기반 load_reliable() 추가
    """

    def __init__(self, store_dir: str = None, market: str = "spot"):
        """
        Args:
            store_dir: 파라미터 저장 디렉토리 (기본: results/params/{market})
            market: 마켓 타입 ("spot" 또는 "futures")
        """
        self.market = market

        if store_dir:
            self.store_dir = Path(store_dir)
        else:
            base = Path(__file__).parent.parent.parent.parent
            self.store_dir = base / "results" / "params" / market

        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self.store_dir / "index.json"
        self._load_index()

    def _load_index(self):
        """인덱스 파일 로드"""
        if self._index_file.exists():
            with open(self._index_file, "r", encoding="utf-8") as f:
                self._index = json.load(f)
        else:
            self._index = {}

    def _save_index(self):
        """인덱스 파일 저장"""
        with open(self._index_file, "w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=2, ensure_ascii=False)

    def _make_key(self, symbol: str, timeframe: str) -> str:
        """심볼/타임프레임 키 생성"""
        symbol_clean = symbol.replace("/", "-").replace(":", "-")
        return f"{symbol_clean}_{timeframe}"

    def save(self, optimized: OptimizedParams) -> str:
        """
        최적화된 파라미터 저장

        Args:
            optimized: OptimizedParams 객체

        Returns:
            run_id (타임스탬프 형식: YYYYmmdd_HHMMSS)
            - set_active()에서 이 run_id로 ACTIVE 지정
            - 파일명: {key}_{run_id}.json
        """
        key = self._make_key(optimized.symbol, optimized.timeframe)
        timestamp = optimized.created_at.strftime("%Y%m%d_%H%M%S")

        # 파일명: BTC-USDT_5m_20251231_120000.json
        filename = f"{key}_{timestamp}.json"
        filepath = self.store_dir / filename

        # JSON 저장
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(optimized.to_dict(), f, indent=2, ensure_ascii=False)

        # 인덱스 업데이트 (기존 active 정보 보존)
        # v2.1.1: 전체 덮어쓰기 대신 update()로 active 유지
        if key not in self._index:
            self._index[key] = {}
        self._index[key].update({
            "latest_file": filename,
            "updated_at": timestamp
        })
        self._save_index()

        print(f"[ParamStore] Saved: {filepath}")
        # v2.1: run_id 반환 (파일경로가 아닌 타임스탬프)
        return timestamp

    def load(self, symbol: str, timeframe: str) -> Optional[OptimizedParams]:
        """
        최신 최적화 파라미터 로드

        Args:
            symbol: 심볼 (예: "BTC/USDT")
            timeframe: 타임프레임 (예: "5m")

        Returns:
            OptimizedParams 또는 None
        """
        key = self._make_key(symbol, timeframe)

        if key not in self._index:
            print(f"[ParamStore] No params found for {symbol} {timeframe}")
            return None

        filename = self._index[key]["latest_file"]
        filepath = self.store_dir / filename

        if not filepath.exists():
            print(f"[ParamStore] File not found: {filepath}")
            return None

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        return OptimizedParams.from_dict(data)

    def load_params(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """
        파라미터만 로드 (간편 메서드)

        Args:
            symbol: 심볼
            timeframe: 타임프레임

        Returns:
            파라미터 딕셔너리 또는 None
        """
        optimized = self.load(symbol, timeframe)
        return optimized.params if optimized else None

    def load_reliable(
        self,
        symbol: str,
        timeframe: str,
        min_confidence: float = 60.0,
        fallback_weeks: int = 4,
        default_params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        신뢰도 기반 파라미터 로드

        로직:
        1. 최신 파라미터의 confidence_score >= min_confidence이면 반환
        2. 아니면 최근 N주 히스토리에서 신뢰도 높은 것 찾기
        3. 모두 낮으면 default_params 반환

        Args:
            symbol: 심볼
            timeframe: 타임프레임
            min_confidence: 최소 신뢰도 점수 (0~100)
            fallback_weeks: 히스토리 탐색 범위 (주)
            default_params: 폴백 파라미터 (None이면 None 반환)

        Returns:
            파라미터 딕셔너리 또는 default_params
        """
        # 1. 최신 파라미터 확인
        latest = self.load(symbol, timeframe)
        if latest and latest.confidence_score >= min_confidence:
            print(f"[ParamStore] Using latest params (confidence={latest.confidence_score:.1f})")
            return latest.params

        # 2. 히스토리에서 신뢰도 높은 것 찾기
        history = self.get_history(symbol, timeframe, limit=fallback_weeks)
        for prev in history[1:]:  # 첫 번째는 이미 확인함
            if prev.confidence_score >= min_confidence:
                print(f"[ParamStore] Using historical params from {prev.created_at.date()} "
                      f"(confidence={prev.confidence_score:.1f})")
                return prev.params

        # 3. 모두 낮으면 폴백
        if latest:
            print(f"[ParamStore] Warning: Low confidence ({latest.confidence_score:.1f}), "
                  f"using {'default params' if default_params else 'latest anyway'}")
            return default_params if default_params else latest.params

        print(f"[ParamStore] No params found, using default")
        return default_params

    def load_reliable_as_bars(
        self,
        symbol: str,
        timeframe: str,
        min_confidence: float = 60.0,
        fallback_weeks: int = 4,
        default_params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        신뢰도 기반 파라미터 로드 (봉 단위로 변환)

        v2 단위 변환 경계:
        - 저장: 분(minutes) 단위
        - 로드: 봉(bars) 단위로 변환하여 반환

        Args:
            symbol: 심볼
            timeframe: 타임프레임 (변환에 사용)
            min_confidence: 최소 신뢰도 점수 (0~100)
            fallback_weeks: 히스토리 탐색 범위 (주)
            default_params: 폴백 파라미터 (None이면 None 반환)

        Returns:
            봉 단위로 변환된 파라미터 딕셔너리
        """
        from .param_schema import convert_params_to_bars

        # 분 단위로 로드
        params_minutes = self.load_reliable(
            symbol=symbol,
            timeframe=timeframe,
            min_confidence=min_confidence,
            fallback_weeks=fallback_weeks,
            default_params=default_params
        )

        if params_minutes is None:
            return None

        # 분 → 봉 변환
        params_bars = convert_params_to_bars(params_minutes, timeframe)
        print(f"[ParamStore] Converted to bars: box_window_minutes {params_minutes.get('box_window_minutes')} -> box_L {params_bars.get('box_L')}")

        return params_bars

    def get_history(self, symbol: str, timeframe: str, limit: int = 10) -> List[OptimizedParams]:
        """
        파라미터 히스토리 조회

        Args:
            symbol: 심볼
            timeframe: 타임프레임
            limit: 최대 개수

        Returns:
            OptimizedParams 리스트 (최신순)
        """
        key = self._make_key(symbol, timeframe)
        pattern = f"{key}_*.json"

        files = sorted(self.store_dir.glob(pattern), reverse=True)[:limit]

        history = []
        for filepath in files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                history.append(OptimizedParams.from_dict(data))
            except Exception as e:
                print(f"[ParamStore] Error loading {filepath}: {e}")

        return history

    def list_symbols(self) -> List[str]:
        """저장된 심볼 목록"""
        symbols = set()
        for key in self._index.keys():
            parts = key.rsplit("_", 1)
            if len(parts) == 2:
                symbols.add(parts[0].replace("-", "/"))
        return sorted(symbols)

    def list_timeframes(self, symbol: str) -> List[str]:
        """심볼의 타임프레임 목록"""
        symbol_clean = symbol.replace("/", "-").replace(":", "-")
        timeframes = []
        for key in self._index.keys():
            if key.startswith(symbol_clean + "_"):
                tf = key.split("_")[-1]
                timeframes.append(tf)
        return sorted(timeframes)

    def delete(self, symbol: str, timeframe: str, keep_history: bool = True):
        """
        파라미터 삭제

        Args:
            symbol: 심볼
            timeframe: 타임프레임
            keep_history: True면 인덱스만 삭제, False면 파일도 삭제
        """
        key = self._make_key(symbol, timeframe)

        if key in self._index:
            if not keep_history:
                pattern = f"{key}_*.json"
                for filepath in self.store_dir.glob(pattern):
                    filepath.unlink()
                    print(f"[ParamStore] Deleted: {filepath}")

            del self._index[key]
            self._save_index()

    def export_all(self, output_file: str = None) -> Dict:
        """
        모든 파라미터 내보내기

        Args:
            output_file: 출력 파일 (JSON)

        Returns:
            전체 파라미터 딕셔너리
        """
        all_params = {}

        for key in self._index.keys():
            parts = key.rsplit("_", 1)
            if len(parts) == 2:
                symbol = parts[0].replace("-", "/")
                timeframe = parts[1]
                optimized = self.load(symbol, timeframe)
                if optimized:
                    all_params[key] = optimized.to_dict()

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_params, f, indent=2, ensure_ascii=False)
            print(f"[ParamStore] Exported to: {output_file}")

        return all_params

    # ================================================================
    # ACTIVE 버전 관리 (v2.1 OOS 추적용)
    # ================================================================

    def set_active(self, symbol: str, timeframe: str, run_id: str = None) -> bool:
        """
        실제 사용 중인 파라미터를 ACTIVE로 지정

        Args:
            symbol: 심볼
            timeframe: 타임프레임
            run_id: 실행 ID (None이면 최신 파일 사용)

        Returns:
            성공 여부
        """
        key = self._make_key(symbol, timeframe)

        if key not in self._index:
            print(f"[ParamStore] No params found for {symbol} {timeframe}")
            return False

        # ACTIVE 정보 업데이트
        if "active" not in self._index[key]:
            self._index[key]["active"] = {}

        self._index[key]["active"] = {
            "file": self._index[key]["latest_file"] if run_id is None else f"{key}_{run_id}.json",
            "set_at": datetime.now().isoformat(),
            "run_id": run_id or self._index[key].get("updated_at", ""),
        }
        self._save_index()

        print(f"[ParamStore] Set ACTIVE: {symbol}/{timeframe}")
        return True

    def get_active(self, symbol: str, timeframe: str) -> Optional[OptimizedParams]:
        """
        ACTIVE 파라미터 조회

        Args:
            symbol: 심볼
            timeframe: 타임프레임

        Returns:
            OptimizedParams 또는 None
        """
        key = self._make_key(symbol, timeframe)

        if key not in self._index:
            return None

        active_info = self._index[key].get("active")
        if not active_info:
            # ACTIVE가 없으면 최신 파일 반환
            return self.load(symbol, timeframe)

        filepath = self.store_dir / active_info["file"]
        if not filepath.exists():
            print(f"[ParamStore] ACTIVE file not found: {filepath}")
            return self.load(symbol, timeframe)

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        optimized = OptimizedParams.from_dict(data)
        # run_id 추가
        optimized.run_id = active_info.get("run_id", "")
        return optimized

    def update_oos(
        self,
        symbol: str,
        timeframe: str,
        oos_metrics: Dict[str, Any],
        oos_period: tuple = None,
        degradation: float = None
    ) -> bool:
        """
        OOS 성능 업데이트

        Args:
            symbol: 심볼
            timeframe: 타임프레임
            oos_metrics: OOS 성능 메트릭
            oos_period: (start, end) 튜플
            degradation: 성능 저하율

        Returns:
            성공 여부
        """
        key = self._make_key(symbol, timeframe)

        if key not in self._index:
            print(f"[ParamStore] No params found for {symbol} {timeframe}")
            return False

        # ACTIVE 파일 경로 결정
        active_info = self._index[key].get("active")
        if active_info:
            filepath = self.store_dir / active_info["file"]
        else:
            filepath = self.store_dir / self._index[key]["latest_file"]

        if not filepath.exists():
            print(f"[ParamStore] File not found: {filepath}")
            return False

        # 파일 로드
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # OOS 업데이트
        if "performance" not in data:
            data["performance"] = {"train": {}, "val": {}, "oos": None}

        data["performance"]["oos"] = oos_metrics

        if oos_period:
            data["performance"]["oos_period"] = {
                "start": oos_period[0].isoformat() if hasattr(oos_period[0], 'isoformat') else str(oos_period[0]),
                "end": oos_period[1].isoformat() if hasattr(oos_period[1], 'isoformat') else str(oos_period[1]),
            }

        # confidence 업데이트 (oos_performance 반영)
        if "confidence" not in data:
            data["confidence"] = {}

        # v2.1.1: missing_data는 oos_performance를 None으로 유지 (패널티 적용)
        oos_status = oos_metrics.get("status")
        oos_sharpe = oos_metrics.get("sharpe_ratio", 0)  # 미리 선언 (UnboundLocalError 방지)

        if oos_status == "missing_data":
            # 데이터 부족: oos_performance는 None 유지 → 15점 패널티 적용됨
            # oos_pending 플래그로 재시도 가능 표시
            data["confidence"]["oos_performance"] = None
            data["confidence"]["oos_pending"] = True
            data["confidence"]["oos_skip_reason"] = "missing_data"
        else:
            # OOS 성능 점수 계산 (sharpe 기준)
            # 정규화: sharpe 2.0 이상이면 만점
            oos_score = min(1.0, max(0, oos_sharpe / 2.0))
            data["confidence"]["oos_performance"] = oos_score
            data["confidence"]["oos_pending"] = False
            data["confidence"]["oos_skip_reason"] = ""  # 성공 시 skip_reason 클리어

        if degradation is not None:
            data["confidence"]["oos_degradation"] = degradation

        data["confidence"]["oos_updated_at"] = datetime.now().isoformat()

        # 파일 저장
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"[ParamStore] Updated OOS: {symbol}/{timeframe} (sharpe={oos_sharpe:.4f})")
        return True

    def list_pending_oos(self) -> List[Dict[str, str]]:
        """
        OOS가 아직 채워지지 않은 ACTIVE 파라미터 목록

        Returns:
            [{"symbol": ..., "timeframe": ..., "file": ...}, ...]
        """
        pending = []

        for key, info in self._index.items():
            active_info = info.get("active")
            if not active_info:
                continue

            filepath = self.store_dir / active_info["file"]
            if not filepath.exists():
                continue

            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            oos = data.get("performance", {}).get("oos")
            if oos is None:
                parts = key.rsplit("_", 1)
                if len(parts) == 2:
                    pending.append({
                        "symbol": parts[0].replace("-", "/"),
                        "timeframe": parts[1],
                        "file": active_info["file"],
                    })

        return pending

    # ================================================================
    # Champion/Challenger 버전 관리 (v2.2 A/B Testing)
    # ================================================================

    def mark_as_champion(self, symbol: str, timeframe: str, run_id: str) -> bool:
        """
        파라미터를 Champion으로 마킹

        Champion은 검증 완료된 운영 버전입니다.
        ACTIVE와 달리, Champion은 OOS 테스트를 통과한 버전입니다.

        Args:
            symbol: 심볼
            timeframe: 타임프레임
            run_id: Champion으로 지정할 run_id

        Returns:
            성공 여부
        """
        key = self._make_key(symbol, timeframe)

        if key not in self._index:
            print(f"[ParamStore] No params found for {symbol} {timeframe}")
            return False

        # champion 정보 업데이트
        if "champion" not in self._index[key]:
            self._index[key]["champion"] = {}

        self._index[key]["champion"] = {
            "file": f"{key}_{run_id}.json",
            "run_id": run_id,
            "promoted_at": datetime.now().isoformat(),
        }

        # 파일에도 is_champion 플래그 저장
        filepath = self.store_dir / f"{key}_{run_id}.json"
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            data["is_champion"] = True
            data["promoted_at"] = datetime.now().isoformat()

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        self._save_index()
        print(f"[ParamStore] Marked as Champion: {symbol}/{timeframe} run_id={run_id}")
        return True

    def get_champion(self, symbol: str, timeframe: str) -> Optional[OptimizedParams]:
        """
        Champion 파라미터 조회

        Args:
            symbol: 심볼
            timeframe: 타임프레임

        Returns:
            OptimizedParams 또는 None
        """
        key = self._make_key(symbol, timeframe)

        if key not in self._index:
            return None

        champion_info = self._index[key].get("champion")
        if not champion_info:
            return None

        filepath = self.store_dir / champion_info["file"]
        if not filepath.exists():
            print(f"[ParamStore] Champion file not found: {filepath}")
            return None

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        optimized = OptimizedParams.from_dict(data)
        optimized.run_id = champion_info.get("run_id", "")
        return optimized

    def demote_champion(self, symbol: str, timeframe: str) -> bool:
        """
        Champion 지위 해제

        롤백 시 호출됩니다.

        Args:
            symbol: 심볼
            timeframe: 타임프레임

        Returns:
            성공 여부
        """
        key = self._make_key(symbol, timeframe)

        if key not in self._index:
            return False

        champion_info = self._index[key].get("champion")
        if not champion_info:
            return False

        # 파일에서 is_champion 플래그 제거
        filepath = self.store_dir / champion_info["file"]
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            data["is_champion"] = False
            data["demoted_at"] = datetime.now().isoformat()

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        # 인덱스에서 champion 제거
        del self._index[key]["champion"]
        self._save_index()

        print(f"[ParamStore] Demoted Champion: {symbol}/{timeframe}")
        return True

    def is_champion(self, symbol: str, timeframe: str, run_id: str) -> bool:
        """
        특정 run_id가 Champion인지 확인

        Args:
            symbol: 심볼
            timeframe: 타임프레임
            run_id: 확인할 run_id

        Returns:
            Champion 여부
        """
        key = self._make_key(symbol, timeframe)

        if key not in self._index:
            return False

        champion_info = self._index[key].get("champion")
        if not champion_info:
            return False

        return champion_info.get("run_id") == run_id

    @property
    def _base_path(self) -> Path:
        """버전 관리에서 사용하는 기본 경로"""
        return self.store_dir


# 전역 싱글톤 인스턴스 (market별 분리)
_stores: Dict[str, ParamStore] = {}


def get_param_store(store_dir: str = None, market: str = "spot") -> ParamStore:
    """
    기본 ParamStore 인스턴스 반환 (market별 분리)

    Args:
        store_dir: 저장 디렉토리 (기본: results/params/{market})
        market: 마켓 타입 ("spot" 또는 "futures")

    Note:
        v2.1.1: market별로 별도 싱글톤 관리
        - spot/futures 각각 독립적인 ParamStore 인스턴스
    """
    global _stores

    # store_dir이 명시되면 항상 새로 생성
    if store_dir:
        return ParamStore(store_dir, market=market)

    # market별 싱글톤
    if market not in _stores:
        _stores[market] = ParamStore(store_dir, market=market)

    return _stores[market]


def save_optimized_params(
    symbol: str,
    timeframe: str,
    params: Dict[str, Any],
    train_score: float = 0.0,
    test_score: float = 0.0,
    metrics: Optional[Dict[str, Any]] = None,
    source: str = "walk_forward",
    unit: str = "minutes",
    market: str = "spot",
    confidence: Optional[ParamConfidence] = None,
    performance: Optional[ParamPerformance] = None,
    **kwargs
) -> str:
    """
    간편 저장 함수 (v2)

    Args:
        symbol: 심볼
        timeframe: 타임프레임
        params: 파라미터 딕셔너리
        train_score: 훈련 점수 (레거시)
        test_score: 테스트 점수 (레거시)
        metrics: 성능 메트릭 {"avg_train_score", "avg_test_score", "overfit_ratio"}
        source: 최적화 소스 (예: "adaptive_tuning_v2")
        unit: 파라미터 단위 ("minutes" 또는 "bars")
        market: 마켓 타입 ("spot" 또는 "futures")
        confidence: ParamConfidence 객체
        performance: ParamPerformance 객체
        **kwargs: 추가 메타데이터

    Returns:
        run_id (타임스탬프 형식: YYYYmmdd_HHMMSS)
    """
    store = get_param_store(market=market)

    # metrics에서 confidence/performance 구성
    if confidence is None and metrics:
        overfit = metrics.get("overfit_ratio", 0.0)
        confidence = ParamConfidence(
            overfit_ratio=overfit,
            cv_consistency=metrics.get("cv_consistency", 0.0),
            temporal_stability=metrics.get("temporal_stability", 0.0),
            fold_consistency=metrics.get("fold_consistency", 0.0),
            fft_reliability=metrics.get("fft_reliability", 0.0),
            n_complete_cycles=metrics.get("n_complete_cycles", 0),
            n_high_purity_cycles=metrics.get("n_high_purity_cycles", 0),
        )

    if performance is None and metrics:
        performance = ParamPerformance(
            train={"score": metrics.get("avg_train_score", train_score)},
            val={"score": metrics.get("avg_test_score", test_score)}
        )

    # OptimizedParams에 전달할 인자만 필터링
    valid_kwargs = {}
    valid_fields = {"n_folds", "data_period", "valid_until", "optimization_method"}
    for k, v in kwargs.items():
        if k in valid_fields:
            valid_kwargs[k] = v

    optimized = OptimizedParams(
        symbol=symbol,
        timeframe=timeframe,
        params=params,
        schema_version=SCHEMA_VERSION,
        market=market,
        unit=unit,
        optimization_method=source,
        confidence=confidence or ParamConfidence(),
        performance=performance or ParamPerformance(),
        train_score=train_score,
        test_score=test_score,
        **valid_kwargs
    )

    return store.save(optimized)


def load_optimized_params(symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
    """
    간편 로드 함수

    Args:
        symbol: 심볼
        timeframe: 타임프레임

    Returns:
        파라미터 딕셔너리 또는 None
    """
    store = get_param_store()
    return store.load_params(symbol, timeframe)
