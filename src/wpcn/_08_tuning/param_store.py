"""
파라미터 저장소 모듈
- 최적화된 파라미터 저장/로드
- DB 및 JSON 파일 지원
- 심볼/타임프레임별 관리
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Dict, Optional, List, Any
from pathlib import Path
import json


@dataclass
class OptimizedParams:
    """최적화된 파라미터"""
    symbol: str
    timeframe: str
    params: Dict[str, Any]

    # 메타데이터
    created_at: datetime = field(default_factory=datetime.now)
    valid_until: Optional[datetime] = None
    optimization_method: str = "walk_forward"

    # 성능 지표
    train_score: float = 0.0
    test_score: float = 0.0
    overfit_ratio: float = 0.0

    # 추가 정보
    n_folds: int = 0
    data_period: str = ""

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "params": self.params,
            "created_at": self.created_at.isoformat(),
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "optimization_method": self.optimization_method,
            "train_score": self.train_score,
            "test_score": self.test_score,
            "overfit_ratio": self.overfit_ratio,
            "n_folds": self.n_folds,
            "data_period": self.data_period
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "OptimizedParams":
        return cls(
            symbol=data["symbol"],
            timeframe=data["timeframe"],
            params=data["params"],
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            valid_until=datetime.fromisoformat(data["valid_until"]) if data.get("valid_until") else None,
            optimization_method=data.get("optimization_method", "walk_forward"),
            train_score=data.get("train_score", 0.0),
            test_score=data.get("test_score", 0.0),
            overfit_ratio=data.get("overfit_ratio", 0.0),
            n_folds=data.get("n_folds", 0),
            data_period=data.get("data_period", "")
        )


class ParamStore:
    """
    파라미터 저장소

    최적화된 파라미터를 저장하고 로드하는 인터페이스
    - JSON 파일 기반 저장
    - 심볼/타임프레임별 관리
    - 히스토리 보관
    """

    def __init__(self, store_dir: str = None):
        """
        Args:
            store_dir: 파라미터 저장 디렉토리 (기본: results/params)
        """
        if store_dir:
            self.store_dir = Path(store_dir)
        else:
            base = Path(__file__).parent.parent.parent.parent
            self.store_dir = base / "results" / "params"

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
            저장된 파일 경로
        """
        key = self._make_key(optimized.symbol, optimized.timeframe)
        timestamp = optimized.created_at.strftime("%Y%m%d_%H%M%S")

        # 파일명: BTC-USDT_5m_20251231_120000.json
        filename = f"{key}_{timestamp}.json"
        filepath = self.store_dir / filename

        # JSON 저장
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(optimized.to_dict(), f, indent=2, ensure_ascii=False)

        # 인덱스 업데이트 (최신 파일만 기록)
        self._index[key] = {
            "latest_file": filename,
            "updated_at": timestamp
        }
        self._save_index()

        print(f"[ParamStore] Saved: {filepath}")
        return str(filepath)

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


# 전역 싱글톤 인스턴스
_default_store: Optional[ParamStore] = None


def get_param_store(store_dir: str = None) -> ParamStore:
    """기본 ParamStore 인스턴스 반환"""
    global _default_store
    if _default_store is None or store_dir:
        _default_store = ParamStore(store_dir)
    return _default_store


def save_optimized_params(
    symbol: str,
    timeframe: str,
    params: Dict[str, Any],
    train_score: float = 0.0,
    test_score: float = 0.0,
    **kwargs
) -> str:
    """
    간편 저장 함수

    Args:
        symbol: 심볼
        timeframe: 타임프레임
        params: 파라미터 딕셔너리
        train_score: 훈련 점수
        test_score: 테스트 점수
        **kwargs: 추가 메타데이터

    Returns:
        저장된 파일 경로
    """
    store = get_param_store()

    optimized = OptimizedParams(
        symbol=symbol,
        timeframe=timeframe,
        params=params,
        train_score=train_score,
        test_score=test_score,
        **kwargs
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
