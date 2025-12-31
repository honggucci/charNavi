"""
시장 국면 분류
- 상승장 / 하락장 / 횡보장
- 고변동성 / 저변동성
"""

from dataclasses import dataclass
from typing import Literal
import pandas as pd
import numpy as np


@dataclass
class MarketRegime:
    """시장 국면"""
    trend: Literal["bullish", "bearish", "ranging"]
    volatility: Literal["low", "medium", "high"]
    strength: float  # 0~1 (1이 가장 강함)
    description: str


class MarketRegimeClassifier:
    """
    시장 국면 분류기

    기준:
    - 추세: EMA 크로스, 가격 위치, ADX
    - 변동성: ATR 백분위
    """

    def __init__(
        self,
        ema_short: int = 20,
        ema_long: int = 50,
        adx_period: int = 14,
        atr_period: int = 14,
        lookback: int = 100
    ):
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.adx_period = adx_period
        self.atr_period = atr_period
        self.lookback = lookback

    def classify(self, df: pd.DataFrame) -> MarketRegime:
        """시장 국면 분류"""
        df = df.copy()

        # 지표 계산
        df = self._add_indicators(df)

        # 최신 값
        last = df.iloc[-1]
        close = last["close"]
        ema_short = last["ema_short"]
        ema_long = last["ema_long"]
        adx = last["adx"]
        atr_pct = last["atr_pct"]

        # 추세 판단
        trend, trend_strength = self._classify_trend(close, ema_short, ema_long, adx)

        # 변동성 판단
        volatility = self._classify_volatility(df, atr_pct)

        # 설명
        description = self._generate_description(trend, volatility, trend_strength, adx)

        return MarketRegime(
            trend=trend,
            volatility=volatility,
            strength=trend_strength,
            description=description
        )

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """지표 추가"""
        # EMA
        df["ema_short"] = df["close"].ewm(span=self.ema_short).mean()
        df["ema_long"] = df["close"].ewm(span=self.ema_long).mean()

        # ATR
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr"] = tr.rolling(self.atr_period).mean()
        df["atr_pct"] = df["atr"] / df["close"]

        # ADX
        df["adx"] = self._calculate_adx(df)

        return df

    def _calculate_adx(self, df: pd.DataFrame) -> pd.Series:
        """ADX 계산"""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # +DM, -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        # TR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Smoothed
        atr = tr.rolling(self.adx_period).mean()
        plus_di = 100 * (plus_dm.rolling(self.adx_period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(self.adx_period).mean() / atr)

        # DX, ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(self.adx_period).mean()

        return adx

    def _classify_trend(
        self,
        close: float,
        ema_short: float,
        ema_long: float,
        adx: float
    ) -> tuple:
        """추세 분류"""
        # ADX < 20: 횡보
        if adx < 20:
            return "ranging", 0.3

        # EMA 관계
        if close > ema_short > ema_long:
            trend = "bullish"
            strength = min(adx / 50, 1.0)  # ADX 50 이상이면 강한 추세
        elif close < ema_short < ema_long:
            trend = "bearish"
            strength = min(adx / 50, 1.0)
        else:
            trend = "ranging"
            strength = 0.3

        return trend, strength

    def _classify_volatility(self, df: pd.DataFrame, current_atr_pct: float) -> str:
        """변동성 분류"""
        # 과거 ATR% 분포
        historical_atr_pct = df["atr_pct"].dropna().tail(self.lookback)

        if len(historical_atr_pct) < 20:
            return "medium"

        percentiles = np.percentile(historical_atr_pct, [33, 66])

        if current_atr_pct < percentiles[0]:
            return "low"
        elif current_atr_pct < percentiles[1]:
            return "medium"
        else:
            return "high"

    def _generate_description(
        self,
        trend: str,
        volatility: str,
        strength: float,
        adx: float
    ) -> str:
        """설명 생성"""
        trend_desc = {
            "bullish": "상승 추세",
            "bearish": "하락 추세",
            "ranging": "횡보 구간"
        }

        vol_desc = {
            "low": "저변동성",
            "medium": "보통 변동성",
            "high": "고변동성"
        }

        strength_desc = "강한" if strength > 0.6 else "보통" if strength > 0.3 else "약한"

        return f"{vol_desc[volatility]}, {strength_desc} {trend_desc[trend]} (ADX: {adx:.1f})"

    def get_regime_stats(
        self,
        df: pd.DataFrame,
        window: int = 20
    ) -> pd.DataFrame:
        """기간별 시장 국면 통계"""
        regimes = []

        for i in range(window, len(df), window):
            subset = df.iloc[i-window:i]
            regime = self.classify(subset)

            regimes.append({
                "start": subset.iloc[0]["timestamp"] if "timestamp" in subset.columns else i-window,
                "end": subset.iloc[-1]["timestamp"] if "timestamp" in subset.columns else i,
                "trend": regime.trend,
                "volatility": regime.volatility,
                "strength": regime.strength
            })

        return pd.DataFrame(regimes)
