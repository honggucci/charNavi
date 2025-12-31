"""
선물 백테스트 엔진 V3
Futures Backtest Engine V3

V2 대비 개선사항:
- 비용 모델 정확 반영 (수수료, 슬리피지, 펀딩비)
- 동적 포지션 사이징 (Kelly/고정비율)
- ATR 기반 동적 TP/SL
- 포지션 관리 (최대 N개, 일일 손실 한도)
- 성과 지표 확장 (Sharpe, Sortino, Calmar)
- DB 로깅
- 시장 국면 분류
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import pandas as pd

# 기존 모듈
from wpcn.core.types import Theta, BacktestCosts, BacktestConfig
from wpcn.features.indicators import atr, rsi, detect_rsi_divergence, stoch_rsi
from wpcn.wyckoff.phases import detect_wyckoff_phase, DIRECTION_RATIOS
from wpcn.wyckoff.box import box_engine_freeze

# 새 모듈 - 숫자폴더 직접 import 불가로 인한 동적 로드
import sys
import importlib.util
from pathlib import Path

def _import_from_numbered_folder(relative_path: str, module_name: str):
    """숫자로 시작하는 폴더에서 모듈 임포트"""
    base_dir = Path(__file__).parent.parent  # wpcn/
    full_path = base_dir / relative_path
    spec = importlib.util.spec_from_file_location(module_name, full_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# 동적 로드
_cost_model = _import_from_numbered_folder("003_common/006_risk/cost_model.py", "cost_model")
_position_sizing = _import_from_numbered_folder("003_common/006_risk/position_sizing.py", "position_sizing")
_dynamic_sl_tp = _import_from_numbered_folder("003_common/006_risk/dynamic_sl_tp.py", "dynamic_sl_tp")
_metrics = _import_from_numbered_folder("003_common/007_validation/metrics.py", "metrics")
_market_regime = _import_from_numbered_folder("003_common/007_validation/market_regime.py", "market_regime")

# 클래스/함수 추출
BinanceFuturesCost = _cost_model.BinanceFuturesCost
FixedRisk = _position_sizing.FixedRisk
PositionManager = _position_sizing.PositionManager
DynamicSLTP = _dynamic_sl_tp.DynamicSLTP
calculate_metrics = _metrics.calculate_metrics
PerformanceMetrics = _metrics.PerformanceMetrics
MarketRegimeClassifier = _market_regime.MarketRegimeClassifier


@dataclass
class FuturesConfigV3:
    """선물 백테스트 설정 V3"""
    # 레버리지
    leverage: float = 10.0
    margin_mode: str = 'isolated'

    # 리스크 관리
    risk_per_trade: float = 0.01  # 거래당 1% 리스크
    max_position_pct: float = 0.30  # 최대 30%
    max_positions: int = 3  # 최대 동시 포지션
    max_daily_loss_pct: float = 0.05  # 일일 손실 한도 5%

    # 동적 TP/SL
    use_dynamic_sl_tp: bool = True
    min_rr_ratio: float = 1.5  # 최소 손익비
    sl_atr_mult: float = 2.0
    tp_atr_mult: float = 3.0

    # 고정 TP/SL (동적 사용 안할 때)
    tp_pct: float = 0.015
    sl_pct: float = 0.010

    # 시간 청산 (5분봉 기준)
    max_hold_bars: int = 72  # 6시간

    # 진입 조건
    rsi_oversold: float = 40
    rsi_overbought: float = 60
    min_score: float = 4.0

    # 쿨다운
    entry_cooldown: int = 12  # 1시간


@dataclass
class Position:
    """포지션"""
    id: str
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    quantity: float
    margin: float
    leverage: float
    tp_price: float
    sl_price: float
    entry_bar: int
    entry_time: datetime
    signal_score: float
    market_regime: str


class FuturesBacktestV3:
    """
    선물 백테스트 엔진 V3

    사용법:
        engine = FuturesBacktestV3(config)
        result = engine.run(df, theta)
    """

    def __init__(self, config: FuturesConfigV3 = None):
        self.config = config or FuturesConfigV3()

        # 비용 모델
        self.cost_model = BinanceFuturesCost()

        # 포지션 사이징
        self.position_sizer = FixedRisk(
            risk_per_trade=self.config.risk_per_trade,
            max_position_pct=self.config.max_position_pct,
            max_leverage=self.config.leverage
        )

        # 포지션 관리자
        self.position_manager = PositionManager(
            max_positions=self.config.max_positions,
            max_daily_loss_pct=self.config.max_daily_loss_pct
        )

        # 동적 SL/TP
        self.dynamic_sl_tp = DynamicSLTP(
            min_rr_ratio=self.config.min_rr_ratio
        )

        # 시장 국면 분류기
        self.regime_classifier = MarketRegimeClassifier()

        # 상태
        self.positions: Dict[str, Position] = {}
        self.trades: List[Dict] = []
        self.signals: List[Dict] = []
        self.equity_history: List[Dict] = []

    def run(
        self,
        df_5m: pd.DataFrame,
        theta: Theta,
        initial_capital: float = 10000.0,
        save_to_db: bool = False
    ) -> Dict[str, Any]:
        """
        백테스트 실행

        Args:
            df_5m: 5분봉 OHLCV 데이터
            theta: 전략 파라미터
            initial_capital: 초기 자본
            save_to_db: DB에 저장 여부

        Returns:
            {"equity_df": ..., "trades_df": ..., "metrics": ..., "stats": ...}
        """
        # 초기화
        self.positions = {}
        self.trades = []
        self.signals = []
        self.equity_history = []
        self.position_manager.current_positions = []
        self.position_manager.daily_pnl = 0.0

        cash = initial_capital
        last_entry_bar = -100
        current_day = None

        # 15분봉 리샘플링
        df_15m = self._resample_to_15m(df_5m)

        # 지표 계산
        phases_df = detect_wyckoff_phase(df_15m, theta)
        box_df = box_engine_freeze(df_15m, theta)
        div_5m = detect_rsi_divergence(df_5m, rsi_len=14, lookback=20)
        rsi_5m = rsi(df_5m['close'], 14)
        stoch_k, stoch_d = stoch_rsi(df_5m['close'], 14, 14, 3, 3)  # 튜플 반환
        stoch_5m = pd.DataFrame({'stoch_k': stoch_k, 'stoch_d': stoch_d}, index=df_5m.index)
        atr_5m = atr(df_5m, 14)  # DataFrame과 length만 전달

        n = len(df_5m)
        idx = df_5m.index

        for i in range(50, n):
            t = idx[i]
            row = df_5m.iloc[i]
            o, h, l, c = row['open'], row['high'], row['low'], row['close']
            current_atr = atr_5m.iloc[i] if i < len(atr_5m) else c * 0.01

            # 일일 초기화
            if current_day != t.date():
                current_day = t.date()
                self.position_manager.reset_daily()

            # === 포지션 관리 ===
            closed_positions = []
            for pos_id, pos in list(self.positions.items()):
                result = self._check_position_exit(pos, h, l, c, i, t)
                if result:
                    exit_price, exit_reason, pnl, pnl_pct = result
                    cash += pos.margin + pnl
                    self.position_manager.remove_position(pos.symbol, pnl)

                    self.trades.append({
                        'id': pos.id,
                        'symbol': pos.symbol,
                        'side': pos.side,
                        'entry_price': pos.entry_price,
                        'exit_price': exit_price,
                        'quantity': pos.quantity,
                        'tp_price': pos.tp_price,
                        'sl_price': pos.sl_price,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'fee': self._calculate_total_fee(pos, exit_price),
                        'entry_time': pos.entry_time,
                        'exit_time': t,
                        'exit_reason': exit_reason,
                        'signal_score': pos.signal_score,
                        'market_regime': pos.market_regime,
                        'leverage': pos.leverage,
                        'hold_bars': i - pos.entry_bar
                    })
                    closed_positions.append(pos_id)

            for pos_id in closed_positions:
                del self.positions[pos_id]

            # === 펀딩비 처리 (8시간마다) ===
            if i % 96 == 0:  # 5분봉 96봉 = 8시간
                for pos in self.positions.values():
                    funding = self._calculate_funding(pos, c)
                    cash -= funding

            # === 진입 신호 분석 ===
            if i - last_entry_bar >= self.config.entry_cooldown:
                signal = self._analyze_signal(
                    df_5m, df_15m, phases_df, box_df, div_5m,
                    rsi_5m, stoch_5m, i, t
                )

                if signal and signal['action'] != 'none':
                    self.signals.append(signal)

                    # 포지션 오픈 가능 여부 체크
                    can_open, reason = self.position_manager.can_open_position(
                        signal['symbol'], cash
                    )

                    if can_open and cash > 0:
                        # 동적 SL/TP 계산
                        if self.config.use_dynamic_sl_tp:
                            sl_tp = self.dynamic_sl_tp.calculate(
                                df_5m.iloc[max(0, i-50):i+1],
                                c,
                                signal['action']
                            )
                            tp_price = sl_tp.take_profit
                            sl_price = sl_tp.stop_loss
                        else:
                            if signal['action'] == 'long':
                                tp_price = c * (1 + self.config.tp_pct)
                                sl_price = c * (1 - self.config.sl_pct)
                            else:
                                tp_price = c * (1 - self.config.tp_pct)
                                sl_price = c * (1 + self.config.sl_pct)

                        # 포지션 사이징
                        pos_size = self.position_sizer.calculate(
                            cash, c, sl_price, self.config.leverage
                        )

                        if pos_size.quantity > 0:
                            # 비용 적용
                            entry_cost = self.cost_model.calculate_entry_cost(
                                c, pos_size.quantity, signal['action'], 'limit'
                            )

                            # 포지션 생성
                            pos_id = f"{signal['symbol']}_{i}"
                            pos = Position(
                                id=pos_id,
                                symbol=signal['symbol'],
                                side=signal['action'],
                                entry_price=entry_cost.effective_price,
                                quantity=pos_size.quantity,
                                margin=pos_size.position_value / self.config.leverage,
                                leverage=self.config.leverage,
                                tp_price=tp_price,
                                sl_price=sl_price,
                                entry_bar=i,
                                entry_time=t,
                                signal_score=signal['score'],
                                market_regime=signal.get('market_regime', 'unknown')
                            )

                            self.positions[pos_id] = pos
                            self.position_manager.add_position(signal['symbol'])
                            cash -= pos.margin + entry_cost.fee
                            last_entry_bar = i

                            signal['executed'] = True
                        else:
                            signal['executed'] = False
                    else:
                        signal['executed'] = False
                        signal['reject_reason'] = reason

            # === Equity 기록 ===
            equity = self._calculate_equity(cash, c)
            self.equity_history.append({
                'time': t,
                'equity': equity,
                'cash': cash,
                'positions': len(self.positions)
            })

        # 결과 생성
        return self._generate_result(initial_capital, save_to_db)

    def _resample_to_15m(self, df_5m: pd.DataFrame) -> pd.DataFrame:
        """5분봉 -> 15분봉"""
        return df_5m.resample('15min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

    def _check_position_exit(
        self,
        pos: Position,
        high: float,
        low: float,
        close: float,
        bar: int,
        time: datetime
    ) -> Optional[Tuple[float, str, float, float]]:
        """포지션 청산 체크"""
        cfg = self.config

        if pos.side == 'long':
            # 청산가
            liq_price = pos.entry_price * (1 - 1/pos.leverage + 0.005)

            # 청산
            if low <= liq_price:
                pnl = -pos.margin
                return liq_price, 'liquidation', pnl, -100.0

            # 손절
            if low <= pos.sl_price:
                exit_cost = self.cost_model.calculate_exit_cost(
                    pos.sl_price, pos.quantity, 'long', 'market'
                )
                pnl = (exit_cost.effective_price - pos.entry_price) * pos.quantity - exit_cost.fee
                pnl_pct = pnl / pos.margin * 100
                return exit_cost.effective_price, 'sl', pnl, pnl_pct

            # 익절
            if high >= pos.tp_price:
                exit_cost = self.cost_model.calculate_exit_cost(
                    pos.tp_price, pos.quantity, 'long', 'limit'
                )
                pnl = (exit_cost.effective_price - pos.entry_price) * pos.quantity - exit_cost.fee
                pnl_pct = pnl / pos.margin * 100
                return exit_cost.effective_price, 'tp', pnl, pnl_pct

        else:  # short
            liq_price = pos.entry_price * (1 + 1/pos.leverage - 0.005)

            if high >= liq_price:
                pnl = -pos.margin
                return liq_price, 'liquidation', pnl, -100.0

            if high >= pos.sl_price:
                exit_cost = self.cost_model.calculate_exit_cost(
                    pos.sl_price, pos.quantity, 'short', 'market'
                )
                pnl = (pos.entry_price - exit_cost.effective_price) * pos.quantity - exit_cost.fee
                pnl_pct = pnl / pos.margin * 100
                return exit_cost.effective_price, 'sl', pnl, pnl_pct

            if low <= pos.tp_price:
                exit_cost = self.cost_model.calculate_exit_cost(
                    pos.tp_price, pos.quantity, 'short', 'limit'
                )
                pnl = (pos.entry_price - exit_cost.effective_price) * pos.quantity - exit_cost.fee
                pnl_pct = pnl / pos.margin * 100
                return exit_cost.effective_price, 'tp', pnl, pnl_pct

        # 시간 청산
        if bar - pos.entry_bar >= cfg.max_hold_bars:
            exit_cost = self.cost_model.calculate_exit_cost(
                close, pos.quantity, pos.side, 'market'
            )
            if pos.side == 'long':
                pnl = (exit_cost.effective_price - pos.entry_price) * pos.quantity - exit_cost.fee
            else:
                pnl = (pos.entry_price - exit_cost.effective_price) * pos.quantity - exit_cost.fee
            pnl_pct = pnl / pos.margin * 100
            return exit_cost.effective_price, 'timeout', pnl, pnl_pct

        return None

    def _analyze_signal(
        self,
        df_5m: pd.DataFrame,
        df_15m: pd.DataFrame,
        phases_df: pd.DataFrame,
        box_df: pd.DataFrame,
        div_5m: pd.DataFrame,
        rsi_5m: pd.Series,
        stoch_5m: pd.DataFrame,
        i: int,
        t: datetime
    ) -> Optional[Dict]:
        """신호 분석"""
        cfg = self.config
        close = df_5m['close'].iloc[i]
        t_15m = t.floor('15min')

        # 15분봉 정보
        if t_15m in phases_df.index:
            phase = phases_df.loc[t_15m, 'phase']
            direction = phases_df.loc[t_15m, 'direction']
        else:
            return None

        # 5분봉 지표
        curr_rsi = rsi_5m.iloc[i] if i < len(rsi_5m) else 50
        curr_stoch = stoch_5m['stoch_k'].iloc[i] if i < len(stoch_5m) else 50
        bullish_div = div_5m.iloc[i].get('bullish_div', False) if i < len(div_5m) else False
        bearish_div = div_5m.iloc[i].get('bearish_div', False) if i < len(div_5m) else False

        # 시장 국면
        try:
            regime = self.regime_classifier.classify(df_5m.iloc[max(0, i-100):i+1])
            market_regime = regime.trend
        except:
            market_regime = 'unknown'

        # 점수 계산
        long_score = 0.0
        short_score = 0.0
        reasons = {}

        # 15분봉 Wyckoff
        if direction in ['accumulation', 're_accumulation']:
            long_score += 2.0
            reasons['wyckoff'] = 'accumulation'
        elif direction in ['distribution', 're_distribution']:
            short_score += 2.0
            reasons['wyckoff'] = 'distribution'

        # 5분봉 RSI
        if curr_rsi < cfg.rsi_oversold:
            long_score += 1.0
            reasons['rsi'] = f'oversold ({curr_rsi:.1f})'
        elif curr_rsi > cfg.rsi_overbought:
            short_score += 1.0
            reasons['rsi'] = f'overbought ({curr_rsi:.1f})'

        # 다이버전스
        if bullish_div:
            long_score += 1.5
            reasons['divergence'] = 'bullish'
        if bearish_div:
            short_score += 1.5
            reasons['divergence'] = 'bearish'

        # Stochastic
        if curr_stoch < 30:
            long_score += 0.5
        elif curr_stoch > 70:
            short_score += 0.5

        # 시장 국면 보너스
        if market_regime == 'bullish':
            long_score += 0.5
        elif market_regime == 'bearish':
            short_score += 0.5

        # 신호 결정
        action = 'none'
        score = 0.0

        if long_score >= cfg.min_score and long_score > short_score:
            action = 'long'
            score = long_score
        elif short_score >= cfg.min_score and short_score > long_score:
            action = 'short'
            score = short_score

        return {
            'symbol': 'BTC/USDT',  # TODO: 심볼 파라미터화
            'timeframe': '5m',
            'timestamp': t,
            'action': action,
            'score': score,
            'long_score': long_score,
            'short_score': short_score,
            'reason': reasons,
            'tf_alignment': len([v for v in reasons.values() if v]),
            'market_regime': market_regime,
            'executed': False
        }

    def _calculate_funding(self, pos: Position, current_price: float) -> float:
        """펀딩비 계산"""
        position_value = pos.quantity * current_price
        return self.cost_model.calculate_funding(position_value, hours_held=8)

    def _calculate_total_fee(self, pos: Position, exit_price: float) -> float:
        """총 수수료 계산"""
        entry_fee = pos.entry_price * pos.quantity * 0.0002  # Maker
        exit_fee = exit_price * pos.quantity * 0.0004  # Taker (SL은 시장가)
        return entry_fee + exit_fee

    def _calculate_equity(self, cash: float, current_price: float) -> float:
        """현재 자산 계산"""
        equity = cash
        for pos in self.positions.values():
            if pos.side == 'long':
                unrealized = (current_price - pos.entry_price) * pos.quantity
            else:
                unrealized = (pos.entry_price - current_price) * pos.quantity
            equity += pos.margin + unrealized
        return equity

    def _generate_result(
        self,
        initial_capital: float,
        save_to_db: bool
    ) -> Dict[str, Any]:
        """결과 생성"""
        # DataFrame 생성
        equity_df = pd.DataFrame(self.equity_history)
        if len(equity_df) > 0:
            equity_df = equity_df.set_index('time')

        trades_df = pd.DataFrame(self.trades)
        signals_df = pd.DataFrame(self.signals)

        # 성과 지표 계산
        if len(equity_df) > 0 and len(trades_df) > 0:
            metrics = calculate_metrics(
                equity_df['equity'],
                trades_df,
                periods_per_year=365 * 24 * 12  # 5분봉
            )
        else:
            metrics = None

        # 기본 통계
        if len(equity_df) > 0:
            final_equity = equity_df['equity'].iloc[-1]
            total_return = (final_equity - initial_capital) / initial_capital * 100
            running_max = equity_df['equity'].cummax()
            drawdown = (equity_df['equity'] - running_max) / running_max * 100
            max_drawdown = drawdown.min()
        else:
            final_equity = initial_capital
            total_return = 0
            max_drawdown = 0

        if len(trades_df) > 0 and 'pnl' in trades_df.columns:
            wins = trades_df[trades_df['pnl'] > 0]
            win_rate = len(wins) / len(trades_df) * 100
            avg_pnl = trades_df['pnl'].mean()
            total_fees = trades_df['fee'].sum() if 'fee' in trades_df.columns else 0
        else:
            win_rate = 0
            avg_pnl = 0
            total_fees = 0

        stats = {
            'initial_capital': initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'total_trades': len(trades_df),
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'total_fees': total_fees,
            'total_signals': len(signals_df),
            'executed_signals': len(signals_df[signals_df['executed'] == True]) if len(signals_df) > 0 else 0
        }

        # DB 저장
        if save_to_db:
            self._save_to_db(trades_df, signals_df)

        return {
            'equity_df': equity_df,
            'trades_df': trades_df,
            'signals_df': signals_df,
            'metrics': metrics,
            'stats': stats
        }

    def _save_to_db(self, trades_df: pd.DataFrame, signals_df: pd.DataFrame):
        """DB에 저장"""
        try:
            _repository = _import_from_numbered_folder("003_common/008_database/repository.py", "repository")
            _models = _import_from_numbered_folder("003_common/008_database/models.py", "models")
            TradeRepository = _repository.TradeRepository
            SignalRepository = _repository.SignalRepository
            Trade = _models.Trade
            Signal = _models.Signal
            create_tables = _models.create_tables

            create_tables()

            # 거래 저장
            for _, row in trades_df.iterrows():
                trade = Trade(
                    symbol=row['symbol'],
                    side=row['side'],
                    entry_price=row['entry_price'],
                    exit_price=row.get('exit_price'),
                    quantity=row['quantity'],
                    tp_price=row['tp_price'],
                    sl_price=row['sl_price'],
                    pnl=row.get('pnl'),
                    pnl_pct=row.get('pnl_pct'),
                    fee=row.get('fee', 0),
                    entry_time=row['entry_time'],
                    exit_time=row.get('exit_time'),
                    exit_reason=row.get('exit_reason'),
                    signal_score=row['signal_score'],
                    market_regime=row.get('market_regime'),
                    timeframe='5m',
                    strategy_version='v3'
                )
                TradeRepository.insert(trade)

            # 신호 저장
            for _, row in signals_df.iterrows():
                signal = Signal(
                    symbol=row['symbol'],
                    timeframe=row['timeframe'],
                    timestamp=row['timestamp'],
                    action=row['action'],
                    score=row['score'],
                    long_score=row['long_score'],
                    short_score=row['short_score'],
                    reason=row['reason'],
                    tf_alignment=row['tf_alignment'],
                    executed=row['executed']
                )
                SignalRepository.insert(signal)

            print("DB 저장 완료")

        except Exception as e:
            print(f"DB 저장 실패: {e}")
