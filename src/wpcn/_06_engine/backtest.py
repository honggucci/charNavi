from __future__ import annotations
from dataclasses import asdict
from typing import Optional, Dict, Any, Tuple

import pandas as pd

from wpcn._03_common._01_core.types import RunConfig, Theta
from wpcn._04_execution.broker_sim import simulate
from wpcn._06_engine.run_manager import make_run_id, ensure_run_dir, dump_resolved_config, save_parquet as save_pq, save_json, write_review_packet
from wpcn._03_common._05_metrics.performance import summarize_performance
from wpcn._07_reporting.review_packet import build_review_prompt
from wpcn._07_reporting.charts import plot_backtest_results

# 기본 Theta (ParamStore 로드 실패 시 폴백)
DEFAULT_THETA = Theta(
    pivot_lr=4, box_L=96, m_freeze=32, atr_len=14,
    x_atr=0.35, m_bw=0.10, N_reclaim=3,
    N_fill=5, F_min=0.70
)


def normalize_symbol(symbol: str) -> str:
    """
    심볼 포맷 정규화 (v2.2.1)

    ParamStore는 "BTC-USDT" 형식을 사용.
    다양한 입력 포맷을 통일된 형식으로 변환.

    Examples:
        "BTCUSDT" -> "BTC-USDT"
        "BTC/USDT" -> "BTC-USDT"
        "BTC:USDT" -> "BTC-USDT"
        "BTC-USDT" -> "BTC-USDT" (그대로)

    Args:
        symbol: 원본 심볼

    Returns:
        정규화된 심볼 (예: "BTC-USDT")
    """
    # 이미 하이픈 형식이면 그대로
    if "-" in symbol:
        return symbol.replace("/", "-").replace(":", "-")

    # 슬래시나 콜론이 있으면 하이픈으로
    if "/" in symbol or ":" in symbol:
        return symbol.replace("/", "-").replace(":", "-")

    # BTCUSDT 형식 → BTC-USDT 변환
    # USDT, USDC, BUSD, USD 순서로 시도 (긴 것부터)
    quote_currencies = ["USDT", "USDC", "BUSD", "USD", "BTC", "ETH", "BNB"]
    for quote in quote_currencies:
        if symbol.endswith(quote):
            base = symbol[:-len(quote)]
            if base:  # base가 비어있지 않으면
                return f"{base}-{quote}"

    # 변환 불가능하면 원본 반환
    return symbol


def load_champion_theta(
    symbol: str,
    timeframe: str,
    market: str = "spot"
) -> Tuple[Optional[Theta], Optional[str]]:
    """
    Champion 파라미터 우선 로드 (v2.2.1 Live Trading 연동)

    Champion은 OOS 테스트를 통과한 검증된 파라미터입니다.
    Live Trading에서는 Champion을 우선 사용해야 합니다.

    v2.2.1: 심볼 자동 정규화 (BTCUSDT → BTC-USDT)

    Args:
        symbol: 심볼 (예: "BTCUSDT", "BTC-USDT", "BTC/USDT" 모두 지원)
        timeframe: 타임프레임 (예: "15m")
        market: 마켓 ("spot" 또는 "futures")

    Returns:
        (Theta 또는 None, run_id 또는 None)
    """
    try:
        from wpcn._08_tuning.param_store import get_param_store
        from wpcn._08_tuning.param_schema import convert_params_to_bars, FIXED_PARAMS

        # v2.2.1: 심볼 정규화
        normalized_symbol = normalize_symbol(symbol)

        store = get_param_store(market=market)

        # Champion 파라미터 조회
        champion = store.get_champion(normalized_symbol, timeframe)
        if champion is None:
            return None, None

        # 분 단위 파라미터 → 봉 단위 변환
        params_bars = convert_params_to_bars(champion.params, timeframe)

        # Theta 생성
        theta = Theta(
            pivot_lr=params_bars.get("pivot_lr", DEFAULT_THETA.pivot_lr),
            box_L=params_bars.get("box_L", DEFAULT_THETA.box_L),
            m_freeze=params_bars.get("m_freeze", DEFAULT_THETA.m_freeze),
            atr_len=params_bars.get("atr_len", FIXED_PARAMS.get("atr_len", 14)),
            x_atr=params_bars.get("x_atr", DEFAULT_THETA.x_atr),
            m_bw=params_bars.get("m_bw", DEFAULT_THETA.m_bw),
            N_reclaim=params_bars.get("N_reclaim", DEFAULT_THETA.N_reclaim),
            N_fill=params_bars.get("N_fill", DEFAULT_THETA.N_fill),
            F_min=params_bars.get("F_min", DEFAULT_THETA.F_min),
        )

        run_id = getattr(champion, 'run_id', None)
        print(f"[BacktestEngine] Loaded Champion Theta: box_L={theta.box_L}, run_id={run_id}")
        return theta, run_id

    except ImportError:
        return None, None
    except Exception as e:
        print(f"[BacktestEngine] Champion load error: {e}")
        return None, None


def load_theta_from_store(
    symbol: str,
    timeframe: str,
    market: str = "spot",
    min_confidence: float = 60.0,
    prefer_champion: bool = True
) -> Theta:
    """
    ParamStore에서 최적화된 Theta 로드 (v2.2.1)

    v2.2.1 변경: 심볼 자동 정규화
    v2.2 변경: Champion 파라미터 우선 로드
    - prefer_champion=True (기본값): Champion 우선, 없으면 ACTIVE/reliable 사용
    - prefer_champion=False: 기존 방식 (load_reliable_as_bars)

    Note: run_id도 필요하면 load_theta_with_run_id() 사용

    Args:
        symbol: 심볼 (예: "BTCUSDT", "BTC-USDT" 모두 지원)
        timeframe: 타임프레임 (예: "15m")
        market: 마켓 ("spot" 또는 "futures")
        min_confidence: 최소 신뢰도 점수 (0~100)
        prefer_champion: Champion 파라미터 우선 사용 여부 (기본 True)

    Returns:
        Theta 객체 (로드 실패 시 DEFAULT_THETA)
    """
    theta, _ = load_theta_with_run_id(symbol, timeframe, market, min_confidence, prefer_champion)
    return theta


def load_theta_with_run_id(
    symbol: str,
    timeframe: str,
    market: str = "spot",
    min_confidence: float = 60.0,
    prefer_champion: bool = True
) -> Tuple[Theta, Optional[str]]:
    """
    ParamStore에서 최적화된 Theta와 run_id 함께 로드 (v2.2.1)

    Live Trading에서 어떤 파라미터로 매매했는지 추적이 필요할 때 사용.
    run_id는 손실 구간 디버깅, 감사 로그 등에 활용.

    Args:
        symbol: 심볼 (예: "BTCUSDT", "BTC-USDT" 모두 지원)
        timeframe: 타임프레임 (예: "15m")
        market: 마켓 ("spot" 또는 "futures")
        min_confidence: 최소 신뢰도 점수 (0~100)
        prefer_champion: Champion 파라미터 우선 사용 여부 (기본 True)

    Returns:
        (Theta, run_id 또는 None)
        - Champion 사용 시: (theta, champion_run_id)
        - reliable 사용 시: (theta, None)
        - DEFAULT_THETA 폴백 시: (DEFAULT_THETA, None)
    """
    # v2.2.1: 심볼 정규화
    normalized_symbol = normalize_symbol(symbol)

    # v2.2: Champion 우선 로드 (Live Trading 권장)
    if prefer_champion:
        champion_theta, run_id = load_champion_theta(normalized_symbol, timeframe, market)
        if champion_theta is not None:
            return champion_theta, run_id
        print(f"[BacktestEngine] No Champion found, falling back to reliable params")

    try:
        from wpcn._08_tuning.param_store import get_param_store
        from wpcn._08_tuning.param_schema import FIXED_PARAMS

        store = get_param_store(market=market)

        # 봉 단위로 변환된 파라미터 로드
        params = store.load_reliable_as_bars(
            symbol=normalized_symbol,
            timeframe=timeframe,
            min_confidence=min_confidence,
            default_params=None  # 폴백은 DEFAULT_THETA 사용
        )

        if params is None:
            print(f"[BacktestEngine] No reliable params found, using DEFAULT_THETA")
            return DEFAULT_THETA, None

        # Theta 생성 (FIXED_PARAMS 병합)
        theta = Theta(
            pivot_lr=params.get("pivot_lr", DEFAULT_THETA.pivot_lr),
            box_L=params.get("box_L", DEFAULT_THETA.box_L),
            m_freeze=params.get("m_freeze", DEFAULT_THETA.m_freeze),
            atr_len=params.get("atr_len", FIXED_PARAMS.get("atr_len", 14)),
            x_atr=params.get("x_atr", DEFAULT_THETA.x_atr),
            m_bw=params.get("m_bw", DEFAULT_THETA.m_bw),
            N_reclaim=params.get("N_reclaim", DEFAULT_THETA.N_reclaim),
            N_fill=params.get("N_fill", DEFAULT_THETA.N_fill),
            F_min=params.get("F_min", DEFAULT_THETA.F_min),
        )

        print(f"[BacktestEngine] Loaded optimized Theta: box_L={theta.box_L}, m_freeze={theta.m_freeze}")
        return theta, None  # reliable params는 run_id 없음

    except ImportError as e:
        print(f"[BacktestEngine] ParamStore import failed: {e}, using DEFAULT_THETA")
        return DEFAULT_THETA, None
    except Exception as e:
        print(f"[BacktestEngine] Error loading params: {e}, using DEFAULT_THETA")
        return DEFAULT_THETA, None

class BacktestEngine:
    def __init__(self, runs_root: str = "runs"):
        self.runs_root = runs_root

    def run(self, df: pd.DataFrame, cfg: RunConfig, theta: Optional[Theta] = None, scalping_mode: bool = False, use_phase_accumulation: bool = True) -> str:
        theta = theta or DEFAULT_THETA
        run_id = make_run_id(cfg.exchange_id, cfg.symbol, cfg.timeframe)
        run_dir = ensure_run_dir(self.runs_root, run_id)

        resolved = {
            "exchange_id": cfg.exchange_id, "symbol": cfg.symbol, "timeframe": cfg.timeframe, "days": cfg.days,
            "costs": asdict(cfg.costs),
            "bt": asdict(cfg.bt),
            "use_tuning": cfg.use_tuning,
            "wf": cfg.wf,
            "mtf": cfg.mtf,
            "theta": asdict(theta),
            "data_path": cfg.data_path,
            "scalping_mode": scalping_mode,
            "use_phase_accumulation": use_phase_accumulation,
        }
        dump_resolved_config(run_dir, resolved)

        equity_df, trades_df, signals_df, nav_df = simulate(
            df, theta, cfg.costs, cfg.bt,
            mtf=cfg.mtf,
            scalping_mode=scalping_mode,
            use_phase_accumulation=use_phase_accumulation
        )

        save_pq(df, run_dir / "candles.parquet")
        save_pq(equity_df, run_dir / "equity.parquet")
        save_pq(trades_df, run_dir / "trades.parquet")
        save_pq(signals_df, run_dir / "signals.parquet")
        save_pq(nav_df, run_dir / "nav.parquet")

        # quick UI snapshot (last bar)
        last = nav_df.iloc[-1].to_dict()
        save_json(last, run_dir / "reports" / "nav_last.json")

        metrics = summarize_performance(equity_df, trades_df)
        save_json(metrics, run_dir / "reports" / "key_metrics.json")

        summary_md = f"""# WPCN Backtest Summary
- Run: {run_id}
- Exchange: {cfg.exchange_id}
- Symbol: {cfg.symbol}
- Timeframe: {cfg.timeframe}
- Bars: {len(df)}

## Key Metrics
```json
{metrics}
```
"""
        prompt = build_review_prompt()
        write_review_packet(run_dir, summary_md, metrics, prompt)

        # Generate charts
        plot_backtest_results(run_dir, df, equity_df, trades_df, nav_df)

        return run_id
