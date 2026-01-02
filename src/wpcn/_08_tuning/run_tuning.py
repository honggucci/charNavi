"""
Adaptive Tuning Pipeline (v2 - GPT 지적 반영)
=============================================

v2 사이클 분석 + Walk-Forward 최적화 통합 파이프라인

GPT 지적 반영:
1. hints는 param_space와 분리
2. m_freeze <= box_L 제약 검증
3. build_theta_and_config() 어댑터 사용
4. complete + high_purity 사이클만 사용
5. validate_params()로 invalid 조합 차단

Usage:
    python -m wpcn._08_tuning.run_tuning --symbol BTC-USDT --days 90 --market spot
"""
from __future__ import annotations
import argparse
from datetime import datetime
import json
from pathlib import Path

# 같은 폴더 내 모듈 import
from .cycle_analyzer import run_analysis, load_theta
from .adaptive_space import (
    generate_adaptive_space,
    get_adaptive_param_space,
    build_theta_and_config,
    validate_params,
    create_constrained_sampler,
    ParamHints,
)
from .walk_forward import (
    WalkForwardConfig,
    WalkForwardOptimizer,
    run_walk_forward_v8
)
from .param_store import save_optimized_params
from .param_schema import convert_params_to_minutes  # v2: 저장 시 분 단위 변환
from wpcn._00_config.config import PATHS


def create_backtest_func():
    """
    백테스트 함수 생성 (V8 엔진 연동)

    GPT 지적 #3 반영: build_theta_and_config() 사용
    GPT 지적 #2 반영: validate_params()로 invalid 조합 차단
    """
    from wpcn._03_common._01_core.types import Theta, BacktestConfig, BacktestCosts
    from wpcn._04_execution.broker_sim import simulate
    from wpcn._03_common._05_metrics.performance import summarize_performance

    def backtest_func(df, params):
        """백테스트 실행 (제약 조건 검증 포함)"""

        # GPT 지적 #2: invalid 조합 차단
        is_valid, reason = validate_params(params)
        if not is_valid:
            return {
                "total_return": -100,
                "max_drawdown": 100,
                "sharpe_ratio": -10,
                "win_rate": 0,
                "profit_factor": 0,
                "total_trades": 0,
                "_invalid_reason": reason,
            }

        # GPT 지적 #3: 어댑터로 분리
        theta_dict, config_dict = build_theta_and_config(params)

        # Theta 생성 (기본값 병합)
        theta = Theta(
            pivot_lr=theta_dict.get("pivot_lr", 4),
            box_L=theta_dict.get("box_L", 96),
            m_freeze=theta_dict.get("m_freeze", 32),
            atr_len=theta_dict.get("atr_len", 14),
            x_atr=theta_dict.get("x_atr", 0.35),
            m_bw=theta_dict.get("m_bw", 0.10),
            N_reclaim=theta_dict.get("N_reclaim", 3),
            N_fill=theta_dict.get("N_fill", 5),
            F_min=theta_dict.get("F_min", 0.70),
        )

        costs = BacktestCosts()
        bt_config = BacktestConfig(
            tp_pct=config_dict.get("tp_pct", 0.015),
            sl_pct=config_dict.get("sl_pct", 0.010),
        )

        try:
            equity_df, trades_df, _, _ = simulate(df, theta, costs, bt_config)
            perf = summarize_performance(equity_df, trades_df)

            return {
                "total_return": perf.get("total_return_pct", 0),
                "max_drawdown": perf.get("max_drawdown_pct", 0),
                "sharpe_ratio": perf.get("sharpe_ratio", 0),
                "win_rate": perf.get("win_rate", 0),
                "profit_factor": perf.get("profit_factor", 0),
                "total_trades": perf.get("trade_entries", 0),
            }
        except Exception as e:
            print(f"  Backtest error: {e}")
            return {
                "total_return": -100,
                "max_drawdown": 100,
                "sharpe_ratio": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "total_trades": 0,
            }

    return backtest_func


def load_ohlcv_data(symbol: str, timeframe: str, days: int, market: str):
    """OHLCV 데이터 로드"""
    from wpcn._00_config.config import get_data_path
    import pandas as pd

    data_path = get_data_path("bronze", "binance", market, symbol, timeframe)

    if not data_path.exists():
        print(f"[Error] Data path not found: {data_path}")
        return None

    dfs = []
    for year_dir in data_path.iterdir():
        if not year_dir.is_dir():
            continue
        for month_file in year_dir.glob("*.parquet"):
            try:
                df = pd.read_parquet(month_file)
                dfs.append(df)
            except Exception as e:
                print(f"  Failed to load {month_file}: {e}")

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp")

    # 최근 N일 필터링
    cutoff = df["timestamp"].max() - pd.Timedelta(days=days)
    df = df[df["timestamp"] >= cutoff]

    return df


def run_adaptive_tuning(
    symbol: str = "BTC-USDT",
    days: int = 90,
    market: str = "spot",
    timeframe: str = "15m",
    n_iterations: int = 30,
    train_weeks: int = 8,
    test_weeks: int = 2,
    output_dir: str = None
):
    """
    적응형 튜닝 파이프라인 실행

    GPT 지적 반영:
    - (param_space, hints) 분리 처리
    - validate_params로 invalid 조합 차단
    - complete + high_purity 사이클만 사용
    """
    print("="*80)
    print("Adaptive Tuning Pipeline (v2 - GPT 지적 반영)")
    print("="*80)
    print(f"Symbol: {symbol}")
    print(f"Market: {market}")
    print(f"Days: {days}")
    print(f"Timeframe: {timeframe}")
    print("="*80)

    # ============================================================
    # Phase 1: 사이클 분석
    # ============================================================
    print("\n[Phase 1] Running Wyckoff Cycle Analysis...")

    analysis_result = run_analysis(
        symbol=symbol,
        days=days,
        market=market,
        timeframes=[timeframe, "1h", "4h"],
        save_json=True
    )

    if not analysis_result.get("timeframes"):
        print("[Error] No analysis results")
        return None

    # ============================================================
    # Phase 2: (param_space, hints) 분리 생성
    # ============================================================
    print("\n[Phase 2] Generating Adaptive Parameter Space...")

    # GPT 지적 #1, #4 반영: hints 분리 + quality cycles만 사용
    space, hints = generate_adaptive_space(analysis_result, timeframe)
    param_space = space.to_optimizer_space()

    print(f"  Source: {space.source}")
    print(f"  Confidence: {space.confidence}")
    print(f"  box_L range: {space.box_L}")
    print(f"  m_freeze range: {space.m_freeze}")
    print(f"  Hints (분리됨):")
    print(f"    - recommended_box_L: {hints.recommended_box_L}")
    print(f"    - fft_dominant: {hints.fft_dominant_cycle}")
    print(f"    - fft_reliable: {hints.fft_reliable}")

    # ============================================================
    # Phase 3: 데이터 로드
    # ============================================================
    print(f"\n[Phase 3] Loading OHLCV data for {timeframe}...")

    df = load_ohlcv_data(symbol, timeframe, days, market)
    if df is None or len(df) < 5000:
        print(f"[Error] Insufficient data: {len(df) if df is not None else 0} bars")
        return None

    print(f"  Loaded {len(df):,} bars")
    print(f"  Period: {df['timestamp'].min()} ~ {df['timestamp'].max()}")

    # ============================================================
    # Phase 4: Walk-Forward 최적화
    # ============================================================
    print("\n[Phase 4] Running Walk-Forward Optimization...")
    print("  (GPT 지적 반영: validate_params로 invalid 조합 차단)")

    config = WalkForwardConfig(
        train_weeks=train_weeks,
        test_weeks=test_weeks,
        step_weeks=test_weeks,
        optimizer_type="random",
        n_iterations=n_iterations,
        objective="sharpe"
    )

    backtest_func = create_backtest_func()

    wf_result = run_walk_forward_v8(
        df=df,
        backtest_func=backtest_func,
        param_space=param_space,
        config=config,
        output_dir=output_dir or str(PATHS.PROJECT_ROOT / "results" / "tuning")
    )

    # ============================================================
    # Phase 4.5: Sensitivity Analysis (v2.1 추가)
    # ============================================================
    sensitivity_result = None
    if wf_result.robust_params:
        print("\n[Phase 4.5] Running Sensitivity Analysis...")
        try:
            from .sensitivity_analyzer import SensitivityAnalyzer

            analyzer = SensitivityAnalyzer(
                backtest_func=backtest_func,
                score_key="sharpe_ratio"
            )
            sensitivity_result = analyzer.analyze(
                df=df,
                params=wf_result.robust_params,
                verbose=True
            )
            print(f"  Stability Grade: {sensitivity_result.stability_grade}")
            if sensitivity_result.unstable_params:
                print(f"  Unstable Params: {sensitivity_result.unstable_params}")
        except Exception as e:
            print(f"  [Warning] Sensitivity analysis failed: {e}")
            sensitivity_result = None

    # ============================================================
    # Phase 5: 결과 저장
    # ============================================================
    print("\n[Phase 5] Saving Results...")

    run_id = None  # v2.1: run_id 추적용
    if wf_result.robust_params:
        # robust_params 유효성 최종 검증
        is_valid, reason = validate_params(wf_result.robust_params)
        if is_valid:
            # v2: 봉 단위 → 분 단위 변환 후 저장 (단위 변환 경계 중앙집중화)
            params_minutes = convert_params_to_minutes(wf_result.robust_params, timeframe)
            print(f"  Converting to minutes: box_L {wf_result.robust_params.get('box_L')} bars -> {params_minutes.get('box_window_minutes')} min")

            # metrics 구성 (v2.1: sensitivity 포함)
            metrics = {
                "avg_train_score": wf_result.avg_train_score,
                "avg_test_score": wf_result.avg_test_score,
                "overfit_ratio": wf_result.overfit_ratio,
            }

            # sensitivity 결과 추가
            if sensitivity_result:
                metrics["sensitivity_score"] = sensitivity_result.overall_score
                metrics["stability_grade"] = sensitivity_result.stability_grade
                metrics["unstable_params"] = sensitivity_result.unstable_params

            # v2.1: run_id 반환받음 (scheduler에서 set_active에 사용)
            run_id = save_optimized_params(
                symbol=symbol,
                timeframe=timeframe,
                params=params_minutes,  # 분 단위로 저장
                metrics=metrics,
                source="adaptive_tuning_v2.1",
                unit="minutes",  # 단위 명시
                market=market,   # v2: market 전달
            )
            print(f"  Saved to ParamStore: {symbol}/{timeframe} (unit=minutes, run_id={run_id})")
        else:
            print(f"  [Warning] robust_params invalid: {reason}")

    # 통합 결과 JSON 저장
    output_path = PATHS.PROJECT_ROOT / "results" / "tuning" / f"adaptive_tuning_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    final_result = {
        "timestamp": datetime.now().isoformat(),
        "version": "v2.1_sensitivity",
        "run_id": run_id,  # v2.1: scheduler에서 set_active에 사용
        "symbol": symbol,
        "market": market,
        "timeframe": timeframe,
        "analysis": {
            "confidence": space.confidence,
            "box_L_range": space.box_L,
            "m_freeze_range": space.m_freeze,
        },
        "hints": {
            "recommended_box_L": hints.recommended_box_L,
            "fft_dominant_cycle": hints.fft_dominant_cycle,
            "fft_reliable": hints.fft_reliable,
            "median_cycle": hints.median_cycle,
        },
        "optimization": wf_result.to_dict(),
        "robust_params": wf_result.robust_params,
        "sensitivity": sensitivity_result.to_dict() if sensitivity_result else None,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=2, default=str, ensure_ascii=False)

    print(f"  Result saved: {output_path}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*80)
    print("ADAPTIVE TUNING COMPLETE (v2.1 - Sensitivity Analysis)")
    print("="*80)
    print(f"Robust Parameters:")
    for k, v in wf_result.robust_params.items():
        print(f"  {k}: {v}")
    print(f"\nPerformance:")
    print(f"  Avg Train Score: {wf_result.avg_train_score:.4f}")
    print(f"  Avg Test Score:  {wf_result.avg_test_score:.4f}")
    print(f"  Overfit Ratio:   {wf_result.overfit_ratio:.1%}")
    if sensitivity_result:
        print(f"\nSensitivity:")
        print(f"  Overall Score:   {sensitivity_result.overall_score:.3f}")
        print(f"  Stability Grade: {sensitivity_result.stability_grade}")
        print(f"  Stable Params:   {len(sensitivity_result.stable_params)}")
        print(f"  Unstable Params: {len(sensitivity_result.unstable_params)}")
        if sensitivity_result.unstable_params:
            print(f"    Warning: {sensitivity_result.unstable_params}")
    print("="*80)

    return final_result


def main():
    parser = argparse.ArgumentParser(description="Adaptive Tuning Pipeline (v2)")
    parser.add_argument("--symbol", default="BTC-USDT", help="Trading symbol")
    parser.add_argument("--days", type=int, default=90, help="Analysis period (days)")
    parser.add_argument("--market", default="spot", choices=["spot", "futures"])
    parser.add_argument("--timeframe", default="15m", help="Primary timeframe")
    parser.add_argument("--iterations", type=int, default=30, help="Optimization iterations")
    parser.add_argument("--train-weeks", type=int, default=8, help="Training period (weeks)")
    parser.add_argument("--test-weeks", type=int, default=2, help="Test period (weeks)")
    parser.add_argument("--output-dir", default=None, help="Output directory")

    args = parser.parse_args()

    run_adaptive_tuning(
        symbol=args.symbol,
        days=args.days,
        market=args.market,
        timeframe=args.timeframe,
        n_iterations=args.iterations,
        train_weeks=args.train_weeks,
        test_weeks=args.test_weeks,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
