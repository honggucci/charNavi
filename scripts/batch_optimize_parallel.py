"""
BTC-USDT Spot 전체 타임프레임 병렬 최적화
==========================================

5m, 15m, 1h, 4h, 1d, 1w 동시 실행 (multiprocessing)

Usage:
    python scripts/batch_optimize_parallel.py
    python scripts/batch_optimize_parallel.py --optimizer optuna
    python scripts/batch_optimize_parallel.py --days 365
"""
import sys
from pathlib import Path
from datetime import datetime
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import traceback

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# 타임프레임별 최적화 설정
TIMEFRAME_CONFIGS = {
    "5m": {
        "train_weeks": 4,
        "test_weeks": 1,
        "n_iterations": 50,
        "description": "스캘핑 (5분봉)"
    },
    "15m": {
        "train_weeks": 8,
        "test_weeks": 2,
        "n_iterations": 40,
        "description": "단기 트레이딩 (15분봉)"
    },
    "1h": {
        "train_weeks": 12,
        "test_weeks": 3,
        "n_iterations": 35,
        "description": "일중 트레이딩 (1시간봉)"
    },
    "4h": {
        "train_weeks": 16,
        "test_weeks": 4,
        "n_iterations": 30,
        "description": "스윙 트레이딩 (4시간봉)"
    },
    "1d": {
        "train_weeks": 24,
        "test_weeks": 6,
        "n_iterations": 25,
        "description": "포지션 트레이딩 (일봉)"
    },
    "1w": {
        "train_weeks": 52,
        "test_weeks": 12,
        "n_iterations": 20,
        "description": "장기 투자 (주봉)"
    }
}


def get_total_days() -> int:
    """2019년 1월 1일부터 현재까지 일수"""
    start = datetime(2019, 1, 1)
    now = datetime.now()
    return (now - start).days + 30


def run_single_optimization(args_tuple):
    """
    단일 타임프레임 최적화 (multiprocessing worker)

    args_tuple: (symbol, timeframe, market, days, config, optimizer_type)
    """
    symbol, timeframe, market, days, config, optimizer_type = args_tuple

    # 각 프로세스에서 import (fork 후 import 해야 함)
    import sys
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    start_time = datetime.now()

    print(f"\n{'='*60}")
    print(f"[{timeframe}] START - {config['description']}")
    print(f"{'='*60}")
    print(f"  Optimizer: {optimizer_type}")
    print(f"  Train: {config['train_weeks']}w, Test: {config['test_weeks']}w")
    print(f"  Iterations: {config['n_iterations']}")

    try:
        from wpcn._08_tuning.run_tuning import run_adaptive_tuning

        result = run_adaptive_tuning(
            symbol=symbol,
            days=days,
            market=market,
            timeframe=timeframe,
            n_iterations=config["n_iterations"],
            train_weeks=config["train_weeks"],
            test_weeks=config["test_weeks"],
            output_dir=str(PROJECT_ROOT / "results" / "tuning" / "batch_parallel"),
            optimizer_type=optimizer_type,
        )

        elapsed = datetime.now() - start_time

        if result:
            print(f"\n[{timeframe}] DONE in {elapsed}")
            return {
                "status": "success",
                "timeframe": timeframe,
                "elapsed_seconds": elapsed.total_seconds(),
                "run_id": result.get("run_id"),
                "robust_params": result.get("robust_params"),
                "optimization": {
                    "avg_train_score": result.get("optimization", {}).get("avg_train_score"),
                    "avg_test_score": result.get("optimization", {}).get("avg_test_score"),
                    "overfit_ratio": result.get("optimization", {}).get("overfit_ratio"),
                },
                "sensitivity": result.get("sensitivity"),
            }
        else:
            return {
                "status": "failed",
                "timeframe": timeframe,
                "elapsed_seconds": elapsed.total_seconds(),
                "error": "No result returned"
            }

    except Exception as e:
        elapsed = datetime.now() - start_time
        print(f"\n[{timeframe}] ERROR: {e}")
        traceback.print_exc()
        return {
            "status": "error",
            "timeframe": timeframe,
            "elapsed_seconds": elapsed.total_seconds(),
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def main():
    parser = argparse.ArgumentParser(description="BTC-USDT 전체 타임프레임 병렬 최적화")
    parser.add_argument("--symbol", default="BTC-USDT")
    parser.add_argument("--market", default="spot", choices=["spot", "futures"])
    parser.add_argument("--days", type=int, default=None, help="데이터 일수 (기본: 2019년부터)")
    parser.add_argument("--optimizer", default="random", choices=["random", "optuna", "bayesian", "grid"])
    parser.add_argument("--workers", type=int, default=None, help="병렬 워커 수 (기본: CPU 코어 수)")
    parser.add_argument("--timeframes", nargs="+", default=None, help="실행할 타임프레임 (예: 5m 15m 1h)")

    args = parser.parse_args()

    days = args.days or get_total_days()
    workers = args.workers or min(6, cpu_count())
    timeframes = args.timeframes or ["5m", "15m", "1h", "4h", "1d", "1w"]

    # 유효한 타임프레임만 필터
    timeframes = [tf for tf in timeframes if tf in TIMEFRAME_CONFIGS]

    print("="*70)
    print("BTC-USDT 전체 타임프레임 병렬 최적화")
    print("="*70)
    print(f"Symbol: {args.symbol}")
    print(f"Market: {args.market}")
    print(f"Days: {days} (~{days//365}년)")
    print(f"Optimizer: {args.optimizer}")
    print(f"Workers: {workers}")
    print(f"Timeframes: {timeframes}")
    print("="*70)

    # 작업 목록 생성
    tasks = [
        (args.symbol, tf, args.market, days, TIMEFRAME_CONFIGS[tf], args.optimizer)
        for tf in timeframes
    ]

    start_time = datetime.now()

    # 병렬 실행 (ThreadPoolExecutor - Windows 호환)
    print(f"\n[START] {len(tasks)} timeframes with {workers} workers...")

    results_list = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_single_optimization, task): task[1] for task in tasks}
        for future in as_completed(futures):
            tf = futures[future]
            try:
                result = future.result()
                results_list.append(result)
                print(f"  [{tf}] Completed")
            except Exception as e:
                print(f"  [{tf}] Exception: {e}")
                results_list.append({"status": "error", "timeframe": tf, "error": str(e)})

    # 결과 정리
    results = {r["timeframe"]: r for r in results_list}
    elapsed = datetime.now() - start_time

    # 최종 요약
    print("\n" + "="*70)
    print("병렬 최적화 완료")
    print("="*70)
    print(f"Total Time: {elapsed}")
    print(f"\nResults:")

    success_count = 0
    for tf in timeframes:
        r = results.get(tf, {})
        status = r.get("status", "unknown")
        icon = "[OK]" if status == "success" else "[FAIL]"
        elapsed_s = r.get("elapsed_seconds", 0)

        if status == "success":
            success_count += 1
            params = r.get("robust_params", {})
            opt = r.get("optimization", {})
            print(f"  [{tf}] {icon} {elapsed_s:.0f}s")
            print(f"       box_L={params.get('box_L')}, m_freeze={params.get('m_freeze')}, x_atr={params.get('x_atr')}")
            print(f"       train={opt.get('avg_train_score', 0):.3f}, test={opt.get('avg_test_score', 0):.3f}, overfit={opt.get('overfit_ratio', 0):.1%}")
        else:
            print(f"  [{tf}] {icon} {status}: {r.get('error', 'unknown')}")

    print(f"\nSuccess: {success_count}/{len(timeframes)}")

    # 결과 저장
    output_path = PROJECT_ROOT / "results" / "tuning" / "batch_parallel" / f"batch_parallel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "symbol": args.symbol,
        "market": args.market,
        "days": days,
        "optimizer": args.optimizer,
        "workers": workers,
        "elapsed_seconds": elapsed.total_seconds(),
        "success_count": success_count,
        "total_count": len(timeframes),
        "results": results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str, ensure_ascii=False)

    print(f"\nSummary saved: {output_path}")
    print("="*70)

    return results


if __name__ == "__main__":
    main()
