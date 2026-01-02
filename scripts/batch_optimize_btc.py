"""
BTC-USDT Spot 전체 타임프레임 배치 최적화
==========================================

2019년부터 현재까지 전체 데이터에 대해 모든 타임프레임별 파라미터 최적화

타임프레임: 5m, 15m, 1h, 4h, 1d, 1w

Usage:
    python scripts/batch_optimize_btc.py
    python scripts/batch_optimize_btc.py --timeframe 15m  # 단일 타임프레임
    python scripts/batch_optimize_btc.py --dry-run  # 테스트 모드
"""
import sys
from pathlib import Path
from datetime import datetime
import json
import argparse

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# 타임프레임별 최적화 설정
TIMEFRAME_CONFIGS = {
    "5m": {
        "train_weeks": 4,      # 짧은 훈련 기간 (빠른 시장 변화)
        "test_weeks": 1,
        "n_iterations": 50,    # 더 많은 iteration
        "description": "스캘핑 전용 (5분봉)"
    },
    "15m": {
        "train_weeks": 8,
        "test_weeks": 2,
        "n_iterations": 40,
        "description": "단기 트레이딩 (15분봉)"
    },
    "1h": {
        "train_weeks": 12,     # 더 긴 훈련 기간
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
        "train_weeks": 24,     # 약 6개월 훈련
        "test_weeks": 6,
        "n_iterations": 25,
        "description": "포지션 트레이딩 (일봉)"
    },
    "1w": {
        "train_weeks": 52,     # 1년 훈련
        "test_weeks": 12,
        "n_iterations": 20,
        "description": "장기 투자 (주봉)"
    }
}


def get_total_days() -> int:
    """2019년 1월 1일부터 현재까지 일수 계산"""
    start = datetime(2019, 1, 1)
    now = datetime.now()
    return (now - start).days + 30  # 여유분 추가


def run_optimization(
    symbol: str,
    timeframe: str,
    market: str,
    days: int,
    config: dict,
    dry_run: bool = False
) -> dict:
    """단일 타임프레임 최적화 실행"""

    print(f"\n{'='*70}")
    print(f"[{timeframe}] {config['description']}")
    print(f"{'='*70}")
    print(f"  Symbol: {symbol}")
    print(f"  Market: {market}")
    print(f"  Days: {days} (~{days//365}년)")
    print(f"  Train: {config['train_weeks']}주, Test: {config['test_weeks']}주")
    print(f"  Iterations: {config['n_iterations']}")

    if dry_run:
        print(f"  [DRY RUN] 실제 최적화 건너뜀")
        return {"status": "dry_run", "timeframe": timeframe}

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
            output_dir=str(PROJECT_ROOT / "results" / "tuning" / "batch_btc")
        )

        if result:
            return {
                "status": "success",
                "timeframe": timeframe,
                "run_id": result.get("run_id"),
                "robust_params": result.get("robust_params"),
                "optimization": {
                    "avg_train_score": result.get("optimization", {}).get("avg_train_score"),
                    "avg_test_score": result.get("optimization", {}).get("avg_test_score"),
                    "overfit_ratio": result.get("optimization", {}).get("overfit_ratio"),
                }
            }
        else:
            return {"status": "failed", "timeframe": timeframe, "error": "No result returned"}

    except Exception as e:
        import traceback
        print(f"  [ERROR] {e}")
        traceback.print_exc()
        return {"status": "error", "timeframe": timeframe, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="BTC-USDT 전체 타임프레임 배치 최적화")
    parser.add_argument("--symbol", default="BTC-USDT", help="심볼")
    parser.add_argument("--market", default="spot", choices=["spot", "futures"])
    parser.add_argument("--timeframe", default=None, help="단일 타임프레임만 실행 (예: 15m)")
    parser.add_argument("--dry-run", action="store_true", help="테스트 모드 (실제 최적화 안함)")
    parser.add_argument("--days", type=int, default=None, help="데이터 일수 (기본: 2019년부터)")

    args = parser.parse_args()

    # 전체 기간 계산
    days = args.days or get_total_days()

    print("="*70)
    print("BTC-USDT Spot 전체 타임프레임 배치 최적화")
    print("="*70)
    print(f"Symbol: {args.symbol}")
    print(f"Market: {args.market}")
    print(f"Data Period: 2019-01-01 ~ Now ({days}일)")
    print(f"Timeframes: {list(TIMEFRAME_CONFIGS.keys()) if not args.timeframe else [args.timeframe]}")
    print(f"Dry Run: {args.dry_run}")
    print("="*70)

    # 실행할 타임프레임 결정
    if args.timeframe:
        if args.timeframe not in TIMEFRAME_CONFIGS:
            print(f"[Error] Unknown timeframe: {args.timeframe}")
            print(f"Available: {list(TIMEFRAME_CONFIGS.keys())}")
            return
        timeframes = [args.timeframe]
    else:
        # 전체 타임프레임 (짧은 것부터)
        timeframes = ["5m", "15m", "1h", "4h", "1d", "1w"]

    # 결과 저장
    results = {}
    start_time = datetime.now()

    for i, tf in enumerate(timeframes, 1):
        print(f"\n[{i}/{len(timeframes)}] Starting {tf} optimization...")

        config = TIMEFRAME_CONFIGS[tf]
        result = run_optimization(
            symbol=args.symbol,
            timeframe=tf,
            market=args.market,
            days=days,
            config=config,
            dry_run=args.dry_run
        )
        results[tf] = result

        if result["status"] == "success":
            print(f"[{tf}] ✓ Success - run_id: {result.get('run_id')}")
        elif result["status"] == "dry_run":
            print(f"[{tf}] ○ Dry run completed")
        else:
            print(f"[{tf}] ✗ Failed: {result.get('error')}")

    # 최종 요약
    elapsed = datetime.now() - start_time

    print("\n" + "="*70)
    print("배치 최적화 완료")
    print("="*70)
    print(f"Total Time: {elapsed}")
    print(f"Results:")

    for tf, result in results.items():
        status_icon = "✓" if result["status"] == "success" else "○" if result["status"] == "dry_run" else "✗"
        print(f"  [{tf}] {status_icon} {result['status']}")
        if result.get("robust_params"):
            params = result["robust_params"]
            print(f"       box_L={params.get('box_L')}, m_freeze={params.get('m_freeze')}, x_atr={params.get('x_atr')}")

    # 결과 JSON 저장
    output_path = PROJECT_ROOT / "results" / "tuning" / "batch_btc" / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "symbol": args.symbol,
        "market": args.market,
        "days": days,
        "elapsed_seconds": elapsed.total_seconds(),
        "results": results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str, ensure_ascii=False)

    print(f"\nSummary saved: {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()
