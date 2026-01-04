"""
현물 백테스트 실행 스크립트
Navigation Gate 완화 설정으로 BTC-USDT 15m 백테스트 실행
"""
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import json
import pandas as pd

def run_spot_backtest():
    """현물 백테스트 실행"""
    print("=" * 60)
    print("WPCN 현물 백테스트 실행")
    print("=" * 60)
    
    # 설정
    config = {
        "symbol": "BTC-USDT",
        "timeframe": "15m",
        "initial_capital": 10000,
        "spot_mode": True,
        # Navigation Gate 완화 설정
        "conf_min": 0.3,
        "min_score": 3.5,
        "min_rr_ratio": 1.2,
        "min_context_score": 1.5,
        "min_trigger_score": 1.0,
    }
    
    print(f"\n[설정]")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 데이터 로드 시도
    print(f"\n[데이터 로드]")
    
    # bronze 폴더에서 데이터 찾기
    data_dir = project_root / "data" / "bronze"
    
    # 가능한 파일 패턴들
    possible_patterns = [
        f"{config['symbol'].replace('-', '_')}_{config['timeframe']}.parquet",
        f"{config['symbol']}_{config['timeframe']}.parquet",
        f"BTC_USDT_{config['timeframe']}.parquet",
        f"BTCUSDT_{config['timeframe']}.parquet",
    ]
    
    data_file = None
    for pattern in possible_patterns:
        candidate = data_dir / pattern
        if candidate.exists():
            data_file = candidate
            break
    
    # 파일 목록 출력
    print(f"  데이터 디렉토리: {data_dir}")
    if data_dir.exists():
        files = list(data_dir.glob("*.parquet"))
        print(f"  발견된 parquet 파일들:")
        for f in files[:10]:  # 최대 10개만 표시
            print(f"    - {f.name}")
        if len(files) > 10:
            print(f"    ... 외 {len(files) - 10}개")
    
    if data_file:
        print(f"  사용할 데이터 파일: {data_file}")
        
        # 데이터 로드
        try:
            df = pd.read_parquet(data_file)
            print(f"  데이터 로드 완료: {len(df)} rows")
            print(f"  기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
            
            # 최근 3개월 필터링
            end_date = df['timestamp'].max()
            start_date = end_date - timedelta(days=90)
            df_filtered = df[df['timestamp'] >= start_date].copy()
            print(f"  최근 3개월 필터링: {len(df_filtered)} rows")
            print(f"  필터링 기간: {df_filtered['timestamp'].min()} ~ {df_filtered['timestamp'].max()}")
            
        except Exception as e:
            print(f"  데이터 로드 실패: {e}")
            return None
    else:
        print(f"  데이터 파일을 찾을 수 없습니다.")
        print(f"  시도한 패턴: {possible_patterns}")
        return None
    
    # 백테스트 엔진 임포트 및 실행
    print(f"\n[백테스트 엔진 로드]")
    try:
        from wpcn._06_engine.backtest import BacktestEngine
        print("  BacktestEngine 임포트 성공")
    except ImportError as e:
        print(f"  BacktestEngine 임포트 실패: {e}")
        # 대안: 직접 백테스트 로직 실행
        return run_simple_backtest(df_filtered, config)
    
    # 백테스트 실행
    print(f"\n[백테스트 실행]")
    try:
        engine = BacktestEngine(
            initial_capital=config["initial_capital"],
            spot_mode=config["spot_mode"],
            conf_min=config["conf_min"],
            min_score=config["min_score"],
            min_rr_ratio=config["min_rr_ratio"],
            min_context_score=config["min_context_score"],
            min_trigger_score=config["min_trigger_score"],
        )
        
        results = engine.run(df_filtered)
        
        # 결과 출력
        print_results(results)
        
        return results
        
    except Exception as e:
        print(f"  백테스트 실행 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_simple_backtest(df: pd.DataFrame, config: dict) -> dict:
    """간단한 백테스트 실행 (BacktestEngine 사용 불가 시)"""
    print("\n[간단한 백테스트 모드]")
    
    # 기본 통계
    results = {
        "data_points": len(df),
        "start_date": str(df['timestamp'].min()),
        "end_date": str(df['timestamp'].max()),
        "initial_capital": config["initial_capital"],
        "config": config,
    }
    
    # 가격 변동 분석
    if 'close' in df.columns:
        df['returns'] = df['close'].pct_change()
        
        results["price_stats"] = {
            "start_price": float(df['close'].iloc[0]),
            "end_price": float(df['close'].iloc[-1]),
            "price_change_pct": float((df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100),
            "max_price": float(df['close'].max()),
            "min_price": float(df['close'].min()),
            "volatility": float(df['returns'].std() * 100),
        }
    
    return results


def print_results(results: dict):
    """결과 출력"""
    print("\n" + "=" * 60)
    print("백테스트 결과")
    print("=" * 60)
    
    if isinstance(results, dict):
        for key, value in results.items():
            if isinstance(value, dict):
                print(f"\n[{key}]")
                for k, v in value.items():
                    if isinstance(v, float):
                        print(f"  {k}: {v:.4f}")
                    else:
                        print(f"  {k}: {v}")
            elif isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
    else:
        print(results)


if __name__ == "__main__":
    results = run_spot_backtest()
    
    if results:
        # 결과 저장
        output_file = Path(__file__).parent.parent / "results" / "spot_backtest_result.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n결과 저장: {output_file}")