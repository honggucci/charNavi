@echo off
REM BTC-USDT 전체 타임프레임 병렬 최적화 (Windows batch)
REM 6개 타임프레임을 각각 별도 프로세스로 실행

echo ============================================================
echo BTC-USDT 전체 타임프레임 병렬 최적화 시작
echo ============================================================

cd /d %~dp0..

REM 각 타임프레임을 백그라운드로 실행
start "5m" cmd /c "python -m wpcn._08_tuning.run_tuning --symbol BTC-USDT --timeframe 5m --days 2200 --optimizer optuna --train-weeks 4 --test-weeks 1 --iterations 50 > results\tuning\log_5m.txt 2>&1"
start "15m" cmd /c "python -m wpcn._08_tuning.run_tuning --symbol BTC-USDT --timeframe 15m --days 2200 --optimizer optuna --train-weeks 8 --test-weeks 2 --iterations 40 > results\tuning\log_15m.txt 2>&1"
start "1h" cmd /c "python -m wpcn._08_tuning.run_tuning --symbol BTC-USDT --timeframe 1h --days 2200 --optimizer optuna --train-weeks 12 --test-weeks 3 --iterations 35 > results\tuning\log_1h.txt 2>&1"
start "4h" cmd /c "python -m wpcn._08_tuning.run_tuning --symbol BTC-USDT --timeframe 4h --days 2200 --optimizer optuna --train-weeks 16 --test-weeks 4 --iterations 30 > results\tuning\log_4h.txt 2>&1"
start "1d" cmd /c "python -m wpcn._08_tuning.run_tuning --symbol BTC-USDT --timeframe 1d --days 2200 --optimizer optuna --train-weeks 24 --test-weeks 6 --iterations 25 > results\tuning\log_1d.txt 2>&1"
start "1w" cmd /c "python -m wpcn._08_tuning.run_tuning --symbol BTC-USDT --timeframe 1w --days 2200 --optimizer optuna --train-weeks 52 --test-weeks 12 --iterations 20 > results\tuning\log_1w.txt 2>&1"

echo.
echo 6개 타임프레임 최적화 프로세스 시작됨
echo 진행 상황: results\tuning\log_*.txt 확인
echo.
