import os
import logging

# 로깅 설정
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def find_wyckoff_spot_strategy(root_dir: str) -> str or None:
    """
    주어진 디렉토리에서 wyckoff_spot_strategy.py 파일을 찾습니다.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == "wyckoff_spot_strategy.py":
                logging.debug(f"파일 찾음: {os.path.join(dirpath, filename)}")
                return os.path.join(dirpath, filename)
    logging.warning("wyckoff_spot_strategy.py 파일을 찾을 수 없습니다.")
    return None


def read_file(file_path: str) -> str:
    """
    주어진 경로의 파일을 읽고 내용을 문자열로 반환합니다.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logging.error(f"파일을 찾을 수 없습니다: {file_path}")
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
    except Exception as e:
        logging.error(f"파일을 읽는 중 에러 발생: {file_path} - {e}")
        raise RuntimeError(f"파일을 읽는 중 에러 발생: {file_path} - {e}")


def run_debug_backtest(strategy_file: str) -> None:
    """
    주어진 전략 파일을 사용하여 디버그 모드로 백테스팅을 실행합니다.
    """
    try:
        # 백테스팅 실행 명령 (디버그 모드)
        backtest_command = f"python scripts/run_backtest.py --strategy {strategy_file} --debug"
        logging.info(f"백테스팅 실행: {backtest_command}")
        os.system(backtest_command + " > backtest_debug.log 2>&1") # 로그 리디렉션
    except Exception as e:
        logging.error(f"백테스팅 실행 중 에러 발생: {e}")
        raise RuntimeError(f"백테스팅 실행 중 에러 발생: {e}")


def analyze_backtest_results(log_file: str) -> None:
    """
    백테스팅 결과 로그를 분석합니다.
    """
    try:
        log_content = read_file(log_file)
        logging.info(f"로그 파일 분석: {log_file}")
        print(log_content)  # 로그 내용 출력 (실제 분석 로직 필요)
    except Exception as e:
        logging.error(f"로그 파일 분석 중 에러 발생: {e}")
        raise RuntimeError(f"로그 파일 분석 중 에러 발생: {e}")


# 1. 파일 찾기
root_dir = "C:/Users/hahonggu/Desktop/coin_master/projects/wpcn-backtester-cli-noflask" # 프로젝트 루트 디렉토리로 변경
strategy_file = find_wyckoff_spot_strategy(root_dir)

if not strategy_file:
    print("에러: wyckoff_spot_strategy.py 파일을 찾을 수 없습니다.")
    exit()

print(f"wyckoff_spot_strategy.py 파일을 찾았습니다: {strategy_file}")

# 2. 파일 읽기
try:
    strategy_code = read_file(strategy_file)
    #print(f"파일 내용:\n{strategy_code}") # 파일 내용이 너무 길 수 있으므로 주석 처리
except Exception as e:
    print(f"에러: 파일 읽기 실패 - {e}")
    exit()

# 3. 디버그 모드 백테스팅 실행
try:
    # 파일 존재 여부 재확인
    if not os.path.exists(strategy_file):
        print(f"에러: 전략 파일이 존재하지 않습니다: {strategy_file}")
        exit()

    # 백테스팅 실행
    run_debug_backtest(strategy_file)

except Exception as e:
    print(f"에러: 디버그 모드 백테스팅 실패 - {e}")
    exit()

# 4. 결과 분석
try:
    analyze_backtest_results("backtest_debug.log")
except Exception as e:
    print(f"에러: 결과 분석 실패 - {e}")
    exit()