"""
MSSQL 데이터베이스 연결 모듈
"""

import os
from typing import Optional, List, Any
from contextlib import contextmanager
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


def get_connection():
    """MSSQL 연결 객체 반환"""
    import pyodbc

    server = os.getenv("MSSQL_SERVER")
    database = os.getenv("MSSQL_DATABASE")
    user = os.getenv("MSSQL_USER")
    password = os.getenv("MSSQL_PASSWORD")

    conn_str = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={user};"
        f"PWD={password};"
        f"TrustServerCertificate=yes;"
    )

    return pyodbc.connect(conn_str)


@contextmanager
def get_cursor():
    """커서를 자동으로 닫는 컨텍스트 매니저"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        yield cursor
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()


def execute_query(query: str, params: Optional[tuple] = None) -> List[Any]:
    """SELECT 쿼리 실행 후 결과 반환"""
    with get_cursor() as cursor:
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)

        columns = [column[0] for column in cursor.description] if cursor.description else []
        rows = cursor.fetchall()

        # dict 형태로 반환
        return [dict(zip(columns, row)) for row in rows]


def execute_many(query: str, params_list: List[tuple]) -> int:
    """INSERT/UPDATE 여러 행 실행"""
    with get_cursor() as cursor:
        cursor.executemany(query, params_list)
        return cursor.rowcount


def execute_non_query(query: str, params: Optional[tuple] = None) -> int:
    """INSERT/UPDATE/DELETE 실행 후 영향받은 행 수 반환"""
    with get_cursor() as cursor:
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        return cursor.rowcount


def test_connection() -> bool:
    """연결 테스트"""
    try:
        with get_cursor() as cursor:
            cursor.execute("SELECT 1")
            return True
    except Exception as e:
        print(f"DB 연결 실패: {e}")
        return False
