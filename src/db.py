# src/db.py
# 프로젝트 DB

import os
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# sql 폴더
SQL_DIR = Path(__file__).resolve().parent.parent / 'sql'
if not SQL_DIR.exists():
    SQL_DIR.mkdir(parents=True, exist_ok=True)

# sql 불러오기
def load_sql(filename):
    path = (SQL_DIR / filename).resolve()
    if path.parent != SQL_DIR.resolve() or path.suffix != '.sql':
        raise ValueError('sql폴더의 .sql파일명만 지정하세요')
    return path.read_text(encoding='utf-8')

def has_database_url():
    return bool(DATABASE_URL)

def get_connection():
    if not has_database_url():
        raise ValueError('DATABASE_URL이 설정되지 않았습니다')
    return psycopg.connect(DATABASE_URL)

# 테이블 생성
def create_table():
    sql = load_sql('schema.sql')
    with get_connection() as conn:
        with conn.cursor() as cur:
            for statement in sql.split(';'):
                if statement.strip():
                    cur.execute(statement)

# 메모 입력
def add_memo(content):
    sql = load_sql('add_memo.sql')
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (content,))
            return cur.fetchone()[0]

# 할 일 생성
def add_task(title, content=None, due_date=None):
    sql = load_sql('add_task.sql')
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (title, content, due_date))
            return cur.fetchone()[0]

# 최근 메모 조회
def list_memo(limit=10):
    sql = load_sql('list_memo.sql')
    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, (limit,))
            return cur.fetchall()

# 날짜별 할 일 조회
def list_task(due_date):
    sql = load_sql('list_task.sql')
    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, (due_date,))
            return cur.fetchall()

# 전체 할 일 목록 조회 (화면 UI용)
def list_all_tasks(limit=100):
    sql = load_sql('list_all_tasks.sql')
    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, (limit,))
            return cur.fetchall()

# 할 일 수정 및 완료 처리
def update_task(data_id, due_date=None, content=None, completed=None):
    sql = load_sql('update_task.sql')
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (due_date, content, completed, data_id))
            return cur.rowcount

# 메모 삭제
def delete_memo(data_id):
    sql = load_sql('delete_memo.sql')
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (data_id,))
            return cur.rowcount

# 할 일 삭제
def delete_task(data_id):
    sql = load_sql('delete_task.sql')
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (data_id,))
            return cur.rowcount

# 음성 명령 처리 이력 입력
def add_voice_runs(transcript, intent, status, response_text=None, latency_ms=None, error_message=None):
    sql = load_sql('add_voice_run.sql')
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (transcript, intent, status, response_text, latency_ms, error_message))
            return cur.fetchone()[0]

# 음성 명령 처리 이력 조회
def list_voice_runs(limit=100):
    sql = load_sql('list_voice_runs.sql')
    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, (limit,))
            return cur.fetchall()
