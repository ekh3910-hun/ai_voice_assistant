# src/db.py
import os
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# SQL 파일 디렉토리 (project/sql 폴더)
SQL_DIR = Path(__file__).resolve().parent.parent / 'sql'

def load_sql(filename):
    path = (SQL_DIR / filename).resolve()
    return path.read_text(encoding='utf-8')

def get_connection():
    if not DATABASE_URL:
        raise ValueError('DATABASE_URL이 설정되지 않았습니다!')
    return psycopg.connect(DATABASE_URL)

# 공통 실행 헬퍼
def _fetch_one(filename, params=None):
    """INSERT/UPDATE/DELETE ... RETURNING id 용. 대상이 없으면 None"""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(load_sql(filename), params)
            row = cur.fetchone()
            return row[0] if row else None

def _fetch_all(filename, params=None):
    """SELECT 용. dict 리스트 반환"""
    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(load_sql(filename), params)
            return cur.fetchall()

# 1. 초기 테이블 생성
def init_db():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(load_sql('schema.sql'))

# 2. 메모(memos)
def insert_memo(content):
    return _fetch_one('add_memo.sql', (content,))

def get_recent_memos(limit=10):
    return _fetch_all('list_memo.sql', (limit,))

def delete_memo(memo_id):
    return _fetch_one('delete_memo.sql', (memo_id,))

# 3. 할 일(tasks)
def insert_task(title, due_date, content):
    return _fetch_one('add_task.sql', (title, due_date, content))

def get_tasks_by_date(target_date):
    return _fetch_all('list_task.sql', (target_date,))

def update_task(task_id, title=None, content=None, due_date=None, completed=None):
    """바꿀 값만 넘기면 됨. 예) update_task(id, completed=True)"""
    return _fetch_one('update_task.sql', (title, content, due_date, completed, task_id))

def delete_task(task_id):
    return _fetch_one('delete_task.sql', (task_id,))

# 4. 음성 처리 이력(voice_runs)
def insert_voice_run(transcript, intent, status, response_text, latency_ms, error_message=None):
    params = (transcript, intent, status, response_text, latency_ms, error_message)
    return _fetch_one('insert_voice_run.sql', params)
