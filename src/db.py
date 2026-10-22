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
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(load_sql(filename), params)
            row = cur.fetchone()
            return row[0] if row else None

def _fetch_all(filename, params=None):
    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(load_sql(filename), params)
            return cur.fetchall()

# 1. 초기 테이블 생성
def init_db():
    sql = load_sql('schema.sql')
    with get_connection() as conn:
        with conn.cursor() as cur:
            # 복수 DDL 문장 순차 실행
            for statement in sql.split(';'):
                if statement.strip():
                    cur.execute(statement)

# 2. 메모(memos)
def insert_memo(content):
    return _fetch_one('add_memo.sql', (content,))

def get_recent_memos(limit=10):
    return _fetch_all('list_memo.sql', (limit,))

def delete_memo(memo_id):
    return _fetch_one('delete_memo.sql', (memo_id,))

# 3. 할 일(tasks)
def insert_task(title, due_date=None, content=None):
    return _fetch_one('add_task.sql', (title, due_date, content))

def get_tasks_by_date(target_date):
    """특정 날짜 할 일 조회 (음성 비서 명령용)"""
    return _fetch_all('list_task.sql', (target_date,))

def get_all_tasks(limit=100):
    """전체 할 일 목록 조회 (화면 UI용)"""
    return _fetch_all('list_all_tasks.sql', (limit,))

def update_task(task_id, title=None, content=None, due_date=None, completed=None):
    return _fetch_one('update_task.sql', (title, content, due_date, completed, task_id))

def delete_task(task_id):
    return _fetch_one('delete_task.sql', (task_id,))

# 4. 음성 처리 이력(voice_runs)
def insert_voice_run(transcript, intent, status, response_text=None, latency_ms=None, error_message=None):
    params = (transcript, intent, status, response_text, latency_ms, error_message)
    return _fetch_one('insert_voice_run.sql', params)

def list_voice_runs(limit=100):
    """이력 화면용 최근 음성 처리 로그 조회"""
    return _fetch_all('list_voice_runs.sql', (limit,))

# 5. 팀원 코드 호환용 별칭 (Alias)
save_memo = insert_memo
add_memo = insert_memo
list_memos = get_recent_memos

save_task = insert_task
add_task = insert_task
list_tasks = get_all_tasks

save_voice_run = insert_voice_run
create_table = init_db
