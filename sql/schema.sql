CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- 1. memos 테이블
CREATE TABLE IF NOT EXISTS memos (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 2. tasks 테이블
CREATE TABLE IF NOT EXISTS tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    due_date DATE,
    completed BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    content TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. voice_runs 테이블
CREATE TABLE IF NOT EXISTS voice_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    transcript TEXT,
    intent TEXT,
    status TEXT NOT NULL,
    response_text TEXT,
    latency_ms DOUBLE PRECISION,
    error_message TEXT,
    input_type TEXT              -- [수정] 입력 방식: 텍스트 / 직접 녹음 / 파일 업로드
);

-- [수정] 이미 만들어진 voice_runs 테이블에 컬럼 추가 (Neon 기존 DB용, 여러 번 실행해도 안전)
ALTER TABLE voice_runs ADD COLUMN IF NOT EXISTS input_type TEXT;