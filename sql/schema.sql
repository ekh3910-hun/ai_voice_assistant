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
    error_message TEXT
);

-- 기존 Neon DB에 옛날 스키마로 생성되어 있을 경우를 대비한 컬럼 추가 안전장치
ALTER TABLE tasks ADD COLUMN IF NOT EXISTS content TEXT;
ALTER TABLE tasks ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE voice_runs ADD COLUMN IF NOT EXISTS error_message TEXT;
