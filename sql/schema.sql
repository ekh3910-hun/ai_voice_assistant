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
