CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. memos 테이블
CREATE TABLE IF NOT EXISTS memos (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 2. tasks 테이블
CREATE TABLE IF NOT EXISTS tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title TEXT NOT NULL,
    due_date DATE,
    completed BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    content TEXT NOT NULL,
    updated_at TIMESTAMPTZ
);

-- 3. voice_runs 테이블
CREATE TABLE IF NOT EXISTS voice_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    transcript TEXT NOT NULL,
    intent TEXT NOT NULL,
    status TEXT NOT NULL,
    response_text TEXT,
    latency_ms DOUBLE PRECISION NOT NULL,
    error_message TEXT
);
