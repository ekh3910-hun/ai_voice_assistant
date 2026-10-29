-- [수정] input_type 컬럼 조회 추가
SELECT id, transcript, intent, response_text, status, latency_ms, error_message, input_type, created_at
FROM voice_runs
ORDER BY created_at DESC
LIMIT %s;
