SELECT id, transcript, intent, response_text, status, latency_ms, error_message, created_at
FROM voice_runs
ORDER BY created_at DESC
LIMIT %s;
