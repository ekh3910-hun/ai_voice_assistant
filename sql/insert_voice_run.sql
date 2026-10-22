INSERT INTO voice_runs (transcript, intent, status, response_text, latency_ms, error_message)
VALUES (%s, %s, %s, %s, %s, %s)
RETURNING id;
