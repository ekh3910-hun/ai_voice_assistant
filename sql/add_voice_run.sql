-- [수정] input_type 컬럼 추가 (값 7개)
INSERT INTO voice_runs (transcript, intent, status, response_text, latency_ms, error_message, input_type)
VALUES (%s, %s, %s, %s, %s, %s, %s)
RETURNING id;
