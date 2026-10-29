UPDATE memos
SET content = %s
WHERE id = %s
RETURNING id;