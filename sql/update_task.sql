UPDATE tasks
SET due_date   = COALESCE(%s, due_date),
    content    = COALESCE(%s, content),
    completed  = COALESCE(%s, completed),
    updated_at = NOW()
WHERE id = %s
RETURNING id;
