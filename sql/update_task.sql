UPDATE tasks
SET title      = COALESCE(%s, title),
    content    = COALESCE(%s, content),
    due_date   = COALESCE(%s, due_date),
    completed  = COALESCE(%s, completed),
    updated_at = NOW()
WHERE id = %s
RETURNING id;
