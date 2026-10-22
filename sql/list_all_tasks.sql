SELECT id, title, content, due_date, completed, created_at, updated_at
FROM tasks
ORDER BY created_at DESC
LIMIT %s;
