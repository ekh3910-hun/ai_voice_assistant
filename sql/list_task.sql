SELECT id, title, content, due_date, completed, created_at, updated_at
FROM tasks
WHERE due_date = %s
ORDER BY created_at DESC;
