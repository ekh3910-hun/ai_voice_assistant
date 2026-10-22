INSERT INTO tasks (title, content, due_date)
VALUES (%s, %s, %s)
RETURNING id;
