INSERT INTO tasks (title, due_date, content) VALUES (%s, %s, %s) RETURNING id;
