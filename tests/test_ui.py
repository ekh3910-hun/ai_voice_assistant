"""외부 DB 없이 실제 Streamlit 페이지 렌더링을 검증합니다."""
from contextlib import ExitStack
from pathlib import Path
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from src import db
from pages import memo, todo, history


class PageTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for module in (db, memo, todo, history):
            self.stack.enter_context(patch.object(module, 'has_database_url', return_value=False))

    def test_home_page(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=30)
        self.assertEqual(len(app.exception), 0, [e.message for e in app.exception])

    def test_legacy_deployment_entry(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app1_b.py')).run(timeout=30)
        self.assertEqual(len(app.exception), 0, [e.message for e in app.exception])

    def test_memo_page(self):
        app = AppTest.from_string('from pages.memo import render_memo_view\nrender_memo_view()').run(timeout=30)
        self.assertEqual(len(app.exception), 0, [e.message for e in app.exception])

    def test_task_page(self):
        app = AppTest.from_string('from pages.todo import render_todo_view\nrender_todo_view()').run(timeout=30)
        self.assertEqual(len(app.exception), 0, [e.message for e in app.exception])

    def test_history_page(self):
        app = AppTest.from_string('from pages.history import render_history_view\nrender_history_view()').run(timeout=30)
        self.assertEqual(len(app.exception), 0, [e.message for e in app.exception])
        self.assertTrue(app.button[0].disabled)

    def test_model_evaluation_page(self):
        app = AppTest.from_string('from pages.model_evaluate import render_evaluate_view\nrender_evaluate_view()').run(timeout=30)
        self.assertEqual(len(app.exception), 0, [e.message for e in app.exception])


if __name__ == '__main__':
    unittest.main()
