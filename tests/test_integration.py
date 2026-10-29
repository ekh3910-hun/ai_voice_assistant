"""두 저장소 통합 기능 회귀 테스트. 외부 DB 및 음성 API를 호출하지 않습니다."""
import io
from datetime import date, datetime
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

from src import audio, core, db
from pages import history, home, memo, todo


class State(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class IntegrationTests(unittest.TestCase):
    def test_day_only_and_korean_dates(self):
        self.assertEqual(core.convert_date('30일', date(2026, 10, 29)), '2026-10-30')
        self.assertEqual(core.convert_date('28일', date(2026, 10, 29)), '2026-11-28')
        self.assertEqual(core.normalize_text('이십구일 운동 추가해'), '29일 운동 추가해')
        result = core.predict_intent('이십구일 운동 추가해')
        self.assertIn('DUE_DATE', result['slots'])
        self.assertEqual(result['slots']['TITLE'], '운동')

    def test_rename_preserves_target_date(self):
        slots = core.extract_slots('30일 장보기 이름을 마트 가기로 바꿔', 'UPDATE_TASK')
        self.assertEqual(slots['DUE_DATE'], core.convert_date('30일'))
        self.assertNotIn('NEW_DUE_DATE', slots)
        self.assertEqual(slots['TITLE'], '장보기')
        self.assertEqual(slots['NEW_TITLE'], '마트 가기')

    def test_move_has_old_and_new_dates(self):
        slots = core.extract_slots('30일 운동 날짜를 31일로 바꿔', 'UPDATE_TASK')
        self.assertEqual(slots['DUE_DATE'], core.convert_date('30일'))
        self.assertEqual(slots['NEW_DUE_DATE'], core.convert_date('31일'))
        self.assertEqual(slots['TITLE'], '운동')

    def test_memo_body_is_data(self):
        scenarios = [('내일 회의 삭제 확인이라고 메모해', '내일 회의 삭제 확인'),
                     ('메모 추가: 내일 자료 추가 확인', '내일 자료 추가 확인'),
                     ('1일 1커밋 하기 메모 해줘', '1일 1커밋 하기'),
                     ('메모 등록 내일 회의 준비', '내일 회의 준비')]
        for text, expected in scenarios:
            with self.subTest(text=text):
                result = core.predict_intent(text)
                self.assertEqual(result['intent'], 'ADD_MEMO')
                self.assertEqual(result['slots']['MEMO_CONTENT'], expected)
                self.assertFalse(result['needs_confirmation'])

    def test_complete_all_only_pending(self):
        rows = [{'id': 'a', 'completed': False}, {'id': 'b', 'completed': False},
                {'id': 'c', 'completed': True}]
        with patch.object(core, 'task_matches', return_value=rows), \
             patch.object(core, 'complete_tasks', return_value=2) as complete:
            response, data = core.execute('COMPLETE_TASK', {'TITLE': '운동', 'ALL': True})
        complete.assert_called_once_with(['a', 'b'])
        self.assertEqual(data['count'], 2)

    def test_ambiguous_delete_does_not_mutate(self):
        rows = [{'id': 'a', 'due_date': date(2026, 10, 30)},
                {'id': 'b', 'due_date': date(2026, 10, 31)}]
        with patch.object(core, 'task_matches', return_value=rows), \
             patch.object(core, 'delete_task') as single, patch.object(core, 'delete_tasks') as bulk:
            response, data = core.execute('DELETE_TASK', {'TITLE': '운동'})
        self.assertIsNone(data)
        self.assertIn('2개', response)
        single.assert_not_called()
        bulk.assert_not_called()

    def test_delete_all_tasks(self):
        with patch.object(core, 'task_matches', return_value=[{'id': 'a'}, {'id': 'b'}]), \
             patch.object(core, 'delete_tasks', return_value=2) as delete:
            _, data = core.execute('DELETE_TASK', {'TITLE': '운동', 'ALL': True})
        delete.assert_called_once_with(['a', 'b'])
        self.assertEqual(data['count'], 2)

    def test_latest_memo_deletes_one(self):
        slots = core.extract_slots('최근 메모 삭제해', 'DELETE_MEMO')
        with patch.object(core, 'memo_matches', return_value=[{'id': 'new', 'content': '최신'}, {'id': 'old', 'content': '이전'}]), \
             patch.object(core, 'delete_memo', return_value=1) as delete:
            _, data = core.execute('DELETE_MEMO', slots)
        delete.assert_called_once_with('new')
        self.assertEqual(data['id'], 'new')

    def test_all_matching_memos(self):
        slots = core.extract_slots('회의 관련 메모 모두 삭제해', 'DELETE_MEMO')
        with patch.object(core, 'memo_matches', return_value=[{'id': 'a'}, {'id': 'b'}]), \
             patch.object(core, 'delete_memos', return_value=2) as delete:
            _, data = core.execute('DELETE_MEMO', slots)
        delete.assert_called_once_with(['a', 'b'])
        self.assertEqual(data['count'], 2)

    def test_batch_failure_rolls_back_transaction(self):
        conn = MagicMock()
        cur = conn.__enter__.return_value.cursor.return_value.__enter__.return_value
        cur.rowcount = 1
        cur.execute.side_effect = [None, RuntimeError('offline')]
        with patch.object(db, 'get_connection', return_value=conn):
            with self.assertRaises(RuntimeError):
                db.delete_tasks(['a', 'b'])
        self.assertEqual(conn.__exit__.call_args.args[0], RuntimeError)

    def test_batch_missing_row_is_not_success(self):
        conn = MagicMock()
        cur = conn.__enter__.return_value.cursor.return_value.__enter__.return_value
        cur.rowcount = 0
        with patch.object(db, 'get_connection', return_value=conn):
            with self.assertRaises(ValueError):
                db.complete_tasks(['missing'])
        self.assertEqual(conn.__exit__.call_args.args[0], ValueError)

    def test_mp3_preprocessing(self):
        samples = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000) * 0.2
        buffer = io.BytesIO()
        sf.write(buffer, samples, 16000, format='MP3')
        output, info = audio.preprocess(buffer.getvalue(), check_duplicate=False)
        self.assertEqual(sf.info(io.BytesIO(output)).format, 'WAV')

    def test_empty_states_have_no_demo_records(self):
        state = State()
        for module, method in [(memo, memo.init_memo_state), (todo, todo.init_task_state)]:
            with patch.object(module, 'st') as ui, patch.object(module, 'has_database_url', return_value=False):
                ui.session_state = state
                method()
        with patch.object(history, 'st') as ui:
            ui.session_state = state
            history.init_history_state()
        self.assertEqual(state, dict(memos=[], tasks=[], history=[]))

    def test_history_merges_unsaved_and_deduplicates_saved(self):
        state = State(history=[{'id': '1', 'transcript': '重複'}, {'id': None, 'transcript': '未保存'}])
        with patch.object(history, 'st') as ui, patch.object(history, 'has_database_url', return_value=True), \
             patch.object(history, 'list_voice_runs', return_value=[{'id': '1', 'transcript': 'DB'}]):
            ui.session_state = state
            records = history.fetch_history_records()
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]['transcript'], 'DB')

    def test_history_failure_preserves_session(self):
        state = State(history=[{'transcript': '세션 이력'}])
        with patch.object(history, 'st') as ui, patch.object(history, 'has_database_url', return_value=True), \
             patch.object(history, 'list_voice_runs', side_effect=RuntimeError('offline')):
            ui.session_state = state
            self.assertEqual(history.fetch_history_records(), state.history)
            ui.error.assert_called_once()

    def test_success_syncs_from_db(self):
        result = dict(intent='DELETE_TASK', slots={'TITLE': '운동'}, status='성공', response='삭제', data={'id': '1'})
        state = State(history=[])
        with patch.object(home, 'st') as ui, patch.object(home, 'core_process_text', return_value=result), \
             patch.object(home, 'init_task_state') as sync:
            ui.session_state = state
            home.handle_process_text('운동 삭제해')
        sync.assert_called_once_with(force=True)

    def test_history_write_failure_returns_warning(self):
        prediction = dict(intent='HELP', slots={}, needs_confirmation=False)
        with patch.object(core, 'predict_intent', return_value=prediction), \
             patch.object(core, 'has_database_url', return_value=True), \
             patch.object(core, 'add_voice_runs', side_effect=RuntimeError('offline')):
            result = core.process_text('도움말')
        self.assertEqual(result['status'], '성공')
        self.assertIsNotNone(result['history_warning'])


if __name__ == '__main__':
    unittest.main()
