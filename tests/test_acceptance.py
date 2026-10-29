"""실행: python -m unittest discover -s tests -p test_acceptance.py -v

외부 STT/TTS 및 DB는 대역을 사용하므로 네트워크와 실제 DB를 변경하지 않습니다.
"""
import asyncio
import io
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from src import audio, core, db, stt, tts
from pages import home, memo, todo


def wav(samples=None, sample_rate=16000):
    if samples is None:
        samples = 0.2 * np.sin(2 * np.pi * 440 * np.arange(sample_rate) / sample_rate)
    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format='WAV')
    return buffer.getvalue()


def prediction(intent='ADD_TASK', slots=None, confirm=False):
    return dict(intent=intent, slots=slots or {'TITLE': '운동'},
                needs_confirmation=confirm, model_name='test',
                confidence_or_score=0.9, score_type='probability')


class AcceptanceTests(unittest.TestCase):
    def test_tc03_saved_model_scenarios(self):
        scenarios = [('내일 할 일에 발표 준비 추가해 줘', 'ADD_TASK'),
                     ('메모 추가: 내일 회의 준비', 'ADD_MEMO'),
                     ('회의 준비라고 메모해', 'ADD_MEMO'),
                     ('완료한 할 일 보여줘', 'LIST_TASK'),
                     ('내일 미완료 할 일 보여줘', 'LIST_TASK'),
                     ('장보기 이름을 마트 가기로 바꿔', 'UPDATE_TASK'),
                     ('사용할 수 있는 명령 알려 줘', 'HELP')]
        for text, expected in scenarios:
            with self.subTest(text=text):
                result = core.predict_intent(text)
                self.assertEqual(result['intent'], expected)
                self.assertFalse(result['needs_confirmation'], result)

    def test_tc01_valid_wav(self):
        data, info = audio.preprocess(wav(), check_duplicate=False)
        self.assertEqual(sf.info(io.BytesIO(data)).samplerate, 16000)
        self.assertGreaterEqual(info['prepared_sec'], 0.5)

    def test_tc01_stereo_resampling(self):
        samples = np.sin(2 * np.pi * 440 * np.arange(44100) / 44100) * 0.2
        data, _ = audio.preprocess(wav(np.column_stack([samples, samples]), 44100), check_duplicate=False)
        info = sf.info(io.BytesIO(data))
        self.assertEqual((info.channels, info.samplerate), (1, 16000))

    def test_tc02_stt_contract(self):
        model = MagicMock()
        model.transcribe.return_value = ([SimpleNamespace(text=' 운동 추가해. ')], None)
        with patch.object(stt, 'load_stt_model', return_value=model):
            text, elapsed = stt.transcript_wav(wav())
        self.assertEqual(text, '운동 추가해')
        self.assertGreaterEqual(elapsed, 0)

    def test_tc02_empty_transcript(self):
        model = MagicMock()
        model.transcribe.return_value = ([], None)
        with patch.object(stt, 'load_stt_model', return_value=model):
            with self.assertRaises(ValueError):
                stt.transcript_wav(wav())

    def test_tc03_memo_preserves_date_and_content(self):
        for text, expected in [('내일 3시 치과 예약이라고 메모해', '내일 3시 치과 예약'),
                               ('회의 준비라고 메모해', '회의 준비'),
                               ('메모 추가: 내일 자료 추가 확인', '내일 자료 추가 확인')]:
            with self.subTest(text=text):
                self.assertEqual(core.extract_slots(text, 'ADD_MEMO')['MEMO_CONTENT'], expected)

    def test_tc03_completion_filter_is_list(self):
        model = MagicMock()
        model.predict.return_value = ['COMPLETE_TASK']
        model.predict_proba.return_value = np.array([[0.9]])
        with patch.object(core, 'load_intent_model', return_value=(model, 'LogisticRegression')):
            result = core.predict_intent('미완료 할 일 보여줘')
        self.assertEqual(result['intent'], 'LIST_TASK')

    def test_tc03_help_is_not_task_list(self):
        model = MagicMock()
        model.predict.return_value = ['HELP']
        model.predict_proba.return_value = np.array([[0.9]])
        with patch.object(core, 'load_intent_model', return_value=(model, 'LogisticRegression')):
            self.assertEqual(core.predict_intent('사용할 수 있는 명령 알려 줘')['intent'], 'HELP')

    def test_tc03_confirmation_blocks_execution(self):
        with patch.object(core, 'predict_intent', return_value=prediction(confirm=True)), \
             patch.object(core, 'execute') as execute, patch.object(core, 'has_database_url', return_value=False):
            result = core.process_text('운동 추가해')
        execute.assert_not_called()
        self.assertEqual(result['status'], '확인 필요')

    def test_tc03_invalid_date_blocks_execution(self):
        with patch.object(core, 'predict_intent', return_value=prediction()), \
             patch.object(core, 'execute') as execute, patch.object(core, 'has_database_url', return_value=False):
            result = core.process_text('2월 30일 운동 추가해')
        execute.assert_not_called()
        self.assertEqual(result['status'], '확인 필요')

    def test_tc03_missing_match_not_success(self):
        with patch.object(core, 'predict_intent', return_value=prediction('DELETE_TASK')), \
             patch.object(core, 'init_db'), patch.object(core, 'list_all_tasks', return_value=[]), \
             patch.object(core, 'has_database_url', return_value=False):
            self.assertEqual(core.process_text('운동 삭제해')['status'], '확인 필요')

    def test_tc03_status_spacing(self):
        self.assertEqual(core.extract_slots('안끝낸 할 일 보여줘', 'LIST_TASK')['STATUS'], '안 끝낸')

    def test_tc08_memo_update_failure_preserves_content(self):
        state = SimpleNamespace(memos=[{'id': '1', 'content': '원본'}])
        with patch.object(memo, 'st') as ui, patch.object(memo, 'has_database_url', return_value=True), \
             patch.object(memo, 'db_update_memo', side_effect=RuntimeError('offline')):
            ui.session_state = state
            memo.update_memo('1', '수정')
            ui.error.assert_called_once()
        self.assertEqual(state.memos[0]['content'], '원본')

    def test_tc05_title_update(self):
        row = {'id': 'task-1', 'title': '운동', 'content': '운동', 'completed': False}
        with patch.object(core, 'task_matches', return_value=[row]), \
             patch.object(core, 'update_task', return_value=1) as update:
            core.execute('UPDATE_TASK', {'TITLE': '운동', 'NEW_TITLE': '수영'})
        self.assertEqual(update.call_args.kwargs['title'], '수영')

    def test_tc05_sql_parameter_order(self):
        conn = MagicMock()
        cursor = conn.__enter__.return_value.cursor.return_value.__enter__.return_value
        cursor.rowcount = 1
        with patch.object(db, 'get_connection', return_value=conn):
            db.update_task('task-1', content='수영', title='수영')
        sql, params = cursor.execute.call_args.args
        self.assertIn('title      = COALESCE(%s, title)', sql)
        self.assertEqual(params, (None, '수영', None, '수영', 'task-1'))

    def test_tc05_history_written(self):
        with patch.object(core, 'predict_intent', return_value=prediction()), \
             patch.object(core, 'init_db'), patch.object(core, 'execute', return_value=('추가했습니다', {'id': '1'})), \
             patch.object(core, 'has_database_url', return_value=True), patch.object(core, 'add_voice_runs') as log:
            result = core.process_text('운동 추가해', input_type='파일 업로드')
        self.assertEqual(result['status'], '성공')
        self.assertEqual(log.call_args.kwargs['input_type'], '파일 업로드')

    def test_tc06_invalid_wav(self):
        for value in [b'', b'not a wav', 'invalid']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                audio.preprocess(value, check_duplicate=False)

    def test_tc07_silence(self):
        with self.assertRaises(ValueError):
            audio.preprocess(wav(np.zeros(16000)), check_duplicate=False)

    def test_tc07_empty_text(self):
        with patch.object(core, 'has_database_url', return_value=False), patch.object(core, 'predict_intent') as predict:
            self.assertEqual(core.process_text('  ')['status'], '실패')
        predict.assert_not_called()

    def test_tc08_db_initialization_failure(self):
        with patch.object(core, 'predict_intent', return_value=prediction()), \
             patch.object(core, 'init_db', side_effect=RuntimeError('offline')), \
             patch.object(core, 'has_database_url', return_value=False):
            result = core.process_text('운동 추가해')
        self.assertEqual(result['status'], '실패')

    def test_tc08_model_failure(self):
        with patch.object(core, 'predict_intent', side_effect=RuntimeError('model')), \
             patch.object(core, 'has_database_url', return_value=False):
            self.assertEqual(core.process_text('운동 추가해')['status'], '실패')

    def test_tc08_audio_ui_recovers(self):
        state = SimpleNamespace(edited_text='이전 명령', stt_time=3)
        with patch.object(home, 'st') as ui, patch.object(home, 'transcript_wav', side_effect=ValueError('잘못된 WAV')):
            ui.session_state = state
            self.assertFalse(home.transcribe_input(b'bad'))
            ui.warning.assert_called_once()
        self.assertEqual(state.edited_text, '')

    def test_tc08_memo_failure_preserves_state(self):
        state = SimpleNamespace(memos=[{'id': '1', 'content': '원본'}])
        with patch.object(memo, 'st') as ui, patch.object(memo, 'has_database_url', return_value=True), \
             patch.object(memo, 'db_delete_memo', side_effect=RuntimeError('offline')):
            ui.session_state = state
            memo.delete_memo('1')
            ui.error.assert_called_once()
        self.assertEqual(len(state.memos), 1)

    def test_tc04_tts(self):
        async def fake_generate(text, voice):
            return b'mp3'
        with patch.object(tts, 'generate', side_effect=fake_generate):
            self.assertEqual(tts.text_to_speech('안녕하세요'), b'mp3')

    def test_tc04_tts_in_running_loop(self):
        async def fake_generate(text, voice):
            return b'mp3'
        async def call():
            return tts.text_to_speech('안녕하세요')
        with patch.object(tts, 'generate', side_effect=fake_generate):
            self.assertEqual(asyncio.run(call()), b'mp3')

    def test_tc08_tts_api_failure(self):
        with patch.object(tts, 'generate', side_effect=RuntimeError('offline')):
            with self.assertRaises(RuntimeError):
                tts.text_to_speech('안녕하세요')
        self.assertFalse(tts.tts_record[-1]['success'])

    def test_tc04_emoji_only_rejected(self):
        with patch.object(tts, 'generate') as generate:
            with self.assertRaises(ValueError):
                tts.text_to_speech('✅ 📌')
        generate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
