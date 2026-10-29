import os
import io
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from src import tts
from src import stt
import numpy as np
import soundfile as sf
from test_stt import main as stt_main

ROOT = Path(__file__).resolve().parents[1]


class SpeechTests(unittest.TestCase):
    def test_stt_decodes_audio_and_uses_korean_command_context(self):
        class Segment:
            text = ' 내일 할 일에 운동 추가해 줘. '
        class Model:
            def transcribe(self, audio, **kwargs):
                self_test.assertEqual(kwargs['language'], 'ko')
                self_test.assertTrue(kwargs['vad_filter'])
                self_test.assertIn('할 일', kwargs['initial_prompt'])
                self_test.assertEqual(len(audio), 16000)
                return iter([Segment()]), None
        self_test = self
        audio = io.BytesIO()
        sf.write(audio, np.sin(np.arange(16000) * 0.05) * 0.1, 16000, format='WAV')
        with patch.object(stt, 'load_stt_model', return_value=Model()) as load:
            text, elapsed = stt.transcript_wav(audio.getvalue())
        load.assert_called_once_with('small')
        self.assertEqual(text, '내일 할 일에 운동 추가해 줘')
        self.assertGreaterEqual(elapsed, 0)

    def test_silence_never_reaches_model(self):
        audio = io.BytesIO()
        sf.write(audio, np.zeros(16000), 16000, format='WAV')
        with patch.object(stt, 'load_stt_model') as load:
            with self.assertRaisesRegex(ValueError, '무음'):
                stt.transcript_wav(audio.getvalue())
        load.assert_not_called()

    def test_tts_app_contract(self):
        class Communicate:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self_test.assertEqual(kwargs['proxy'], 'http://proxy:8080')
                self_test.assertEqual(kwargs['voice'], 'ko-KR-SunHiNeural')
            async def stream(self):
                yield {'type': 'audio', 'data': b'first'}
                yield {'type': 'SentenceBoundary', 'text': 'ignored'}
                yield {'type': 'audio', 'data': b'second'}
        self_test = self
        with patch.dict(os.environ, {'HTTPS_PROXY': 'http://proxy:8080'}), patch.object(tts.edge_tts, 'Communicate', Communicate):
            audio, elapsed = tts.synthesize_speech('안녕하세요')
        self.assertEqual(audio, b'firstsecond')
        self.assertGreaterEqual(elapsed, 0)

    def test_empty_audio_is_failure(self):
        class EmptyCommunicate:
            def __init__(self, **kwargs): pass
            async def stream(self):
                yield {'type': 'SentenceBoundary'}
        with patch.object(tts.edge_tts, 'Communicate', EmptyCommunicate):
            with self.assertRaises(RuntimeError):
                tts.text_to_speech('안녕하세요')

    def test_stt_runner_reports_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / 'sample.wav').write_bytes(b'not-used')
            with patch('test_stt.transcript_wav', side_effect=RuntimeError('model unavailable')):
                self.assertEqual(stt_main(['--audio-dir', temp, '--output', str(path / 'result.csv')]), 1)
            self.assertIn('model unavailable', (path / 'result.csv').read_text())
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(stt_main(['--audio-dir', temp]), 1)


class AppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / 'test.sqlite3'
        self.env = patch.dict(os.environ, {'ASSISTANT_DB_PATH': str(self.db)})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def command_app(self, name='app1_b.py'):
        # Execute the real app and expose command results through Streamlit state.
        source = (ROOT / name).read_text().replace(
            'ROOT = Path(__file__).resolve().parent', f'ROOT = Path({str(ROOT)!r})'
        )
        source += '\nst.session_state.result = process_message(st.session_state.get("command", ""))\n'
        return AppTest.from_string(source, default_timeout=30)

    def test_commands_and_no_accidental_writes(self):
        app = self.command_app()
        for text, intent in [('내일 오후 3시 병원 예약 추가해 줘', 'ADD_TASK'), ('우유 사기 메모해 줘', 'ADD_MEMO')]:
            app.session_state.command = text
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state.result[1], intent)
            self.assertTrue(app.session_state.result[2])
        for text in ('메모 삭제해 줘', '내일 할 일 완료했어', '내일 24시 할 일 추가해 줘', '내일 오후 13시 할 일 추가해 줘'):
            app.session_state.command = text
            app.run()
            self.assertFalse(app.session_state.result[2], text)
        with sqlite3.connect(self.db) as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM tasks').fetchone()[0], 1)
            self.assertEqual(conn.execute('SELECT count(*) FROM memos').fetchone()[0], 1)
            self.assertEqual(conn.execute('SELECT due_time FROM tasks').fetchone()[0], '15:00')

    def test_both_apps_render(self):
        for name in ('app1.py', 'app1_b.py'):
            app = AppTest.from_file(str(ROOT / name), default_timeout=30).run()
            self.assertFalse(app.exception, name)
            for page in ('메모 관리', '할 일 관리', '처리 이력'):
                app.button(key='nav_' + page).click().run()
                self.assertFalse(app.exception, page)

    def test_memo_edit_delete_and_task_complete_delete(self):
        app = AppTest.from_file(str(ROOT / 'app1_b.py'), default_timeout=30).run()
        app.button(key='nav_메모 관리').click().run()
        next(x for x in app.text_input if x.label == '메모 제목').input('검증 메모')
        next(x for x in app.text_area if x.label == '메모 내용').input('원본 내용')
        next(x for x in app.button if x.label == '메모 저장').click().run()
        with sqlite3.connect(self.db) as conn:
            memo_id = conn.execute('SELECT id FROM memos').fetchone()[0]
        app.button(key=f'edit_{memo_id}').click().run()
        next(x for x in app.text_input if x.label == '제목').input('수정한 메모')
        next(x for x in app.button if x.label == '수정 저장').click().run()
        with sqlite3.connect(self.db) as conn:
            self.assertEqual(conn.execute('SELECT title FROM memos').fetchone()[0], '수정한 메모')
        app.button(key=f'del_{memo_id}').click().run()
        app.button(key=f'yes_{memo_id}').click().run()
        app.button(key='nav_할 일 관리').click().run()
        next(x for x in app.text_input if x.label == '할 일 제목').input('검증 일정')
        next(x for x in app.button if x.label == '일정 저장').click().run()
        with sqlite3.connect(self.db) as conn:
            task_id = conn.execute('SELECT id FROM tasks').fetchone()[0]
        app.checkbox(key=f'task_{task_id}').check().run()
        with sqlite3.connect(self.db) as conn:
            self.assertEqual(conn.execute('SELECT completed FROM tasks').fetchone()[0], 1)
        app.button(key=f'del_task_{task_id}').click().run()
        self.assertFalse(app.exception)
        with sqlite3.connect(self.db) as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM memos').fetchone()[0], 0)
            self.assertEqual(conn.execute('SELECT count(*) FROM tasks').fetchone()[0], 0)


if __name__ == '__main__':
    unittest.main()
