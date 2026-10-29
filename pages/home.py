# views/home.py
# 음성 비서 홈 페이지 (입력, STT, 문장 수정, AI 의도 분석 연동)

import time
import uuid
from datetime import date, datetime
import streamlit as st
from src.stt import transcript_wav
from src.core import process_text as core_process_text
from src.tts import text_to_speech, VOICES


def handle_process_text(text: str):
    result = core_process_text(text)
    intent = result.get('intent', 'UNKNOWN')
    slots = result.get('slots', {})
    status = result.get('status', '실패')
    response_msg = result.get('response', '')
    latency_ms = result.get('latency_ms', 0)

    # 1. 이력 세션 상태 동기화
    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.insert(0, {
        "transcript": text,
        "intent": intent,
        "status": status,
        "latency_ms": latency_ms,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "response": response_msg
    })

    # 2. 메모 / 할 일 세션 상태 동기화 (성공 시)
    if status == '성공':
        if "memos" not in st.session_state:
            st.session_state.memos = []
        if "tasks" not in st.session_state:
            st.session_state.tasks = []

        if intent == 'ADD_MEMO':
            content = slots.get('MEMO_CONTENT', text)
            memo_id = result.get('data', {}).get('id') if isinstance(result.get('data'), dict) else str(uuid.uuid4())
            st.session_state.memos.insert(0, {
                "id": str(memo_id),
                "content": content,
                "created_at": datetime.now()
            })

        elif intent == 'ADD_TASK':
            title = slots.get('TITLE', text)
            due_str = slots.get('DUE_DATE')
            try:
                due_date = date.fromisoformat(due_str) if due_str else date.today()
            except Exception:
                due_date = date.today()

            task_id = result.get('data', {}).get('id') if isinstance(result.get('data'), dict) else str(uuid.uuid4())
            st.session_state.tasks.append({
                'id': str(task_id),
                'content': title,
                'due_date': due_date,
                'due_time': None,
                'is_done': False
            })

        elif intent == 'COMPLETE_TASK':
            title = slots.get('TITLE', '')
            for t in st.session_state.tasks:
                if title in t.get('content', ''):
                    t['is_done'] = True

        elif intent == 'DELETE_TASK':
            title = slots.get('TITLE', '')
            st.session_state.tasks = [t for t in st.session_state.tasks if title not in t.get('content', '')]

        elif intent == 'DELETE_MEMO':
            keyword = slots.get('KEYWORD', '')
            if keyword:
                st.session_state.memos = [m for m in st.session_state.memos if keyword not in m.get('content', '')]

    return result


def render_home_view():
    st.header('음성 비서 홈')

    tab1, tab2 = st.tabs(['녹음/텍스트입력', '파일 업로드'])

    if "recorded_audio" not in st.session_state:
        st.session_state.recorded_audio = None

    with tab1:
        prompt = st.chat_input(
            '명령어를 입력하거나 음성 녹음을 실행하세요',
            accept_audio=True,
            key='voice_prompt'
        )
        if prompt:
            if prompt.text:
                st.session_state.edited_text = prompt.text
                st.session_state.stt_time = 0.0
                st.session_state.recorded_audio = None
                st.rerun()

            if prompt.audio:
                audio_bytes = prompt.audio.getvalue()
                st.session_state.recorded_audio = audio_bytes
                with st.spinner('음성 인식 처리 중...'):
                    text, elapsed = transcript_wav(audio_bytes)
                st.session_state.edited_text = text
                st.session_state.stt_time = elapsed
                st.rerun()

        if st.session_state.get('recorded_audio'):
            st.caption("🎙️ 녹음된 음성 확인:")
            st.audio(st.session_state.recorded_audio, format="audio/wav")

    with tab2:
        wav_file = st.file_uploader(
            'wav파일을 업로드하세요',
            type=['wav', 'mp3']
        )

        if wav_file is not None:
            st.audio(wav_file)

        if st.button('파일 음성 변환 처리 🎧'):
            if wav_file is None:
                st.warning('파일을 먼저 업로드해주세요')
            else:
                with st.spinner('파일 분석 중...'):
                    text, elapsed = transcript_wav(wav_file.getvalue())
                st.session_state.edited_text = text
                st.session_state.stt_time = elapsed
                st.toast(f"전사 완료 ({elapsed}초)", icon="✨")
                st.rerun()

    st.divider()

    st.subheader('문장 미리보기 및 수정')

    # 저장 후 텍스트 안전 초기화
    if st.session_state.get('clear_edited_text'):
        st.session_state.edited_text = ""
        st.session_state.clear_edited_text = False

    edited_text = st.text_area(
        '인식된 문장을 확인하고 수정하세요',
        key='edited_text',
        height=120
    )

    if st.session_state.stt_time > 0:
        st.caption(f"⚡ STT 소요 시간: {st.session_state.stt_time:.3f}초")

    if st.button('저장', type='primary', use_container_width=True):
        text = edited_text.strip()

        if not text:
            st.warning('음성녹음/텍스트 입력, 파일업로드를 먼저 해주세요')
        else:
            with st.spinner('AI 의도 분석 및 등록 중...'):
                result = handle_process_text(text)
            st.session_state.result = result
            st.session_state.clear_edited_text = True

    if st.session_state.result is not None:
        result = st.session_state.result
        if result['status'] == '성공':
            st.success(f"✅ {result['response']}")
        else:
            st.error(f"❌ {result['response']}")

    st.divider()

    voice_name = st.selectbox(
        '목소리 선택',
        list(VOICES.keys())
    )

    voice = VOICES[voice_name]

    if st.button('음성 재생'):
        if result is None:
            st.warning('작업을 먼저 실행해주세요')
        else:
            with st.spinner('음성 생성 중...'):
                audio_bytes = text_to_speech(
                    result['response'],
                    voice=voice
                )

            st.audio(audio_bytes, format="audio/mp3", autoplay=True)