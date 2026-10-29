# views/home.py
# 음성 비서 홈 페이지 (입력, STT, 문장 수정, AI 의도 분석 연동)

import time
import uuid
from datetime import date, datetime
import streamlit as st
import pandas as pd
from src.stt import transcript_wav
from src.core import process_text as core_process_text
from src.tts import text_to_speech, VOICES, evaluate_tts
from pages.memo import init_memo_state
from pages.todo import init_task_state


def transcribe_input(audio):
    # 실패한 새 입력에 이전 전사문이 남아 잘못 저장되지 않도록 초기화합니다.
    st.session_state.edited_text = ''
    st.session_state.stt_time = 0.0
    try:
        text, elapsed = transcript_wav(audio)
    except ValueError as exc:
        st.warning(str(exc))
        return False
    except Exception:
        st.error('음성 인식에 실패했습니다. 모델과 입력 파일을 확인한 후 다시 시도해 주세요.')
        return False
    st.session_state.edited_text = text
    st.session_state.stt_time = elapsed
    return True


def handle_process_text(text: str, input_type: str = "직접 녹음"):
    result = core_process_text(text, input_type=input_type)
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
        "input_type": input_type,
        "intent": intent,
        "status": status,
        "latency_ms": latency_ms,
        "created_at": datetime.now(),
        "response": response_msg,
        "response_text": response_msg,
        "error_message": "" if status == '성공' else response_msg
    })

    # 실제 DB 결과를 다시 읽어 동일 제목의 다른 항목까지 변경하지 않습니다.
    if status == '성공':
        if intent in {'ADD_MEMO', 'DELETE_MEMO'}:
            init_memo_state(force=True)
        elif intent in {'ADD_TASK', 'UPDATE_TASK', 'COMPLETE_TASK', 'DELETE_TASK'}:
            init_task_state(force=True)

    return result


def render_home_view():
    st.title('🎙️ 나만의 AI 음성 비서')
    st.caption('한국어 음성 명령으로 메모와 일정을 간편하게 관리하세요.')

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
                st.session_state.current_input_type = '텍스트'
                st.rerun()

            if prompt.audio:
                audio_bytes = prompt.audio.getvalue()
                st.session_state.recorded_audio = audio_bytes
                st.session_state.current_input_type = '직접 녹음'
                with st.spinner('음성 인식 처리 중...'):
                    transcribed = transcribe_input(audio_bytes)
                if transcribed:
                    st.rerun()

        if st.session_state.get('recorded_audio'):
            st.caption("🎙️ 녹음된 음성 확인:")
            st.audio(st.session_state.recorded_audio, format="audio/wav")

    with tab2:
        wav_file = st.file_uploader(
            'wav파일을 업로드하세요',
            type=['wav']
        )

        if wav_file is not None:
            st.audio(wav_file)

        if st.button('파일 음성 변환 처리 🎧'):
            if wav_file is None:
                st.warning('파일을 먼저 업로드해주세요')
            else:
                st.session_state.current_input_type = '파일 업로드'
                with st.spinner('파일 분석 중...'):
                    transcribed = transcribe_input(wav_file.getvalue())
                if transcribed:
                    st.toast(f"전사 완료 ({st.session_state.stt_time}초)", icon="✨")
                    st.rerun()

    st.divider()

    st.subheader('문장 미리보기 및 수정')

    # 저장 후 텍스트 안전 초기화
    if st.session_state.get('clear_edited_text'):
        st.session_state.edited_text = ""
        st.session_state.clear_edited_text = False

    with st.expander('📌 사용 가능한 명령어 예시'):
        st.caption('예시에 맞게 문장을 수정하세요')
        st.info('예시에 맞게 문장을 수정하세요')
        st.markdown("""
        **📅 할 일 관리**

        **1. 할 일 추가**
        - 할 일에 장보기 추가해
        - 내일 회의 등록해
        - 할 일 추가: 운동하기

        **2. 할 일 조회**
        - 오늘 할 일 알려줘
        - 내일 미완료 할 일 보여줘
        - 완료한 할 일 보여줘

        **3. 할 일 수정**
        - 장보기 이름을 마트 가기로 바꿔
        - 회의 날짜를 내일로 변경해

        **4. 할 일 완료**
        - 장보기 완료
        - 운동 다 했어
        - 할 일 중에 회의 완료

        **5. 할 일 삭제**
        - 할 일에서 장보기 삭제
        - 운동 할 일 지워
        - 장보기 삭제

        ---

        **📝 메모 관리**

        **1. 메모 추가**
        - 우유 사기 메모해
        - 메모 추가: 회의 준비
        - 회의 준비라고 메모해
        - 메모장에 운동 계획 적어

        **2. 메모 조회**
        - 메모 목록 보여줘
        - 회의 관련 메모 보여줘

        **3. 메모 삭제**
        - 회의 관련 메모 삭제해
        - 메모 삭제해
        """)

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
            input_type = st.session_state.get('current_input_type', '텍스트')
            with st.spinner('AI 의도 분석 및 등록 중...'):
                result = handle_process_text(text, input_type=input_type)
            st.session_state.result = result
            st.session_state.clear_edited_text = result['status'] == '성공'
            st.session_state.last_tts_audio = None
            st.session_state.current_input_type = '텍스트'
            st.rerun()

    if st.session_state.result is not None:
        result = st.session_state.result
        if result['status'] == '성공':
            st.success(result['response'])
        elif result['status'] == '확인 필요':
            st.warning(result['response'])
        else:
            st.error(result['response'])

    voice_name = st.selectbox(
        '원하는 목소리를 선택하세요',
        list(VOICES.keys())
    )

    voice = VOICES[voice_name]

    if st.button('음성 재생'):
        if st.session_state.get('result') is None:
            st.warning('작업을 먼저 실행해주세요')
        else:
            with st.spinner('음성 생성 중...'):
                try:
                    t0 = time.perf_counter()
                    audio_bytes = text_to_speech(
                        st.session_state.result['response'],
                        voice=voice
                    )
                    tts_sec = time.perf_counter() - t0
                    st.session_state.last_tts_audio = audio_bytes
                    st.session_state.last_tts_sec = tts_sec
                except Exception as e:
                    st.error(f'음성 생성에 실패했습니다: {e}')

    if st.session_state.get('last_tts_audio'):
        st.audio(st.session_state.last_tts_audio, format="audio/mp3")
        if st.session_state.get('last_tts_sec'):
            st.caption(f"⚡ TTS 소요 시간: {st.session_state.last_tts_sec:.3f}초")

    st.divider()

    # ─────────────────────────────────────────────────────────────
    # TTS 생성 성능 (평가 상단으로 위치 이동)
    # ─────────────────────────────────────────────────────────────
    st.subheader("TTS 생성 성능")

    stats = evaluate_tts()

    c1, c2, c3 = st.columns(3)
    c1.metric("생성 시도", f"{stats['total']}건")
    c2.metric("평균 생성 시간", f"{stats['avg_elapsed_sec']:.2f}초")
    c3.metric("실패율", f"{stats['failure_rate']:.1f}%")

    st.divider()

    # ─────────────────────────────────────────────────────────────
    # TTS 청취 평가
    # ─────────────────────────────────────────────────────────────
    st.subheader('TTS 청취 평가')
    st.caption('청취자의 주관적인 평가입니다 (1~5점)')

    default_evals = [
        {'평가자': '문윤성', '자연스러움': 5, '명확도': 5},
        {'평가자': '이태규', '자연스러움': 5, '명확도': 4},
        {'평가자': '최준혁', '자연스러움': 4, '명확도': 5},
        {'평가자': '엄기훈', '자연스러움': 4, '명확도': 5},
    ]

    if 'tts_evaluation' not in st.session_state or len(st.session_state.tts_evaluation) <= 1:
        st.session_state.tts_evaluation = list(default_evals)

    with st.form('tts_eval_form'):
        evaluator = st.text_input('평가자 ID', placeholder='예) 사용자 이름 또는 아이디', key='input_evaluator')

        naturalness = st.number_input(
            '자연스러움',
            min_value=1,
            max_value=5,
            step=1,
            value=3
        )

        clarity = st.number_input(
            '명확도',
            min_value=1,
            max_value=5,
            step=1,
            value=3
        )
        submitted = st.form_submit_button('평가 저장', type='primary')

        if submitted:
            evaluator = evaluator.strip()

            if not evaluator:
                st.warning('평가자의 ID를 입력해주세요')
            elif any(row['평가자'] == evaluator for row in st.session_state.tts_evaluation):
                st.warning('이미 평가한 ID 입니다')
            else:
                st.session_state.tts_evaluation.append({
                    '평가자': evaluator,
                    '자연스러움': naturalness,
                    '명확도': clarity
                })
                st.success('평가를 성공적으로 저장했습니다')

    evaluation = st.session_state.tts_evaluation

    if len(evaluation) >= 1:
        st.write(f'👥 청취 평가 참여자 수: {len(evaluation)}명')
        df = pd.DataFrame(evaluation)
        m1, m2 = st.columns(2)
        m1.metric("평균 자연스러움", f"{df['자연스러움'].mean():.2f} / 5.0")
        m2.metric("평균 명료도", f"{df['명확도'].mean():.2f} / 5.0")
        st.dataframe(df, hide_index=True, use_container_width=True)
