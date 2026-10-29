# views/history.py
# 처리 이력 페이지 (음성 명령 및 AI 처리 로그)

import streamlit as st


def init_history_state():
    if "history" not in st.session_state:
        st.session_state.history = [
            {
                "transcript": "오늘 회의 내용 메모해 줘",
                "intent": "ADD_MEMO",
                "status": "성공",
                "latency_ms": 142,
                "created_at": "2026-10-26 11:30:10",
                "response": "메모를 저장했습니다: '10월 26일 회의 내용'"
            },
            {
                "transcript": "내일 오후 3시 병원 예약 할 일에 추가해 줘",
                "intent": "ADD_TASK",
                "status": "성공",
                "latency_ms": 178,
                "created_at": "2026-10-26 14:15:22",
                "response": "할 일을 등록했습니다: '병원 예약' (2026-10-27)"
            }
        ]


def render_history_view():
    init_history_state()

    st.subheader('처리 이력')

    if not st.session_state.history:
        st.info("기록된 음성 처리 이력이 없습니다.")
        return

    for item in st.session_state.history:
        status_color = "#10B981" if item["status"] == "성공" else "#F59E0B"
        st.markdown(f"""
        <div style="background: #FFFFFF; border-radius: 12px; padding: 14px 18px; margin-bottom: 12px; border: 1px solid #E2E8F0; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-weight: 700; color: #1E293B; font-size: 1rem;">🗣️ "{item['transcript']}"</span>
                <span style="background: {status_color}20; color: {status_color}; font-weight: 700; font-size: 0.78rem; padding: 3px 10px; border-radius: 14px;">
                    {item['status']}
                </span>
            </div>
            <div style="font-size: 0.85rem; color: #64748B; display: flex; gap: 16px;">
                <span>🎯 의도: <b>{item['intent']}</b></span>
                <span>⚡ 지연시간: <b>{item['latency_ms']}ms</b></span>
                <span>📅 시간: {item['created_at']}</span>
            </div>
            <div style="margin-top: 8px; font-size: 0.9rem; color: #4338CA; background: #EEF2FF; padding: 7px 12px; border-radius: 8px;">
                🤖 결과: {item['response']}
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == '__main__':
    st.set_page_config(page_title="처리 이력", layout="centered")
    render_history_view()