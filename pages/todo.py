# pages/todo.py
# 할 일 관리 페이지 (FullCalendar 및 진행 중 / 끝낸 일 분리 목록)

from datetime import date, datetime, time, timedelta
import uuid
import streamlit as st

from src.db import has_database_url, list_all_tasks, add_task as db_add_task, update_task as db_update_task, delete_task as db_delete_task

PALETTE = ['#c7b8f5', '#b8e6c9', '#ffd9a8', '#f7c6d9', '#b8d8f7']   # 달력 칩 색상
DONE_COLOR = '#94A3B8'


def tasks_to_events(tasks):
    """할 일 목록 → FullCalendar 이벤트 변환"""
    events = []
    for i, t in enumerate(tasks):
        due = t.get('due_date')
        if not due:
            continue
        due_time = t.get('due_time')
        is_done = t.get('is_done', False)
        event = {
            'title': f"{'✓ ' if is_done else ''}{t.get('content', '할 일')}",
            'color': DONE_COLOR if is_done else PALETTE[i % len(PALETTE)],
            'textColor': '#FFFFFF' if is_done else '#1E1B4B',
        }
        if due_time:
            event['start'] = datetime.combine(due, due_time).isoformat()
        else:
            event['start'] = due.isoformat() if isinstance(due, (date, datetime)) else str(due)
            event['allDay'] = True
        events.append(event)
    return events


def render_calendar(tasks, key='task_calendar'):
    """FullCalendar 렌더링"""
    from streamlit_calendar import calendar as full_cal

    today_str = date.today().isoformat()
    options = {
        'initialView': 'dayGridMonth',
        'initialDate': today_str,
        'locale': 'ko',
        'firstDay': 0,
        'contentHeight': 620,
        'headerToolbar': {'left': 'prev,next today', 'center': 'title', 'right': ''},
        'displayEventTime': True,
    }
    custom_css = """
    .fc {
        background-color: #ffffff;
        border-radius: 14px;
        padding: 16px;
        color: #252943;
    }
    .fc-theme-standard th {
        background: #eef2ff !important;
        color: #252943 !important;
        font-weight: 600 !important;
        padding: 8px 0 !important;
    }
    .fc-col-header-cell-cushion, .fc-daygrid-day-number {
        color: #252943 !important;
        text-decoration: none !important;
    }
    .fc-toolbar-title {
        color: #252943 !important;
        font-weight: 700 !important;
    }
    .fc-button-primary {
        background-color: #7867ed !important;
        border-color: #7867ed !important;
    }
    .fc-button-primary:hover {
        background-color: #6654d9 !important;
        border-color: #6654d9 !important;
    }
    """
    return full_cal(events=tasks_to_events(tasks), options=options, custom_css=custom_css, key=key)


def init_task_state(force=False):
    # 이미 세션에 할 일이 로드되어 있으면 불필요한 원격 DB 반복 조회 방지 (속도 최적화)
    if not force and "tasks" in st.session_state:
        return
    # DB 연결 시 항상 실제 DB 데이터와 동기화
    if has_database_url():
        try:
            db_tasks = list_all_tasks(limit=100)
            loaded = []
            if db_tasks:
                for row in db_tasks:
                    d_val = row.get('due_date')
                    if isinstance(d_val, str):
                        try:
                            d_val = date.fromisoformat(d_val)
                        except Exception:
                            d_val = None
                    elif isinstance(d_val, datetime):
                        d_val = d_val.date()

                    loaded.append({
                        'id': str(row.get('id')),
                        'content': row.get('title') or row.get('content') or '(제목 없음)',
                        'due_date': d_val,
                        'due_time': None,
                        'is_done': bool(row.get('completed', False)),
                    })
            st.session_state.tasks = loaded
            return
        except Exception:
            st.error('할 일을 불러오지 못했습니다. DB 연결을 확인해 주세요.')
            st.session_state.setdefault('tasks', [])
            return

    # DB 미연결 시에만 데모 데이터 사용
    if "tasks" not in st.session_state:
        st.session_state.tasks = []


def format_date_ko(d):
    """날짜를 'M월 D일' 형태로 포맷"""
    if not d:
        return "-"
    if isinstance(d, str):
        try:
            d = date.fromisoformat(d)
        except Exception:
            return str(d)
    if isinstance(d, datetime):
        d = d.date()
    if isinstance(d, date):
        return f"{d.month}월 {d.day}일"
    return str(d)


def get_task_status_badge(due_date):
    """예정일 기준 상태 배지 반환 (기한 지남 / 오늘 / 예정)"""
    if not due_date:
        return '<span style="color: #94A3B8;">-</span>'

    if isinstance(due_date, str):
        try:
            due_date = date.fromisoformat(due_date)
        except Exception:
            return '<span style="color: #94A3B8;">-</span>'
    elif isinstance(due_date, datetime):
        due_date = due_date.date()
    elif not isinstance(due_date, date):
        return '<span style="color: #94A3B8;">-</span>'

    today = date.today()
    if due_date < today:
        return '<span style="color: #EF4444; font-weight: 700;">🔴 기한 지남</span>'
    elif due_date == today:
        return '<span style="color: #F59E0B; font-weight: 700;">🟠 오늘</span>'
    else:
        return '<span style="color: #94A3B8;">예정</span>'


def render_todo_view():
    init_task_state(force=True)

    st.subheader('할 일 관리(업무 관리)')

    # 새 일정 등록 간이 폼
    with st.expander("➕ 새 일정 등록", expanded=False):
        c1, c2, c3 = st.columns([3, 2, 1])
        with c1:
            new_title = st.text_input("일정 내용", placeholder="예: 팀 회의 자료 준비", key="new_task_content")
        with c2:
            new_due = st.date_input("예정일", value=date.today(), key="new_task_due")
        with c3:
            st.write("")
            st.write("")
            if st.button("추가", type="primary", key="btn_add_task", use_container_width=True):
                if new_title.strip():
                    task_id = str(uuid.uuid4())
                    if has_database_url():
                        try:
                            saved_id = db_add_task(title=new_title.strip(), due_date=new_due)
                            if saved_id:
                                task_id = str(saved_id)
                        except Exception:
                            st.error('일정 저장에 실패했습니다. 다시 시도해 주세요.')
                            return

                    st.session_state.tasks.insert(0, {
                        'id': task_id,
                        'content': new_title.strip(),
                        'due_date': new_due,
                        'due_time': None,
                        'is_done': False
                    })
                    st.toast("새 일정이 등록되었습니다.", icon="📅")
                    st.rerun()

    # 버튼 글자 잘림 방지 CSS
    st.markdown("""
    <style>
    div[data-testid="column"] button {
        padding: 4px 4px !important;
        font-size: 0.85rem !important;
        white-space: nowrap !important;
        min-height: 36px !important;
    }
    div[data-testid="column"] button p {
        font-size: 0.85rem !important;
        white-space: nowrap !important;
    }
    </style>
    """, unsafe_allow_html=True)

    tab_cal, tab_list = st.tabs(["📅 캘린더", "📋 목록"])

    with tab_cal:
        render_calendar(st.session_state.tasks)

    with tab_list:
        tasks = st.session_state.tasks
        pending_tasks = [(i, t) for i, t in enumerate(tasks) if not t.get('is_done')]
        completed_tasks = [(i, t) for i, t in enumerate(tasks) if t.get('is_done')]

        # ─────────────────────────────────────────────────────────────
        # 1. 진행 중인 할 일 (해야 할 일)
        # ─────────────────────────────────────────────────────────────
        st.markdown(f"#### ⏳ 해야 할 일 <span style='font-size: 0.9rem; color: #94A3B8;'>({len(pending_tasks)}건)</span>", unsafe_allow_html=True)
        st.caption("아직 완료되지 않은 진행 중인 일정입니다.")

        if not pending_tasks:
            st.info("진행 중인 할 일이 없습니다! 모두 완료했습니다. 🎉")
        else:
            # 테이블 헤더 (플랫 컬럼으로 글자 잘림 완벽 해결)
            h_c1, h_c2, h_c3, h_c4, h_c5, h_c6 = st.columns([3.8, 1.8, 1.8, 1.3, 1.3, 1.3], vertical_alignment="center")
            with h_c1:
                st.markdown("**할 일**")
            with h_c2:
                st.markdown("**예정일**")
            with h_c3:
                st.markdown("**상태**")
            with h_c4:
                st.markdown("<div style='text-align: center; font-weight: bold;'>완료</div>", unsafe_allow_html=True)
            with h_c5:
                st.markdown("<div style='text-align: center; font-weight: bold;'>수정</div>", unsafe_allow_html=True)
            with h_c6:
                st.markdown("<div style='text-align: center; font-weight: bold;'>삭제</div>", unsafe_allow_html=True)
            st.markdown("<hr style='margin: 4px 0 12px 0; border: 0; border-top: 1px solid #334155;'>", unsafe_allow_html=True)

            for original_idx, t in pending_tasks:
                task_id = t.get('id', str(original_idx))
                r_c1, r_c2, r_c3, r_c4, r_c5, r_c6 = st.columns([3.8, 1.8, 1.8, 1.3, 1.3, 1.3], vertical_alignment="center")

                # 수정 모드 확인
                is_editing = st.session_state.get(f"editing_task_{task_id}", False)

                if is_editing:
                    with r_c1:
                        edit_text = st.text_input("수정할 내용", value=t.get('content', ''), key=f"edit_txt_{task_id}", label_visibility="collapsed")
                    with r_c2:
                        edit_due = st.date_input("수정할 날짜", value=t.get('due_date') or date.today(), key=f"edit_due_{task_id}", label_visibility="collapsed")
                    with r_c3:
                        st.caption("수정 중")
                    with r_c4:
                        if st.button("저장", key=f"save_{task_id}", type="primary", use_container_width=True):
                            if not edit_text.strip():
                                st.warning('일정 내용을 입력해 주세요.')
                                return
                            if has_database_url() and not str(task_id).startswith("demo-"):
                                try:
                                    if not db_update_task(data_id=task_id, due_date=edit_due, content=edit_text.strip(), title=edit_text.strip()):
                                        st.warning('수정할 일정이 없습니다.')
                                        return
                                except Exception:
                                    st.error('일정 수정에 실패했습니다. 다시 시도해 주세요.')
                                    return
                            t['content'] = edit_text.strip()
                            t['due_date'] = edit_due
                            st.session_state[f"editing_task_{task_id}"] = False
                            st.rerun()
                    with r_c5:
                        if st.button("취소", key=f"cancel_{task_id}", use_container_width=True):
                            st.session_state[f"editing_task_{task_id}"] = False
                            st.rerun()
                    with r_c6:
                        st.write("")
                else:
                    with r_c1:
                        st.markdown(f"<span style='font-weight: 600; font-size: 0.98rem;'>{t.get('content', '')}</span>", unsafe_allow_html=True)
                    with r_c2:
                        st.markdown(f"<span>{format_date_ko(t.get('due_date'))}</span>", unsafe_allow_html=True)
                    with r_c3:
                        st.markdown(get_task_status_badge(t.get('due_date')), unsafe_allow_html=True)
                    with r_c4:
                        if st.button("✓", key=f"btn_done_{task_id}", help="완료 처리", use_container_width=True):
                            if has_database_url() and not str(task_id).startswith("demo-"):
                                try:
                                    if not db_update_task(data_id=task_id, completed=True):
                                        st.warning('완료할 일정이 없습니다.')
                                        return
                                except Exception:
                                    st.error('완료 처리에 실패했습니다. 다시 시도해 주세요.')
                                    return
                            t['is_done'] = True
                            st.rerun()
                    with r_c5:
                        if st.button("✏️", key=f"btn_edit_{task_id}", help="할 일 수정", use_container_width=True):
                            st.session_state[f"editing_task_{task_id}"] = True
                            st.rerun()
                    with r_c6:
                        if st.button("🗑️", key=f"btn_del_{task_id}", help="할 일 삭제", use_container_width=True):
                            if has_database_url() and not str(task_id).startswith("demo-"):
                                try:
                                    if not db_delete_task(data_id=task_id):
                                        st.warning('삭제할 일정이 없습니다.')
                                        return
                                except Exception:
                                    st.error('일정 삭제에 실패했습니다. 다시 시도해 주세요.')
                                    return
                            st.session_state.tasks = [item for item in st.session_state.tasks if str(item.get('id')) != str(task_id)]
                            st.rerun()

                st.markdown("<hr style='margin: 6px 0; border: 0; border-top: 1px solid rgba(255, 255, 255, 0.05);'>", unsafe_allow_html=True)

        st.write("")
        st.write("")

        # ─────────────────────────────────────────────────────────────
        # 2. 끝낸 일 (완료된 할 일)
        # ─────────────────────────────────────────────────────────────
        st.markdown(f"#### ✅ 끝낸 일 <span style='font-size: 0.9rem; color: #94A3B8;'>({len(completed_tasks)}건)</span>", unsafe_allow_html=True)
        st.caption("완료된 일정입니다. 취소선으로 구분되어 표시됩니다.")

        if not completed_tasks:
            st.caption("아직 완료된 일정이 없습니다.")
        else:
            # 테이블 헤더
            h_d1, h_d2, h_d3, h_d4, h_d5 = st.columns([3.8, 1.8, 1.8, 2.0, 1.9], vertical_alignment="center")
            with h_d1:
                st.markdown("**할 일**")
            with h_d2:
                st.markdown("**예정일**")
            with h_d3:
                st.markdown("**상태**")
            with h_d4:
                st.markdown("<div style='text-align: center; font-weight: bold;'>복원</div>", unsafe_allow_html=True)
            with h_d5:
                st.markdown("<div style='text-align: center; font-weight: bold;'>삭제</div>", unsafe_allow_html=True)
            st.markdown("<hr style='margin: 4px 0 12px 0; border: 0; border-top: 1px solid #334155;'>", unsafe_allow_html=True)

            for original_idx, t in completed_tasks:
                task_id = t.get('id', str(original_idx))
                r_d1, r_d2, r_d3, r_d4, r_d5 = st.columns([3.8, 1.8, 1.8, 2.0, 1.9], vertical_alignment="center")

                with r_d1:
                    # 끝낸 일에는 일자로 선 그어주기 (취소선 line-through)
                    content_str = t.get('content', '')
                    st.markdown(f"<span style='text-decoration: line-through; color: #94A3B8; font-size: 0.98rem;'>{content_str}</span>", unsafe_allow_html=True)
                with r_d2:
                    st.markdown(f"<span style='color: #64748B;'>{format_date_ko(t.get('due_date'))}</span>", unsafe_allow_html=True)
                with r_d3:
                    st.markdown("<span style='color: #10B981; font-weight: 700;'>✅ 완료</span>", unsafe_allow_html=True)
                with r_d4:
                    if st.button("↩️ 되돌리기", key=f"btn_undo_{task_id}", help="미완료로 복원", use_container_width=True):
                        if has_database_url() and not str(task_id).startswith("demo-"):
                            try:
                                if not db_update_task(data_id=task_id, completed=False):
                                    st.warning('복원할 일정이 없습니다.')
                                    return
                            except Exception:
                                st.error('일정 복원에 실패했습니다. 다시 시도해 주세요.')
                                return
                        t['is_done'] = False
                        st.rerun()
                with r_d5:
                    if st.button("🗑️ 삭제", key=f"btn_del_done_{task_id}", help="할 일 삭제", use_container_width=True):
                        if has_database_url() and not str(task_id).startswith("demo-"):
                            try:
                                if not db_delete_task(data_id=task_id):
                                    st.warning('삭제할 일정이 없습니다.')
                                    return
                            except Exception:
                                st.error('일정 삭제에 실패했습니다. 다시 시도해 주세요.')
                                return
                        st.session_state.tasks = [item for item in st.session_state.tasks if str(item.get('id')) != str(task_id)]
                        st.rerun()

                st.markdown("<hr style='margin: 6px 0; border: 0; border-top: 1px solid rgba(255, 255, 255, 0.05);'>", unsafe_allow_html=True)


if __name__ == '__main__':
    st.set_page_config(page_title="할 일 관리", layout="centered")
    render_todo_view()
