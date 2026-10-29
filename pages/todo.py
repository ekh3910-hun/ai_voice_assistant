# views/todo.py
# 할 일 관리 페이지 (FullCalendar & HTML 격자 뷰)

import calendar as py_calendar
import html
from datetime import date, datetime, time
import streamlit as st

PALETTE = ['#c7b8f5', '#b8e6c9', '#ffd9a8', '#f7c6d9', '#b8d8f7']   # 달력 칩 색상
DONE_COLOR = '#d9d9d9'


def tasks_to_events(tasks):
    """할 일 목록 → FullCalendar 이벤트 변환"""
    events = []
    for i, t in enumerate(tasks):
        due = t.get('due_date')
        if not due:
            continue
        due_time = t.get('due_time')
        event = {
            'title': t.get('content', '할 일'),
            'color': DONE_COLOR if t.get('is_done') else PALETTE[i % len(PALETTE)],
            'textColor': '#1E1B4B',
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
    return full_cal(events=tasks_to_events(tasks), options=options, key=key)


_CSS = """<style>
.cal{display:grid;grid-template-columns:repeat(7,1fr);gap:1px;background:#e2e8f0;border:1px solid #cbd5e1;border-radius:12px;overflow:hidden;margin-top:10px}
.cal .h{background:#f1f5f9;text-align:center;padding:10px 0;font-size:14px;font-weight:700;color:#334155}
.cal .d{background:#ffffff;min-height:92px;padding:6px 8px;font-size:12px;position:relative}
.cal .d.o{background:#f8fafc;color:#94a3b8}
.cal .d.t b{background:#6366f1;color:#ffffff;border-radius:50%;padding:2px 7px}
.cal .e{margin-top:4px;padding:3px 6px;border-radius:6px;color:#1e1b4b;font-size:11px;font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
</style>"""


def month_grid_html(year, month, tasks, today=None):
    today = today or date.today()
    by_day = {}
    for i, t in enumerate(tasks):
        d_val = t.get('due_date')
        if d_val:
            by_day.setdefault(d_val, []).append((i, t))

    cells = [f'<div class="h">{name}</div>' for name in '일월화수목금토']
    for d in py_calendar.Calendar(firstweekday=6).itermonthdates(year, month):
        classes = 'd' + (' o' if d.month != month else '') + (' t' if d == today else '')
        chips = ''
        for i, t in by_day.get(d, []):
            color = DONE_COLOR if t.get('is_done') else PALETTE[i % len(PALETTE)]
            when = f"{t['due_time']:%H:%M} " if t.get('due_time') else ''
            chips += f'<div class="e" style="background:{color}">{html.escape(when + t["content"])}</div>'
        cells.append(f'<div class="{classes}"><b>{d.day}</b>{chips}</div>')

    return _CSS + '<div class="cal">' + ''.join(cells) + '</div>'


def _demo_tasks():
    today = date.today()
    return [
        {'content': '팀 회의', 'due_date': today, 'due_time': time(10, 0), 'is_done': False},
        {'content': '보고서 작성', 'due_date': today, 'due_time': time(14, 0), 'is_done': True},
        {'content': '병원 예약', 'due_date': today.replace(day=min(today.day + 1, 28)), 'due_time': time(15, 0), 'is_done': False},
        {'content': '운동하기', 'due_date': today.replace(day=min(today.day + 3, 28)), 'due_time': None, 'is_done': False},
    ]


def init_task_state():
    if "tasks" not in st.session_state:
        st.session_state.tasks = _demo_tasks()


def render_todo_view():
    init_task_state()

    st.subheader('할 일 관리')

    # 새 일정 등록 간이 폼
    with st.expander("➕ 새 일정 등록", expanded=False):
        c1, c2, c3 = st.columns([3, 2, 1])
        with c1:
            new_title = st.text_input("일정 내용", placeholder="예: 치과 검진", key="new_task_content")
        with c2:
            new_due = st.date_input("날짜", value=date.today(), key="new_task_due")
        with c3:
            st.write("")
            st.write("")
            if st.button("추가", type="primary", key="btn_add_task"):
                if new_title.strip():
                    st.session_state.tasks.append({
                        'content': new_title.strip(),
                        'due_date': new_due,
                        'due_time': None,
                        'is_done': False
                    })
                    st.toast("일정을 등록했습니다.", icon="📅")
                    st.rerun()

    view_mode = st.segmented_control(
        "달력 보기 방식",
        ["인터랙티브 캘린더 (FullCalendar)", "HTML 격자 뷰"],
        default="인터랙티브 캘린더 (FullCalendar)",
        label_visibility="collapsed"
    )

    if view_mode == "인터랙티브 캘린더 (FullCalendar)":
        render_calendar(st.session_state.tasks)
    else:
        today = date.today()
        st.markdown(month_grid_html(today.year, today.month, st.session_state.tasks), unsafe_allow_html=True)


if __name__ == '__main__':
    st.set_page_config(page_title="할 일 관리", layout="centered")
    render_todo_view()