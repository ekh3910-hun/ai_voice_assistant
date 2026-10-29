# views/memo.py
# 메모 페이지 (UI 전용)
# - 메모를 추가할 때마다 카드가 하나씩 쌓이고, 작성 날짜별로 묶여서 보입니다.
# - 카드를 누르면 메모 전체 내용이 팝업(dialog)으로 열립니다.
# DB는 아직 연결하지 않았고, 예시 메모를 st.session_state에 보관합니다.
import uuid
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import streamlit as st
from src.db import has_database_url, list_memo, add_memo as db_add_memo, update_memo as db_update_memo, delete_memo as db_delete_memo

WEEKDAYS = "월화수목금토일"
COLS = 2            # 한 줄에 놓을 카드 수
PREVIEW_LEN = 40    # 카드에 보여 줄 글자 수


def normalize_datetime(dt_val):
    """타임존 충돌 방지: timezone-aware 객체는 한국 시간(KST)으로 변환 후 naive datetime으로 정규화"""
    if dt_val is None:
        return datetime.now()
    if isinstance(dt_val, str):
        try:
            dt_val = datetime.fromisoformat(dt_val)
        except Exception:
            return datetime.now()
    if hasattr(dt_val, "tzinfo") and dt_val.tzinfo is not None:
        try:
            return dt_val.astimezone(ZoneInfo("Asia/Seoul")).replace(tzinfo=None)
        except Exception:
            return dt_val.replace(tzinfo=None)
    return dt_val


# ---------------------------------------------------------------------------
# 데이터 초기화 및 DB 동기화
# ---------------------------------------------------------------------------
def init_memo_state(force=False):
    # 이미 세션에 메모가 로드되어 있으면 불필요한 원격 DB 반복 조회 방지 (속도 최적화)
    if not force and "memos" in st.session_state:
        return
    # DB 연결 시 항상 실제 DB 데이터와 실시간 동기화
    if has_database_url():
        try:
            db_memos = list_memo(limit=100)
            loaded = []
            if db_memos:
                for row in db_memos:
                    c_val = normalize_datetime(row.get('created_at'))

                    loaded.append({
                        "id": str(row.get('id')),
                        "content": row.get('content') or '',
                        "created_at": c_val
                    })
            st.session_state.memos = loaded
            return
        except Exception:
            st.error('메모를 불러오지 못했습니다. DB 연결을 확인해 주세요.')
            st.session_state.setdefault('memos', [])
            return

    # DB 미연결 시에만 예시 샘플 표시
    if "memos" not in st.session_state:
        now = datetime.now().replace(second=0, microsecond=0)
        today = now.replace(hour=0, minute=0)

        def at(days_ago, h, m):
            return today - timedelta(days=days_ago) + timedelta(hours=h, minutes=m)

        samples = [
            ("다음 발표 주제는 음성 인식", at(0, 11, 2)),
            ("프린터 잉크 사야 함", at(0, 9, 40)),
            ("교수님 면담은 목요일", at(1, 20, 15)),
            ("주차 위치 지하 2층", at(1, 14, 30)),
            ("회의실 예약 확인 필요", at(3, 16, 5)),
            ("우유랑 계란 사기", at(3, 10, 20)),
        ]
        st.session_state.memos = [
            {"id": f"sample-{uuid.uuid4()}", "content": c, "created_at": t} for c, t in samples
        ]


# ---------------------------------------------------------------------------
# 동작
# ---------------------------------------------------------------------------
def add_memo():
    text = st.session_state.new_memo.strip()
    if not text:
        return

    memo_id = str(uuid.uuid4())
    if has_database_url():
        try:
            saved_id = db_add_memo(text)
            if saved_id:
                memo_id = str(saved_id)
        except Exception:
            st.error('메모 저장에 실패했습니다. 다시 시도해 주세요.')
            return

    st.session_state.memos.insert(
        0, {"id": memo_id, "content": text, "created_at": datetime.now()}
    )
    st.session_state.new_memo = ""
    st.toast("메모를 저장했습니다.", icon="✅")


def delete_memo(memo_id):
    if has_database_url() and not str(memo_id).startswith("sample-"):
        try:
            if not db_delete_memo(memo_id):
                st.warning('삭제할 메모가 없습니다. 목록을 새로 확인해 주세요.')
                return
        except Exception:
            st.error('메모 삭제에 실패했습니다. 다시 시도해 주세요.')
            return

    st.session_state.memos = [m for m in st.session_state.memos if m["id"] != memo_id]
    st.toast("메모를 삭제했습니다.", icon="🗑️")


def update_memo(memo_id, new_content):
    edit_key = f"edit_content_{memo_id}"
    if new_content is None:
        new_content = st.session_state.get(edit_key, "")
    new_content = new_content.strip()

    if not new_content:
        st.session_state.pending_toast = ('수정할 내용을 입력해 주세요', '⚠️')
        return

    for m in st.session_state.memos:
        if m['id'] == memo_id:
            if m['content'] == new_content:  # 바뀐 내용이 없으면 조용히 종료
                return
            if has_database_url() and not str(memo_id).startswith("sample-"):
                try:
                    if not db_update_memo(data_id=memo_id, content=new_content):
                        st.warning('수정할 메모가 없습니다.')
                        return
                except Exception:
                    st.error('메모 수정에 실패했습니다. 다시 시도해 주세요.')
                    return
            m['content'] = new_content
            st.session_state.pop(edit_key, None)  # 카드 입력칸이 새 내용으로 다시 채워지도록
            st.session_state.pending_toast = ('메모를 수정했습니다', '✏️')
            return


# ---------------------------------------------------------------------------
# 표시 도우미
# ---------------------------------------------------------------------------
def day_label(d: date):
    diff = (date.today() - d).days
    base = f"{d.month}/{d.day}({WEEKDAYS[d.weekday()]})"
    return {0: f"오늘 · {base}", 1: f"어제 · {base}"}.get(diff, base)


def time_label(t: datetime):
    ampm = "오전" if t.hour < 12 else "오후"
    hour = t.hour % 12 or 12
    return f"{ampm} {hour}:{t.minute:02d}"


def preview(text):
    return text if len(text) <= PREVIEW_LEN else text[:PREVIEW_LEN] + "…"


@st.dialog("메모")
def open_memo(memo):
    t = memo["created_at"]
    st.caption(f"{day_label(t.date())} {time_label(t)}")
    st.markdown(memo["content"])
    st.divider()
    left, middle, right = st.columns(3)
    if left.button("삭제", icon=":material/delete:", use_container_width=True):
        delete_memo(memo["id"])
        st.rerun()
    if middle.button('수정', icon="✏️", use_container_width=True, key=f"correction_{memo['id']}"):
        st.session_state[f'editing_{memo["id"]}'] = True
    if st.session_state.get(f'editing_{memo["id"]}', False):
        new_content = st.text_input(
            '수정할 내용',
            value=memo['content'],
            key=f'dialog_edit_{memo["id"]}'
        )
        col1, col2 = st.columns(2)

        with col1:
            if st.button('저장', key=f'dialog_save_{memo["id"]}', type="tertiary"):
                if new_content.strip():
                    update_memo(memo['id'], new_content)
                    st.session_state[f'editing_{memo["id"]}'] = False
                    st.rerun()
                else:
                    st.warning('수정할 내용을 입력해주세요')
        with col2:
            if st.button('취소', key=f'dialog_cancel_{memo["id"]}', type="tertiary"):
                st.session_state[f'editing_{memo["id"]}'] = False
                st.rerun()
    if right.button("닫기", type="primary", use_container_width=True):
        st.rerun()



# ---------------------------------------------------------------------------
# 화면 렌더링 함수
# ---------------------------------------------------------------------------
def render_memo_view():
    init_memo_state()

    pending = st.session_state.pop("pending_toast", None)
    if pending:
        st.toast(pending[0], icon=pending[1])

    # 카드 스타일: key가 "memo_"로 시작하는 컨테이너를 노란 메모지처럼 꾸밉니다
    st.markdown(
        """
        <style>
        div[class*="st-key-memo_"] {
            background: #FFF8E1;
            border: 1px solid #EEDFAE;
            border-radius: 12px;
            padding: 14px 14px 6px;
            min-height: 128px;
            justify-content: space-between;
        }
        div[class*="st-key-memo_"] * { color: #3A2F0B !important; }
        div[class*="st-key-memo_"] input {color: #FFFFFF !important;
        background-color: #262730 !important;
}
        div[class*="st-key-memo_"] button[kind="tertiary"] p { font-size: 16px; text-align: left; }
        div[class*="st-key-open_"] button { justify-content: flex-start; width: 100%; min-height: 56px; align-items: flex-start; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("메모 관리")

    # 1. 새 메모
    with st.container(border=True):
        st.text_area("새 메모", key="new_memo", height=90,
                     placeholder="예: 회의실 예약 확인 필요", label_visibility="collapsed")
        st.button("저장", type="primary", on_click=add_memo)

    # 2. 필터
    f1, f2 = st.columns([1, 2], vertical_alignment="bottom")
    with f1:
        period = st.segmented_control("작성 날짜", ["전체", "오늘", "이번 주"],
                                      default="전체", label_visibility="collapsed")
    with f2:
        keyword = st.text_input("키워드 검색", placeholder="키워드로 찾기 (예: 회의)",
                                label_visibility="collapsed", key='find')

    memos = st.session_state.memos
    today = date.today()
    if period == "오늘":
        memos = [m for m in memos if m["created_at"].date() == today]
    elif period == "이번 주":
        monday = today - timedelta(days=today.weekday())
        memos = [m for m in memos if m["created_at"].date() >= monday]
    if keyword:
        memos = [m for m in memos if keyword in m["content"]]

    # 3. 날짜별 카드 목록 (최신 날짜·최신 메모가 위로)
    if not memos:
        st.info("메모가 없습니다. 위에서 새 메모를 추가해 보세요.")

    for m in memos:
        m["created_at"] = normalize_datetime(m.get("created_at"))

    groups = {}
    for m in sorted(memos, key=lambda x: normalize_datetime(x.get("created_at")), reverse=True):
        groups.setdefault(m["created_at"].date(), []).append(m)

    for d, items in groups.items():
        st.markdown(f"**{day_label(d)}** &nbsp; <span style='color:#8A8B93;font-size:13px'>{len(items)}개</span>",
                    unsafe_allow_html=True)
        cols = st.columns(COLS)
        for i, memo in enumerate(items):
            with (cols[i % COLS]):
                with st.container(key=f"memo_{memo['id']}"):
                    # 카드 본문을 누르면 전체 내용 팝업
                    if st.button(preview(memo["content"]), key=f"open_{memo['id']}", type="tertiary"):
                        open_memo(memo)
                    bottom_l, bottom_r = st.columns([3.2, 1.2], vertical_alignment="center")
                    bottom_l.caption(time_label(memo["created_at"]))
                    with bottom_r:
                        left, right = st.columns(2)
                        with left:
                            if st.button("", icon=":material/edit:", type="tertiary",
                                      key=f"update_{memo['id']}", help="메모 수정"):
                                st.session_state[f'editing_{memo["id"]}'] = True
                                st.rerun()
                        with right:
                            st.button("", icon=":material/delete:", key=f"del_{memo['id']}",
                                    type="tertiary", help="삭제",
                                    on_click=delete_memo, args=(memo["id"],))
                    if st.session_state.get(f'editing_{memo["id"]}', False):
                        new_content = st.text_input(
                            '수정할 내용',
                            value=memo['content'],
                            key=f'card_edit_{memo["id"]}'
                        )
                        col1, col2 = st.columns(2)

                        with col1:
                            if st.button('저장', key=f'card_save_{memo["id"]}', type="tertiary"):
                                if new_content.strip():
                                    update_memo(memo['id'], new_content)
                                    st.session_state[f'editing_{memo["id"]}'] = False
                                    st.rerun()
                                else:
                                    st.warning('수정할 내용을 입력해주세요')
                        with col2:
                            if st.button('취소', key=f'card_cancel_{memo["id"]}', type="tertiary"):
                                st.session_state[f'editing_{memo["id"]}'] = False
                                st.rerun()

        st.write("")


if __name__ == "__main__":
    st.set_page_config(page_title="메모 관리", layout="centered")
    render_memo_view()
