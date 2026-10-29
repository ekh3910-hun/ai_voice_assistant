# views/memo.py
# 메모 페이지 (UI 전용)
# - 메모를 추가할 때마다 카드가 하나씩 쌓이고, 작성 날짜별로 묶여서 보입니다.
# - 카드를 누르면 메모 전체 내용이 팝업(dialog)으로 열립니다.
# DB는 아직 연결하지 않았고, 예시 메모를 st.session_state에 보관합니다.
import uuid
from datetime import date, datetime, timedelta

import streamlit as st

WEEKDAYS = "월화수목금토일"
COLS = 2            # 한 줄에 놓을 카드 수
PREVIEW_LEN = 40    # 카드에 보여 줄 글자 수


# ---------------------------------------------------------------------------
# 예시 데이터 초기화
# ---------------------------------------------------------------------------
def init_memo_state():
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
            {"id": str(uuid.uuid4()), "content": c, "created_at": t} for c, t in samples
        ]


# ---------------------------------------------------------------------------
# 동작
# ---------------------------------------------------------------------------
def add_memo():
    text = st.session_state.new_memo.strip()
    if not text:
        return
    st.session_state.memos.append(
        {"id": str(uuid.uuid4()), "content": text, "created_at": datetime.now()}
    )
    st.session_state.new_memo = ""
    st.toast("메모를 저장했습니다.", icon="✅")


def delete_memo(memo_id):
    st.session_state.memos = [m for m in st.session_state.memos if m["id"] != memo_id]
    st.toast("메모를 삭제했습니다.", icon="🗑️")


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
    left, right = st.columns(2)
    if left.button("삭제", icon=":material/delete:", use_container_width=True):
        delete_memo(memo["id"])
        st.rerun()
    if right.button("닫기", type="primary", use_container_width=True):
        st.rerun()


# ---------------------------------------------------------------------------
# 화면 렌더링 함수
# ---------------------------------------------------------------------------
def render_memo_view():
    init_memo_state()

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
                                label_visibility="collapsed")

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

    groups = {}
    for m in sorted(memos, key=lambda x: x["created_at"], reverse=True):
        groups.setdefault(m["created_at"].date(), []).append(m)

    for d, items in groups.items():
        st.markdown(f"**{day_label(d)}** &nbsp; <span style='color:#8A8B93;font-size:13px'>{len(items)}개</span>",
                    unsafe_allow_html=True)
        cols = st.columns(COLS)
        for i, memo in enumerate(items):
            with cols[i % COLS]:
                with st.container(key=f"memo_{memo['id']}"):
                    # 카드 본문을 누르면 전체 내용 팝업
                    if st.button(preview(memo["content"]), key=f"open_{memo['id']}", type="tertiary"):
                        open_memo(memo)
                    bottom_l, bottom_r = st.columns([4, 1], vertical_alignment="center")
                    bottom_l.caption(time_label(memo["created_at"]))
                    bottom_r.button("", icon=":material/delete:", key=f"del_{memo['id']}",
                                    type="tertiary", help="삭제",
                                    on_click=delete_memo, args=(memo["id"],))
        st.write("")


if __name__ == "__main__":
    st.set_page_config(page_title="메모 관리", layout="centered")
    render_memo_view()