# pages/history.py
# 처리 이력 페이지 (음성 명령 및 AI 처리 로그)

from datetime import date, datetime
from zoneinfo import ZoneInfo
import pandas as pd
import streamlit as st

from src.db import has_database_url, list_voice_runs


def normalize_datetime(dt_val):
    """타임존 충돌 방지: timezone-aware 객체는 한국 시간(KST)으로 변환 후 naive datetime으로 정규화"""
    if dt_val is None or pd.isna(dt_val):
        return datetime.now()
    if isinstance(dt_val, str):
        try:
            dt_val = pd.to_datetime(dt_val)
        except Exception:
            return datetime.now()
    if hasattr(dt_val, "tzinfo") and dt_val.tzinfo is not None:
        try:
            return dt_val.astimezone(ZoneInfo("Asia/Seoul")).replace(tzinfo=None)
        except Exception:
            return dt_val.tz_localize(None) if hasattr(dt_val, "tz_localize") else dt_val
    if hasattr(dt_val, "to_pydatetime"):
        return dt_val.to_pydatetime()
    return dt_val


def init_history_state():
    if "history" not in st.session_state:
        st.session_state.history = [
            {
                "created_at": datetime(2026, 10, 26, 14, 30),
                "input_type": "직접 녹음",
                "transcript": "내일 장보기 추가해 줘",
                "intent": "ADD_TASK",
                "status": "성공",
                "response_text": "내일 할 일에 장보기를 추가했어요.",
                "latency_ms": 2300,
                "error_message": "",
            },
            {
                "created_at": datetime(2026, 10, 26, 14, 25),
                "input_type": "텍스트",
                "transcript": "회의 준비라고 메모해 줘",
                "intent": "ADD_MEMO",
                "status": "성공",
                "response_text": "회의 준비를 메모로 저장했어요.",
                "latency_ms": 450,
                "error_message": "",
            },
            {
                "created_at": datetime(2026, 10, 26, 14, 20),
                "input_type": "파일 업로드",
                "transcript": "오늘 할 일 알려줘",
                "intent": "LIST_TASK",
                "status": "실패",
                "response_text": "잠시 후 다시 시도해 주세요.",
                "latency_ms": 1800,
                "error_message": "DB 연결에 실패했습니다.",
            },
        ]


STATUS_KO = {"success": "성공", "fail": "실패", "need_confirm": "확인 필요",
             "성공": "성공", "실패": "실패", "확인 필요": "확인 필요"}


def fetch_history_records():
    """DB 연결 시 voice_runs만 사용 (세션 이력과 섞지 않음), 없으면 예시 이력"""
    if not has_database_url():
        init_history_state()
        return list(st.session_state.history)

    try:
        db_runs = list_voice_runs(limit=500)
    except Exception as e:
        st.error(f"처리 이력을 불러오지 못했습니다: {type(e).__name__}")
        return []

    return [{
        "created_at": normalize_datetime(r.get("created_at")),
        "input_type": r.get("input_type") or "-",
        "transcript": r.get("transcript") or "(내용 없음)",
        "intent": r.get("intent") or "UNKNOWN",
        "status": STATUS_KO.get(r.get("status"), "실패"),
        "response_text": r.get("response_text") or "",
        "latency_ms": float(r.get("latency_ms") or 0),
        "error_message": r.get("error_message") or "",
    } for r in db_runs]


def render_history_view():
    init_history_state()

    st.subheader('🕒 처리 이력')
    st.caption("음성·텍스트 명령의 실행 결과를 확인하세요.")

    history_records = fetch_history_records()
    if not history_records:
        st.info("기록된 음성 처리 이력이 없습니다.")
        return

    df = pd.DataFrame(history_records)

    # 1) 필수 컬럼 기본값 보장 (KeyError 방지)
    required_defaults = {
        "created_at": datetime.now(),
        "input_type": "직접 녹음",
        "transcript": "(내용 없음)",
        "intent": "UNKNOWN",
        "status": "성공",
        "response_text": "",
        "latency_ms": 0.0,
        "error_message": "",
    }
    for col, default_val in required_defaults.items():
        if col not in df.columns:
            df[col] = default_val
        else:
            df[col] = df[col].fillna(default_val)

    # 2) 일시 컬럼 datetime 변환 (정규화된 naive datetime)
    df["created_at"] = pd.to_datetime(df["created_at"].apply(normalize_datetime))

    # ── 1. 검색 · 필터 ─────────────────────────────────────────────
    min_date = df["created_at"].min().date() if not df.empty else date.today()
    max_date = max(df["created_at"].max().date(), date.today()) if not df.empty else date.today()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("시작일", value=min_date, key="history_start")
    with col2:
        end_date = st.date_input("종료일", value=max_date, key="history_end")

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        input_type = st.selectbox("입력 방식", ["전체", "직접 녹음", "텍스트", "파일 업로드"],
                                  key="history_input_type")
    with col2:
        status = st.selectbox("처리 결과", ["전체", "성공", "실패", "확인 필요"], key="history_status")

    with col3:
        keyword = st.text_input("명령 검색", placeholder="검색할 문장을 입력하세요",
                                key="history_keyword")

    if start_date > end_date:
        st.warning("종료일은 시작일 이후로 선택해 주세요.")
        filtered = df.iloc[0:0]
    else:
        dates = df["created_at"].dt.date
        filtered = df[(dates >= start_date) & (dates <= end_date)].copy()

        if input_type != "전체":
            filtered = filtered[filtered["input_type"] == input_type]

        if status != "전체":
            filtered = filtered[filtered["status"] == status]

        if keyword.strip():
            filtered = filtered[filtered["transcript"].str.contains(
                keyword.strip(), regex=False, na=False)]

        filtered = filtered.sort_values("created_at", ascending=False)

    # ── 2. 요약 카드 ──────────────────────────────────────────────
    st.divider()
    st.caption("선택한 검색 조건 기준")
    col1, col2, col3 = st.columns(3)
    col1.metric("전체", f"{len(filtered)}건")
    col2.metric("성공", f"{(filtered['status'] == '성공').sum()}건")
    col3.metric("실패", f"{(filtered['status'] == '실패').sum()}건")

    # ── 3. 목록 ───────────────────────────────────────────────────
    st.divider()
    if filtered.empty:
        st.info("조건에 맞는 처리 이력이 없습니다.")
        return

    view = filtered[["created_at", "input_type", "transcript", "intent", "status"]].copy()
    view["created_at"] = view["created_at"].dt.strftime("%m/%d %H:%M")
    view.columns = ["실행 시간", "입력 방식", "명령 문장", "의도", "결과"]
    st.dataframe(view, hide_index=True, use_container_width=True)

    # ── 4. 상세 보기 ──────────────────────────────────────────────
    st.subheader("기록 상세 보기")
    for row in filtered.to_dict("records"):
        icon = "🟢" if row.get("status") == "성공" else "🔴"
        created_dt = row.get("created_at")
        time_str = created_dt.strftime('%m/%d %H:%M') if hasattr(created_dt, 'strftime') else str(created_dt)[:16]
        title = f"{icon} {time_str} · {row.get('transcript', '')}"

        with st.expander(title):
            st.write("입력 방식:", row.get("input_type", "직접 녹음"))
            st.write("입력·인식 문장:", row.get("transcript", ""))
            st.write("분류된 의도:", row.get("intent", ""))

            resp = row.get("response_text") or row.get("response") or "응답 없음"
            st.write("시스템 응답:", resp)

            latency = row.get("latency_ms") or 0
            if latency >= 1000:
                st.write("처리 시간:", f"{latency / 1000:.2f}초")
            else:
                st.write("처리 시간:", f"{latency:.0f}ms")

            err = row.get("error_message")
            if err:
                st.error(err)


if __name__ == '__main__':
    st.set_page_config(page_title="처리 이력", layout="centered")
    render_history_view()
