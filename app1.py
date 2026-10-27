"""AI 음성 비서: 이미지 시안 기반 Streamlit UI.
로컬 SQLite에 저장. 팀 Neon DB/STT/TTS는 아직 미연결.
실행: python -m streamlit run app.py
"""
import calendar
import base64
import hashlib
import html
import re
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from time import perf_counter

import streamlit as st
import streamlit.components.v1 as components

if 'page' not in st.session_state:
    st.session_state.page = '대화하기'

ROOT = Path(__file__).resolve().parent
chat_bar_component = components.declare_component('voice_chat_bar', path=str(ROOT / 'chat_bar'))
DB_FILE = ROOT / 'assistant_data.sqlite3'
COLORS = ['#FFF2BE', '#FFE0E8', '#DBECFF', '#DDF6DD', '#E9E2FF']
st.set_page_config(page_title='AI 음성 비서', page_icon='🤖', layout='wide', initial_sidebar_state='expanded')
st.set_option('client.showSidebarNavigation', False)

def db():
    conn = sqlite3.connect(DB_FILE, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute('''CREATE TABLE IF NOT EXISTS memos (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, content TEXT NOT NULL, color TEXT NOT NULL, created_at TEXT NOT NULL)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS tasks (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, due_date TEXT NOT NULL, due_time TEXT DEFAULT '', completed INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS processing_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        input_type TEXT NOT NULL,
        transcript TEXT NOT NULL,
        intent TEXT NOT NULL,
        status TEXT NOT NULL,
        stt_ms REAL NOT NULL DEFAULT 0,
        intent_ms REAL NOT NULL DEFAULT 0,
        db_ms REAL NOT NULL DEFAULT 0,
        total_ms REAL NOT NULL DEFAULT 0,
        response_text TEXT NOT NULL DEFAULT '',
        error_type TEXT NOT NULL DEFAULT ''
    )''')
    conn.commit()
    return conn


def query(sql, params=()):
    with db() as conn:
        return conn.execute(sql, params).fetchall()


def execute(sql, params=()):
    with db() as conn:
        conn.execute(sql, params)
        conn.commit()


def esc(s):
    return html.escape(str(s), quote=True)


def parse_date(text):
    today = date.today()
    if '모레' in text: return today + timedelta(days=2)
    if '내일' in text: return today + timedelta(days=1)
    if '오늘' in text: return today
    m = re.search(r'(\d{4})[.\-/년]\s*(\d{1,2})[.\-/월]\s*(\d{1,2})일?', text)
    if m:
        try: return date(*map(int, m.groups()))
        except ValueError: return None
    m = re.search(r'(\d{1,2})월\s*(\d{1,2})일', text)
    if m:
        try:
            d = date(today.year, int(m.group(1)), int(m.group(2)))
            return d if d >= today else date(today.year + 1, d.month, d.day)
        except ValueError: return None
    m = re.search(r'(이번\s*주|다음\s*주)?\s*([월화수목금토일])요일', text)
    if m:
        day_index = '월화수목금토일'.index(m.group(2))
        base = today - timedelta(days=today.weekday())
        if m.group(1) and '다음' in m.group(1): base += timedelta(days=7)
        d = base + timedelta(days=day_index)
        if d < today and not m.group(1): d += timedelta(days=7)
        return d
    return None


def parse_time(text):
    m = re.search(r'(오전|오후)?\s*(\d{1,2})시(?:\s*(\d{1,2})분)?', text)
    if not m: return ''
    hour = int(m.group(2))
    if hour > 23: return ''
    if m.group(1) == '오후' and hour < 12: hour += 12
    if m.group(1) == '오전' and hour == 12: hour = 0
    minute = int(m.group(3) or 0)
    return f'{hour:02d}:{minute:02d}' if minute < 60 else ''


def clean_content(text, kind):
    s = text.strip()
    s = re.sub(r'(?:해\s*줘|해주세요|해줘|줘|주세요|해\s*줘요)[.!?\s]*$', '', s).strip()
    if kind == 'memo':
        s = re.sub(r'(?:메모(?:로)?|기록(?:으로)?)\s*(?:해|남겨|저장해|적어|추가해)?\s*$', '', s).strip()
        s = re.sub(r'^(?:메모(?:해|로)?|기록해)\s*', '', s).strip()
    else:
        s = re.sub(r'(?:할\s*일(?:에|로)?|일정(?:에|으로)?)\s*', '', s)
        s = re.sub(r'(?:추가|등록|저장|잡아|넣어)(?:해)?\s*$', '', s).strip()
        s = re.sub(r'(오늘|내일|모레|\d{1,2}월\s*\d{1,2}일|(?:이번|다음)\s*주\s*[월화수목금토일]요일)', '', s)
        s = re.sub(r'(오전|오후)?\s*\d{1,2}시(?:\s*\d{1,2}분)?(?:에|부터)?', '', s)
    s = re.sub(r'\s+', ' ', s).strip(' ,.-')
    return s or text.strip()


def process_message(text):
    """규칙 기반 명령 처리. (응답, 의도, 성공여부, 의도분류ms, DBms) 반환."""
    start = perf_counter()
    msg = text.strip()
    is_memo = bool(re.search(r'메모|기록해|기록해줘|적어\s*줘', msg))
    is_task = bool(re.search(r'할\s*일|일정|예약|등록|추가|해야\s*해|까지|완료', msg))
    is_list = bool(re.search(r'보여\s*줘|조회|목록|알려\s*줘', msg))
    intent = ('LIST_MEMO' if is_list else 'ADD_MEMO') if is_memo else (
        ('LIST_TASK' if is_list else 'ADD_TASK') if is_task else 'UNKNOWN'
    )
    intent_ms = (perf_counter() - start) * 1000
    db_ms = 0.0
    if not msg:
        return '내용을 입력해 주세요.', 'UNKNOWN', False, intent_ms, db_ms
    if is_list:
        if is_memo:
            return '메모 관리 탭에서 저장된 메모를 확인해 주세요.', intent, True, intent_ms, db_ms
        if is_task:
            return '할 일 관리 탭의 달력에서 확인해 주세요.', intent, True, intent_ms, db_ms
    if is_memo:
        content = clean_content(msg, 'memo')
        title = (content[:20] + '…') if len(content) > 20 else content
        if not title:
            return '저장할 메모 내용을 알려 주세요.', intent, False, intent_ms, db_ms
        t = perf_counter()
        execute('INSERT INTO memos(title,content,color,created_at) VALUES(?,?,?,?)',
                (title, content, COLORS[0], datetime.now().isoformat(timespec='seconds')))
        db_ms = (perf_counter()-t)*1000
        return f'📝 메모를 저장했어요! 「{title}」 — 메모 관리 탭에서 확인할 수 있어요.', intent, True, intent_ms, db_ms
    if is_task:
        when = parse_date(msg)
        if when is None:
            return '📅 날짜를 알려 주세요. 예: "내일 오후 3시 병원 예약 추가해 줘"', intent, False, intent_ms, db_ms
        title = clean_content(msg, 'task')
        if not title:
            return '등록할 할 일 내용을 알려 주세요.', intent, False, intent_ms, db_ms
        tm = parse_time(msg)
        t = perf_counter()
        execute('INSERT INTO tasks(title,due_date,due_time,created_at) VALUES(?,?,?,?)',
                (title, when.isoformat(), tm, datetime.now().isoformat(timespec='seconds')))
        db_ms = (perf_counter()-t)*1000
        return f'📅 {when:%m월 %d일} {tm}에 「{title}」 할 일을 등록했어요! 달력에서 확인해 주세요.', intent, True, intent_ms, db_ms
    return '메모나 할 일 명령을 입력해 주세요. 예: "회의 내용 메모해 줘", "내일 병원 예약 추가해 줘"', intent, False, intent_ms, db_ms


def log_history(input_type, transcript, intent, success, stt_ms, intent_ms, db_ms, total_ms, response, error_type=''):
    execute("""INSERT INTO processing_history
        (created_at,input_type,transcript,intent,status,stt_ms,intent_ms,db_ms,total_ms,response_text,error_type)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (datetime.now().isoformat(timespec='seconds'), input_type, transcript, intent,
         '성공' if success else '실패', stt_ms, intent_ms, db_ms, total_ms, response, error_type))



import base64

def image_uri(path):
    return 'data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode() if path.exists() else ''

ROBOT = image_uri(ROOT / 'assets' / 'robot.png')

st.markdown('''<style>
[data-testid="stSidebarNav"] {
    display: none !important;
}
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"],.stApp {font-family:'Noto Sans KR',sans-serif!important}
.stApp {background:linear-gradient(120deg,#e6ebff 0%,#f7f8ff 38%,#eff0ff 100%);color:#25345b}
#MainMenu,footer {visibility:hidden}
[data-testid="stMainBlockContainer"] {max-width:1250px!important;padding:20px 28px 55px!important}
[data-testid="stSidebar"] {background:linear-gradient(160deg,#dce6ff,#ecf0ff 68%,#ded8ff)!important;border-right:1px solid #d8dfff!important;width:286px!important}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap:.55rem}
[data-testid="stSidebar"] .stButton>button {width:100%;border:none!important;box-shadow:none!important;text-align:left;justify-content:flex-start;background:transparent;color:#344c9e;border-radius:13px;padding:13px 17px;font-size:15px;font-weight:600}
[data-testid="stSidebar"] .stButton>button:hover {background:#e0e5ff}
[data-testid="stSidebar"] .stButton>button[kind="primary"] {background:#cdd5ff!important;color:#4153c5!important}
.stButton>button,.stFormSubmitButton>button {border-radius:12px!important;border:1px solid #dce2fa!important;font-weight:700}
.stButton>button[kind="primary"],.stFormSubmitButton>button[kind="primary"] {background:linear-gradient(110deg,#6e7af2,#8572f7)!important;border:none!important;color:white!important;box-shadow:0 6px 16px #7776ec33}
.stButton>button:hover {border-color:#aebafa!important}
[data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea {background:#fff!important;border:1px solid #e3e6fa!important;border-radius:12px!important}
[data-testid="stForm"] {background:#fff!important;border:1px solid #e6e8fb!important;border-radius:22px!important;padding:15px 19px!important;box-shadow:0 8px 28px #5864aa12}
[data-testid="stExpander"] {background:#ffffffb8;border-radius:16px;border:1px solid #e1e6fb}
[data-testid="stHorizontalBlock"] {align-items:stretch}
.hero {background:linear-gradient(120deg,#fff 10%,#f2f1ff 100%);border-radius:22px;padding:23px 30px;border:1px solid #ececff;box-shadow:0 10px 26px #737baf10;margin-bottom:19px}
.hero-title {font-size:27px;font-weight:800;color:#1e2c54;display:flex;align-items:center;gap:14px}
.hero-icon {background:linear-gradient(130deg,#e7e7ff,#f3eaff);padding:8px 13px;border-radius:17px;font-size:27px}
.hero-desc {font-size:14px;color:#7d87a9;margin:8px 0 0 63px}
.side-logo {font-size:21px;font-weight:800;color:#24396e;padding-top:5px}
.side-desc {font-size:12px;color:#7884a5;margin:2px 0 13px}
.side-robot {margin-top:22px;padding:8px;background:linear-gradient(155deg,#edf3ff,#e2dcff);border-radius:27px;overflow:hidden}
.side-robot img {width:100%;border-radius:22px;display:block}
.side-footer {color:#8a92ae;font-size:11px;line-height:1.7;margin-top:9px}
.chat-panel {background:#ffffffb8;border-radius:23px;border:1px solid #e8eafa;min-height:310px;padding:22px 24px;margin-bottom:13px;box-shadow:0 8px 26px #5865a511}
.chat-row {display:flex;align-items:flex-start;gap:12px;margin:10px 0 18px}
.chat-row.user {justify-content:flex-end}
.chat-avatar {flex:0 0 37px;width:37px;height:37px;border-radius:50%;background:#e9ebff;display:flex;align-items:center;justify-content:center;font-size:22px}
.chat-avatar img {width:37px;height:37px;object-fit:cover;border-radius:50%}
.chat-bubble {max-width:77%;padding:13px 18px;border-radius:19px;background:#f2f3fb;color:#344061;font-size:15px;line-height:1.7;white-space:pre-wrap;overflow-wrap:anywhere}
.chat-row.user .chat-bubble {background:#dfe2ff;border-radius:19px 19px 5px 19px}
.chat-row.assistant .chat-bubble {border-radius:5px 19px 19px 19px}
.chat-caption {color:#98a0bb;font-size:11px;margin-top:6px}
.section-line {font-size:20px;font-weight:800;color:#273862;margin:8px 0 14px}
.notes-wrap {margin-top:12px}
.note {position:relative;border-radius:11px;min-height:230px;padding:25px 18px 14px;box-shadow:0 8px 18px #53639722;overflow-wrap:anywhere;margin:15px 0 4px;color:#30344c}
.pin {position:absolute;top:-1px;left:50%;transform:translateX(-50%);font-size:23px;filter:drop-shadow(1px 2px 1px #aaa)}
.note-title {font-size:17px;font-weight:800;margin:6px 0 12px}
.note-content {font-size:13px;line-height:1.85;white-space:pre-wrap;min-height:110px}
.note-date {font-size:11px;color:#7d7f9a;margin-top:9px}
.cal-cell {background:#fff;border:1px solid #e6e8f7;border-radius:8px;min-height:108px;padding:8px 8px;margin:3px 0;color:#36446b;box-shadow:0 2px 5px #6069a008}
.cal-cell.today {outline:2px solid #7b85f6;outline-offset:-2px}
.cal-day {font-size:13px;font-weight:800;margin-bottom:5px}
.cal-event {padding:4px 6px;margin-top:4px;border-radius:5px;background:#dfe4ff;color:#3c52a5;font-size:11px;line-height:1.4;overflow-wrap:anywhere}
.cal-event:nth-child(3n) {background:#e0f4ed;color:#3a8b72}
.cal-event:nth-child(4n) {background:#ffe8b7;color:#a57d2b}
.cal-weekday {text-align:center;font-size:13px;color:#6b769d;font-weight:800;padding:9px}
.small-muted {font-size:13px;color:#7c88a8}
.empty {padding:25px;background:#fff;border-radius:16px;color:#7380a5}


/* 단일 입력바: 마이크 + 텍스트 + 전송을 한 줄에 */
.st-key-unified_chat_bar {
  background:#fff!important; border:1px solid #e0e4fb!important;
  border-radius:24px!important; padding:8px 12px!important;
  box-shadow:0 8px 22px #6472a01a!important;
}
.st-key-unified_chat_bar [data-testid="stHorizontalBlock"] {align-items:center!important}
.st-key-unified_chat_bar [data-testid="stForm"] {
  padding:0!important; border:0!important; box-shadow:none!important;
  background:transparent!important;
}
.st-key-unified_chat_bar [data-testid="stForm"] [data-testid="stVerticalBlock"] {
  display:grid!important; grid-template-columns:minmax(0,1fr) auto!important;
  align-items:center!important; gap:8px!important;
}
.st-key-unified_chat_bar [data-testid="stForm"] [data-testid="stTextInput"] {grid-column:1}
.st-key-unified_chat_bar [data-testid="stForm"] [data-testid="stFormSubmitButton"] {grid-column:2}
.st-key-unified_chat_bar [data-testid="stForm"] input {
  border:0!important; box-shadow:none!important; background:transparent!important;
}
.st-key-unified_chat_bar [data-testid="stAudioInput"] {min-width:46px}
.st-key-unified_chat_bar [data-testid="stAudioInput"] button {border-radius:50%!important}
.st-key-unified_chat_bar [data-testid="stFormSubmitButton"] button {
  border-radius:50%!important; min-width:47px!important;
  width:47px!important; height:47px!important; font-size:22px!important;
  color:white!important; padding:0!important;
}

/* 파스텔 모던 + 미니멀 화이트 디자인 */
.stApp {background:#f5f7ff!important}
[data-testid="stSidebar"] {background:linear-gradient(160deg,#e4e9ff,#f3f4ff 75%,#e9e6ff)!important}
[data-testid="stMainBlockContainer"] {max-width:1510px!important;padding:22px 32px 60px!important}
.hero {background:#fff!important;border-radius:24px!important;box-shadow:0 7px 25px #6472b012!important}
.stButton>button {transition:transform .15s ease,box-shadow .15s ease}
.stButton>button:hover {transform:translateY(-1px);box-shadow:0 5px 15px #7778bc19}
.note {border-radius:15px!important;min-height:190px!important;margin:4px 0!important;box-shadow:0 8px 18px #6070a019!important}
.note-content {min-height:90px!important}
.note-title {font-size:16px!important}
.st-key-note_search input {border-radius:30px!important}
.st-key-unified_chat_bar {border-radius:30px!important}
.st-key-unified_chat_bar [data-testid="stHorizontalBlock"] {align-items:center!important}
.st-key-unified_chat_bar [data-testid="stForm"] [data-testid="stVerticalBlock"] {display:grid!important;grid-template-columns:minmax(0,1fr) 50px!important}
.st-key-unified_chat_bar [data-testid="stFormSubmitButton"] button {background:linear-gradient(135deg,#8a82f5,#666ce7)!important}
.st-key-unified_chat_bar [data-testid="stFileUploader"] {max-width:300px}
.st-key-unified_chat_bar [data-testid="stAudioInput"] {min-width:42px}
[data-testid="stMetric"] {background:#fff;border-radius:18px;padding:12px;border:1px solid #e7eaff}

/* 입력바 세부 조정: 마이크·+ 아이콘만 노출, 좌측 여백 최소화 */
.st-key-unified_chat_bar {
  padding:4px 6px!important;
  border-radius:28px!important;
}
.st-key-unified_chat_bar [data-testid="stHorizontalBlock"] {
  gap:0.15rem!important;
  align-items:center!important;
}
.st-key-unified_chat_bar [data-testid="column"] {
  min-width:0!important;
  padding:0!important;
}
.st-key-unified_chat_bar [data-testid="stAudioInput"] {
  min-width:0!important;
  width:100%!important;
}
.st-key-unified_chat_bar [data-testid="stAudioInput"] button {
  border:0!important;
  box-shadow:none!important;
  background:#e9edff!important;
  color:#6c75e5!important;
  min-width:42px!important;
  width:42px!important;
  min-height:42px!important;
  height:42px!important;
  padding:8px!important;
  border-radius:50%!important;
}
.st-key-unified_chat_bar [data-testid="stAudioInput"] button:hover {
  background:#dfe4ff!important;
}
.st-key-unified_chat_bar [data-testid="stAudioInput"] button svg {
  color:#6c75e5!important;
  width:22px!important;
  height:22px!important;
}
.st-key-unified_chat_bar [data-testid="stPopover"] button {
  border:0!important;
  box-shadow:none!important;
  background:transparent!important;
  color:#6c75e5!important;
  font-size:27px!important;
  font-weight:400!important;
  padding:0!important;
  min-width:36px!important;
  width:36px!important;
  min-height:42px!important;
  height:42px!important;
  border-radius:50%!important;
}
.st-key-unified_chat_bar [data-testid="stPopover"] button:hover {
  background:#f0f1ff!important;
}
.st-key-unified_chat_bar [data-testid="stForm"] {
  margin:0!important;
  padding:0!important;
}
.st-key-unified_chat_bar [data-testid="stForm"] [data-testid="stVerticalBlock"] {
  gap:0!important;
  grid-template-columns:minmax(0,1fr) 47px!important;
}
.st-key-unified_chat_bar [data-testid="stTextInput"] input {
  padding-left:4px!important;
  border:0!important;
  box-shadow:none!important;
}
/* 녹음 시간 표시는 Streamlit 기본 audio_input의 녹음 UI에서 제공 */


 /* 메모 카드 안쪽 하단의 아이콘 버튼 */
 [class*="st-key-memo_card_"] .note {padding-bottom:58px!important;min-height:245px!important;margin-bottom:0!important}
 [class*="st-key-memo_card_"] [data-testid="stHorizontalBlock"] {
   position:relative!important;
   margin-top:-51px!important;
   padding:0 13px 12px!important;
   gap:6px!important;
   z-index:2;
 }
 [class*="st-key-memo_card_"] [data-testid="stButton"] button {
   background:#ffffffa6!important;
   color:#5d638b!important;
   border:1px solid #ffffff9c!important;
   box-shadow:none!important;
   height:34px!important;
   min-height:34px!important;
   padding:2px!important;
   border-radius:9px!important;
   font-size:16px!important;
 }
 [class*="st-key-memo_card_"] [data-testid="stButton"] button:hover {background:#fff!important}


/* 첨부 이미지와 유사한 파스텔 화이트 대시보드 */
.stApp {background:linear-gradient(115deg,#eef1ff 0%,#f7f8ff 50%,#e8edff 100%)!important}
[data-testid="stSidebar"] {background:linear-gradient(165deg,#dce2ff 0%,#edf0ff 62%,#d8deff 100%)!important}
[data-testid="stMainBlockContainer"] {max-width:1580px!important;padding:84px 24px 60px!important}
[data-testid="stSidebar"] .stButton>button[kind="primary"] {
 background:linear-gradient(110deg,#7775ed,#8e83f4)!important;
 color:#fff!important;box-shadow:0 8px 22px #7777d444!important
}
.side-robot {background:transparent!important;padding:0!important;margin-top:12px!important}
.side-robot img {max-height:175px;object-fit:contain}
.side-logo {font-size:24px!important;text-align:center}
.side-desc {text-align:center}
.hero {padding:18px 22px!important;margin-bottom:12px!important;min-height:96px}
.hero-title {font-size:23px!important}
.hero-desc {margin-left:55px!important}
.chat-panel {min-height:435px!important;border-radius:22px!important;background:#fff!important;padding:24px 18px!important;margin-bottom:9px!important}
.chat-row {margin:8px 0 20px!important}
.chat-bubble {font-size:13px!important;padding:13px 16px!important}
[data-testid="stVerticalBlockBorderWrapper"] {border-color:#e8eafa}
.dashboard-panel-title {font-size:19px;font-weight:800;color:#22335e;background:#fff;border-radius:20px 20px 0 0;padding:16px 18px 12px;margin-bottom:0}
.dashboard-subtitle {font-size:12px;font-weight:700;color:#34436e;margin:8px 0 5px}
.dashboard-month {font-size:17px;font-weight:800;color:#24335e;text-align:center;margin:12px 0 8px}
.mini-calendar {display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:4px;background:#fff;border:1px solid #eef0fa;border-radius:15px;padding:10px 5px}
.mini-weekday {text-align:center;font-size:11px;color:#8893b7;padding:7px 0}
.mini-day {position:relative;text-align:center;font-size:12px;color:#34446d;min-height:31px;padding:7px 0;border-radius:50%}
.mini-today {background:#807cf0;color:#fff!important;box-shadow:0 4px 10px #807cf044}
.mini-selected:not(.mini-today) {background:#eef0ff}
.mini-dot {position:absolute;bottom:1px;left:calc(50% - 2px);height:4px;width:4px;border-radius:50%;background:#4fb88c}
.dashboard-task-head {font-weight:800;font-size:14px;color:#27365c;margin:20px 0 12px}
.dashboard-task-row {display:flex;justify-content:space-between;gap:6px;font-size:12px;color:#334365;background:#fff;border-radius:10px;padding:8px 5px}
.dashboard-task-row span {color:#8791ad;white-space:nowrap}
.dashboard-empty {padding:18px 10px;color:#8a93ad;font-size:12px;background:#fff;border-radius:12px}
.dashboard-bottom-space {height:16px}
.dashboard-sticky {height:172px;border-radius:12px;box-shadow:0 5px 12px #6c79a122;padding:17px 10px 9px;position:relative;display:flex;flex-direction:column;overflow:hidden}
.sticky-pin {position:absolute;top:0;left:45%;font-size:15px}
.dashboard-sticky strong {font-size:12px;color:#293451;margin-top:9px}
.dashboard-sticky p {font-size:11px;white-space:pre-wrap;color:#4e5870;line-height:1.6;flex:1;overflow:hidden}
.dashboard-sticky small {font-size:9px;color:#818aa0}
[class*="st-key-memo_card_"] .note {border-radius:12px!important}

</style>''',unsafe_allow_html=True)

for key, default in {'pages':'대화하기','messages':[], 'month':date.today().replace(day=1), 'editing_memo':None,'viewing_memo':None}.items():
    if key not in st.session_state: st.session_state[key] = default

with st.sidebar:
    if ROBOT: st.markdown(f'<div class="side-robot"><img src="{ROBOT}" alt="AI 로봇"></div>',unsafe_allow_html=True)
    st.markdown('<div class="side-logo">AI 음성 비서</div><div class="side-desc">나만의 스마트 메모 · 일정 도우미</div>',unsafe_allow_html=True)
    for name,icon in [('대화하기','💬'),('메모 관리','📝'),('할 일 관리','🗓️'),('처리 이력','📊')]:
        if st.button(f'{icon}  {name}',key='nav_'+name,type='primary' if st.session_state.page==name else 'secondary',use_container_width=True):
            st.session_state.page=name
            st.rerun()
    st.markdown('---')
    st.markdown('<div class="side-footer">음성이나 텍스트로 입력한 메모와 할 일을<br>자동으로 정리해 드려요.</div>',unsafe_allow_html=True)


def hero(icon,title,description):
    st.markdown(f'<div class="hero"><div class="hero-title"><span class="hero-icon">{icon}</span>{title}</div><div class="hero-desc">{description}</div></div>',unsafe_allow_html=True)


def add_message(text, input_type='문자', stt_ms=0.0, started_at=None):
    start = started_at if started_at is not None else perf_counter()
    st.session_state.messages.append({'role':'user','text':text})
    intent = 'UNKNOWN'
    intent_ms = db_ms = 0.0
    success = False
    error_type = ''
    try:
        reply, intent, success, intent_ms, db_ms = process_message(text)
    except Exception as exc:
        error_type = type(exc).__name__
        reply = f'처리 중 오류가 발생했어요: {exc}'
    total_ms = (perf_counter()-start)*1000
    st.session_state.messages.append({'role':'assistant','text':reply})
    try:
        log_history(input_type, text, intent, success, stt_ms, intent_ms, db_ms, total_ms, reply, error_type)
    except Exception as exc:
        st.error(f'처리 이력 저장 실패: {exc}')
    st.session_state.page='대화하기'


def log_voice_failure(message, stt_ms, started_at, error_type):
    log_history('음성', '(인식 실패)', 'UNKNOWN', False, stt_ms, 0, 0,
                (perf_counter()-started_at)*1000, message, error_type)

page=st.session_state.page
if page=='대화하기':
    hero('💬','대화하기','음성이나 텍스트로 메모나 할 일을 말해보세요.')
    chat='<div class="chat-panel">'
    messages=st.session_state.messages[-24:]
    if not messages:
        messages=[{'role':'assistant','text':'안녕하세요! 👋\n“오늘 회의 내용 메모해 줘” 또는 “내일 오후 3시 병원 예약 추가해 줘”라고 입력해 보세요.'}]
    for m in messages:
        role=m['role']
        avatar='<img src="'+ROBOT+'" alt="AI">' if role=='assistant' and ROBOT else ('🤖' if role=='assistant' else '👤')
        bubble=f'<div class="chat-bubble">{esc(m["text"])}</div>'
        chat+=f'<div class="chat-row {role}">'+(f'<div class="chat-avatar">{avatar}</div>'+bubble if role=='assistant' else bubble+f'<div class="chat-avatar">{avatar}</div>')+'</div>'
    chat+='</div>'
    st.markdown(chat,unsafe_allow_html=True)
    # 단일 커스텀 입력바: 브라우저 MediaRecorder + 숨겨진 파일 선택 + 문자 전송
    # 이벤트 ID로 Streamlit rerun 시 중복 처리 방지
    chat_event = chat_bar_component(key='voice_chat_bar', default=None)
    selected_audio = None
    selected_kind = '음성 녹음'
    if isinstance(chat_event, dict):
        event_id = chat_event.get('id')
        if event_id and event_id != st.session_state.get('last_chat_event_id'):
            st.session_state.last_chat_event_id = event_id
            if chat_event.get('type') == 'text':
                content = str(chat_event.get('text') or '').strip()
                if content:
                    add_message(content)
                    st.rerun()
            elif chat_event.get('type') == 'audio':
                encoded = chat_event.get('data') or ''
                if isinstance(encoded, str) and len(encoded) <= 15_000_000:
                    try:
                        raw_audio = base64.b64decode(encoded, validate=True)
                        if raw_audio:
                            selected_audio = raw_audio
                            selected_kind = '음성 파일' if chat_event.get('source') == 'file' else '음성 녹음'
                            audio_ext = str(chat_event.get('extension') or 'webm').lower()
                            if audio_ext not in ('wav','mp3','m4a','webm','ogg','mp4'):
                                audio_ext = 'webm'
                    except (ValueError, base64.binascii.Error):
                        st.error('음성 파일을 읽을 수 없어요.')
                else:
                    st.warning('음성 파일은 10MB 이하만 업로드할 수 있어요.')

    # 음성 녹음/파일 → STT → 명령 처리 → 이력 저장

    if selected_audio is not None:
        audio_bytes = selected_audio
        audio_key = hashlib.sha256(audio_bytes).hexdigest()
        if st.session_state.get('last_audio_hash') != audio_key:
            if len(audio_bytes) > 10 * 1024 * 1024:
                st.warning('음성 파일은 10MB 이하만 업로드할 수 있어요.')
            else:
                st.session_state.last_audio_hash = audio_key
                started = perf_counter()
                stt_ms = 0.0
                try:
                    from src.stt import transcript_wav

                    with st.spinner('음성 인식 중...'):
                        recognized, elapsed = transcript_wav(
                            audio_bytes,
                            model_size='base'
                        )
                        stt_ms = elapsed * 1000
                except ImportError as exc:
                    st.error(f"음성 인식 모듈 불러오기 실패: {exc}")
                    st.exception(exc)
                    log_voice_failure(
                            str(exc), stt_ms, started, "ImportError"
                        )

                except Exception as exc:
                    st.error(f"음성 인식 오류: {exc}")
                    st.exception(exc)
                    log_voice_failure(
                        str(exc), stt_ms, started, type(exc).__name__
                    )

elif page=='메모 관리':
    hero('📝','메모 관리','저장한 메모를 한눈에 확인하세요. 음성으로 만든 메모도 여기서 관리할 수 있어요.')
    top_left, top_right = st.columns([5, 1.6], vertical_alignment='center')
    with top_left:
        search = st.text_input('메모 검색', placeholder='🔎 메모 제목이나 내용 검색...', label_visibility='collapsed', key='note_search')
    with top_right:
        with st.popover('＋ 메모 추가', use_container_width=True):
            with st.form('new_note', clear_on_submit=True):
                st.markdown('#### 새 메모 작성')
                title = st.text_input('메모 제목')
                content = st.text_area('메모 내용', height=160)
                color_name = st.selectbox('포스트잇 색상', ['노랑','분홍','하늘','연두','보라'])
                if st.form_submit_button('메모 저장', type='primary', use_container_width=True):
                    if not content.strip():
                        st.warning('메모 내용을 입력해 주세요.')
                    else:
                        execute('INSERT INTO memos(title,content,color,created_at) VALUES(?,?,?,?)',
                                (title.strip() or content.strip()[:20], content.strip(), COLORS[['노랑','분홍','하늘','연두','보라'].index(color_name)], datetime.now().isoformat(timespec='seconds')))
                        st.rerun()
    notes = query('SELECT * FROM memos WHERE title LIKE ? OR content LIKE ? ORDER BY id DESC',
                  (f'%{search}%', f'%{search}%'))
    st.caption(f'메모 {len(notes)}개' + (f' · 검색어: {search}' if search else ''))
    if not notes:
        st.info('검색된 메모가 없어요. 오른쪽 위의 ＋ 메모 추가를 눌러 작성해 보세요.')
    else:
        cols = st.columns(4, gap='medium')
        for i, n in enumerate(notes):
            with cols[i % 4]:
                snippet = n['content'][:110] + ('…' if len(n['content']) > 110 else '')
                with st.container(key=f'memo_card_{n["id"]}'):
                    st.markdown(f'<div class="note" style="background:{esc(n["color"])}"><div class="pin">📌</div><div class="note-title">{esc(n["title"])}</div><div class="note-content">{esc(snippet)}</div><div class="note-date">{esc(n["created_at"][:10])}</div></div>', unsafe_allow_html=True)
                    a,b,c = st.columns(3,gap='small')
                    if a.button('🔍', key=f'view_{n["id"]}', help='메모 보기', use_container_width=True):
                        st.session_state.viewing_memo=n['id']; st.session_state.editing_memo=None; st.rerun()
                    if b.button('✏️', key=f'edit_{n["id"]}', help='메모 수정', use_container_width=True):
                        st.session_state.editing_memo=n['id']; st.session_state.viewing_memo=None; st.rerun()
                    if c.button('🗑️', key=f'del_{n["id"]}', help='메모 삭제', use_container_width=True):
                        st.session_state[f'confirm_delete_{n["id"]}']=True; st.rerun()
                if st.session_state.get(f'confirm_delete_{n["id"]}',False):
                    st.warning('정말 삭제할까요?')
                    yes,no=st.columns(2)
                    if yes.button('삭제 확인',key=f'yes_{n["id"]}'):
                        execute('DELETE FROM memos WHERE id=?',(n['id'],))
                        st.session_state.pop(f'confirm_delete_{n["id"]}',None)
                        st.rerun()
                    if no.button('취소',key=f'no_{n["id"]}'):
                        st.session_state.pop(f'confirm_delete_{n["id"]}',None)
                        st.rerun()
    if st.session_state.viewing_memo is not None:
        found=query('SELECT * FROM memos WHERE id=?',(st.session_state.viewing_memo,))
        if found:
            n=found[0]
            with st.container(border=True):
                st.markdown(f'### 📌 {esc(n["title"])}')
                st.caption(n['created_at'].replace('T',' '))
                st.text(n['content'])
                if st.button('닫기',key='close_note'):
                    st.session_state.viewing_memo=None; st.rerun()
    if st.session_state.editing_memo is not None:
        found=query('SELECT * FROM memos WHERE id=?',(st.session_state.editing_memo,))
        if found:
            n=found[0]
            with st.container(border=True):
                st.markdown('### ✏️ 메모 수정')
                with st.form('edit_form'):
                    nt=st.text_input('제목',value=n['title'])
                    nc=st.text_area('내용',value=n['content'],height=170)
                    color_names=['노랑','분홍','하늘','연두','보라']
                    index=COLORS.index(n['color']) if n['color'] in COLORS else 0
                    color_name=st.selectbox('색상',color_names,index=index)
                    if st.form_submit_button('수정 저장',type='primary'):
                        if nt.strip() and nc.strip():
                            execute('UPDATE memos SET title=?,content=?,color=? WHERE id=?',
                                    (nt.strip(),nc.strip(),COLORS[color_names.index(color_name)],n['id']))
                            st.session_state.editing_memo=None; st.rerun()
                        else:
                            st.warning('제목과 내용을 입력해 주세요.')
                if st.button('수정 취소',key='cancel_edit'):
                    st.session_state.editing_memo=None; st.rerun()

elif page=='할 일 관리':
    hero('🗓️','할 일 관리','일정을 추가하고 날짜별로 한눈에 관리할 수 있어요.')
    # 연/월 제목을 누르는 대신 날짜 선택 달력으로 원하는 달에 즉시 이동
    selected_date = st.date_input(
        '📅 날짜 선택',
        value=st.session_state.month,
        key='calendar_month_picker',
        help='날짜를 클릭하면 달력이 열립니다. 연도와 월을 선택해 이동할 수 있어요.',
    )
    month = selected_date.replace(day=1)
    st.session_state.month = month
    st.markdown(
        f'<h3 style="text-align:center;color:#26365d;margin:6px 0 16px">{month.year}년 {month.month}월</h3>',
        unsafe_allow_html=True,
    )
    with st.expander('✚ 새 일정 추가'):
        with st.form('new_task',clear_on_submit=True):
            title=st.text_input('할 일 제목');d=st.date_input('날짜',value=date.today());t=st.time_input('시간 (선택)',value=None)
            if st.form_submit_button('일정 저장',type='primary'):
                if not title.strip():st.warning('제목을 입력하세요.')
                else:
                    execute('INSERT INTO tasks(title,due_date,due_time,created_at) VALUES(?,?,?,?)',(title.strip(),d.isoformat(),t.strftime('%H:%M') if t else '',datetime.now().isoformat(timespec='seconds')))
                    st.session_state.month=d.replace(day=1);st.session_state.pop('calendar_month_picker', None);st.rerun()
    calendar_col, monthly_col = st.columns([2.1, 1], gap='large', vertical_alignment='top')
    with calendar_col:
        tasks=query('SELECT * FROM tasks WHERE substr(due_date,1,7)=? ORDER BY due_date,due_time,id',(month.strftime('%Y-%m'),))
        byday={}
        for task in tasks:byday.setdefault(int(task['due_date'][-2:]),[]).append(task)
        for col,name in zip(st.columns(7,gap='small'),['일','월','화','수','목','금','토']):col.markdown(f'<div class="cal-weekday">{name}</div>',unsafe_allow_html=True)
        for week in calendar.Calendar(firstweekday=6).monthdayscalendar(month.year,month.month):
            for col,day in zip(st.columns(7,gap='small'),week):
                items=byday.get(day,[]) if day else []
                inner=f'<div class="cal-day">{day if day else "&nbsp;"}</div>'
                for task in items[:3]:
                    label=('✓ ' if task['completed'] else '')+esc(task['title'])
                    inner+=f'<div class="cal-event">{label}<br><small>{esc(task["due_time"] or "")}</small></div>'
                if len(items)>3:inner+=f'<small>+{len(items)-3}개 더</small>'
                is_today=bool(day and date.today()==date(month.year,month.month,day))
                col.markdown(f'<div class="cal-cell {"today" if is_today else ""}">{inner}</div>',unsafe_allow_html=True)
    with monthly_col:
        st.markdown('<div class="section-line" style="margin-top:4px">이번 달 할 일</div>',unsafe_allow_html=True)
        if not tasks:st.info('등록된 일정이 없어요. 대화하기에서 "내일 오후 3시 병원 예약 추가해 줘"라고 입력해 보세요.')
        for task in tasks:
            a,b,c=st.columns([1,8,1],vertical_alignment='center')
            checked=a.checkbox('완료',value=bool(task['completed']),key=f'task_{task["id"]}',label_visibility='collapsed')
            if checked!=bool(task['completed']):execute('UPDATE tasks SET completed=? WHERE id=?',(int(checked),task['id']));st.rerun()
            b.markdown(f'{"~~" if checked else ""}**{esc(task["title"])}** · {esc(task["due_date"])} {esc(task["due_time"] or "")}{"~~" if checked else ""}')
            if c.button('삭제',key=f'del_task_{task["id"]}'):execute('DELETE FROM tasks WHERE id=?',(task['id'],));st.rerun()
elif page=='처리 이력':
    hero('📊','처리 이력','입력 시간, 인식 문장, 의도, 성공 여부, 단계별 처리 시간을 확인할 수 있어요.')
    history = query('SELECT * FROM processing_history ORDER BY id DESC LIMIT 500')
    if not history:
        st.info('아직 처리 이력이 없어요. 대화하기에서 문자 또는 음성으로 명령을 입력해 보세요.')
    else:
        import pandas as pd
        rows = []
        for h in history:
            rows.append({
                '시간': h['created_at'].replace('T', ' '),
                '입력': h['input_type'],
                '인식 문장': h['transcript'],
                '의도': h['intent'],
                '성공/실패': h['status'],
                'STT(ms)': round(h['stt_ms'], 1),
                '의도 분류(ms)': round(h['intent_ms'], 1),
                'DB(ms)': round(h['db_ms'], 1),
                '전체(ms)': round(h['total_ms'], 1),
                '오류 유형': h['error_type'],
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True,
                     column_config={'인식 문장': st.column_config.TextColumn(width='large')})
        st.download_button('CSV 다운로드', data=df.to_csv(index=False).encode('utf-8-sig'),
                           file_name='processing_history.csv', mime='text/csv')
        with st.expander('최근 처리 응답 자세히 보기'):
            for h in history[:15]:
                st.markdown(f'**{h["created_at"].replace("T", " ")} | {h["intent"]} | {h["status"]}**')
                st.write(h['response_text'])
                st.divider()
