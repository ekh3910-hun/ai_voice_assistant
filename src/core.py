# core.py

import time
import sys
from functools import lru_cache
from pathlib import Path
from datetime import date, timedelta
import re
import joblib
BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src.db import *
MODEL_PATH = BASE / 'models' / 'intent_model.pkl'


@lru_cache(maxsize=1)
def load_intent_model():
    """학습되어 저장된 intent_model.pkl 로드 (없을 경우 1회 자동 학습)"""
    if not MODEL_PATH.exists():
        from models.model import train_model
        return train_model()
    model = joblib.load(MODEL_PATH)
    clf = model.named_steps.get('model', model)
    model_name = clf.__class__.__name__
    return model, model_name


WEEKDAYS = {'월': 0, '화': 1, '수': 2, '목': 3, '금': 4, '토': 5, '일': 6}
DATE_PATTERN = re.compile(
    r'\d{4}[-./]\d{1,2}[-./]\d{1,2}|'
    r'\d{4}년\s*\d{1,2}월\s*\d{1,2}일|'
    r'(?:다음\s*달|이번\s*달)\s*\d{1,2}일|'
    r'\d{1,2}월\s*\d{1,2}일|'
    r'(?:이번\s*주|다음\s*주|지난\s*주)\s*[월화수목금토일]요일|'
    r'(?<![\d월])\d{1,2}일(?!\s*(?:간|동안|마다|째|연속|치))|'   # [수정] '29일', '30일' 처럼 일(日)만 말한 경우
    r'오늘|내일|모레|어제|그저께|방금'
)
# [수정] 메모 내용 추출에는 기존 날짜 패턴만 사용 ('1일 1커밋 하기 메모해' 같은 내용이 잘리지 않도록)
DATE_PATTERN_LEGACY = re.compile(
    r'\d{4}[-./]\d{1,2}[-./]\d{1,2}|'
    r'\d{4}년\s*\d{1,2}월\s*\d{1,2}일|'
    r'(?:다음\s*달|이번\s*달)\s*\d{1,2}일|'
    r'\d{1,2}월\s*\d{1,2}일|'
    r'(?:이번\s*주|다음\s*주|지난\s*주)\s*[월화수목금토일]요일|'
    r'오늘|내일|모레|어제|그저께|방금'
)

# [수정] STT가 날짜를 한글 숫자로 전사한 경우("이십구일") 아라비아 숫자로 정규화
_KO_DIGIT = {'일': 1, '이': 2, '삼': 3, '사': 4, '오': 5, '육': 6, '칠': 7, '팔': 8, '구': 9}
_KO_DAY_PATTERN = re.compile(r'(?<![가-힣])([이삼]?십[일이삼사오육칠팔구]?)(일|월)')


def _ko_num(s):
    tens, _, ones = s.partition('십')
    return (_KO_DIGIT.get(tens, 1) if tens else 1) * 10 + (_KO_DIGIT.get(ones, 0) if ones else 0)


def normalize_text(text):
    """'이십구일' → '29일', '십일월 삼일' 같은 표현은 그대로 둠(10 이상만 변환해 오인식 방지)"""
    return _KO_DAY_PATTERN.sub(lambda m: f'{_ko_num(m.group(1))}{m.group(2)}', text)


def convert_date(date_text, today=None):
    today = today or date.today()
    s = re.sub(r'\s+', '', date_text)
    offsets = {'그저께': -2, '어제': -1, '오늘': 0,
               '방금': 0, '내일': 1, '모레': 2}
    if s in offsets:
        return (today + timedelta(days=offsets[s])).isoformat()

    m = re.fullmatch(r'(이번주|다음주|지난주)([월화수목금토일])요일', s)
    if m:
        week_offset = {'지난주': -1, '이번주': 0, '다음주': 1}[m.group(1)]
        monday = today - timedelta(days=today.weekday())
        return (monday + timedelta(weeks=week_offset, days=WEEKDAYS[m.group(2)])).isoformat()

    m = re.fullmatch(r'(이번달|다음달)(\d{1,2})일', s)
    if m:
        year, month = today.year, today.month
        if m.group(1) == '다음달':
            month += 1
            if month == 13:
                year, month = year + 1, 1
        try:
            return date(year, month, int(m.group(2))).isoformat()
        except ValueError:
            return None

    m = re.fullmatch(r'(\d{4})년(\d{1,2})월(\d{1,2})일', s)
    if not m:
        m = re.fullmatch(r'(\d{4})[-./](\d{1,2})[-./](\d{1,2})', s)
    if m:
        try:
            return date(*map(int, m.groups())).isoformat()
        except ValueError:
            return None

    m = re.fullmatch(r'(\d{1,2})월(\d{1,2})일', s)
    if m:
        month, day = map(int, m.groups())
        try:
            result = date(today.year, month, day)
            if result < today:
                result = date(today.year + 1, month, day)
            return result.isoformat()
        except ValueError:
            return None

    # [수정] 'N일'만 말한 경우: 이번 달 N일, 이미 지났으면 다음 달 N일
    m = re.fullmatch(r'(\d{1,2})일', s)
    if m:
        day = int(m.group(1))
        year, month = today.year, today.month
        for _ in range(12):  # 해당 일이 없는 달(예: 11월 31일)은 건너뜀
            try:
                result = date(year, month, day)
                if result >= today:
                    return result.isoformat()
            except ValueError:
                pass
            month += 1
            if month == 13:
                year, month = year + 1, 1
        return None
    return None


def clean_title(value):
    value = value.strip(' ,，.!?')
    value = re.sub(r'^(?:할\s*일(?:에|에서|중에|로|을|이)?|메모(?:에|장에)?)\s*', '', value)
    value = re.sub(r'^(?:에|를|을|은|는)\s+', '', value)
    value = re.sub(r'\s*(?:을|를|은|는|이|가|에|로|으로|까지)$', '', value)
    value = re.sub(r'\s*(?:추가\s*(?:해줘|해달라|해주세요|해라|해|해주|하기|할래|좀)?|등록\s*(?:해줘|해달라|해주세요|해라|해|해주|하기|할래|좀)?|넣어\s*(?:줘|주세요|달라|라)?|추가|등록).*$', '', value)
    value = re.sub(r'^(?:추가|등록|할\s*일)\s*', '', value)
    return value.strip(' ,，.!?')


def _strip_quote_josa(s):
    """[수정] 인용 조사 '이라고'의 '이' 제거: '있음이' → '있음' (앞 글자에 받침이 있을 때만, '아이'는 유지)"""
    if len(s) >= 2 and s.endswith('이'):
        prev = s[-2]
        if '가' <= prev <= '힣' and (ord(prev) - 0xAC00) % 28 != 0:
            return s[:-1]
        if prev.isdigit() or prev.isascii() and prev.isalpha():
            return s[:-1]
    return s


def extract_slots(text, intent):
    slots = {}
    working = text.strip()
    date_pattern = DATE_PATTERN_LEGACY if intent == 'ADD_MEMO' else DATE_PATTERN  # [수정]
    dates = list(date_pattern.finditer(text))

    if intent in {'ADD_TASK', 'LIST_TASK', 'UPDATE_TASK', 'COMPLETE_TASK', 'DELETE_TASK'}:
        if intent == 'UPDATE_TASK':
            if dates:
                # [수정] 날짜 뒤에 '로/으로'가 붙으면 새 마감일, 아니면 대상 할 일을 찾는 날짜
                #   예) '30일 장보기 이름을 마트 가기로 바꿔' → DUE_DATE=30일 (이름만 변경)
                #       '오늘 회의 날짜를 내일로 변경해'     → DUE_DATE=오늘, NEW_DUE_DATE=내일
                is_rename = bool(re.search(r'(?:이름|제목)\s*(?:을|를)', text))
                for d in dates:
                    followed_by_ro = re.match(r'\s*(?:으로|로)', text[d.end():])
                    key = 'NEW_DUE_DATE' if followed_by_ro else 'DUE_DATE'
                    slots.setdefault(key, convert_date(d.group()))
                if len(dates) == 1 and not is_rename and 'NEW_DUE_DATE' not in slots:
                    slots['NEW_DUE_DATE'] = slots.pop('DUE_DATE')   # '회의 내일 변경해' 등 기존 동작 유지
        elif dates:
            slots['DUE_DATE'] = convert_date(dates[0].group())
    elif intent in {'LIST_MEMO', 'DELETE_MEMO'} and dates:
        if not (intent == 'DELETE_MEMO' and re.search(r'방금', text)):
            slots['MEMO_DATE'] = convert_date(dates[0].group())

    # [수정] '모두/전부' → 일치하는 항목 일괄 처리, '최근/마지막/방금' → 가장 최근 메모
    #        (완료·삭제 명령에서만 적용 → 추가 명령의 '가족 모두 모임' 같은 내용은 그대로 유지)
    if intent in {'COMPLETE_TASK', 'DELETE_TASK', 'DELETE_MEMO'} and re.search(r'모두|전부|싹\s*다', text):
        slots['ALL'] = True
        working = re.sub(r'\s*(?:모두|전부|싹\s*다)\s*', ' ', working)
    if intent == 'DELETE_MEMO' and re.search(r'최근|마지막|방금', text):
        slots['TARGET'] = 'LATEST'

    if intent != 'ADD_MEMO':
        working = date_pattern.sub(' ', working)
    working = re.sub(r'\s+', ' ', working).strip()
    working = re.sub(r'^(?:까지|에|부터)\s+', '', working)

    if intent == 'HELP':
        return {}

    if intent == 'LIST_TASK':
        m = re.search(r'안\s*끝낸|미완료|완료한|완료된', working)
        if m:
            slots['STATUS'] = re.sub(r'안\s*끝낸', '안 끝낸', m.group())
        return {k: v for k, v in slots.items() if v is not None}

    if intent in {'LIST_MEMO', 'DELETE_MEMO'}:
        m = re.search(r'^(.+?)\s*(?:관련|들어간)?\s*메모', working)
        if m:
            keyword = clean_title(m.group(1))
            keyword = re.sub(r'^(?:최근|마지막|방금)\s*(?:에)?\s*(?:쓴|적은|작성한|저장한)?\s*', '', keyword).strip()
            # [수정] '회의라고 적어 둔/쓴 메모' → 키워드 '회의'
            keyword = _strip_quote_josa(re.sub(r'\s*라고\s*(?:적어\s*둔|적어\s*놓은|적은|쓴|써\s*둔|메모한|저장한)$', '', keyword).strip())
            if keyword and keyword not in {'쓴', '적어 둔', '작성한', '방금', '최근', '마지막', '전체'}:
                slots['KEYWORD'] = keyword
        return {k: v for k, v in slots.items() if v is not None}

    if intent == 'ADD_MEMO':
        # [수정] '~라고 메모해' 형식: 따옴표처럼 말한 내용은 날짜 단어까지 그대로 저장
        #        '준비라고'(받침 없음)·'있음이라고'(받침 있음) 모두 처리, '메모 해줘'처럼 띄어 써도 인식
        m = re.search(r'^(.+?)\s*라고\s*(?:메모|적어|써|남겨)', text.strip())
        if m:
            content = _strip_quote_josa(m.group(1).strip(' ,，.!?"\''))
            if content:
                slots['MEMO_CONTENT'] = content
                return slots

        memo_actions = r'(?:추가\s*(?:해줘|해주세요|해달라|해라|해|해주|하기|할래|좀)?|등록\s*(?:해줘|해주세요|해달라|해라|해|해주|하기|할래|좀)?|남겨\s*(?:줘|주세요|달라|라|둬|놓아)?|적어\s*(?:줘|주세요|달라|라|둬|놓아)?|써\s*(?:줘|주세요|달라|라|둬|놓아)?|해\s*(?:줘|주세요|둬|놓아|라)?|추가|등록)'
        patterns = [
            # 1) 메모 추가: 회의 준비 / 메모 추가 회의준비 / 메모 등록 회의준비
            rf'^(?:메모\s*추가|메모\s*등록)\s*[,，:]?\s*(.+)$',
            # 2) 메모에 회의준비 추가해줘 / 메모장에 회의준비 적어줘
            rf'^(?:메모장|메모)(?:에|로)?\s*(.+?)\s*{memo_actions}$',
            # 3) 회의준비 메모 추가해줘 / 회의준비 메모 등록
            rf'^(.+?)\s*(?:메모장|메모)(?:에|로|를|을)?\s*{memo_actions}$',
            # 4) 회의준비 메모해 / 우유 사기 메모
            rf'^(.+?)\s*[,，]?\s*메모\s*(?:좀\s*)?해',
            rf'^(.+?)\s*메모(?:해|로\s*남겨|에\s*적어)',
            # 5) "회의준비 메모" 처럼 입력된 경우
            rf'^(.+?)\s*메모$',
            # 6) 메모 회의준비
            rf'^메모\s+(.+)$',
        ]
        for pattern in patterns:
            m = re.search(pattern, working)
            if m:
                extracted = m.group(1).strip(' ,，.!?')
                if extracted:
                    slots['MEMO_CONTENT'] = extracted
                    break
        return slots

    if intent == 'UPDATE_TASK':
        m = re.search(r'^(.+?)\s*(?:이름을|제목을|을|를)\s*(.+?)(?:으로|로)\s*(?:바꿔|수정해|변경해|고쳐)', working)
        if m and 'NEW_DUE_DATE' not in slots:
            slots['TITLE'] = clean_title(m.group(1))
            slots['NEW_TITLE'] = clean_title(m.group(2))
        else:
            m = re.match(r'^(.+?)\s*(?:마감을|날짜를|기한을|일정을|로\s*미뤄|옮겨|바꿔|수정해)', working)
            if m:
                slots['TITLE'] = clean_title(m.group(1))
            elif 'NEW_DUE_DATE' in slots:
                title = re.sub(r'\s*(?:마감을|날짜를|기한을|일정을|로|으로|바꿔|옮겨|미뤄|수정해|변경해).*$', '', working)
                if clean_title(title):
                    slots['TITLE'] = clean_title(title)
        return {k: v for k, v in slots.items() if v is not None}

    if intent == 'ADD_TASK':
        action_verbs = r'(?:추가\s*(?:해줘|해주세요|해달라|해라|해|해주|하기|할래|좀)?|등록\s*(?:해줘|해주세요|해달라|해라|해|해주|하기|할래|좀)?|넣어\s*(?:줘|주세요|달라|라|줄래)?|추가|등록)'
        patterns = [
            rf'할\s*일(?:에|로|을)?\s*(.+?)\s*{action_verbs}',
            rf'(.+?)\s*할\s*일(?:에|로|을)?\s*{action_verbs}',
            rf'^(.+?)\s*{action_verbs}',
            rf'할\s*일\s*(?:추가|등록)\s*[,，:\s]\s*(.+)$',
            rf'^(?:추가|등록)\s*(.+)$',
            r'^(.+?)\s*(?:해야\s*돼|끝내야).*?(?:할\s*일에|등록해|넣어)',
        ]
    elif intent == 'COMPLETE_TASK':
        complete_verbs = r'(?:완료\s*(?:해줘|했어|처리|됐어|로)?|끝\s*(?:났어|냈어|남)?|다\s*했어)'
        patterns = [
            rf'할\s*일\s*중에\s*(.+?)\s*{complete_verbs}',
            rf'^(.+?)\s*{complete_verbs}',
            rf'(.+?)\s*할\s*일\s*{complete_verbs}'
        ]
    elif intent == 'DELETE_TASK':
        delete_verbs = r'(?:지워\s*(?:줘)?|삭제\s*(?:해줘|해)?|빼\s*(?:줘)?)'
        patterns = [
            rf'할\s*일에서\s*(.+?)\s*{delete_verbs}',
            rf'^(.+?)\s*할\s*일\s*{delete_verbs}',
            rf'^(.+?)\s*{delete_verbs}'
        ]
    else:
        patterns = []

    for pattern in patterns:
        m = re.search(pattern, working)
        if m:
            title = clean_title(m.group(1))
            if title:
                slots['TITLE'] = title
                break
    return {k: v for k, v in slots.items() if v is not None}


def predict_intent(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('텍스트를 입력하세요')
    model, model_name = load_intent_model()
    intent = model.predict([text])[0]
    if model_name == 'LogisticRegression':
        score = float(model.predict_proba([text]).max())
        score_type = 'probability'
    else:
        score = float(model.decision_function([text]).max())
        score_type = 'decision_score'

    # 명확한 의도 키워드 기반 오분류 보정 (하이브리드 NLU)
    add_kw = ['추가', '등록', '넣어']
    complete_kw = ['완료', '다 했어', '끝났어', '끝냈어']
    delete_kw = ['삭제', '지워', '빼줘', '빼']
    list_kw = ['목록', '보여줘', '알려줘', '조회', '뭐 있어', '확인']

    compact = re.sub(r'\s+', '', text)
    rule_matched = True
    quote = re.search(r'라고\s*(?:메모|적어|써|남겨)', text)
    if quote and not re.search(r'삭제|지워|조회|보여|둔|놓은|있는', text[quote.end():]):
        intent = 'ADD_MEMO'
    elif re.match(r'^메모\s*(?:추가|등록)(?:\s*[:,，]\s*|\s+)\S', text):
        intent = 'ADD_MEMO'
    elif re.search(r'도움말|사용할수있는명령|사용법', compact):
        intent = 'HELP'
    elif re.search(r'메모(?:해|장에)|메모추가[:：,，]', compact) and not re.search(r'삭제|지워|조회|보여', compact):
        intent = 'ADD_MEMO'
    elif '메모' not in text and re.search(r'바꿔|변경해|수정해|미뤄|옮겨', compact):
        intent = 'UPDATE_TASK'
    elif ('메모' not in text and any(k in compact for k in ['미완료', '완료한', '완료된', '안끝낸'])
          and any(k in compact for k in ['보여', '알려', '조회', '목록', '뭐있어', '확인'])):
        intent = 'LIST_TASK'
    elif any(k in text for k in add_kw) and not any(k in text for k in complete_kw + delete_kw + list_kw):
        intent = 'ADD_MEMO' if '메모' in text else 'ADD_TASK'
    elif any(k in text for k in complete_kw) and not any(k in text for k in add_kw + delete_kw):
        intent = 'COMPLETE_TASK'
    elif any(k in text for k in delete_kw) and not any(k in text for k in add_kw + complete_kw):
        intent = 'DELETE_MEMO' if '메모' in text else 'DELETE_TASK'
    elif any(k in text for k in list_kw) and not any(k in text for k in add_kw + complete_kw + delete_kw):
        intent = 'LIST_MEMO' if '메모' in text else 'LIST_TASK'
    else:
        rule_matched = False

    slot_text = text if intent == 'ADD_MEMO' else normalize_text(text)
    slots = extract_slots(slot_text, intent)
    needs_confirm = (
        (not rule_matched and score < 0.4)
        or (intent in {'ADD_TASK', 'COMPLETE_TASK', 'DELETE_TASK', 'UPDATE_TASK'} and 'TITLE' not in slots)
        or (intent == 'ADD_MEMO' and 'MEMO_CONTENT' not in slots)
    )

    return {
        'model_name': model_name,
        'intent': intent,
        'confidence_or_score': round(score, 4),
        'score_type': score_type,
        'slots': slots,
        'needs_confirmation': needs_confirm
    }


HELP_TEXT = (
    '할 일 추가, 조회, 수정, 완료, 삭제 / '
    '메모 추가, 조회, 삭제 명령을 사용할 수 있습니다'
)


# 할 일 전체 조회 및 검색
def task_matches(title, due_date=None):
    rows = list_all_tasks()
    matches = [row for row in rows if row.get('title') == title]
    if not matches and title:
        matches = [row for row in rows if title in row.get('title', '')]

    if due_date:
        matches = [row for row in matches if row.get('due_date') and
                   str(row['due_date']) == str(due_date)]

    return matches


# 메모 조회 및 검색
def memo_matches(keyword=None, memo_date=None):
    rows = list_memo(limit=100)
    if keyword:
        rows = [row for row in rows if keyword in row.get('content', '')]
    if memo_date:
        rows = [row for row in rows if str(row['created_at'].date()) == str(memo_date)]

    return rows


def select_one(rows, entity_name='항목'):
    if not rows:
        return None, f"해당하는 {entity_name}을(를) 찾을 수 없습니다"
    if len(rows) > 1:
        return None, f"일치하는 {entity_name}이(가) 여러 개 있습니다. 날짜나 키워드를 함께 말씀해 주세요"
    return rows[0], None


# [수정] 중복 후보가 있을 때 후보 목록과 다시 말할 예시를 함께 안내
def _fmt_date(d):
    if not d:
        return '날짜 없음'
    d = d if isinstance(d, date) else date.fromisoformat(str(d)[:10])
    diff = (d - date.today()).days
    label = {0: '오늘', 1: '내일', 2: '모레', -1: '어제'}.get(diff)
    return f"{d.month}월 {d.day}일" + (f"({label})" if label else '')


def ambiguous_task_message(title, rows, intent, slots=None):
    verb = {'COMPLETE_TASK': '완료', 'DELETE_TASK': '삭제', 'UPDATE_TASK': '수정'}.get(intent, '처리')
    dates = [_fmt_date(r.get('due_date')) for r in rows]
    example_day = next((r['due_date'] for r in rows if r.get('due_date')), None)
    example = f"{example_day.day}일 {title} {verb}" if isinstance(example_day, date) else f"{title} {verb}"
    all_hint = f", 모두 처리하려면 '{title} 모두 {verb}'" if intent != 'UPDATE_TASK' else ''
    if intent == 'UPDATE_TASK' and isinstance(example_day, date):
        slots = slots or {}
        if slots.get('NEW_TITLE'):
            example = f"{example_day.day}일 {title} 이름을 {slots['NEW_TITLE']}로 바꿔"
        elif slots.get('NEW_DUE_DATE'):
            new_d = date.fromisoformat(slots['NEW_DUE_DATE'])
            example = f"{example_day.day}일 {title} 날짜를 {new_d.day}일로 바꿔"
    return (f"'{title}' 할 일이 {len(rows)}개 있습니다: {', '.join(dates)}. "
            f"어느 날짜인지 함께 말씀해 주세요 (예: '{example}'{all_hint})")


def ambiguous_memo_message(rows, keyword=None):
    preview = ' / '.join(f"{i}) {r['content'][:15]}" for i, r in enumerate(rows[:5], start=1))
    head = f"'{keyword}' 관련 메모가 {len(rows)}개 있습니다" if keyword else f"메모가 {len(rows)}개 있습니다"
    return (f"{head}: {preview}. 더 구체적인 키워드나 날짜를 말씀해 주세요 "
            f"(예: '{(rows[0]['content'].split() or ['회의'])[0]} 관련 메모 삭제해', '최근 메모 삭제해', "
            f"'{keyword or '회의'} 관련 메모 모두 삭제해')")


def execute(intent, slots):
    slots = slots or {}
    title = slots.get('TITLE')
    due_date = slots.get('DUE_DATE')

    if intent == 'HELP':
        return HELP_TEXT, None

    if intent == 'ADD_TASK':
        if not title:
            return '추가할 할 일의 제목을 말씀해주세요', None

        # 중복 체크: 동일한 제목과 마감일을 가진 미완료 할 일이 이미 있는지 확인
        existing_tasks = list_all_tasks()
        for t in existing_tasks:
            if not t.get('completed') and t.get('title', '').strip() == title.strip():
                t_due = str(t.get('due_date')) if t.get('due_date') else None
                new_due = str(due_date) if due_date else None
                if t_due == new_due:
                    return f'이미 등록된 할 일입니다: {title}', {'id': str(t['id']), 'already_exists': True}

        task_id = add_task(title, title, due_date)
        return f'할 일: {title}을(를) 추가했습니다', {'id': str(task_id)}

    if intent == 'LIST_TASK':
        tasks = list_all_tasks()
        if due_date:
            tasks = [row for row in tasks if row.get('due_date') and
                     str(row['due_date']) == str(due_date)]
        status = slots.get('STATUS', '')
        if status in ('완료한', '완료된'):
            tasks = [row for row in tasks if row.get('completed')]
        elif status in ('안 끝낸', '미완료'):
            tasks = [row for row in tasks if not row.get('completed')]
        if not tasks:
            return '조회된 할 일이 없습니다', []
        lines = [f"{i}. {row['title']} ({'완료' if row.get('completed') else '미완료'}, 마감: {row.get('due_date') or '미지정'})"
                 for i, row in enumerate(tasks, start=1)]
        return '\n\n'.join(lines), tasks

    if intent in ('UPDATE_TASK', 'COMPLETE_TASK', 'DELETE_TASK'):
        if not title:
            return '처리할 할 일의 제목을 말씀해 주세요', None
        matches = task_matches(title, due_date)
        # [수정] 완료 처리는 아직 안 끝낸 할 일만 대상으로 (이미 완료한 같은 이름 할 일 제외)
        if intent == 'COMPLETE_TASK':
            pending = [r for r in matches if not r.get('completed')]
            if matches and not pending:
                return f"'{title}' 할 일은 이미 완료되어 있습니다", None
            matches = pending
        if not matches:
            where = f"{_fmt_date(due_date)}에 " if due_date else ''
            return f"{where}'{title}' 할 일을 찾을 수 없습니다", None

        # [수정] '모두/전부'라고 말한 경우에만 여러 개를 한 번에 처리 (완료·삭제)
        if len(matches) > 1 and slots.get('ALL') and intent in ('COMPLETE_TASK', 'DELETE_TASK'):
            ids = [str(r['id']) for r in matches]
            if intent == 'DELETE_TASK':
                delete_tasks(ids)
            else:
                complete_tasks(ids)
            verb = '삭제' if intent == 'DELETE_TASK' else '완료 처리'
            return f"'{title}' 할 일 {len(ids)}개를 모두 {verb}했습니다", {'ids': ids, 'count': len(ids)}

        if len(matches) > 1:
            matches.sort(key=lambda r: (r.get('due_date') is None, str(r.get('due_date'))))
            return ambiguous_task_message(title, matches, intent, slots), None
        row = matches[0]
        old_title = row['title']

        if intent == 'DELETE_TASK':
            count = delete_task(row['id'])
            if count:
                return f"{_fmt_date(row.get('due_date'))} 할 일: {row['title']}을(를) 삭제했습니다", {'id': str(row['id']), 'count': count}
            return '삭제할 할 일이 없습니다', None

        new_title = (
            slots.get('NEW_TITLE')
            if intent == 'UPDATE_TASK'
            else None
        )

        new_due = (
            slots.get('NEW_DUE_DATE')
            if intent == 'UPDATE_TASK'
            else None
        )

        if intent == 'UPDATE_TASK' and not (new_title or new_due):
            return '변경할 제목이나 마감일을 입력해주세요', None

        # [수정] 이름 변경 시 화면에 표시되는 title 컬럼도 함께 변경 (기존엔 content만 바뀌어 반영 안 됨)
        changed = update_task(
            row['id'],
            due_date=new_due or row.get('due_date'),
            content=new_title or row.get('content'),
            completed=True if intent == 'COMPLETE_TASK' else row.get('completed'),
            title=new_title
        )

        if not changed:
            return '변경에 실패했습니다', None

        title = old_title
        if intent == 'COMPLETE_TASK':
            return f"{_fmt_date(row.get('due_date'))} 할 일: {title}을(를) 완료 처리했습니다", {'id': str(row['id'])}

        if new_title and new_due:
            return (
                f'할 일 {title}을 {new_title}로, '
                f'마감일을 {new_due}로 변경했습니다',
                {'id': str(row['id'])}
            )

        if new_title:
            return f'할 일 {title}을 {new_title}로 수정했습니다', {'id': str(row['id'])}

        if new_due:
            return f'{title}의 마감일을 {_fmt_date(new_due)}로 변경했습니다', {'id': str(row['id'])}

    if intent == 'ADD_MEMO':
        content = slots.get('MEMO_CONTENT')
        if not content:
            return '메모 내용을 입력해주세요', None

        # 중복 체크: 동일한 내용의 메모가 이미 있는지 확인
        existing_memos = list_memo(limit=50)
        for m in existing_memos:
            if m.get('content', '').strip() == content.strip():
                return f'이미 저장된 메모입니다: {content}', {'id': str(m['id']), 'already_exists': True}

        memo_id = add_memo(content)
        return '메모를 저장했습니다', {'id': str(memo_id)}

    if intent == 'LIST_MEMO':
        keyword = slots.get('KEYWORD')
        memo_date = slots.get('MEMO_DATE')
        memos = memo_matches(keyword, memo_date)
        if not memos:
            return '조회할 메모가 없습니다', []
        lines = [f"{i}. {row['content']}" for i, row in enumerate(memos, start=1)]
        return '\n\n'.join(lines), memos

    if intent == 'DELETE_MEMO':
        keyword = slots.get('KEYWORD')
        memo_date = slots.get('MEMO_DATE')
        memos = memo_matches(keyword, memo_date)

        # [수정] '최근/마지막/방금 메모 삭제' → 가장 최근 메모 1개
        if slots.get('TARGET') == 'LATEST' and memos:
            memos = memos[:1]
        elif not (keyword or memo_date):
            # [수정] '메모 삭제해'처럼 대상이 없으면 최근 메모를 보여주며 다시 말하도록 안내
            if not memos:
                return '삭제할 메모가 없습니다', None
            return ambiguous_memo_message(memos), None

        if not memos:
            return (f"'{keyword}' 관련 메모를 찾을 수 없습니다" if keyword
                    else '해당 날짜의 메모를 찾을 수 없습니다'), None

        if len(memos) > 1:
            if not slots.get('ALL'):
                return ambiguous_memo_message(memos, keyword), None
            ids = [str(r['id']) for r in memos]
            delete_memos(ids)
            return f'{len(ids)}개 메모를 모두 삭제했습니다', {'ids': ids, 'count': len(ids)}

        row = memos[0]
        count = delete_memo(row['id'])
        if count:
            return f"메모 '{row['content'][:20]}'을(를) 삭제했습니다", {'id': str(row['id']), 'count': count}
        return '메모 삭제에 실패했습니다', None

    return f'작업을 수행할 수 없습니다: {intent}', None

@lru_cache(maxsize=1)
def init_db():
    if has_database_url():
        create_table()

def process_text(text, input_type='텍스트'):  # [수정] input_type 인자 추가 (텍스트 / 직접 녹음 / 파일 업로드)
    t0 = time.perf_counter()
    result = {'model_name': None, 'intent': 'UNKNOWN', 'slots': {},
              'needs_confirmation': False, 'confidence_or_score': 0.0,
              'score_type': 'unavailable'}
    intent = 'UNKNOWN'

    try:
        if not isinstance(text, str) or not text.strip():
            raise ValueError('텍스트를 입력하세요')
        result = predict_intent(text)
        intent = result['intent']
        if any(convert_date(m.group()) is None for m in DATE_PATTERN.finditer(normalize_text(text))) and intent != 'ADD_MEMO':
            result['needs_confirmation'] = True
            response, data, status = '유효한 날짜를 입력해 주세요.', None, '확인 필요'
        elif result['needs_confirmation']:
            response, data, status = '명령을 확실히 이해하지 못했습니다. 제목과 내용을 확인해 다시 입력해 주세요.', None, '확인 필요'
        else:
            if intent != 'HELP':
                init_db()
            response, data = execute(intent, result['slots'])
            status = '성공' if data is not None or intent == 'HELP' else '확인 필요'
    except ValueError as exc:
        response = str(exc)
        data = None
        status = '실패'
    except Exception:
        response = '명령 처리에 실패했습니다. 모델 및 DB 연결 상태를 확인한 후 다시 시도해 주세요.'
        data = None
        status = '실패'

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    run_id = None
    history_warning = None
    try:
        if has_database_url():
            run_id = add_voice_runs(
                transcript=text,
                intent=intent,
                status=status,
                response_text=response,
                latency_ms=elapsed_ms,
                error_message=None if status == '성공' else response,
                input_type=input_type  # [수정] 처리 이력에 입력 방식 저장
            )
    except Exception:
        history_warning = '처리 이력을 DB에 저장하지 못했습니다. 현재 세션에서만 확인할 수 있습니다.'

    return {
        **result,
        'text': text,
        'response': response,
        'data': data,
        'status': status,
        'run_id': str(run_id) if run_id else None,
        'history_warning': history_warning,
        'latency_ms': elapsed_ms,
        'input_type': input_type  # [수정]
    }


if __name__ == '__main__':
    if sys.stdin.isatty():
        try:
            text = input('명령을 입력하세요: ')
            print(process_text(text))
        except (KeyboardInterrupt, EOFError):
            pass
    else:
        samples = [
            '내일 할 일에 운동 추가해 줘',
            '10월 30일까지 택배 보내기 할 일에 넣어 줘',
            '다음 주 월요일 할 일 뭐 있어?',
            '운동 마감을 이번 주 금요일로 바꿔 줘',
            '보고서 작성 완료했어',
            '과제 제출 할 일 삭제해 줘',
            '메모 추가, 프린터 잉크 사야 함',
            '과제 관련 메모 보여 줘',
            '시험 관련 메모 지워 줘',
            '사용할 수 있는 명령 알려 줘'
        ]
        for t in samples:
            print('\n명령:', t)
            print(process_text(t))
