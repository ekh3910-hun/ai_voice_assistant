# core.py

import time
import sys
from functools import lru_cache
from pathlib import Path
from datetime import date, timedelta
import re
import joblib
from src.db import *
BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
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
    r'오늘|내일|모레|어제|그저께|방금'
)


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
    return None


def clean_title(value):
    value = value.strip(' ,，.!?')
    value = re.sub(r'^(?:할\s*일(?:에|에서|중에|로|을|이)?|메모(?:에|장에)?)\s*', '', value)
    value = re.sub(r'^(?:에|를|을|은|는)\s+', '', value)
    value = re.sub(r'\s*(?:을|를|은|는|이|가|에|로|으로|까지)$', '', value)
    value = re.sub(r'\s*(?:추가\s*(?:해줘|해달라|해주세요|해라|해|해주|하기|할래|좀)?|등록\s*(?:해줘|해달라|해주세요|해라|해|해주|하기|할래|좀)?|넣어\s*(?:줘|주세요|달라|라)?|추가|등록).*$', '', value)
    value = re.sub(r'^(?:추가|등록|할\s*일)\s*', '', value)
    return value.strip(' ,，.!?')


def extract_slots(text, intent):
    slots = {}
    working = text.strip()
    dates = list(DATE_PATTERN.finditer(text))

    if intent in {'ADD_TASK', 'LIST_TASK', 'UPDATE_TASK', 'COMPLETE_TASK', 'DELETE_TASK'}:
        if intent == 'UPDATE_TASK':
            if dates:
                if len(dates) >= 2:
                    slots['DUE_DATE'] = convert_date(dates[0].group())
                    slots['NEW_DUE_DATE'] = convert_date(dates[-1].group())
                else:
                    slots['NEW_DUE_DATE'] = convert_date(dates[0].group())
        elif dates:
            slots['DUE_DATE'] = convert_date(dates[0].group())
    elif intent in {'LIST_MEMO', 'DELETE_MEMO'} and dates:
        slots['MEMO_DATE'] = convert_date(dates[0].group())

    # 메모 본문에 포함된 날짜는 사용자 데이터이므로 보존합니다.
    if intent != 'ADD_MEMO':
        working = DATE_PATTERN.sub(' ', working)
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
            if keyword and keyword not in {'쓴', '적어 둔', '작성한', '방금'}:
                slots['KEYWORD'] = keyword
        return {k: v for k, v in slots.items() if v is not None}

    if intent == 'ADD_MEMO':
        patterns = [
            r'^(.+?)\s*(?:이라고|라고)\s*(?:메모|적어)',
            r'^메모\s*추가\s*[,，:]\s*(.+)$',
            r'^(?:메모장|메모)(?:에)?\s*(.+?)\s*(?:남겨|적어|써)',
            r'^(.+?)\s*[,，]?\s*메모해',
            r'^(.+?)\s*메모(?:해|로\s*남겨|에\s*적어)'
        ]
        for pattern in patterns:
            m = re.search(pattern, working)
            if m:
                slots['MEMO_CONTENT'] = m.group(1).strip(' ,，.!?')
                break
        return slots

    if intent == 'UPDATE_TASK':
        m = re.search(r'^(.+?)\s*(?:이름을|제목을|을|를)\s*(.+?)(?:로|으로)\s*(?:바꿔|수정해|변경해)', working)
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
    if re.search(r'도움말|사용할수있는명령|사용법', compact):
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

    slots = extract_slots(text, intent)
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
        row, error = select_one(task_matches(title, due_date), '할 일')
        if error:
            return error, None

        if intent == 'DELETE_TASK':
            count = delete_task(row['id'])
            if count:
                return f'할 일: {title}을 삭제했습니다', {'count': count}
            return '삭제할 할 일이 없습니다', {'count': count}

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

        changed = update_task(
            row['id'],
            due_date=new_due or row.get('due_date'),
            content=new_title or row.get('content'),
            title=new_title,
            completed=True if intent == 'COMPLETE_TASK' else row.get('completed')
        )

        if not changed:
            return '변경에 실패했습니다', None

        if intent == 'COMPLETE_TASK':
            return f'할 일: {title}을 완료 처리했습니다', {'id': str(row['id'])}

        if new_title and new_due:
            return (
                f'할 일 {title}을 {new_title}로, '
                f'마감일을 {new_due}로 변경했습니다',
                {'id': str(row['id'])}
            )

        if new_title:
            return f'할 일 {title}을 {new_title}로 수정했습니다', {'id': str(row['id'])}

        if new_due:
            return f'{title}의 마감일을 {new_due}로 변경했습니다', {'id': str(row['id'])}

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
        if not (keyword or memo_date):
            return '삭제할 메모의 내용이나 날짜를 알려주세요', None
        memos = memo_matches(keyword, memo_date)
        row, error = select_one(memos, '메모')
        if error:
            return error, None
        count = delete_memo(row['id'])
        if count:
            return f'{count}개 메모를 삭제했습니다', {'count': count}
        return '메모 삭제에 실패했습니다', {'count': 0}

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
        if any(convert_date(m.group()) is None for m in DATE_PATTERN.finditer(text)) and intent != 'ADD_MEMO':
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

    try:
        if has_database_url():
            add_voice_runs(
                transcript=text,
                intent=intent,
                status=status,
                response_text=response,
                latency_ms=elapsed_ms,
                error_message=None if status == '성공' else response,
                input_type=input_type  # [수정] 처리 이력에 입력 방식 저장
            )
    except Exception as e:
        print(f"[voice_runs 저장 오류] {type(e).__name__}: {e}")

    return {
        **result,
        'text': text,
        'response': response,
        'data': data,
        'status': status,
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
