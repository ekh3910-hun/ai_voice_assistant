import time
import sys
from functools import lru_cache
from pathlib import Path
from datetime import date, timedelta
import re
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from src.db import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, precision_recall_fscore_support,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

BASE = Path(__file__).resolve().parent.parent
CSV_PATH = BASE / 'data' / 'intent_train.csv'
REPORT_PATH = BASE / 'reports'
MODEL_PATH = BASE / 'models' / 'intent_model.joblib'
TOKEN_PATTERN = r'(?u)\b\w+\b'


@lru_cache(maxsize=1)
def train_model(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    X, y = df['text'], df['intent']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logistic = Pipeline([
        ('tfidf', TfidfVectorizer(token_pattern=TOKEN_PATTERN, ngram_range=(1, 2))),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ])
    linear = Pipeline([
        ('tfidf', TfidfVectorizer(token_pattern=TOKEN_PATTERN, ngram_range=(1, 2))),
        ('model', LinearSVC(max_iter=1000, random_state=42, dual=False))
    ])

    results = []
    trained = {}
    for name, candidate in [('LogisticRegression', logistic), ('LinearSVC', linear)]:
        candidate.fit(X_train, y_train)
        pred = candidate.predict(X_test)
        trained[name] = candidate
        results.append({
            'model': name,
            'Accuracy': accuracy_score(y_test, pred),
            'Precision': precision_score(y_test, pred, average='macro', zero_division=0),
            'Recall': recall_score(y_test, pred, average='macro', zero_division=0),
            'F1': f1_score(y_test, pred, average='macro', zero_division=0)
        })

    print('[모델 비교 평가 (8:2)]')
    print(pd.DataFrame(results).round(4).to_string(index=False))

    best = max(results, key=lambda r: (r['Accuracy'], r['F1']))
    model_name = best['model']
    model = trained[model_name]
    print('선택 모델:', model_name)

    predicted = model.predict(X_test)
    print('테스트 Accuracy:', round(accuracy_score(y_test, predicted), 4))
    print('테스트 Macro F1:', round(f1_score(y_test, predicted, average='macro'), 4))

    p, r, f, support = precision_recall_fscore_support(
        y_test, predicted, labels=model.classes_, zero_division=0
    )
    print('\n[의도별 테스트 성능]')
    metrics_df = pd.DataFrame({
        'intent': model.classes_, 'Precision': p, 'Recall': r,
        'F1': f, 'count': support
    }).round(4)
    print(metrics_df.to_string(index=False))

    # 리포트 및 모델 저장
    REPORT_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(REPORT_PATH / 'intent_metrics.csv', index=False, encoding='utf-8-sig')

    ConfusionMatrixDisplay.from_predictions(y_test, predicted, xticks_rotation=90)
    plt.tight_layout()
    plt.savefig(REPORT_PATH / 'confusion_matrix.png', bbox_inches='tight')
    plt.close()

    errors = pd.DataFrame({
        'text': X_test.to_numpy(), 'intent': y_test.to_numpy(), 'predict': predicted
    })
    errors = errors[errors['intent'] != errors['predict']]
    print('\n오분류 건수:', len(errors))
    if not errors.empty:
        print(errors.to_string(index=False))

    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
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
    value = re.sub(r'^(?:할\s*일(?:에|에서|중에|로)?|메모(?:에|장에)?)\s*', '', value)
    value = re.sub(r'^(?:에|를|을|은|는)\s+', '', value)
    value = re.sub(r'\s*(?:을|를|은|는|이|가|에|로|으로|까지)$', '', value)
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

    working = DATE_PATTERN.sub(' ', working)
    working = re.sub(r'\s+', ' ', working).strip()
    working = re.sub(r'^(?:까지|에|부터)\s+', '', working)

    if intent == 'HELP':
        return {}

    if intent == 'LIST_TASK':
        m = re.search(r'안\s*끝낸|미완료|완료한|완료된', working)
        if m:
            slots['STATUS'] = m.group()
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
            r'^(?:메모장|메모)(?:에)?\s+(.+?)\s*(?:추가|등록|저장)(?:해줘|해|해주세요)?[.!?]?$',
            r'^(.+?)\s*이라고\s*(?:메모|적어)',
            r'^메모\s*추가\s*[,，:]\s*(.+)$',
            r'^(?:메모장|메모)(?:에)?\s*(.+?)\s*(?:남겨|적어|써)',
            r'^(.+?)\s*[,，]?\s*메모해',
            r'^(.+?)\s*메모(?:해|로\s*남겨|에\s*적어)'
        ]
        for pattern in patterns:
            m = re.search(pattern, working)
            if m:
                slots['MEMO_CONTENT'] = clean_title(m.group(1))
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
        patterns = [
            r'할\s*일(?:에|로)?\s*(.+?)\s*(?:추가해|넣어|등록해)',
            r'^(.+?)\s*(?:해야\s*돼|끝내야).*?(?:할\s*일에|등록해|넣어)',
            r'^(.+?)\s*(?:할\s*일로\s*)?(?:등록해|추가해|넣어)',
            r'^할\s*일\s*추가\s*[,，:]\s*(.+)$'
        ]
    elif intent == 'COMPLETE_TASK':
        patterns = [
            r'할\s*일\s*중에\s*(.+?)\s*(?:완료|끝)',
            r'^(.+?)\s*(?:완료|끝났어|다\s*했어)',
            r'^(.+?)\s*(?:완료로|완료\s*처리)'
        ]
    elif intent == 'DELETE_TASK':
        patterns = [
            r'할\s*일에서\s*(.+?)\s*(?:지워|삭제|빼)',
            r'^(.+?)\s*할\s*일\s*(?:삭제|지워)',
            r'^(.+?)\s*(?:안\s*해도\s*돼|삭제|지워|빼)'
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
    model, model_name = train_model()
    intent = model.predict([text])[0]
    if model_name == 'LogisticRegression':
        score = float(model.predict_proba([text]).max())
        score_type = 'probability'
    else:
        score = float(model.decision_function([text]).max())
        score_type = 'decision_score'

    if re.fullmatch(
            r'\s*(?:메모장|메모)(?:에)?\s+.+?\s*'
            r'(?:추가|등록|저장)(?:해줘|해|해주세요)?[.!?]?\s*',
            text
    ):
        intent = 'ADD_MEMO'

    slots = extract_slots(text, intent)
    needs_confirm = (
        (score < 0.4)
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

        task_id = add_task(title, title, due_date)
        return f'할 일: {title}을 추가했습니다', {'id': str(task_id)}

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
        lines = [f"{row['title']} ({'완료' if row.get('completed') else '미완료'}, "
                 f"마감: {row.get('due_date') or '미지정'})" for row in tasks]
        return '\n'.join(lines), tasks

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

        memo_id = add_memo(content)
        return '메모를 저장했습니다', {'id': str(memo_id)}

    if intent in ('LIST_MEMO', 'DELETE_MEMO'):
        keyword = slots.get('KEYWORD')
        memo_date = slots.get('MEMO_DATE')

        if intent == 'DELETE_MEMO' and not (keyword or memo_date):
            return '삭제할 메모의 내용이나 날짜를 알려주세요', None

        memos = memo_matches(keyword, memo_date)

        if intent == 'LIST_MEMO':
            if not memos:
                return '조회할 메모가 없습니다', []
            return '\n'.join(f"- {row['content']}" for row in memos), memos

        row, error = select_one(memos, '메모')
        if error:
            return error, None

        count = delete_memo(row['id'])
        if count:
            return f'{count}개 메모를 삭제했습니다', {'count': count}
        return '메모 삭제에 실패했습니다', {'count': 0}

    return f'작업을 수행할 수 없습니다: {intent}', None


def process_text(text):
    t0 = time.perf_counter()
    result = predict_intent(text)

    intent = result['intent']
    slots = result['slots']

    try:
        response, data = execute(intent, slots)

        if intent == 'HELP':
            status = '안내'
        elif data is None:
            status = '실패'
        elif isinstance(data, dict) and 'count' in data:
            status = '성공' if data['count'] > 0 else '실패'
        else:
            status = '성공'
    except Exception as e:
        response = f'DB 처리 중 오류 발생, {e}'
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
                error_message=None if status == '성공' else response
            )
    except Exception:
        pass

    return {
        **result,
        'text': text,
        'response': response,
        'data': data,
        'status': status,
        'latency_ms': elapsed_ms
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