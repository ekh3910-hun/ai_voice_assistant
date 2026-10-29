"""
models/model.py
- 한국어 음성 명령 의도(Intent) 분류 모델 학습 및 성능 평가 모듈
- TF-IDF + LogisticRegression / LinearSVC 비교 평가
- 최종 모델(models/intent_model.pkl) 및 평가 리포트(reports/) 생성
"""

from functools import lru_cache
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_fscore_support,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

BASE = Path(__file__).resolve().parent.parent
CSV_PATH = BASE / 'data' / 'commands_RE.csv'
REPORT_PATH = BASE / 'reports'
MODEL_PATH = BASE / 'models' / 'intent_model.pkl'
TOKEN_PATTERN = r'(?u)\b\w+\b'


def train_model(csv_path=CSV_PATH):
    print(f"[1/5] 데이터셋 로드: {csv_path}")
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    X, y = df['text'], df['intent']

    # 8:2 분할 (의도 레이블 계층화 추출)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  - 전체 데이터: {len(df)}건 (Train: {len(X_train)}건 / Test: {len(X_test)}건)")

    # 파이프라인 구성
    logistic = Pipeline([
        ('tfidf', TfidfVectorizer(token_pattern=TOKEN_PATTERN, ngram_range=(1, 2))),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ])
    linear = Pipeline([
        ('tfidf', TfidfVectorizer(token_pattern=TOKEN_PATTERN, ngram_range=(1, 2))),
        ('model', LinearSVC(max_iter=1000, random_state=42, dual=False))
    ])

    print("\n[2/5] 모델 학습 및 성능 비교 평가 (8:2)")
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

    compare_df = pd.DataFrame(results).round(4)
    print(compare_df.to_string(index=False))

    # 최고 성능 모델 선정
    best = max(results, key=lambda r: (r['Accuracy'], r['F1']))
    model_name = best['model']
    model = trained[model_name]
    print(f"\n=> 최종 선정 모델: {model_name} (Accuracy: {best['Accuracy']:.4f}, Macro F1: {best['F1']:.4f})")

    predicted = model.predict(X_test)

    # 의도별 상세 평가 지표 산출
    p, r, f, support = precision_recall_fscore_support(
        y_test, predicted, labels=model.classes_, zero_division=0
    )
    metrics_df = pd.DataFrame({
        'intent': model.classes_,
        'Precision': p,
        'Recall': r,
        'F1': f,
        'count': support
    }).round(4)

    print("\n[3/5] 의도별 테스트 세트 상세 지표")
    print(metrics_df.to_string(index=False))

    # 오분류 사례 분석
    errors_df = pd.DataFrame({
        'text': X_test.to_numpy(),
        'true_intent': y_test.to_numpy(),
        'predict_intent': predicted
    })
    errors_df = errors_df[errors_df['true_intent'] != errors_df['predict_intent']]
    print(f"\n[4/5] 오분류 건수: {len(errors_df)}건")
    if not errors_df.empty:
        print(errors_df.to_string(index=False))

    # 리포트 및 아티팩트 저장
    REPORT_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(REPORT_PATH / 'intent_metrics.csv', index=False, encoding='utf-8-sig')
    errors_df.to_csv(REPORT_PATH / 'intent_errors.csv', index=False, encoding='utf-8-sig')

    ConfusionMatrixDisplay.from_predictions(y_test, predicted, xticks_rotation=90)
    plt.tight_layout()
    plt.savefig(REPORT_PATH / 'confusion_matrix.png', bbox_inches='tight')
    plt.close()
    print(f"  - 리포트 저장 완료: {REPORT_PATH}")

    # 전체 데이터로 최종 학습 후 모델 파일 직렬화
    print("\n[5/5] 전체 데이터셋 기반 최종 모델 재학습 및 저장")
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    print(f"  - 모델 파일 저장 완료: {MODEL_PATH}")

    return model, model_name


if __name__ == '__main__':
    train_model()
