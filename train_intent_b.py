
# train_intent_b.py
# 한국어 음성 명령 의도 분류 모델 학습

from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ===== 경로 설정 =====

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "intent_train.csv"
MODEL_PATH = BASE_DIR / "models" / "intent_model.joblib"

RANDOM_STATE = 42


# ===== 1. 학습 데이터 불러오기 =====

def load_data():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"학습 데이터가 없습니다: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    # 필요한 컬럼 확인
    if not {"text", "intent"}.issubset(df.columns):
        raise ValueError(
            "CSV에는 text, intent 컬럼이 필요합니다."
        )

    # 결측값과 중복 데이터 제거
    df = df[["text", "intent"]].dropna().copy()

    df["text"] = df["text"].astype(str).str.strip()
    df["intent"] = df["intent"].astype(str).str.strip()

    df = df[
        (df["text"] != "") &
        (df["intent"] != "")
    ]

    df = df.drop_duplicates()

    if df["intent"].nunique() < 2:
        raise ValueError(
            "최소 2개 이상의 의도가 필요합니다."
        )

    return df


# ===== 2. 모델 생성 =====

def create_model():
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 4),
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        )
    ])

    return model


# ===== 3. 모델 학습 및 정확도 평가 =====

def train_model(df):
    X = df["text"]
    y = df["intent"]

    # 학습용 80%, 평가용 20% 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 모델 생성 및 학습
    model = create_model()
    model.fit(X_train, y_train)

    # 정확도 평가
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("\n===== 모델 평가 =====")
    print(f"정확도: {accuracy:.2%}")
    print(f"평가 문장 수: {len(X_test)}개")

    return accuracy


# ===== 4. 전체 데이터 재학습 =====

def fit_final_model(df):
    model = create_model()

    model.fit(
        df["text"],
        df["intent"],
    )

    return model


# ===== 5. 모델 저장 =====

def save_model(model):
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(model, MODEL_PATH)

    print(f"\n모델 저장 완료: {MODEL_PATH}")

# ===== 6. 메인 실행 =====

def main():
    print("의도 분류 모델 학습 시작")

    # 데이터 불러오기
    df = load_data()

    print(f"전체 학습 데이터: {len(df)}개")
    print(f"의도 종류: {df['intent'].nunique()}개")

    # 모델 정확도 평가
    accuracy = train_model(df)

    # 전체 데이터로 최종 모델 학습
    final_model = fit_final_model(df)

    # 모델 저장
    save_model(final_model)

    print(f"\n평가 정확도: {accuracy:.2%}")
    print("학습 완료")


if __name__ == "__main__":
    main()

