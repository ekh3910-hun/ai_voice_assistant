
from pathlib import Path
import sys
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from src.core import load_intent_model, predict_intent

TEST_CASES = [
    ("오늘 할 일에 운동 추가해줘", "ADD_TASK"),
    ("메모에 장보기 추가해줘", "ADD_MEMO"),
    ("저장한 메모 보여줘", "LIST_MEMO"),
    ("내일 할 일 목록 알려줘", "LIST_TASK"),
    ("운동 할 일 삭제해줘", "DELETE_TASK"),
    ("장보기 메모 삭제해줘", "DELETE_MEMO"),
    ("운동 완료 처리해줘", "COMPLETE_TASK"),
    ("운동 시간을 오후 3시로 변경해줘", "UPDATE_TASK"),
    ("사용 방법 알려줘", "HELP"),
    ("메모 추가하는 방법 알려줘", "HELP"),
    ("할 일 삭제하는 방법 알려줘", "HELP"),
    ("메모 말고 할 일에 운동 추가해줘", "ADD_TASK"),
    ("운동을 추가한 할 일 목록 보여줘", "LIST_TASK"),
    ("완료한 할 일 목록 보여줘", "LIST_TASK"),
    ("저장된 메모를 모아서 읽어줘", "LIST_MEMO"),
]

def main():
    model, model_name = load_intent_model()
    rows = []

    for sentence, expected in TEST_CASES:
        raw = str(model.predict([sentence])[0])
        final = str(predict_intent(sentence)["intent"])

        rows.append({
            "text": sentence,
            "expected": expected,
            "model_only": raw,
            "after_keywords": final,
            "changed": raw != final,
            "model_correct": raw == expected,
            "final_correct": final == expected,
        })

    df = pd.DataFrame(rows)

    print(f"\n모델: {model_name}")
    print(f"테스트 문장: {len(df)}개")
    print(f"키워드가 예측을 변경한 횟수: {df['changed'].sum()}개")
    print(f"모델 단독 정확도: {df['model_correct'].mean():.2%}")
    print(f"키워드 적용 정확도: {df['final_correct'].mean():.2%}")

    print("\n===== 문장별 결과 =====")
    print(df.to_string(index=False))

    report_dir = BASE / "reports"
    report_dir.mkdir(exist_ok=True)

    output = report_dir / "keyword_override_results.csv"
    df.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {output}")

if __name__ == "__main__":
    main()
