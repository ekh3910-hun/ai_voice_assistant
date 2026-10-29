
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher
import re
import unicodedata

import pandas as pd
from jiwer import cer


BASE = Path(__file__).resolve().parent

INPUT_PATH = BASE / "reports" / "stt_speed_all_details.csv"
REPORT_DIR = BASE / "reports"


def normalize(text):
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"\s+", "", text)
    return text.strip()


def analyze_characters(reference, prediction):
    matcher = SequenceMatcher(
        None, reference, prediction, autojunk=False
    )

    changes = []
    counts = {"대체": 0, "삭제": 0, "삽입": 0}

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        old = reference[i1:i2]
        new = prediction[j1:j2]

        if tag == "replace":
            kind = "대체"
            counts[kind] += max(len(old), len(new))
        elif tag == "delete":
            kind = "삭제"
            counts[kind] += len(old)
        else:
            kind = "삽입"
            counts[kind] += len(new)

        changes.append({
            "오류유형": kind,
            "정답부분": old,
            "인식부분": new,
        })

    return changes, counts


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"결과 파일이 없습니다: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")

    # 속도 1.00배 결과만 선택
    df = df[
        (pd.to_numeric(df["속도"]) - 1.0).abs() < 0.0001
    ].copy()

    if df.empty:
        print("1.00배 평가 결과가 없습니다.")
        return

    results = []
    error_details = []
    common_errors = Counter()

    for _, row in df.iterrows():
        filename = row["파일명"]
        reference = str(row["정답"])
        prediction = str(row["인식 결과"])

        ref_clean = normalize(reference)
        pred_clean = normalize(prediction)

        score = cer(ref_clean, pred_clean)

        changes, counts = analyze_characters(
            ref_clean, pred_clean
        )

        results.append({
            "파일명": filename,
            "정답": reference,
            "인식결과": prediction,
            "공백제외_CER": score,
            "대체": counts["대체"],
            "삭제": counts["삭제"],
            "삽입": counts["삽입"],
            "오류내용": " / ".join(
                f"{c['오류유형']}:{c['정답부분']}→{c['인식부분']}"
                for c in changes
            )
        })

        for change in changes:
            error_details.append({
                "파일명": filename,
                **change
            })

            common_errors[
                (
                    change["오류유형"],
                    change["정답부분"],
                    change["인식부분"]
                )
            ] += 1

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(
        "공백제외_CER",
        ascending=False
    )

    print("\n===== 파일별 STT 오류 순위 =====")
    print(
        result_df[
            ["파일명", "공백제외_CER", "대체", "삭제", "삽입"]
        ].round(4).to_string(index=False)
    )

    print("\n===== 정답과 인식 결과 비교 =====")

    for _, row in result_df.iterrows():
        print(f"\n파일: {row['파일명']}")
        print(f"정답: {row['정답']}")
        print(f"인식: {row['인식결과']}")
        print(f"공백 제외 CER: {row['공백제외_CER']:.2%}")
        print(f"차이: {row['오류내용'] or '없음'}")

    print("\n===== 반복되는 오류 패턴 =====")

    repeated = [
        (key, count)
        for key, count in common_errors.most_common()
        if count >= 2
    ]

    if repeated:
        for (kind, old, new), count in repeated:
            print(f"{kind}: {old} → {new} ({count}회)")
    else:
        print("2회 이상 동일하게 나타난 오류 패턴 없음")

    print("\n===== 전체 요약 =====")
    print(f"평가 파일: {len(result_df)}개")
    print(
        f"평균 공백 제외 CER: "
        f"{result_df['공백제외_CER'].mean():.2%}"
    )
    print(
        f"완전 일치: "
        f"{(result_df['공백제외_CER'] == 0).sum()}개"
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(
        REPORT_DIR / "stt_error_analysis.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(
        error_details,
        columns=["파일명", "오류유형", "정답부분", "인식부분"]
    ).to_csv(
        REPORT_DIR / "stt_character_errors.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n분석 결과 저장 완료")


if __name__ == "__main__":
    main()
