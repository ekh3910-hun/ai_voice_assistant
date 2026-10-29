
from pathlib import Path
import re
import unicodedata

import pandas as pd
from jiwer import wer, cer


BASE = Path(__file__).resolve().parent

INPUT_PATH = BASE / "reports" / "stt_speed_all_details.csv"
OUTPUT_PATH = BASE / "reports" / "stt_spacing_comparison.csv"


def normalize(text):
    text = unicodedata.normalize("NFC", str(text))
    return text.strip()


def remove_spaces(text):
    text = normalize(text)
    return re.sub(r"\s+", "", text)


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"결과 파일이 없습니다: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")

    # 원본 속도(1.00배) 결과만 사용
    df = df[
        (df["속도"] - 1.0).abs() < 0.0001
    ].copy()

    results = []

    for _, row in df.iterrows():
        reference = normalize(row["정답"])
        prediction = normalize(row["인식 결과"])

        ref_no_space = remove_spaces(reference)
        pred_no_space = remove_spaces(prediction)

        results.append({
            "파일명": row["파일명"],
            "정답": reference,
            "인식 결과": prediction,

            # 기존 단어 단위 오류율
            "기존_WER": wer(reference, prediction),

            # 기존 문자 단위 오류율
            "기존_CER": cer(reference, prediction),

            # 띄어쓰기를 제외한 문자 오류율
            "띄어쓰기제외_CER": cer(
                ref_no_space,
                pred_no_space
            ),

            "띄어쓰기제외_완전일치": int(
                ref_no_space == pred_no_space
            )
        })

    result_df = pd.DataFrame(results)

    print("\n===== 띄어쓰기 영향 비교 =====")

    print(
        result_df[
            [
                "파일명",
                "기존_WER",
                "기존_CER",
                "띄어쓰기제외_CER"
            ]
        ].round(4).to_string(index=False)
    )

    print("\n===== 전체 평균 =====")

    print(
        result_df[
            [
                "기존_WER",
                "기존_CER",
                "띄어쓰기제외_CER"
            ]
        ].mean().round(4).to_string()
    )

    print(
        "\n띄어쓰기 제외 완전일치:",
        result_df["띄어쓰기제외_완전일치"].sum(),
        "/",
        len(result_df)
    )

    result_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\n결과 저장: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
