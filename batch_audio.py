
# batch_audio.py
# 수집한 WAV 파일 일괄 전처리

from pathlib import Path
import pandas as pd

from src.audio import preprocess, delete_duplicate


def main():
    project_root = Path(__file__).resolve().parent

    input_dir = project_root / "data" / "audio_raw"
    output_dir = project_root / "data" / "audio_processed"
    report_dir = project_root / "reports"

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.wav"))

    if not files:
        print("전처리할 WAV 파일이 없습니다.")
        return

    # 1. 중복 검사
    audio_list = [(path.name, path.read_bytes()) for path in files]
    duplicate_df = delete_duplicate(audio_list)

    duplicate_df.to_csv(
        report_dir / "duplicate_report.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 2. 전처리
    results = []

    for path in files:
        row = duplicate_df.loc[
            duplicate_df["filename"] == path.name
        ].iloc[0]

        if row["status"] != "ok":
            print(f"[제외] {path.name}: {row['note']}")
            results.append({
                "filename": path.name,
                "status": row["status"],
                "note": row["note"]
            })
            continue

        try:
            wav_bytes, info = preprocess(
                path.read_bytes(),
                check_duplicate=False
            )

            (output_dir / path.name).write_bytes(wav_bytes)

            results.append({
                "filename": path.name,
                "status": "ok",
                "duration": info["prepared_sec"],
                "sample_rate": 16000,
                "warnings": ", ".join(info["warnings"])
            })

            print(f"[성공] {path.name}")

        except ValueError as exc:
            results.append({
                "filename": path.name,
                "status": "error",
                "note": str(exc)
            })
            print(f"[오류] {path.name}: {exc}")

    # 3. 결과 저장
    pd.DataFrame(results).to_csv(
        report_dir / "preprocess_report.csv",
        index=False,
        encoding="utf-8-sig"
    )

    success = sum(r["status"] == "ok" for r in results)

    print("\n=== 전처리 결과 ===")
    print(f"전체 파일: {len(files)}개")
    print(f"성공: {success}개")
    print(f"제외/오류: {len(files) - success}개")


if __name__ == "__main__":
    main()
