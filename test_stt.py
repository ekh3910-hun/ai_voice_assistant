# test_stt.py
# 전처리된 WAV 6개 STT 테스트

from pathlib import Path
import pandas as pd

from src.stt import transcribe_audio


def main():
    audio_dir = Path("data/audio_processed")
    files = sorted(audio_dir.glob("*.wav"))[:6]

    if not files:
        print("WAV 파일이 없습니다.")
        return

    results = []

    for file_path in files:
        print(f"\n처리 중: {file_path.name}")

        try:
            text, elapsed = transcribe_audio(file_path)

            results.append({
                "filename": file_path.name,
                "stt_text": text,
                "elapsed_sec": round(elapsed, 2)
            })

            print("인식 결과:", text)
            print(f"처리 시간: {elapsed:.2f}초")

        except Exception as exc:
            print("오류:", exc)
            results.append({
                "filename": file_path.name,
                "stt_text": "",
                "elapsed_sec": None
            })

    output = Path("reports/stt_results.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(results).to_csv(
        output,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nSTT 결과 저장 완료:", output)


if __name__ == "__main__":
    main()