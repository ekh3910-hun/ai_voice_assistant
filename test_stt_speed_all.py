
import io
import re
import time
import unicodedata
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from faster_whisper import WhisperModel
from jiwer import wer, cer

from src.audio import preprocess


# =========================================
# 1. 기본 설정
# =========================================

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
REPORT_DIR = BASE / "reports"

MODEL_SIZE = "base"
CPU_THREADS = 2
BEAM_SIZE = 6

SPEEDS = [0.90, 0.95, 1.00, 1.05, 1.10]

# 이전에 제공한 정답 문장
REFERENCES = {
    "ADD_MEMO.wav": "다음발표주제는 음성인식이라고 적어줘",
    "ADD_TASK1.wav": "오늘할일에 장보기 추가해줘",
    "ADD_TASK2.wav": "과제 제출을 다음주 월요일 할일로 등록해주세요",
    "LIST_MEMO.wav": "그저께 쓴 메모 알려줘",
    "LIST_TASK.wav": "완료한 할일목록 읽어줘",
    "LIST_TASK2.wav": "끝난 할일 목록 읽어줘",
    "T00_ADD_MEMO.wav": "메모에 우유사기라고 적어줘",
    "T01_ADD_MEMO.wav": "장보기 목록 메모해줘",
    "T02_ADD_MEMO.wav": "회의 준비라고 메모해줘",
    "T03_LIST_MEMO.wav": "저장된 메모 보여줘",
    "T04_ADD_TASK.wav": "오늘 할일에 운도 추가해줘",
    "T05_LIST_TASK.wav": "내일 할일 알려줘",
}


# =========================================
# 2. 텍스트 정규화
# =========================================

def normalize_text(text):
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"\s+", " ", text)
    return text.strip().rstrip(".?!,~ ")


# =========================================
# 3. 오디오 로딩 및 전처리
# =========================================

def load_audio(path):
    audio_bytes = path.read_bytes()

    processed_bytes, _ = preprocess(
        audio_bytes,
        check_duplicate=False
    )

    audio_data, sr = sf.read(
        io.BytesIO(processed_bytes),
        dtype="float32"
    )

    if audio_data.ndim == 2:
        audio_data = audio_data.mean(axis=1)

    if sr != 16000:
        raise ValueError(
            f"{path.name}: 예상하지 못한 샘플레이트 {sr}"
        )

    return np.asarray(audio_data, dtype=np.float32), sr


# =========================================
# 4. STT 실행
# =========================================

def transcribe(model, audio_data):
    start = time.perf_counter()

    segments, _ = model.transcribe(
        audio_data,
        language="ko",
        beam_size=BEAM_SIZE,
        temperature=0,
        vad_filter=False,
        initial_prompt=None,
        condition_on_previous_text=False
    )

    text = " ".join(
        segment.text.strip()
        for segment in segments
    )

    elapsed = time.perf_counter() - start

    return normalize_text(text), elapsed


# =========================================
# 5. 전체 WAV 평가
# =========================================

def main():
    print("=" * 65)
    print("base 모델 전체 음성 속도별 STT 성능 비교")
    print("=" * 65)

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"data 폴더가 없습니다: {DATA_DIR}"
        )

    wav_files = sorted(DATA_DIR.rglob("*.wav"))

    print(f"발견한 WAV 파일: {len(wav_files)}개")

    if not wav_files:
        print("WAV 파일이 없습니다.")
        return

    # 정답이 등록된 파일만 평가
    valid_files = []
    skipped = []

    for path in wav_files:
        if path.name in REFERENCES:
            valid_files.append(path)
        else:
            skipped.append(path)

    print(f"정답 등록 파일: {len(valid_files)}개")
    print(f"정답 미등록 파일: {len(skipped)}개")

    if skipped:
        print("\n[정답이 없어 제외한 파일]")
        for path in skipped:
            print(" -", path.relative_to(DATA_DIR))

    if not valid_files:
        print("평가할 파일이 없습니다.")
        return

    print("\n모델 로딩 중...")

    model = WhisperModel(
        MODEL_SIZE,
        device="cpu",
        compute_type="int8",
        cpu_threads=CPU_THREADS
    )

    print("모델 워밍업 중...")

    first_audio, _ = load_audio(valid_files[0])
    transcribe(model, first_audio)

    results = []
    failures = []

    for index, path in enumerate(valid_files, start=1):
        print(
            f"\n{'=' * 60}"
            f"\n[{index}/{len(valid_files)}] {path.name}"
        )

        reference = normalize_text(
            REFERENCES[path.name]
        )

        print("정답:", reference)

        try:
            audio_data, sr = load_audio(path)

            for speed in SPEEDS:
                # 피치를 유지하면서 속도 변경
                if speed == 1.0:
                    modified = audio_data.copy()
                else:
                    modified = librosa.effects.time_stretch(
                        audio_data,
                        rate=speed
                    )

                modified = np.asarray(
                    modified,
                    dtype=np.float32
                )

                duration = len(modified) / sr

                predicted, latency = transcribe(
                    model,
                    modified
                )

                wer_score = wer(reference, predicted)
                cer_score = cer(reference, predicted)

                results.append({
                    "파일명": str(path.relative_to(DATA_DIR)),
                    "속도": speed,
                    "정답": reference,
                    "인식 결과": predicted,
                    "WER": wer_score,
                    "CER": cer_score,
                    "Latency": latency,
                    "RTF": latency / duration,
                    "ExactMatch": int(predicted == reference)
                })

                print(
                    f"[{speed:.2f}배] {predicted}"
                    f" | WER={wer_score:.4f}"
                    f" CER={cer_score:.4f}"
                    f" Time={latency:.2f}s"
                )

        except Exception as e:
            failures.append({
                "파일명": str(path.relative_to(DATA_DIR)),
                "오류": str(e)
            })
            print(f"오류: {e}")

    if not results:
        print("\n평가 결과가 없습니다.")
        return

    # =====================================
    # 6. 전체 결과 집계
    # =====================================

    df = pd.DataFrame(results)

    summary = (
        df.groupby("속도", as_index=False)
        .agg(
            평균_WER=("WER", "mean"),
            평균_CER=("CER", "mean"),
            평균_시간=("Latency", "mean"),
            평균_RTF=("RTF", "mean"),
            정확히일치=("ExactMatch", "sum"),
            평가건수=("파일명", "count")
        )
        .sort_values(
            ["평균_CER", "평균_WER"]
        )
    )

    summary["완전일치율"] = (
        summary["정확히일치"] /
        summary["평가건수"]
    )

    print("\n" + "=" * 65)
    print("===== 전체 음성 속도별 최종 비교 =====")
    print("=" * 65)

    print(summary.round(4).to_string(index=False))

    # 파일별로 어떤 속도가 가장 좋았는지
    best_per_file = (
        df.sort_values(
            ["CER", "WER", "Latency"]
        )
        .drop_duplicates("파일명")
        .sort_values("파일명")
    )

    print("\n===== 파일별 최저 CER 속도 =====")
    print(
        best_per_file[
            ["파일명", "속도", "인식 결과", "WER", "CER"]
        ].round(4).to_string(index=False)
    )

    # =====================================
    # 7. CSV 저장
    # =====================================

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        REPORT_DIR / "stt_speed_all_details.csv",
        index=False,
        encoding="utf-8-sig"
    )

    summary.to_csv(
        REPORT_DIR / "stt_speed_all_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    best_per_file.to_csv(
        REPORT_DIR / "stt_speed_best_per_file.csv",
        index=False,
        encoding="utf-8-sig"
    )

    if failures:
        pd.DataFrame(failures).to_csv(
            REPORT_DIR / "stt_speed_failures.csv",
            index=False,
            encoding="utf-8-sig"
        )

    print("\n결과 저장 완료:", REPORT_DIR)

    if failures:
        print(f"처리 실패 파일: {len(failures)}개")


if __name__ == "__main__":
    main()
