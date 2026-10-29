# tests/test_stt_beam.py
# Beam Size별 Whisper STT 전사 성능(WER, CER, Latency, RTF) 평가 스크립트

import io
from pathlib import Path
import sys
import time

import pandas as pd
import soundfile as sf
from faster_whisper import WhisperModel
from jiwer import wer, cer

# 프로젝트 루트 경로 등록
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.audio import preprocess

AUDIO_DIR = Path(__file__).resolve().parent / "stt_test"
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

# 파일명과 실제 녹음한 문장 (데이터셋 commands_RE.csv / commands_RE.csv 기반 표준 정답 및 오타 수정 반영)
REFERENCES = {
    "ADD_MEMO.wav": "다음 발표 주제는 음성 인식이라고 적어 줘",
    "ADD_TASK1.wav": "오늘 할 일에 장보기 추가해 줘",
    "ADD_TASK2.wav": "과제 제출을 다음 주 월요일 할 일로 등록해 주세요",
    "LIST_MEMO.wav": "그저께 쓴 메모 알려 줘",
    "LIST_TASK.wav": "완료한 할 일 목록 읽어 줘",
    "LIST_TASK2.wav": "끝난 할 일 목록 읽어 줘",
    "T00_ADD_MEMO.wav": "메모에 우유 사기라고 적어 줘",
    "T01_ADD_MEMO.wav": "장보기 목록 메모해 줘",
    "T02_ADD_MEMO.wav": "회의 준비라고 메모해 줘",
    "T03_LIST_MEMO.wav": "저장된 메모 보여 줘",
    "T04_ADD_TASK.wav": "오늘 할 일에 운동 추가해 줘",  # '운도' 오타 -> '운동' 정상 수정
    "T05_LIST_TASK.wav": "내일 할 일 알려 줘",
}


def main():
    model = WhisperModel(
        "base",
        device="cpu",
        compute_type="int8",
        cpu_threads=2,
    )

    results = []

    for beam_size in range(1, 11):
        print(f"\n===== BEAM SIZE {beam_size} =====")

        for filename, reference in REFERENCES.items():
            path = AUDIO_DIR / filename

            if not path.exists():
                print(f"파일 없음: {path}")
                continue

            start = time.perf_counter()

            # 오디오 전처리 (PyAV 에러 방지 및 16kHz Mono Float32 변환)
            raw_bytes = path.read_bytes()
            prep_bytes, info = preprocess(raw_bytes, check_duplicate=False)
            audio_data, _ = sf.read(io.BytesIO(prep_bytes), dtype="float32")

            segments, transcribe_info = model.transcribe(
                audio_data,
                language="ko",
                beam_size=beam_size,
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 500,
                    "speech_pad_ms": 300,
                },
                condition_on_previous_text=False,
                temperature=0,
            )

            prediction = " ".join(
                segment.text.strip() for segment in segments
            ).strip()

            latency = time.perf_counter() - start
            duration = transcribe_info.duration

            result = {
                "beam_size": beam_size,
                "file": filename,
                "reference": reference,
                "prediction": prediction,
                "WER": wer(reference, prediction),
                "CER": cer(reference, prediction),
                "Latency": latency,
                "RTF": latency / duration if duration > 0 else 0,
            }

            results.append(result)

            print(f"파일: {filename}")
            print(f"정답: {reference}")
            print(f"인식: {prediction}")
            print(f"WER: {result['WER']:.2%}")
            print(f"CER: {result['CER']:.2%}")
            print(f"처리시간: {latency:.2f}초")

    if not results:
        print("테스트할 음성 파일이 없습니다.")
        return

    df = pd.DataFrame(results)

    summary = (
        df.groupby("beam_size", as_index=False)
        .agg(
            WER=("WER", "mean"),
            CER=("CER", "mean"),
            Latency=("Latency", "mean"),
            RTF=("RTF", "mean"),
        )
        .sort_values(["CER", "RTF"])
    )

    print("\n===== 최종 비교 결과 =====")
    print(summary.to_string(index=False))

    df.to_csv(
        REPORT_DIR / "stt_beam_details.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        REPORT_DIR / "stt_beam_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n결과 저장 완료!")


if __name__ == "__main__":
    main()
