
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


# ======================================
# 1. 기본 설정
# ======================================

BASE = Path(__file__).resolve().parent

AUDIO_PATH = BASE / "T02_ADD_MEMO.wav"

REPORT_PATH = (
    BASE / "reports" / "stt_speed_comparison.csv"
)

REFERENCE = "회의 준비라고 메모해줘"

MODEL_SIZE = "base"
CPU_THREADS = 2
BEAM_SIZE = 6

SPEEDS = [0.90, 0.95, 1.00, 1.05, 1.10]


# ======================================
# 2. 텍스트 정규화
# ======================================

def normalize_text(text):
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().rstrip(".?!,~ ")


# ======================================
# 3. 기존 audio.py 전처리
# ======================================

def load_audio():
    if not AUDIO_PATH.exists():
        raise FileNotFoundError(
            f"음성 파일을 찾을 수 없습니다: {AUDIO_PATH}"
        )

    audio_bytes = AUDIO_PATH.read_bytes()

    processed_bytes, info = preprocess(
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
            f"샘플링 주파수 오류: {sr}Hz"
        )

    return np.asarray(audio_data, dtype=np.float32), sr


# ======================================
# 4. STT 실행
# ======================================

def transcribe_audio(model, audio_data):
    start = time.perf_counter()

    segments, info = model.transcribe(
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


# ======================================
# 5. 속도 변경 및 평가
# ======================================

def main():
    print("=" * 60)
    print("base 모델 오디오 속도별 STT 비교")
    print("=" * 60)

    audio_data, sr = load_audio()

    original_duration = len(audio_data) / sr

    print(f"음성 파일: {AUDIO_PATH.name}")
    print(f"정답: {REFERENCE}")
    print(f"원본 길이: {original_duration:.2f}초")
    print(f"모델: {MODEL_SIZE}")
    print(f"CPU threads: {CPU_THREADS}")
    print(f"Beam size: {BEAM_SIZE}")

    print("\n모델 로딩 중...")

    model = WhisperModel(
        MODEL_SIZE,
        device="cpu",
        compute_type="int8",
        cpu_threads=CPU_THREADS
    )

    # 모델 워밍업
    print("워밍업 중...")
    transcribe_audio(model, audio_data)

    results = []
    reference = normalize_text(REFERENCE)

    print("\n===== 속도별 테스트 =====")

    for speed in SPEEDS:

        # 피치를 유지하면서 음성 속도 변경
        if speed == 1.0:
            modified_audio = audio_data.copy()
        else:
            modified_audio = librosa.effects.time_stretch(
                audio_data,
                rate=speed
            )

        modified_audio = np.asarray(
            modified_audio,
            dtype=np.float32
        )

        duration = len(modified_audio) / sr

        predicted, latency = transcribe_audio(
            model,
            modified_audio
        )

        wer_score = wer(reference, predicted)
        cer_score = cer(reference, predicted)

        rtf = latency / duration

        results.append({
            "속도": speed,
            "인식 결과": predicted,
            "WER": wer_score,
            "CER": cer_score,
            "Latency": latency,
            "RTF": rtf,
            "입력길이": duration,
            "ExactMatch": int(predicted == reference)
        })

        print(
            f"\n[속도 {speed:.2f}배]"
            f"\n인식: {predicted}"
            f"\nWER={wer_score:.4f}"
            f" CER={cer_score:.4f}"
            f" Latency={latency:.3f}s"
            f" RTF={rtf:.3f}"
        )

    # ==================================
    # 6. 결과 출력 및 저장
    # ==================================

    df = pd.DataFrame(results)

    df = df.sort_values(
        ["CER", "WER", "Latency"]
    )

    print("\n===== 속도별 최종 비교 =====")

    print(
        df.round(4).to_string(index=False)
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        REPORT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\n결과 저장 완료: {REPORT_PATH}")


if __name__ == "__main__":
    main()
