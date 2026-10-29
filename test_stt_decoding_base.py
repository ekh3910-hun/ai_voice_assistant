
import io
import time
import unicodedata
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from faster_whisper import WhisperModel
from jiwer import wer, cer

from src.audio import preprocess


# ==========================================
# 1. 기본 설정
# ==========================================

BASE = Path(__file__).resolve().parent

AUDIO_PATH = BASE / "T02_ADD_MEMO.wav"
REPORT_PATH = BASE / "reports" / "stt_base_decoding_comparison.csv"

REFERENCE = "회의 준비라고 메모해줘"

MODEL_SIZE = "base"
CPU_THREADS = 2
BEAM_SIZE = 6

# temperature=0에서는 best_of가 사용되지 않음
TEST_CONFIGS = [
    {"name": "A_기본", "temperature": 0.0, "best_of": 1, "repeat": 1},
    {"name": "B_샘플링1", "temperature": 0.2, "best_of": 1, "repeat": 3},
    {"name": "C_샘플링5", "temperature": 0.2, "best_of": 5, "repeat": 3},
]


# ==========================================
# 2. 텍스트 정규화
# ==========================================

def normalize_text(text):
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().rstrip(".?!,~ ")


# ==========================================
# 3. 음성 전처리
# ==========================================

def load_audio():
    if not AUDIO_PATH.exists():
        raise FileNotFoundError(
            f"음성 파일을 찾을 수 없습니다: {AUDIO_PATH}"
        )

    audio_bytes = AUDIO_PATH.read_bytes()

    # 기존 프로젝트의 audio.py 전처리 사용
    processed_bytes, info = preprocess(
        audio_bytes,
        check_duplicate=False
    )

    audio_data, sample_rate = sf.read(
        io.BytesIO(processed_bytes),
        dtype="float32"
    )

    if audio_data.ndim == 2:
        audio_data = audio_data.mean(axis=1)

    if sample_rate != 16000:
        raise ValueError(
            f"전처리 후 샘플링 주파수가 16000Hz가 아닙니다: {sample_rate}"
        )

    duration = len(audio_data) / sample_rate

    return np.asarray(audio_data, dtype=np.float32), duration


# ==========================================
# 4. STT 실행
# ==========================================

def transcribe_once(model, audio_data, config):
    start = time.perf_counter()

    segments, info = model.transcribe(
        audio_data,
        language="ko",
        beam_size=BEAM_SIZE,
        best_of=config["best_of"],
        temperature=config["temperature"],
        vad_filter=False,
        initial_prompt=None,
        condition_on_previous_text=False,
    )

    # faster-whisper는 segments를 순회해야
    # 실제 전사가 완료됨
    text = " ".join(
        segment.text.strip()
        for segment in segments
    )

    latency = time.perf_counter() - start

    return normalize_text(text), latency


# ==========================================
# 5. 평가 실행
# ==========================================

def main():
    print("=" * 60)
    print("base 모델 디코딩 설정 비교")
    print("=" * 60)

    print(f"음성 파일: {AUDIO_PATH.name}")
    print(f"정답: {REFERENCE}")
    print(f"모델: {MODEL_SIZE}")
    print(f"cpu_threads: {CPU_THREADS}")
    print(f"beam_size: {BEAM_SIZE}")
    print("VAD: OFF")
    print("initial_prompt: 없음")

    audio_data, duration = load_audio()

    print(f"\n음성 길이: {duration:.2f}초")

    print("\n모델 로딩 중...")

    model = WhisperModel(
        MODEL_SIZE,
        device="cpu",
        compute_type="int8",
        cpu_threads=CPU_THREADS
    )

    # 워밍업: 첫 실행의 초기화 비용을 측정에서 제외
    print("워밍업 중...")
    warmup_config = TEST_CONFIGS[0]
    transcribe_once(model, audio_data, warmup_config)

    results = []
    reference = normalize_text(REFERENCE)

    print("\n===== 개별 테스트 결과 =====")

    for config in TEST_CONFIGS:
        for trial in range(1, config["repeat"] + 1):

            predicted, latency = transcribe_once(
                model,
                audio_data,
                config
            )

            wer_score = wer(reference, predicted)
            cer_score = cer(reference, predicted)
            rtf = latency / duration

            results.append({
                "설정": config["name"],
                "temperature": config["temperature"],
                "best_of": config["best_of"],
                "회차": trial,
                "인식 결과": predicted,
                "WER": wer_score,
                "CER": cer_score,
                "Latency": latency,
                "RTF": rtf,
                "ExactMatch": int(predicted == reference)
            })

            print(
                f"\n[{config['name']} / {trial}회]"
                f"\n인식: {predicted}"
                f"\nWER={wer_score:.4f}"
                f" CER={cer_score:.4f}"
                f" Latency={latency:.3f}s"
                f" RTF={rtf:.3f}"
            )

    # ======================================
    # 6. 결과 정리
    # ======================================

    df = pd.DataFrame(results)

    summary = (
        df.groupby(
            ["설정", "temperature", "best_of"],
            as_index=False
        )
        .agg(
            WER=("WER", "mean"),
            CER=("CER", "mean"),
            Latency=("Latency", "mean"),
            RTF=("RTF", "mean"),
            ExactMatch=("ExactMatch", "mean"),
            실행횟수=("회차", "count")
        )
        .sort_values(["CER", "WER", "Latency"])
    )

    print("\n===== base 디코딩 설정 비교 (평균) =====")
    print(summary.round(4).to_string(index=False))

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        REPORT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    summary_path = (
        REPORT_PATH.parent /
        "stt_base_decoding_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n결과 저장 완료!")
    print(f"개별 결과: {REPORT_PATH}")
    print(f"평균 결과: {summary_path}")


if __name__ == "__main__":
    main()
