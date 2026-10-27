# src/stt.py

import io
import logging
import re
import time
import unicodedata
from functools import lru_cache
from pathlib import Path

import soundfile as sf
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

from src.audio import preprocess


logger = logging.getLogger(__name__)


def report_progress(callback, message):
    logger.warning("[STT] %s", message)

    if callback is not None:
        callback(message)


@lru_cache(maxsize=1)
def load_stt_model(model_size="tiny"):
    return WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",
        cpu_threads=2,
        num_workers=1,
    )


def transcript_wav(
    audio,
    model_size="tiny",
    progress_callback=None,
):
    start = time.perf_counter()

    if isinstance(audio, (str, Path)):
        audio_bytes = Path(audio).read_bytes()
    elif isinstance(audio, (bytes, bytearray)):
        audio_bytes = bytes(audio)
    else:
        raise ValueError("파일 경로 또는 음성 bytes가 필요합니다.")

    if not audio_bytes:
        raise ValueError("녹음 데이터가 비어 있습니다.")

    if len(audio_bytes) > 10 * 1024 * 1024:
        raise ValueError("음성 파일은 10MB 이하여야 합니다.")

    report_progress(
        progress_callback,
        "1/4 녹음 형식 변환 중...",
    )

    decoded_audio = decode_audio(
        io.BytesIO(audio_bytes),
        sampling_rate=16000,
    )

    if decoded_audio.size == 0:
        raise ValueError("음성 데이터가 없습니다.")

    # 기존 프로젝트의 최대 녹음 길이 정책 적용
    if len(decoded_audio) / 16000 > 15:
        raise ValueError("15초 이내로 짧게 녹음해주세요.")

    wav_buffer = io.BytesIO()
    sf.write(
        wav_buffer,
        decoded_audio,
        16000,
        format="WAV",
        subtype="PCM_16",
    )

    report_progress(
        progress_callback,
        "2/4 음성 전처리 중...",
    )

    prepared_audio, _ = preprocess(
        wav_buffer.getvalue(),
        check_duplicate=False,
    )

    audio_data, _ = sf.read(
        io.BytesIO(prepared_audio),
        dtype="float32",
    )

    report_progress(
        progress_callback,
        f"3/4 {model_size} 모델 준비 중... "
        "(첫 실행은 모델 다운로드가 필요합니다)",
    )

    model = load_stt_model(model_size)

    report_progress(
        progress_callback,
        "4/4 한국어 음성 인식 중...",
    )

    segments, _ = model.transcribe(
        audio_data,
        language="ko",
        beam_size=1,
        vad_filter=False,
        condition_on_previous_text=False,
        temperature=0,
    )

    # segments를 순회해야 실제 전사가 완료됨
    transcript = " ".join(
        segment.text.strip()
        for segment in segments
    ).strip()

    text = unicodedata.normalize("NFC", transcript)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.rstrip(".?!,~ ")

    if not text:
        raise ValueError(
            "인식된 문장이 없습니다. "
            "마이크 가까이에서 다시 말씀해주세요."
        )

    elapsed = round(time.perf_counter() - start, 3)

    report_progress(
        progress_callback,
        f"인식 완료: {elapsed:.3f}초",
    )

    return text, elapsed


def evaluate_score(ref, pred):
    # 평가할 때만 평가 패키지 불러오기
    from jiwer import wer, cer

    return {
        "WER": round(wer(ref, pred), 4),
        "CER": round(cer(ref, pred), 4),
    }