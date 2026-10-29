# src/stt.py
"""Shared STT. Team app retains preprocessing; app1 can compare raw decoded audio."""
import io
import logging
import os
import re
import time
import unicodedata
from functools import lru_cache
from pathlib import Path
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

logger = logging.getLogger(__name__)

def report_progress(callback, message):
    logger.warning("[STT] %s", message)
    if callback is not None:
        callback(message)

DEFAULT_MODEL_SIZE = 'small'
COMMAND_CONTEXT = '한국어 메모, 할 일, 일정, 예약, 추가, 조회, 수정, 완료, 삭제, 오늘, 내일, 모레, 오전, 오후, 월요일.'


@lru_cache(maxsize=2)
def _load_stt_model(model_path):
    return WhisperModel(
        model_path, device="cpu", compute_type="int8",
        cpu_threads=min(4, os.cpu_count() or 1), num_workers=1,
    )


def load_stt_model(model_size=None):
    return _load_stt_model(os.getenv('WHISPER_MODEL_PATH') or model_size
                           or os.getenv('WHISPER_MODEL_SIZE', DEFAULT_MODEL_SIZE))


def transcript_wav(audio, model_size=None, progress_callback=None,
                   use_preprocess=False):
    started = time.perf_counter()
    if isinstance(audio, (str, Path)):
        audio_bytes = Path(audio).read_bytes()
    elif isinstance(audio, (bytes, bytearray)):
        audio_bytes = bytes(audio)
    else:
        raise ValueError("파일 경로 또는 음성 bytes가 필요합니다.")
    if not audio_bytes or len(audio_bytes) > 10 * 1024 * 1024:
        raise ValueError("비어 있지 않은 10MB 이하 음성이 필요합니다.")

    report_progress(progress_callback, "1/3 녹음 형식 변환 중...")
    data = decode_audio(io.BytesIO(audio_bytes), sampling_rate=16000)
    if not data.size or not np.isfinite(data).all():
        raise ValueError("유효한 음성 데이터가 없습니다.")
    duration = len(data) / 16000
    if not 0.5 <= duration <= 15:
        raise ValueError("녹음은 0.5초 이상 15초 이내로 해주세요.")
    if float(np.max(np.abs(data))) < 1e-6:
        raise ValueError("녹음이 무음입니다.")
    logger.warning("[STT] decoded %.3fs audio in %.3fs", duration, time.perf_counter()-started)

    if use_preprocess:
        report_progress(progress_callback, "기존 음성 전처리 중...")
        from src.audio import preprocess
        buf = io.BytesIO()
        sf.write(buf, data, 16000, format="WAV", subtype="PCM_16")
        prep_start = time.perf_counter()
        prepared, _ = preprocess(buf.getvalue(), check_duplicate=False)
        data, _ = sf.read(io.BytesIO(prepared), dtype="float32")
        logger.warning("[STT] preprocessing %.3fs", time.perf_counter()-prep_start)

    selected_model = model_size or os.getenv('WHISPER_MODEL_SIZE', DEFAULT_MODEL_SIZE)
    report_progress(progress_callback, f"2/3 {selected_model} 모델 준비 중...")
    model_start = time.perf_counter()
    model = load_stt_model(selected_model)
    logger.warning("[STT] model ready %.3fs", time.perf_counter()-model_start)
    report_progress(progress_callback, "3/3 한국어 음성 인식 중...")
    transcribe_start = time.perf_counter()
    segments, _ = model.transcribe(
        data, language="ko", beam_size=5,
        # Preserve quiet word boundaries; avoid trimming the signal twice.
        vad_filter=not use_preprocess,
        vad_parameters={'min_speech_duration_ms': 150, 'speech_pad_ms': 400},
        initial_prompt=COMMAND_CONTEXT,
        condition_on_previous_text=False, temperature=0,
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    text = re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip().rstrip(".?!,~ ")
    if not text:
        raise ValueError("인식된 문장이 없습니다. 녹음 원본을 확인해주세요.")
    elapsed = round(time.perf_counter()-started, 3)
    logger.warning("[STT] transcription %.3fs total %.3fs", time.perf_counter()-transcribe_start, elapsed)
    report_progress(progress_callback, f"인식 완료 ({elapsed:.3f}초)")
    return text, elapsed

def evaluate_score(ref, pred):
    from jiwer import wer, cer
    return {"WER": round(wer(ref, pred), 4), "CER": round(cer(ref, pred), 4)}
