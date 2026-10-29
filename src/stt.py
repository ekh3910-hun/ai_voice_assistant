# src/stt.py
import io
import re
import sys
import time
import unicodedata
from functools import lru_cache
from pathlib import Path

# 상위 폴더 경로 추가 (모듈 인식용)
sys.path.append(str(Path(__file__).resolve().parent.parent))

import soundfile as sf
from faster_whisper import WhisperModel
from jiwer import wer, cer

from src.audio import preprocess


# 1. 모델 불러오기
@lru_cache(maxsize=1)
def load_stt_model(model_size="base"):
    return WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=2)


# 2. 전사 처리
def transcript_wav(audio, model_size="base"):
    start = time.perf_counter()

    # 파일 경로 또는 bytes 처리
    if isinstance(audio, (str, Path)):
        audio_bytes = Path(audio).read_bytes()
    else:
        audio_bytes = audio

    # audio.py 전처리
    preprocess_audio, info = preprocess(audio_bytes, check_duplicate=False)

    # 16kHz float32 NumPy 배열 변환
    audio_data, _ = sf.read(io.BytesIO(preprocess_audio), dtype="float32")

    model = load_stt_model(model_size)

    # STT 전사 수행
    segments, info = model.transcribe(
        audio_data,
        language="ko",
        beam_size=6,
        vad_filter=False,
        condition_on_previous_text=False,
        temperature=0
    )

    transcript = " ".join(segment.text.strip() for segment in segments).strip()

    # 텍스트 정규화
    text = unicodedata.normalize("NFC", transcript)
    text = re.sub(r"\s+", " ", text).strip().rstrip(".?!,~ ")
    if not text:
        raise ValueError('인식된 문장이 없습니다. 다시 말씀해 주세요.')

    elapsed = round(time.perf_counter() - start, 3)

    return text, elapsed


# 3. wer, cer 반환
def evaluate_score(ref, pred):
    wer_score = wer(ref, pred)
    cer_score = cer(ref, pred)
    return {
        "WER": round(wer_score, 4),
        "CER": round(cer_score, 4)
    }
