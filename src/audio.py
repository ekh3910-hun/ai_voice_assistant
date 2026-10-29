# audio.py
# AI 음성 비서 - 사용자 음성 입력 검사 및 STT 전처리

import hashlib
import io
import math
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal

TARGET_SR = 16000
TARGET_SUBTYPE = "PCM_16"
TOP_DB = 25
MAX_BYTES = 10 * 1024 * 1024
MIN_SEC = 0.5
MAX_SEC = 15.0
LOW_PEAK = 0.02
CLIP_RATIO = 0.001
RECENT_LIMIT = 100

_recent_hashes = {}


def _resample(mono: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """오디오 리샘플링 (scipy polyphase 필터 사용)"""
    if orig_sr == target_sr:
        return mono
    gcd = math.gcd(int(orig_sr), int(target_sr))
    up = target_sr // gcd
    down = orig_sr // gcd
    return signal.resample_poly(mono, up, down).astype(np.float32)


def _trim_silence(y: np.ndarray, top_db: float = TOP_DB, frame_length: int = 2048, hop_length: int = 512):
    """
    무음 구간 제거 (librosa/numba 의존성 없이 순수 NumPy로 동작)
    """
    if len(y) <= frame_length:
        return y, np.array([0, len(y)])

    num_frames = 1 + (len(y) - frame_length) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        y,
        shape=(num_frames, frame_length),
        strides=(y.strides[0] * hop_length, y.strides[0])
    )
    # RMS 에너지 및 상대 dB 계산
    rms = np.sqrt(np.mean(frames ** 2, axis=-1) + 1e-10)
    max_rms = float(np.max(rms))
    if max_rms < 1e-6:
        return y, np.array([0, len(y)])

    rms_db = 20 * np.log10(rms / max_rms + 1e-10)
    non_silent = np.where(rms_db >= -top_db)[0]

    if len(non_silent) == 0:
        return y, np.array([0, len(y)])

    start = max(0, int(non_silent[0] * hop_length))
    end = min(len(y), int(non_silent[-1] * hop_length + frame_length))
    return y[start:end], np.array([start, end])


# 검사
def validate_wav(audio):
    if not isinstance(audio, bytes) or not audio:
        raise ValueError("비어 있지 않은 WAV bytes가 필요합니다.")
    if len(audio) > MAX_BYTES:
        raise ValueError("녹음 파일은 10MB 이하여야 합니다.")

    try:
        wav = sf.info(io.BytesIO(audio))
    except Exception as exc:
        raise ValueError("WAV 파일을 읽을 수 없습니다.") from exc

    if wav.format not in ("WAV", "WAVEX"):
        raise ValueError("WAV 파일만 지원합니다.")
    if wav.channels not in (1, 2) or wav.samplerate <= 0 or wav.frames <= 0:
        raise ValueError("모노 또는 스테레오 WAV 파일이 필요합니다.")

    return {
        "hash_value": hashlib.sha256(audio).hexdigest(),
        "channels": wav.channels,
        "sample_rate": wav.samplerate,
        "duration": round(wav.frames / wav.samplerate, 3),
    }

# 모노/샘플링 레이트 통일, 무음구간 정리
def prepare(audio):
    try:
        y, sr = sf.read(io.BytesIO(audio), dtype="float32", always_2d=True)
    except Exception as exc:
        raise ValueError("음성 데이터를 읽을 수 없습니다.") from exc

    if y.size == 0 or not np.isfinite(y).all():
        raise ValueError("유효하지 않은 음성 데이터입니다.")

    clip_ratio = float(np.mean(np.abs(y) >= 0.999))
    mono = y.mean(axis=1, dtype=np.float32)
    resampled = _resample(mono, orig_sr=sr, target_sr=TARGET_SR)

    if resampled.size == 0 or not np.isfinite(resampled).all():
        raise ValueError("전처리된 음성 데이터가 유효하지 않습니다.")
    if np.max(np.abs(resampled)) < 1e-6:
        raise ValueError("소리가 들리지 않습니다. 다시 말씀해 주세요.")

    trimmed, interval = _trim_silence(resampled, top_db=TOP_DB)
    if trimmed.size == 0:
        raise ValueError("음성 구간을 찾지 못했습니다.")
    return trimmed, interval, clip_ratio

# 길이 음량 검사
def check_quality(trimmed, clip_ratio):
    sec = len(trimmed) / TARGET_SR
    if sec < MIN_SEC:
        raise ValueError("녹음이 너무 짧습니다. 다시 말씀해 주세요.")
    if sec > MAX_SEC:
        raise ValueError("녹음이 너무 깁니다. 짧게 말씀해 주세요.")

    warnings = []
    if float(np.max(np.abs(trimmed))) < LOW_PEAK:
        warnings.append("음량 낮음")
    if clip_ratio > CLIP_RATIO:
        warnings.append(f"클리핑 {clip_ratio:.1%}")
    return round(sec, 3), warnings

# 중복 확인
def check_recent(hash_value):
    if hash_value in _recent_hashes:
        raise ValueError("이미 처리한 녹음입니다.")
    _recent_hashes[hash_value] = True
    if len(_recent_hashes) > RECENT_LIMIT:
        _recent_hashes.pop(next(iter(_recent_hashes)))

# 전처리
def preprocess(audio, check_duplicate=True):
    info = validate_wav(audio)
    trimmed, interval, clip_ratio = prepare(audio)
    prepared_sec, warnings = check_quality(trimmed, clip_ratio)

    output = io.BytesIO()
    sf.write(output, trimmed, TARGET_SR, format="WAV", subtype=TARGET_SUBTYPE)
    result = output.getvalue()

    if check_duplicate:
        check_recent(info["hash_value"])

    info.update({
        "prepared_sec": prepared_sec,
        "interval": interval.tolist(),
        "warnings": warnings,
    })
    return result, info

# 중복 삭제
def delete_duplicate(audio_list):
    columns = ["filename", "status", "note", "hash_value", "channels", "sample_rate", "duration"]
    rows = []
    seen = {}

    for name, audio in audio_list:
        row = {"filename": name, "status": "ok", "note": ""}
        try:
            row.update(validate_wav(audio))
            if row["hash_value"] in seen:
                row["status"] = "duplicate"
                row["note"] = f'{seen[row["hash_value"]]}와 같은 파일'
            else:
                seen[row["hash_value"]] = name
        except ValueError as exc:
            row.update({"status": "error", "note": str(exc)})
        rows.append(row)

    return pd.DataFrame(rows, columns=columns)