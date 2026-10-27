# audio.py
# AI 음성 비서 - 사용자 음성 입력 검사 및 STT 전처리

import hashlib
import io
import librosa
import numpy as np
import pandas as pd
import soundfile as sf

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
    resampled = (
        librosa.resample(mono, orig_sr=sr, target_sr=TARGET_SR)
        if sr != TARGET_SR else mono
    )

    if resampled.size == 0 or not np.isfinite(resampled).all():
        raise ValueError("전처리된 음성 데이터가 유효하지 않습니다.")
    if np.max(np.abs(resampled)) < 1e-6:
        raise ValueError("소리가 들리지 않습니다. 다시 말씀해 주세요.")

    trimmed, interval = librosa.effects.trim(resampled, top_db=TOP_DB)
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