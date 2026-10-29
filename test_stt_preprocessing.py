"""Compare original, peak-normalized, and project's audio.py preprocessing.
Run from the pp project root: python test_stt_preprocessing.py
"""
import io
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd
from faster_whisper import WhisperModel
from jiwer import wer, cer

from src.audio import preprocess

BASE = Path(__file__).resolve().parent
AUDIO_PATH = BASE / 'T02_ADD_MEMO.wav'
REFERENCE = '회의 준비라고 메모해줘'
MODEL_SIZE = 'small'
BEAM_SIZE = 6
CPU_THREADS = 2


def normalize_text(text):
    text = unicodedata.normalize('NFC', text)
    return re.sub(r'\s+', ' ', text).strip().rstrip('.?!,~ ')


def load_audio_16k(raw_bytes):
    audio, sr = sf.read(io.BytesIO(raw_bytes), dtype='float32', always_2d=True)
    mono = audio.mean(axis=1).astype(np.float32)
    if sr != 16000:
        divisor = gcd(sr, 16000)
        mono = resample_poly(mono, 16000 // divisor, sr // divisor).astype(np.float32)
    return mono


def peak_normalize(audio, target_peak=0.8):
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-8:
        return audio.copy()
    return (audio * (target_peak / peak)).astype(np.float32)


def main():
    if not AUDIO_PATH.exists():
        raise FileNotFoundError(f'음성 파일을 찾을 수 없습니다: {AUDIO_PATH}\n'
                                '테스트 WAV를 pp 폴더에 복사하거나 AUDIO_PATH를 수정하세요.')

    raw_bytes = AUDIO_PATH.read_bytes()
    original = load_audio_16k(raw_bytes)
    normalized = peak_normalize(original)
    processed_bytes, prep_info = preprocess(raw_bytes, check_duplicate=False)
    processed, processed_sr = sf.read(io.BytesIO(processed_bytes), dtype='float32')
    if processed_sr != 16000:
        raise ValueError(f'전처리 출력 샘플레이트 오류: {processed_sr}')

    print(f'모델={MODEL_SIZE}, beam_size={BEAM_SIZE}, cpu_threads={CPU_THREADS}')
    print(f'원본 길이={len(original)/16000:.2f}초, 전처리 길이={len(processed)/16000:.2f}초')
    print(f'전처리 무음 제거 구간(16kHz 샘플)={prep_info.get("interval")}')
    print(f'정답: {REFERENCE}')
    print('모델 로딩 중... (첫 실행에는 다운로드 시간이 걸릴 수 있습니다)')
    model = WhisperModel(MODEL_SIZE, device='cpu', compute_type='int8', cpu_threads=CPU_THREADS)

    conditions = [
        ('원본', original),
        ('음량 정규화', normalized),
        ('기존 audio.py 전처리', np.asarray(processed, dtype=np.float32)),
    ]
    rows = []
    for name, audio in conditions:
        start = time.perf_counter()
        segments, _ = model.transcribe(
            audio, language='ko', beam_size=BEAM_SIZE,
            vad_filter=False, condition_on_previous_text=False,
            temperature=0,
        )
        prediction = normalize_text(' '.join(s.text.strip() for s in segments))
        elapsed = time.perf_counter() - start
        duration = len(audio) / 16000
        rows.append({
            '방식': name,
            '인식 결과': prediction,
            'WER': round(wer(normalize_text(REFERENCE), prediction), 4),
            'CER': round(cer(normalize_text(REFERENCE), prediction), 4),
            'Latency(s)': round(elapsed, 3),
            'RTF': round(elapsed / duration, 3),
            '입력길이(s)': round(duration, 3),
        })
        print(f'\n[{name}] {prediction}\nWER={rows[-1]["WER"]:.4f}, CER={rows[-1]["CER"]:.4f}, '
              f'Latency={elapsed:.3f}s, RTF={rows[-1]["RTF"]:.3f}')

    df = pd.DataFrame(rows)
    print('\n===== 전처리 방식별 비교 =====')
    print(df.to_string(index=False))
    output = BASE / 'reports' / 'stt_preprocessing_comparison.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, encoding='utf-8-sig')
    print(f'\n결과 저장: {output}')


if __name__ == '__main__':
    main()
