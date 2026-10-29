"""STT preprocessing ablation: baseline, noise reduction, high-pass, peak normalize.
Place in pp project root and run: python test_stt_noise_filters.py
"""
import io
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, sosfiltfilt
from faster_whisper import WhisperModel
from jiwer import wer, cer
from src.audio import preprocess

BASE = Path(__file__).resolve().parent
AUDIO_PATH = BASE / 'T02_ADD_MEMO.wav'
REFERENCE = '회의 준비라고 메모해줘'
SR = 16000


def clean_text(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', text)).strip().rstrip('.?!,~ ')


def normalize_peak(x, peak=0.8):
    level = float(np.max(np.abs(x)))
    if level < 1e-8:
        return x.copy()
    return (x * (peak / level)).astype(np.float32)


def highpass(x, cutoff=80):
    sos = butter(3, cutoff, btype='highpass', fs=SR, output='sos')
    return sosfiltfilt(sos, x).astype(np.float32)


def main():
    if not AUDIO_PATH.is_file():
        raise FileNotFoundError(f'파일을 찾을 수 없습니다: {AUDIO_PATH}')

    try:
        import noisereduce as nr
    except ImportError as exc:
        raise SystemExit('잡음 제거 라이브러리가 없습니다. 먼저 실행: python -m pip install noisereduce') from exc

    processed_bytes, prep_info = preprocess(AUDIO_PATH.read_bytes(), check_duplicate=False)
    audio, sr = sf.read(io.BytesIO(processed_bytes), dtype='float32', always_2d=True)
    if sr != SR:
        raise ValueError(f'audio.py 출력 샘플레이트가 {sr}Hz입니다. 16000Hz가 필요합니다.')
    baseline = audio.mean(axis=1).astype(np.float32)

    # Stationary noise assumption is not required. Nonstationary spectral gating
    # estimates noise adaptively; it can still damage quiet speech consonants.
    reduced = nr.reduce_noise(y=baseline, sr=SR, stationary=False, prop_decrease=0.6)
    conditions = [
        ('A_기존전처리', baseline),
        ('B_잡음제거', np.asarray(reduced, dtype=np.float32)),
        ('C_저주파필터80Hz', highpass(baseline)),
        ('D_음량정규화', normalize_peak(baseline)),
    ]

    print('small / beam_size=6 / cpu_threads=2 / VAD OFF / initial_prompt 없음')
    print(f'파일: {AUDIO_PATH.name} / 정답: {REFERENCE}')
    print(f'길이: {len(baseline)/SR:.2f}초 / 전처리 구간: {prep_info.get("interval")}')
    print('모델 로딩 중...')
    model = WhisperModel('small', device='cpu', compute_type='int8', cpu_threads=2)
    reference = clean_text(REFERENCE)
    rows = []
    for name, signal in conditions:
        start = time.perf_counter()
        segments, _ = model.transcribe(
            np.ascontiguousarray(signal), language='ko', beam_size=6,
            vad_filter=False, condition_on_previous_text=False,
            temperature=0, initial_prompt=None,
        )
        prediction = clean_text(' '.join(s.text.strip() for s in segments))
        elapsed = time.perf_counter() - start
        row = {
            '방식': name, '인식 결과': prediction,
            'WER': round(wer(reference, prediction), 4),
            'CER': round(cer(reference, prediction), 4),
            'Latency(s)': round(elapsed, 3),
            'RTF': round(elapsed / (len(signal) / SR), 3),
        }
        rows.append(row)
        print(f'[{name}] {prediction}')
        print(f'  WER={row["WER"]:.4f} CER={row["CER"]:.4f} '
              f'Latency={row["Latency(s)"]:.3f}s RTF={row["RTF"]:.3f}')

    df = pd.DataFrame(rows)
    print('\n===== 잡음 제거 및 전처리 비교 =====')
    print(df.to_string(index=False))
    out = BASE / 'reports' / 'stt_noise_filters_comparison.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding='utf-8-sig')
    print(f'\n결과 저장: {out}')
    print('주의: Latency는 STT 전사 시간만 측정하며 전처리 계산 시간은 포함하지 않습니다.')


if __name__ == '__main__':
    main()
