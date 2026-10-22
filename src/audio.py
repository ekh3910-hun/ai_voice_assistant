from pathlib import Path
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


TARGET_SR = 16000


def load_audio(path):
    audio, sr = sf.read(path, dtype="float32")
    return audio, sr


def convert_to_mono(audio):
    if audio.ndim == 2:
        return audio.mean(axis=1)
    return audio


def resample_audio(audio, sr):
    if sr == TARGET_SR:
        return audio

    divisor = gcd(sr, TARGET_SR)

    return resample_poly(
        audio,
        TARGET_SR // divisor,
        sr // divisor
    ).astype(np.float32)


def trim_silence(audio, sr):
    """앞뒤의 긴 무음만 제거하고 여유 구간은 남긴다."""
    if len(audio) == 0:
        return audio

    peak = np.max(np.abs(audio))

    if peak < 1e-5:
        return audio

    threshold = max(peak * 0.01, 0.001)
    active = np.where(np.abs(audio) > threshold)[0]

    if len(active) == 0:
        return audio

    padding = int(sr * 0.2)

    start = max(0, active[0] - padding)
    end = min(len(audio), active[-1] + padding + 1)

    return audio[start:end]


def preprocess_audio(input_path, output_path):
    audio, sr = load_audio(input_path)

    original_duration = len(audio) / sr

    audio = convert_to_mono(audio)
    audio = resample_audio(audio, sr)
    audio = trim_silence(audio, TARGET_SR)

    sf.write(
        output_path,
        audio,
        TARGET_SR,
        subtype="PCM_16"
    )

    return original_duration, len(audio) / TARGET_SR


def main():
    root = Path(__file__).resolve().parent.parent
    input_dir = root / "data" / "audio_raw"
    output_dir = root / "data" / "audio_processed"

    output_dir.mkdir(parents=True, exist_ok=True)

    for input_path in sorted(input_dir.glob("*.wav")):
        output_path = output_dir / input_path.name

        before, after = preprocess_audio(
            input_path,
            output_path
        )

        print(
            f"{input_path.name}: "
            f"{before:.2f}초 → {after:.2f}초"
        )


if __name__ == "__main__":
    main()