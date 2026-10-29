# src/tts.py

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import edge_tts

VOICES = {
    "선희(여성)": "ko-KR-SunHiNeural",
    "인준(남성)": "ko-KR-InJoonNeural",
    "현수(남성)": "ko-KR-HyunsuNeural"
}

tts_record = []


async def generate(text, voice):
    communicate = edge_tts.Communicate(text=text, voice=voice)

    audio = bytearray()

    async for chunk in communicate.stream():
        if chunk['type'] == 'audio':
            audio.extend(chunk['data'])

    return bytes(audio)


def text_to_speech(text, voice='ko-KR-SunHiNeural'):

    start = time.perf_counter()
    try:
        if not isinstance(text, str) or not text.strip():
            raise ValueError('변환할 텍스트가 없습니다')

        if voice not in VOICES.values():
            if voice in VOICES:
                voice = VOICES[voice]
            else:
                raise ValueError('지원하지않는 목소리입니다')

        # 이모지 제거 (음성 합성 오류 방지)
        clean_text = text.strip()
        for emoji in ["✅", "❌", "📌", "📅", "📝", "⚡"]:
            clean_text = clean_text.replace(emoji, "")

        if not clean_text.strip():
            raise ValueError('변환할 텍스트가 없습니다')
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            audio = asyncio.run(generate(clean_text.strip(), voice))
        else:
            # 실행 중인 이벤트 루프와 분리된 스레드에서 합성합니다.
            with ThreadPoolExecutor(max_workers=1) as pool:
                audio = pool.submit(lambda: asyncio.run(generate(clean_text.strip(), voice))).result()

        if not audio:
            raise ValueError('음성 데이터가 없습니다')

        elapsed = time.perf_counter() - start

        tts_record.append({
            'voice': voice,
            'elapsed': elapsed,
            'success': True
        })

        return audio

    except Exception:
        elapsed = time.perf_counter() - start

        tts_record.append({
            'voice': voice,
            'elapsed': elapsed,
            'success': False
        })
        raise


def evaluate_tts():
    total = len(tts_record)
    fail = sum(not r['success'] for r in tts_record)

    return {
        'total': total,
        'failures': fail,
        'failure_rate': (fail / total * 100) if total else 0.0,
        'avg_elapsed_sec': (
            sum(r['elapsed'] for r in tts_record) / total
            if total else 0.0
        )
    }


if __name__ == "__main__":
    sample_text = "내일 3시 치과 예약 일정을 등록했습니다."
    print(f"TTS 변환 테스트 시작: '{sample_text}'")
    for name, v_id in VOICES.items():
        data = text_to_speech(sample_text, voice=v_id)
        print(f"[{name}] 성공: {len(data):,} bytes")
    print("평가 결과:", evaluate_tts())
