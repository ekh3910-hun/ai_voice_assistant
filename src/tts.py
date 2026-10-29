# tts.py

import asyncio
import os
from time import perf_counter
import edge_tts

VOICES = {
    '선희(여성)': 'ko-KR-SunHiNeural',
    '인준(남성)': 'ko-KR-InJoonNeural',
    '현수(남성)': 'ko-KR-HyunsuNeural'
}


async def generate(text, voice):
    communicate = edge_tts.Communicate(
        text=text, voice=voice,
        proxy=os.getenv('HTTPS_PROXY') or os.getenv('https_proxy'),
        connect_timeout=10, receive_timeout=20,
    )

    audio = bytearray()

    async for chunk in communicate.stream():
        if chunk['type'] == 'audio':
            audio.extend(chunk['data'])

    if not audio:
        raise RuntimeError('음성 서비스에서 오디오를 받지 못했습니다')
    return bytes(audio)


async def _generate_with_timeout(text, voice):
    return await asyncio.wait_for(generate(text, voice), timeout=30)

def text_to_speech(text, voice='ko-KR-SunHiNeural'):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('변환할 텍스트가 없습니다')

    if voice not in VOICES.values():
        raise ValueError('지원하지않는 목소리입니다')

    return asyncio.run(_generate_with_timeout(text.strip(), voice))


def synthesize_speech(text, voice='ko-KR-SunHiNeural'):
    """Return MP3 bytes and elapsed seconds for the Streamlit applications."""
    started = perf_counter()
    return text_to_speech(text, voice), round(perf_counter() - started, 3)
