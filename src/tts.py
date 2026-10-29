# tts.py

import asyncio
import edge_tts

VOICES = {
    '선희(여성)': 'ko-KR-SunHiNeural',
    '인준(남성)': 'ko-KR-InJoonNeural',
    '현수(남성)': 'ko-KR-HyunsuNeural'
}


async def generate(text, voice):
    communicate = edge_tts.Communicate(text=text, voice=voice)

    audio = bytearray()

    async for chunk in communicate.stream():
        if chunk['type'] == 'audio':
            audio.extend(chunk['data'])

    return bytes(audio)

def text_to_speech(text, voice='ko-KR-SunHiNeural'):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('변환할 텍스트가 없습니다')

    if voice not in VOICES.values():
        raise ValueError('지원하지않는 목소리입니다')

    return asyncio.run(generate(text.strip(), voice))