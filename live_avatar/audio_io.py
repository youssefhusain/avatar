# -*- coding: utf-8 -*-
"""
audio_io.py
-----------
تسجيل من المايك وستريم الصوت كـ chunks، باستخدام sounddevice.
لو مفيش مايك متاح (سيرفر بدون صوت)، استخدم `text_input_loop` بدل
`mic_stream` وابعت نص للاختبار (تجاوز الـ STT بالكامل مؤقتًا).
"""

import asyncio
import queue

import numpy as np
import sounddevice as sd

CHUNK_MS = 100  # حجم كل جزء صوت مبعوت لـ Gemini (بالميلي ثانية)


async def mic_stream(sample_rate: int = 16000, chunk_ms: int = CHUNK_MS):
    """
    Async generator بيرجع chunks من الصوت (raw PCM 16-bit) من المايك،
    ستريم مباشر، لحد ما توقف البرنامج (Ctrl+C).
    """
    block_size = int(sample_rate * chunk_ms / 1000)
    q: "queue.Queue[bytes]" = queue.Queue()

    def _callback(indata, frames, time_info, status):
        if status:
            print(f"[mic] status: {status}")
        q.put(bytes(indata))

    stream = sd.RawInputStream(
        samplerate=sample_rate,
        blocksize=block_size,
        dtype="int16",
        channels=1,
        callback=_callback,
    )
    stream.start()
    try:
        loop = asyncio.get_event_loop()
        while True:
            chunk = await loop.run_in_executor(None, q.get)
            yield chunk
    finally:
        stream.stop()
        stream.close()


def play_pcm(pcm_bytes: bytes, sample_rate: int):
    """تشغيل صوت مباشر من السماعة (اختياري - للتجربة السريعة بدون فيديو)."""
    audio = np.frombuffer(pcm_bytes, dtype=np.int16)
    sd.play(audio, samplerate=sample_rate, blocking=True)
