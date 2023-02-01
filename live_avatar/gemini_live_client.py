# -*- coding: utf-8 -*-
"""
gemini_live_client.py
----------------------
غلاف بسيط حوالين Gemini Live API (Speech-To-Speech).
بياخد صوت المستخدم (مايك) ستريم، ويرجّع صوت رد Gemini كـ chunks من
raw PCM (16-bit, 24kHz mono) — نفس الفورمات اللي محتاجاها FlashTalk
بعد إعادة الـ resample لـ 16kHz.

الاستخدام الأساسي هو async generator: كل "دور" (turn) بيكلامك فيه
Gemini، بيرجعلك الصوت الكامل بتاعه لما يخلص، جاهز إنه يتحول لفيديو.
"""

import asyncio
import io
import os
import wave
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional

from google import genai
from google.genai import types

# --------------------------------------------------------------------------
# إعدادات ثابتة بتاعة الـ Live API (خدها زي ما هي - دي مواصفات Gemini نفسه)
# --------------------------------------------------------------------------
GEMINI_INPUT_SAMPLE_RATE = 16000   # اللي بنبعته لـ Gemini (مايك)
GEMINI_OUTPUT_SAMPLE_RATE = 24000  # اللي Gemini بيرجعه (صوت الرد)


@dataclass
class TurnAudio:
    """صوت رد واحد كامل من Gemini (دور واحد في المحادثة)."""
    pcm_bytes: bytes                 # raw PCM 16-bit mono @ 24kHz
    sample_rate: int = GEMINI_OUTPUT_SAMPLE_RATE
    text: str = ""                   # الترانسكريبت (لو محتاجه للّوج)


class GeminiLiveClient:
    """
    عميل يفتح جلسة Gemini Live، يبعتلها صوت المايك ستريم،
    ويطلعلك صوت الرد كامل لكل دور (turn) عن طريق async generator.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash-native-audio-preview-09-2025",
        voice_name: str = "Puck",
        system_instruction: str = (
            "انت مساعد صوتي بيتكلم باللهجة المصرية العامية، "
            "ردودك قصيرة وواضحة ومناسبة لأفاتار بيتكلم لايف."
        ),
        language_code: str = "ar-EG",
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY مش موجود. حطه في .env أو مرره لـ GeminiLiveClient(api_key=...)"
            )

        self.client = genai.Client(api_key=self.api_key)
        self.model = model

        self.config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_name
                    )
                ),
                language_code=language_code,
            ),
            system_instruction=types.Content(
                parts=[types.Part(text=system_instruction)]
            ),
            # بيخلي Gemini يبعتلك ترانسكريبت نص الرد جنب الصوت - مفيد للّوج
            output_audio_transcription=types.AudioTranscriptionConfig(),
        )

        self._session_cm = None
        self.session = None

    async def __aenter__(self):
        self._session_cm = self.client.aio.live.connect(
            model=self.model, config=self.config
        )
        self.session = await self._session_cm.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._session_cm:
            await self._session_cm.__aexit__(exc_type, exc, tb)

    async def send_audio_chunk(self, pcm16_chunk: bytes):
        """ابعت جزء صوت من المايك (raw PCM 16-bit mono @ 16kHz)."""
        await self.session.send_realtime_input(
            audio=types.Blob(
                data=pcm16_chunk,
                mime_type=f"audio/pcm;rate={GEMINI_INPUT_SAMPLE_RATE}",
            )
        )

    async def end_user_turn(self):
        """قول لـ Gemini إن المستخدم خلص كلامه (لو مش هتعتمد على VAD التلقائي)."""
        await self.session.send_realtime_input(audio_stream_end=True)

    async def receive_turns(self) -> AsyncGenerator[TurnAudio, None]:
        """
        Async generator: كل مرة Gemini يخلص رد كامل (turn)، بيرجع TurnAudio
        فيه كل الصوت مجمّع + الترانسكريبت. استخدمها في حلقة `async for`.
        """
        audio_buf = io.BytesIO()
        text_buf = []

        async for message in self.session.receive():
            if message.server_content is None:
                continue

            # صوت جزئي (chunk) - جمّعه
            if message.server_content.model_turn:
                for part in message.server_content.model_turn.parts:
                    if part.inline_data and part.inline_data.data:
                        audio_buf.write(part.inline_data.data)

            # ترانسكريبت جزئي
            if message.server_content.output_transcription:
                t = message.server_content.output_transcription.text
                if t:
                    text_buf.append(t)

            # خلص الدور بالكامل
            if message.server_content.turn_complete:
                pcm_bytes = audio_buf.getvalue()
                if pcm_bytes:
                    yield TurnAudio(
                        pcm_bytes=pcm_bytes,
                        sample_rate=GEMINI_OUTPUT_SAMPLE_RATE,
                        text="".join(text_buf),
                    )
                audio_buf = io.BytesIO()
                text_buf = []


def pcm_to_wav_bytes(pcm_bytes: bytes, sample_rate: int, channels: int = 1) -> bytes:
    """يحول raw PCM لملف WAV صحيح في الميموري (مطلوب لأي مكتبة resample/تحميل)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()
