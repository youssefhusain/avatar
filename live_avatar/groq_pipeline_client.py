# -*- coding: utf-8 -*-
"""
groq_pipeline_client.py
------------------------
بديل عن Gemini Live، بيستخدم Groq بالكامل:

    صوت المستخدم -> Whisper (STT) -> LLM نصي (Groq) -> Orpheus (TTS)

مهم: ده مش صوت-لصوت مباشر زي Gemini Live - دي 3 خطوات منفصلة.
Whisper و Orpheus مش streaming حقيقي (بياخدوا الصوت/النص كامل مرة
واحدة)، فالاستخدام العملي هو "دور بدور" (turn-based) مش تدفق مستمر:
تسجّل جملة -> تبعتها -> تاخد رد -> تحوله فيديو.

الأصوات العربية (Orpheus Arabic Saudi) محدودة بـ 200 حرف للطلب
الواحد، فالكود بيقسم أي رد طويل لجمل ويولّد كل جزء لوحده ويجمعهم.
"""

import io
import os
import wave
from dataclasses import dataclass
from typing import List, Optional

from groq import Groq

# --------------------------------------------------------------------------
GROQ_TTS_SAMPLE_RATE = 24000  # الفورمات اللي Orpheus بيرجعها كـ WAV


@dataclass
class TurnAudio:
    """نفس الشكل المستخدم مع Gemini، عشان باقي الكود (FlashTalk) يفضل زي ما هو."""
    pcm_bytes: bytes
    sample_rate: int = GROQ_TTS_SAMPLE_RATE
    text: str = ""


class GroqPipelineClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        stt_model: str = "whisper-large-v3-turbo",
        llm_model: str = "llama-3.3-70b-versatile",
        tts_model: str = "canopylabs/orpheus-arabic-saudi",
        tts_voice: str = "fahad",
        system_prompt: str = (
            "انت مساعد صوتي بيتكلم باللهجة المصرية العامية، "
            "ردودك قصيرة (جملة أو اتنين) وواضحة ومناسبة لأفاتار بيتكلم لايف."
        ),
    ):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "GROQ_API_KEY مش موجود. حطه في .env أو مرره لـ GroqPipelineClient(api_key=...)"
            )

        self.client = Groq(api_key=self.api_key)
        self.stt_model = stt_model
        self.llm_model = llm_model
        self.tts_model = tts_model
        self.tts_voice = tts_voice
        self.system_prompt = system_prompt

        self.history: List[dict] = [{"role": "system", "content": system_prompt}]

    # ---------------------------------------------------------------- STT
    def transcribe(self, wav_bytes: bytes, filename: str = "audio.wav") -> str:
        """يحول ملف صوت (WAV bytes) لنص عن طريق Whisper."""
        transcription = self.client.audio.transcriptions.create(
            file=(filename, wav_bytes),
            model=self.stt_model,
            language="ar",
            response_format="text",
        )
        # لما response_format="text"، الـ SDK بيرجع string مباشرة أو object فيه .text
        return transcription if isinstance(transcription, str) else transcription.text

    # ---------------------------------------------------------------- LLM
    def chat(self, user_text: str) -> str:
        """يبعت نص المستخدم لموديل Groq النصي، ويرجع رد المساعد كنص."""
        self.history.append({"role": "user", "content": user_text})

        completion = self.client.chat.completions.create(
            model=self.llm_model,
            messages=self.history,
            temperature=0.7,
            max_tokens=300,
        )
        reply = completion.choices[0].message.content
        self.history.append({"role": "assistant", "content": reply})
        return reply

    # ---------------------------------------------------------------- TTS
    def synthesize(self, text: str) -> TurnAudio:
        """
        يحول نص لصوت عن طريق Orpheus. بيقسم النص لجمل قصيرة (حد أقصى
        ~180 حرف للجملة العربية) ويجمع الصوت الناتج في ملف واحد.
        """
        chunks = self._split_text(text, max_len=180)
        pcm_parts = []
        sample_rate = GROQ_TTS_SAMPLE_RATE

        for chunk in chunks:
            response = self.client.audio.speech.create(
                model=self.tts_model,
                voice=self.tts_voice,
                input=chunk,
                response_format="wav",
            )
            wav_bytes = response.read() if hasattr(response, "read") else response.content
            pcm, sr = self._wav_bytes_to_pcm(wav_bytes)
            sample_rate = sr
            pcm_parts.append(pcm)

        return TurnAudio(pcm_bytes=b"".join(pcm_parts), sample_rate=sample_rate, text=text)

    # -------------------------------------------------------- دور كامل
    def run_turn(self, wav_bytes: Optional[bytes] = None, text: Optional[str] = None) -> TurnAudio:
        """
        دور محادثة كامل: صوت أو نص من المستخدم -> رد نصي -> رد صوتي.
        استخدم ده بدل استدعاء transcribe/chat/synthesize يدويًا.
        """
        if text is None:
            if wav_bytes is None:
                raise ValueError("لازم تمرر wav_bytes أو text")
            text = self.transcribe(wav_bytes)
            print(f"📝 المستخدم قال: {text}")

        reply_text = self.chat(text)
        return self.synthesize(reply_text)

    # ------------------------------------------------------------ Utils
    @staticmethod
    def _split_text(text: str, max_len: int = 180) -> List[str]:
        """يقسم نص طويل لجمل بحد أقصى max_len حرف (حد Orpheus Arabic)."""
        sentences = [s.strip() for s in text.replace("،", ".").split(".") if s.strip()]
        chunks, current = [], ""
        for s in sentences:
            candidate = (current + ". " + s).strip(". ") if current else s
            if len(candidate) > max_len:
                if current:
                    chunks.append(current)
                current = s
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks or [text[:max_len]]

    @staticmethod
    def _wav_bytes_to_pcm(wav_bytes: bytes):
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            sr = wf.getframerate()
            pcm = wf.readframes(wf.getnframes())
        return pcm, sr


def pcm_to_wav_bytes(pcm_bytes: bytes, sample_rate: int, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()
