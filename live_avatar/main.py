# -*- coding: utf-8 -*-
"""
main.py
-------
شغّل ده عشان تكلم الأفاتار لايف:

    python live_avatar/main.py

الـ pipeline:
    مايكروفون
      -> Gemini Live (Speech-To-Speech، بيسمعك ويرد بصوت)
      -> صوت الرد الكامل لكل دور (turn)
      -> FlashTalk (بيحرك صورة الأفاتار بصوت الرد)
      -> فيديو mp4 محفوظ في live_avatar/output/ (ويتفتح تلقائي)

شرط أساسي: لازم يكون عندك GPU بمواصفات كافية (64GB+ VRAM لكارت
واحد مع --cpu_offload، أو 8xH800 للسرعة الحقيقية real-time) وموديلات
FlashTalk متنزّلة زي ما موضح في README.md الأصلي.
"""

import asyncio
import os
import subprocess
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_avatar.gemini_live_client import GeminiLiveClient, pcm_to_wav_bytes
from live_avatar.flashtalk_bridge import FlashTalkBridge, make_output_path
from live_avatar.audio_io import mic_stream

load_dotenv()


def open_video(path: str):
    """يفتح الفيديو تلقائي بعد ما يتولّد (Linux/Mac/Windows)."""
    try:
        if sys.platform.startswith("linux"):
            subprocess.Popen(["xdg-open", path])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        elif sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore
    except Exception as e:
        print(f"[تنبيه] مقدرتش أفتح الفيديو تلقائي: {e}")


async def run():
    cfg = dict(
        ckpt_dir=os.environ["FLASHTALK_CKPT_DIR"],
        wav2vec_dir=os.environ["WAV2VEC_DIR"],
        avatar_image=os.environ["AVATAR_IMAGE"],
    )
    output_dir = os.environ.get("OUTPUT_DIR", "./live_avatar/output")

    print("جاري تحميل موديل FlashTalk... (خطوة واحدة، هتاخد وقت)")
    bridge = FlashTalkBridge(**cfg)

    model = os.environ.get(
        "GEMINI_LIVE_MODEL", "gemini-2.5-flash-native-audio-preview-09-2025"
    )
    voice = os.environ.get("GEMINI_VOICE_NAME", "Puck")

    async with GeminiLiveClient(model=model, voice_name=voice) as gemini:
        print("جاهز! اتكلم في المايك... (Ctrl+C للخروج)")

        async def sender():
            async for chunk in mic_stream(sample_rate=16000):
                await gemini.send_audio_chunk(chunk)

        async def receiver():
            async for turn in gemini.receive_turns():
                print(f"\n🤖 Gemini: {turn.text}")
                wav_bytes = pcm_to_wav_bytes(turn.pcm_bytes, turn.sample_rate)

                out_path = make_output_path(output_dir)
                print("جاري تحريك الأفاتار بالصوت ده...")
                loop = asyncio.get_event_loop()
                # التوليد تقيل (GPU-bound) - شغّله في executor عشان
                # ميوقفش استقبال صوت المايك في نفس الوقت
                await loop.run_in_executor(
                    None, bridge.audio_bytes_to_video, wav_bytes, out_path
                )
                print(f"الفيديو جاهز: {out_path}")
                open_video(out_path)

        await asyncio.gather(sender(), receiver())


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nتم الإيقاف.")
