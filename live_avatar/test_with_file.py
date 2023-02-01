# -*- coding: utf-8 -*-
"""
test_with_file.py
------------------
نسخة اختبار بدون مايك حي: بتاخد رسالة نصية (أو ملف صوت جاهز)،
تبعتها لمزود الصوت/LLM اللي تختاره، وتاخد الرد الصوتي، وتولّد
فيديو الأفتار بيه.

الاستخدام:
    # Groq (الافتراضي) - STT + LLM + TTS كلهم من Groq
    python live_avatar/test_with_file.py --text "أهلاً، عرفني بنفسك"
    python live_avatar/test_with_file.py --audio_file my_question.wav

    # Gemini Live (صوت لصوت مباشر)
    python live_avatar/test_with_file.py --provider gemini --text "أهلاً"
"""

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_avatar.flashtalk_bridge import FlashTalkBridge, make_output_path

load_dotenv()


def _load_wav_as_pcm16(path: str, target_sr: int):
    import librosa
    import numpy as np

    audio, _ = librosa.load(path, sr=target_sr, mono=True)
    audio_int16 = (audio * 32767).astype(np.int16)
    return audio_int16.tobytes()


def run_groq(text: str, audio_file: str, bridge: FlashTalkBridge, output_dir: str):
    """المسار البسيط: كل حاجة sync، مفيش async محتاج ليه هنا."""
    from live_avatar.groq_pipeline_client import GroqPipelineClient, pcm_to_wav_bytes

    client = GroqPipelineClient(
        stt_model=os.environ.get("GROQ_STT_MODEL", "whisper-large-v3-turbo"),
        llm_model=os.environ.get("GROQ_LLM_MODEL", "llama-3.3-70b-versatile"),
        tts_model=os.environ.get("GROQ_TTS_MODEL", "canopylabs/orpheus-arabic-saudi"),
        tts_voice=os.environ.get("GROQ_TTS_VOICE", "fahad"),
    )

    if audio_file:
        wav_bytes = open(audio_file, "rb").read()
        turn = client.run_turn(wav_bytes=wav_bytes)
    else:
        print(f"بابعت لـ Groq النص: {text}")
        turn = client.run_turn(text=text)

    print(f"\n🤖 الرد: {turn.text}")
    out_wav = pcm_to_wav_bytes(turn.pcm_bytes, turn.sample_rate)

    out_path = make_output_path(output_dir)
    print("جاري تحريك الأفتار بالصوت ده... (ده هياخد شوية وقت)")
    bridge.audio_bytes_to_video(out_wav, out_path)
    print(f"\n✅ الفيديو جاهز: {out_path}")


async def run_gemini(text: str, audio_file: str, bridge: FlashTalkBridge, output_dir: str):
    from live_avatar.gemini_live_client import (
        GeminiLiveClient,
        pcm_to_wav_bytes,
        GEMINI_INPUT_SAMPLE_RATE,
    )

    model = os.environ.get("GEMINI_LIVE_MODEL", "gemini-2.5-flash-native-audio-preview-09-2025")
    voice = os.environ.get("GEMINI_VOICE_NAME", "Puck")

    async with GeminiLiveClient(model=model, voice_name=voice) as gemini:
        if text:
            print(f"بابعت لـ Gemini النص: {text}")
            await gemini.session.send_client_content(
                turns={"role": "user", "parts": [{"text": text}]},
                turn_complete=True,
            )
        else:
            print(f"بابعت لـ Gemini ملف الصوت: {audio_file}")
            pcm = _load_wav_as_pcm16(audio_file, GEMINI_INPUT_SAMPLE_RATE)
            chunk_size = GEMINI_INPUT_SAMPLE_RATE * 2 // 10
            for i in range(0, len(pcm), chunk_size):
                await gemini.send_audio_chunk(pcm[i : i + chunk_size])
            await gemini.end_user_turn()

        print("مستني رد Gemini الصوتي...")
        async for turn in gemini.receive_turns():
            print(f"\n🤖 Gemini قال: {turn.text}")
            wav_bytes = pcm_to_wav_bytes(turn.pcm_bytes, turn.sample_rate)

            out_path = make_output_path(output_dir)
            print("جاري تحريك الأفتار بالصوت ده...")
            bridge.audio_bytes_to_video(wav_bytes, out_path)
            print(f"\n✅ الفيديو جاهز: {out_path}")
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["groq", "gemini"], default="groq")
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--audio_file", type=str, default=None)
    args = parser.parse_args()

    if not args.text and not args.audio_file:
        parser.error("لازم تحدد --text أو --audio_file")

    cfg = dict(
        ckpt_dir=os.environ["FLASHTALK_CKPT_DIR"],
        wav2vec_dir=os.environ["WAV2VEC_DIR"],
        avatar_image=os.environ["AVATAR_IMAGE"],
    )
    output_dir = os.environ.get("OUTPUT_DIR", "./live_avatar/output")

    print("جاري تحميل موديل FlashTalk...")
    bridge = FlashTalkBridge(**cfg)

    if args.provider == "groq":
        run_groq(args.text, args.audio_file, bridge, output_dir)
    else:
        asyncio.run(run_gemini(args.text, args.audio_file, bridge, output_dir))


if __name__ == "__main__":
    main()
