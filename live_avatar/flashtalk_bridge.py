# -*- coding: utf-8 -*-
"""
flashtalk_bridge.py
--------------------
يحمّل موديل FlashTalk مرة واحدة، وبعدين ياخد أي صوت (raw PCM من
Gemini Live) ويحوله لفيديو أفاتار بيتكلم بيه.

ملحوظة مهمة:
    الموديل ده مش بيشتغل frame-by-frame وهو الصوت لسه بيتسجل (مش
    streaming حقيقي مع المايك). هو بياخد "دور كلام" كامل (turn) من
    Gemini، ويولّد الفيديو بتاعه على شكل chunks صوتية (33 فريم، بعدين
    28 فريم...)، فبيبقى فيه تأخير (latency) طبيعي قد مدة رد Gemini +
    وقت التوليد. ده الحد الأقصى للـ "live" الممكن بالموديل ده حاليًا.
"""

import os
import io
import time
from datetime import datetime

import numpy as np
import librosa
import imageio
import torch
from loguru import logger

from flash_talk.inference import (
    get_pipeline,
    get_base_data,
    get_audio_embedding,
    run_pipeline,
    infer_params,
)


class FlashTalkBridge:
    def __init__(
        self,
        ckpt_dir: str,
        wav2vec_dir: str,
        avatar_image: str,
        cpu_offload: bool = False,
        input_prompt: str = (
            "A person is talking. Only the foreground characters are "
            "moving, the background remains static."
        ),
        base_seed: int = 9999,
    ):
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.rank = int(os.environ.get("RANK", 0))

        logger.info("جاري تحميل موديل FlashTalk... (ده بياخد وقت ومساحة VRAM كبيرة)")
        self.pipeline = get_pipeline(
            world_size=world_size,
            ckpt_dir=ckpt_dir,
            wav2vec_dir=wav2vec_dir,
            cpu_offload=cpu_offload,
        )

        self.sample_rate = infer_params["sample_rate"]  # 16000 - ثابت في الموديل
        self.tgt_fps = infer_params["tgt_fps"]
        self.frame_num = infer_params["frame_num"]
        self.motion_frames_num = infer_params["motion_frames_num"]
        self.slice_len = self.frame_num - self.motion_frames_num

        get_base_data(
            self.pipeline,
            input_prompt=input_prompt,
            cond_image=avatar_image,
            base_seed=base_seed,
        )
        logger.info("الموديل جاهز.")

    def audio_bytes_to_video(self, wav_bytes: bytes, out_path: str) -> str:
        """
        ياخد ملف صوت (WAV bytes، أي sample rate) وينتج فيديو mp4
        للأفاتار بيتكلم بالصوت ده، ويحفظه في out_path.
        """
        human_speech_array, _ = librosa.load(
            io.BytesIO(wav_bytes), sr=self.sample_rate, mono=True
        )

        slice_len = self.slice_len
        frame_num = self.frame_num
        sample_rate = self.sample_rate
        tgt_fps = self.tgt_fps

        human_speech_array_slice_len = slice_len * sample_rate // tgt_fps
        human_speech_array_frame_num = frame_num * sample_rate // tgt_fps

        # بادينج بالصمت عشان ما نقصش آخر جزء من الكلام
        remainder = (
            len(human_speech_array) - human_speech_array_frame_num
        ) % human_speech_array_slice_len
        if remainder > 0:
            pad_length = human_speech_array_slice_len - remainder
            human_speech_array = np.concatenate(
                [human_speech_array, np.zeros(pad_length, dtype=human_speech_array.dtype)]
            )

        audio_embedding_all = get_audio_embedding(self.pipeline, human_speech_array)

        n_chunks = (audio_embedding_all.shape[1] - frame_num) // slice_len
        generated_list = []

        for chunk_idx in range(max(n_chunks, 1)):
            start = chunk_idx * slice_len
            audio_embedding_chunk = audio_embedding_all[
                :, start : start + frame_num
            ].contiguous()

            torch.cuda.synchronize()
            t0 = time.time()
            video = run_pipeline(self.pipeline, audio_embedding_chunk)
            if chunk_idx != 0:
                video = video[self.motion_frames_num :]
            torch.cuda.synchronize()

            if self.rank == 0:
                logger.info(
                    f"chunk-{chunk_idx} اتولّد في {(time.time() - t0):.2f}s"
                )
            generated_list.append(video.cpu())

        if self.rank == 0:
            self._save_video(generated_list, out_path, wav_bytes, tgt_fps)
        return out_path

    def _save_video(self, frames_list, video_path, wav_bytes, fps):
        os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
        temp_video_path = video_path.replace(".mp4", "_noaudio.mp4")

        with imageio.get_writer(
            temp_video_path, format="mp4", mode="I", fps=fps, codec="h264",
            ffmpeg_params=["-bf", "0"],
        ) as writer:
            for frames in frames_list:
                frames = frames.numpy().astype(np.uint8)
                for i in range(frames.shape[0]):
                    writer.append_data(frames[i])

        # اكتب الصوت الأصلي لملف مؤقت وادمجه مع الفيديو
        tmp_wav_path = video_path.replace(".mp4", "_audio.wav")
        with open(tmp_wav_path, "wb") as f:
            f.write(wav_bytes)

        import subprocess

        cmd = [
            "ffmpeg", "-y",
            "-i", temp_video_path,
            "-i", tmp_wav_path,
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            video_path,
        ]
        subprocess.run(cmd, check=True)
        os.remove(temp_video_path)
        os.remove(tmp_wav_path)


def make_output_path(output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    return os.path.join(output_dir, f"turn_{ts}.mp4")
