# Live Avatar (Gemini Live STS + FlashTalk)

بيوصل صوتك لايف بـ Gemini (يسمعك ويرد بصوت مباشرة، من غير STT/LLM/TTS
منفصلين)، وبعدين بيحرك صورة أفاتار بصوت الرد ده عن طريق FlashTalk.

## 1. المتطلبات

- كل متطلبات FlashTalk الأصلية (شوف `../README.md`) + موديلات
  `SoulX-FlashTalk-14B` و `chinese-wav2vec2-base` متنزّلة.
- GPU: 64GB+ VRAM (مع `--cpu_offload` في الكود لو أقل)، أو 8xH800
  للسرعة الـ real-time الكاملة.
- مايكروفون وسماعة على الجهاز اللي هتشغل الكود عليه.

```bash
pip install -r live_avatar/requirements.txt
```

## 2. الإعداد

```bash
cp live_avatar/.env.example live_avatar/.env
# افتح .env واملى:
#   - GEMINI_API_KEY (مفتاحك من Google AI Studio)
#   - FLASHTALK_CKPT_DIR و WAV2VEC_DIR (مسارات الموديلات بعد التنزيل)
#   - AVATAR_IMAGE (صورة الأفاتار اللي عايز يتحرك)
```

⚠️ **متسيبش المفتاح مكتوب في أي كود أو تتشاركه في شات/جروب** — أي حد
شافه يقدر يستخدمه على حسابك. لو حصل كده قبل كده، اعمله regenerate من
Google AI Studio.

## 3. التشغيل

```bash
python live_avatar/main.py
```

اتكلم في المايك، Gemini هيرد بصوته، والفيديو هيتولّد ويتفتح تلقائي
في `live_avatar/output/`.

## 4. حدود مهمة تعرفها

- **مش streaming حقيقي على مستوى الفريم**: FlashTalk بياخد صوت الدور
  (turn) كامل من Gemini، فبيبقى فيه تأخير طبيعي = وقت رد Gemini +
  وقت توليد الفيديو (ثواني على 8xH800، أكتر بكتير على كارت واحد).
- الصوت اللي بيرجعه Gemini بصوته الجاهز (زي "Puck")، مش صوتك ولا صوت
  مخصص — لو محتاج صوت معيّن، استبدل الخطوة دي بـ TTS منفصل بدل الـ
  Gemini Live الكامل.
- لو الجهاز مفيهوش GPU كفاية، جرب أول `SoulX-FlashHead` (نسخة أخف من
  نفس الفريق بتشتغل على RTX 4090/5090) بدل الموديل الأساسي.
