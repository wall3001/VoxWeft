"""Design showcase voices with VoxCPM2's caption mode — no reference audio.

Official usage (model card): put a natural-language voice description in
parentheses at the start of the text. Everything after the closing paren is
spoken; the description is not. We generate N candidates per (language,
role), transcribe each back with Qwen3-ASR, and write a manifest. A human
listens and picks; the chosen wav + its text become the clone reference for
that role in that language.

Roles:  src  — mature male, calm broadcaster (the speaker being interpreted)
        tgt  — young female, warm and clear (the interpreter's voice)

    VOXCPM2_MODEL=… VOXWEFT_ASR_MODEL=… python scripts/design_voices.py \\
        --langs zh,en,de,fr,es,ja,ko,vi,th,id,ms --n 3 --out samples/voices

Produces samples/voices/<lang>_<role>_<k>.wav and manifest.json with
caption, text, ASR transcript, CER, and a `leak` flag if caption words
appear in the transcript.
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Two-sentence self-introduction per language. Short on purpose: ~6–9 s of
# speech is what a clone reference wants.
INTRO = {
    "zh": "VoxWeft 是一套跑在苹果芯片上的端侧同声传译系统。你说话，它一边听一边翻译，全程不联网，数据不出这台机器。",
    "en": "VoxWeft is an on-device simultaneous interpretation system for Apple Silicon. You speak, it translates while you talk, and nothing ever leaves the machine.",
    "de": "VoxWeft ist ein System für Simultandolmetschen, das vollständig auf Apple-Silicon-Geräten läuft. Sie sprechen, es übersetzt währenddessen, und nichts verlässt jemals das Gerät.",
    "fr": "VoxWeft est un système d'interprétation simultanée qui fonctionne entièrement sur puce Apple. Vous parlez, il traduit pendant que vous parlez, et rien ne quitte jamais la machine.",
    "es": "VoxWeft es un sistema de interpretación simultánea que funciona por completo en chips de Apple. Usted habla, él traduce mientras habla, y nada sale nunca de la máquina.",
    "it": "VoxWeft è un sistema di interpretazione simultanea che gira interamente su chip Apple. Tu parli, lui traduce mentre parli, e nulla lascia mai la macchina.",
    "pt": "VoxWeft é um sistema de interpretação simultânea que roda inteiramente em chips Apple. Você fala, ele traduz enquanto você fala, e nada sai da máquina.",
    "ja": "VoxWeft は、Apple シリコン上で完結する同時通訳システムです。話している間に翻訳が進み、データは一切この端末の外に出ません。",
    "ko": "VoxWeft는 애플 실리콘에서 완전히 동작하는 동시통역 시스템입니다. 말하는 동안 번역이 진행되며, 데이터는 절대 이 기기 밖으로 나가지 않습니다.",
    "vi": "VoxWeft là hệ thống phiên dịch đồng thời chạy hoàn toàn trên chip Apple. Bạn nói, nó dịch ngay trong lúc bạn nói, và không có gì rời khỏi chiếc máy này.",
    "th": "VoxWeft คือระบบล่ามพร้อมกันที่ทำงานทั้งหมดบนชิป Apple คุณพูด มันแปลไปพร้อมกัน และไม่มีข้อมูลใดออกจากเครื่องนี้เลย",
    "id": "VoxWeft adalah sistem penerjemahan simultan yang berjalan sepenuhnya di chip Apple. Anda berbicara, ia menerjemahkan saat Anda bicara, dan tidak ada data yang keluar dari mesin ini.",
    "ms": "VoxWeft ialah sistem terjemahan serentak yang berjalan sepenuhnya pada cip Apple. Anda bercakap, ia menterjemah semasa anda bercakap, dan tiada data keluar dari mesin ini.",
}

# Accent phrase the caption uses per language. Kept to the dominant standard
# variety; regional accents are a later pass.
ACCENT = {
    "zh": "standard Mandarin, Beijing accent", "en": "American English",
    "de": "standard German", "fr": "Parisian French", "es": "Castilian Spanish",
    "it": "standard Italian", "pt": "Brazilian Portuguese", "ja": "Tokyo Japanese",
    "ko": "Seoul Korean", "vi": "Hanoi Vietnamese", "th": "Bangkok Thai",
    "id": "Jakarta Indonesian", "ms": "Kuala Lumpur Malay",
}

ROLES = {
    "src": "A mature man in his fifties, calm and steady, clear broadcaster voice, unhurried, {accent}",
    "tgt": "A young woman in her twenties, warm and bright, clear and natural, moderate pace, {accent}",
}

LEAK_WORDS = {"man", "woman", "voice", "broadcaster", "accent", "twenties", "fifties",
              "mature", "young", "calm", "warm", "bright", "pace", "unhurried"}

ASR_NAME = {"zh": "Chinese", "en": "English", "de": "German", "fr": "French", "es": "Spanish",
            "it": "Italian", "pt": "Portuguese", "ja": "Japanese", "ko": "Korean",
            "vi": "Vietnamese", "th": "Thai", "id": "Indonesian", "ms": "Malay"}


def cer(ref: str, hyp: str) -> float:
    from rapidfuzz.distance import Levenshtein
    norm = lambda s: re.sub(r"[\s\W_]+", "", s).lower()
    r, h = norm(ref), norm(hyp)
    return Levenshtein.distance(r, h) / max(1, len(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="zh,en,de,fr,es,ja,ko,vi,th,id,ms")
    ap.add_argument("--roles", default="src,tgt")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", default="samples/voices")
    ap.add_argument("--steps", type=int, default=10, help="CFM steps; 10 for quality")
    ap.add_argument("--no-asr", action="store_true")
    a = ap.parse_args()

    from voxweft.core import VoxCPM2Pipeline
    model = os.environ.get("VOXCPM2_MODEL", os.path.expanduser("~/.voxweft/models/VoxCPM2"))
    pipe = VoxCPM2Pipeline(model, inference_timesteps=a.steps, quantize_lm=4)
    os.makedirs(a.out, exist_ok=True)

    asr = None
    if not a.no_asr:
        from mlx_audio.stt.utils import load_model
        asr_path = os.environ.get("VOXWEFT_ASR_MODEL", "")
        if asr_path:
            asr = load_model(asr_path)
            supported = {s.lower() for s in (getattr(asr.config, "support_languages", None) or [])}

    def transcribe(wav48, lang):
        if asr is None:
            return ""
        from scipy.signal import resample_poly
        a16 = resample_poly(wav48, 1, 3).astype(np.float32)
        name = ASR_NAME.get(lang)
        hint = name if name and name.lower() in supported else None
        r = asr.generate(a16, language=hint)
        t = str(getattr(r, "text", r))
        _, sep, tail = t.partition("<asr_text>")
        return (tail if sep else t).strip()

    manifest = []
    langs = [l for l in a.langs.split(",") if l in INTRO]
    for lang in langs:
        for role in a.roles.split(","):
            caption = ROLES[role].format(accent=ACCENT[lang])
            text = INTRO[lang]
            for k in range(1, a.n + 1):
                t0 = time.perf_counter()
                wav, stats = pipe.generate(f"({caption}){text}", cfg_value=2.0)
                gen_s = time.perf_counter() - t0
                fn = f"{lang}_{role}_{k}.wav"
                sf.write(os.path.join(a.out, fn), wav, 48000)
                hyp = transcribe(wav, lang)
                c = cer(text, hyp) if hyp else None
                leak = bool(hyp) and any(w in hyp.lower().split() for w in LEAK_WORDS)
                row = {"file": fn, "lang": lang, "role": role, "k": k, "caption": caption,
                       "text": text, "audio_s": round(stats["audio_s"], 2),
                       "gen_s": round(gen_s, 1), "asr": hyp, "cer": None if c is None else round(c, 4),
                       "leak": leak, "pick": False}
                manifest.append(row)
                flag = " ⚠️LEAK" if leak else ""
                cs = f"CER {c:5.1%}" if c is not None else "CER   —  "
                print(f"{fn:14s} {stats['audio_s']:5.1f}s  {cs}{flag} | {hyp[:60]}", flush=True)
                json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"),
                          ensure_ascii=False, indent=1)
    print(f"\n{len(manifest)} candidates → {a.out}/manifest.json. Listen, set \"pick\": true on one per (lang, role).")


if __name__ == "__main__":
    main()
