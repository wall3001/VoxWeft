"""TTS 可懂度回环：判定 ASR 把 tts_out/*.wav 转回文字，与原文算 CER。
VoxCPM2 官方就是这么评 30 语可懂度的（他们用 Gemini ASR）。
    python judge_tts.py
"""
import json, os, glob
import numpy as np, soundfile as sf
from bench_asr import cer, load_asr

from paths import RESULTS, TTS_OUT, model_kind, model_path
# Same loader as the ASR bench, so the judge applies the same "only send a
# language label the model lists" rule instead of forcing lo/km/my.
_judge = os.environ.get("VOXWEFT_JUDGE_ASR", "qwen17")
_run = load_asr(model_kind(_judge), model_path(_judge))

def asr(wav, lang):
    a, sr = sf.read(wav, dtype="float32")
    return _run(a, lang)

out = []
for f in sorted(glob.glob(f"{RESULTS}/tts_*.json")):
    rows = json.load(open(f))
    for r in rows:
        hyp = asr(f"{TTS_OUT}/{r['wav']}", r["lang"]); c = cer(r["text"], hyp)
        r["asr_hyp"] = hyp; r["cer"] = round(c, 4); out.append(r)
        print(f"{r['engine']:9s} {r['lang']} #{r['i']}  CER {c:6.1%}  RTF {r['rtf']:.2f}  首包 {r['first_s']:.2f}s | {hyp[:50]}")
by = {}
for r in out: by.setdefault((r["engine"], r["lang"]), []).append(r["cer"])
print("\n== 可懂度汇总（CER 越低越好）")
for (e, l), v in sorted(by.items()): print(f"   {e:9s} {l}: CER {np.mean(v):.1%}  (n={len(v)})")
json.dump(out, open(f"{RESULTS}/tts_judged.json", "w"), ensure_ascii=False, indent=1)
