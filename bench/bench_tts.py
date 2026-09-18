"""TTS 评测（Apple Silicon）。

    python bench_tts.py --engine ours      # voxweft.core（我们的移植，4-bit，流式）
    python bench_tts.py --engine mlxaudio  # mlx-audio 内置 voxcpm2，用 mlx-community/VoxCPM2-4bit
    python judge_tts.py                    # 回环 ASR 算 CER（可懂度）

每条记录：首包延迟（流式第一块音频到手）、总耗时、音频时长、RTF。
输出 wav 重采样到 16k 存 tts_out/，供 judge 用。
"""
import argparse, json, os, time, glob
import numpy as np
import soundfile as sf

from paths import RESULTS, TTS_OUT as OUT, ensure_dirs, manifest, model_path
ensure_dirs()

def texts():
    """Text set for the TTS bench: every manifest item that has a `ref`,
    text-only items included. Long refs are cut to their first sentence so
    the run stays short and per-item RTF is comparable."""
    seps = {"km": ["។"], "lo": ["\n", ". "], "zh": ["。", "！", "？"]}

    def first_sent(s, lang, maxlen=110):
        s = s.strip()
        for sep in seps.get(lang, [". ", "\n"]):
            i = s.find(sep)
            if 0 < i < maxlen:
                return s[:i + len(sep)].strip()
        cut = s[:maxlen]
        j = cut.rfind(" ")
        return (cut[:j] if j > 40 else cut).strip()

    out = []
    for it in manifest():
        if it.get("ref"):
            out.append((it["lang"], first_sent(it["ref"], it["lang"])))
    return out

def resample_to_16k(a: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000: return a
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(sr, 16000); return resample_poly(a, 16000 // g, sr // g).astype(np.float32)

def run_ours(steps):
    import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from voxweft.core import VoxCPM2Pipeline
    t0 = time.perf_counter()
    pipe = VoxCPM2Pipeline(model_path("voxcpm2"), inference_timesteps=steps, quantize_lm=4)
    load = time.perf_counter() - t0
    def synth(text, lang=None):
        chunks = []; t1 = time.perf_counter(); first = None
        for chunk, _idx in pipe.generate_streaming(text):
            if first is None: first = time.perf_counter() - t1
            chunks.append(np.asarray(chunk, dtype=np.float32))
        total = time.perf_counter() - t1
        return np.concatenate(chunks) if chunks else np.zeros(0, np.float32), 48000, first, total
    return synth, load

def run_mlxaudio(steps):
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model
    path = model_path("voxcpm2-mlxaudio")
    t0 = time.perf_counter(); model = load_model(path); load = time.perf_counter() - t0
    sr = int(model.sample_rate)
    def synth(text, lang=None):
        chunks = []; t1 = time.perf_counter(); first = None
        for r in model.generate(text=text, inference_timesteps=steps, cfg_value=2.0):
            a = np.asarray(r.audio, dtype=np.float32).reshape(-1)
            if first is None and len(a): first = time.perf_counter() - t1
            chunks.append(a)
        total = time.perf_counter() - t1
        return np.concatenate(chunks) if chunks else np.zeros(0, np.float32), sr, first, total
    return synth, load

def run_omnivoice(steps):
    """k2-fsa OmniVoice（0.6B Qwen3 骨干，非自回归掩码扩散，646 语，Apache-2.0）—— M2 级机器的轻量 TTS 候选。
    steps 这里映射到 num_steps（默认 32）。lo/km/my 训练数据只有 7–12 小时，可懂度要靠回环 ASR 判。"""
    from mlx_audio.tts.utils import load_model
    path = model_path("omnivoice")
    t0 = time.perf_counter(); model = load_model(path); load = time.perf_counter() - t0
    sr = int(model.sample_rate)
    LANG = {"zh": "Chinese", "vi": "Vietnamese", "lo": "Lao", "km": "Khmer"}
    def synth(text, lang="zh"):
        chunks = []; t1 = time.perf_counter(); first = None
        for r in model.generate(text=text, language=LANG.get(lang, "None"), num_steps=(steps if steps >= 8 else 32)):
            a = np.asarray(r.audio, dtype=np.float32).reshape(-1)
            if first is None and len(a): first = time.perf_counter() - t1
            chunks.append(a)
        total = time.perf_counter() - t1
        return np.concatenate(chunks) if chunks else np.zeros(0, np.float32), sr, first, total
    return synth, load

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--engine", required=True, choices=["ours", "mlxaudio", "omnivoice"])
    ap.add_argument("--steps", type=int, default=6)
    a = ap.parse_args()
    synth, load = {"ours": run_ours, "mlxaudio": run_mlxaudio, "omnivoice": run_omnivoice}[a.engine](a.steps)
    print(f"== {a.engine} steps={a.steps}  load {load:.1f}s", flush=True)
    synth("你好。")  # 预热
    rows = []
    for i, (lang, text) in enumerate(texts()):
        audio, sr, first, total = synth(text, lang)
        dur = len(audio) / sr
        a16 = resample_to_16k(audio, sr)
        fn = f"{OUT}/{a.engine}{"-s"+str(a.steps) if a.engine=="omnivoice" and a.steps>=8 else ""}_{lang}_{i:02d}.wav"; sf.write(fn, a16, 16000, subtype="PCM_16")
        rtf = total / dur if dur > 0 else float("nan")
        rows.append({"engine": a.engine, "lang": lang, "i": i, "text": text, "wav": os.path.basename(fn),
                     "first_s": round(first or 0, 3), "total_s": round(total, 2), "audio_s": round(dur, 2), "rtf": round(rtf, 2)})
        print(f"   {lang} #{i}  首包 {first or 0:.2f}s  总 {total:5.1f}s  音频 {dur:5.1f}s  RTF {rtf:.2f} | {text[:40]}", flush=True)
    import statistics as st
    print(f"\n== {a.engine} 汇总: 首包中位 {st.median(r['first_s'] for r in rows):.2f}s  RTF 中位 {st.median(r['rtf'] for r in rows):.2f}")
    json.dump(rows, open(f"{RESULTS}/tts_{a.engine}{"-s"+str(a.steps) if a.engine=="omnivoice" and a.steps>=8 else ""}.json", "w"), ensure_ascii=False, indent=1)

if __name__ == "__main__":
    main()
