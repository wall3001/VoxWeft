"""Onset probe — is there really a transient in the first patches?

Measures, per run: first-audio latency, RMS/peak of patch 1, patch 2, and the
rest, and the max adjacent-sample jump inside the first two patches (a click
is a jump comparable to the signal peak). Zero-shot and, if REF_WAV is set,
voice clone. This is the measurement behind DEVLOG §2.3.

    VOXCPM2_MODEL=~/.voxweft/models/VoxCPM2 REF_WAV=ref.wav python scripts/probe_onset.py
"""
import os
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from voxweft.core import VoxCPM2Pipeline  # noqa: E402

MODEL = os.environ.get("VOXCPM2_MODEL", os.path.expanduser("~/.voxweft/models/VoxCPM2"))
TEXT = os.environ.get("TEXT", "各位来宾，欢迎参加东盟数字经济国际会议。")
PATCH = 7680                     # 160 ms @ 48 kHz
RUNS = int(os.environ.get("RUNS", "3"))
WARMUP = int(os.environ.get("WARMUP", "0"))
FADE = float(os.environ.get("FADE", "0"))

pipe = VoxCPM2Pipeline(MODEL, inference_timesteps=6, quantize_lm=4)

ref_lat = None
if os.environ.get("REF_WAV"):
    w, sr = sf.read(os.environ["REF_WAV"], dtype="float32")
    if w.ndim > 1:
        w = w.mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample
        w = resample(w, int(len(w) * 16000 / sr)).astype(np.float32)
    ref_lat = pipe.encode_reference(w[:16000 * 6])


def gen(ref):
    t0, first, chunks = time.perf_counter(), None, []
    for c, _ in pipe.generate_streaming(TEXT, ref_latents=ref, warmup_patches=WARMUP,
                                        fade_in_ms=FADE):
        if first is None:
            first = time.perf_counter() - t0
        chunks.append(c)
    return first, np.concatenate(chunks)


def rms(x):
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0


def pk(x):
    return float(np.abs(x).max()) if len(x) else 0.0


cases = [("zero-shot", None)] + ([("clone", ref_lat)] if ref_lat is not None else [])
gen(None)                        # compile / warm, not measured
for label, ref in cases:
    print(f"--- {label} (warmup={WARMUP}, fade={FADE}ms) ---")
    for i in range(RUNS):
        f, w = gen(ref)
        p1, p2, rest = w[:PATCH], w[PATCH:2 * PATCH], w[2 * PATCH:]
        jump = float(np.abs(np.diff(w[:2 * PATCH])).max())
        print(f"run{i}: first_audio={f * 1000:.0f}ms  "
              f"p1 rms={rms(p1):.4f} pk={pk(p1):.3f} | p2 rms={rms(p2):.4f} pk={pk(p2):.3f} | "
              f"rest rms={rms(rest):.4f} pk={pk(rest):.3f} | first-1ms pk={pk(w[:48]):.4f} | "
              f"max|Δ| p1-2={jump:.3f}")
