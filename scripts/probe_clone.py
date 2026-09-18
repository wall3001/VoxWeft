"""Encoder golden check + voice cloning e2e. Set VOXCPM2_MODEL and REF_WAV."""
import os, sys, time
import numpy as np
import soundfile as sf
import mlx.core as mx
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from voxweft.core import VoxCPM2Pipeline

MODEL = os.environ.get("VOXCPM2_MODEL", os.path.expanduser("~/.voxweft/models/VoxCPM2"))
pipe = VoxCPM2Pipeline(MODEL)

# 1) encoder golden
g = np.load(os.path.join(MODEL, "enc_golden.npz"))
mu = pipe.encode_reference(g["wav"])            # [T, 64]
ref = g["mu"].T                                  # [T, 64]
d = np.abs(mu - ref)
corr = np.corrcoef(mu.flatten(), ref.flatten())[0, 1]
print(f"encoder: max_diff={d.max():.4f} corr={corr:.6f}")
assert corr > 0.999, "ENCODER MISMATCH"
print("ENCODER GOLDEN PASS")

# 2) clone synthesis
wav, sr = sf.read(os.environ.get("REF_WAV", "samples/ref_5s.wav"), dtype="float32")
if wav.ndim > 1: wav = wav.mean(axis=1)
if sr != 16000:
    from scipy.signal import resample
    wav = resample(wav, int(len(wav) * 16000 / sr)).astype(np.float32)
ref_latents = pipe.encode_reference(wav)
print(f"ref latents: {ref_latents.shape}")

for name, text in [("zh", "各位来宾，欢迎参加东盟数字经济国际会议。"),
                   ("lo", "ສະບາຍດີທຸກໆທ່ານ ຍິນດີຕ້ອນຮັບເຂົ້າຮ່ວມກອງປະຊຸມ")]:
    out, stats = pipe.generate(text, ref_latents=ref_latents)
    sf.write(f"/tmp/mlx_clone_{name}.wav", out, 48000)
    print(f"[clone-{name}] {stats['audio_s']:.2f}s audio, rtf={stats['rtf']:.2f}")
print("CLONE DONE")
