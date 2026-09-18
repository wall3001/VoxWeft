"""streaming VAE consistency + streaming generation latency."""
import os, sys, time
import numpy as np
import soundfile as sf
import mlx.core as mx
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from voxweft.core import VoxCPM2Pipeline
from voxweft.core.vae import StreamingVAEDecoder, StreamingVAEState

MODEL = os.environ.get("VOXCPM2_MODEL", os.path.expanduser("~/.voxweft/models/VoxCPM2"))
pipe = VoxCPM2Pipeline(MODEL, inference_timesteps=6, quantize_lm=4)

# 1) consistency: full decode vs chunked streaming decode on same latents
g = np.load(MODEL + "/vae_golden.npz")
z = g["z"].transpose(0, 2, 1)               # [1, 8, 64] NLC
full = np.array(pipe.vae(mx.array(z))[0])
stream = StreamingVAEDecoder(pipe.vae)
st = StreamingVAEState()
chunks = []
for i in range(0, 8, 4):                     # two 4-frame patches
    w = stream.decode_chunk(mx.array(z[:, i:i+4, :]), st)
    chunks.append(np.array(w[0]))
sv = np.concatenate(chunks)
d = np.abs(full - sv)
corr = np.corrcoef(full, sv)[0, 1]
print(f"stream-vs-full: len {len(full)}/{len(sv)} max_diff={d.max():.2e} corr={corr:.6f}")
assert corr > 0.9999, "STREAM MISMATCH"
print("STREAM CONSISTENCY PASS")

# 2) streaming generation latency
TEXT = "各位来宾，欢迎参加东盟数字经济国际会议。"
t0 = time.perf_counter()
first = None
chunks = []
for wav_chunk, n in pipe.generate_streaming(TEXT):
    if first is None:
        first = time.perf_counter() - t0
    chunks.append(wav_chunk)
total = time.perf_counter() - t0
audio = np.concatenate(chunks)
sf.write("/tmp/mlx_stream.wav", audio, 48000)
print(f"streaming: first_audio={first*1000:.0f}ms total={total:.1f}s "
      f"audio={len(audio)/48000:.2f}s rtf={total/(len(audio)/48000):.2f}")
print("STREAM DONE")
