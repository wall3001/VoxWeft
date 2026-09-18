"""RTF comparison — baseline vs compile vs compile+4bit."""
import os, sys, time
import numpy as np
import soundfile as sf
import mlx.core as mx
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from voxweft.core import VoxCPM2Pipeline

TEXT = "各位来宾，欢迎参加东盟数字经济国际会议。"
MODEL = os.environ.get("VOXCPM2_MODEL", os.path.expanduser("~/.voxweft/models/VoxCPM2"))

def bench(tag, **kw):
    pipe = VoxCPM2Pipeline(MODEL, **kw)
    mx.random.seed(7)
    wav, s = pipe.generate(TEXT)          # warm (includes compile)
    mx.random.seed(7)
    wav, s = pipe.generate(TEXT)          # measured
    sf.write(f"/tmp/perf_{tag}.wav", wav, 48000)
    peak = mx.get_peak_memory() / 1e9
    print(f"[{tag:12s}] rtf={s['rtf']:.2f} audio={s['audio_s']:.2f}s "
          f"wall={s['wall_s']:.1f}s prefill={s['prefill_s']*1000:.0f}ms peak_mem={peak:.1f}GB")
    del pipe
    mx.clear_cache()
    return s

bench("compile")                       # compile now always on
bench("compile+4bit", quantize_lm=4)
bench("compile+8bit", quantize_lm=8)
print("PERF DONE")
