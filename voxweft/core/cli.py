"""voxweft say — command-line TTS (L1, no runtime deps).

  voxweft say "你好世界" -o out.wav
  voxweft say "ສະບາຍດີ" --ref my_voice.wav --play
  voxweft say "Guten Tag" --stream -o out.wav      # streaming path, prints first-audio ms
"""

import argparse
import os
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="voxweft say",
                                 description="VoxCPM2 TTS on Apple Silicon (MLX)")
    ap.add_argument("text", help="text to synthesize")
    ap.add_argument("-m", "--model", default=os.environ.get("VOXCPM2_MODEL", ""),
                    help="model dir (or set VOXCPM2_MODEL)")
    ap.add_argument("-o", "--out", default="out.wav", help="output wav path")
    ap.add_argument("--ref", default=None,
                    help="reference wav for voice cloning (any sr, mono/stereo)")
    ap.add_argument("--steps", type=int, default=6, help="CFM timesteps (6-10)")
    ap.add_argument("--quant", type=int, default=4, choices=(0, 4, 8),
                    help="LM quantization bits (0=bf16)")
    ap.add_argument("--cfg", type=float, default=2.0, help="CFG value")
    ap.add_argument("--play", action="store_true", help="play after synthesis (afplay)")
    ap.add_argument("--stream", action="store_true",
                    help="use the streaming decoder and report first-audio latency")
    args = ap.parse_args(argv)

    if not args.model:
        sys.exit("error: --model or VOXCPM2_MODEL required")

    import numpy as np
    import soundfile as sf
    from .generate import VoxCPM2Pipeline

    pipe = VoxCPM2Pipeline(args.model, inference_timesteps=args.steps,
                           quantize_lm=args.quant or None)

    ref_latents = None
    if args.ref:
        wav, sr = sf.read(args.ref, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            from scipy.signal import resample
            wav = resample(wav, int(len(wav) * 16000 / sr)).astype(np.float32)
        ref_latents = pipe.encode_reference(wav)

    if args.stream:
        import time
        t0 = time.perf_counter()
        chunks, t_first = [], None
        for chunk, _ in pipe.generate_streaming(args.text, ref_latents=ref_latents,
                                                cfg_value=args.cfg):
            if t_first is None:
                t_first = time.perf_counter() - t0
            chunks.append(chunk)
        total = time.perf_counter() - t0
        out = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
        sf.write(args.out, out, 48000)
        audio_s = len(out) / 48000
        print(f"{args.out}: {audio_s:.2f}s audio, first_audio={t_first*1000:.0f}ms, "
              f"rtf={total/audio_s:.2f}" if audio_s else f"{args.out}: empty")
    else:
        out, stats = pipe.generate(args.text, ref_latents=ref_latents,
                                   cfg_value=args.cfg)
        sf.write(args.out, out, 48000)
        print(f"{args.out}: {stats['audio_s']:.2f}s audio, rtf={stats['rtf']:.2f}")

    if args.play:
        os.system(f"afplay '{args.out}'")


if __name__ == "__main__":
    main()
