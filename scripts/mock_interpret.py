"""Mic-free interpretation run against a live `voxweft serve`.

Two modes:

* **text** (default) — feeds a text set as `text_input`: skips ASR, exercises
  MT → streaming TTS with the events a spoken session produces.
* **audio** (`--wav file.wav`) — streams a 16 kHz wav through `ptt_start` /
  PCM frames / `ptt_end` at real-time pace, i.e. exactly what the app's
  microphone does. This is the full pipeline: segmenter → ASR → MT → TTS.
  Feed it the designed source voice (`samples/voices/<lang>_src_*.wav`) to
  record a demo with a consistent "speaker" and no microphone.

    voxweft bridge &  voxweft serve &
    python scripts/mock_interpret.py --src zh --tgt en                     # text, intro_zh.txt
    python scripts/mock_interpret.py --src es --tgt ms --wav samples/voices/es_src_2.wav
    python scripts/mock_interpret.py --src en --tgt zh --mode agent --save-wav out/

Requires: pip install websockets
"""
import argparse
import asyncio
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


async def run(a):
    import websockets
    text_file = a.text_file or os.path.join(HERE, "..", "bench", "data", "intro", f"intro_{a.src}.txt")
    sents = [l.strip() for l in open(text_file, encoding="utf-8") if l.strip()]
    if a.save_wav:
        os.makedirs(a.save_wav, exist_ok=True)

    async with websockets.connect(a.url, max_size=None) as ws:
        ready = json.loads(await ws.recv())
        assert ready["type"] == "ready", ready
        await ws.send(json.dumps({"type": "session_config", "mode": a.mode,
                                  "src_lang": a.src, "tgt_lang": a.tgt}))
        ack = json.loads(await ws.recv())
        print(f"routes: {ack.get('routes')}")

        if a.wav:
            await run_audio(ws, a)
            return

        rows = []
        for i, s in enumerate(sents):
            t0 = time.perf_counter()
            await ws.send(json.dumps({"type": "text_input", "text": s}))
            first_audio = None
            out_text = ""
            pcm = bytearray()
            while True:
                msg = await ws.recv()
                if isinstance(msg, bytes):
                    if first_audio is None:
                        first_audio = time.perf_counter() - t0
                    pcm += msg
                    continue
                ev = json.loads(msg)
                t = ev.get("type")
                if t in ("translation", "llm"):
                    out_text = ev.get("text", "")
                elif t == "error":
                    out_text = f"ERROR {ev.get('stage')}: {ev.get('message')}"
                    break
                elif t == "turn_end":
                    break
            total = time.perf_counter() - t0
            rows.append((s, out_text, first_audio, total, len(pcm) / 2 / 48000))
            fa = f"{first_audio * 1000:.0f}ms" if first_audio else "—"
            print(f"[{i + 1}/{len(sents)}] first_audio={fa} total={total * 1000:.0f}ms "
                  f"audio={len(pcm) / 2 / 48000:.2f}s\n    src: {s[:70]}\n    out: {out_text[:70]}")
            if a.save_wav and pcm:
                import numpy as np
                import soundfile as sf
                sf.write(os.path.join(a.save_wav, f"{i + 1:02d}.wav"),
                         np.frombuffer(bytes(pcm), dtype=np.int16), 48000)

    fas = [r[2] for r in rows if r[2]]
    if fas:
        fas.sort()
        print(f"\nfirst_audio median {fas[len(fas) // 2] * 1000:.0f}ms  "
              f"p90 {fas[int(len(fas) * 0.9) - 1] * 1000:.0f}ms  n={len(fas)}/{len(rows)}")
    if a.json_out:
        json.dump([{"src": r[0], "out": r[1], "first_audio_s": r[2], "total_s": r[3],
                    "audio_s": r[4]} for r in rows],
                  open(a.json_out, "w"), ensure_ascii=False, indent=1)


async def run_audio(ws, a):
    import numpy as np
    import soundfile as sf
    wav, sr = sf.read(a.wav, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample_poly
        wav = resample_poly(wav, 16000, sr).astype(np.float32)
    pcm = (np.clip(wav, -1, 1) * 32767).astype(np.int16).tobytes()
    frame = 1024                                # 32 ms @ 16 kHz int16
    t0 = time.perf_counter()
    await ws.send(json.dumps({"type": "ptt_start"}))
    for off in range(0, len(pcm), frame):
        await ws.send(pcm[off:off + frame])
        if a.pace > 0:
            due = t0 + (off + frame) / 32000.0 / a.pace
            delay = due - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
    await ws.send(json.dumps({"type": "ptt_end"}))
    t_release = time.perf_counter()
    print(f"streamed {len(wav) / 16000:.1f}s of audio in {t_release - t0:.1f}s; waiting for turns…")

    segs = {}
    out_pcm = bytearray()
    idle_since = time.perf_counter()
    while True:
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
        except asyncio.TimeoutError:
            if time.perf_counter() - idle_since > 20:
                break
            continue
        idle_since = time.perf_counter()
        if isinstance(msg, bytes):
            out_pcm += msg
            continue
        ev = json.loads(msg)
        t, seq = ev.get("type"), ev.get("seq", 0)
        now = time.perf_counter()
        d = segs.setdefault(seq, {"t_commit": None, "asr": "", "mt": "", "first_audio": None})
        if t == "segment_committed":
            d["t_commit"] = now
        elif t == "asr":
            d["asr"] = ev.get("text", "")
        elif t == "translation":
            d["mt"] = ev.get("text", "")
        elif t == "audio_start" and d["first_audio"] is None and d["t_commit"]:
            d["first_audio"] = now - d["t_commit"]
            print(f"  seg {seq}: first audio {d['first_audio'] * 1000:.0f} ms after commit\n"
                  f"    asr: {d['asr'][:70]}\n    mt : {d['mt'][:70]}")
        elif t == "no_speech":
            print(f"  no_speech: {ev}")
        elif t == "error":
            print(f"  ERROR {ev.get('stage')}: {ev.get('message')}")
        elif t == "turn_end" and not a.wav:
            break
    if a.save_wav and out_pcm:
        os.makedirs(a.save_wav, exist_ok=True)
        sf.write(os.path.join(a.save_wav, "interpretation.wav"),
                 np.frombuffer(bytes(out_pcm), dtype=np.int16), 48000)
    fas = sorted(v["first_audio"] for v in segs.values() if v["first_audio"])
    if fas:
        print(f"\n{len(segs)} segments; first-audio-after-commit median {fas[len(fas) // 2] * 1000:.0f} ms, "
              f"max {fas[-1] * 1000:.0f} ms; {len(out_pcm) / 2 / 48000:.1f}s of interpretation audio")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("VOXWEFT_WS", "ws://127.0.0.1:3100/ws/voice/v2"))
    ap.add_argument("--src", default="zh")
    ap.add_argument("--tgt", default="en")
    ap.add_argument("--mode", default="interpreter", choices=["interpreter", "agent"])
    ap.add_argument("--text-file")
    ap.add_argument("--wav", help="16 kHz mono wav to stream as microphone input (audio mode)")
    ap.add_argument("--pace", type=float, default=1.0, help="audio mode: 1.0 = real time, 0 = as fast as possible")
    ap.add_argument("--save-wav")
    ap.add_argument("--json-out")
    a = ap.parse_args()
    asyncio.run(run(a))


if __name__ == "__main__":
    sys.exit(main())
