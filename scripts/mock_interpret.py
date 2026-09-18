"""Mic-free interpretation run against a live `voxweft serve`.

Feeds a text set through /ws/voice/v2 as `text_input` — skipping ASR but
exercising segment → MT → streaming TTS with exactly the events a spoken
session produces — and reports per-sentence latency to first audio and to
`turn_end`. Use it to check a deployment, to record a demo without a
microphone, or as the TTS/MT half of an end-to-end regression.

    voxweft bridge &  voxweft serve &
    python scripts/mock_interpret.py --src zh --tgt en            # bench/data/intro/intro_zh.txt
    python scripts/mock_interpret.py --src en --tgt zh --mode agent
    python scripts/mock_interpret.py --text-file my.txt --save-wav out/

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("VOXWEFT_WS", "ws://127.0.0.1:3100/ws/voice/v2"))
    ap.add_argument("--src", default="zh")
    ap.add_argument("--tgt", default="en")
    ap.add_argument("--mode", default="interpreter", choices=["interpreter", "agent"])
    ap.add_argument("--text-file")
    ap.add_argument("--save-wav")
    ap.add_argument("--json-out")
    a = ap.parse_args()
    asyncio.run(run(a))


if __name__ == "__main__":
    sys.exit(main())
