"""VoxWeft runtime — WebSocket server for simultaneous interpretation & dialogue.

    voxweft serve --port 3100          (expects `voxweft bridge` on :8133)

One endpoint, two modes, one event vocabulary:

    /ws/voice/v2   mode=interpreter   speech → ASR → MT → TTS, emitted per segment
                   mode=agent         speech → ASR → LLM → TTS, per sentence

Client → server (JSON text frames unless noted):
    session_config {mode, src_lang, tgt_lang, hotwords}
    ptt_start / ptt_end                 push-to-talk; ptt_start implies interrupt
    <binary>                            int16 mono 16 kHz PCM, any framing
    interrupt {reason} · reset · text_input {text} · speak {text, seq} · canon {pairs}

Server → client:
    ready · session_config_ack · ptt_ack · segment_committed · asr · translation
    llm_sentence · audio_start / <binary int16 48 kHz> / audio_end · turn_end
    interrupted · no_speech · error

Why two processes (this + the bridge): MLX GPU streams are thread-bound and
the models are not thread-safe. The bridge pins each model to one executor
thread; this process never touches MLX for TTS/ASR — only the in-process
translator/LLM (mlx_lm) lives here. See docs/DEVLOG.md.
"""
from __future__ import annotations

import asyncio
import base64
import hmac
import io
import json
import logging
import os
import time

import numpy as np
import uvicorn
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

log = logging.getLogger("voxweft.server")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s %(message)s")

# ── config ────────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("VOXWEFT_API_KEY", "")
BRIDGE_URL = os.environ.get("VOXWEFT_BRIDGE_URL", "http://localhost:8133")
LLM_DIR = os.environ.get("VOXWEFT_LLM_DIR", "")
GLOSSARY_PATH = os.environ.get(
    "VOXWEFT_GLOSSARY_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "glossary.json"))
USE_LLM_TRANSLATOR = bool(LLM_DIR)
USE_LLM_ZH_TO_SEA = os.environ.get("VOXWEFT_USE_LLM_ZH_TO_SEA", "0") == "1"
# SmartTurn v3 endpointing. Measured net-negative for interpretation (it is a
# dialogue "did they finish?" model; in continuous speech it blocks the silence
# cut and everything falls through to the forced cut). Off by default; flip on
# for dialogue experiments and compare segment counts on the same audio first.
SMART_TURN = os.environ.get("VOXWEFT_SMART_TURN", "0") == "1"
MAX_UTTERANCE_S = os.environ.get("VOXWEFT_MAX_UTTERANCE_S", "6.0")
os.environ.setdefault("VOXWEFT_MAX_UTTERANCE_S", MAX_UTTERANCE_S)

FRAME_BYTES = 1024            # 32 ms @ 16 kHz int16
TTS_CHUNK_SAMPLES = 9600      # 200 ms @ 48 kHz per binary frame

AGENT_SYSTEM_PROMPT = os.environ.get(
    "VOXWEFT_AGENT_PROMPT",
    "你是一个友善、自然的语音助手。回答要口语化、简短（一到两句），"
    "像面对面聊天那样自然。不要输出 Markdown、列表或表情符号——你的回答会被读成语音。")

# Default ASR hotwords by source language (FLORES-200 keys). Sparse on
# purpose: a handful of high-value proper nouns beats a flood — long hotword
# lists perturb decoding on short segments. Each entry below fixed a real
# mis-recognition (e.g. Lao "AI" heard as the single letter ເອ; Chinese 端侧 →
# 端测, 切段 → 切断, 出声 → 出生, which then translated into nonsense).
DEFAULT_HOTWORDS = {
    "lao_Laoo": ["ວຽງຈັນ", "ນະຄອນຫຼວງວຽງຈັນ", "ຫຼວງພະບາງ", "ອາຊຽນ", "ກອງປະຊຸມ",
                 "ເສດຖະກິດດິຈິຕອນ", "ປັນຍາປະດິດ", "ເອໄອ", "ພື້ນຖານໂຄງລ່າງ"],
    "zho_Hans": ["东盟", "数字经济", "人工智能", "基础设施", "同声传译",
                 "端侧", "切段", "出声", "端到端", "时延", "本地推理"],
    "tha_Thai": ["อาเซียน", "การประชุม"],
    "khm_Khmr": ["អាស៊ាន", "សេដ្ឋកិច្ចឌីជីថល"],
    "mya_Mymr": ["အာဆီယံ", "စီးပွားရေး"],
}

app = FastAPI(title="voxweft runtime")
_engines: dict = {}
_engines_lock = asyncio.Lock()


def _check_api_key(token: str | None) -> bool:
    if not API_KEY:
        return True
    return bool(token) and hmac.compare_digest(token, API_KEY)


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "service": "voxweft-runtime",
                         "translator_loaded": "translator" in _engines})


# ── bridge clients (ASR / TTS live in the other process) ──────────────────────

def _bridge_asr(audio: np.ndarray, language: str | None, hotwords) -> dict:
    import requests
    import soundfile as sf
    t0 = time.perf_counter()
    buf = io.BytesIO()
    sf.write(buf, (np.clip(audio, -1, 1) * 32767).astype(np.int16), 16000,
             format="WAV", subtype="PCM_16")
    url = "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()
    r = requests.post(f"{BRIDGE_URL}/v1/chat/completions", timeout=300, json={
        "model": "asr",
        "messages": [{"role": "user", "content": [
            {"type": "audio_url", "audio_url": {"url": url}}]}],
        "audio_duration_sec": len(audio) / 16000,
        "language": language or "chinese",
        "hotword": list(hotwords or []),
    })
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    _, sep, text = content.partition("<asr_text>")
    return {"text": (text if sep else content).strip(),
            "latency_ms": (time.perf_counter() - t0) * 1000}


def _bridge_tts_stream(text: str):
    """POST /generate stream=true; bridge emits int16 @48k, we yield float32."""
    import requests
    r = requests.post(f"{BRIDGE_URL}/generate", stream=True, timeout=180,
                      json={"target_text": text, "stream": True,
                            "temperature": 1.0, "cfg_value": 2.0})
    r.raise_for_status()
    buf, step = b"", TTS_CHUNK_SAMPLES * 2
    for chunk in r.iter_content(chunk_size=step):
        if not chunk:
            continue
        buf += chunk
        while len(buf) >= step:
            yield np.frombuffer(buf[:step], dtype=np.int16).astype(np.float32) / 32767.0
            buf = buf[step:]
    if len(buf) >= 2:
        yield np.frombuffer(buf[: len(buf) // 2 * 2], dtype=np.int16).astype(np.float32) / 32767.0


class _ASREngine:
    def transcribe(self, audio, language=None, hotwords=None):
        return _bridge_asr(audio, language, hotwords)


class _TTSEngine:
    def synthesize_stream(self, text):
        return _bridge_tts_stream(text)


# ── SmartTurn (optional) ──────────────────────────────────────────────────────

class _SmartTurnAdapter:
    """Wrap pipecat SmartTurn v3 into the segmenter's should_endpoint().

    Measured, do not "fix":
    1. append_audio()'s return is NOT the model verdict — it only returns
       COMPLETE once accumulated silence exceeds stop_secs, i.e. a silence
       fallback equivalent to our own. The model verdict is analyze_end_of_turn().
    2. So never feed silence frames: that trips the fallback and everything is
       COMPLETE (118/118 in our test).
    3. Correct use: append speech frames, then await analyze_end_of_turn().
       On Lao conference audio: 37/118 COMPLETE — real discrimination.
    4. One verdict is ~50 ms, not the ~10 ms the docs imply, so the segmenter
       runs it on a thread pool and only once silence is already established.
    """

    def __init__(self, analyzer):
        self._a = analyzer

    def should_endpoint(self, buf, is_speech_active=False) -> bool:
        from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
        self._a.clear()
        for chunk in buf:
            i16 = (np.clip(chunk, -1, 1) * 32767).astype(np.int16)
            self._a.append_audio(i16.tobytes(), is_speech=True)

        async def _go():
            st, _ = await self._a.analyze_end_of_turn()
            return st

        try:
            st = asyncio.run(_go())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                st = loop.run_until_complete(_go())
            finally:
                loop.close()
        return st == EndOfTurnState.COMPLETE


_smart_turn = None
_smart_turn_tried = False


def _get_smart_turn():
    global _smart_turn, _smart_turn_tried
    if _smart_turn_tried:
        return _smart_turn
    _smart_turn_tried = True
    if not SMART_TURN:
        return None
    try:
        from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
        a = LocalSmartTurnAnalyzerV3()
        a.set_sample_rate(16000)
        _smart_turn = _SmartTurnAdapter(a)
        log.info("SmartTurn v3 loaded")
    except Exception as e:
        log.warning("SmartTurn unavailable, silence-only endpointing: %s", e)
    return _smart_turn


# ── engines ───────────────────────────────────────────────────────────────────

async def _load_engines():
    async with _engines_lock:
        if "translator" in _engines:
            return
        loop = asyncio.get_event_loop()
        if USE_LLM_TRANSLATOR:
            from voxweft.runtime.translator import HybridTranslator
            log.info("translator: HybridTranslator LLM=%s (NLLB fallback opt-in)", LLM_DIR)
            tr = HybridTranslator(LLM_DIR, GLOSSARY_PATH,
                                  use_llm_zh_to_lo=USE_LLM_ZH_TO_SEA)
            # Pre-warm so the first real turn doesn't pay load + first-generation
            # cold start (~1 s for a 4B model on M3).
            try:
                await loop.run_in_executor(None, tr.warmup)
            except Exception as e:
                log.warning("translator warmup failed: %s", e)
        else:
            from voxweft.runtime.translator_nllb import NLLBTranslator
            nllb_dir = os.environ.get("VOXWEFT_NLLB_DIR",
                                      os.path.expanduser("~/.voxweft/models/nllb-200-1.3B"))
            log.warning("VOXWEFT_LLM_DIR not set — falling back to NLLB at %s "
                        "(CC-BY-NC-4.0: fine for research, not for redistribution)", nllb_dir)
            tr = NLLBTranslator(nllb_dir, device=os.environ.get("VOXWEFT_NLLB_DEVICE", "mps"))
            await loop.run_in_executor(None, tr.load)
        _engines["translator"] = tr
        log.info("engines ready (bridge=%s)", BRIDGE_URL)


# ── the endpoint ──────────────────────────────────────────────────────────────

@app.websocket("/ws/voice/v2")
async def ws_voice_v2(ws: WebSocket, token: str = Query(default="")):
    if not _check_api_key(token):
        await ws.close(code=4003, reason="Invalid API key")
        return
    await ws.accept()
    session_id = f"vw-{int(time.time() * 1000)}"
    slog = logging.LoggerAdapter(log, {"session": session_id})
    slog.info("session connected")

    await _load_engines()
    _dead = False

    async def send_json(data: dict):
        nonlocal _dead
        if _dead:
            return
        try:
            await ws.send_text(json.dumps(data, ensure_ascii=False))
        except Exception as e:
            _dead = True
            slog.warning("send_json failed: %s", e)

    async def send_bytes(data: bytes):
        nonlocal _dead
        if _dead:
            return
        try:
            await ws.send_bytes(data)
        except Exception as e:
            _dead = True
            slog.warning("send_bytes failed: %s", e)

    state = {"mode": "interpreter", "src": "zho_Hans", "tgt": "eng_Latn",
             "hotwords": [], "orch": None}
    ptt_active = False
    frame_buf = b""

    async def _build_orch():
        """Rebuild the orchestrator for the current session state (language or
        mode changed ⇒ routes changed). The translation canon is deliberately
        NOT inherited: it is keyed to a target language, and the most common
        reason to rebuild is that the target language changed."""
        from voxweft.runtime.orchestrator import InProcessChatLLM, Orchestrator
        old = state["orch"]
        engines = {"asr": _ASREngine(), "qwen3asr": _ASREngine(),
                   "whisper": _ASREngine(), "tts": _TTSEngine(),
                   "translate": _engines["translator"], "filler": None}
        if state["mode"] == "agent":
            tr = _engines["translator"]
            if hasattr(tr, "chat_stream"):
                engines["llm"] = InProcessChatLLM(tr)
            else:
                slog.warning("agent mode needs an LLM translator; set VOXWEFT_LLM_DIR")
        orch = Orchestrator(
            mode=state["mode"], src_lang=state["src"], tgt_lang=state["tgt"],
            send_json=send_json, send_bytes=send_bytes, engines=engines,
            hotwords=state["hotwords"] or None, default_hotwords=DEFAULT_HOTWORDS,
            smart_turn=_get_smart_turn(), use_smart_turn=SMART_TURN,
            use_llm_translator=USE_LLM_TRANSLATOR, use_llm_zh_to_sea=USE_LLM_ZH_TO_SEA,
            voice_role="agent_default" if state["mode"] == "agent" else "interpreter",
            system_prompt=AGENT_SYSTEM_PROMPT)
        state["orch"] = orch
        if old is not None:
            await old.aclose()
        slog.info("orchestrator: %s", orch.routes.describe())
        return orch

    await _build_orch()
    await send_json({"type": "ready", "session_id": session_id, "version": "voxweft-0.1",
                     "mode": state["mode"], "src_lang": state["src"],
                     "tgt_lang": state["tgt"], "tts_sample_rate": 48000})

    async def _feed_bytes(data: bytes):
        nonlocal frame_buf
        frame_buf += data
        orch = state["orch"]
        while len(frame_buf) >= FRAME_BYTES:
            raw, frame_buf = frame_buf[:FRAME_BYTES], frame_buf[FRAME_BYTES:]
            f32 = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            # Track peak frame RMS per press: the segmenter's silence rule is
            # per-frame RMS, so this is the only evidence that can answer
            # "nothing was said" vs "said, but too quiet" when no segment comes out.
            rms = float(np.sqrt(np.mean(f32 * f32)))
            state["press_peak_rms"] = max(state.get("press_peak_rms", 0.0), rms)
            state["press_samples"] = state.get("press_samples", 0) + len(f32)
            orch.feed(f32)

    async def _flush_tail():
        nonlocal frame_buf
        orch = state["orch"]
        tail, frame_buf = frame_buf, b""
        if len(tail) >= 4:
            even = tail[: len(tail) // 2 * 2]
            orch.feed(np.frombuffer(even, dtype=np.int16).astype(np.float32) / 32768.0)
        await orch.flush()

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            data = msg.get("bytes")
            if data is not None:
                if ptt_active:
                    await _feed_bytes(data)
                else:
                    n = state["orphan_frames"] = state.get("orphan_frames", 0) + 1
                    if n in (1, 100, 1000):
                        slog.warning("%d audio frames received outside PTT — dropped. "
                                     "Check client ptt_start ordering.", n)
                continue
            text = msg.get("text")
            if not text:
                continue
            try:
                obj = json.loads(text)
            except Exception:
                continue
            mt = obj.get("type")

            if mt == "session_config":
                state["mode"] = obj.get("mode") or state["mode"]
                state["src"] = obj.get("src_lang") or state["src"]
                state["tgt"] = obj.get("tgt_lang") or state["tgt"]
                state["hotwords"] = obj.get("hotwords") or []
                orch = await _build_orch()
                await send_json({"type": "session_config_ack", "success": True,
                                 "mode": state["mode"], "routes": orch.routes.describe()})

            elif mt == "ptt_start":
                # Pressing implies interrupt: the user is about to say the next
                # thing, so the previous translation need not finish playing.
                frame_buf = b""
                orch = state["orch"]
                state.update(orphan_frames=0, press_peak_rms=0.0, press_samples=0,
                             press_seq0=orch.committed_count)
                await orch.interrupt("ptt_start")
                ptt_active = True
                await send_json({"type": "ptt_ack", "recording": True})

            elif mt == "ptt_end":
                ptt_active = False
                orch = state["orch"]
                await _flush_tail()
                await send_json({"type": "ptt_ack", "recording": False})
                # No segment at all must be reported explicitly — otherwise the
                # client sits on "recognizing…" forever. Ship level + threshold
                # so "too quiet" and "nothing said" can be told apart on the spot.
                if orch.committed_count == state.get("press_seq0", 0):
                    secs = state.get("press_samples", 0) / 16000.0
                    peak = state.get("press_peak_rms", 0.0)
                    thresh = orch.segmenter.sil_thresh
                    slog.warning("no segment from this press: %.2fs audio, peak RMS %.4f "
                                 "(silence threshold %.3f)", secs, peak, thresh)
                    await send_json({"type": "no_speech", "seconds": round(secs, 2),
                                     "peak_rms": round(peak, 4), "threshold": thresh,
                                     "too_quiet": bool(secs > 0.3 and peak < thresh)})

            elif mt == "interrupt":
                await state["orch"].interrupt(obj.get("reason") or "client")

            elif mt == "reset":
                ptt_active = False
                frame_buf = b""
                await state["orch"].interrupt("reset")
                await send_json({"type": "reset_ack"})

            elif mt == "text_input":
                txt = (obj.get("text") or "").strip()
                if txt:
                    asyncio.ensure_future(_text_turn(state["orch"], txt, send_json))

            elif mt == "speak":
                # Client fallback: render given text directly, skipping ASR and MT.
                # Not routed via text_input on purpose — that would re-run the
                # translation that just failed. Deliberately no interrupt either:
                # interrupt() resets the segmenter and would drop audio still
                # buffered from the same utterance.
                txt = (obj.get("text") or "").strip()
                if txt:
                    asyncio.ensure_future(_speak_text(state["orch"], txt,
                                                      int(obj.get("seq") or 0), send_json))

            elif mt == "canon":
                # Verified source→target pairs for a known demo clip. Hit ⇒ skip
                # MT. Keyed by normalized source text, not segment index: if the
                # segmenter cuts 6 where the canon has 5, index-matching would
                # voice the wrong sentence with zero latency and no error.
                from voxweft.runtime.orchestrator import canon_key
                pairs = obj.get("pairs") or {}
                orch = state["orch"]
                orch.canon = {canon_key(k): v for k, v in pairs.items()
                              if isinstance(k, str) and isinstance(v, str) and v.strip()}
                await send_json({"type": "canon_ack", "accepted": len(orch.canon)})

    except WebSocketDisconnect:
        slog.info("disconnected")
    except Exception as e:
        slog.error("session error: %s", e, exc_info=True)
    finally:
        if state.get("orch") is not None:
            await state["orch"].aclose()
        slog.info("session closed")


async def _text_turn(orch, text: str, send_json):
    """text_input: skip ASR. Interpreter ⇒ MT ⇒ TTS; agent ⇒ LLM ⇒ TTS."""
    try:
        if orch.mode == "agent":
            llm = orch.engines.get("llm")
            if llm is None:
                return
            full, n = "", 0
            for s in llm.stream_sentences(text, system_prompt=orch.system_prompt,
                                          history=orch._history):
                full += s["sentence"]
                n += 1
                await send_json({"type": "llm_sentence", "text": s["sentence"],
                                 "sentence_idx": n - 1})
            orch._history.append({"role": "user", "content": text})
            orch._history.append({"role": "assistant", "content": full})
            await send_json({"type": "llm", "text": full, "sentences": n})
            await orch.send_json({"type": "turn_end", "sentences": n, "total_ms": 0})
        else:
            tr = await orch._run_blocking(orch.engines["translate"].translate,
                                          text, orch.routes.src, orch.routes.tgt)
            await send_json({"type": "translation", "seq": 0, "src": text,
                             "text": tr.get("text", ""),
                             "latency_ms": round(tr.get("latency_ms", 0))})
            await orch._synth_and_send(0, tr.get("text", ""), orch.routes.voice_role)
            await send_json({"type": "turn_end", "sentences": 1, "total_ms": 0})
    except Exception as e:
        await send_json({"type": "error", "stage": "text_input", "message": str(e)})


async def _speak_text(orch, text: str, seq: int, send_json):
    try:
        await orch._synth_and_send(seq, text, orch.routes.voice_role)
        await send_json({"type": "turn_end", "sentences": 1, "total_ms": 0})
    except Exception as e:
        await send_json({"type": "error", "stage": "speak", "message": str(e)})


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="voxweft serve")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "3100")))
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args(argv)
    log.info("voxweft runtime on ws://%s:%d/ws/voice/v2 (bridge %s, LLM %s)",
             args.host, args.port, BRIDGE_URL, LLM_DIR or "<none: NLLB fallback>")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
