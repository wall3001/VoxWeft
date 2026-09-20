#!/usr/bin/env python
"""VoxWeft bridge — VoxCPM2 MLX TTS + Qwen3-ASR MLX behind one HTTP process.

Only needs MLX + transformers, no CUDA/vLLM. Runs on Apple Silicon.
Endpoints:
  POST /generate                    text → 48 kHz WAV (stream optional)
  POST /v1/asr/stream/{start,chunk,finish}   float32 packet streaming
  POST /v1/chat/completions         (offline convenience)
  GET  /health
"""
import io, json, os, time, base64, asyncio, tempfile, logging, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pathlib import Path

import uvicorn
import soundfile as sf
from scipy.signal import resample_poly
from fastapi import FastAPI, Request
from fastapi.responses import Response, JSONResponse, StreamingResponse
from pydantic import BaseModel

from mlx_audio.stt.utils import load_model as load_asr_model

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("voxweft.bridge")

BASE = Path(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = Path(os.environ.get("VOXWEFT_MODEL_DIR", os.environ.get("MODEL_DIR", str(Path.home() / ".voxweft/models"))))
VoxCPM2_DIR = Path(os.environ.get("VOXCPM2_MODEL", str(MODEL_DIR / "VoxCPM2")))
ASR_DIR = Path(os.environ.get("VOXWEFT_ASR_MODEL", str(MODEL_DIR / "Qwen3-ASR-1.7B")))
WHISPER_DIR = Path(os.environ.get("VOXWEFT_WHISPER_MODEL", str(MODEL_DIR / "whisper-large-v3-turbo")))

app = FastAPI(title="voxweft bridge")


@app.on_event("shutdown")
def _shutdown_executors():
    """退出时关掉推理线程池。

    线程池默认非 daemon —— 不关的话进程退出会被挂住，launchd 的
    kickstart/重启会明显变慢（实测：重启要等好几秒）。
    """
    _tts_executor.shutdown(wait=False, cancel_futures=True)
    _asr_executor.shutdown(wait=False, cancel_futures=True)


@app.get("/health")
async def health():
    """进程活着就返回 ok。**不代表模型可用** —— 预热完成前也返回 ok。
    需要「能不能干活」的判断请用 /ready。"""
    return {"status": "ok", "service": "voxweft-bridge"}


@app.get("/ready")
async def ready():
    """/health 的严格版：两个模型都加载好、且预热跑过才算 ready。

    为什么需要它：惰性加载下，/health 从进程起来那一刻就是 ok 的，但此时
    第一个请求仍要付几秒的加载 + 图编译。watchdog 与客户端启动检查若只看
    /health，会误以为服务可用，用户就吃上那几秒。

    注意用的是 non-blocking 的检查（只看有没有加载好），不触发加载 ——
    否则一个健康探针会变成「把模型加载起来」的副作用，那是预热该干的事。
    """
    st = {
        "tts": _tts is not None,
        "asr": _asr is not None,
        "whisper": _whisper is not None,
        "warmup_done": _warmup_done.is_set(),
    }
    ok = st["tts"] and st["asr"] and st["warmup_done"]
    return JSONResponse(
        {"status": "ready" if ok else "warming", "components": st},
        status_code=200 if ok else 503,
    )

# ---------------------------------------------------------------------------
# VoxCPM2 MLX TTS
# ---------------------------------------------------------------------------
# 推理串行化：MLX 不是线程安全的，两个线程同时进同一个模型会段错误。
#
# 用**单线程 executor** 而不是全局锁：
#   * 锁会被 async 生成器跨越（在 yield 处挂起时仍持有），导致死锁
#     ——实测过一次，整个 bridge 卡死无日志
#   * 单线程 executor 让「谁在跑模型」天然互斥，且不跨越 await：
#     提交一个任务、等它完成，期间事件循环照常服务其它请求
# TTS 与 ASR 各一个（不同模型对象，互不干扰，也让 ASR 不必排在长 TTS 后面）。
_tts_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")
_asr_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr")
# 预热完成标志。/ready 用它区分「进程活着」和「能干活」。
_warmup_done = threading.Event()

# 模型加载 —— 必须用**线程锁**，不是 asyncio.Lock。
#
# 这里踩过一个会让整个服务崩溃的坑（实测）：开机预热跑在后台线程（不阻塞
# uvicorn 起来），而 uvicorn 一旦起来就开始接请求。两边若同时进加载路径，
# 会各构造一份模型实例——MLX 不是线程安全的，结果是段错误，整个 bridge
# 进程消失（launchd 记 exit -9），所有语言链路一起断。曾误判为「某段中文
# 音频有毒」，其实与音频无关。
#
# 旧写法两个问题：
#   1. asyncio.Lock 在 run_in_executor 的线程里不是同一个事件循环，锁形同虚设
#   2. 预热直接构造模型，完全绕过锁
# 现在加载一律走 threading.Lock，预热与请求路径共用同一把。
_tts = None
_tts_lock = threading.Lock()
_voice_ref_latents = None          # the "default" voice (backwards compatible)
_voice_external_id = "default"
# Named voices: id → reference latents. Filled from VOXWEFT_VOICE_DIR/*.wav
# at warm-up (id = file stem) and from POST /add_prompt. `/generate` picks
# by `voice`; unknown/absent → default → zero-shot.
_voices: dict = {}
_voice_dir = os.environ.get("VOXWEFT_VOICE_DIR", "")
_voice_prompt_wav = os.environ.get(
    "VOXWEFT_VOICE_PROMPT_WAV",
    "",
)
_tts_temperature = float(os.environ.get("VOXWEFT_TTS_TEMPERATURE", "1.0"))
_tts_cfg = float(os.environ.get("VOXWEFT_TTS_CFG", "2.0"))
_tts_warmup_patches = int(os.environ.get("VOXWEFT_TTS_WARMUP_PATCHES", "0"))
_tts_inference_steps = int(os.environ.get("VOXWEFT_TTS_STEPS", "6"))
_tts_warmup_text = os.environ.get("VOXWEFT_TTS_WARMUP_TEXT",
                                  "欢迎使用语音合成服务，这是一次预热。")


def _decode_audio_16k(payload: bytes) -> np.ndarray:
    """WAV/FLAC/OGG bytes → float32 mono @16 kHz. Falls back to raw int16
    @16 kHz only when the bytes are not a recognizable container."""
    try:
        audio, sr = sf.read(io.BytesIO(payload), dtype="float32")
    except Exception:
        return np.frombuffer(payload, dtype=np.int16).astype(np.float32) / 32768.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = resample_poly(audio, 16000, sr).astype(np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32)


_asr_supported: set | None = None


def _asr_language_hint(asr, name: str | None) -> str | None:
    """Return `name` if the loaded Qwen3-ASR lists it in config.support_languages,
    else None (auto language-ID). mlx-audio passes unknown names through into
    the decoder prefix verbatim, and a label the model never saw yields
    plausible-looking output in the wrong script."""
    global _asr_supported
    if not name:
        return None
    if _asr_supported is None:
        langs = getattr(getattr(asr, "config", None), "support_languages", None) or []
        _asr_supported = {str(l).lower() for l in langs}
    if not _asr_supported or name.lower() in _asr_supported:
        return name
    return None


def _load_tts_sync():
    """同步加载 TTS。**必须在 _tts_executor 线程里调用**。

    为什么线程要紧：MLX 的 GPU stream 是线程绑定的（`Stream(gpu, 0)`
    属于创建它的线程）。模型在 A 线程加载、在 B 线程推理，会报
    「There is no Stream(gpu, 0) in current thread」并失败。
    所以加载与推理都在同一个单线程 executor 上，天然串行也天然同线程。
    """
    global _tts
    from voxweft.core import VoxCPM2Pipeline
    log.info("loading VoxCPM2 MLX from %s", VoxCPM2_DIR)
    _tts = VoxCPM2Pipeline(str(VoxCPM2_DIR),
                           inference_timesteps=_tts_inference_steps,
                           quantize_lm=4)
    return _tts


def _ensure_tts_locked():
    with _tts_lock:
        if _tts is None:
            _load_tts_sync()
    return _tts


def _prepare_voice_prompt(wav_bytes: bytes) -> np.ndarray:
    """Decode a prompt and normalize it to VoxCPM2's 16 kHz mono input."""
    audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = resample_poly(audio, 16000, sr).astype(np.float32)
    audio = np.asarray(np.clip(audio, -1.0, 1.0), dtype=np.float32)

    # Keep a small context at both edges, but do not encode long leading or
    # trailing silence as part of the speaker prompt. This follows the
    # reference VoxCPM2 preprocessing and prevents silence from becoming an
    # unwanted onset condition.
    frame = 512
    rms = np.sqrt(np.convolve(
        audio * audio, np.ones(frame, dtype=np.float32) / frame, mode="same"))
    active = np.flatnonzero(rms >= max(0.008, float(rms.max()) * 10 ** (-35 / 20)))
    if active.size:
        margin = int(0.2 * 16000)
        start = max(0, int(active[0]) - margin)
        end = min(len(audio), int(active[-1]) + margin)
        audio = audio[start:end]
    return audio


def _install_voice_prompt_sync(wav_bytes: bytes, external_id: str = "default"):
    """Encode and cache one stable speaker prompt on the TTS executor."""
    global _voice_ref_latents, _voice_external_id
    tts = _ensure_tts_locked()
    wav = _prepare_voice_prompt(wav_bytes)
    latents = tts.encode_reference(wav)
    patch = tts.patch
    chunk = tts._encoder.chunk
    if latents.ndim != 2 or latents.shape[1] != tts.feat_dim:
        raise ValueError(f"unexpected reference latent shape: {latents.shape}")
    if latents.shape[0] % patch != 0:
        raise ValueError(
            f"reference latent frames not patch-aligned: "
            f"frames={latents.shape[0]} patch={patch}")
    external_id = external_id or "default"
    _voices[external_id] = latents
    if external_id == "default" or _voice_ref_latents is None:
        _voice_ref_latents = latents
        _voice_external_id = external_id
    log.info("voice prompt installed: id=%s samples=%d (%.2fs) "
             "encoder_chunk=%d patch=%d latent_frames=%d",
             _voice_external_id, len(wav), len(wav) / 16000,
             chunk, patch, len(latents))
    return _voice_external_id


def _load_voice_dir_sync():
    """Install every <id>.wav in VOXWEFT_VOICE_DIR as a named voice."""
    if not _voice_dir:
        return
    d = Path(_voice_dir).expanduser()
    if not d.is_dir():
        log.warning("VOXWEFT_VOICE_DIR not a directory: %s", d)
        return
    for wav in sorted(d.glob("*.wav")):
        try:
            _install_voice_prompt_sync(wav.read_bytes(), wav.stem)
        except Exception as e:  # noqa: BLE001
            log.warning("voice %s not installed: %s", wav.name, e)


def _pick_voice(voice: str | None):
    if voice and voice in _voices:
        return _voices[voice]
    return _voice_ref_latents


def _load_default_voice_sync():
    _load_voice_dir_sync()
    if not _voice_prompt_wav:
        if not _voices:
            log.info("no VOXWEFT_VOICE_PROMPT_WAV / VOXWEFT_VOICE_DIR — zero-shot voice until /add_prompt")
        return
    path = Path(_voice_prompt_wav)
    if not path.exists():
        log.warning("voice prompt not found: %s", path)
        return
    _install_voice_prompt_sync(path.read_bytes(), "default")


async def _get_tts():
    if _tts is not None:
        return _tts
    # 丢到 _tts_executor（而不是默认线程池）——加载必须发生在实际跑推理的
    # 那个线程上，见 _load_tts_sync 的说明。
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_tts_executor, _ensure_tts_locked)


class GenReq(BaseModel):
    target_text: str
    voice_type: str | None = None
    external_id: str | None = None
    voice: str | None = None            # named voice from VOXWEFT_VOICE_DIR / add_prompt
    stream: bool = False
    # Defaults follow the process env so a bare {"target_text"} behaves like
    # the operator configured; an explicit value in the body wins.
    temperature: float = _tts_temperature
    cfg_value: float = _tts_cfg
    do_normalize: bool = True


@app.post("/generate")
async def generate(req: GenReq):
    tts = await _get_tts()
    loop = asyncio.get_running_loop()

    if req.stream:
        # 流式 TTS 在**线程**里跑，每个 chunk 通过队列送回事件循环。
        #
        # 为什么不用 async 生成器 + 全局推理锁（曾经写成那样，结果是死锁）：
        # async 生成器会在每个 `yield` 处被挂起，若把锁持有到整个响应结束，
        # 那么客户端（ws_server）在收 chunk 的间隙去发 ASR 请求时，ASR 也要
        # 同一把锁 —— 自己等自己，整个 bridge 卡死。实测：/ready 超时、
        # CPU 50% 空转、无任何错误日志。
        #
        # 正确做法：让模型调用在**单个线程**里串行（MLX 非线程安全，
        # 这是真需求），通过队列把结果送回事件循环。锁不跨越 await。
        aq: asyncio.Queue = asyncio.Queue(maxsize=64)

        def _produce():
            try:
                for chunk_f32, _ in tts.generate_streaming(
                        req.target_text,
                        ref_latents=_pick_voice(req.voice or req.external_id),
                        temperature=req.temperature,
                        cfg_value=req.cfg_value,
                        warmup_patches=_tts_warmup_patches):
                    i16 = (np.clip(chunk_f32, -1, 1) * 32767).astype(np.int16)
                    loop.call_soon_threadsafe(aq.put_nowait, i16.tobytes())
            except Exception as e:            # noqa: BLE001
                loop.call_soon_threadsafe(aq.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(aq.put_nowait, None)

        async def _stream_pcm():
            # 全部 TTS 生成都丢到同一个单线程 executor（见 _tts_executor），
            # 天然串行，不需要额外加锁。
            loop.run_in_executor(_tts_executor, _produce)
            while True:
                item = await aq.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    # Status 200 is already on the wire. Raising here makes
                    # Starlette drop the connection mid-body, which the client
                    # sees as a transport error — the only signal we can still
                    # send. Swallowing it would look like a clean short sentence.
                    log.error("TTS stream failed: %s", item)
                    raise item
                yield item

        return StreamingResponse(_stream_pcm(), media_type="application/octet-stream")

    def _full():
        return tts.generate(req.target_text,
                            ref_latents=_pick_voice(req.voice or req.external_id),
                            temperature=req.temperature,
                            cfg_value=req.cfg_value)

    wav, _ = await loop.run_in_executor(_tts_executor, _full)
    int16 = (np.clip(wav, -1, 1) * 32767).astype(np.int16)
    bio = io.BytesIO()
    sf.write(bio, int16, 48000, format="WAV", subtype="PCM_16")
    return Response(content=bio.getvalue(), media_type="audio/wav")


@app.post("/add_prompt")
async def add_prompt(req: Request):
    """Install a voice prompt into the MLX pipeline."""
    body = await req.json()
    wav_b64 = body.get("wav_base64") or body.get("audio_base64")
    external_id = body.get("external_id", "default")
    if not wav_b64:
        return JSONResponse({"error": "wav_base64 is required"}, status_code=400)
    try:
        wav_bytes = base64.b64decode(wav_b64)
        loop = asyncio.get_running_loop()
        installed = await loop.run_in_executor(
            _tts_executor,
            lambda: _install_voice_prompt_sync(wav_bytes, external_id),
        )
        return {"external_id": installed, "installed": True}
    except Exception as e:
        log.exception("voice prompt install failed")
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/voices")
@app.get("/available_prompts")
async def voices():
    ids = list(_voices)
    return {"prompts": [{"external_id": i} for i in ids],
            "voices": [{"external_id": i} for i in ids],
            "default": _voice_external_id if _voice_ref_latents is not None else None}


# ---------------------------------------------------------------------------
# Qwen3 ASR (MLX) — OpenAI-compatible chat completions (audio_url data URI)
# ---------------------------------------------------------------------------
_asr = None
_asr_lock = threading.Lock()


def _load_asr_sync():
    """同步加载 ASR。调用方必须持 _asr_lock。"""
    global _asr
    log.info("loading MLX ASR from %s", ASR_DIR)
    _asr = load_asr_model(str(ASR_DIR))
    return _asr


def _ensure_asr_locked():
    with _asr_lock:
        if _asr is None:
            _load_asr_sync()
    return _asr


async def _get_asr():
    if _asr is not None:
        return _asr
    # 同 TTS：加载必须发生在 _asr_executor 线程上（MLX stream 线程绑定）
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_asr_executor, _ensure_asr_locked)


# Whisper-large-v3 (MLX) for languages the loaded Qwen3-ASR does not list —
# stock Qwen3-ASR has no Lao / Khmer / Burmese; Whisper does.
_whisper = None
_whisper_lock = threading.Lock()


class _Whisper:
    """Thin wrapper over mlx_whisper so the call shape matches Qwen3-ASR's.

    Why mlx_whisper and not mlx_audio's whisper: the latter needs the
    HuggingFace processor/tokenizer files next to the weights, which the
    ModelScope mirrors of mlx-community/whisper-* do not ship; mlx_whisper
    only needs config.json + weights.safetensors and is what bench/ validated.
    """

    def __init__(self, path: str):
        import mlx_whisper  # noqa: F401  (import here: optional dependency)
        self._path = path

    def generate(self, audio: np.ndarray, language: str | None = None):
        import mlx_whisper
        r = mlx_whisper.transcribe(
            audio, path_or_hf_repo=self._path, language=language, task="transcribe",
            fp16=True, condition_on_previous_text=False, temperature=0.0,
            no_speech_threshold=0.6)
        return (r.get("text") or "").strip()


def _load_whisper_sync():
    """同步加载 Whisper。调用方必须持 _whisper_lock。"""
    global _whisper
    if not (WHISPER_DIR / "config.json").exists():
        raise RuntimeError("whisper not downloaded: %s" % WHISPER_DIR)
    log.info("loading Whisper (mlx_whisper) from %s", WHISPER_DIR)
    _whisper = _Whisper(str(WHISPER_DIR))
    return _whisper


def _ensure_whisper_locked():
    with _whisper_lock:
        if _whisper is None:
            _load_whisper_sync()
    return _whisper


async def _get_whisper():
    if _whisper is not None:
        return _whisper
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_asr_executor, _ensure_whisper_locked)


@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    """Accept Qwen3RemoteASR format: {model, messages:[{content:[{type:audio_url,
    audio_url:{url: data:audio/wav;base64,...}}]}], language, hotword}"""
    body = await req.json()
    audio_url = ""
    for m in body.get("messages", []):
        for c in m.get("content", []):
            if isinstance(c, dict) and c.get("type") == "audio_url":
                audio_url = c["audio_url"]["url"]
    if not audio_url.startswith("data:"):
        return JSONResponse({"error": "expected data URL"}, status_code=400)
    b64 = audio_url.split(",", 1)[1]
    wav_bytes = base64.b64decode(b64)
    # Decode as a container, not raw PCM: the runtime sends a RIFF WAV, and
    # np.frombuffer on the whole payload turns the 44-byte header into 22
    # near-full-scale samples — a click prepended to every ASR segment.
    audio = _decode_audio_16k(wav_bytes)

    lang = body.get("language")
    hotwords = body.get("hotword") or []
    asr = await _get_asr()
    # Auto-LID is unreliable on 1–3 s clips (Lao↔Thai), so the runtime always
    # sends an explicit language; direct HTTP callers get a configurable default.
    if lang in (None, "", "auto"):
        lang = os.environ.get("VOXWEFT_ASR_DEFAULT_LANG", "Chinese")
    from voxweft.runtime.router import FLORES_TO_NAME, normalize, whisper_lang
    code = normalize(str(lang))
    name = FLORES_TO_NAME.get(code, str(lang))

    # Both models are thread-pinned (MLX is not thread-safe); ASR has its own
    # executor so it never queues behind a long TTS generation.
    loop = asyncio.get_running_loop()
    hint = _asr_language_hint(asr, name)
    have_whisper = _whisper is not None or (WHISPER_DIR / "config.json").exists()
    if hint is not None or not have_whisper:
        # Qwen3-ASR: with the label when it knows it, auto-LID otherwise
        # (also the path when no Whisper is installed).
        if hint is None:
            log.warning("ASR label %r not in Qwen3-ASR support_languages and no "
                        "Whisper at %s — auto-LID", name, WHISPER_DIR)
        text = await loop.run_in_executor(
            _asr_executor,
            lambda: asr.generate(audio, language=hint, hotwords=hotwords))
    else:
        w = await _get_whisper()
        iso = whisper_lang(code)
        res = await loop.run_in_executor(
            _asr_executor,
            lambda: w.generate(audio, language=iso if len(iso) == 2 else None))
        text = getattr(res, "text", str(res))
    text = getattr(text, "text", str(text)).strip()
    return {"choices": [{"message": {"content": f"language {lang}<asr_text>{text}"}}]}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="voxweft bridge",
                                 description="VoxCPM2 TTS + Qwen3-ASR over HTTP (MLX, thread-pinned)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8133")))
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args(argv)
    port = args.port

    # ── 开机预热 ────────────────────────────────────────────────────────────
    #
    # 不预热的话，服务第一个真实请求要付全部冷启动：加载权重 + 首次推理的
    # 图编译。实测 TTS 首块延迟 2149ms（冷）vs 62ms（热），差 35 倍——展厅
    # 演示时这就是「第一句话卡住」。
    #
    # 预热在一次性的后台线程里跑，不阻塞 uvicorn 起来（否则 launchd 的
    # 健康检查会以为服务半天没起）。没预热完就来的请求会正常走懒加载，
    # 只是慢——不会出错。
    def _warmup():
        """预热：加载两个模型 + 各跑一次真实推理。

        走的是和请求路径**同一套** _ensure_*_locked()，所以两边不会同时
        构造模型实例。这是之前崩溃（段错误、整个进程消失）的根因——
        旧版预热直接构造模型绕过了锁。

        放后台线程是为了不阻塞 uvicorn 起来：launchd 的健康检查要在几秒内
        看到端口活着，而加载要几秒。没预热完就来的请求会等同一把锁，
        不会出错，只是慢一点。
        """
        def _go():
            t0 = time.perf_counter()
            try:
                # TTS：加载 + 跑前几个 patch（编译/缓存在这一步建好）。
                # **整个流程都提交给 _tts_executor** —— 加载与推理必须同线程
                # （MLX stream 线程绑定），所以不能在别的线程先加载再传进来。
                t1 = time.perf_counter()

                def _prime_tts():
                    tts = _ensure_tts_locked()
                    log.info("warmup: TTS loaded in %.1fs", time.perf_counter() - t1)
                    _load_default_voice_sync()
                    t2 = time.perf_counter()
                    # Run a realistic sentence to completion. Priming with
                    # "你好" and breaking after 3 patches left the first real
                    # request paying ~450 ms of compile (M2: 747 ms first
                    # byte vs 305 ms for every request after it). The
                    # stop head and a normal-length prefill only get
                    # exercised on a full generation. DEVLOG §3.1.
                    n_patches = 0
                    for _ in tts.generate_streaming(
                            _tts_warmup_text, ref_latents=_voice_ref_latents,
                            temperature=_tts_temperature, cfg_value=_tts_cfg,
                            warmup_patches=_tts_warmup_patches):
                        n_patches += 1
                    log.info("warmup: TTS primed in %.1fs (%d patches)",
                             time.perf_counter() - t2, n_patches)

                _tts_executor.submit(_prime_tts).result(timeout=300)

                t3 = time.perf_counter()

                def _prime_asr():
                    asr = _ensure_asr_locked()
                    asr.generate(np.zeros(16000, np.float32),
                                 language="chinese", hotwords=[])
                    log.info("warmup: ASR primed in %.1fs", time.perf_counter() - t3)

                _asr_executor.submit(_prime_asr).result(timeout=300)

                # Re-prime TTS *after* ASR has loaded. Measured on M2 16 GB:
                # with TTS primed first and ASR loaded second, the first real
                # TTS request still paid ~450 ms (720 ms vs 305 ms steady)
                # regardless of how long the TTS warm-up ran. Loading a second
                # multi-GB model evicts the first one's MLX buffer cache; the
                # cheapest fix is one short generation once both are resident.
                t4 = time.perf_counter()

                def _reprime_tts():
                    for _ in _tts.generate_streaming(
                            "再热一次。", ref_latents=_voice_ref_latents,
                            temperature=_tts_temperature, cfg_value=_tts_cfg,
                            warmup_patches=_tts_warmup_patches):
                        pass

                _tts_executor.submit(_reprime_tts).result(timeout=120)
                log.info("warmup: TTS re-primed after ASR in %.1fs", time.perf_counter() - t4)

                log.info("warmup done (total %.1fs)", time.perf_counter() - t0)
                _warmup_done.set()
            except Exception as e:
                # 预热失败不影响服务——回退懒加载
                log.warning("warmup failed (will load lazily): %s", e)

        threading.Thread(target=_go, daemon=True, name="warmup").start()

    _warmup()
    uvicorn.run(app, host=args.host, port=port, log_level="info")


if __name__ == "__main__":
    main()
