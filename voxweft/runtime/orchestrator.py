"""VoxWeft 编排 — 同传(interpreter) + 语音对话(agent)，同一端点两种 mode。

替代 ws_server.py 里的 `ws_crossweft` 批量实现。区别在输入侧：

    旧 (v1):  ptt_start 攒 → ptt_end 一次性 detect_full 切段 → 逐段出
    新 (v2):  边说边切（IncrementalSegmenter）→ 段一到就进流水线

模式 A 同传：段进 → ASR → 翻译 → TTS，级联重叠，段 i 的音频还在合成时
段 i+1 已经在识别。没有上下文、没有人格——它转述用户的话。

模式 B 对话：段进 → ASR → LLM(流式切句) → 逐句 TTS。有历史、有人格、
有语气词填空窗。它回应用户的话。

两条链共用 LanguageRouter（选后端）和下行事件词汇（客户端一套解析）。
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import unicodedata
from typing import Awaitable, Callable, Optional

import numpy as np

from voxweft.runtime import router as LR
from voxweft.runtime.segmenter import IncrementalSegmenter, CommittedSegment

log = logging.getLogger("voxweft.v2")

SR = 16000
OUT_SR = 48000


def canon_key(s: str) -> str:
    """译文底本的匹配键：去标点、去空白、去符号、转小写。

    ⚠️ 这个函数和 App 里 SessionController.sameSentence 的归一化**必须一致**，
    两边算出不同的键 = 底本永远不命中，而且是无声地不命中（表现为「开关打开了
    但延迟没变」，没人会往归一化上想）。所以：

      · 这边按 Unicode 大类删（P* 标点 / Z* 分隔 / S* 符号），Swift 那边用的是
        CharacterSet.punctuationCharacters ∪ .whitespacesAndNewlines ∪ .symbols
        ——正是同样的三类，这不是巧合，是照着对方写的。
      · 服务端在下发底本时会回一条 canon_ack 报告收下了几条，客户端首次命中
        也会打日志。两头都留了痕迹，对不齐能当场看出来。

    为什么要归一化而不是直接 ==：ASR 的标点是模型**加**上去的，同一段音频
    两次识别可能一次给「在万象，」一次给「在万象」。为这点差别放弃命中不值。
    """
    out = []
    for ch in unicodedata.normalize("NFKC", s):
        if ch.isspace():
            continue
        if unicodedata.category(ch)[0] in ("P", "Z", "S"):
            continue
        out.append(ch)
    return "".join(out).lower()


class _Cancelled(Exception):
    """用户打断——刻意穿过流水线的信号，不是错误。"""


class Orchestrator:
    """一条会话。构造时锁定模式与语言对，之后只喂帧和事件。"""

    def __init__(
        self,
        *,
        mode: str,
        src_lang: str,
        tgt_lang: str,
        send_json: Callable[[dict], Awaitable[None]],
        send_bytes: Callable[[bytes], Awaitable[None]],
        engines: dict,
        hotwords: Optional[list[str]] = None,
        default_hotwords: Optional[dict] = None,
        voice_role: Optional[str] = None,
        smart_turn=None,
        use_smart_turn: bool = True,
        use_llm_translator: bool = True,
        use_llm_zh_to_sea: bool = False,
        system_prompt: str = "",
        max_history: int = 20,
        reply_lang: Optional[str] = None,
    ):
        self.mode = mode
        self.send_json = send_json
        self.send_bytes = send_bytes
        self.engines = engines          # {"asr":…, "sea_asr":…, "whisper":…, "translate":…, "tts":…, "llm":…}

        self.routes = LR.SessionRoutes(
            src_lang, tgt_lang, mode=mode, hotwords=hotwords,
            default_hotwords=default_hotwords,
            use_llm_translator=use_llm_translator,
            use_llm_zh_to_sea=use_llm_zh_to_sea,
            voice_role=voice_role,
        )
        # 对话模式的回复语言默认等于用户语言（同语言问答），也可显式指定
        self.reply_lang = LR.normalize(reply_lang or tgt_lang)

        self._seq = 0
        self._cancel = asyncio.Event()
        self._seg_queue: asyncio.Queue = asyncio.Queue()
        self._tasks: set[asyncio.Task] = set()
        self._dead = False
        self._loop = asyncio.get_event_loop()

        # 同一时刻只允许一路在合成。
        #
        # worker 本来就是串行的，正常那条路上这把锁一次都不会争用——它是为
        # **worker 之外**的合成准备的：ws_server 的 `speak`（客户端兜底通道）
        # 是 ensure_future 起的，和正在跑的那一轮并行。两路同时往 send_bytes
        # 里灌块，客户端那边是同一个播放节点、按到达顺序排队，两段语音会被
        # 交织成一串谁也听不懂的东西。排队会让兜底那句晚一点出来，但至少是
        # 一句完整的话。
        self._synth_lock = asyncio.Lock()

        # 故障注入档位（见 _turn_interpreter）。默认空 = 关。
        self._fault = os.environ.get("VOXWEFT_FAULT_INJECT", "").strip().lower()
        if self._fault:
            log.warning("⚠️⚠️ 故障注入已开启：VOXWEFT_FAULT_INJECT=%s "
                        "——这是演练开关，展会前务必确认它是空的", self._fault)

        # 译文底本（canon）。键是**归一化后的源文**，值是核对过的译文。
        #
        # 客户端在开始播放一条示范素材时下发这一整条素材的底本，播完清空。
        # 命中时跳过翻译那一环直接进 TTS，未命中照常走翻译。
        #
        # 为什么是「按源文匹配」而不是「按段号」：段号是 VAD 切出来的，同一个
        # 音频文件理论上每次切一样，但只要有一次切成了 6 段而底本是 5 条，
        # 按段号取就会把第 4 句的译文配到第 3 句上——机器会流利地说出一句和
        # 屏幕上原文无关的话。按源文匹配的话，切得不一样 = 匹配不上 = 自动
        # 退回实时翻译，最坏情况只是没省下那一次推理。
        #
        # 这同时解释了为什么 PT 设备说的话永远不会命中：现场没有人会一字不差
        # 地说出素材里的句子。「素材走底本、设备走实时」这条规则不需要另加
        # 判断，它是匹配规则的自然结果。
        self.canon: dict[str, str] = {}

        # 对话模式的状态
        self.llm_missing = False          # set by the host when agent mode has no LLM
        self.system_prompt = system_prompt
        self._history: list[dict] = []
        self._max_history = max_history

        self.segmenter = IncrementalSegmenter(
            self._on_segment,
            # Keep long conference turns responsive. Natural silence/SmartTurn
            # still wins; this is only the fallback for uninterrupted speech.
            max_utterance_s=float(os.environ.get("VOXWEFT_MAX_UTTERANCE_S", "6.0")),
            # Adaptive noise floor: measured CER 16.3% → 10% on phone-mic
            # audio vs the fixed 0.035 stage-mic threshold (DEVLOG §4.3).
            adaptive=os.environ.get("VOXWEFT_ADAPTIVE_THRESH", "1") == "1",
            smart_turn=smart_turn,
            use_smart_turn=use_smart_turn)

        # 常驻 worker：段一到就开工，不用每次起任务
        self._worker = self._track(self._run_worker(), name="worker")

    # ── 生命周期 ────────────────────────────────────────────────────────────

    def _track(self, coro, *, name: str = "") -> asyncio.Task:
        t = asyncio.ensure_future(coro)
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)
        return t

    async def aclose(self) -> None:
        self._dead = True
        self._cancel.set()
        for t in list(self._tasks):
            if not t.done():
                t.cancel()
        self._seg_queue.put_nowait(None)   # 唤醒 worker 退出
        try:
            await asyncio.wait_for(asyncio.shield(self._worker), timeout=1.0)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            pass
        self._tasks.clear()

    # ── 输入 ────────────────────────────────────────────────────────────────

    @property
    def committed_count(self) -> int:
        """本会话至今切出过多少段。

        调用方拿它做差，判断「这一次按压到底有没有产生过语音段」。
        为什么需要这个判断：切分器把整段都判成静音时，flush 是**静悄悄**
        返回的——没有段、没有回调、没有报错（见 IncrementalSegmenter.flush
        的 else 分支）。客户端那边发完 ptt_end 就把界面点到 ASR 那一格等着，
        而这一格永远不会被点灭。展台上的表现是「按了、说了、屏幕一直在闪」。

        所以「什么都没听到」必须是一个**能被观察到的事件**，而不是一段
        什么都没发生的沉默。
        """
        return self._seq

    def feed(self, float32_frame: np.ndarray) -> None:
        """收一帧 16k float32。热路径，必须快。"""
        if self._dead:
            return
        self.segmenter.feed(float32_frame)

    async def flush(self) -> None:
        """ptt_end / 会话结束：把尾段冲出来等它做完。"""
        await self.segmenter.flush()

    async def interrupt(self, reason: str = "client") -> None:
        """打断：停当前轮、清下游队列、通知客户端停播。

        同传模式：丢弃未播的译文（无上下文，丢了不影响后续）。
        对话模式：已生成的文本进历史（已说出口的话要记住，否则下轮错乱）。
        """
        self._cancel.set()
        self._drain_queue()
        self.segmenter.reset()
        await self.send_json({"type": "interrupted", "reason": reason})
        log.info("interrupted (%s)", reason)
        # 新一轮：清掉 cancel 让 worker 继续
        self._cancel = asyncio.Event()

    def _drain_queue(self) -> None:
        """丢掉未开工的段。**不能吃掉 None 哨兵**——那是 aclose() 用来唤醒
        worker 退出的信号，被 drain 掉的话 worker 会永远阻塞在 get() 上。"""
        kept_none = False
        while True:
            try:
                item = self._seg_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is None:
                kept_none = True
        if kept_none:
            self._seg_queue.put_nowait(None)

    def _on_segment(self, seg: CommittedSegment) -> None:
        """切分器回调（同步）。交给常驻 worker。"""
        self._seq += 1
        self._seg_queue.put_nowait((self._seq, seg))
        # 让客户端看见切分发生——增量是否按预期工作，联调时全靠这个
        self._track(self.send_json({
            "type": "segment_committed", "seq": self._seq,
            "ms": round(seg.duration_s * 1000), "reason": seg.reason,
        }))

    # ── worker：串行取段 → 派发到流水线 ──────────────────────────────────────

    async def _run_worker(self) -> None:
        while not self._dead:
            item = await self._seg_queue.get()
            if item is None:
                break
            seq, seg = item
            try:
                if self.mode == "agent":
                    await self._turn_agent(seq, seg)
                else:
                    await self._turn_interpreter(seq, seg)
            except _Cancelled:
                log.info("turn %d cancelled", seq)
            except Exception as e:
                log.exception("turn %d failed: %s", seq, e)
                await self.send_json({
                    "type": "error", "seq": seq,
                    "stage": "pipeline", "message": str(e)})

    # ── 模式 A · 同传 ───────────────────────────────────────────────────────

    async def _turn_interpreter(self, seq: int, seg: CommittedSegment) -> None:
        """一段话：ASR → 翻译 → TTS。级联由 worker 串行保证顺序。"""
        t0 = time.perf_counter()

        asr_res = await self._run_blocking(self._transcribe, seg.audio)
        if self._cancel.is_set():
            raise _Cancelled()
        src_text = (asr_res.get("text") or "").strip()
        await self.send_json({
            "type": "asr", "seq": seq, "text": src_text,
            "latency_ms": round(asr_res.get("latency_ms", 0))})
        if not src_text:
            return

        # 故障注入。**只有显式设了环境变量才生效**，默认是死的。
        #
        # 存在的理由：客户端有一套兜底逻辑（翻译崩了就从本地底账取之前跑通
        # 过的译文、交给 TTS 现场念）。没有这个开关就没有任何办法验证它——
        # 而一个没验证过的兜底比没有兜底更糟：现场真崩的时候才发现它自己
        # 也有 bug，等于白写。
        #
        #   VOXWEFT_FAULT_INJECT=translate   翻译这一环必抛（最常演练的一种）
        #   VOXWEFT_FAULT_INJECT=empty       翻译返回空串（更阴的一种：不报错）
        #
        # 每次触发都打一行 WARNING，开着忘了关不至于无声无息。
        if self._fault == "translate":
            log.warning("⚠️ 故障注入生效：translate（seq=%d）", seq)
            raise RuntimeError("fault injection: translate")

        # 底本命中就不跑翻译。见 __init__ 里 self.canon 的注释。
        #
        # 注意这一段在故障注入**之后**：演练兜底时要能看到翻译真的崩了，
        # 底本抢在前面会把演练变成空跑。
        hit = self.canon.get(canon_key(src_text)) if self.canon else None
        if hit:
            log.info("📖 底本命中 seq=%d：%s", seq, hit[:40])
            await self.send_json({
                "type": "translation", "seq": seq, "src": src_text,
                "text": hit, "latency_ms": 0, "origin": "canon"})
            await self._synth_and_send(seq, hit, self.routes.voice_role)
            if seq == self._seq:
                await self.send_json({
                    "type": "turn_end", "sentences": 1,
                    "total_ms": round((time.perf_counter() - t0) * 1000)})
            return

        tr = await self._run_blocking(
            self.engines["translate"].translate,
            src_text, self.routes.src, self.routes.tgt)
        if self._cancel.is_set():
            raise _Cancelled()
        tgt_text = (tr.get("text") or "").strip()
        if self._fault == "empty":
            log.warning("⚠️ 故障注入生效：empty（seq=%d）", seq)
            tgt_text = ""
        await self.send_json({
            "type": "translation", "seq": seq, "src": src_text,
            "text": tgt_text, "latency_ms": round(tr.get("latency_ms", 0))})
        if not tgt_text:
            # Translators report failure as {"text": "", "error": ...} rather
            # than raising (e.g. NLLB fallback with torch not installed).
            # Surface it: an empty translation with no error and no turn_end
            # leaves the client on "translating" forever.
            err = tr.get("error")
            if err:
                await self.send_json({"type": "error", "seq": seq,
                                      "stage": "translate", "message": str(err)})
            if seq == self._seq:
                await self.send_json({
                    "type": "turn_end", "sentences": 0,
                    "total_ms": round((time.perf_counter() - t0) * 1000)})
            return

        await self._synth_and_send(seq, tgt_text, self.routes.voice_role)
        if seq == self._seq:
            await self.send_json({
                "type": "turn_end", "sentences": 1,
                "total_ms": round((time.perf_counter() - t0) * 1000)})

    # ── 模式 B · 对话 ───────────────────────────────────────────────────────

    async def _turn_agent(self, seq: int, seg: CommittedSegment) -> None:
        """一段话：ASR → LLM 流式切句 → 逐句 TTS。有历史、有人格。"""
        t0 = time.perf_counter()

        asr_res = await self._run_blocking(self._transcribe, seg.audio)
        if self._cancel.is_set():
            raise _Cancelled()
        user_text = (asr_res.get("text") or "").strip()
        await self.send_json({
            "type": "asr", "seq": seq, "text": user_text,
            "latency_ms": round(asr_res.get("latency_ms", 0))})
        if len(user_text) < 2:
            return

        await self.send_json({"type": "state", "state": "thinking"})

        # 语气词：LLM 首 token 前的空窗，用预生成的音频填上
        filler = self.engines.get("filler")
        if filler is not None:
            await self._send_filler(filler)

        await self.send_json({"type": "state", "state": "speaking"})

        # LLM 流式切句 → 逐句 TTS。句在生成时就能开合成，不等整段。
        n_sent = 0
        full = ""
        llm = self.engines.get("llm")
        if llm is None:
            # Agent mode was configured without an LLM (NLLB-only translator).
            # Say so per turn instead of dying with KeyError after
            # state=speaking and leaving the client stuck on "thinking".
            await self.send_json({
                "type": "error", "seq": seq, "stage": "llm",
                "message": "agent mode needs an LLM (set VOXWEFT_LLM_DIR)"})
            await self.send_json({"type": "state", "state": "idle"})
            if seq == self._seq:
                await self.send_json({"type": "turn_end", "sentences": 0,
                                      "total_ms": round((time.perf_counter() - t0) * 1000)})
            return
        for s in llm.stream_sentences(user_text, system_prompt=self.system_prompt,
                                      history=self._history):
            if self._cancel.is_set():
                raise _Cancelled()
            sentence = s["sentence"]
            full += sentence
            n_sent += 1
            await self.send_json({
                "type": "llm_sentence", "text": sentence, "seq": seq,
                "sentence_idx": n_sent - 1,
                "latency_ms": round(s["ttfs_ms"]) if n_sent == 1 else 0})
            await self._synth_and_send(seq, sentence, self.routes.voice_role)

        if full:
            self._history.append({"role": "user", "content": user_text})
            self._history.append({"role": "assistant", "content": full})
            self._history = self._history[-self._max_history:]

        await self.send_json({
            "type": "llm", "text": full, "seq": seq, "sentences": n_sent})
        await self.send_json({"type": "state", "state": "idle"})
        await self.send_json({
            "type": "turn_end", "sentences": n_sent,
            "total_ms": round((time.perf_counter() - t0) * 1000)})

    async def _send_filler(self, filler) -> None:
        try:
            text, audio = filler.get_filler()
            if audio:
                await self.send_json({"type": "filler", "text": text})
                await self.send_bytes(audio)
        except Exception as e:
            log.debug("filler skipped: %s", e)

    # ── 共享：ASR / TTS ─────────────────────────────────────────────────────

    def _transcribe(self, audio: np.ndarray) -> dict:
        """按 LanguageRouter 选的后端做识别。阻塞，由 run_blocking 包。"""
        route = self.routes.asr
        if route.backend == "qwen3asr":
            eng = self.engines.get("qwen3asr") or self.engines["asr"]
        else:
            eng = self.engines.get("whisper")
            if eng is None:
                log.warning("whisper 未加载，回退 Qwen3-ASR 自动语言识别")
                eng = self.engines.get("qwen3asr") or self.engines["asr"]
        return eng.transcribe(audio, language=route.lang, hotwords=route.hotwords)

    async def _synth_and_send(self, seq: int, text: str, voice_role: str) -> None:
        """流式 TTS：边合成边发，首块一出来就 audio_start。

        全程持锁——见 `_synth_lock`。串行 worker 下这把锁不争用，它防的是
        兜底通道那条并行路径。
        """
        async with self._synth_lock:
            await self._synth_locked(seq, text, voice_role)

    async def _synth_locked(self, seq: int, text: str, voice_role: str) -> None:
        tts = self.engines["tts"]
        aq: asyncio.Queue = asyncio.Queue()
        spoken = text or ""

        def _producer():
            t0 = time.perf_counter()
            first = None
            try:
                # Voice: the interpreter speaks in the *target* language's
                # registered voice (VOXWEFT_VOICE_DIR/<lang>_tgt.wav), then
                # any voice named after the role, else the bridge default.
                voice = self.routes.voice_id(voice_role)
                try:
                    it = tts.synthesize_stream(spoken, voice)
                except TypeError:          # engines that predate the voice arg
                    it = tts.synthesize_stream(spoken)
                for chunk in it:
                    if first is None:
                        first = (time.perf_counter() - t0) * 1000
                    self._loop.call_soon_threadsafe(
                        aq.put_nowait, ("chunk", chunk, first))
                self._loop.call_soon_threadsafe(aq.put_nowait, ("done", None, first or 0))
            except Exception as e:
                self._loop.call_soon_threadsafe(aq.put_nowait, ("err", e, 0))

        fut = self._loop.run_in_executor(None, _producer)
        started = False
        n = 0
        while True:
            kind, val, first = await aq.get()
            if kind == "chunk":
                if self._cancel.is_set():
                    break
                if not started:
                    await self.send_json({
                        "type": "audio_start", "seq": seq,
                        "sample_rate": OUT_SR, "streamed": True,
                        "tts_latency_ms": round(first or 0)})
                    started = True
                arr = val if isinstance(val, np.ndarray) else np.asarray(val)
                i16 = (np.clip(arr, -1, 1) * 32767).astype(np.int16)
                n += len(i16)
                await self.send_bytes(i16.tobytes())
            elif kind == "done":
                if started:
                    await self.send_json({
                        "type": "audio_end", "seq": seq, "samples": n,
                        "tts_latency_ms": round(first or 0)})
                break
            else:
                log.error("TTS failed (seq=%d): %s", seq, val)
                await self.send_json({
                    "type": "error", "stage": "tts", "seq": seq,
                    "message": str(val)})
                break
        try:
            await fut
        except Exception:
            pass

    async def _run_blocking(self, fn, *a, **kw):
        return await self._loop.run_in_executor(None, lambda: fn(*a, **kw))


# ── LLM 适配：把进程内的 mlx LLM 接成「逐句 yield」───────────────────────────

_SENT_END = "。！？!?；;\n"


class InProcessChatLLM:
    """复用 translator 已加载的 LLM 做对话（agent 模式）。

    为什么不另起 mlx_lm.server：模型加载几十秒 + 几 GB 内存，而 M3 上
    翻译与对话不会同时高负载（用户要么在说、要么在听）。translator 的
    warmup() 已经把权重预加载好了，复用能让对话首句直接受益。

    切句在客户端做：bridge/LLM 吐的是 token 流，而语音要在「够合成一句」
    时立刻开工，不能等整段生成完。
    """

    def __init__(self, translator, max_tokens: int = 200):
        self._tr = translator
        self._max_tokens = max_tokens

    def stream_sentences(self, user_text: str, *, system_prompt: str = "",
                         history: Optional[list[dict]] = None):
        """阻塞生成器，逐句 yield {'sentence', 'ttfs_ms'}。在线程池里跑。"""
        t0 = time.perf_counter()
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.extend(history or [])
        msgs.append({"role": "user", "content": user_text})

        buf = ""
        first = True
        for piece in self._tr.chat_stream(msgs, max_tokens=self._max_tokens):
            buf += piece
            while True:
                idx = _first_sent_end(buf)
                if idx < 0:
                    break
                sentence, buf = buf[:idx + 1].strip(), buf[idx + 1:].lstrip()
                if len(sentence) >= 2:
                    yield {"sentence": sentence,
                           "ttfs_ms": (time.perf_counter() - t0) * 1000 if first else 0}
                    first = False
        tail = buf.strip()
        if len(tail) >= 2:
            yield {"sentence": tail,
                   "ttfs_ms": (time.perf_counter() - t0) * 1000 if first else 0}


def _first_sent_end(s: str) -> int:
    for i, ch in enumerate(s):
        if ch in _SENT_END:
            return i
    return -1
