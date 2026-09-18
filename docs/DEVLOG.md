# DEVLOG — what broke, what we were wrong about, and what it cost

> This file is a first-class deliverable, not an appendix. A working demo can
> be written by any LLM; *where it went wrong and why* cannot. Every entry
> below was hit for real on M2 / M3 Ultra hardware between 2026-08 and 2026-09.
> Each entry: **symptom → root cause → fix → what it cost us**. Entries marked
> ⚠️ are the ones most likely to bite you again.

Structure per entry follows the triage we ended up using for everything:
**confirmed / disproved / unverified**. Writing down what we *disproved*
turned out to matter as much as what we fixed — it is the only thing that
stops the next person from re-running the same dead-end investigation.

---

## 1. MLX

### ⚠️ 1.1 GPU stream is thread-bound — load and infer on the same thread

**Symptom.** `There is no Stream(gpu, 0) in current thread`, raised from the
first inference call. Loading succeeded; inference failed.

**Root cause.** MLX's default GPU stream belongs to the thread that created
it. A model loaded in thread A (e.g. a FastAPI startup hook or a warm-up
thread) and run in thread B (a request handler's `run_in_executor` default
pool) is on the wrong stream.

**Fix.** One `ThreadPoolExecutor(max_workers=1)` **per model**. Both the load
*and* every inference are submitted to that executor, so they are trivially on
the same thread. TTS and ASR each get their own executor so a long TTS
generation does not queue ASR behind it. See `voxweft/runtime/bridge.py`.

**Cost.** Half a day, because the error message points at the inference site
while the mistake is at the load site.

### ⚠️ 1.2 Holding a lock across an async-generator `yield` deadlocks the whole process

**Symptom.** Bridge process alive, CPU ~50%, `/health` returns ok, `/ready`
times out, zero log lines. Both TTS and ASR dead simultaneously.

**Root cause.** The first serialization attempt was a single `_infer_lock`
around inference. The streaming TTS handler is an async generator: it
acquires the lock, `yield`s a chunk, and is suspended *still holding the
lock* while the client reads. An ASR request then waits on the same lock
forever, and the event loop has nothing to run.

**Fix.** Remove the lock entirely. The single-thread executors from 1.1
already give mutual exclusion, and `await executor.submit(...)` releases
control between chunks instead of pinning it.

**Cost.** This one was misdiagnosed for a day as "a particular Chinese
audio clip is poisonous" because the deadlock happened to be triggered by
a request that arrived while a stream was open.

### 1.3 Two threads constructing the same MLX model ⇒ segfault, process gone

**Symptom.** Bridge disappears with `exit -9`. launchd restarts it; it
disappears again on the next warm request.

**Root cause.** Warm-up ran in a background thread so that uvicorn could
start immediately. uvicorn started accepting requests immediately. If a real
request arrived during warm-up, both paths entered the lazy-load code and
constructed two model instances concurrently. MLX is not thread-safe here.
The original guard was an `asyncio.Lock`, which is per-event-loop and does
nothing across threads.

**Fix.** `threading.Lock` around load, shared by warm-up and request paths.
Warm-up goes through the *same* `_ensure_*_locked()` the requests use — no
private construction path.

### 1.4 `mlx_lm` 0.31 changed the sampling API

**Symptom.** `TypeError: generate() got an unexpected keyword argument 'temp'`.

**Root cause.** `mlx_lm.generate(..., temp=x)` was removed in 0.31.x in
favour of `sampler=make_sampler(temp=x)`.

**Fix.** `from mlx_lm.sample_utils import make_sampler` and pass `sampler=`.
Pin `mlx-lm>=0.31` in `pyproject.toml` and do not support both.

### 1.5 `mx.eval` placement decides whether streaming is actually streaming

Latents kept on-device (`mx.transpose` instead of `np.array(...).T`) and a
single device→host copy at the end saved a measurable slice of RTF in the
non-streaming path. In the streaming path the opposite holds: you *must*
`mx.eval` each chunk before yielding it or the first audio is delayed until
the graph is forced later. This is why `generate()` and `generate_streaming()`
in `voxweft/core/generate.py` deliberately look different.

---

## 2. VoxCPM2 model behaviour

### ⚠️ 2.1 `temperature` is not a sampling temperature

**Symptom.** Trying to reduce output variance by lowering `temperature`.
At `t=0.01` generation never emits `stop`, runs to the 80 s cap, and 80% of
frames are silence.

**Root cause.** In VoxCPM2 `temperature` scales the **initial noise of the
flow-matching ODE** (`z = randn(...) * temperature`), not a softmax. Near
zero the CFM has no stochasticity to work with and does not converge.

**Fix.** Leave it at the official `1.0`. Documented in `TTS-FINDINGS.md` §2.
This knob *looks* like it treats jitter; it only makes things worse.

### 2.2 Voice-style captions are read aloud as speech

**Symptom.** Every utterance began with "一位沉稳清晰的播音员" ("a calm, clear
announcer") spoken out loud.

**Root cause.** We were prepending a parenthesised style caption to the text
on the assumption that VoxCPM2 would treat it as a control prefix. It does
not — it is text, so it is synthesized. A code comment claimed otherwise;
the comment was wrong.

**How we proved it.** ASR round-trip: synthesize with and without the
prefix, transcribe both. With prefix: 4.80 s of audio, transcript contains
"播音员". Without: 2.56 s. Not "sounds a bit longer" — measured.

**Fix.** `apply_voice()` is now the identity and `voice_caption()` returns
`""`. Unit tests pin this (`test_voice_caption_is_empty`,
`test_apply_voice_is_identity` in `tests/test_runtime_router.py`); the ASR
round-trip itself is `scripts/probe_caption_leak.py`. Voice control goes
through reference-audio cloning only.

**Cost.** Also a lesson about trusting comments: the one that said "this is
not read aloud" was written by the person who introduced the bug.

### ⚠️ 2.3 The "first-frame pop" — what it was and what it wasn't

**Symptom.** Audible click at the start of streamed audio, reported on the
M3 stage box.

**What we believed.** The first one or two patches carry a VAE/CFM onset
transient; skip emitting them (`warmup_patches=2`, following mlx-audio's
VoxCPM2 implementation). This shipped in the stage build.

**What we measured** (`scripts/probe_onset.py`, M2, 6 steps, 4-bit, zero-shot
and clone, 3 runs each, `warmup=0 fade=0`):

| | patch 1 RMS | patch 2 RMS | rest RMS | max adjacent-sample jump in patches 1–2 | signal peak |
|---|---|---|---|---|---|
| zero-shot ×3 | 0.004 / 0.112 / 0.061 | 0.015 / 0.085 / 0.073 | 0.070 / 0.067 / 0.036 | 0.024 / 0.057 / 0.073 | 0.5–0.8 |
| clone ×3 | 0.002 / 0.001 / 0.052 | 0.122 / 0.023 / 0.107 | 0.068 / 0.049 / 0.054 | 0.051 / 0.028 / 0.101 | 0.4–0.6 |

No transient. Patch 1 is usually *quieter* than the body (speech starts from
natural silence); the largest sample-to-sample jump in the first two patches
is 0.02–0.10 against peaks of 0.4–0.8 — normal speech slope, not a click. The
first millisecond peaks at 0.0014: the streaming VAE starts from zero cleanly.

**What `warmup_patches=2` actually did.** Two things, both bad on M2:
1. Started playback mid-speech, so the first emitted sample is at speech
   amplitude — first-1 ms peak **0.0437** — i.e. it *creates* the step it was
   meant to remove.
2. Cost **~470 ms** of first-audio latency (warm 304 ms → 773 ms), not the
   160 ms of audio it skips, because on a base M2 each patch is ~230 ms of
   wall time. On M3 Ultra (RTF 0.2) the cost is small, which is why nobody
   noticed on the stage box.

**Fix.** `warmup_patches=0` (default), `fade_in_ms=8` (default): an 8 ms
raised-cosine on the first emitted chunk. With the fade, first-1 ms peak is
0.0002 whatever the stream start. The knob stays for anyone who wants to
reproduce mlx-audio's behaviour.

**Where the stage click most likely came from.** The client-side jitter
buffer (§3.3): fragments scheduled at WebSocket message boundaries, not at
200 ms chunk boundaries. That fix was written but not verified on the box
before it shipped to the venue; the server-side fade is belt-and-braces.

**Lesson.** We inherited a default from a reference implementation and
documented a mechanism for it that we had not measured. The measurement
took 20 minutes.

### 2.4 4-bit quantization is not the source of variance

Disproved. `quantize_lm=4` vs bf16, 5 runs each: duration jitter 1.44 s vs
1.76 s, SNR 29.0 dB vs 26.6 dB. bf16 is not better. The variance that *is*
there (same text, same params, 3.84–6.40 s duration across 6 runs) is the
model sampling normally without a seed. Keep the 4-bit; keep the memory.

### 2.5 Acronyms are read as words

TTS round-trip on the intro text: "We do not compete on BLEU" came back as
"We do not compete on being young" — VoxCPM2 read `BLEU` as the French word,
the ASR heard what it heard. Text normalization (spell out or hyphenate
acronyms before synthesis) is the client's job; the runtime does not do it.
CER on that sentence: 40%. Everything else in the set: 0–9%.

---

## 3. Streaming TTS engineering

### 3.1 Cold start: 2149 ms first chunk vs 62 ms warm

**Root cause.** Lazy singletons — first request paid weight load + graph
compile.

**Fix.** Warm-up on process start that runs the *real* inference path once
(not just weight load: compilation and caches are built on first inference).
Runs in a background thread so the port is up for health checks within
seconds. First playable chunk after warm-up: 124–168 ms.

**Measurement discipline.** Two latency figures were being confused:
HTTP first byte (~50 ms earlier) vs first *playable* chunk on the client.
Only the second one is the north-star number. Say which one you mean.

### 3.2 Streaming VAE: stateful causal convs, bit-identical to full decode

The streaming decoder carries causal-conv state between chunks. Verified:
same latents through per-frame streaming, all-at-once streaming, and
non-streaming full decode produce identical output (0.0% difference).
This is what makes 0.51 s first-audio possible vs 9.3 s for the upstream
non-streaming path — and it is *not* where any audible artefact comes from.
Golden test: `tests/test_core_golden.py::test_streaming_equals_full`.

### ⚠️ 3.4 Loading a second model evicts the first one's buffer cache

**Symptom.** Bridge first real TTS request 720–762 ms first byte; every
request after it 305 ms; in-process streaming on the same machine 286 ms.
Warm-up had run and `/ready` was 200.

**Disproved first.** "Warm-up didn't exercise the full path" — we changed
warm-up from 3 patches of "你好" to a full 23-patch sentence. First request
still 720 ms. Not compile.

**Root cause.** Order of warm-up: TTS loaded and primed, *then* ASR (another
~3 GB) loaded. On a 16 GB M2 the second load evicts the first model's MLX
buffer cache; the next TTS inference re-allocates. A plain ASR *inference*
afterwards does not evict again (measured: 301 ms TTS after an ASR call) —
it is specifically the load.

**Fix.** After all models are loaded, run one short TTS generation
(`_reprime_tts`, ~2.3 s at start-up, once). First request: 311 ms. Any
process that hosts more than one MLX model should prime *after the last
load*, not after each one.

**Cost.** 450 ms on every cold session for as long as the stage build ran,
invisible on M3 Ultra because 512 GB never evicts anything.

### 3.3 Client jitter buffer: periodic "click" was a scheduling artefact

Server sends 200 ms chunks; WebSocket message boundaries do not align with
them. Scheduling each received fragment as its own `AVAudioPCMBuffer` makes
the player node stitch at every fragment boundary — audible as periodic
clicks. Fix: accumulate to 200 ms before `scheduleBuffer`. Also:
`player.stop()` must be immediately followed by `play()` or later buffers
queue without playing.

---

## 4. Endpointing / segmentation

### ⚠️ 4.1 SmartTurn v3 is net-negative for simultaneous interpretation

**Expectation.** A learned end-of-turn model would beat RMS silence.

**Measurement.** Same Lao conference audio: silence rule → 28 segments;
SmartTurn on → 8 segments, all exactly 15 s = every one of them was the
forced cut. The model answered "not finished" on almost every check, blocked
the silence cut, and everything fell through to `max_utterance`.

**Why.** SmartTurn is a *dialogue* model: "has the other person stopped and
is waiting for me?" In continuous speech the speaker is *not* waiting, so it
is correctly saying "not done" — and that is the wrong question for
interpretation, where we want to cut at clause boundaries while they talk.

**Decision.** Off by default (`VOXWEFT_SMART_TURN=0`). Kept as an option for
the dialogue mode. Before enabling anywhere, compare segment counts on the
same audio.

### 4.2 Three SmartTurn integration traps (if you do turn it on)

1. `append_audio()`'s return value is *not* the model verdict — it is a
   silence-duration fallback. The verdict is `analyze_end_of_turn()`.
2. Therefore never feed it silence frames: 118/118 COMPLETE.
3. Feeding the whole accumulated buffer (15 s, several sentences) ⇒ 26/26
   INCOMPLETE. Feeding the buffer *tail* ⇒ it is all silence by the time
   you trigger, 96 ms of nothing. You need a separate rolling ring of the
   last ~4 s of *speech* frames. That is `_speech_ring` in the segmenter.
4. One verdict is ~50 ms, not ~10 ms. Thread pool, and only once silence
   is already established.

### 4.3 Fixed RMS threshold drops words when the microphone changes

`sil_thresh=0.035` was tuned for a close-talk stage mic. On phone
recordings the median speech-frame RMS is 0.02 — 70% of speech frames were
classified as silence. Commercial ASR CER on the same material: 4.7% → 16.3%.

**Fix.** Adaptive threshold (`adaptive=True`): track a noise floor and
clamp the effective threshold to `[floor, sil_thresh]`; soft forced cuts
look for the quietest frame in a window instead of cutting at an arbitrary
sample. CER back to 10% on phone audio; 12/12 segmenter tests pass.

### 4.4 Segments too short were silently deleted — including real words

"各位专家，" (0.45 s of speech) followed by a 0.58 s pause became its own
segment, failed `min_seg_s=0.60`, and was **discarded**. The screen showed
"下面介绍…" instead of "各位专家，下面介绍…". No error anywhere. Real
speech has exactly this shape ("So,", "Okay,", vocatives).

**Fix.** Short segments are *carried* into the next segment (`_carry`), not
dropped. They must live in a separate buffer — putting them back into `_buf`
makes the silence state machine think speech already started and the
threshold never triggers again. Carry is capped at 2 s and its speech
content is re-counted, not estimated.

### 4.5 `asyncio.ensure_future` silently binds to the wrong loop

**Symptom.** SmartTurn never called once; only forced cuts.

**Root cause.** `ensure_future` takes "the current event loop", which is the
right one when called from inside a loop callback and a *wrong/closed one*
when called from a thread or early in startup. The task is created, never
runs, never errors.

**Fix.** `asyncio.get_running_loop()` explicitly; on `RuntimeError`, fall
back to the silence rule and log it.

### 4.6 Do not `Task.cancel()` a not-yet-started coroutine to invalidate it

Cancelling a coroutine that has not begun executing shows as `done()` but
never ran; the scheduler then sees "no pending check" and schedules another,
racing the cancel. Use a generation counter (`_turn_check_gen`) and let the
task discard its own result.

### 4.7 The config-ordering bug that produced garbage

**Symptom.** Lao audio recognized as Chinese: "拜拜你他你他".

**Root cause.** Client pushed `session_config` mid-utterance; the
orchestrator was rebuilt with the wrong source language for audio already
in flight.

**Fix.** Config changes rebuild the orchestrator only between presses; the
Swift client gates on it. Any client you write must not send
`session_config` between `ptt_start` and `ptt_end`.

---

## 5. Translation

### 5.1 "Repetition means the model is too small" — wrong

**Claim.** NLLB-1.3B produces repetitive output on short fragments; 3.3B
would fix it.

**First test (7 hand-written samples).** 1 repetition. Concluded "decoding
params, not size". *Premature — sample too small.*

**Second test (28 real ASR fragments from production audio).** 0/28
repetitions on **both** 1.3B and 3.3B. The repetition people had seen was
the routing bug in 4.7 feeding wrong-language text into the translator.

**Decision.** Keep 1.3B where NLLB is used at all. Record this as a
reminder that a 7-sample test can only tell you a hypothesis is *not
obviously false*.

### 5.2 Small LLMs guess the wrong language for zh→low-resource

A 4B chat model asked for Lao would sometimes emit Thai, or Lao script
with Thai vocabulary, confidently. NLLB never does this. Hence the router:
zh→SEA via NLLB or a dedicated MT model; SEA→zh via the LLM (where 4B is
strong: 267–310 ms per sentence). See `voxweft/runtime/router.py`.

### 5.3 "thinking" mode wrecks translation quality

With a reasoning-enabled chat model, letting it think before translating
made translations *worse* — it rationalises, then "improves" the sentence.
Disable thinking for MT, or train a small LoRA that suppresses it.

### 5.4 Glossary injection: pre-model pinning vs post-model patching

Proper nouns go in *before* the model as target-script substitutions so NLLB
copies them; common words pinned this way get copied in Chinese word order
and produce broken sentences. Post-model patching is only for terms the
model copied back untranslated. Keep both tables short — a long glossary in
the prompt biases short-sentence decoding.

---

## 6. ASR

### 6.1 Language label case decides the *script* you get back

Qwen3-ASR fine-tuned with lowercase training labels: `lao` → Lao script;
`Lao`, `lo`, `lao_Laoo` → **Thai script**, silently. Normalize every
incoming form to the training label before it reaches the model
(`router.normalize`, `test_normalize_all_lao_forms` is the most important
test in the runtime suite).

### 6.2 Auto language-ID on short clips is unreliable between Lao and Thai

Do not rely on LID for segments of 1–3 s. Always pass an explicit language
from the session.

### 6.3 Hotwords: a few high-value proper nouns beat a long list

"AI" in Lao was heard as the single letter "ເອ", and the translator then
"repaired" the whole clause into nonsense. Adding the two Lao forms of "AI"
fixed it at the source. But long hotword lists perturb decoding on short
segments; keep them sparse. Chinese homophone traps that were real:
端侧→端测, 切段→切断, 出声→出生.

---

## 7. Build & deploy on macOS

### ⚠️ 7.1 SwiftPM incremental build reports success without compiling

**Symptom.** `swift build` → "Build complete (2.8s)". App runs old code.
Days of "I can see the log line in the source, the user sees nothing".

**Root cause.** Files copied via `scp` keep an *older* mtime than the
existing build artefacts, so SwiftPM decides nothing changed.

**Fix.** `rm -rf .build` before every deploy. A real compile of this app is
40+ s; if you see 2.8 s, nothing was built. Do not use `strings | grep` on
the binary to check whether Chinese literals made it in — it misses UTF-8;
use Python `b"...".encode("utf-8") in data`.

### ⚠️ 7.2 macOS 26 (Tahoe) + Swift 6.3: SwiftUI `Button` aborts on click

`_ButtonGesture.internalBody → MainActor.assumeIsolated →
swift_task_isCurrentExecutorWithFlags → _objc_fatal → SIGABRT`. Every
`Button` in the app. Workaround: replace with a `DragGesture`-based
`.tappable` modifier or `onTapGesture`. (First attributed to
`SystemSegmentedControl`; that was wrong — it was `Button` all along.)

### 7.3 `Task { @MainActor in }` created inside `SwiftUI.Update.dispatchActions()` never runs

Not cancelled, just never executed. `DispatchSourceTimer` also fails there.
`Timer.scheduledTimer` on `RunLoop.main` in `.common` mode works.

### 7.4 Deleting a "leftover" `stopFlag = false` broke the demo player

It was the reset. The task exited immediately and the generation counter
advanced by 2–3 per press instead of 1. If a flag reset looks redundant,
find who reads it before deleting it.

### 7.5 Command Line Tools ≠ a Swift toolchain

On one M2 with CLT only (`swiftlang-6.0.3.1.10`), the SDK was built by
`swiftlang-6.0.3.1.5`: SwiftPM could not even link `Package.swift`
(undefined `PackageDescription.Package.__allocating_init`), and plain
`swiftc` failed on `CoreFoundation` with "this SDK is not supported by the
compiler". Nothing in the project is at fault; the CLT install is internally
inconsistent. `app/build.sh` exists so a `swiftc`-only path is available, but
if you see those two errors, reinstall CLT or use a machine with Xcode.

### 7.6 Health probes: `/health` ≠ `/ready`

`/health` is "process alive" and is true during the 1.7 s cold start;
`/ready` is "models loaded and warmed". Watchdogs and client start-up
checks must probe `/ready`, or users eat the cold start.

### 7.7 Non-daemon executor threads make restarts slow

`ThreadPoolExecutor` threads are non-daemon; without an explicit
`shutdown(wait=False, cancel_futures=True)` on process exit, launchd
`kickstart` waits several seconds.

---

## 8. Measurement mistakes we made (so you don't)

* Concluding from 7 samples (§5.1).
* Trusting a code comment over an ASR round-trip (§2.2).
* Confusing HTTP-first-byte with first-playable-chunk (§3.1).
* Checking deployment with `strings | grep` on UTF-8 (§7.1).
* Attributing a crash to the most visible frame in the stack (§7.2).
* Attributing a deadlock to the input that happened to trigger it (§1.2).

---

## Appendix — numbers that anchor this repo

| what | M2 16 GB | M3 Ultra |
|---|---|---|
| VoxCPM2 4-bit, 6 CFM steps, RTF | 1.48 (streaming) / 1.35 | ≈ 0.2 |
| first audio, streaming, warm (`voxweft say --stream`) | 304 ms zero-shot · 403 ms clone | 124–168 ms |
| first audio, cold (includes `mx.compile`) | ~700 ms | 2149 ms → 124 ms with warm-up |
| upstream non-streaming first audio | 9.3 s | — |
| cold start first chunk (no warm-up) | — | 2149 ms |
| TTS ASR-round-trip intelligibility (zh/vi/lo/km) | ✅ verbatim | ✅ |
| segmenter, phone audio, fixed vs adaptive threshold (ASR CER) | 16.3% → 10% | — |

RTF 1.48 on a base M2 means the audio is produced at 2/3 of playback speed:
streaming will stutter on that machine. That is the hardware, not the port —
the upstream implementation measures the same RTF. On M3 Ultra it is
real-time with margin. See `docs/HARDWARE.md`.

---

## Open — measured but not yet explained

* **Loudness variance between runs** (same text/params: whole-signal RMS
  0.006–0.11, peaks up to 0.97). Known model behaviour without a seed
  (`TTS-FINDINGS.md` §3); a peak of 0.97 is one bad run from clipping.
  A light limiter on the output path would be cheap; not done.
* **M-series Pro/Max RTF** is unmeasured (`docs/HARDWARE.md`). We only have a
  base M2 and an M3 Ultra.
