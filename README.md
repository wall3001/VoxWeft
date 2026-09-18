# VoxWeft

![VoxWeft — voice cloning on Apple Silicon](docs/banner.png)

**On-device simultaneous interpretation and voice dialogue on Apple Silicon.
Speech in, translated speech out, streaming, nothing leaves the machine.**

At the core is our port of [VoxCPM2](https://github.com/OpenBMB/VoxCPM) —
a 2B tokenizer-free TTS covering 30 languages — to [MLX](https://github.com/ml-explore/mlx):
zero weight conversion for the LM, bit-exact AudioVAE, a **stateful streaming
decoder** that gets first audio out in ~0.3–0.4 s on a base M2 and ~150 ms
on M3 Ultra, voice cloning from ~5 s of reference audio, 4-bit
quantization. Verified module-by-module against the official PyTorch
implementation.

Around it: a runtime that turns a microphone into segmented, translated,
spoken output while the speaker is still talking.

There is no PyPI or npm package, on purpose: the source is the distribution.
Clone, install editable, point your agent at the clone.

```bash
git clone https://github.com/OpenBMB/VoxWeft && cd VoxWeft
python -m venv .venv && source .venv/bin/activate
pip install -e .              # L1: TTS, clone, `voxweft say`
pip install -e ".[mcp]"       # + MCP server for agent harnesses
pip install -e ".[runtime]"   # + segmenter, ASR/LLM adapters, WebSocket server
```

If you are an agent (Claude Code, Codex, Cursor, …) setting this up for a
user, read [`docs/AGENT-SETUP.md`](docs/AGENT-SETUP.md) — it is written for you.

---

## Three layers, one repo

```
              ┌──────────────────────────────────────────────────────┐
   L3         │  CLI  ·  MCP server  ·  macOS shell (app/)           │  wrappers — thin, replaceable
              └──────────────────────────────────────────────────────┘
              ┌──────────────────────────────────────────────────────┐
   L2         │  segmenter / endpointing  →  ASR  →  MT | LLM        │  runtime — MLX only,
   runtime    │  language router · orchestrator · /ws/voice/v2       │  models are yours to pick
              └──────────────────────────────────────────────────────┘
              ┌──────────────────────────────────────────────────────┐
   L1         │  VoxCPM2 on MLX: streaming VAE · clone · 4-bit       │  core — ours end to end
   core       │  `voxweft say`                                       │
              └──────────────────────────────────────────────────────┘
```

**L1 is the reason this repo exists.** It is complete, tested against
goldens, and useful on its own — any agent harness can get a local voice
through the MCP server in one JSON block.

**L2 is Lego.** We ship two defaults each for ASR and LLM and the adapters to
plug them in, but you download the weights. Everything runs on MLX; we do not
support Ollama or other CPU-first runtimes because they cannot hit the
latency budget on this hardware.

**L3 is whatever fits.** CLI, MCP, and a ~500-line SwiftUI shell in `app/`
with hold-to-talk, latency chips and a mic-free demo. Disposable by design.

## Quick start — L1

```bash
export VOXCPM2_MODEL=~/.voxweft/models/VoxCPM2      # see docs/MODELS.md
python scripts/convert_vae.py $VOXCPM2_MODEL        # one-time, audiovae.pth → MLX

voxweft say "你好，欢迎使用语音合成服务。" --play
voxweft say "Guten Morgen, wie geht es Ihnen?" --stream -o out.wav   # prints first-audio ms
voxweft say "ສະບາຍດີທຸກໆທ່ານ" --ref my_voice.wav -o lao.wav          # clone from ~5 s
```

```python
from voxweft.core import VoxCPM2Pipeline

pipe = VoxCPM2Pipeline(model_dir, inference_timesteps=6, quantize_lm=4)
for chunk, i in pipe.generate_streaming("text", ref_latents=pipe.encode_reference(ref_16k)):
    play(chunk)                       # float32 @ 48 kHz
```

MCP, for Claude Code / Codex / Cursor / anything that speaks MCP — point the
harness at the venv inside your clone, no global install:

```jsonc
{"mcpServers": {"voxweft": {
    "command": "/path/to/VoxWeft/.venv/bin/voxweft", "args": ["mcp"],
    "env": {"VOXCPM2_MODEL": "/path/to/models/VoxCPM2"}}}}
```

## Quick start — L2 runtime

Two processes. The **bridge** owns the MLX models for TTS and ASR, each pinned
to its own thread (MLX GPU streams are thread-bound — see the devlog). The
**server** owns segmentation, routing, the in-process MLX LLM/MT, and the
WebSocket.

```bash
pip install -e ".[runtime]"
export VOXWEFT_MODEL_DIR=~/.voxweft/models         # VoxCPM2/, Qwen3-ASR-1.7B/, <llm>/
export VOXWEFT_LLM_DIR=$VOXWEFT_MODEL_DIR/Qwen3-4B-Instruct-2507-4bit
export VOXWEFT_VOICE_PROMPT_WAV=~/my_voice.wav   # ~5 s reference; pins one speaker for the session

voxweft bridge --port 8133 &      # wait for GET /ready → 200
voxweft serve  --port 3100        # ws://localhost:3100/ws/voice/v2
```

Protocol is in the docstring of `voxweft/runtime/server.py`: `session_config`,
`ptt_start`, int16 PCM frames, `ptt_end`; back come `asr`, `translation`,
`audio_start` + 48 kHz PCM chunks + `audio_end`, `turn_end`, and `no_speech`
with the measured level when nothing was cut.

### Default models

| slot | default | alt | licence |
|---|---|---|---|
| TTS | VoxCPM2 (4-bit) | — | Apache-2.0 |
| ASR | Qwen3-ASR-1.7B via mlx-audio (zh/en/vi/th/id/ms + 20 more) | Qwen3-ASR-0.6B | Apache-2.0 |
| ASR, lo/km/my | whisper-large-v3-turbo (MLX) — routed automatically; **quality is poor** (Lao often comes back in Thai script) | Omnilingual-ASR CTC via sherpa-onnx is the better open option (adapter not written yet) | MIT |
| LLM / MT | Qwen3-4B-Instruct (mlx-lm) | Qwen3-8B · Hy-MT2-1.8B | Apache-2.0 |
| MT fallback | NLLB-200-1.3B | — | **CC-BY-NC-4.0**, opt-in, research only |

No model weights are in this repo. `docs/MODELS.md` has download notes
(ModelScope mirrors where HF is slow).

## What the runtime does that a model download doesn't

* **Cuts while you speak.** Per-frame RMS with an adaptive noise floor,
  0.45 s silence rule, forced cut at the first ≥100 ms dip after 6 s (hard cut at 9 s), and a *carry* buffer so a
  0.4 s "So," is prepended to the next segment instead of silently dropped.
  Learned end-of-turn (SmartTurn v3) is wired in and **off by default** — we
  measured it net-negative for interpretation and say why.
* **Routes by language pair.** Low-resource → zh through the LLM (where 4B
  is strong); zh → low-resource through dedicated MT (where small LLMs
  guess the wrong language); SEA↔SEA direct, never via Chinese.
* **Keeps MLX alive under concurrency.** Thread-pinned executors, no locks
  across `yield`, warm-up through the same code path as requests. Each of
  those is a crash we had.
* **Reports failure loudly.** A press with no segment returns
  `no_speech {peak_rms, threshold, too_quiet}` so "nothing said" and "too
  quiet" can be told apart on the spot instead of in the server log.

## Numbers

| | M2 16 GB | M3 Ultra |
|---|---|---|
| TTS first audio, streaming, warm | 304 ms (zero-shot) · 403 ms (clone) | 124–168 ms |
| TTS RTF (4-bit, 6 steps) | 1.48 | ≈ 0.2 |
| ASR → MT → first TTS chunk, per segment | — | ≈ 0.85 s |
| text → MT (Qwen3-4B) → first TTS chunk, per sentence | 2.35 s median | — |
| TTS ASR-round-trip CER on the repo intro (zh / en) | 3.4% / 7.8% | |
| Segmenter, phone mic, ASR CER fixed→adaptive threshold | 16.3% → 10% | |

RTF 1.48 means a base M2 produces audio slower than it plays — the streaming
path will stutter there, and that is the silicon (upstream measures the
same). Read `docs/HARDWARE.md` before choosing a machine.

We do **not** publish BLEU. Translation quality is the property of the model
you plug in; what this repo owns is that the pipeline does not lose it on
the way: ASR-round-trip intelligibility of the TTS, first-audio latency, and
RTF. `bench/` has the scripts.

## The devlog is part of the product

[`docs/DEVLOG.md`](docs/DEVLOG.md) — every failure we hit, what we were wrong
about, and what it cost: MLX stream thread binding, a lock held across an
async `yield` that froze the process, `temperature` that is not a
temperature, text that the model dutifully read aloud, SwiftPM "building" in 2.8 s
without compiling, SwiftUI `Button` aborting on macOS 26, a learned
endpointer that made interpretation worse. A working demo can be
regenerated by any model; this list cannot.

## Layout

```
voxweft/
  core/        model.py · vae.py · generate.py · cli.py          L1
  runtime/     segmenter · router · orchestrator · translator     L2
               bridge.py (TTS+ASR HTTP) · server.py (WebSocket)
  wrappers/    mcp_server.py                                      L3
app/           SwiftUI macOS shell (hold-to-talk, latency chips, demo)  L3
tests/         golden alignment (needs model) · segmenter · router · orchestrator (no model)
bench/         asr / mt / tts benches + ASR-round-trip judge
scripts/       convert_vae.py · probe_{onset,stream,clone,perf}.py · mock_interpret.py
docs/          DEVLOG.md · TTS-FINDINGS.md · ARCHITECTURE.md · HARDWARE.md · MODELS.md
samples/       zero-shot zh / lo output, round-trip verified
```

## Credits

- [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) — the model. Tokenizer-free
  turned out to be exactly right for low-resource languages: no RVQ codebook
  to be out-of-distribution on.
- [ml-explore/mlx](https://github.com/ml-explore/mlx) — the array framework
  that made the port a one-day job (and the debugging a two-week one).
- Qwen3-ASR, mlx-lm, mlx-audio, pipecat SmartTurn — the Lego.

Apache-2.0.
