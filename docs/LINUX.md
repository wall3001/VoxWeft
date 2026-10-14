# Linux partial install, and the Mac runbook

VoxWeft's speech pipeline is Apple Silicon / MLX. `docs/HARDWARE.md` lists
Linux as unsupported. This file records what a Linux machine can still do,
and the exact commands to finish setup on a Mac.

Checked on a Cursor cloud VM, 2026-10-14:

| check | result |
|---|---|
| `uname -m` | `x86_64` (AGENT-SETUP step 0 requires `arm64`) |
| `sw_vers` | not installed |
| `swift` | not installed |
| Python | 3.12.3 |
| RAM | 15 GiB |
| GPU | none (`nvidia-smi` absent; `mx.metal.is_available()` is `False`) |

No model weights were downloaded. `~/.voxweft` and `~/.cache/huggingface`
were absent after the checks below.

## What ran on Linux

```bash
sudo apt-get update
sudo apt-get install -y python3.12-venv   # Debian/Ubuntu: venv needs ensurepip
cd /path/to/VoxWeft
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -e ".[dev,mcp,runtime]"
python -m pip install 'mlx[cpu]==0.32.3'
```

The plain `mlx` manylinux wheel installs, then `import mlx.core` fails with:

```text
ImportError: libmlx.so: cannot open shared object file: No such file or directory
```

`libmlx.so` comes from a backend extra. This VM has no NVIDIA GPU, so the
CPU extra was installed (`mlx-cpu==0.32.3`). On macOS, skip that extra:
the `mlx` package pulls `mlx-metal` when `platform_system == "Darwin"`.

Versions that resolved here: voxweft 0.1.0, mlx 0.32.3, mlx-cpu 0.32.3,
mlx-lm 0.32.0, mlx-audio 0.5.9, mlx-whisper 0.4.3, mcp 2.3.0, fastapi
0.143.0, uvicorn 0.54.0, torch 2.14.1, transformers 5.19.0, numpy 2.5.4.
`[runtime]` pulls `mlx-whisper`, which pulls a large torch/CUDA wheel set.
Those wheels installed; they do not provide a Metal GPU.

Smoke that completed:

```bash
voxweft                         # prints the four verbs, exit 0
python -m voxweft.db show       # 22 seed voices, intro langs, showcase routes
python - <<'PY'
import mlx.core as mx
print(mx.default_device())      # Device(cpu, 0)
a = mx.arange(8, dtype=mx.float32)
b = a * 2 + 1
mx.eval(b)
print([float(x) for x in b])   # 1,3,5,7,9,11,13,15
print("metal", mx.metal.is_available())  # False
PY
python -m pytest tests/test_runtime_router.py \
                 tests/test_runtime_segmenter.py \
                 tests/test_runtime_orchestrator.py \
                 tests/test_core_fade.py \
                 tests/test_core_golden.py -q
# 46 passed, 2 skipped
```

The two skips are `tests/test_core_golden.py`: `vae_golden.npz` is not under
`~/.voxweft/models/VoxCPM2`. Router, segmenter, and orchestrator tests use
fakes and need no weights. The fade tests are NumPy; they import MLX because
`voxweft.core.generate` does.

`voxweft say` with no model exits 1:

```text
error: --model or VOXCPM2_MODEL required
```

The WebSocket server process starts without weights. Hit `/health` only.
Opening `/ws/voice/v2` loads a translator (see below).

```bash
voxweft serve --host 127.0.0.1 --port 3100
# other shell:
curl -sS http://127.0.0.1:3100/health
# {"status":"ok","service":"voxweft-runtime","translator_loaded":false}
```

Log line at startup:

```text
voxweft runtime on ws://127.0.0.1:3100/ws/voice/v2 (bridge http://localhost:8133, LLM <none: NLLB fallback>)
```

Stop the process after the health check. `.venv/` is gitignored.

## What still requires a Mac

These were not started here. They need Apple Silicon, the model files in
`docs/MODELS.md`, or both.

* **L1 TTS.** `voxweft say`, voice clone, streaming first-audio / RTF, and
  `afplay`. A CPU array add is not that pipeline. The core is written for
  MLX's Metal GPU (`docs/HARDWARE.md`).
* **Bridge (port 8133).** `voxweft bridge` warm-up loads VoxCPM2 and
  Qwen3-ASR on MLX threads. `mlx_audio.stt.utils.load_model` treats a missing
  directory as a Hugging Face repo id and will download multi-GB weights.
  Leave the bridge off until `VoxCPM2/` and `Qwen3-ASR-1.7B-8bit/` are on disk.
* **A real `/ws/voice/v2` turn.** The first WebSocket client calls
  `_load_engines()`. With `VOXWEFT_LLM_DIR` unset, that loads NLLB-200-1.3B
  (`from_pretrained`, CC-BY-NC-4.0) onto `VOXWEFT_NLLB_DEVICE` (default
  `mps`). `scripts/mock_interpret.py` uses this path.
* **MCP `synthesize`.** It builds `VoxCPM2Pipeline` from `VOXCPM2_MODEL`.
  The server was not registered in this harness.
* **SwiftUI app.** `app/build.sh` needs Xcode or the Command Line Tools, then
  `open build/VoxWeft.app`. There is no Swift toolchain on this VM.
* **Golden tests.** They need `vae_mlx.safetensors` plus
  `vae_golden.npz` / `lm_golden.npz` / `enc_golden.npz` in the model dir.

## Environment, ports, models

Nothing is bundled. Layout root is `~/.voxweft/models/` (`VOXWEFT_MODEL_DIR`).
If Hugging Face is slow, use the ModelScope mirrors named in `docs/MODELS.md`.

| slot | directory | source | approx. size | env |
|---|---|---|---|---|
| TTS (L1 minimum) | `VoxCPM2/` | [OpenBMB/VoxCPM2](https://huggingface.co/OpenBMB/VoxCPM2) | multi-GB; not pinned here | `VOXCPM2_MODEL` |
| ASR | `Qwen3-ASR-1.7B-8bit/` | mlx-community/Qwen3-ASR-1.7B-8bit | 2.5 GB | `VOXWEFT_ASR_MODEL` |
| ASR alt | `whisper-large-v3-turbo/` | mlx-community | — | `VOXWEFT_WHISPER_MODEL` |
| LLM / MT | `Qwen3-4B-Instruct-2507-4bit/` | mlx-community | ~2.3 GB resident with the others | `VOXWEFT_LLM_DIR` |
| MT alt | `Hy-MT2-1.8B-4bit/` | mlx-community | ~1.3 GB | point `VOXWEFT_LLM_DIR` here |
| MT fallback | `nllb-200-1.3B/` | facebook, **CC-BY-NC-4.0** | 1.3B | `VOXWEFT_NLLB_DIR` |

One-time, after `VoxCPM2/` is downloaded (needs PyTorch, which `[runtime]`
installs via `mlx-whisper`):

```bash
export VOXCPM2_MODEL=~/.voxweft/models/VoxCPM2
python scripts/convert_vae.py "$VOXCPM2_MODEL"
# audiovae.pth → vae_mlx.safetensors
```

| variable | default | role |
|---|---|---|
| `VOXWEFT_MODEL_DIR` | `~/.voxweft/models` | model root |
| `VOXCPM2_MODEL` | `$VOXWEFT_MODEL_DIR/VoxCPM2` | TTS weights; required for `voxweft say` |
| `VOXWEFT_ASR_MODEL` | `$VOXWEFT_MODEL_DIR/Qwen3-ASR-1.7B-8bit` | bridge ASR |
| `VOXWEFT_WHISPER_MODEL` | `$VOXWEFT_MODEL_DIR/whisper-large-v3-turbo` | lo/km/my and other non-Qwen languages |
| `VOXWEFT_LLM_DIR` | empty | MLX LLM dir; empty selects NLLB |
| `VOXWEFT_NLLB_DIR` | `~/.voxweft/models/nllb-200-1.3B` | NLLB weights |
| `VOXWEFT_NLLB_DEVICE` | `mps` | NLLB device (`mps` is Apple only) |
| `VOXWEFT_VOICE_DIR` | empty | `<lang>_{src,tgt}.wav`; repo ships `samples/voices/` |
| `VOXWEFT_VOICE_PROMPT_WAV` | empty | optional ~5 s clone reference |
| `VOXWEFT_BRIDGE_URL` | `http://localhost:8133` | server → bridge |
| `VOXWEFT_API_KEY` | empty | empty allows every WebSocket |
| `VOXWEFT_DB` | `~/.voxweft/voxweft.db` | copied from repo `voxweft.db` on first use |
| `VOXWEFT_SMART_TURN` | `0` | learned endpointer; off on purpose |
| `PORT` | `8133` for bridge, `3100` for serve | CLI `--port` overrides |

| process | port | probe |
|---|---|---|
| `voxweft bridge` | 8133 | `GET /health` is process-up. `GET /ready` is 200 only after TTS and ASR warm-up; otherwise 503 |
| `voxweft serve` | 3100 | `GET /health`. Speech is `ws://127.0.0.1:3100/ws/voice/v2` |

Poll `/ready`, not `/health`, before sending audio to the bridge.

## Mac runbook

On an Apple Silicon Mac, macOS 14 or newer, Python 3.10+, 16 GB RAM. Follow
`docs/AGENT-SETUP.md` top to bottom. The commands are:

```bash
uname -m                 # arm64
sw_vers -productVersion  # 14.0 or newer
python3 --version
sysctl -n hw.memsize | awk '{print $1/1073741824 " GB"}'

python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[mcp,runtime]"

mkdir -p ~/.voxweft/models
# download the directories in the table above into ~/.voxweft/models/
export VOXCPM2_MODEL=~/.voxweft/models/VoxCPM2
python scripts/convert_vae.py "$VOXCPM2_MODEL"

voxweft say "VoxWeft is running on this machine." --stream -o /tmp/vw.wav
afplay /tmp/vw.wav
```

Expected on a base M2: `first_audio` 300–500 ms, `rtf` ≈ 1.5. On M3 Ultra:
~150 ms, rtf ≈ 0.2. If first audio is above 2 s, the model is cold — run it
twice and keep the second number.

MCP, pointed at this clone's venv (Claude Code: `.mcp.json`; Cursor: the
project MCP config). Restart the harness after writing it.

```json
{"mcpServers": {"voxweft": {
    "command": "<absolute path to clone>/.venv/bin/voxweft",
    "args": ["mcp"],
    "env": {"VOXCPM2_MODEL": "<absolute path>/VoxCPM2"}}}}
```

The tool is `synthesize(text, ref_wav_path?, out_path?, play?)`.

Interpretation runtime, after the ASR and LLM directories exist:

```bash
export VOXWEFT_MODEL_DIR=~/.voxweft/models
export VOXWEFT_LLM_DIR=$VOXWEFT_MODEL_DIR/Qwen3-4B-Instruct-2507-4bit
export VOXWEFT_ASR_MODEL=$VOXWEFT_MODEL_DIR/Qwen3-ASR-1.7B-8bit
export VOXWEFT_VOICE_DIR=samples/voices
# optional:
# export VOXWEFT_VOICE_PROMPT_WAV=/path/to/5s_reference.wav

voxweft bridge --port 8133 &
# until HTTP 200:
# curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8133/ready
voxweft serve --port 3100 &
python scripts/mock_interpret.py --src zh --tgt en
```

`mock_interpret.py` prints per-sentence first-audio latency. Desktop shell,
after the two processes are up:

```bash
cd app && ./build.sh && open build/VoxWeft.app
```

If `swift build` finishes in about 3 seconds, it did not compile. `rm -rf app/.build` and build again (`docs/DEVLOG.md` §7.1, `app/README.md`).

Report, in this order: chip and memory; `voxweft say` first-audio and RTF;
whether MCP is wired and to which harness; if the runtime was started, the
`mock_interpret` median first-audio and the language pair. Quote the error
and the AGENT-SETUP step number for anything that failed.

## Report from this Linux VM

* Chip and memory: `x86_64` Linux, 15 GiB. Not `arm64`. No Metal GPU.
* `voxweft say`: not synthesized. Without `VOXCPM2_MODEL` the CLI exits 1
  with `error: --model or VOXCPM2_MODEL required`.
* MCP: not wired. No weights, and `synthesize` needs the MLX TTS pipeline.
* Runtime: `voxweft serve` on port 3100 answered `GET /health` 200 with
  `translator_loaded: false`. Bridge was not started. `mock_interpret` was
  not run. No language pair was configured.
* Also: `python -m voxweft.db show` succeeded. Pytest: 46 passed, 2 skipped
  (missing VoxCPM2 goldens). MLX CPU device executed one arithmetic op.
