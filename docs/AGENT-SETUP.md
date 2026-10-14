# Setting up VoxWeft — instructions for an agent

You are an AI coding agent asked to get VoxWeft running on this Mac. This
file is the whole procedure; follow it top to bottom and report the numbers
at the end. There is no package registry step anywhere — everything builds
from the clone you are sitting in.

## 0. Preconditions (check, do not assume)

```bash
uname -m                 # must print arm64 — MLX is Apple Silicon only
sw_vers -productVersion  # 14.0 or newer
python3 --version        # 3.10+
sysctl -n hw.memsize | awk '{print $1/1073741824 " GB"}'   # 16 GB minimum; see docs/HARDWARE.md
```

If `uname -m` is not `arm64`, stop the Mac procedure and tell the user: this
project does not run on Intel Macs or Linux. A Linux host can still install
the package and run the model-free tests; that partial procedure, and the
handoff back to this file, is in [`LINUX.md`](LINUX.md).

## 1. Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[mcp,runtime]"
```

Only the base install (`pip install -e .`) is needed for TTS / cloning / the
CLI. `[mcp]` adds the MCP server; `[runtime]` adds the interpretation
pipeline (mlx-lm, mlx-audio, mlx-whisper, fastapi).

## 2. Models

Nothing is bundled. Create `~/.voxweft/models/` and download into it —
`docs/MODELS.md` lists sources and the exact directory names the code
expects. Minimum for L1: `VoxCPM2/`. Then run the one-time VAE conversion:

```bash
export VOXCPM2_MODEL=~/.voxweft/models/VoxCPM2
python scripts/convert_vae.py "$VOXCPM2_MODEL"
```

If Hugging Face is slow or blocked from this machine, use the ModelScope
mirrors named in MODELS.md rather than retrying HF.

## 3. Verify L1 (do this before anything else)

```bash
voxweft say "VoxWeft is running on this machine." --stream -o /tmp/vw.wav
```

Expected on a base M2: `first_audio` 300–500 ms, `rtf` ≈ 1.5. On M3 Ultra:
~150 ms, rtf ≈ 0.2. If first audio is > 2 s, the model is loading from a
cold disk cache — run it twice and report the second number.

Play it (`afplay /tmp/vw.wav`) or ask the user to; you cannot judge audio.

## 4. Wire the MCP server into the harness you are running in

Write the harness's MCP config pointing at **this clone's venv binary**:

```jsonc
{"mcpServers": {"voxweft": {
    "command": "<absolute path to clone>/.venv/bin/voxweft",
    "args": ["mcp"],
    "env": {"VOXCPM2_MODEL": "<absolute path>/VoxCPM2"}}}}
```

Claude Code: `.mcp.json` in the project root. Codex / Cursor: their
equivalent. Restart the harness. The tool exposed is `synthesize(text,
ref_wav_path?, out_path?, play?)`.

## 5. Optional — the interpretation runtime

Two processes, both from the venv:

```bash
export VOXWEFT_MODEL_DIR=~/.voxweft/models
export VOXWEFT_LLM_DIR=$VOXWEFT_MODEL_DIR/Qwen3-4B-Instruct-2507-4bit
export VOXWEFT_ASR_MODEL=$VOXWEFT_MODEL_DIR/Qwen3-ASR-1.7B-8bit
export VOXWEFT_VOICE_PROMPT_WAV=/path/to/5s_reference.wav   # optional but recommended

voxweft bridge --port 8133 &        # poll GET http://127.0.0.1:8133/ready until 200
voxweft serve  --port 3100 &
python scripts/mock_interpret.py --src zh --tgt en            # mic-free end-to-end check
```

`mock_interpret.py` prints per-sentence first-audio latency; report the
median. Then, if the user wants the desktop shell: `cd app && ./build.sh &&
open build/VoxWeft.app` (needs Xcode or a working Command Line Tools install;
see `app/README.md` for the CLT failure mode).

## 6. Report

Tell the user, in this order: chip and memory; `voxweft say` first-audio and
RTF; whether MCP is wired and to which harness; if the runtime was started,
`mock_interpret` median first-audio and which languages were configured.
Anything that failed: quote the exact error and the step number.

## Things that look like errors and are not

* `swift build` finishing in ~3 s after you copied sources: it did not
  compile. `rm -rf .build` and build again (DEVLOG §7.1).
* First TTS request after start-up slower than the rest: warm-up not yet
  complete. Poll `/ready`, not `/health`.
* `ASR label 'Lao' not in Qwen3-ASR support_languages`: expected; those
  languages route to Whisper. Their open-model quality is poor — say so.
