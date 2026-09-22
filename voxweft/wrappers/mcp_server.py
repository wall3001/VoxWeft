"""MCP server (L3 wrapper) — plug VoxWeft TTS into any agent harness
(Claude Code / Codex / Cursor / anything speaking MCP).

Config example (Claude Code `.mcp.json`):

    {"mcpServers": {"voxweft": {
        "command": "voxweft", "args": ["mcp"],
        "env": {"VOXCPM2_MODEL": "/path/to/VoxCPM2"}}}}
"""

import os
import tempfile

# mcp 2.x renamed FastMCP → MCPServer (mcp.server.mcpserver) and removed the
# old module; 1.x only has FastMCP. The three calls we make — construct,
# `.tool()`, `.run()` — have the same shape in both, so one alias covers it.
try:
    from mcp.server.mcpserver import MCPServer as _Server        # mcp >= 2.0
except ImportError:                                               # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

mcp = _Server("voxweft")
_pipe = None


def _pipeline():
    global _pipe
    if _pipe is None:
        from voxweft.core import VoxCPM2Pipeline
        model = os.environ.get("VOXCPM2_MODEL")
        if not model:
            raise RuntimeError("set VOXCPM2_MODEL to the model directory")
        _pipe = VoxCPM2Pipeline(model, inference_timesteps=6, quantize_lm=4)
    return _pipe


@mcp.tool()
def synthesize(text: str, ref_wav_path: str = "", out_path: str = "",
               play: bool = False) -> str:
    """Synthesize speech from text (any of 30 languages incl. zh/en/th/lo/km/my).
    Optional ref_wav_path clones the voice from a short (~5s) sample.
    Returns the output wav path (48 kHz)."""
    import numpy as np
    import soundfile as sf

    pipe = _pipeline()
    ref_latents = None
    if ref_wav_path:
        wav, sr = sf.read(ref_wav_path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            from scipy.signal import resample
            wav = resample(wav, int(len(wav) * 16000 / sr)).astype(np.float32)
        ref_latents = pipe.encode_reference(wav)

    out, stats = pipe.generate(text, ref_latents=ref_latents)
    if not out_path:
        out_path = tempfile.mktemp(suffix=".wav", prefix="voxweft_")
    sf.write(out_path, out, 48000)
    if play:
        os.system(f"afplay '{out_path}' &")
    return f"{out_path} ({stats['audio_s']:.2f}s, rtf {stats['rtf']:.2f})"


def main():
    mcp.run()


if __name__ == "__main__":
    main()
