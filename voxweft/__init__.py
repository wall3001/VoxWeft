"""VoxWeft — on-device simultaneous interpretation & voice dialogue on Apple Silicon.

Three layers, one repo:

    voxweft.core      L1  VoxCPM2 on MLX — tokenizer-free TTS, streaming VAE,
                          voice clone, caption. The part that is ours end to end.
    voxweft.runtime   L2  Segmenter / endpointing, ASR + LLM adapters (MLX only),
                          language router, orchestrator, WebSocket server.
    voxweft.wrappers  L3  CLI, MCP server, (Swift app lives out of tree).
"""
__version__ = "0.1.0"
