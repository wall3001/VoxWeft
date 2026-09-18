"""`voxweft` — one entry point, four verbs.

  voxweft say  "text" [--ref voice.wav] [-o out.wav] [--stream]   L1 TTS
  voxweft bridge   [--port 8133]        TTS + ASR HTTP (thread-pinned MLX)
  voxweft serve    [--port 3100]        WebSocket runtime: /ws/voice/v2
  voxweft mcp                           MCP server (stdio) for agent harnesses
"""
import sys


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0
    verb, rest = argv[0], argv[1:]
    if verb == "say":
        from .core.cli import main as say
        return say(rest)
    if verb == "bridge":
        from .runtime.bridge import main as bridge
        return bridge(rest)
    if verb == "serve":
        from .runtime.server import main as serve
        return serve(rest)
    if verb == "mcp":
        from .wrappers.mcp_server import main as mcp
        return mcp()
    print(f"unknown verb {verb!r}\n\n{__doc__.strip()}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main() or 0)
