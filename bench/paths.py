"""Bench paths — one place that knows where things live.

Layout (all overridable):

    VOXWEFT_BENCH_DIR   default: this directory
        data/manifest.json   test-set index (id, lang, wav, ref, …)
        data/wav/            16 kHz mono clips referenced by the manifest
        results/             *.json outputs
        tts_out/             synthesized wavs for the round-trip judge
    VOXWEFT_MODEL_DIR   default: ~/.voxweft/models

Model lookup: `model_path("qwen17")` resolves a logical name through
`models.json` (list of candidate globs, relative to VOXWEFT_MODEL_DIR unless
absolute), after checking `VOXWEFT_MODEL_QWEN17` for a direct override.
Missing models raise with the candidates that were tried — no silent
fallback to a path that does not exist.
"""
import glob
import json
import os

BENCH = os.environ.get("VOXWEFT_BENCH_DIR", os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.expanduser(os.environ.get("VOXWEFT_MODEL_DIR", "~/.voxweft/models"))
DATA = os.path.join(BENCH, "data")
WAV = os.path.join(DATA, "wav")
RESULTS = os.path.join(BENCH, "results")
TTS_OUT = os.path.join(BENCH, "tts_out")
MANIFEST = os.path.join(DATA, "manifest.json")

_REGISTRY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.json")


def manifest():
    with open(MANIFEST, encoding="utf-8") as f:
        return json.load(f)


def wav_path(item):
    return item["wav"] if os.path.isabs(item["wav"]) else os.path.join(WAV, item["wav"])


def model_path(name: str) -> str:
    env = os.environ.get("VOXWEFT_MODEL_" + name.upper().replace("-", "_"))
    if env:
        return os.path.expanduser(env)
    with open(_REGISTRY, encoding="utf-8") as f:
        reg = json.load(f)
    if name not in reg:
        raise KeyError(f"unknown model {name!r}; known: {sorted(reg)}")
    tried = []
    for cand in reg[name]["paths"]:
        cand = os.path.expanduser(cand)
        if not os.path.isabs(cand):
            cand = os.path.join(MODELS, cand)
        tried.append(cand)
        hits = sorted(glob.glob(cand))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"model {name!r} not found. Tried:\n  " + "\n  ".join(tried)
        + f"\nSet VOXWEFT_MODEL_{name.upper().replace('-', '_')} or see docs/MODELS.md")


def model_kind(name: str) -> str:
    with open(_REGISTRY, encoding="utf-8") as f:
        return json.load(f)[name]["kind"]


def ensure_dirs():
    for d in (RESULTS, TTS_OUT):
        os.makedirs(d, exist_ok=True)
