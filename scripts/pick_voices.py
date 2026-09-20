"""Copy the picked candidates from design_voices.py into their final names.

    python scripts/pick_voices.py samples/voices

Reads <dir>/manifest.json, takes every row with "pick": true, writes
<dir>/<lang>_<role>.wav and <dir>/voices.json (id → caption, text). Refuses
to run if a (lang, role) has more than one pick.
"""
import json
import os
import shutil
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "samples/voices"
rows = json.load(open(os.path.join(d, "manifest.json"), encoding="utf-8"))
picks = {}
for r in rows:
    if r.get("pick"):
        key = (r["lang"], r["role"])
        if key in picks:
            sys.exit(f"two picks for {key}: {picks[key]['file']} and {r['file']}")
        picks[key] = r
out = {}
for (lang, role), r in sorted(picks.items()):
    dst = f"{lang}_{role}.wav"
    shutil.copy(os.path.join(d, r["file"]), os.path.join(d, dst))
    out[f"{lang}_{role}"] = {"from": r["file"], "caption": r["caption"], "text": r["text"],
                             "cer": r.get("cer")}
    print(f"{dst:14s} ← {r['file']}")
json.dump(out, open(os.path.join(d, "voices.json"), "w"), ensure_ascii=False, indent=1)
missing = [(l, ro) for l in {r["lang"] for r in rows} for ro in ("src", "tgt") if (l, ro) not in picks]
if missing:
    print("no pick yet for:", ", ".join(f"{l}_{ro}" for l, ro in missing))
