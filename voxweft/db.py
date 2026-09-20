"""voxweft.db — the seed database: voices, showcase texts, demo routes.

One SQLite file, `voxweft.db`, shipped at the repo root and copied to
`~/.voxweft/voxweft.db` on first use. It is the default configuration for
the runtime and the source of truth for the showcase:

    voices     id, lang, role, caption, text, wav (path), cer, source
    intro      lang, idx, text            — the repo's self-introduction, per language
    showcase   src, tgt, title, note      — the demo routes we recommend

Why a database and not JSON: the bridge, the server, the app and the bench
all need the same answers ("which voice speaks Malay?", "what does the
Spanish intro say?"), and a single file with a schema is easier to keep
consistent than four loaders. sqlite3 is in the standard library; no dep.

    python -m voxweft.db build      # regenerate voxweft.db from samples/voices + bench/data/intro
    python -m voxweft.db show       # dump
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SEED = REPO / "voxweft.db"
USER = Path(os.environ.get("VOXWEFT_DB", os.path.expanduser("~/.voxweft/voxweft.db")))

SCHEMA = """
CREATE TABLE IF NOT EXISTS voices (
    id      TEXT PRIMARY KEY,        -- e.g. es_tgt
    lang    TEXT NOT NULL,           -- ISO-639-1
    role    TEXT NOT NULL,           -- src | tgt
    caption TEXT NOT NULL,           -- the voice-design caption that produced it
    text    TEXT NOT NULL,           -- exact transcript of wav (clone prompt_text)
    wav     TEXT NOT NULL,           -- path relative to repo root
    cer     REAL,                    -- ASR round-trip CER at design time
    source  TEXT NOT NULL DEFAULT 'designed'
);
CREATE TABLE IF NOT EXISTS intro (
    lang TEXT NOT NULL, idx INTEGER NOT NULL, text TEXT NOT NULL,
    PRIMARY KEY (lang, idx)
);
CREATE TABLE IF NOT EXISTS showcase (
    src TEXT NOT NULL, tgt TEXT NOT NULL, title TEXT NOT NULL, note TEXT,
    ord INTEGER NOT NULL, PRIMARY KEY (src, tgt)
);
"""

# The demo matrix. Three regions, equal footing; the last block is the point
# of the router — pairs where neither side is zh or en, translated in one hop.
SHOWCASE = [
    ("zh", "en", "baseline", "the pair everyone tries first"),
    ("en", "zh", "baseline", ""),
    ("en", "de", "Europe", "accented voices"),
    ("de", "fr", "Europe", ""),
    ("es", "it", "Europe", ""),
    ("ja", "ko", "East Asia", "both CJK, mutually unintelligible"),
    ("zh", "ja", "East Asia", ""),
    ("zh", "vi", "ASEAN", ""),
    ("en", "th", "ASEAN", ""),
    ("vi", "id", "ASEAN", ""),
    ("es", "ms", "no pivot", "neither side is zh/en; one hop, not es→en→ms"),
    ("de", "vi", "no pivot", ""),
    ("ja", "fr", "no pivot", ""),
]


def connect(path: Path | None = None) -> sqlite3.Connection:
    """Open the user DB, seeding it from the repo copy on first use."""
    p = path or USER
    if not p.exists() and SEED.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(SEED, p)
    c = sqlite3.connect(p)
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    return c


def build(out: Path = SEED) -> None:
    if out.exists():
        out.unlink()
    c = sqlite3.connect(out)
    c.executescript(SCHEMA)
    vj = REPO / "samples" / "voices" / "voices.json"
    if vj.exists():
        for vid, v in json.load(open(vj, encoding="utf-8")).items():
            lang, role = vid.split("_", 1)
            c.execute("INSERT INTO voices VALUES (?,?,?,?,?,?,?,?)",
                      (vid, lang, role, v["caption"], v["text"],
                       f"samples/voices/{vid}.wav", v.get("cer"), "designed"))
    intro = REPO / "bench" / "data" / "intro"
    for f in sorted(intro.glob("intro_*.txt")):
        lang = f.stem.split("_", 1)[1]
        for i, line in enumerate([l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.strip()], 1):
            c.execute("INSERT INTO intro VALUES (?,?,?)", (lang, i, line))
    for i, (s, t, title, note) in enumerate(SHOWCASE, 1):
        c.execute("INSERT INTO showcase VALUES (?,?,?,?,?)", (s, t, title, note, i))
    c.commit()
    n = {t: c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("voices", "intro", "showcase")}
    c.close()
    print(f"{out.relative_to(REPO)}: {n}")


# ── queries the runtime uses ─────────────────────────────────────────────────

def voice_for(lang: str, role: str = "tgt", c: sqlite3.Connection | None = None):
    c = c or connect()
    return c.execute("SELECT * FROM voices WHERE lang=? AND role=?", (lang, role)).fetchone()


def voices(c: sqlite3.Connection | None = None):
    c = c or connect()
    return c.execute("SELECT * FROM voices ORDER BY lang, role").fetchall()


def intro_lines(lang: str, c: sqlite3.Connection | None = None) -> list[str]:
    c = c or connect()
    return [r["text"] for r in c.execute("SELECT text FROM intro WHERE lang=? ORDER BY idx", (lang,))]


def showcase(c: sqlite3.Connection | None = None):
    c = c or connect()
    return c.execute("SELECT * FROM showcase ORDER BY ord").fetchall()


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    cmd = argv[0] if argv else "show"
    if cmd == "build":
        build()
        return 0
    c = connect(SEED if SEED.exists() else None)
    print("voices:")
    for r in voices(c):
        print(f"  {r['id']:8s} cer={r['cer']!s:6s} {r['wav']}")
    print("intro languages:", ", ".join(sorted({r[0] for r in c.execute('SELECT DISTINCT lang FROM intro')})))
    print("showcase:")
    for r in showcase(c):
        print(f"  {r['src']}→{r['tgt']:3s} {r['title']:10s} {r['note'] or ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
