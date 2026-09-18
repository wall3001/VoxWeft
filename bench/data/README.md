# bench/data

`manifest.json` — the test-set index every bench script reads. Fields:
`id`, `lang` (ISO-639-1), `wav` (file under `wav/`, or `null` for text-only),
`ref` (ground truth), optional `ref_zh` (Chinese parallel for MT chrF),
`dur_s`, `speaker`.

## What ships

`intro/` — the repository's own description, Chinese (6 sentences, ~190
chars) and English (6 sentences, ~110 words). Text-only. Used for:

* TTS round-trip: `bench_tts.py` synthesizes each sentence, `judge_tts.py`
  transcribes it back and reports CER — the intelligibility number we publish.
* Mic-free interpretation: `scripts/mock_interpret.py` feeds these sentences
  through `/ws/voice/v2` as `text_input`, exercising MT → TTS with the same
  events a live session produces, and reports per-sentence first-audio latency.

## What does not ship

The 18 real-speaker recordings (vi ×3 / km ×12 / lo ×3, ~28 min) used in
`docs/DEVLOG.md` and the ASR/MT numbers were collected internally and are not
licensed for redistribution. To reproduce those runs, place your own 16 kHz
mono clips in `wav/` and add them to the manifest with `ref` text. Public
alternatives: FLEURS (vi/km/lo/th/my, CC-BY-4.0), Common Voice (vi/th).

Model weights: see `../models.json` and `docs/MODELS.md`.

## Accent and language references (for voice-clone demos)

Clean read speech with speaker metadata works best as a ~5 s clone
reference. Checked on Hugging Face 2026-09-18 (licence from the dataset card
API; verify before redistribution):

| dataset | what | licence | use |
|---|---|---|---|
| `CSTR-Edinburgh/vctk` | 110 English speakers, `accent` field (English, Scottish, Irish, Indian, American, …), ~400 read sentences each | CC-BY-4.0 | accent clone references |
| `google/fleurs` | 102 languages incl. vi/th/lo/km/my/id/ms/ja/ko/de/fr/es, read sentences | CC-BY-4.0 | per-language references and ASR/MT eval |
| `ylacombe/english_dialects` | British Isles dialects, 120 speakers, 31 h | CC-BY-SA-4.0 | accent references (share-alike) |
| `edinburghcstr/edacc` | 40 h conversational English, linguist-annotated accent + L1 | not stated on card — check homepage | ASR robustness eval, not clean enough for cloning |
| `KoelLabs/L2Arctic` | 24 L2 speakers (vi/ko/zh/es/hi/ar L1), phone-level | **CC-BY-NC-4.0** | do not ship; fine for local experiments |

`scripts/fetch_refs.py` (to do) will stream one clip per accent/language from
VCTK and FLEURS into `samples/refs/`.
