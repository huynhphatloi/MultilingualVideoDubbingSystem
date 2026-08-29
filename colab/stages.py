"""ASR and translation on the GPU - the two models that were eating the disk.

The TTS engines live in `engines/` because there are seven of them and the
whole point is comparing them. These two are different: there is one obvious
model for each job, nobody is benchmarking them against each other, and they
are here for one reason only - whisper-medium is 1.4 GB and
NLLB-200-distilled-600M is 2.3 GB, and that is 3.7 GB of weights a laptop
should not be storing for models it cannot run quickly anyway.

Both are lazy and both unload. A T4 has 16 GB and a cloning TTS model wants 3
of them; loading Whisper, NLLB and viXTTS at once and then wondering why the
notebook OOMs is the failure this avoids.

The wire format keeps the *decision* logic on the client:

* ``/transcribe`` returns raw decoder WINDOWS, not utterances. Turning windows
  into utterances is pure data manipulation with its own tests in the AI
  service; re-implementing it here would give two versions to keep in step.
* ``/translate`` returns the candidate list as well as the pick, so
  duration-aware selection still happens client-side.
"""
from __future__ import annotations

import gc
import threading

_LOCK = threading.Lock()
_WHISPER: dict[str, object] = {}
_MT: dict[str, tuple] = {}


def free_vram() -> list[str]:
    """Drop both models. Called before a TTS engine loads."""
    freed = []
    with _LOCK:
        if _WHISPER:
            freed += [f"whisper:{k}" for k in _WHISPER]
            _WHISPER.clear()
        if _MT:
            freed += [f"mt:{k}" for k in _MT]
            _MT.clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return freed


def loaded() -> dict:
    return {"whisper": sorted(_WHISPER), "translation": sorted(_MT)}


# ------------------------------------------------------------------ ASR ----
def _whisper(size: str):  # noqa: ANN202
    if size not in _WHISPER:
        import torch
        from faster_whisper import WhisperModel

        cuda = torch.cuda.is_available()
        _WHISPER[size] = WhisperModel(
            size,
            device="cuda" if cuda else "cpu",
            # float16 on a T4 is roughly 4x int8-on-CPU and loses nothing that
            # matters for dubbing timestamps.
            compute_type="float16" if cuda else "int8",
        )
    return _WHISPER[size]


def transcribe(audio_path: str, *, language: str | None, model: str,
               word_timestamps: bool, beam_size: int, vad_filter: bool,
               condition_on_previous_text: bool) -> dict:
    """Decode to raw windows. Segmentation stays on the client - see module doc."""
    with _LOCK:
        whisper = _whisper(model)
        segments, info = whisper.transcribe(
            audio_path,
            language=language or None,
            beam_size=beam_size,
            vad_filter=vad_filter,
            vad_parameters={"min_silence_duration_ms": 500} if vad_filter else None,
            word_timestamps=word_timestamps,
            condition_on_previous_text=condition_on_previous_text,
        )

        windows = []
        for seg in segments:
            text = (seg.text or "").strip()
            if not text:
                continue
            item = {
                "start": float(seg.start),
                "end": float(seg.end),
                "source_text": text,
                "avg_logprob": getattr(seg, "avg_logprob", None),
                "no_speech_prob": getattr(seg, "no_speech_prob", None),
            }
            if word_timestamps and getattr(seg, "words", None):
                item["words"] = [
                    {"word": w.word, "start": round(float(w.start), 3),
                     "end": round(float(w.end), 3),
                     "probability": getattr(w, "probability", None)}
                    for w in seg.words if w.start is not None and w.end is not None
                ]
            windows.append(item)

    return {
        "language": info.language,
        "language_confidence": float(getattr(info, "language_probability", 0.0) or 0.0),
        "duration": float(getattr(info, "duration", 0.0) or 0.0),
        "model": model,
        "windows": windows,
    }


# ---------------------------------------------------------- translation ----
#: ISO-639-1 -> the FLORES-200 code NLLB actually wants. Only the languages
#: this project targets; an unlisted one is a 400 rather than a silent
#: mistranslation into whatever the tokenizer guessed.
_FLORES = {
    "en": "eng_Latn", "vi": "vie_Latn", "fr": "fra_Latn", "de": "deu_Latn",
    "es": "spa_Latn", "it": "ita_Latn", "pt": "por_Latn", "ru": "rus_Cyrl",
    "ja": "jpn_Jpan", "ko": "kor_Hang", "zh": "zho_Hans", "th": "tha_Thai",
    "id": "ind_Latn", "hi": "hin_Deva", "ar": "arb_Arab", "tr": "tur_Latn",
    "nl": "nld_Latn", "pl": "pol_Latn", "uk": "ukr_Cyrl", "cs": "ces_Latn",
    "ro": "ron_Latn", "hu": "hun_Latn", "el": "ell_Grek", "sv": "swe_Latn",
    "da": "dan_Latn", "fi": "fin_Latn", "no": "nob_Latn", "he": "heb_Hebr",
    "fa": "pes_Arab", "ms": "zsm_Latn", "ta": "tam_Taml", "bn": "ben_Beng",
    "mn": "khk_Cyrl", "km": "khm_Khmr", "lo": "lao_Laoo", "my": "mya_Mymr",
}

_REPOS = {
    "nllb": "facebook/nllb-200-distilled-600M",
    "seamless": "facebook/hf-seamless-m4t-medium",
}


def flores(code: str) -> str | None:
    return _FLORES.get((code or "").lower().split("-")[0])


def _translator(engine: str):  # noqa: ANN202
    if engine not in _REPOS:
        raise ValueError(f"unknown translation engine {engine!r}; have {sorted(_REPOS)}")
    if engine not in _MT:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        repo = _REPOS[engine]
        tokenizer = AutoTokenizer.from_pretrained(repo)
        model = AutoModelForSeq2SeqLM.from_pretrained(repo)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        _MT[engine] = (tokenizer, model)
    return _MT[engine]


def translate(items: list[dict], *, source_language: str, target_language: str,
              engine: str = "nllb", num_beams: int = 4,
              max_new_tokens: int = 256) -> dict:
    """Translate a whole job in one call.

    Batched on purpose: the pipeline hands over every segment at once, so the
    round-trip is paid once per job instead of once per line - which over a
    home connection is the difference between a 2-second stage and a 40-second
    one.

    `char_budget` asks for a rendering that fits the original's time slot. Beam
    search already produces several hypotheses on the way to its answer, so
    returning them costs nothing extra and lets the client pick the one closest
    to the budget with the duration-aware logic it already has.
    """
    src, tgt = flores(source_language), flores(target_language)
    if not src or not tgt:
        raise ValueError(
            f"no FLORES-200 code for {source_language!r} -> {target_language!r}; "
            f"known: {sorted(_FLORES)}"
        )

    texts = [(item.get("text") or "").strip() for item in items]
    wants_candidates = any(item.get("char_budget") for item in items)

    with _LOCK:
        import torch

        tokenizer, model = _translator(engine)
        tokenizer.src_lang = src
        bos = _target_token(tokenizer, tgt)

        results: list[dict] = []
        # Modest batches: NLLB's memory grows with the longest line in the
        # batch, and subtitle lines vary from one word to thirty.
        for start in range(0, len(texts), 16):
            chunk = texts[start:start + 16]
            blank = [i for i, t in enumerate(chunk) if not t]
            live = [t for t in chunk if t]
            decoded: list[list[str]] = []
            if live:
                encoded = tokenizer(live, return_tensors="pt", padding=True,
                                    truncation=True, max_length=512).to(model.device)
                returns = num_beams if wants_candidates else 1
                with torch.inference_mode():
                    output = model.generate(
                        **encoded,
                        forced_bos_token_id=bos,
                        num_beams=num_beams,
                        num_return_sequences=returns,
                        max_new_tokens=max_new_tokens,
                    )
                flat = tokenizer.batch_decode(output, skip_special_tokens=True)
                decoded = [flat[i * returns:(i + 1) * returns] for i in range(len(live))]

            cursor = 0
            for i in range(len(chunk)):
                if i in blank:
                    results.append({"text": "", "candidates": []})
                else:
                    options = decoded[cursor]
                    cursor += 1
                    results.append({"text": options[0], "candidates": options})

    return {"engine": engine, "source": src, "target": tgt, "results": results}


def _target_token(tokenizer, code: str) -> int:  # noqa: ANN001
    """The forced BOS id for the target language.

    transformers moved this API twice: `lang_code_to_id` on the NLLB tokenizer,
    then `convert_tokens_to_ids`. Trying both means the notebook is not pinned
    to one transformers release, which on Colab is not something we control.
    """
    table = getattr(tokenizer, "lang_code_to_id", None)
    if isinstance(table, dict) and code in table:
        return table[code]
    token_id = tokenizer.convert_tokens_to_ids(code)
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise ValueError(f"tokenizer does not know the language token {code!r}")
    return token_id
