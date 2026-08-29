"""Objective metrics - the part that turns "nghe cũng được" into a conclusion.

A listening test is the ground truth for TTS, but it does not scale and it does
not go in a results table. These three numbers do, and between them they cover
the three ways a dubbing engine can fail:

    RTF        - is it fast enough to be usable at all?
    SECS       - does the cloned voice sound like the person it copied?
    WER / CER  - did it say the words, or did it garble them?

WER is the one that decides usability for this project. Vietnamese cloning
models are trained mostly on read speech and are documented to break down on
short inputs; subtitle lines are overwhelmingly short. An engine that scores
beautifully on naturalness and drops words on 4-word lines is unusable for
dubbing no matter how good the demo sounded.

None of these replaces listening. They rank candidates so that the listening
test is spent on the two or three engines worth listening to.

Every metric degrades gracefully: if the dependency is missing the field comes
back ``None`` and the report prints "-" rather than the run dying at line 200.
"""
from __future__ import annotations

import functools
import logging
import re
import unicodedata
from pathlib import Path

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ speed ----
def rtf(compute_seconds: float, audio_seconds: float) -> float | None:
    """Real-time factor. Below 1.0 means synthesis outruns playback.

    For a 10-minute film with ~6 minutes of speech, RTF 0.3 is ~2 minutes of
    synthesis; RTF 3.0 is 18 minutes and the pipeline stops being interactive.
    """
    return round(compute_seconds / audio_seconds, 3) if audio_seconds else None


# -------------------------------------------------------- speaker similarity --
@functools.lru_cache(maxsize=1)
def _speaker_encoder():  # noqa: ANN202
    """ECAPA-TDNN, the standard speaker-verification embedding.

    Same family of model the diarization stage already uses, so a similarity
    score here is comparable with how the pipeline itself tells voices apart.
    """
    try:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        log.warning("speechbrain not installed - speaker similarity disabled")
        return None
    try:
        return EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="/tmp/spkrec-ecapa",  # noqa: S108 - Colab scratch
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("could not load the speaker encoder: %s", exc)
        return None


def speaker_similarity(reference: Path, generated: Path) -> float | None:
    """Cosine similarity in ECAPA space, in [-1, 1]. Higher is a closer voice.

    Rules of thumb from speaker-verification practice, for reading the column:

        > 0.80   same speaker, confidently
        0.65-0.80  recognisably the same person
        0.45-0.65  same gender and register, not the same person
        < 0.45   cloning did not happen

    Meaningless for non-cloning engines (MMS, Piper, Edge): they were never
    asked to copy the reference, so their score just measures how similar the
    stock voice happens to be. The report prints it as "n/a" for them.
    """
    encoder = _speaker_encoder()
    if encoder is None:
        return None
    try:
        import torch
        import torchaudio

        def embed(path: Path):  # noqa: ANN202
            wav, sr = torchaudio.load(str(path))
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            return encoder.encode_batch(wav).squeeze()

        cos = torch.nn.functional.cosine_similarity(
            embed(reference), embed(generated), dim=0,
        )
        return round(float(cos), 4)
    except Exception as exc:  # noqa: BLE001
        log.warning("speaker similarity failed: %s", exc)
        return None


# --------------------------------------------------------- intelligibility ---
@functools.lru_cache(maxsize=1)
def _asr(model_size: str = "medium"):  # noqa: ANN202
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        log.warning("faster-whisper not installed - WER/CER disabled")
        return None
    try:
        import torch

        cuda = torch.cuda.is_available()
        return WhisperModel(model_size,
                            device="cuda" if cuda else "cpu",
                            compute_type="float16" if cuda else "int8")
    except Exception as exc:  # noqa: BLE001
        log.warning("could not load the ASR model: %s", exc)
        return None


def _normalize(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace, keep the diacritics.

    Vietnamese diacritics are NOT noise - "ma", "má", "mà", "mã" are different
    words, and an engine that flattens tone is making exactly the error this
    metric exists to catch. NFC first so that composed and decomposed forms of
    the same syllable do not count as a substitution.
    """
    text = unicodedata.normalize("NFC", text.lower())
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _edit_distance(a: list, b: list) -> int:
    """Levenshtein, iterative, O(min(len)) memory."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, token_a in enumerate(a, 1):
        current = [i]
        for j, token_b in enumerate(b, 1):
            current.append(min(
                previous[j] + 1,                                  # deletion
                current[j - 1] + 1,                               # insertion
                previous[j - 1] + (token_a != token_b),           # substitution
            ))
        previous = current
    return previous[-1]


def transcribe(wav: Path, language: str, model_size: str = "medium") -> str | None:
    model = _asr(model_size)
    if model is None:
        return None
    try:
        segments, _ = model.transcribe(str(wav), language=language, beam_size=5)
        return " ".join(s.text for s in segments).strip()
    except Exception as exc:  # noqa: BLE001
        log.warning("ASR round-trip failed on %s: %s", wav.name, exc)
        return None


def intelligibility(expected: str, wav: Path, language: str,
                    model_size: str = "medium") -> dict:
    """Synthesise -> transcribe -> compare. Lower is better; 0.0 is perfect.

    This measures the pair (TTS, ASR), not the TTS alone, so the absolute value
    is not comparable with published WER. Across engines measured with the SAME
    ASR on the SAME sentences the ranking is fair, and the ranking is what the
    comparison needs.
    """
    heard = transcribe(wav, language, model_size)
    if heard is None:
        return {"wer": None, "cer": None, "heard": None}

    ref_w, hyp_w = _normalize(expected).split(), _normalize(heard).split()
    ref_c, hyp_c = list(_normalize(expected)), list(_normalize(heard))
    return {
        "wer": round(_edit_distance(ref_w, hyp_w) / len(ref_w), 4) if ref_w else None,
        "cer": round(_edit_distance(ref_c, hyp_c) / len(ref_c), 4) if ref_c else None,
        "heard": heard,
    }


# ------------------------------------------------------------- naturalness ---
@functools.lru_cache(maxsize=1)
def _mos_predictor():  # noqa: ANN202
    try:
        import torch

        return torch.hub.load("tarepan/SpeechMOS:v1.2.0",
                              "utmos22_strong", trust_repo=True)
    except Exception as exc:  # noqa: BLE001
        log.warning("UTMOS unavailable (%s) - naturalness column will be blank", exc)
        return None


def predicted_mos(wav: Path) -> float | None:
    """UTMOS22: a model trained to predict what listeners score, 1-5.

    Treat it as a coarse screen, not a verdict. It was fitted on English VCC/
    BVCC data, so on Vietnamese it is being used out of domain - fine for
    "is this obviously broken", not fine for "engine A beats B by 0.1 MOS".
    Real conclusions about naturalness still need real listeners.
    """
    predictor = _mos_predictor()
    if predictor is None:
        return None
    try:
        import torchaudio

        wave, sr = torchaudio.load(str(wav))
        if wave.shape[0] > 1:
            wave = wave.mean(dim=0, keepdim=True)
        return round(float(predictor(wave, sr)), 3)
    except Exception as exc:  # noqa: BLE001
        log.warning("MOS prediction failed on %s: %s", wav.name, exc)
        return None
