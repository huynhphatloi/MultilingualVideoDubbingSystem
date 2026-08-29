"""Turn raw Whisper output into utterance-sized dubbing segments.

Why this exists
---------------
faster-whisper does not return utterances, it returns *decoder windows*. With
``vad_filter=True`` the audio is de-silenced before decoding and the timestamps
are mapped back onto the original timeline afterwards, so a single returned
segment routinely spans the silence it was never spoken over::

    [ 61.38 - 115.05]  "Call it, Captain."      <- 3 words, 54 seconds

For a subtitle burner that is merely ugly. For a dubbing pipeline it is fatal:

* ``duration_ratio = generated / (end - start)`` becomes ~0.02, so every such
  segment is classified ``retranslate``, burns both adapt attempts, and finally
  gets force-stretched - a 1 s take slowed down to fill a 54 s hole;
* the take is placed at ``start`` and the remaining ~53 s carry background
  music only. That is exactly the "there is music but nobody speaks" failure.

So the ASR stage never hands its raw windows downstream. This module re-cuts
them into real utterances using the word timestamps Whisper already produces:

* tighten every segment onto its first/last word (kills the trailing silence);
* start a new utterance whenever the pause between two words exceeds
  ``ASR_SPLIT_GAP_SECONDS``;
* also split at sentence punctuation once an utterance is long enough, and
  hard-split anything still longer than ``ASR_MAX_SEGMENT_SECONDS``;
* drop empty/duplicated windows (Whisper's classic repetition loop).

Everything here is pure data - no models, no I/O - so it is cheap to unit-test.
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

#: Sentence-final punctuation, incl. the CJK / Arabic / Devanagari variants.
_CLOSERS = r"['\"”’)\]]*\s*$"
_SENTENCE_END = re.compile(r"[.!?。．！？…؟।]" + _CLOSERS)
#: Clause-level punctuation - a weaker split point, used only when a segment
#: would otherwise blow past the hard length cap.
_CLAUSE_END = re.compile(r"[,;:，、；：]" + _CLOSERS)


#: Whisper only calls a window "silence" when the no-speech token is likely
#: AND the decode itself went badly. Using no_speech_prob on its own is wrong:
#: on a Demucs vocals stem a whole scene of real dialogue comes back at
#: no_speech_prob ~0.84 and would be thrown away.
_HALLUCINATION_LOGPROB = -1.0


def build_utterances(
    segments: list[dict],
    *,
    max_seconds: float = 12.0,
    split_gap: float = 0.7,
    min_seconds: float = 0.35,
    no_speech_threshold: float = 0.8,
    total_duration: float | None = None,
) -> list[dict]:
    """Re-cut raw Whisper segments into utterance-sized dubbing segments.

    ``segments`` are dicts as produced by :func:`app.services.asr.transcribe`
    (``start``/``end``/``source_text`` and optionally ``words``).
    """
    utterances: list[dict] = []

    for seg in segments:
        if _is_noise(seg, no_speech_threshold):
            log.debug("dropping non-speech window [%.2f-%.2f] %r",
                      seg.get("start", 0), seg.get("end", 0), seg.get("source_text"))
            continue

        words = _usable_words(seg)
        if words:
            utterances.extend(_split_on_words(seg, words, max_seconds, split_gap))
        else:
            utterances.extend(_split_without_words(seg, max_seconds))

    utterances = _drop_repetitions(utterances)
    utterances = _merge_slivers(utterances, min_seconds, split_gap)
    utterances = _enforce_monotonic(utterances, total_duration)

    for index, utt in enumerate(utterances):
        utt["segment_id"] = index
        utt["start"] = round(utt["start"], 3)
        utt["end"] = round(utt["end"], 3)
        utt["duration"] = round(utt["end"] - utt["start"], 3)

    return utterances


# ------------------------------------------------------------------ filters --
def _is_noise(seg: dict, no_speech_threshold: float) -> bool:
    """Whisper's own silence rule: unlikely speech AND a bad decode.

    Both conditions matter. ``no_speech_prob`` is computed per 30 s decoder
    window and runs high on anything that does not sound like clean studio
    speech - a Demucs vocals stem, a phone call, a crowd. Dropping on it alone
    deletes real dialogue; requiring a poor ``avg_logprob`` as well leaves only
    the actual hallucinations ("Thanks for watching!", subtitle credits).
    """
    if not (seg.get("source_text") or "").strip():
        return True

    prob = seg.get("no_speech_prob")
    logprob = seg.get("avg_logprob")
    if prob is None or float(prob) < no_speech_threshold:
        return False
    return logprob is not None and float(logprob) < _HALLUCINATION_LOGPROB


def _usable_words(seg: dict) -> list[dict]:
    """Words with sane, monotonically increasing timestamps."""
    out: list[dict] = []
    cursor = float(seg.get("start", 0.0))
    for word in seg.get("words") or []:
        if word.get("start") is None or word.get("end") is None:
            continue
        start, end = float(word["start"]), float(word["end"])
        if end <= start:
            end = start + 0.02
        if start < cursor - 0.25:        # timestamps went backwards - untrustworthy
            return []
        text = str(word.get("word") or "")
        if not text.strip():
            continue
        out.append({**word, "start": start, "end": end, "word": text})
        cursor = end
    return out


# ------------------------------------------------------------------- splits --
def _ends_utterance(*, gap: float, span: float, text_so_far: str,
                    max_seconds: float, split_gap: float) -> bool:
    """Should the utterance being built end *before* the next word?

    Four reasons, most trustworthy first. They are spelled out rather than
    folded into one boolean expression because the thresholds are the tuning
    surface of this whole module.
    """
    # 1. A real pause. The only signal that always means "new utterance".
    if gap >= split_gap:
        return True
    # 2. Sentence punctuation, once the utterance is worth its own take. Below
    #    half the cap, splitting on every full stop produces one-word takes.
    if span > max_seconds * 0.5 and _SENTENCE_END.search(text_so_far):
        return True
    # 3. Over the cap: accept a weaker boundary - a comma, or any small pause.
    if span > max_seconds and (_CLAUSE_END.search(text_so_far) or gap >= 0.2):
        return True
    # 4. Still going. Cut mid-clause rather than hand the synchroniser a take
    #    it can neither place nor stretch.
    return span > max_seconds * 1.5


def _split_on_words(seg: dict, words: list[dict], max_seconds: float,
                    split_gap: float) -> list[dict]:
    chunks: list[list[dict]] = []
    current: list[dict] = []

    for word in words:
        if current and _ends_utterance(
            gap=word["start"] - current[-1]["end"],
            span=word["end"] - current[0]["start"],
            text_so_far="".join(w["word"] for w in current),
            max_seconds=max_seconds,
            split_gap=split_gap,
        ):
            chunks.append(current)
            current = []
        current.append(word)

    if current:
        chunks.append(current)

    return [_from_words(seg, chunk) for chunk in chunks if _text_of(chunk)]


def _split_without_words(seg: dict, max_seconds: float) -> list[dict]:
    """No word timings: split proportionally by sentence, keeping the span."""
    start, end = float(seg["start"]), float(seg["end"])
    text = (seg.get("source_text") or "").strip()
    span = end - start

    if span <= max_seconds:
        return [_carry(seg, start, end, text)]

    sentences = [s for s in re.split(r"(?<=[.!?。！？…])\s+", text) if s.strip()]
    if len(sentences) < 2:
        # One long unpunctuated blob: keep the text but clamp the slot, so the
        # ratio maths downstream stays meaningful instead of dividing by 60 s.
        return [_carry(seg, start, min(end, start + max_seconds), text)]

    total_chars = sum(len(s) for s in sentences)
    out, cursor = [], start
    for sentence in sentences:
        share = len(sentence) / total_chars
        chunk_end = min(end, cursor + span * share)
        out.append(_carry(seg, cursor, chunk_end, sentence.strip()))
        cursor = chunk_end
    return out


def _from_words(seg: dict, words: list[dict]) -> dict:
    item = _carry(seg, words[0]["start"], words[-1]["end"], _text_of(words))
    item["words"] = words
    return item


def _carry(seg: dict, start: float, end: float, text: str) -> dict:
    """Copy the per-window metadata that stays valid after a split."""
    return {
        "start": float(start),
        "end": float(max(end, start + 0.05)),
        "source_text": text,
        "avg_logprob": seg.get("avg_logprob"),
        "no_speech_prob": seg.get("no_speech_prob"),
    }


def _text_of(words: list[dict]) -> str:
    return "".join(w["word"] for w in words).strip()


# ------------------------------------------------------------------ tidy up --
def _drop_repetitions(utterances: list[dict]) -> list[dict]:
    """Whisper's repetition loop emits the same line over and over."""
    out: list[dict] = []
    for utt in utterances:
        key = utt["source_text"].strip().lower()
        if out and key and key == out[-1]["source_text"].strip().lower():
            log.debug("dropping repeated line %r at %.2fs", utt["source_text"], utt["start"])
            continue
        out.append(utt)
    return out


def _merge_slivers(utterances: list[dict], min_seconds: float,
                   split_gap: float) -> list[dict]:
    """Fold sub-``min_seconds`` fragments into the neighbour they belong to."""
    out: list[dict] = []
    for utt in utterances:
        too_short = (utt["end"] - utt["start"]) < min_seconds
        if too_short and out and (utt["start"] - out[-1]["end"]) < split_gap:
            previous = out[-1]
            previous["end"] = max(previous["end"], utt["end"])
            previous["source_text"] = f"{previous['source_text']} {utt['source_text']}".strip()
            if previous.get("words") and utt.get("words"):
                previous["words"] = previous["words"] + utt["words"]
            continue
        out.append(utt)
    return out


def _enforce_monotonic(utterances: list[dict], total_duration: float | None) -> list[dict]:
    """Guarantee 0 <= start < end and no overlap: the sync stage relies on it."""
    out: list[dict] = []
    for utt in sorted(utterances, key=lambda u: (u["start"], u["end"])):
        start = max(0.0, float(utt["start"]))
        end = float(utt["end"])
        if out and start < out[-1]["end"]:
            start = out[-1]["end"]
        if total_duration:
            end = min(end, total_duration)
            start = min(start, max(0.0, total_duration - 0.05))
        if end - start < 0.05:
            # Fully swallowed by its predecessor - keep the text by appending it
            # rather than silently losing a line of dialogue.
            if out:
                out[-1]["source_text"] = (
                    f"{out[-1]['source_text']} {utt['source_text']}".strip()
                )
            continue
        utt["start"], utt["end"] = start, end
        out.append(utt)
    return out
