"""Stage 6: translation of every segment, with fallback and duration awareness."""
from __future__ import annotations

import logging

from app.core import languages
from app.core.config import settings
from app.core.errors import TranslationError
from app.services.translation import duration_aware, glossary, router
from app.services.translation.base import TranslationRequest
from app.services.workspace import JobWorkspace

log = logging.getLogger(__name__)


def translate_job(job_id: str, segments_key: str, target_language: str,
                  source_language: str | None = None, engine: str = "auto",
                  duration_aware_mode: bool = True) -> dict:
    ws = JobWorkspace(job_id)
    document = ws.pull_json(segments_key)
    segments = document.get("segments", [])
    if not segments:
        raise TranslationError("No segments to translate", details={"key": segments_key})

    source = languages.normalize(source_language or document.get("source_language"))
    target = languages.normalize(target_language)
    if not source:
        raise TranslationError("Source language is unknown - run transcription first.")
    if not target:
        raise TranslationError(f"Unsupported target language '{target_language}'")

    if source == target:
        log.warning("source == target (%s); copying source text through", source)
        for seg in segments:
            seg["translated_text"] = seg["source_text"]
            seg["target_language"] = target
        used_engine = "passthrough"
    else:
        terms = glossary.terms_for(source)
        masks: list[dict[str, str]] = []
        protected: list[str] = []
        for seg in segments:
            masked, mapping = glossary.protect(seg["source_text"], terms)
            protected.append(masked)
            masks.append(mapping)
        if terms:
            log.info("glossary: %d protected term(s), applied to %d segment(s)",
                     len(terms), sum(1 for m in masks if m))

        requests = [
            TranslationRequest(
                text=text,
                source_language=source,
                target_language=target,
                char_budget=(
                    duration_aware.estimate_budget(
                        duration=max(0.2, seg["end"] - seg["start"]),
                        target_language=target,
                        source_text=seg["source_text"],
                        source_language=source,
                        target_chars_per_second=document.get("tts_chars_per_second"),
                    ) if duration_aware_mode else None
                ),
            )
            for seg, text in zip(segments, protected, strict=True)
        ]
        translations, used_engine = _run_with_fallback(requests, engine, source, target)
        for seg, result, mapping in zip(segments, translations, masks, strict=True):
            seg["translated_text"] = glossary.restore(result.text, mapping)
            seg["target_language"] = target
            seg["char_budget"] = result.char_budget
            seg["translation_engine"] = result.engine

    document["target_language"] = target
    document["source_language"] = source
    document["translation_engine"] = used_engine
    document["segments"] = segments

    segments_out = ws.push_json(document, ws.layout.segments)
    translation_out = ws.push_json(
        {
            "job_id": job_id,
            "source_language": source,
            "target_language": target,
            "engine": used_engine,
            "segments": [
                {
                    "segment_id": s["segment_id"],
                    "start": s["start"],
                    "end": s["end"],
                    "duration": round(s["end"] - s["start"], 3),
                    "speaker_id": s["speaker_id"],
                    "source_text": s["source_text"],
                    "translated_text": s["translated_text"],
                }
                for s in segments
            ],
        },
        ws.layout.translation(target),
    )

    return {
        "translation_key": translation_out,
        "segments_key": segments_out,
        "engine": used_engine,
        "segment_count": len(segments),
        "source_language": source,
        "target_language": target,
    }


def adapt(job_id: str, segments_key: str, segment_ids: list[int],
          engine: str = "auto") -> dict:
    """Re-translate the segments whose synthesised audio missed its slot.

    The new character budget comes from the *measured* ratio, so the second
    take is aimed at the real speaking rate of the TTS voice.
    """
    ws = JobWorkspace(job_id)
    document = ws.pull_json(segments_key)
    segments = document.get("segments", [])
    by_id = {s["segment_id"]: s for s in segments}

    source = languages.normalize(document.get("source_language"))
    target = languages.normalize(document.get("target_language"))
    if not source or not target:
        raise TranslationError("Job has no language pair yet - translate first.")

    targets = [by_id[i] for i in segment_ids if i in by_id]
    targets = [s for s in targets
               if s.get("sync_action") == "retranslate" or s.get("duration_ratio")]
    if not targets:
        return {"segments_key": segments_key, "adapted": [], "unchanged": list(segment_ids)}

    # The engine's own measured rate, when synthesis has already run once.
    rate = document.get("tts_chars_per_second")

    requests, adapting = [], []
    for seg in targets:
        if seg.get("attempts", 0) >= settings.sync_max_retranslate_attempts:
            log.info("segment %s hit the retranslate cap", seg["segment_id"])
            continue
        ratio = seg.get("duration_ratio") or 1.0
        if rate:
            # We know how many characters this voice fits into this slot.
            budget = duration_aware.budget_for(
                max(0.2, seg["end"] - seg["start"]), target, rate)
        else:
            budget = duration_aware.budget_from_ratio(
                seg.get("translated_text") or "", ratio)
        masked, mapping = glossary.protect(seg["source_text"], glossary.terms_for(source))
        requests.append(TranslationRequest(
            text=masked, source_language=source, target_language=target,
            char_budget=budget,
            metadata={"hint": "shorter" if ratio > 1 else "longer", "mask": mapping},
        ))
        adapting.append(seg)

    if not requests:
        return {"segments_key": segments_key, "adapted": [], "unchanged": list(segment_ids)}

    results, used_engine = _run_with_fallback(requests, engine, source, target)

    adapted_ids: list[int] = []
    identical: list[int] = []
    for seg, result, request in zip(adapting, results, requests, strict=True):
        previous = seg.get("translated_text")
        request_mask = request.metadata.get("mask") or {}
        # A shorter budget does not guarantee a shorter translation: for a
        # five-word line NLLB returns the same sentence whatever length penalty
        # it is given. Re-voicing identical text would cost a full synthesis
        # pass and one of the two retranslate attempts to produce byte-identical
        # audio, so the segment is left alone and handed to the time-stretcher.
        restored = glossary.restore(result.text, request_mask)
        if (restored or "").strip() == (previous or "").strip():
            seg["attempts"] = settings.sync_max_retranslate_attempts
            seg["sync_action"] = "stretch"
            seg["forced_stretch"] = True
            identical.append(seg["segment_id"])
            continue

        seg["previous_translations"] = (seg.get("previous_translations") or []) + [previous]
        seg["translated_text"] = restored
        seg["char_budget"] = result.char_budget
        seg["translation_engine"] = result.engine
        seg["attempts"] = seg.get("attempts", 0) + 1
        seg["sync_action"] = "pending"
        seg["generated_audio"] = None
        seg["generated_duration"] = None
        seg["duration_ratio"] = None
        adapted_ids.append(seg["segment_id"])

    document["segments"] = segments
    key = ws.push_json(document, ws.layout.segments)
    log.info("adapted %d segment(s) with %s (%d came back unchanged)",
             len(adapted_ids), used_engine, len(identical))
    return {
        "segments_key": key,
        "adapted": adapted_ids,
        "unchanged": [i for i in segment_ids if i not in adapted_ids],
        "identical": identical,
        "engine": used_engine,
    }


def apply_review(job_id: str, edits: list[dict]) -> dict:
    """Human-in-the-loop: overwrite machine translations with user edits."""
    ws = JobWorkspace(job_id)
    document = ws.pull_json(ws.layout.segments)
    by_id = {s["segment_id"]: s for s in document.get("segments", [])}

    applied: list[int] = []
    for edit in edits:
        seg = by_id.get(int(edit.get("segment_id", -1)))
        if seg is None:
            continue
        text = (edit.get("translated_text") or "").strip()
        if not text or text == seg.get("translated_text"):
            continue
        seg["previous_translations"] = (seg.get("previous_translations") or []) + [
            seg.get("translated_text")
        ]
        seg["translated_text"] = text
        seg["edited_by_user"] = True
        seg["sync_action"] = "pending"
        seg["generated_audio"] = None
        seg["generated_duration"] = None
        seg["duration_ratio"] = None
        applied.append(seg["segment_id"])

    key = ws.push_json(document, ws.layout.segments)
    target = document.get("target_language")
    if target:
        ws.push_json(
            {
                "job_id": job_id,
                "source_language": document.get("source_language"),
                "target_language": target,
                "engine": "human-reviewed",
                "segments": [
                    {k: s.get(k) for k in
                     ("segment_id", "start", "end", "speaker_id", "source_text", "translated_text")}
                    for s in document.get("segments", [])
                ],
            },
            ws.layout.translation(target),
        )
    return {"segments_key": key, "edited": applied, "count": len(applied)}


# ---------------------------------------------------------------- internals --
def _run_with_fallback(requests, engine: str, source: str, target: str):  # noqa: ANN001
    chain = router.resolve_chain(engine)
    errors: list[str] = []
    for candidate in chain:
        if not candidate.supports(source, target):
            errors.append(f"{candidate.name}: language pair unsupported")
            continue
        try:
            log.info("translating %d segment(s) with %s", len(requests), candidate.name)
            return candidate.translate_batch(requests), candidate.name
        except Exception as exc:  # noqa: BLE001 - deliberately try the next engine
            log.warning("engine %s failed: %s", candidate.name, exc)
            errors.append(f"{candidate.name}: {exc}")
    raise TranslationError(
        f"All translation engines failed for {source}->{target}",
        details={"attempts": errors},
    )
