"""Translation on the Colab GPU - the cheapest stage to move, by far.

NLLB-200-distilled-600M is 2.3 GB on disk, the single largest model this
pipeline pulls. Moving it is also the least invasive of the remote stages,
because translation is the one that sends **only text**: a whole film's
subtitles are a few tens of kilobytes. There is no privacy trade here and no
bandwidth trade either - just 2.3 GB of disk that stops existing.

The batch shape matters. The pipeline translates every segment of a job in one
call, so the round-trip cost is paid once per job rather than once per line;
sending them individually would turn a 2-second stage into a 40-second one over
a home connection.

Duration-aware translation (`char_budget`) travels with each item. The remote
side returns the candidate list as well as its pick, so the local
duration-aware logic can still choose differently if it wants to - the decision
stays here, only the decoding moves.
"""
from __future__ import annotations

import logging

from app.core import languages
from app.core.config import settings
from app.core.errors import TranslationError
from app.services import remote_gpu
from app.services.translation.base import (
    TranslationEngine,
    TranslationRequest,
    TranslationResult,
)

log = logging.getLogger(__name__)


class RemoteTranslationEngine(TranslationEngine):
    name = "remote"

    def supports(self, source_language: str, target_language: str) -> bool:
        if not settings.remote_translation_enabled or not remote_gpu.configured():
            return False
        # The endpoint serves NLLB-200 (200 languages) or SeamlessM4T; rather
        # than mirror either list here and let it rot, trust the operator who
        # pointed us at it and let a 4xx be the answer for what it cannot do.
        return bool(languages.normalize(source_language)
                    and languages.normalize(target_language))

    def unavailable_reason(self) -> str:
        if not settings.remote_translation_enabled:
            return "disabled — set REMOTE_TRANSLATION_ENABLED=true"
        return "no GPU endpoint — set REMOTE_URL (see colab/README.md)"

    def translate_batch(self, requests: list[TranslationRequest]) -> list[TranslationResult]:
        if not requests:
            return []
        if not settings.remote_translation_enabled:
            raise TranslationError("Remote translation is disabled "
                                   "(set REMOTE_TRANSLATION_ENABLED=true)")

        source = languages.normalize(requests[0].source_language) or requests[0].source_language
        target = languages.normalize(requests[0].target_language) or requests[0].target_language

        body = {
            "source_language": source,
            "target_language": target,
            "engine": settings.remote_translation_engine,
            "num_beams": settings.translation_num_beams,
            "max_new_tokens": settings.translation_max_new_tokens,
            "items": [
                {"text": r.text, "char_budget": r.char_budget} for r in requests
            ],
        }

        payload = remote_gpu.post(
            "/translate", json_body=body,
            error=TranslationError, stage="remote translation",
        )
        results = payload.get("results") or []
        if len(results) != len(requests):
            # A silent length mismatch would shift every subtitle by one line -
            # a failure that renders perfectly and is wrong throughout.
            raise TranslationError(
                f"Remote translation returned {len(results)} result(s) for "
                f"{len(requests)} request(s)",
                details={"engine": payload.get("engine")},
            )

        engine = payload.get("engine") or self.name
        return [
            TranslationResult(
                text=(item.get("text") or "").strip(),
                engine=f"{self.name}:{engine}",
                candidates=[c for c in (item.get("candidates") or []) if c],
                char_budget=request.char_budget,
            )
            for item, request in zip(results, requests, strict=True)
        ]
