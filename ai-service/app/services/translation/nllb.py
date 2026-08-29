"""NLLB-200 adapter (facebook/nllb-200-distilled-600M by default).

Supports 200 languages, which is why it is the primary engine: it can serve
any target the TTS router is able to speak.
"""
from __future__ import annotations

import logging

from app.core import languages
from app.core.config import settings
from app.core.device import resolve_device_for
from app.core.errors import ModelLoadError, TranslationError
from app.services.models_registry import get_or_load
from app.services.translation.base import (
    TranslationEngine,
    TranslationRequest,
    TranslationResult,
)

log = logging.getLogger(__name__)

#: length_penalty values explored when a character budget is requested. Beam
#: search with a low length penalty is the only knob that reliably yields a
#: *shorter* rendering of the same sentence, and "shorter" is the whole point
#: of duration-aware translation.
_LENGTH_PENALTIES = (0.4, 0.7, 1.0, 1.4)
#: Beams kept per length penalty. Returning every beam rather than only the
#: best one is what gives the budget something to choose between - with a
#: single return sequence, every penalty came back with byte-identical text on
#: short lines and the adapt loop could never shorten anything.
_BUDGET_BEAMS = 4
#: The length penalty that represents "just translate it normally".
_NEUTRAL_PENALTY = 1.0
#: How much a candidate's rank inside its own beam search counts against it.
#:
#: Length alone is a bad objective. Chasing the budget with no regard for the
#: model's own ranking turned "anything gets more than three blocks out" into
#: "ba thùng" (three *barrels*) simply because that beam was two characters
#: shorter. Beam 0 is the model's best answer; each step down costs
#: 1/_BUDGET_BEAMS of this weight, so a lower beam has to be *meaningfully*
#: closer to the budget - not marginally - before it wins.
_RANK_WEIGHT = 0.60
_BATCH_SIZE = 8


class NLLBEngine(TranslationEngine):
    name = "nllb"

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.nllb_model

    # ------------------------------------------------------------ loading ---
    def _bundle(self):  # noqa: ANN201
        def loader():  # noqa: ANN202
            try:
                import torch
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            except ImportError as exc:  # pragma: no cover
                raise ModelLoadError("transformers is not installed") from exc

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            device = resolve_device_for("translation")
            model = model.to(device)
            model.eval()
            return {"tokenizer": tokenizer, "model": model, "torch": torch, "device": device}

        return get_or_load(f"nllb::{self.model_name}", loader)

    # ----------------------------------------------------------- capability --
    def supports(self, source_language: str, target_language: str) -> bool:
        try:
            languages.to_flores(source_language)
            languages.to_flores(target_language)
            return True
        except Exception:
            return False

    # ---------------------------------------------------------- translating --
    def translate_batch(self, requests: list[TranslationRequest]) -> list[TranslationResult]:
        if not requests:
            return []
        bundle = self._bundle()
        tokenizer, model, torch = bundle["tokenizer"], bundle["model"], bundle["torch"]
        device = bundle["device"]

        results: list[TranslationResult | None] = [None] * len(requests)

        # Free-form requests can be batched; budgeted ones need their own search.
        plain_idx = [i for i, r in enumerate(requests) if not r.char_budget]
        budget_idx = [i for i, r in enumerate(requests) if r.char_budget]

        for start in range(0, len(plain_idx), _BATCH_SIZE):
            chunk = plain_idx[start:start + _BATCH_SIZE]
            group = [requests[i] for i in chunk]
            outputs = self._generate(
                tokenizer, model, torch, device, group,
                num_beams=settings.translation_num_beams, length_penalty=1.0,
                num_return_sequences=1,
            )
            for position, text in zip(chunk, outputs, strict=True):
                results[position] = TranslationResult(text=text, engine=self.name)

        for index in budget_idx:
            request = requests[index]
            beams = max(_BUDGET_BEAMS, settings.translation_num_beams)
            budget = max(1, request.char_budget)

            # text -> best (lowest) rank it achieved in any beam search.
            ranked: dict[str, int] = {}
            anchor = ""
            for penalty in _LENGTH_PENALTIES:
                outputs = self._generate(
                    tokenizer, model, torch, device, [request],
                    num_beams=beams, length_penalty=penalty,
                    num_return_sequences=beams,
                )
                # The neutral penalty's best beam is what the model would say if
                # nobody asked it to be brief. That is the quality anchor.
                if penalty == _NEUTRAL_PENALTY and outputs:
                    anchor = outputs[0]
                for rank, text in enumerate(outputs):
                    if not text:
                        continue
                    ranked[text] = min(ranked.get(text, rank), rank)

            if not ranked:
                raise TranslationError("NLLB returned no output",
                                       details={"text": request.text[:200]})
            anchor = anchor or min(ranked, key=lambda t: ranked[t])

            best = _choose(ranked, anchor=anchor, budget=budget, beams=beams)
            results[index] = TranslationResult(
                text=best, engine=self.name, candidates=list(ranked),
                char_budget=request.char_budget,
            )

        # One result per request, in order. Silently dropping the gaps used to
        # be possible here, and the caller zips this list against the segments
        # with strict=True - so a hole would misalign every later translation
        # (or explode) instead of failing on the segment that actually broke.
        missing = [i for i, r in enumerate(results) if r is None]
        if missing:
            raise TranslationError(
                f"NLLB produced no output for {len(missing)} segment(s)",
                details={"indexes": missing[:20],
                         "first_text": requests[missing[0]].text[:200]},
            )
        return results

    def _generate(self, tokenizer, model, torch, device,  # noqa: ANN001
                  requests: list[TranslationRequest], *, num_beams: int,
                  length_penalty: float, num_return_sequences: int) -> list[str]:
        src = languages.to_flores(requests[0].source_language)
        tgt = languages.to_flores(requests[0].target_language)
        tokenizer.src_lang = src

        encoded = tokenizer(
            [r.text for r in requests], return_tensors="pt",
            padding=True, truncation=True, max_length=512,
        ).to(device)

        bos = _target_token_id(tokenizer, tgt)
        try:
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    forced_bos_token_id=bos,
                    max_new_tokens=settings.translation_max_new_tokens,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    num_return_sequences=num_return_sequences,
                    early_stopping=True,
                )
        except Exception as exc:
            raise TranslationError(f"NLLB generation failed: {exc}",
                                   details={"source": src, "target": tgt}) from exc

        return [t.strip() for t in tokenizer.batch_decode(generated, skip_special_tokens=True)]



def _choose(ranked: dict[str, int], *, anchor: str, budget: int, beams: int) -> str:
    """Pick a rendering that fits the slot WITHOUT throwing content away.

    The budget used to be an unconditional objective, and it quietly wrecked
    short lines. Two real examples from a 37 s clip:

        "All right, listen up."  budget 11 -> "Được rồi."   (lost "listen up")
        "Smash."                 budget 22 -> "Đúng rồi."   (means "that's right")

    Both are shorter. Neither is a translation. So:

    * if the natural rendering already fits, use it and stop looking;
    * never accept a candidate that is a small fraction of the natural one -
      that is content deletion, not concision. The synchroniser can compress a
      take by ~35%, and the TTS engine can read ~45% faster, so a line that
      overruns is recoverable; a line whose meaning was deleted is not.
    """
    if len(anchor) <= budget * _BUDGET_TOLERANCE:
        return anchor

    floor = len(anchor) * settings.translation_min_keep_ratio
    viable = {t: r for t, r in ranked.items() if len(t) >= floor}
    viable.setdefault(anchor, ranked.get(anchor, 0))

    return min(viable.items(), key=lambda item: _selection_cost(
        item[0], item[1], budget=budget, beams=beams))[0]


#: A line may exceed its budget by this much before we go shopping for a
#: shorter one - the downstream stages absorb small overruns for free.
_BUDGET_TOLERANCE = 1.25


def _selection_cost(text: str, rank: int, *, budget: int, beams: int) -> tuple[float, int]:
    """Rank a candidate translation on length AND on the model's own opinion.

    ``length_cost`` is *relative*, so the trade-off means the same thing for a
    two-word interjection and for a full sentence. ``quality_cost`` grows with
    the candidate's position in its beam search, which is what stops the budget
    from picking a nonsense rendering just because it is two characters shorter.
    The final element ties on length: an over-long line has to be compressed by
    the time-stretcher, a slightly short one only leaves a natural pause.
    """
    length_cost = abs(len(text) - budget) / max(1, budget)
    quality_cost = _RANK_WEIGHT * rank / max(1, beams)
    return (length_cost + quality_cost, len(text))


def _target_token_id(tokenizer, flores_code: str) -> int:  # noqa: ANN001
    """transformers moved this API around; support both old and new."""
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is not None:
        token_id = convert(flores_code)
        if token_id is not None and token_id != getattr(tokenizer, "unk_token_id", None):
            return token_id
    lang_code_to_id = getattr(tokenizer, "lang_code_to_id", None)
    if lang_code_to_id and flores_code in lang_code_to_id:
        return lang_code_to_id[flores_code]
    raise TranslationError(
        f"NLLB tokenizer does not know target language '{flores_code}'",
        details={"flores": flores_code},
    )
