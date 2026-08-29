"""MMS-TTS adapter (facebook/mms-tts-XXX) - the multilingual safety net.

No voice cloning, but Meta ships single-speaker VITS checkpoints for over a
thousand languages, which is what makes an NLLB-200 target list realistic.
"""
from __future__ import annotations

import logging
from pathlib import Path

from app.core import languages
from app.core.config import settings
from app.core.device import resolve_device_for
from app.core.errors import ModelLoadError, SynthesisError
from app.services import ffmpeg
from app.services.models_registry import get_or_load
from app.services.tts.base import SpeechGenerator, SynthesisRequest, SynthesisResult

log = logging.getLogger(__name__)

#: Remembers languages whose checkpoint does not exist so we stop retrying.
_MISSING: set[str] = set()


class MMSAdapter(SpeechGenerator):
    name = "mms_tts"
    title = "MMS-TTS (local, one voice)"
    blurb = ("~1100 languages and always available, but ONE voice per "
             "language - every character is read by the same person. This is "
             "the safety net, not a choice you make on purpose.")
    supports_cloning = False

    def _repo(self, language: str) -> str:
        return settings.mms_tts_model_template.format(iso3=languages.to_iso3(language))

    def _model(self, language: str):  # noqa: ANN201
        repo = self._repo(language)

        def loader():  # noqa: ANN202
            try:
                import torch
                from transformers import AutoTokenizer, VitsModel
            except ImportError as exc:  # pragma: no cover
                raise ModelLoadError("transformers is not installed") from exc
            try:
                model = VitsModel.from_pretrained(repo).to(resolve_device_for("tts"))
                tokenizer = AutoTokenizer.from_pretrained(repo)
            except Exception as exc:
                _MISSING.add(repo)
                raise ModelLoadError(f"No MMS-TTS checkpoint for '{repo}': {exc}") from exc
            model.eval()
            return {"model": model, "tokenizer": tokenizer, "torch": torch}

        return get_or_load(f"mms::{repo}", loader)

    def supports(self, language: str) -> bool:
        try:
            return self._repo(language) not in _MISSING
        except Exception:
            return False

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        bundle = self._model(request.language)
        model, tokenizer, torch = bundle["model"], bundle["tokenizer"], bundle["torch"]

        out = Path(request.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # VitsModel computes `length_scale = 1.0 / speaking_rate`, so a higher
        # rate is faster (shorter) speech. It is a module attribute rather than
        # a forward() argument, which is why it is set and restored around the
        # call instead of being passed in. This is the cheapest way to make a
        # Vietnamese line fit an English slot: MMS-TTS reads noticeably slower
        # than the actors it is replacing.
        previous_rate = getattr(model, "speaking_rate", 1.0)
        try:
            model.speaking_rate = previous_rate * max(0.5, float(request.speed or 1.0))
            inputs = tokenizer(request.text, return_tensors="pt").to(resolve_device_for("tts"))
            with torch.inference_mode():
                waveform = model(**inputs).waveform
            audio = waveform.squeeze().detach().float().cpu().numpy()
        except Exception as exc:
            raise SynthesisError(f"MMS-TTS synthesis failed: {exc}") from exc
        finally:
            model.speaking_rate = previous_rate

        sample_rate = int(model.config.sampling_rate)
        _write_wav(out, audio, sample_rate)

        # Normalise to the project sample rate so downstream ffmpeg graphs
        # never have to deal with per-language rates.
        if sample_rate != settings.tts_sample_rate:
            tmp = out.with_name(out.stem + "_rs.wav")
            ffmpeg.resample(out, tmp, settings.tts_sample_rate, channels=1)
            tmp.replace(out)
            sample_rate = settings.tts_sample_rate

        return SynthesisResult(
            path=out, duration=ffmpeg.duration_of(out), model=self.name,
            voice_cloned=False, sample_rate=sample_rate,
        )


def _write_wav(path: Path, audio, sample_rate: int) -> None:  # noqa: ANN001
    try:
        import soundfile as sf

        sf.write(str(path), audio, sample_rate)
        return
    except ImportError:
        pass

    import wave

    import numpy as np

    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(pcm.tobytes())
