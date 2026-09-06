from __future__ import annotations

import sys
from types import SimpleNamespace

from providers.base import ModelSpec
from providers.tts.edge import EdgeProvider


VOICES = [
    {"ShortName": "vi-VN-NamMinhNeural", "Gender": "Male"},
    {"ShortName": "en-US-JennyNeural", "Gender": "Female"},
    {"ShortName": "vi-VN-HoaiMyNeural", "Gender": "Female"},
]


def provider(monkeypatch) -> EdgeProvider:  # noqa: ANN001
    async def list_voices():  # noqa: ANN202
        return list(VOICES)

    monkeypatch.setitem(sys.modules, "edge_tts", SimpleNamespace(list_voices=list_voices))
    return EdgeProvider(ModelSpec(
        id="edge", task="tts", provider="edge", display_name="Edge TTS"
    ))


def test_no_speaker_uses_the_existing_female_default(monkeypatch):
    engine = provider(monkeypatch)
    assert engine.voice_for("vi") == "vi-VN-HoaiMyNeural"


def test_each_pyannote_speaker_gets_a_stable_voice(monkeypatch):
    engine = provider(monkeypatch)
    first = engine.voice_for("vi", "SPEAKER_00")
    second = engine.voice_for("vi", "SPEAKER_01")

    assert first != second
    assert engine.voice_for("vi", "SPEAKER_00") == first
    assert engine.voice_for("vi", "SPEAKER_01") == second


def test_speakers_reuse_voices_deterministically_when_the_pool_is_exhausted(
    monkeypatch,
):
    engine = provider(monkeypatch)
    assert engine.voice_for("vi", "SPEAKER_02") == engine.voice_for("vi", "SPEAKER_00")
