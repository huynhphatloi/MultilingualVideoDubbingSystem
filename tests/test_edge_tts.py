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


def test_single_voice_mode_keeps_the_existing_female_default(monkeypatch):
    engine = provider(monkeypatch)
    assert engine.voice_for("vi", "SPEAKER_01", multi_voice=False) == (
        "vi-VN-HoaiMyNeural"
    )


def test_multi_voice_assigns_a_stable_voice_to_each_pyannote_speaker(monkeypatch):
    engine = provider(monkeypatch)
    first = engine.voice_for("vi", "SPEAKER_00", multi_voice=True)
    second = engine.voice_for("vi", "SPEAKER_01", multi_voice=True)

    assert first != second
    assert engine.voice_for("vi", "SPEAKER_00", multi_voice=True) == first
    assert engine.voice_for("vi", "SPEAKER_01", multi_voice=True) == second


def test_multi_voice_reuses_voices_deterministically_when_speakers_outnumber_them(
    monkeypatch,
):
    engine = provider(monkeypatch)
    assert engine.voice_for("vi", "SPEAKER_02", multi_voice=True) == engine.voice_for(
        "vi", "SPEAKER_00", multi_voice=True
    )
