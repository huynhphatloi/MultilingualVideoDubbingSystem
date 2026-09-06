from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from providers.base import ModelSpec
from providers.diarization import pyannote


def spec() -> ModelSpec:
    return ModelSpec(
        id="pyannote_3_1",
        task="diarization",
        provider="pyannote",
        display_name="pyannote speaker-diarization 3.1",
        repo_id="pyannote/speaker-diarization-3.1",
    )


def install_fake_module(monkeypatch, pipeline_class):  # noqa: ANN001, ANN201
    package = ModuleType("pyannote")
    audio = ModuleType("pyannote.audio")
    audio.Pipeline = pipeline_class
    package.audio = audio
    monkeypatch.setitem(sys.modules, "pyannote", package)
    monkeypatch.setitem(sys.modules, "pyannote.audio", audio)
    monkeypatch.setattr(pyannote, "_token", lambda: "test-token")
    monkeypatch.setattr(pyannote, "device", lambda: "cpu")


def test_pyannote_4_uses_the_token_keyword(monkeypatch):
    received = {}

    class Pipeline:
        @classmethod
        def from_pretrained(cls, checkpoint, token=None):  # noqa: ANN001, ANN206
            received.update({"checkpoint": checkpoint, "token": token})
            return object()

    install_fake_module(monkeypatch, Pipeline)
    pyannote.PyannoteProvider(spec())

    assert received == {
        "checkpoint": "pyannote/speaker-diarization-3.1",
        "token": "test-token",
    }


def test_pyannote_3_uses_the_legacy_auth_keyword(monkeypatch):
    received = {}

    class Pipeline:
        @classmethod
        def from_pretrained(  # noqa: ANN001, ANN206
            cls, checkpoint, use_auth_token=None
        ):
            received.update({
                "checkpoint": checkpoint,
                "use_auth_token": use_auth_token,
            })
            return object()

    install_fake_module(monkeypatch, Pipeline)
    pyannote.PyannoteProvider(spec())

    assert received["use_auth_token"] == "test-token"


def test_pyannote_4_reads_the_exclusive_diarization_output():
    class Turn:
        start = 1.25
        end = 2.75

    class Annotation:
        def itertracks(self, yield_label=False):  # noqa: ANN001, ANN201
            assert yield_label is True
            return [(Turn(), None, "SPEAKER_03")]

    result = SimpleNamespace(
        exclusive_speaker_diarization=Annotation(),
        speaker_diarization=SimpleNamespace(
            itertracks=lambda **_: (_ for _ in ()).throw(AssertionError("not exclusive"))
        ),
    )
    engine = object.__new__(pyannote.PyannoteProvider)
    engine.pipeline = lambda *_args, **_kwargs: result

    assert engine.diarize(Path("audio.wav")) == [
        {
            "speaker_id": "SPEAKER_03",
            "start": 1.25,
            "end": 2.75,
            "duration": 1.5,
        }
    ]
