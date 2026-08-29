"""Unit tests for the pure logic of the pipeline.

These deliberately need no models, no ffmpeg and no database, so they run in
about a second:

    docker compose exec ai-service python -m pytest tests -q
    # or locally:  pip install pytest pydantic pydantic-settings && pytest ai-service/tests -q
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from app.core import device, languages
from app.core.errors import NoSpeechDetected, TranslationError
from app.services import alignment, diarization, ffmpeg, subtitles
from app.services.translation import duration_aware
from app.storage.layout import JobLayout


# ------------------------------------------------------------- languages ----
@pytest.mark.parametrize(("raw", "expected"), [
    ("EN", "en"), ("zh-CN", "zh"), ("vie", "vi"), ("vie_Latn", "vi"),
    ("en-US", "en"), ("pt-BR", "pt"), ("klingon", None),
])
def test_normalize(raw, expected):
    assert languages.normalize(raw) == expected


def test_model_specific_codes():
    assert languages.to_flores("vi") == "vie_Latn"
    assert languages.to_flores("eng_Latn") == "eng_Latn"   # FLORES codes pass through
    assert languages.to_iso3("ja") == "jpn"
    assert languages.to_xtts("zh") == "zh-cn"


def test_voice_cloning_support_drives_the_router():
    assert languages.supports_voice_cloning("en")
    # Vietnamese has no XTTS checkpoint -> the router must fall back to MMS-TTS
    assert not languages.supports_voice_cloning("vi")


# --------------------------------------------------- duration awareness -----
def test_cjk_budget_is_tighter_than_latin():
    assert duration_aware.estimate_budget(2.5, "ja") < duration_aware.estimate_budget(2.5, "en")


def test_fast_speaker_gets_a_bigger_budget():
    slow = duration_aware.estimate_budget(
        2.5, "vi", source_text="Hello there.", source_language="en")
    fast = duration_aware.estimate_budget(2.5, "vi", source_text="H" * 60, source_language="en")
    assert fast > slow


@pytest.mark.parametrize(("ratio", "action"), [
    (1.00, "accept"), (0.95, "accept"), (1.10, "accept"),
    (1.15, "stretch"), (0.85, "stretch"),
    (1.40, "retranslate"), (0.60, "retranslate"),
])
def test_classify(ratio, action):
    assert duration_aware.classify(ratio, 0.90, 1.10, 0.80, 1.20) == action


def test_empirical_budget_correction():
    text = "x" * 100
    assert duration_aware.budget_from_ratio(text, 1.24) < 100   # take was long -> ask for less
    assert duration_aware.budget_from_ratio(text, 0.70) > 100   # take was short -> ask for more
    assert duration_aware.budget_from_ratio(text, 0.0) == 100   # degenerate input is safe


# --------------------------------------------------------------- ffmpeg -----
def test_atempo_chain_stays_within_ffmpeg_limits():
    assert ffmpeg.atempo_chain(1.2) == "atempo=1.200000"
    for factor in (3.0, 0.3, 4.0, 0.25):
        chain = ffmpeg.atempo_chain(factor)
        values = [float(part.split("=")[1]) for part in chain.split(",")]
        assert all(0.5 <= v <= 2.0 for v in values)
        product = 1.0
        for v in values:
            product *= v
        assert product == pytest.approx(factor, rel=1e-6)


# ------------------------------------------------------------ subtitles -----
def test_cue_overlap_is_repaired():
    cues = subtitles._cues([
        {"start": 0.0, "end": 2.5, "translated_text": "Xin chao."},
        {"start": 2.4, "end": 5.0, "translated_text": "He thong long tieng."},
    ], "translated_text", 42)
    assert cues[0]["end"] < cues[1]["start"]


def test_render_formats():
    cues = subtitles._cues([{"start": 0.0, "end": 2.0, "translated_text": "hi"}],
                   "translated_text", 42)
    assert subtitles._render(cues, "srt").startswith("1\n00:00:00,000 --> ")
    assert subtitles._render(cues, "vtt").startswith("WEBVTT")
    assert subtitles._ts(3661.5, ",") == "01:01:01,500"


def test_long_lines_are_wrapped():
    cues = subtitles._cues([{"start": 0, "end": 3, "translated_text": "word " * 40}],
                   "translated_text", 42)
    for line in cues[0]["text"].split("\n"):
        assert len(line) <= 42
    assert cues[0]["text"].count("\n") <= 2


# ------------------------------------------------------------ alignment -----
TURNS = [
    {"start": 0.0, "end": 3.0, "speaker_id": "SPEAKER_00"},
    {"start": 3.0, "end": 6.0, "speaker_id": "SPEAKER_01"},
]


def test_segment_is_split_on_a_real_speaker_change():
    seg = {
        "segment_id": 0, "start": 2.0, "end": 4.0,
        "source_text": "hello there how are you",
        "words": [
            {"word": "hello", "start": 2.0, "end": 2.4},
            {"word": " there", "start": 2.4, "end": 2.9},
            {"word": " how", "start": 3.1, "end": 3.4},
            {"word": " are", "start": 3.4, "end": 3.7},
            {"word": " you", "start": 3.7, "end": 4.0},
        ],
    }
    chunks = alignment._assign_speakers(seg, TURNS)
    assert [c["speaker_id"] for c in chunks] == ["SPEAKER_00", "SPEAKER_01"]
    assert chunks[0]["source_text"] == "hello there"


def test_single_stray_word_does_not_split_a_segment():
    seg = {
        "segment_id": 0, "start": 0.2, "end": 3.1,
        "source_text": "one two three four five six seven eight nine ten",
        "words": [{"word": f" w{i}", "start": 0.2 + i * 0.28, "end": 0.4 + i * 0.28}
                  for i in range(10)],
    }
    assert len(alignment._assign_speakers(seg, TURNS)) == 1


def test_no_word_timestamps_falls_back_to_dominant_overlap():
    seg = {"segment_id": 1, "start": 3.2, "end": 5.0, "source_text": "hi"}
    chunks = alignment._assign_speakers(seg, TURNS)
    assert len(chunks) == 1 and chunks[0]["speaker_id"] == "SPEAKER_01"


def test_no_diarization_at_all_is_still_usable():
    seg = {"segment_id": 1, "start": 0.0, "end": 1.0, "source_text": "hi"}
    assert alignment._assign_speakers(seg, [])[0]["speaker_id"] == "SPEAKER_00"


# --------------------------------------------------------------- layout -----
def test_object_key_layout_is_stable():
    lay = JobLayout("abc123")
    assert lay.source_video("mp4") == "jobs/abc123/source/input.mp4"
    assert lay.original_audio == "jobs/abc123/audio/original.wav"
    assert lay.generated_segment(7, 1) == "jobs/abc123/generated/segment_0007_v1.wav"
    assert lay.speaker_reference("SPEAKER_01") == "jobs/abc123/speakers/speaker_01.wav"
    assert lay.subtitle("vi", "vtt") == "jobs/abc123/subtitles/vi.vtt"
    assert lay.output_video("vi") == "jobs/abc123/output/dubbed_vi.mp4"


# --------------------------------------------------------------- errors -----
def test_error_payloads_are_branchable_by_n8n():
    payload = NoSpeechDetected("nothing to dub").to_payload()
    assert payload["error"]["code"] == "no_speech_detected"
    assert payload["error"]["retryable"] is False
    assert TranslationError("boom").to_payload()["error"]["retryable"] is True


# ------------------------------------------------- diarization output shapes -
class _Seg:
    """Stand-in for pyannote.core.Segment."""

    def __init__(self, start, end):
        self.start, self.end = start, end


class _LegacyAnnotation:
    """pyannote 3.x: itertracks(yield_label=True) -> (segment, track, label)."""

    def __init__(self, items):
        self._items = items

    def itertracks(self, yield_label=False):
        for seg, label in self._items:
            yield (seg, "_", label) if yield_label else (seg, "_")


class _PairAnnotation:
    """pyannote 4.x style: iterating yields (segment, label)."""

    def __init__(self, items):
        self._items = items

    def __iter__(self):
        return iter(self._items)


class _CommunityOutput:
    """community-1 returns BOTH annotations on one object."""

    def __init__(self, regular, exclusive):
        self.speaker_diarization = regular
        self.exclusive_speaker_diarization = exclusive


ITEMS = [(_Seg(1.0, 3.0), "SPEAKER_00"), (_Seg(3.0, 6.0), "SPEAKER_01")]


def test_turns_parsed_from_legacy_annotation():
    turns = diarization._turns_of(_LegacyAnnotation(ITEMS))
    assert [t["speaker_id"] for t in turns] == ["SPEAKER_00", "SPEAKER_01"]
    assert turns[0] == {"start": 1.0, "end": 3.0, "duration": 2.0, "speaker_id": "SPEAKER_00"}


def test_turns_parsed_from_pair_iteration():
    assert (diarization._turns_of(_PairAnnotation(ITEMS))
            == diarization._turns_of(_LegacyAnnotation(ITEMS)))


def test_turns_are_sorted_and_noise_is_dropped():
    noisy = [(_Seg(5.0, 5.02), "SPEAKER_02")] + list(reversed(ITEMS))
    turns = diarization._turns_of(_PairAnnotation(noisy))
    assert [t["start"] for t in turns] == [1.0, 3.0]     # 20 ms turn dropped, order fixed


def test_missing_annotation_is_not_an_error():
    assert diarization._turns_of(None) == []


def test_community_output_exposes_both_views():
    out = _CommunityOutput(_PairAnnotation(ITEMS), _PairAnnotation(ITEMS[:1]))
    assert len(diarization._turns_of(getattr(out, "speaker_diarization", out))) == 2
    assert len(diarization._turns_of(getattr(out, "exclusive_speaker_diarization", None))) == 1


# ------------------------------------------------------- turn source policy --
def test_exclusive_turns_win_when_available():
    doc = {"turns": [{"start": 0, "end": 1, "speaker_id": "A"}],
           "exclusive_turns": [{"start": 0, "end": 2, "speaker_id": "B"}]}
    turns, source = alignment._pick_turns(doc)
    assert source == "exclusive" and turns[0]["speaker_id"] == "B"


def test_falls_back_to_regular_turns_for_legacy_models():
    doc = {"turns": [{"start": 0, "end": 1, "speaker_id": "A"}], "exclusive_turns": []}
    turns, source = alignment._pick_turns(doc)
    assert source == "regular" and turns[0]["speaker_id"] == "A"


def test_no_turns_at_all_is_reported_not_crashed():
    assert alignment._pick_turns({}) == ([], "none")


# ------------------------------------------------------- device resolution --
@pytest.fixture
def host(monkeypatch):
    """Pretend to be a machine with a given set of accelerators."""

    def _host(cuda=False, mps=False, **overrides):
        monkeypatch.setattr(device, "_probe", lambda: {"cuda": cuda, "mps": mps})
        settings = device.settings
        for name in ("device", "asr_device", "separation_device", "diarization_device",
                     "translation_device", "tts_device"):
            monkeypatch.setattr(settings, name, overrides.get(name, "auto"), raising=False)
        monkeypatch.setattr(settings, "compute_type", overrides.get("compute_type", "auto"),
                            raising=False)
        resolved = "cuda" if cuda else ("mps" if mps else "cpu")
        if overrides.get("device", "auto") not in ("auto", ""):
            wanted = overrides["device"]
            resolved = wanted if {"cuda": cuda, "mps": mps}.get(wanted, wanted == "cpu") else "cpu"
        monkeypatch.setattr(device, "resolve_device", lambda: resolved)
        return resolved

    return _host


ALL_COMPONENTS = ("asr", "separation", "diarization", "translation", "tts")


def test_apple_silicon_uses_metal_except_for_asr(host):
    host(mps=True)
    # CTranslate2 (faster-whisper) has no Metal backend - it must land on cpu
    # instead of raising at model-load time.
    assert device.resolve_device_for("asr") == "cpu"
    for component in ("separation", "diarization", "translation", "tts"):
        assert device.resolve_device_for(component) == "mps"


def test_cuda_host_uses_cuda_everywhere(host):
    host(cuda=True)
    assert all(device.resolve_device_for(c) == "cuda" for c in ALL_COMPONENTS)


def test_plain_cpu_host(host):
    host()
    assert all(device.resolve_device_for(c) == "cpu" for c in ALL_COMPONENTS)


def test_per_component_override_wins(host):
    host(mps=True, tts_device="cpu")
    assert device.resolve_device_for("tts") == "cpu"
    assert device.resolve_device_for("separation") == "mps"


def test_requesting_an_absent_device_degrades_instead_of_crashing(host):
    host(mps=True, tts_device="cuda")     # no NVIDIA GPU on this host
    assert device.resolve_device_for("tts") == "cpu"


def test_compute_type_follows_the_asr_device(host):
    host(cuda=True)
    device.resolve_compute_type.cache_clear()
    assert device.resolve_compute_type() == "float16"
    host(mps=True)
    device.resolve_compute_type.cache_clear()
    assert device.resolve_compute_type() == "int8"   # asr fell back to cpu
    device.resolve_compute_type.cache_clear()


def test_unknown_component_is_a_programming_error(host):
    host()
    with pytest.raises(ValueError):
        device.resolve_device_for("nope")


def test_mps_fallback_flag_is_set(monkeypatch):
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    device.configure_threads()
    import os
    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"


# --------------------------------------------------------- model cache -------
def test_unwritable_cache_falls_back_instead_of_dying_mid_pipeline(monkeypatch, tmp_path):
    """`/models` exists in the image but not on a Mac - the read-only root there
    made torch.hub blow up inside whichever stage downloaded a model first."""
    from app.core import paths

    monkeypatch.setenv("HF_HOME", "/proc/definitely-not-writable/hf")
    monkeypatch.setenv("TORCH_HOME", "/proc/definitely-not-writable/torch")
    monkeypatch.setenv("XDG_CACHE_HOME", "/proc/definitely-not-writable/cache")
    monkeypatch.setattr(paths, "FALLBACK_ROOT", tmp_path / "fallback")

    resolved = paths.ensure_model_cache()

    for var, path in resolved.items():
        assert str(tmp_path) in path, f"{var} did not fall back"
        assert Path(path).is_dir()
        assert os.environ[var] == path


def test_relative_cache_paths_are_made_absolute(monkeypatch, tmp_path):
    from app.core import paths

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_HOME", "./models/huggingface")
    monkeypatch.setenv("TORCH_HOME", "./models/torch")
    monkeypatch.setenv("XDG_CACHE_HOME", "./models/cache")

    resolved = paths.ensure_model_cache()

    for path in resolved.values():
        assert Path(path).is_absolute()
        assert Path(path).is_dir()


def test_cache_report_flags_a_bad_path(monkeypatch, tmp_path):
    from app.core import paths

    good = tmp_path / "good"
    good.mkdir()
    monkeypatch.setenv("HF_HOME", str(good))
    monkeypatch.setenv("TORCH_HOME", "/proc/nope")
    monkeypatch.setenv("XDG_CACHE_HOME", str(good))

    report = paths.cache_report()
    assert report["HF_HOME"]["writable"] is True
    assert report["TORCH_HOME"]["writable"] is False
