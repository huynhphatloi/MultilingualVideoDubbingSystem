"""Central, typed configuration. Every tunable comes from the environment."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # ---------------------------------------------------------------- app ---
    app_name: str = "Multilingual Video Dubbing - AI Service"
    app_version: str = "1.0.0"
    log_level: str = "INFO"
    log_json: bool = True

    # --------------------------------------------------------------- data ---
    database_url: str = "postgresql+psycopg://dubbing:dubbing@postgres:5432/dubbing"

    storage_backend: Literal["minio", "local"] = "minio"
    local_storage_root: str = "/data/jobs"
    scratch_root: str = "/data/jobs/_scratch"
    #: Scratch holds local copies of objects that already live in storage, so it
    #: is disposable. Without this every finished job left 60-150 MB behind for
    #: good (6 jobs = 529 MB on the dev machine). Set false while debugging a
    #: stage so the intermediate wavs stay inspectable.
    cleanup_scratch_after_render: bool = True

    minio_endpoint: str = "minio:9000"
    minio_public_endpoint: str = "localhost:47900"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin123"
    minio_bucket: str = "dubbing-jobs"
    minio_secure: bool = False
    presign_expiry_seconds: int = 24 * 3600

    # ------------------------------------------------------------- device ---
    #: auto -> cuda > mps > cpu
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    #: Per-component overrides. "auto" inherits `device`. Useful when one
    #: backend misbehaves on Metal but the rest is fine.
    asr_device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    separation_device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    diarization_device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    translation_device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    tts_device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    compute_type: str = "auto"
    torch_num_threads: int = 4

    # ---------------------------------------------------------------- asr ---
    #: Which audio Whisper reads.
    #:   "original" - the untouched mix. THE DEFAULT, and it is not a
    #:                compromise: Whisper was trained on noisy audio, while
    #:                Demucs' vocal isolation leaves phase artefacts it has
    #:                never seen. Measured on a 37 s action scene from the
    #:                sample clip: original mix -> 18 clean, punctuated lines;
    #:                vocals stem -> 11 run-on lines with no punctuation
    #:                ("eyes call it captain all right listen up until we can
    #:                close that porta"). Everything downstream translates and
    #:                speaks whatever this stage produces, so a bad transcript
    #:                is a bad dub, however good the rest of the pipeline is.
    #:   "speech"   - the Demucs vocals stem. Worth trying when the source has
    #:                extremely loud music over quiet dialogue.
    #: Diarization and the voice references keep using the vocals stem either
    #: way: there the isolation genuinely helps.
    asr_audio_source: Literal["original", "speech"] = "original"
    whisper_model: str = "medium"
    whisper_vad_filter: bool = True
    whisper_beam_size: int = 5
    whisper_condition_on_previous_text: bool = False

    # Whisper returns decoder windows, not utterances: with the VAD filter on, a
    # window routinely spans the silence it was never spoken over (a 3-word line
    # reported as 54 s long). Dubbing divides by that span, so the raw windows
    # are re-cut into utterances before anything else sees them.
    # See app/services/segmentation.py.
    #: Hard ceiling for one dubbing segment. Longer windows are split.
    asr_max_segment_seconds: float = 12.0
    #: A pause of at least this many seconds between two words ends an utterance.
    asr_split_gap_seconds: float = 0.7
    #: Fragments shorter than this are folded into their neighbour.
    asr_min_segment_seconds: float = 0.35
    #: Windows Whisper itself flags as non-speech above this probability are
    #: dropped - they are the usual source of hallucinated subtitles.
    asr_no_speech_threshold: float = 0.8

    # -------------------------------------------------------- diarization ---
    diarization_enabled: bool = True
    #: pyannote.audio 4.x community pipeline. Self-contained (segmentation +
    #: embedding + clustering), so only this ONE gated repo has to be accepted.
    #: Legacy value that still works: "pyannote/speaker-diarization-3.1".
    diarization_model: str = "pyannote/speaker-diarization-community-1"
    #: Use `output.exclusive_speaker_diarization` (one speaker per instant) when
    #: the pipeline provides it - far easier to reconcile with ASR timestamps.
    diarization_use_exclusive: bool = True
    huggingface_token: str | None = None
    max_speaker_reference_seconds: float = 12.0

    # --------------------------------------------------------- separation ---
    demucs_enabled: bool = True
    demucs_model: str = "htdemucs"
    demucs_shifts: int = 0
    #: How far the ORIGINAL mix is ducked when separation is unavailable and we
    #: fall back to voice-over. Deeper = the original dialogue is less
    #: distracting, but the music goes with it.
    #:
    #: Kept shallow ON PURPOSE. The mixer now ducks the background dynamically
    #: (MIX_DUCK_*), i.e. only while the dub is speaking. Stacking a deep static
    #: duck on top of that plus BACKGROUND_GAIN_DB buried the soundtrack
    #: completely: at -10 dB the mixed result correlated 0.02 with the source
    #: audio - the original was, for practical purposes, gone.
    voiceover_duck_db: float = -4.0
    demucs_segment: int | None = None

    # -------------------------------------------------------- translation ---
    translation_primary: Literal["nllb", "seamless"] = "nllb"
    translation_fallback: Literal["nllb", "seamless", "none"] = "seamless"
    nllb_model: str = "facebook/nllb-200-distilled-600M"
    seamless_model: str = "facebook/hf-seamless-m4t-medium"
    translation_max_new_tokens: int = 256
    translation_num_beams: int = 4
    #: Floor on how much shorter a duration-aware rendering may be than the
    #: natural one, as a fraction of its length. Below this the model is not
    #: being concise, it is dropping clauses ("All right, listen up." came back
    #: as "Được rồi."). Timing problems are recoverable downstream; deleted
    #: meaning is not.
    translation_min_keep_ratio: float = 0.75

    # ---------------------------------------------------------------- tts ---
    tts_prefer_voice_clone: bool = True
    xtts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    mms_tts_model_template: str = "facebook/mms-tts-{iso3}"
    f5_tts_enabled: bool = False
    tts_sample_rate: int = 24000

    # ------------------------------------------------------- remote GPU ---
    #: ONE Colab notebook serves every heavy stage, so one base URL and one
    #: token drive all of them. Empty disables every remote stage, which is why
    #: it is safe to leave them wired in.
    #:
    #: The point is disk, not just speed: whisper-medium is 1.4 GB and
    #: NLLB-200-distilled 2.3 GB. Running them remotely means this machine
    #: caches no weights at all.
    remote_url: str = ""
    remote_token: str = ""
    #: Shared retry/timeout budget. A Colab tunnel dies on session timeout, so
    #: a dropped connection is the normal case rather than the exceptional one.
    remote_timeout: float = 600.0
    remote_retries: int = 3

    #: Per-stage switches. ASR is opt-in ON ITS OWN because it is the one that
    #: uploads audio rather than text - see app/services/remote_gpu.py. Anyone
    #: dubbing material they cannot share should leave it off and accept the
    #: local Whisper download.
    remote_asr_enabled: bool = False
    remote_translation_enabled: bool = False
    #: Opus bitrate for the speech track sent to remote ASR. 24 kbps mono is
    #: ~1.8 MB for ten minutes of film against 19 MB of wav, and Whisper's
    #: accuracy is unaffected at this rate for speech.
    remote_asr_opus_kbps: int = 24
    #: Which model the remote side loads for translation. Same names as the
    #: local engines, because it is the same two models - just not on this disk.
    remote_translation_engine: str = "nllb"

    #: Synthesis endpoint. Empty falls back to remote_url - they are normally
    #: the same notebook, and splitting them is only useful when synthesis is
    #: pinned to a bigger GPU than the rest.
    remote_tts_url: str = ""
    remote_tts_token: str = ""
    #: Comma-separated engine ids the endpoint serves ("vixtts,f5_vi"). Each one
    #: becomes its own adapter named `remote:<id>`, so comparing two engines on
    #: a real job is `force_model="remote:f5_vi"` rather than a redeploy. Empty
    #: registers a single unnamed `remote` adapter and lets the server choose,
    #: which is the right setting when the endpoint only hosts one model.
    remote_tts_engines: str = ""
    #: Comma-separated ISO-639-1 codes the endpoint can speak. Empty means
    #: "trust the endpoint" and let it answer 4xx for what it cannot do.
    remote_tts_languages: str = ""
    #: Whether the remote engine clones. Drives router preference AND the
    #: per-speaker pitch shift, which must stay off when voices really differ.
    remote_tts_voice_clone: bool = True
    #: A cold Colab wakes the model on the first call; later calls are seconds.
    remote_tts_timeout: float = 300.0
    remote_tts_retries: int = 3

    #: TTS engines wrap every take in silence (MMS-TTS: ~0.41 s before, ~0.28 s
    #: after). Counted as speech, that padding alone can exceed the slot the
    #: line has to fit into. Trimming it is the single cheapest win in the
    #: whole synthesis stage.
    tts_trim_silence: bool = True
    tts_trim_threshold_db: float = -45.0
    #: Silence deliberately left at each end so consonants are not clipped.
    tts_trim_keep_ms: int = 40
    #: When a take still overruns its slot, re-read it faster before resorting
    #: to a post-hoc time-stretch (which degrades the voice) or a re-translation
    #: (which costs another model round-trip).
    tts_speed_adaptation: bool = True
    #: Past ~1.45x even a synthetic voice sounds panicked.
    tts_max_speaking_rate: float = 1.45

    #: Tell speakers apart when the engine cannot clone. MMS-TTS ships exactly
    #: ONE voice per language, so every character in a Vietnamese dub is read by
    #: the same person - a two-hander becomes someone arguing with themselves,
    #: which is what "trộn giọng" describes. Shifting each speaker's pitch by a
    #: couple of semitones restores the ability to follow who is talking.
    #: Ignored when the engine clones (XTTS already gives each speaker a voice).
    tts_speaker_pitch_enabled: bool = True
    #: Semitone offset per speaker, in order of screen time. Kept small: past
    #: about 4 semitones a shifted voice stops sounding like a person.
    tts_speaker_pitch_steps: str = "0,-2.5,2.5,-4,4,-1.5,1.5"

    @property
    def tts_endpoint(self) -> str:
        """Where synthesis goes: its own URL, or the shared notebook."""
        return (self.remote_tts_url or self.remote_url or "").rstrip("/")

    @property
    def tts_endpoint_token(self) -> str:
        return self.remote_tts_token or self.remote_token or ""

    @property
    def remote_tts_engine_list(self) -> list[str]:
        seen: list[str] = []
        for chunk in (self.remote_tts_engines or "").split(","):
            chunk = chunk.strip()
            if chunk and chunk not in seen:
                seen.append(chunk)
        return seen

    @property
    def speaker_pitch_steps(self) -> list[float]:
        out: list[float] = []
        for chunk in (self.tts_speaker_pitch_steps or "").split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                out.append(float(chunk))
            except ValueError:
                continue
        return out or [0.0]

    # --------------------------------------------------------------- sync ---
    sync_ratio_accept_min: float = 0.90
    sync_ratio_accept_max: float = 1.10
    sync_ratio_stretch_min: float = 0.80
    sync_ratio_stretch_max: float = 1.20
    sync_max_retranslate_attempts: int = 2
    sync_sample_rate: int = 48000

    # ------------------------------------------------------------- mixing ---
    background_gain_db: float = -6.0
    speech_gain_db: float = 0.0
    #: TTS output level varies wildly between engines and even between lines.
    #: Bring the whole dubbed track to a known peak before mixing, otherwise the
    #: background decides how audible the dub is.
    speech_auto_gain: bool = True
    speech_target_peak_dbfs: float = -3.0
    #: Max correction auto-gain may apply, so a near-silent take cannot be
    #: amplified into noise.
    speech_auto_gain_limit_db: float = 12.0
    #: Duck the background *while the dub speaks* (sidechain compression) rather
    #: than turning the whole soundtrack down. This is what makes the voice
    #: intelligible over an action-scene score without losing the score.
    mix_duck_enabled: bool = True
    mix_duck_ratio: float = 8.0
    mix_duck_threshold: float = 0.03
    mix_duck_attack_ms: float = 20.0
    mix_duck_release_ms: float = 400.0
    #: Wet/dry blend of the side-chain compressor. Below 1.0 it bounds how far
    #: the background can ever be pushed down, so a dense scene cannot silence
    #: the soundtrack outright.
    mix_duck_mix: float = 0.85
    loudnorm_enabled: bool = True
    loudnorm_i: float = -16.0
    loudnorm_tp: float = -1.5
    loudnorm_lra: float = 11.0

    # ------------------------------------------------------------ runtime ---
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"

    @field_validator("huggingface_token", mode="before")
    @classmethod
    def _blank_to_none(cls, v):  # noqa: ANN001
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @property
    def has_hf_token(self) -> bool:
        return bool(self.huggingface_token)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
