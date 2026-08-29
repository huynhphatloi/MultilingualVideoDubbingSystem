#!/usr/bin/env python3
"""End-to-end verification of the pipeline on a REAL video, without the models.

What is real here
-----------------
Everything this project actually wrote: ffmpeg audio extraction, the segment
document, speaker/voice-reference handling, the duration-ratio decision table,
the atempo time-stretch, the "audio canvas" overlay at original timestamps, the
background mix, subtitle timing, and the final mux. The input is a real video
and the output is a real playable file.

What is faked
-------------
Only the three model calls (ASR, translation, TTS) - the parts that are
third-party weights rather than our logic:

* ASR       -> segments derived from ffmpeg `silencedetect` on the REAL audio,
               so the timings are the clip's actual speech/pause structure.
* Translate -> deterministically expands/shrinks text, on purpose, to drive
               every branch of the sync decision table.
* TTS       -> a real wav of a precisely controlled duration, so the ratio
               logic gets exercised with known-good numbers.

**The output of this script is NOT a dub.** Its "voice" is a 180 Hz sine tone
and its "translation" is English with Vietnamese filler words appended. It
proves the ffmpeg/timing/mixing code is correct and nothing else. For a real
dub, run the models: ``./scripts/smoke-test.sh data/samples/<clip>.mp4 vi``.

Run it:

    .venv/bin/python scripts/verify_pipeline.py data/samples/avengers_60s.mp4

Exit code is non-zero if any assertion about the output fails.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ai-service"))

SOURCE = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data/samples/avengers_60s.mp4")
WORK = Path(tempfile.mkdtemp(prefix="mvds-verify-"))

# Local storage + SQLite: no MinIO, no Postgres, no network.
os.environ.update({
    "STORAGE_BACKEND": "local",
    "LOCAL_STORAGE_ROOT": str(WORK / "objects"),
    "SCRATCH_ROOT": str(WORK / "scratch"),
    "DEMUCS_ENABLED": "false",          # passthrough; Demucs is third-party
    "DIARIZATION_ENABLED": "false",     # exercised via the fallback path
    "HF_HOME": str(WORK / "cache/hf"),
    "TORCH_HOME": str(WORK / "cache/torch"),
    "XDG_CACHE_HOME": str(WORK / "cache/xdg"),
    # keep the intermediate wavs so the assertions below can measure them
    "CLEANUP_SCRATCH_AFTER_RENDER": "false",
    "LOG_JSON": "false",
    "LOG_LEVEL": "WARNING",
})

from sqlalchemy import create_engine  # noqa: E402

from app.core.paths import ensure_model_cache  # noqa: E402

ensure_model_cache()

from app.jobs import db as jobs_db  # noqa: E402
from app.jobs import service as jobs  # noqa: E402
from app.services import alignment, ffmpeg, media, mixing, render, separation  # noqa: E402
from app.services import subtitles as subs  # noqa: E402
from app.services import sync  # noqa: E402
from app.services.translation import duration_aware  # noqa: E402
from app.services.workspace import JobWorkspace  # noqa: E402

PASS, FAIL = "\033[1;32m  ok  \033[0m", "\033[1;31m FAIL \033[0m"
failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"{PASS if condition else FAIL}{label}{(' — ' + detail) if detail else ''}")
    if not condition:
        failures.append(label)


def step(title: str) -> None:
    print(f"\n\033[1;36m== {title}\033[0m")


# ---------------------------------------------------------------- fake ASR ---
def speech_segments_from_audio(wav: Path, total: float) -> list[dict]:
    """Derive real speech spans from the clip using ffmpeg silencedetect."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(wav),
         "-af", "silencedetect=noise=-30dB:d=0.35", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    starts = [float(m) for m in re.findall(r"silence_start: ([\d.]+)", proc.stderr)]
    ends = [float(m) for m in re.findall(r"silence_end: ([\d.]+)", proc.stderr)]

    spans, cursor = [], 0.0
    for s, e in zip(starts, ends + [total] * (len(starts) - len(ends))):
        if s - cursor > 0.8:
            spans.append((cursor, s))
        cursor = e
    if total - cursor > 0.8:
        spans.append((cursor, total))

    # Long uninterrupted spans are not realistic ASR output; split them the way
    # Whisper would, into utterance-sized chunks.
    chunks: list[tuple[float, float]] = []
    for start, end in spans:
        length = end - start
        pieces = max(1, int(math.ceil(length / 5.0)))
        width = length / pieces
        chunks.extend((start + i * width, start + (i + 1) * width) for i in range(pieces))

    words = ("we need to solve this problem right now before it gets worse "
             "and everyone here has to agree on the plan").split()
    out = []
    for i, (start, end) in enumerate(chunks):
        n = max(3, int((end - start) * 2.6))
        text = " ".join(words[(i * 3) % len(words):][:n]) or "we need to solve this"
        out.append({
            "segment_id": i,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
            "source_text": text.capitalize() + ".",
        })
    return out


# -------------------------------------------------------------- fake models --
def fake_translate(text: str, factor: float) -> str:
    """Deterministically stretch/shrink so every sync branch gets hit."""
    words = text.split()
    target = max(2, int(round(len(words) * factor)))
    if target <= len(words):
        return " ".join(words[:target])
    filler = ["thêm", "nữa", "rồi", "đấy", "nhé"]
    return " ".join(words + [filler[i % len(filler)] for i in range(target - len(words))])


def fake_tts(path: Path, duration: float) -> None:
    """A real wav of an exact duration - that is all the sync code cares about."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", f"sine=frequency=180:sample_rate=24000:duration={max(0.15, duration):.3f}",
         "-af", "tremolo=f=5:d=0.7", "-acodec", "pcm_s16le", str(path)],
        check=True,
    )


# ===================================================================== run ====
def main() -> int:
    if not SOURCE.exists():
        print(f"No such file: {SOURCE}")
        return 2

    print(f"source : {SOURCE}")
    print(f"workdir: {WORK}")

    engine = create_engine(f"sqlite:///{WORK/'jobs.db'}", future=True)
    jobs_db.Base.metadata.create_all(engine)
    jobs_db.configure(engine)

    # ---------------------------------------------------------- 1. job ------
    step("1 · create job + store video")
    with jobs_db.session_scope() as session:
        job = jobs.create_job(session, source_filename=SOURCE.name,
                              target_language="vi", options={"origin": "verify"})
        job_id = job.id
    ws = JobWorkspace(job_id)
    video_key = ws.push(SOURCE, ws.layout.source_video("mp4"))
    info = ffmpeg.media_info(SOURCE)
    jobs.update_job(job_id, video_key=video_key, duration_seconds=info["duration"])
    check("video registered", ws.storage.exists(video_key),
          f"{info['duration']:.1f}s {info['video']['width']}x{info['video']['height']}")

    # ------------------------------------------------------ 2. extract ------
    step("2 · extract audio (real ffmpeg)")
    with jobs.track(job_id, "extract_audio"):
        res = media.extract_audio(job_id, video_key)
    total = res["duration_seconds"]
    check("audio extracted", ws.storage.exists(res["audio_key"]),
          f"{total:.1f}s, peak {res['peak_dbfs']} dBFS")
    check("duration matches the video", abs(total - info["duration"]) < 0.5)

    # ----------------------------------------------------- 3. separate ------
    step("3 · separate (passthrough - Demucs is third-party)")
    with jobs.track(job_id, "separate_sources"):
        sep = separation.separate(job_id, res["audio_key"])
    check("speech + background produced",
          ws.storage.exists(sep["speech_key"]) and ws.storage.exists(sep["background_key"]))
    check("fallback mode is reported honestly", sep.get("mode") == "voiceover",
          f"mode={sep.get('mode')} · original dialogue still present="
          f"{sep.get('original_dialogue_present')}")
    bg = ws.pull(sep["background_key"])
    check("background spans the whole clip", abs(ffmpeg.duration_of(bg) - total) < 1.0,
          f"{ffmpeg.duration_of(bg):.1f}s vs {total:.1f}s")

    # ----------------------------------------------------- 4. ASR (fake) ----
    step("4 · transcript from the clip's real speech/pause structure")
    speech = ws.pull(sep["speech_key"])
    segments = speech_segments_from_audio(speech, total)
    ws.push_json({"job_id": job_id, "language": "en", "language_confidence": 0.97,
                  "model": "fake", "duration": total, "segments": segments},
                 ws.layout.transcript)
    check("segments found", len(segments) >= 5, f"{len(segments)} segments")
    check("timestamps inside the clip",
          all(0 <= s["start"] < s["end"] <= total + 0.1 for s in segments))
    check("segments do not overlap",
          all(a["end"] <= b["start"] + 1e-6 for a, b in zip(segments, segments[1:])))

    # -------------------------------------------------- 5. diarize/merge ----
    step("5 · diarization fallback + merge (real code)")
    from app.services import diarization
    with jobs.track(job_id, "diarize"):
        dia = diarization.diarize(job_id, sep["speech_key"])
    check("degrades to one speaker without a token", dia["speaker_count"] == 1,
          f"enabled={dia['enabled']}")

    # Pretend the model found two speakers so the multi-speaker path is covered.
    turns, flip = [], 0.0
    while flip < total:
        turns.append({"start": round(flip, 3), "end": round(min(flip + 7, total), 3),
                      "duration": 7.0,
                      "speaker_id": f"SPEAKER_0{len(turns) % 2}"})
        flip += 7
    ws.push_json({"job_id": job_id, "model": "fake", "speakers": ["SPEAKER_00", "SPEAKER_01"],
                  "speaker_count": 2, "turns": turns, "exclusive_turns": turns,
                  "has_exclusive": True, "enabled": True}, ws.layout.diarization)

    with jobs.track(job_id, "merge_segments"):
        merged = alignment.merge(job_id, ws.layout.transcript, ws.layout.diarization)
    check("merge used exclusive turns", merged["turn_source"] == "exclusive")
    check("two speakers carried through", merged["speaker_count"] == 2)
    doc = ws.storage.get_json(merged["segments_key"])
    refs = [s["voice_reference"] for s in doc["speakers"]]
    check("a voice reference was cut per speaker", all(refs) and len(refs) == 2)
    for key in refs:
        ref = ws.pull(key)
        check(f"  {Path(key).name} is real audio", ffmpeg.duration_of(ref) > 1.0,
              f"{ffmpeg.duration_of(ref):.1f}s")

    # -------------------------------------------- 6. translate + 7. TTS -----
    step("6 · translation with a deliberately awkward length + 7 · synthesis")
    # factor per segment: on target, far too long, slightly long, far too short
    factors = [1.0, 1.6, 1.15, 0.5]
    for i, seg in enumerate(doc["segments"]):
        seg["translated_text"] = fake_translate(seg["source_text"], factors[i % 4])
        seg["target_language"] = "vi"
        seg["char_budget"] = duration_aware.estimate_budget(
            seg["end"] - seg["start"], "vi", seg["source_text"], "en")
    doc["target_language"] = "vi"
    doc["source_language"] = "en"
    ws.push_json(doc, ws.layout.segments)

    generated = ws.subdir("generated")
    actions: dict[str, int] = {}
    for i, seg in enumerate(doc["segments"]):
        slot = seg["end"] - seg["start"]
        ratio_target = factors[i % 4]
        wav = generated / f"segment_{seg['segment_id']:04d}.wav"
        fake_tts(wav, slot * ratio_target)
        real = ffmpeg.duration_of(wav)
        seg["generated_audio"] = ws.push(
            wav, ws.layout.generated_segment(seg["segment_id"]))
        seg["generated_duration"] = round(real, 3)
        seg["duration_ratio"] = round(real / slot, 4)
        seg["tts_model"] = "fake"
        seg["sync_action"] = duration_aware.classify(real / slot, 0.90, 1.10, 0.80, 1.20)
        actions[seg["sync_action"]] = actions.get(seg["sync_action"], 0) + 1
    ws.push_json(doc, ws.layout.segments)
    check("decision table hit every branch", len(actions) >= 3, json.dumps(actions))

    # ------------------------------------------------- 7b. the adapt loop ---
    # This is what the n8n IF node does: send the off-target segments back to
    # translation with a corrected budget, re-voice them, re-measure. Without
    # it the sync stage is asked to fix something it cannot fix.
    step("7b · adapt loop (what the n8n IF branch drives)")
    off = [x for x in doc["segments"] if x["sync_action"] == "retranslate"]
    check("some segments needed adapting", bool(off), f"{len(off)} of {len(doc['segments'])}")

    first = off[0]
    budget = duration_aware.budget_from_ratio(first["translated_text"],
                                              first["duration_ratio"])
    check("  budget moves the right way",
          (budget < len(first["translated_text"])) == (first["duration_ratio"] > 1),
          f"ratio {first['duration_ratio']:.2f} -> {budget} chars")

    for seg in off:
        slot = seg["end"] - seg["start"]
        # a good re-translation lands inside the accept window
        seg["translated_text"] = fake_translate(seg["source_text"], 1.0)
        seg["attempts"] = seg.get("attempts", 0) + 1
        wav = generated / f"segment_{seg['segment_id']:04d}_v1.wav"
        fake_tts(wav, slot * 1.02)
        real = ffmpeg.duration_of(wav)
        seg["generated_audio"] = ws.push(
            wav, ws.layout.generated_segment(seg["segment_id"], 1))
        seg["generated_duration"] = round(real, 3)
        seg["duration_ratio"] = round(real / slot, 4)
        seg["sync_action"] = duration_aware.classify(real / slot, 0.90, 1.10, 0.80, 1.20)
    ws.push_json(doc, ws.layout.segments)

    still_off = [x["segment_id"] for x in doc["segments"] if x["sync_action"] == "retranslate"]
    check("adapt cleared every off-target segment", not still_off, str(still_off))

    # ----------------------------------------------------- 8. synchronize ---
    step("8 · place every take at its ORIGINAL timestamp")
    with jobs.track(job_id, "synchronize"):
        s = sync.synchronize(job_id, ws.layout.segments)
    dubbed = ws.pull(s["dubbed_track_key"])
    check("all segments placed", s["placed_segments"] == len(doc["segments"]),
          f"{s['placed_segments']}/{len(doc['segments'])}, {s['stretched_segments']} stretched")
    check("nothing dropped", not s["dropped_segments"])
    check("canvas == video length", abs(ffmpeg.duration_of(dubbed) - total) < 1.0,
          f"{ffmpeg.duration_of(dubbed):.1f}s vs {total:.1f}s")
    doc = ws.storage.get_json(ws.layout.segments)
    stretches = [x["applied_stretch"] for x in doc["segments"] if "applied_stretch" in x]
    check("stretch stayed inside the quality window",
          all(0.89 <= f <= 1.36 for f in stretches),
          f"min {min(stretches):.2f} max {max(stretches):.2f}")
    check("takes stay in chronological order",
          all(a["placed_at"] <= b["placed_at"]
              for a, b in zip(doc["segments"], doc["segments"][1:])))
    check("no take overruns the next one after adapting",
          not s["overlapping_segments"], json.dumps(s["overlapping_segments"]))
    check("no take is cut off at the end of the video",
          not s["truncated_at_end"], str(s["truncated_at_end"]))

    # Prove the placement from the AUDIO, not from the code's own bookkeeping.
    import wave

    import numpy as np
    with wave.open(str(dubbed)) as wf:
        sr = wf.getframerate()
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), "<i2").astype(np.float32) / 32768

    def rms(t0: float, t1: float) -> float:
        a, b = int(t0 * sr), int(t1 * sr)
        chunk = pcm[max(0, a):min(len(pcm), b)]
        return float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) else 0.0

    loud = [rms(x["placed_at"], x["placed_at"] + x["final_duration"]) for x in doc["segments"]]
    gaps = []
    for a, b in zip(doc["segments"], doc["segments"][1:]):
        g0, g1 = a["placed_at"] + a["final_duration"], b["placed_at"]
        if g1 - g0 > 0.15:
            gaps.append(rms(g0 + 0.05, g1 - 0.05))
    check("audio really is present in every take window", min(loud) > 0.005,
          f"quietest take RMS {min(loud):.4f}")
    # NOTE: this is measured on the dubbed-speech CANVAS, before mixing. The
    # gaps must be free of *dubbed* speech - in the final mix they are supposed
    # to carry the original background, which is checked separately above.
    check("gaps carry no dubbed speech (measured on the canvas, pre-mix)",
          not gaps or max(gaps) < min(loud) / 10,
          f"loudest gap RMS {max(gaps):.5f} vs quietest take {min(loud):.4f}")
    check("placement drift is zero",
          all(abs(x["placed_at"] - x["start"]) < 0.001 for x in doc["segments"]))

    # ------------------------------------------------------------ 9. mix ---
    step("9 · mix dubbed voice under the preserved background")
    with jobs.track(job_id, "mix_audio"):
        m = mixing.mix(job_id, s["dubbed_track_key"])
    final = ws.pull(m["final_audio_key"])
    check("background was kept", m["background_used"],
          m.get("background_reason") or "")

    # The boolean above used to pass while the stem was DIGITAL SILENCE, so
    # prove it from the samples: the mix must actually carry the original
    # soundtrack, and it must correlate with the source audio.
    import wave

    import numpy as np

    def load(path, take_mono=True):
        with wave.open(str(path)) as wf:
            ch, sr = wf.getnchannels(), wf.getframerate()
            a = np.frombuffer(wf.readframes(wf.getnframes()), "<i2").astype(np.float32) / 32768
        if ch > 1 and take_mono:
            a = a.reshape(-1, ch).mean(axis=1)
        return a, sr

    bg_path = ws.pull(sep["background_key"])
    bg, _ = load(bg_path)
    bg_peak = 20 * np.log10(max(float(np.max(np.abs(bg))), 1e-9))
    check("background stem is not digital silence", bg_peak > -60,
          f"peak {bg_peak:.1f} dBFS")

    orig, sr_o = load(ws.pull(res["audio_key"]))
    mixed, sr_m = load(final)
    dubbed_track, sr_dub = load(dubbed)

    # The soundtrack is now DUCKED under the dub (sidechaincompress), so a
    # whole-file correlation no longer measures what it used to: this script's
    # stub TTS speaks over ~85% of the timeline, ducking is therefore active
    # almost everywhere, and the coefficient collapses even though the mix is
    # exactly right. Measure the two things that actually matter instead.
    step_ = max(1, sr_m // sr_o)
    orig_ds = orig[: len(mixed) // step_]
    mix_ds = mixed[::step_][: len(orig_ds)]

    # Where the dub is silent, the original must come through essentially intact.
    frame = 0.05
    frames = int(min(len(dubbed_track) / sr_dub, len(orig_ds) / sr_o) / frame)
    quiet = np.zeros(len(orig_ds), dtype=bool)
    for i in range(frames):
        chunk = dubbed_track[int(i * frame * sr_dub):int((i + 1) * frame * sr_dub)]
        if len(chunk) and float(np.sqrt(np.mean(chunk ** 2))) <= 1e-4:
            quiet[int(i * frame * sr_o):int((i + 1) * frame * sr_o)] = True

    if quiet.sum() > sr_o:            # at least a second of dub-free audio
        a = orig_ds[quiet]; b = mix_ds[quiet]
        a = a - a.mean(); b = b - b.mean()
        gap_corr = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        check("original soundtrack survives where the dub is silent", gap_corr > 0.5,
              f"correlation over {quiet.sum() / sr_o:.1f}s of gaps: {gap_corr:.2f}")
    else:
        check("enough dub-free audio to verify the soundtrack", False,
              f"only {quiet.sum() / sr_o:.2f}s")

    # And it must never be ducked into nonexistence.
    bg_in_mix = float(np.sqrt(np.mean(mix_ds[quiet] ** 2))) if quiet.sum() else 0.0
    check("background is still audible between takes",
          bg_in_mix > 10 ** (-60 / 20),
          f"{20 * np.log10(max(bg_in_mix, 1e-9)):.1f} dBFS in the gaps")
    check("loudness normalised", m["loudnorm_applied"])
    peak = ffmpeg.peak_dbfs(final)
    check("mix is audible and not clipping", peak is not None and -12 < peak <= 0.2,
          f"peak {peak} dBFS")
    check("mix spans the clip", abs(ffmpeg.duration_of(final) - total) < 1.0)
    src_ch = (info.get("audio") or {}).get("channels")
    mix_ch = (ffmpeg.media_info(final).get("audio") or {}).get("channels")
    check("stereo image of the original soundtrack survives", mix_ch == src_ch,
          f"source {src_ch}ch -> mix {mix_ch}ch (layout {m['channel_layout']})")

    # ------------------------------------------------------ 10. subtitles ---
    step("10 · subtitles")
    with jobs.track(job_id, "generate_subtitles"):
        sub = subs.generate(job_id, ws.layout.segments, formats=["srt", "vtt"])
    srt = ws.storage.get_bytes(sub["subtitle_keys"]["srt"]).decode()
    check("srt written", sub["cue_count"] == len(doc["segments"]),
          f"{sub['cue_count']} cues")
    check("srt header is well formed",
          bool(re.match(r"^1\n\d\d:\d\d:\d\d,\d\d\d --> \d\d:\d\d:\d\d,\d\d\d\n", srt)))
    check("source subtitles also produced", "srt" in sub["source_subtitle_keys"])
    times = re.findall(r"(\d\d):(\d\d):(\d\d),(\d\d\d) --> (\d\d):(\d\d):(\d\d),(\d\d\d)", srt)
    last_end = max(int(t[4]) * 3600 + int(t[5]) * 60 + int(t[6]) + int(t[7]) / 1000
                   for t in times)
    check("no cue runs past the video", last_end <= total + 1.0,
          f"last cue ends at {last_end:.1f}s")

    # --------------------------------------------------------- 11. render ---
    step("11 · render")
    with jobs.track(job_id, "render_video"):
        r = render.render(job_id, video_key, m["final_audio_key"], target_language="vi",
                          subtitle_key=sub["subtitle_keys"]["srt"])
    jobs.mark_completed(job_id, output_key=r["output_key"],
                        subtitle_key=sub["subtitle_keys"]["srt"])
    out = ws.pull(r["output_key"])
    out_info = ffmpeg.media_info(out)
    check("output exists", out.exists(), f"{r['size_bytes']/1_048_576:.1f} MB")
    check("output has video + audio", out_info["has_video"] and out_info["has_audio"])
    check("duration preserved", abs(out_info["duration"] - info["duration"]) < 1.0,
          f"{out_info['duration']:.1f}s vs {info['duration']:.1f}s")
    check("video stream was NOT re-encoded", r["video_stream_copied"],
          "re-encoding every job is slow and lossy")
    check("subtitle track embedded in the mp4",
          any(st.get("codec_type") == "subtitle"
              for st in ffmpeg.probe(out).get("streams", [])))
    check("video stream copied untouched",
          out_info["video"]["width"] == info["video"]["width"]
          and out_info["video"]["height"] == info["video"]["height"],
          f"{out_info['video']['width']}x{out_info['video']['height']} "
          f"{out_info['video']['codec']}")
    check("audio re-encoded to aac", out_info["audio"]["codec"] == "aac")
    check("output keeps the source channel count",
          out_info["audio"]["channels"] == info["audio"]["channels"],
          f"{out_info['audio']['channels']}ch")

    # ------------------------------------------------------- 12. job state --
    step("12 · job registry")
    with jobs_db.session_scope() as session:
        state = jobs.get_job(session, job_id).to_dict()
    check("job completed", state["status"] == "completed")
    check("every stage that ran is green",
          all(st["status"] in ("completed", "skipped")
              for st in state["stages"] if st["status"] != "pending"),
          f"{state['progress']['completed']}/{state['progress']['total']}")
    check("not reported as waiting", state["progress"]["waiting_for_trigger"] is False)

    # ------------------------------------------------------------ artifact --
    # NOT a dub. The "voice" is a 180 Hz sine tone and the "translation" is
    # English with filler words appended, because this script deliberately
    # replaces ASR/translation/TTS with stubs. It lives in data/verify/ under a
    # name nobody can mistake for a deliverable - an earlier version wrote
    # data/samples/verify_output.mp4 and was repeatedly reported as "the dub is
    # broken: background music, no voice, no translation".
    dest_dir = ROOT / "data/verify"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "SYNTHETIC_no_real_voice.mp4"
    shutil.copyfile(out, dest)
    srt_dest = dest.with_suffix(".srt")
    srt_dest.write_text(srt, encoding="utf-8")

    print("\n\033[1;33m" + "=" * 72)
    print("  THIS IS NOT A DUB. ASR / translation / TTS were STUBBED OUT.")
    print("  The 'voice' is a sine tone; the 'translation' is English + filler.")
    print("  It only proves the ffmpeg + timing + mixing code is correct.")
    print("  For a real dub run:  ./scripts/smoke-test.sh data/samples/<clip>.mp4 vi")
    print("=" * 72 + "\033[0m")
    print(f"\n\033[1msynthetic output:\033[0m {dest}")
    print(f"\033[1msubtitle        :\033[0m {srt_dest}")
    print(f"\033[1mworkdir kept at :\033[0m {WORK}")

    print()
    if failures:
        print(f"\033[1;31m{len(failures)} check(s) failed:\033[0m")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\033[1;32mall checks passed\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
