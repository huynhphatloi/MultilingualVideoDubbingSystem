"""Run every engine over the same lines and write a comparison you can cite.

    python benchmark.py --language vi --reference ref.wav
    python benchmark.py --engines vixtts,f5_vi --language vi --reference ref.wav
    python benchmark.py --language vi --reference ref.wav --no-metrics   # speed only

Outputs land in ``--out`` (default ``results/``):

    audio/<engine>/line_00.wav   every take, so a listening test can follow
    results.json                 one record per (engine, line) - the raw data
    results.csv                  the same, flat, for a spreadsheet or pandas
    report.md                    the tables that go in the thesis

**Engines are run one at a time and unloaded in between.** A T4 has 16 GB and
each cloning engine wants 2.5-3 GB plus activations; holding four at once is
how a benchmark turns into an OOM traceback that looks like a model defect.
The cost is reload time between engines, which is reported separately from
synthesis time and excluded from RTF - loading is a once-per-session cost that
a real pipeline pays once, not per line.

One engine failing is recorded and the run continues. An engine that cannot
speak the language at all is skipped before it loads, not after.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
import time
from pathlib import Path

import engines as registry
import metrics
import sentences

log = logging.getLogger("benchmark")


# ------------------------------------------------------------------- run -----
def run_engine(name: str, language: str, lines, reference: Path | None,
               out_dir: Path, *, measure: bool, asr_model: str) -> list[dict]:
    engine = registry.get(name)
    audio_dir = out_dir / "audio" / name
    audio_dir.mkdir(parents=True, exist_ok=True)

    if engine.needs_reference() and reference is None:
        log.warning("skipping %s: it clones and no --reference was given", name)
        return []

    rows: list[dict] = []
    load_started = time.perf_counter()
    try:
        engine.ensure_loaded()
    except Exception as exc:  # noqa: BLE001 - one engine must not end the run
        log.error("%s failed to load: %s", name, exc)
        return [{"engine": name, "line_index": None, "error": f"load: {exc}"}]
    load_seconds = round(time.perf_counter() - load_started, 1)
    log.info("%s loaded in %.1fs", name, load_seconds)

    for index, line in enumerate(lines):
        wav = audio_dir / f"line_{index:02d}.wav"
        row = {
            "engine": name,
            "title": engine.title,
            "voice_cloning": engine.voice_cloning,
            "language": language,
            "line_index": index,
            "words": line.words,
            "bucket": sentences.bucket_of(line.words),
            "probe": line.probe,
            "text": line.text,
            "load_seconds": load_seconds,
        }
        try:
            spoken = engine.speak(line.text, language, wav,
                                  reference_wav=reference if engine.needs_reference() else None)
        except Exception as exc:  # noqa: BLE001 - a bad line is data, not a crash
            log.error("%s line %d failed: %s", name, index, exc)
            rows.append({**row, "error": str(exc)[:300]})
            continue

        row.update({
            "wav": str(wav),
            "audio_seconds": round(spoken.seconds, 3),
            "compute_seconds": round(spoken.compute_seconds, 3),
            "rtf": metrics.rtf(spoken.compute_seconds, spoken.seconds),
            # Speaking rate makes the length columns comparable: an engine that
            # is "fast" only because it rushes is not actually better.
            "chars_per_second": (round(len(line.text) / spoken.seconds, 2)
                                 if spoken.seconds else None),
            "error": None,
        })

        if measure:
            row.update(metrics.intelligibility(line.text, wav, language, asr_model))
            row["mos"] = metrics.predicted_mos(wav)
            row["speaker_similarity"] = (
                metrics.speaker_similarity(reference, wav)
                if (reference is not None and engine.voice_cloning) else None
            )
        rows.append(row)
        log.info("  [%s] %2d words  %.2fs audio  RTF %s  WER %s",
                 name, line.words, row["audio_seconds"], row.get("rtf"),
                 row.get("wer"))

    freed = registry.unload_all(keep=None)
    log.info("unloaded %s", ", ".join(freed) or "nothing")
    return rows


# ---------------------------------------------------------------- report -----
def _mean(values) -> float | None:
    clean = [v for v in values if isinstance(v, int | float)]
    return round(statistics.fmean(clean), 3) if clean else None


def _fmt(value, spec: str = "") -> str:
    if value is None:
        return "-"
    return format(value, spec) if spec else str(value)


def write_report(rows: list[dict], out: Path, language: str) -> str:
    ok = [r for r in rows if not r.get("error") and r.get("line_index") is not None]
    by_engine: dict[str, list[dict]] = {}
    for row in ok:
        by_engine.setdefault(row["engine"], []).append(row)

    lines = [
        f"# TTS engine comparison - {language}",
        "",
        f"{len(by_engine)} engine(s) x {len({r['line_index'] for r in ok})} line(s).",
        "",
        "## Overall",
        "",
        "RTF is compute / audio seconds, model load excluded (a pipeline pays "
        "that once). WER/CER come from a Whisper round-trip: **lower is better, "
        "0 is perfect**. SECS is cosine similarity to the reference speaker and "
        "is only meaningful for cloning engines. MOS is UTMOS, a coarse "
        "out-of-domain screen for Vietnamese - not a verdict.",
        "",
        "| engine | clones | load | RTF | WER | CER | SECS | MOS | failed |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in registry.names():
        rows_e = by_engine.get(name)
        if not rows_e:
            continue
        failed = len([r for r in rows if r["engine"] == name and r.get("error")])
        lines.append(
            f"| `{name}` "
            f"| {'yes' if rows_e[0]['voice_cloning'] else 'no'} "
            f"| {_fmt(rows_e[0].get('load_seconds'), '.0f')}s "
            f"| {_fmt(_mean(r.get('rtf') for r in rows_e), '.2f')} "
            f"| {_fmt(_mean(r.get('wer') for r in rows_e), '.3f')} "
            f"| {_fmt(_mean(r.get('cer') for r in rows_e), '.3f')} "
            f"| {_fmt(_mean(r.get('speaker_similarity') for r in rows_e), '.3f')} "
            f"| {_fmt(_mean(r.get('mos') for r in rows_e), '.2f')} "
            f"| {failed} |"
        )

    lines += [
        "",
        "## WER by sentence length",
        "",
        "**This is the table that decides the engine choice.** Subtitle lines "
        "are mostly under 10 words, so the two leftmost buckets carry more "
        "weight than the average above. An engine whose WER climbs sharply as "
        "sentences get shorter is unusable for dubbing however good it sounds "
        "on a long paragraph.",
        "",
        "| engine | " + " | ".join(b[0] for b in sentences.BUCKETS) + " |",
        "|---" * (len(sentences.BUCKETS) + 1) + "|",
    ]
    for name in registry.names():
        rows_e = by_engine.get(name)
        if not rows_e:
            continue
        cells = []
        for label, _, _ in sentences.BUCKETS:
            subset = [r for r in rows_e if r["bucket"] == label]
            cells.append(_fmt(_mean(r.get("wer") for r in subset), ".3f")
                         if subset else "-")
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")

    failures = [r for r in rows if r.get("error")]
    if failures:
        lines += ["", "## Failures", ""]
        seen: set[tuple] = set()
        for row in failures:
            key = (row["engine"], row["error"][:80])
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- `{row['engine']}`: {row['error']}")

    lines += [
        "",
        "## What the numbers cannot tell you",
        "",
        "Prosody. Every metric here is blind to whether a line sounds like a "
        "person acting or a person reading. Take the top two engines from the "
        "table and listen to `audio/<engine>/` before concluding anything - "
        "the numbers narrowed the field, they did not pick the winner.",
        "",
    ]

    text = "\n".join(lines)
    (out / "report.md").write_text(text, encoding="utf-8")
    return text


def write_data(rows: list[dict], out: Path) -> None:
    (out / "results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (out / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------------ main -----
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--language", default="vi")
    parser.add_argument("--engines", default="",
                        help="comma-separated; default is every engine that "
                             "supports the language")
    parser.add_argument("--reference", type=Path, default=None,
                        help="speaker wav for the cloning engines (6-15s of "
                             "clean speech works best)")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--max-lines", type=int, default=None)
    parser.add_argument("--asr-model", default="medium",
                        help="faster-whisper size for the WER round-trip")
    parser.add_argument("--no-metrics", action="store_true",
                        help="speed only - skips ASR, SECS and MOS")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    wanted = ([e.strip() for e in args.engines.split(",") if e.strip()]
              or registry.for_language(args.language))
    if not wanted:
        log.error("no engine in the registry speaks %r", args.language)
        return 1

    lines = sentences.get(args.language, max_lines=args.max_lines)
    args.out.mkdir(parents=True, exist_ok=True)
    log.info("engines: %s", ", ".join(wanted))
    log.info("lines:   %d in %s", len(lines), args.language)

    rows: list[dict] = []
    for name in wanted:
        log.info("\n=== %s ===", name)
        rows += run_engine(name, args.language, lines, args.reference, args.out,
                           measure=not args.no_metrics, asr_model=args.asr_model)

    write_data(rows, args.out)
    print("\n" + write_report(rows, args.out, args.language))
    log.info("wrote %s", args.out.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
