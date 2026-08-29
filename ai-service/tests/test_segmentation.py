"""Segmentation is what stands between Whisper's decoder windows and the dub.

Every case here is a shape that was observed on real Whisper output and that
used to reach the synchroniser unchanged.
"""
from __future__ import annotations

from app.services.segmentation import build_utterances


def words(*spec):
    """(text, start, end) triples -> the word dicts asr.transcribe produces."""
    return [{"word": t, "start": s, "end": e} for t, s, e in spec]


def window(start, end, text, w=None, **extra):
    item = {"start": start, "end": end, "source_text": text, **extra}
    if w is not None:
        item["words"] = w
    return item


# --------------------------------------------------------------- the big one --
def test_window_spanning_a_long_silence_is_tightened_onto_its_words():
    """The failure that made dubs sound like 'music, then nobody speaks'.

    Whisper reported "Call it, Captain." as 61.38 -> 115.05 (54 s) because the
    VAD filter had removed the silence in between. Dubbing divides by that span.
    """
    utterances = build_utterances([
        window(61.38, 115.05, "Call it, Captain.",
               words(("Call", 114.37, 114.55), (" it,", 114.55, 114.75),
                     (" Captain.", 114.75, 115.05))),
    ])

    assert len(utterances) == 1
    assert utterances[0]["start"] == 114.37
    assert utterances[0]["end"] == 115.05
    assert utterances[0]["duration"] < 1.0


def test_a_pause_inside_one_window_becomes_two_utterances():
    utterances = build_utterances([
        window(50.65, 55.22, "Dr. Banner, now might be a good time to get angry.",
               words(("Dr.", 50.65, 50.95), (" Banner,", 50.95, 51.39),
                     # 2 s dramatic pause
                     (" now", 53.42, 53.60), (" might", 53.60, 53.85),
                     (" be", 53.85, 54.00), (" a", 54.00, 54.10),
                     (" good", 54.10, 54.40), (" time", 54.40, 54.70),
                     (" to", 54.70, 54.85), (" get", 54.85, 55.00),
                     (" angry.", 55.00, 55.22))),
    ])

    assert len(utterances) == 2
    assert utterances[0]["source_text"] == "Dr. Banner,"
    assert utterances[1]["start"] == 53.42
    # The dead 2 s belongs to neither take.
    assert utterances[0]["end"] < utterances[1]["start"]


def test_long_run_of_speech_is_split_at_sentence_punctuation():
    spec, t = [], 0.0
    for i in range(40):
        text = f" word{i}." if i % 8 == 7 else f" word{i}"
        spec.append((text, t, t + 0.45))
        t += 0.5
    utterances = build_utterances([window(0.0, t, "".join(spec[i][0] for i in range(40)),
                                          words(*spec))], max_seconds=6.0)

    assert len(utterances) > 1
    assert all(u["duration"] <= 6.0 * 1.5 for u in utterances)


# --------------------------------------------------------------- invariants ---
def test_ids_are_contiguous_and_segments_never_overlap():
    utterances = build_utterances([
        window(0.0, 2.0, "one", words(("one", 0.0, 2.0))),
        window(1.5, 3.0, "two", words(("two", 1.5, 3.0))),
        window(5.0, 6.0, "three", words(("three", 5.0, 6.0))),
    ])

    assert [u["segment_id"] for u in utterances] == list(range(len(utterances)))
    for a, b in zip(utterances, utterances[1:], strict=False):
        assert a["end"] <= b["start"] + 1e-9
        assert a["start"] < a["end"]


def test_nothing_runs_past_the_media_duration():
    utterances = build_utterances(
        [window(9.0, 30.0, "trailing", words(("trailing", 9.0, 30.0)))],
        total_duration=10.0,
    )
    assert all(u["end"] <= 10.0 for u in utterances)


# ------------------------------------------------------------------ hygiene ---
def test_hallucinated_window_is_dropped():
    """Both signals must agree: unlikely speech AND a bad decode."""
    utterances = build_utterances([
        window(0.0, 3.0, "Thanks for watching!", no_speech_prob=0.95, avg_logprob=-1.8),
        window(4.0, 5.0, "real line", words(("real line", 4.0, 5.0)),
               no_speech_prob=0.1, avg_logprob=-0.2),
    ])
    assert [u["source_text"] for u in utterances] == ["real line"]


def test_confident_speech_survives_a_high_no_speech_prob():
    """The regression that deleted 6 of 9 windows of real dialogue.

    On a Demucs vocals stem Whisper reports no_speech_prob ~0.84 for lines it
    transcribed perfectly well. Dropping on that alone silently threw away most
    of the scene, and the dub came out as music with three stray lines in it.
    """
    utterances = build_utterances([
        window(0.0, 2.0, "Call it, Captain.", words(("Call it, Captain.", 0.0, 2.0)),
               no_speech_prob=0.84, avg_logprob=-0.35),
    ])
    assert [u["source_text"] for u in utterances] == ["Call it, Captain."]


def test_high_no_speech_prob_without_logprob_information_is_kept():
    utterances = build_utterances([
        window(0.0, 2.0, "keep me", words(("keep me", 0.0, 2.0)), no_speech_prob=0.99),
    ])
    assert len(utterances) == 1


def test_repeated_lines_are_collapsed():
    utterances = build_utterances([
        window(0.0, 1.0, "same", words(("same", 0.0, 1.0))),
        window(2.0, 3.0, "same", words(("same", 2.0, 3.0))),
        window(4.0, 5.0, "different", words(("different", 4.0, 5.0))),
    ])
    assert [u["source_text"] for u in utterances] == ["same", "different"]


def test_slivers_are_folded_into_their_neighbour():
    utterances = build_utterances([
        window(0.0, 1.4, "a real line", words(("a real line", 0.0, 1.4))),
        window(1.5, 1.6, "oh", words(("oh", 1.5, 1.6))),
    ], min_seconds=0.35)
    assert len(utterances) == 1
    assert utterances[0]["source_text"] == "a real line oh"


# --------------------------------------------------- no word timestamps path --
def test_window_without_words_is_split_by_sentence():
    utterances = build_utterances(
        [window(0.0, 30.0, "First sentence here. Second sentence here. Third one.")],
        max_seconds=10.0,
    )
    assert len(utterances) == 3
    assert utterances[0]["start"] == 0.0
    assert abs(utterances[-1]["end"] - 30.0) < 0.01


def test_unpunctuated_blob_without_words_keeps_text_but_clamps_the_slot():
    utterances = build_utterances(
        [window(10.0, 70.0, "a b c d e f g")], max_seconds=12.0,
    )
    assert len(utterances) == 1
    assert utterances[0]["source_text"] == "a b c d e f g"
    assert utterances[0]["duration"] <= 12.0


def test_backwards_word_timestamps_fall_back_to_the_window():
    utterances = build_utterances([
        window(0.0, 4.0, "scrambled",
               words(("scram", 3.0, 3.5), ("bled", 0.1, 0.5))),
    ])
    assert len(utterances) == 1
    assert utterances[0]["source_text"] == "scrambled"


def test_empty_input_is_empty_output():
    assert build_utterances([]) == []
