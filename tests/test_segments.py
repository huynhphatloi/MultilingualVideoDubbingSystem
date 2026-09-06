from __future__ import annotations

from dubflow_core import segments as S


def segment(index, start, end, text="hello"):
    return S.make_segment(index, start, end, text)


def test_make_segment_has_every_canonical_key():
    made = segment(0, 1.2345, 4.8501)
    for key in S.SEGMENT_KEYS:
        assert key in made
    assert made["start"] == 1.234 or made["start"] == 1.235
    assert made["duration"] == round(made["end"] - made["start"], 3)
    assert made["speaker_id"] == S.DEFAULT_SPEAKER


def test_without_turns_every_segment_keeps_the_default_speaker():
    segments = [segment(0, 0, 2), segment(1, 3, 5)]
    S.assign_speakers(segments, [])
    assert {item["speaker_id"] for item in segments} == {S.DEFAULT_SPEAKER}


def test_speaker_is_the_one_holding_most_of_the_segment():
    segments = [segment(0, 0, 4)]
    turns = [
        {"speaker_id": "SPEAKER_00", "start": 0.0, "end": 1.0},
        {"speaker_id": "SPEAKER_01", "start": 1.0, "end": 4.0},
    ]
    S.assign_speakers(segments, turns)
    assert segments[0]["speaker_id"] == "SPEAKER_01"
    assert segments[0]["speaker_confidence"] == 0.75


def test_a_segment_that_overlaps_nothing_takes_the_nearest_turn():
    segments = [segment(0, 10.0, 11.0)]
    turns = [{"speaker_id": "SPEAKER_02", "start": 8.0, "end": 9.5}]
    S.assign_speakers(segments, turns)
    assert segments[0]["speaker_id"] == "SPEAKER_02"

    far = [segment(0, 100.0, 101.0)]
    S.assign_speakers(far, turns)
    assert far[0]["speaker_id"] == S.DEFAULT_SPEAKER


def test_merge_turns_joins_short_pauses_only():
    turns = [
        {"speaker_id": "A", "start": 0.0, "end": 1.0},
        {"speaker_id": "A", "start": 1.1, "end": 2.0},
        {"speaker_id": "A", "start": 5.0, "end": 6.0},
        {"speaker_id": "B", "start": 6.1, "end": 7.0},
    ]
    merged = S.merge_turns(turns)
    assert [(item["speaker_id"], item["start"], item["end"]) for item in merged] == [
        ("A", 0.0, 2.0), ("A", 5.0, 6.0), ("B", 6.1, 7.0)
    ]


def test_renumber_sorts_and_reindexes():
    segments = [segment(9, 5, 6), segment(3, 1, 2)]
    ordered = S.renumber(segments)
    assert [item["id"] for item in ordered] == [0, 1]
    assert ordered[0]["start"] == 1.0


def test_subtitle_prefers_the_translation():
    segments = [segment(0, 0, 1.5, "hello")]
    segments[0]["translated_text"] = "xin chao"
    body = S.subtitle(segments)
    assert "xin chao" in body
    assert "00:00:00,000 --> 00:00:01,500" in body


