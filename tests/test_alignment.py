"""Duration alignment arithmetic."""
from __future__ import annotations

from dubflow_core import alignment as A


def test_a_line_that_already_fits_is_left_alone():
    plan = A.plan(3.4, 3.5)
    assert plan.status == "fits"
    assert plan.speed == 1.0


def test_a_long_line_is_sped_up_within_the_limit():
    plan = A.plan(4.4, 3.5)
    assert plan.status == "aligned"
    assert 1.0 < plan.speed <= A.DEFAULT_MAX_SPEED
    assert plan.overflow == 0.0
    assert round(4.4 / plan.speed, 2) <= 3.55


def test_an_impossible_line_is_clamped_and_the_overrun_recorded():
    plan = A.plan(9.0, 3.5)
    assert plan.status == "clamped"
    assert plan.speed == A.DEFAULT_MAX_SPEED
    assert plan.overflow > 0


def test_short_speech_is_not_stretched_unless_asked():
    assert A.plan(2.0, 3.5).status == "fits"
    stretched = A.plan(2.0, 3.5, A.Limits(allow_stretch=True))
    assert stretched.status == "stretched"
    assert stretched.speed == A.DEFAULT_MIN_SPEED


def test_a_segment_without_a_measured_clip_is_unmeasured():
    assert A.plan(None, 3.5).status == "unmeasured"
    assert A.plan(0.0, 3.5).status == "unmeasured"


def test_the_window_runs_to_the_next_segment_not_the_segment_end():
    assert A.window_for(1.0, 3.0, 6.0, 20.0) == 5.0
    assert A.window_for(1.0, 3.0, None, 20.0) == 19.0
    # An overlapping neighbour never shrinks the window below the segment.
    assert A.window_for(1.0, 3.0, 1.5, 20.0) == 2.0


def test_plan_segments_uses_each_neighbour_in_time_order():
    segments = [
        {"start": 0.0, "end": 2.0, "tts_duration_raw": 4.0},
        {"start": 2.0, "end": 4.0, "tts_duration_raw": 1.0},
    ]
    first, second = A.plan_segments(segments, media_duration=10.0)
    assert first.window == 2.0          # bounded by the next segment
    assert first.status in {"aligned", "clamped"}
    assert second.window == 8.0         # bounded by the media duration
    assert second.status == "fits"


def test_record_writes_the_documented_metadata():
    segment = {"start": 1.0, "end": 4.5, "tts_duration_raw": 4.4}
    plan = A.plan(4.4, 3.5)
    A.record(segment, plan, 3.52)
    assert segment["source_duration"] == 3.5
    assert segment["alignment_speed"] == plan.speed
    assert segment["alignment_status"] == "aligned"
    assert segment["tts_duration_final"] == 3.52
    assert segment["tts_duration"] == 3.52
    assert A.summary([segment]) == {"aligned": 1}
