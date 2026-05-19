"""Tests for the musical quality judge."""
from __future__ import annotations

import pytest

from tt_midi_maker.coherence.judge import (
    PatternReport,
    judge_tracks,
    DENSITY_MIN,
    DENSITY_MAX,
    PITCH_SPAN_MIN,
    CLUSTER_RATIO_MAX,
    DIRECTION_REVERSAL_MAX,
    SILENCE_MIN,
)
from tt_midi_maker.models.track import NoteEvent, RoleTrack

_TPB = 480  # ticks per beat used in judge


def _note(pitch: int, start: int, dur: int = 120, vel: int = 80) -> NoteEvent:
    return NoteEvent(pitch=pitch, velocity=vel, start_tick=start,
                     duration_ticks=dur, channel=1)


def _track(role: str, notes: list[NoteEvent]) -> RoleTrack:
    return RoleTrack(role=role, channel=1, program=0, notes=notes)


def _bars_to_ticks(bars: int) -> int:
    return bars * 4 * _TPB


# ── helpers ──────────────────────────────────────────────────────────────────

def test_empty_track_no_issues():
    """A track with zero notes should produce no issues."""
    tracks = [_track("melody", [])]
    report = judge_tracks(tracks, bars=4, bpm=120)
    assert report.passed
    assert report.issues == []


def test_perfect_scale_run():
    """A clean stepwise 8-note C major scale over 4 bars should score perfectly."""
    # C4=60, D4=62, E4=64, F4=65, G4=67, A4=69, B4=71, C5=72
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]
    bar_ticks = _bars_to_ticks(4) // len(pitches)
    notes = [_note(p, i * bar_ticks, bar_ticks // 2) for i, p in enumerate(pitches)]
    tracks = [_track("melody", notes)]
    report = judge_tracks(tracks, bars=4, bpm=120)
    assert report.passed
    assert not any("zigzag" in iss or "interval" in iss for iss in report.issues)


# ── density ──────────────────────────────────────────────────────────────────

def test_too_sparse_flagged():
    """One note over 4 bars = 0.25 npb — below DENSITY_MIN."""
    notes = [_note(60, 0)]
    report = judge_tracks([_track("melody", notes)], bars=4, bpm=120)
    assert any("sparse" in iss for iss in report.issues)


def test_density_in_range_no_flag():
    """4 notes per bar should be within density range."""
    notes = [_note(60 + i % 7, i * 240, 120) for i in range(16)]
    report = judge_tracks([_track("melody", notes)], bars=4, bpm=120)
    assert not any("sparse" in iss or "machine-gun" in iss for iss in report.issues)


# ── pitch span ───────────────────────────────────────────────────────────────

def test_narrow_pitch_span_flagged():
    """All notes on the same pitch = 0 semitone span — flagged as monotonous."""
    notes = [_note(60, i * 480, 240) for i in range(8)]
    report = judge_tracks([_track("melody", notes)], bars=4, bpm=120)
    assert any("monotonous" in iss or "pitch span" in iss for iss in report.issues)


def test_wide_pitch_span_flagged():
    """Span of 40 semitones should be flagged as too wide."""
    notes = [_note(40, 0, 240), _note(80, 480, 240)]
    report = judge_tracks([_track("melody", notes)], bars=4, bpm=120)
    assert any("scattered" in iss or "too wide" in iss for iss in report.issues)


# ── silence ratio ─────────────────────────────────────────────────────────────

def test_no_silence_flagged():
    """Notes filling every tick = silence ratio ~0% — flagged."""
    loop_ticks = _bars_to_ticks(4)
    # One note spanning the whole loop
    notes = [_note(60, 0, loop_ticks)]
    report = judge_tracks([_track("melody", notes)], bars=4, bpm=120)
    assert any("breathing room" in iss for iss in report.issues)


def test_adequate_silence_ok():
    """Short notes with gaps — should not flag silence ratio."""
    notes = [_note(60 + i, i * 960, 240) for i in range(8)]  # 25% duty cycle
    report = judge_tracks([_track("melody", notes)], bars=4, bpm=120)
    assert not any("breathing room" in iss for iss in report.issues)


# ── direction reversals ───────────────────────────────────────────────────────

def test_zigzag_melody_flagged():
    """Alternating up-down-up-down should trigger direction reversal flag."""
    # C4 E4 C4 E4 C4 E4 ... — 100% reversals
    pitches = [60, 64, 60, 64, 60, 64, 60, 64]
    bar_ticks = _bars_to_ticks(4) // len(pitches)
    notes = [_note(p, i * bar_ticks, bar_ticks // 2) for i, p in enumerate(pitches)]
    report = judge_tracks([_track("melody", notes)], bars=4, bpm=120)
    assert any("zigzag" in iss or "reversal" in iss for iss in report.issues)


def test_unidirectional_melody_ok():
    """Ascending scale — 0% reversals — should not be flagged."""
    notes = [_note(60 + i * 2, i * 480, 240) for i in range(8)]
    report = judge_tracks([_track("melody", notes)], bars=4, bpm=120)
    assert not any("reversal" in iss or "zigzag" in iss for iss in report.issues)


# ── rhythmic clustering ───────────────────────────────────────────────────────

def test_clustered_notes_flagged():
    """All notes starting within 24 ticks of each other should be flagged."""
    notes = [_note(60 + i, i * 5, 10) for i in range(10)]  # all within 50 ticks
    report = judge_tracks([_track("melody", notes)], bars=4, bpm=120)
    assert any("cluster" in iss or "pile-up" in iss for iss in report.issues)


def test_evenly_spaced_notes_ok():
    """Notes spaced a full beat apart — should not flag clustering."""
    notes = [_note(60 + i % 8, i * _TPB, _TPB // 2) for i in range(8)]
    report = judge_tracks([_track("melody", notes)], bars=4, bpm=120)
    assert not any("cluster" in iss for iss in report.issues)


# ── register overlap ──────────────────────────────────────────────────────────

def test_register_overlap_flagged():
    """Bass notes above melody's lowest note = register invasion."""
    melody_notes = [_note(60, i * 480, 240) for i in range(4)]   # C4 and up
    bass_notes   = [_note(70, i * 480, 240) for i in range(4)]   # A#4 — invades melody
    tracks = [_track("melody", melody_notes), _track("bass", bass_notes)]
    report = judge_tracks(tracks, bars=4, bpm=120)
    assert any("register" in iss for iss in report.issues)


def test_proper_register_ok():
    """Bass well below melody — no register overlap."""
    melody_notes = [_note(72, i * 480, 240) for i in range(4)]   # C5
    bass_notes   = [_note(36, i * 480, 240) for i in range(4)]   # C2
    tracks = [_track("melody", melody_notes), _track("bass", bass_notes)]
    report = judge_tracks(tracks, bars=4, bpm=120)
    assert not any("register" in iss for iss in report.issues)


# ── PatternReport ─────────────────────────────────────────────────────────────

def test_score_decreases_with_issues():
    """Rule score should decrease as issues accumulate."""
    # Perfect track
    good_notes = [_note(60 + i * 2, i * 480, 240) for i in range(8)]
    good_report = judge_tracks([_track("melody", good_notes)], bars=4, bpm=120)

    # Dense zigzag with no silence — multiple issues
    dense_notes = [_note(60 + (i % 2) * 4, i * 5, 4) for i in range(40)]
    bad_report  = judge_tracks([_track("melody", dense_notes)], bars=4, bpm=120)

    assert good_report.rule_score > bad_report.rule_score


def test_pass_threshold_respected():
    """Patterns with score below threshold should not pass."""
    # Forced to fail: zigzag + no silence + clustering
    notes = [_note(60 + (i % 2) * 5, i * 3, 2) for i in range(50)]
    report = judge_tracks([_track("melody", notes)], bars=4, bpm=120,
                          pass_threshold=0.9)
    assert not report.passed


def test_report_summary_contains_score():
    notes = [_note(60, 0, 240)]
    report = judge_tracks([_track("melody", notes)], bars=4, bpm=120)
    s = report.summary()
    assert str(round(report.rule_score, 2)) in s
