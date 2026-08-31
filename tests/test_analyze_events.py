"""
Automated Test Suite for Phase 4: Multimodal Timeline Event Analysis
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
import pytest

from ai import (
    BaseAIProvider,
    EventRecord,
    EventsDocument,
    VisualAnalysis,
    EventUncertainty,
    MockAIProvider,
    GeminiAIProvider,
    get_ai_provider,
    MissingAPIKeyError,
    InvalidModelResponseError,
)
from movie_analyzer.analyze_events import (
    analyze_timeline_events,
    match_dialogue_to_timestamp,
    build_rolling_context,
)
from movie_analyzer.sample_frames import sample_movie_frames
from movie_analyzer.extract_dialogue import extract_movie_dialogue


@pytest.fixture(scope="session")
def synthetic_phase4_movie(tmp_path_factory) -> Path:
    """Generate a synthetic test movie with embedded dialogue for Phase 4 tests."""
    temp_dir = tmp_path_factory.mktemp("media_phase4")
    video_file = temp_dir / "Synthetic Action Movie (2026).mkv"

    srt_file = temp_dir / "subs.srt"
    srt_file.write_text(
        "1\n00:00:01,000 --> 00:00:02,500\nWho entered the building?\n\n"
        "2\n00:00:04,000 --> 00:00:05,500\nIt's an unknown operative.\n\n"
        "3\n00:00:07,000 --> 00:00:08,500\nSecure the perimeter!\n",
        encoding="utf-8"
    )

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=9:size=640x360:rate=25",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=9",
        "-i", str(srt_file),
        "-c:v", "libx264", "-c:a", "aac",
        "-c:s", "srt",
        "-map", "0:v:0", "-map", "1:a:0", "-map", "2:s:0",
        "-metadata:s:s:0", "language=eng",
        "-disposition:s:0", "default",
        str(video_file),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    return video_file


# ==============================================================================
# Unit Tests
# ==============================================================================

def test_match_dialogue_to_timestamp():
    """Test dialogue alignment within configurable time window."""
    dialogue = [
        {"start_seconds": 1.0, "end_seconds": 2.5, "text": "Line 1"},
        {"start_seconds": 5.0, "end_seconds": 6.5, "text": "Line 2"},
        {"start_seconds": 10.0, "end_seconds": 12.0, "text": "Line 3"},
    ]

    # Timestamp 2.0s with window 1.5s -> [0.5, 3.5] -> Matches Line 1
    m1 = match_dialogue_to_timestamp(dialogue, timestamp_seconds=2.0, window_seconds=1.5)
    assert len(m1) == 1
    assert m1[0]["text"] == "Line 1"

    # Timestamp 4.0s with window 1.5s -> [2.5, 5.5] -> Matches Line 1 (ends 2.5) and Line 2 (starts 5.0)
    m2 = match_dialogue_to_timestamp(dialogue, timestamp_seconds=4.0, window_seconds=1.5)
    assert len(m2) == 2

    # Timestamp 8.0s with window 1.0s -> [7.0, 9.0] -> No dialogue
    m3 = match_dialogue_to_timestamp(dialogue, timestamp_seconds=8.0, window_seconds=1.0)
    assert len(m3) == 0


def test_build_rolling_context():
    """Test chronological rolling context construction."""
    events = [
        EventRecord(
            index=1,
            timestamp_seconds=0.0,
            timestamp="00:00:00.000",
            frame_file="frames/frame_000001.jpg",
            dialogue=[],
            visual=VisualAnalysis(1, ["PERSON_A"], "room", "standing", [], "neutral"),
            event_summary="PERSON_A enters the room.",
            plot_significance=4,
            uncertainty=EventUncertainty(),
        ),
        EventRecord(
            index=2,
            timestamp_seconds=3.0,
            timestamp="00:00:03.000",
            frame_file="frames/frame_000002.jpg",
            dialogue=[],
            visual=VisualAnalysis(2, ["PERSON_A", "PERSON_B"], "room", "speaking", [], "tense"),
            event_summary="PERSON_A confronts PERSON_B.",
            plot_significance=6,
            uncertainty=EventUncertainty(),
        ),
    ]

    context = build_rolling_context(events, max_items=2)
    assert len(context) == 2
    assert "[00:00:00.000] PERSON_A enters the room." in context[0]
    assert "[00:00:03.000] PERSON_A confronts PERSON_B." in context[1]


def test_missing_api_key_raises_error(monkeypatch, tmp_path: Path):
    """Test that Gemini provider without API key raises MissingAPIKeyError."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    fake_img = tmp_path / "frame.jpg"
    fake_img.write_bytes(b"dummy")

    provider = GeminiAIProvider(api_key="")
    with pytest.raises(MissingAPIKeyError):
        provider.analyze_frame_event(
            image_path=fake_img,
            timestamp_seconds=0.0,
            timestamp_str="00:00:00.000",
            dialogue_context=[],
            previous_context=[],
        )


def test_provider_factory():
    """Test factory creates correct provider instances."""
    mock_p = get_ai_provider("mock")
    assert isinstance(mock_p, MockAIProvider)
    assert mock_p.provider_name == "mock"

    gemini_p = get_ai_provider("gemini", api_key="dummy_key")
    assert isinstance(gemini_p, GeminiAIProvider)
    assert gemini_p.provider_name == "gemini"

    with pytest.raises(ValueError):
        get_ai_provider("unsupported_provider_xyz")


# ==============================================================================
# Pipeline & Integration Tests
# ==============================================================================

def test_analyze_timeline_events_mock(synthetic_phase4_movie: Path, tmp_path: Path):
    """
    Test complete Phase 4 event analysis with Mock AI provider:
    - Merges timeline and dialogue
    - Generates validated events.json
    - Temporary character labels (PERSON_A, PERSON_B)
    - Uncertainty fields present
    - Plot significance 1-10
    """
    out_dir = tmp_path / "analysis_phase4"
    doc = analyze_timeline_events(
        source_path=synthetic_phase4_movie,
        provider_name="mock",
        output_base_dir=out_dir,
    )

    assert doc.provider == "mock"
    assert doc.total_events == 4  # 9s movie @ 3s interval -> 0s, 3s, 6s, 9s (4 events)
    assert len(doc.events) == 4

    # Verify first event schema
    first_evt = doc.events[0]
    assert first_evt.index == 1
    assert first_evt.timestamp_seconds == 0.0
    assert first_evt.timestamp == "00:00:00.000"
    assert first_evt.frame_file == "frames/frame_000001.jpg"
    assert isinstance(first_evt.visual.people_count, int)
    assert "PERSON_A" in first_evt.visual.character_labels
    assert first_evt.uncertainty.character_identity in ("low", "medium", "high")
    assert 1 <= first_evt.plot_significance <= 10
    assert len(first_evt.event_summary) > 0

    # Verify events.json on disk
    events_json = out_dir / doc.movie_id / "events.json"
    assert events_json.exists()
    with open(events_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["movie_id"] == doc.movie_id
    assert data["total_events"] == 4
    assert len(data["events"]) == 4


def test_dry_run_mode(synthetic_phase4_movie: Path, tmp_path: Path, capsys):
    """Test --dry-run produces usage estimate and does not call model or write events.json."""
    out_dir = tmp_path / "analysis_dry_run"
    mock_p = MockAIProvider()

    doc = analyze_timeline_events(
        source_path=synthetic_phase4_movie,
        output_base_dir=out_dir,
        dry_run=True,
        ai_provider=mock_p,
    )

    assert mock_p.call_count == 0  # No AI calls made
    captured = capsys.readouterr()
    assert "[DRY-RUN] PHASE 4" in captured.out
    assert "Expected AI Calls" in captured.out

    # Verify events.json was NOT written
    events_json = out_dir / doc.movie_id / "events.json"
    assert not events_json.exists()


def test_limit_parameter(synthetic_phase4_movie: Path, tmp_path: Path):
    """Test --limit parameter limits number of frames processed."""
    out_dir = tmp_path / "analysis_limit"
    doc = analyze_timeline_events(
        source_path=synthetic_phase4_movie,
        provider_name="mock",
        output_base_dir=out_dir,
        limit=2,
    )

    assert doc.total_events == 2
    assert len(doc.events) == 2


def test_idempotent_resume_and_checkpoint(synthetic_phase4_movie: Path, tmp_path: Path):
    """Test partial checkpointing, resuming from event K+1 without repeating completed frames."""
    out_dir = tmp_path / "analysis_resume"
    mock_p = MockAIProvider()

    # 1. First run with limit=2 (processes frames 1 & 2)
    doc1 = analyze_timeline_events(
        source_path=synthetic_phase4_movie,
        output_base_dir=out_dir,
        limit=2,
        ai_provider=mock_p,
    )
    assert doc1.total_events == 2
    assert mock_p.call_count == 2

    # 2. Second run without force and no limit (should resume and process only frames 3 & 4)
    doc2 = analyze_timeline_events(
        source_path=synthetic_phase4_movie,
        output_base_dir=out_dir,
        force=False,
        ai_provider=mock_p,
    )
    assert doc2.total_events == 4
    # Call count should increase only by 2 (frames 3 and 4)
    assert mock_p.call_count == 4

    # 3. Third run without force (all 4 are done, 0 additional calls)
    doc3 = analyze_timeline_events(
        source_path=synthetic_phase4_movie,
        output_base_dir=out_dir,
        force=False,
        ai_provider=mock_p,
    )
    assert doc3.total_events == 4
    assert mock_p.call_count == 4

    # 4. Fourth run with force=True (rebuilds all 4 frames -> 4 new calls)
    doc4 = analyze_timeline_events(
        source_path=synthetic_phase4_movie,
        output_base_dir=out_dir,
        force=True,
        ai_provider=mock_p,
    )
    assert doc4.total_events == 4
    assert mock_p.call_count == 8


def test_cli_analyze_events_mock(synthetic_phase4_movie: Path, tmp_path: Path):
    """Test CLI execution python -m movie_analyzer.analyze_events."""
    out_dir = tmp_path / "cli_analysis_events"
    cmd = [
        sys.executable,
        "-m", "movie_analyzer.analyze_events",
        str(synthetic_phase4_movie),
        "--provider", "mock",
        "--output-dir", str(out_dir),
        "--limit", "2",
    ]

    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert res.returncode == 0, f"CLI failed: {res.stderr}"
    assert "[SUCCESS] TIMELINE EVENT ANALYSIS COMPLETED" in res.stdout
    assert (out_dir / "synthetic_action_movie_2026" / "events.json").exists()
