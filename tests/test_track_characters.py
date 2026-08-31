"""
Automated Test Suite for Phase 5: Character Identity Tracking & Memory
"""

import json
import subprocess
import sys
from pathlib import Path
import pytest

from ai import (
    EventRecord,
    EventsDocument,
    VisualAnalysis,
    EventUncertainty,
    MockAIProvider,
)
from movie_analyzer.track_characters import (
    CharacterMemoryTracker,
    CharacterRecord,
    CharactersDocument,
    CharacterAppearance,
    NameEvidence,
    extract_name_evidence_from_dialogue,
    track_movie_characters,
)


@pytest.fixture(scope="session")
def synthetic_phase5_movie(tmp_path_factory) -> Path:
    """Generate a synthetic test movie with dialogue and character appearances."""
    temp_dir = tmp_path_factory.mktemp("media_phase5")
    video_file = temp_dir / "Synthetic Character Movie (2026).mkv"

    srt_file = temp_dir / "dialogue.srt"
    srt_file.write_text(
        "1\n00:00:01,000 --> 00:00:02,500\nHey John, look over there.\n\n"
        "2\n00:00:04,000 --> 00:00:05,500\nI see it, Sarah.\n\n"
        "3\n00:00:07,000 --> 00:00:08,500\nLet's move together.\n",
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

def test_extract_name_evidence_from_dialogue():
    """Test extracting names from dialogue patterns and speaker tags."""
    dialogue = [
        {"start_seconds": 1.0, "end_seconds": 2.5, "text": "Hey John, wait up!"},
        {"start_seconds": 4.0, "end_seconds": 5.5, "text": "Understood.", "speaker": "Sarah"},
        {"start_seconds": 7.0, "end_seconds": 8.5, "text": "My name is Detective Miller."},
    ]

    evidences = extract_name_evidence_from_dialogue(dialogue, event_timestamp=2.0)
    names = [e.candidate_name for e in evidences]

    assert "John" in names
    assert "Sarah" in names
    assert "Miller" in names


def test_character_tracker_recurring_identity():
    """Test that recurring temporary labels in close temporal proximity map to same character."""
    tracker = CharacterMemoryTracker(movie_id="test_movie")

    e1 = EventRecord(
        index=1,
        timestamp_seconds=0.0,
        timestamp="00:00:00.000",
        frame_file="frames/frame_000001.jpg",
        dialogue=[{"text": "Hey John!"}],
        visual=VisualAnalysis(1, ["PERSON_A"], "room", "walking", ["jacket"], "neutral"),
        event_summary="PERSON_A walks in.",
        plot_significance=3,
        uncertainty=EventUncertainty(),
    )

    e2 = EventRecord(
        index=2,
        timestamp_seconds=3.0,
        timestamp="00:00:03.000",
        frame_file="frames/frame_000002.jpg",
        dialogue=[],
        visual=VisualAnalysis(1, ["PERSON_A"], "room", "sitting", ["jacket"], "neutral"),
        event_summary="PERSON_A sits down.",
        plot_significance=3,
        uncertainty=EventUncertainty(),
    )

    m1 = tracker.process_event(e1)
    m2 = tracker.process_event(e2)

    assert m1[0][1] == "CHARACTER_001"
    assert m2[0][1] == "CHARACTER_001"  # Preserved recurring identity!

    char1 = tracker.characters["CHARACTER_001"]
    assert len(char1.appearances) == 2
    assert char1.first_seen == 0.0
    assert char1.last_seen == 3.0
    assert char1.canonical_name == "John"
    assert char1.name_confidence in ("low", "medium", "high")


def test_different_characters_remain_separate():
    """Test that distinct character labels are assigned distinct IDs."""
    tracker = CharacterMemoryTracker(movie_id="test_multi_char")

    e = EventRecord(
        index=1,
        timestamp_seconds=10.0,
        timestamp="00:00:10.000",
        frame_file="frames/frame_000004.jpg",
        dialogue=[],
        visual=VisualAnalysis(2, ["PERSON_A", "PERSON_B"], "office", "speaking", [], "tense"),
        event_summary="Two characters meet.",
        plot_significance=5,
        uncertainty=EventUncertainty(),
    )

    mappings = tracker.process_event(e)
    assert len(mappings) == 2
    assert mappings[0][1] == "CHARACTER_001"
    assert mappings[1][1] == "CHARACTER_002"

    assert "CHARACTER_001" in tracker.characters
    assert "CHARACTER_002" in tracker.characters
    assert tracker.characters["CHARACTER_001"].relationships[0].target_character_id == "CHARACTER_002"


def test_no_name_hallucination():
    """Test that character with no dialogue name evidence keeps canonical_name = None."""
    tracker = CharacterMemoryTracker(movie_id="test_silent")

    e = EventRecord(
        index=1,
        timestamp_seconds=5.0,
        timestamp="00:00:05.000",
        frame_file="frames/frame_000002.jpg",
        dialogue=[],  # No dialogue
        visual=VisualAnalysis(1, ["PERSON_A"], "forest", "running", [], "fearful"),
        event_summary="A person runs through forest.",
        plot_significance=6,
        uncertainty=EventUncertainty(),
    )

    tracker.process_event(e)
    char = tracker.characters["CHARACTER_001"]
    assert char.canonical_name is None
    assert char.name_confidence == "unknown"


def test_conservative_merge():
    """Test manual/conservative merge of two character records."""
    tracker = CharacterMemoryTracker(movie_id="test_merge")

    c1 = CharacterRecord(
        character_id="CHARACTER_001",
        canonical_name=None,
        name_confidence="unknown",
        first_seen=0.0,
        last_seen=10.0,
        appearances=[CharacterAppearance(0.0, 1, "frames/frame_000001.jpg", "PERSON_A", 0.85)],
    )
    c2 = CharacterRecord(
        character_id="CHARACTER_002",
        canonical_name="Alex",
        name_confidence="medium",
        first_seen=20.0,
        last_seen=30.0,
        appearances=[CharacterAppearance(20.0, 5, "frames/frame_000005.jpg", "PERSON_A", 0.90)],
        name_evidence=[NameEvidence(20.0, "Hey Alex!", "Alex", None, 0.85)],
    )

    tracker.characters["CHARACTER_001"] = c1
    tracker.characters["CHARACTER_002"] = c2

    res = tracker.propose_conservative_merge(
        "CHARACTER_001", "CHARACTER_002", reason="Visual continuity confirmed across scene"
    )
    assert res is True
    assert "CHARACTER_002" not in tracker.characters
    assert len(tracker.characters["CHARACTER_001"].appearances) == 2
    assert tracker.characters["CHARACTER_001"].canonical_name == "Alex"
    assert len(tracker.merge_history) == 1


# ==============================================================================
# Pipeline & Integration Tests
# ==============================================================================

def test_track_movie_characters_pipeline(synthetic_phase5_movie: Path, tmp_path: Path):
    """
    Test full Phase 5 character tracking pipeline:
    - Runs Phase 4 events analysis with Mock provider
    - Processes character appearances and names
    - Generates validated characters.json
    """
    out_dir = tmp_path / "analysis_phase5"
    mock_p = MockAIProvider()

    doc = track_movie_characters(
        source_path=synthetic_phase5_movie,
        output_base_dir=out_dir,
        ai_provider=mock_p,
    )

    assert doc.total_characters >= 1
    assert len(doc.characters) >= 1

    # Verify characters.json exists on disk
    json_path = out_dir / doc.movie_id / "characters.json"
    assert json_path.exists()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["movie_id"] == doc.movie_id
    assert "characters" in data
    assert len(data["characters"]) >= 1

    first_char = data["characters"][0]
    assert "character_id" in first_char
    assert "first_seen" in first_char
    assert "last_seen" in first_char
    assert "visual_profile" in first_char
    assert "name_evidence" in first_char


def test_character_tracking_dry_run(synthetic_phase5_movie: Path, tmp_path: Path, capsys):
    """Test --dry-run produces character tracking estimate without modifying state."""
    out_dir = tmp_path / "analysis_char_dry_run"
    mock_p = MockAIProvider()

    doc = track_movie_characters(
        source_path=synthetic_phase5_movie,
        output_base_dir=out_dir,
        dry_run=True,
        ai_provider=mock_p,
    )

    captured = capsys.readouterr()
    assert "[DRY-RUN] PHASE 5" in captured.out
    assert "Total Timeline Events" in captured.out

    # Verify characters.json was not created
    json_path = out_dir / doc.movie_id / "characters.json"
    assert not json_path.exists()


def test_character_tracking_limit_and_resume(synthetic_phase5_movie: Path, tmp_path: Path):
    """Test limiting event count and incremental resume."""
    out_dir = tmp_path / "analysis_char_resume"
    mock_p = MockAIProvider()

    # Run 1: limit=2
    doc1 = track_movie_characters(
        source_path=synthetic_phase5_movie,
        output_base_dir=out_dir,
        limit=2,
        ai_provider=mock_p,
    )
    assert doc1.total_characters >= 1

    # Run 2: resume with no limit
    doc2 = track_movie_characters(
        source_path=synthetic_phase5_movie,
        output_base_dir=out_dir,
        force=False,
        ai_provider=mock_p,
    )
    assert doc2.total_characters >= 1


def test_cli_track_characters(synthetic_phase5_movie: Path, tmp_path: Path):
    """Test CLI execution python -m movie_analyzer.track_characters."""
    out_dir = tmp_path / "cli_analysis_char"
    cmd = [
        sys.executable,
        "-m", "movie_analyzer.track_characters",
        str(synthetic_phase5_movie),
        "--provider", "mock",
        "--output-dir", str(out_dir),
    ]

    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert res.returncode == 0, f"CLI failed: {res.stderr}"
    assert "[SUCCESS] CHARACTER TRACKING COMPLETED" in res.stdout
    assert (out_dir / "synthetic_character_movie_2026" / "characters.json").exists()
