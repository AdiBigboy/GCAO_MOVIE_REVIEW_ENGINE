"""
Automated Integration Test Suite for Phase 6: End-to-End Pipeline Runner
Tests end-to-end sequential execution, error handling, order enforcement, and summary creation.
"""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch
import pytest

from ai import MockAIProvider
from movie_analyzer.ingest import (
    SourceFileNotFoundError,
    InvalidMediaFileError,
)
from movie_analyzer.extract_dialogue import DialogueExtractionError
from run_pipeline import run_movie_pipeline, PipelineExecutionSummary


@pytest.fixture(scope="session")
def pipeline_synthetic_movie(tmp_path_factory) -> Path:
    """Generate a synthetic test movie with video, audio, and embedded subtitles."""
    temp_dir = tmp_path_factory.mktemp("media_pipeline")
    video_file = temp_dir / "Pipeline Test Movie (2026).mkv"

    srt_file = temp_dir / "dialogue.srt"
    srt_file.write_text(
        "1\n00:00:01,000 --> 00:00:02,500\nHello Alice, are you ready?\n\n"
        "2\n00:00:04,000 --> 00:00:05,500\nReady Bob, let's go.\n\n"
        "3\n00:00:07,000 --> 00:00:08,500\nMission complete.\n",
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
    assert res.returncode == 0, f"FFmpeg failed: {res.stderr}"
    assert video_file.exists()
    return video_file


def test_pipeline_nonexistent_movie_fails(tmp_path: Path):
    """Test that a nonexistent movie path raises SourceFileNotFoundError."""
    fake_movie = tmp_path / "ghost_movie_xyz.mp4"
    with pytest.raises(SourceFileNotFoundError) as exc_info:
        run_movie_pipeline(
            source_path=fake_movie,
            output_base_dir=tmp_path / "analysis",
            provider_name="mock",
        )
    assert "not found" in str(exc_info.value)


def test_pipeline_invalid_media_file_fails(tmp_path: Path):
    """Test that corrupt media fails at Phase 1 and writes failed summary."""
    corrupt_file = tmp_path / "corrupt_movie.mp4"
    corrupt_file.write_text("NOT A REAL MOVIE FILE", encoding="utf-8")
    out_dir = tmp_path / "analysis_corrupt"

    with pytest.raises(InvalidMediaFileError):
        run_movie_pipeline(
            source_path=corrupt_file,
            output_base_dir=out_dir,
            provider_name="mock",
        )

    # Verify failure summary written
    summary_path = out_dir / "corrupt_movie" / "pipeline_summary.json"
    assert summary_path.exists()
    with open(summary_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    assert summary_data["status"] == "FAILED"
    assert "phase_1_ingest" in summary_data["phases"]
    assert summary_data["phases"]["phase_1_ingest"]["status"] == "FAILED"


def test_pipeline_end_to_end_mock_success(pipeline_synthetic_movie: Path, tmp_path: Path):
    """
    Test successful end-to-end execution of Phase 1 -> 5:
    - Phase 1: movie_metadata.json
    - Phase 2: timeline_index.json & frames/
    - Phase 3: dialogue.json & subtitles/
    - Phase 4: events.json
    - Phase 5: characters.json
    - Phase 6: pipeline_summary.json
    """
    out_dir = tmp_path / "analysis_e2e"
    mock_provider = MockAIProvider()

    summary: PipelineExecutionSummary = run_movie_pipeline(
        source_path=pipeline_synthetic_movie,
        output_base_dir=out_dir,
        interval_seconds=3.0,
        ai_provider=mock_provider,
    )

    assert summary.status == "SUCCESS"
    assert summary.movie_id == "pipeline_test_movie_2026"
    assert summary.total_duration_seconds > 0.0

    movie_dir = out_dir / summary.movie_id

    # Verify all phase artifacts exist on disk
    assert (movie_dir / "movie_metadata.json").exists()
    assert (movie_dir / "timeline_index.json").exists()
    assert (movie_dir / "frames" / "frame_000001.jpg").exists()
    assert (movie_dir / "dialogue.json").exists()
    assert (movie_dir / "subtitles" / "raw_subtitle.srt").exists()
    assert (movie_dir / "events.json").exists()
    assert (movie_dir / "characters.json").exists()
    assert (movie_dir / "story.json").exists()
    assert (movie_dir / "pipeline_summary.json").exists()

    # Validate pipeline summary contents
    summary_path = movie_dir / "pipeline_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["status"] == "SUCCESS"
    assert "phase_1_ingest" in data["phases"]
    assert data["phases"]["phase_1_ingest"]["resolution"] == "640x360"
    assert data["phases"]["phase_2_sample_frames"]["total_samples"] == 4
    assert data["phases"]["phase_3_extract_dialogue"]["total_entries"] == 3
    assert data["phases"]["phase_4_analyze_events"]["total_events"] == 4
    assert data["phases"]["phase_5_track_characters"]["total_characters"] >= 1
    assert "phase_7_reconstruct_story" in data["phases"]
    assert data["phases"]["phase_7_reconstruct_story"]["status"] == "SUCCESS"


def test_pipeline_execution_order_and_call_sequence(pipeline_synthetic_movie: Path, tmp_path: Path):
    """Verify that phases are called strictly in sequential order (1 -> 2 -> 3 -> 4 -> 5 -> 7)."""
    call_order = []

    import run_pipeline

    orig_ingest = run_pipeline.ingest_movie
    orig_sample = run_pipeline.sample_movie_frames
    orig_dialogue = run_pipeline.extract_movie_dialogue
    orig_events = run_pipeline.analyze_timeline_events
    orig_chars = run_pipeline.track_movie_characters
    orig_story = run_pipeline.reconstruct_movie_story

    def wrap_ingest(*args, **kwargs):
        call_order.append(1)
        return orig_ingest(*args, **kwargs)

    def wrap_sample(*args, **kwargs):
        call_order.append(2)
        return orig_sample(*args, **kwargs)

    def wrap_dialogue(*args, **kwargs):
        call_order.append(3)
        return orig_dialogue(*args, **kwargs)

    def wrap_events(*args, **kwargs):
        call_order.append(4)
        return orig_events(*args, **kwargs)

    def wrap_chars(*args, **kwargs):
        call_order.append(5)
        return orig_chars(*args, **kwargs)

    def wrap_story(*args, **kwargs):
        call_order.append(6)
        return orig_story(*args, **kwargs)

    out_dir = tmp_path / "analysis_order"
    mock_provider = MockAIProvider()

    with patch("run_pipeline.ingest_movie", side_effect=wrap_ingest), \
         patch("run_pipeline.sample_movie_frames", side_effect=wrap_sample), \
         patch("run_pipeline.extract_movie_dialogue", side_effect=wrap_dialogue), \
         patch("run_pipeline.analyze_timeline_events", side_effect=wrap_events), \
         patch("run_pipeline.track_movie_characters", side_effect=wrap_chars), \
         patch("run_pipeline.reconstruct_movie_story", side_effect=wrap_story):

        run_movie_pipeline(
            source_path=pipeline_synthetic_movie,
            output_base_dir=out_dir,
            ai_provider=mock_provider,
        )

    assert call_order == [1, 2, 3, 4, 5, 6], f"Phase call order was incorrect: {call_order}"


def test_pipeline_intermediate_failure_propagation(pipeline_synthetic_movie: Path, tmp_path: Path):
    """Verify that failure in Phase 3 stops pipeline immediately and records failure state."""
    out_dir = tmp_path / "analysis_fail_mid"

    def fail_dialogue(*args, **kwargs):
        raise DialogueExtractionError("Simulated subtitle decoder failure")

    with patch("run_pipeline.extract_movie_dialogue", side_effect=fail_dialogue):
        with pytest.raises(DialogueExtractionError):
            run_movie_pipeline(
                source_path=pipeline_synthetic_movie,
                output_base_dir=out_dir,
                ai_provider=MockAIProvider(),
            )

    summary_file = out_dir / "pipeline_test_movie_2026" / "pipeline_summary.json"
    assert summary_file.exists()
    with open(summary_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["status"] == "FAILED"
    assert data["phases"]["phase_1_ingest"]["status"] == "SUCCESS"
    assert data["phases"]["phase_2_sample_frames"]["status"] == "SUCCESS"
    assert data["phases"]["phase_3_extract_dialogue"]["status"] == "FAILED"
    # Phase 4, 5, 7 must not have executed
    assert "phase_4_analyze_events" not in data["phases"]
    assert "phase_5_track_characters" not in data["phases"]
    assert "phase_7_reconstruct_story" not in data["phases"]


def test_pipeline_dry_run_mode(pipeline_synthetic_movie: Path, tmp_path: Path):
    """Test that --dry-run runs phases without live AI calls and records DRY_RUN status."""
    out_dir = tmp_path / "analysis_dry_run_pipeline"
    mock_p = MockAIProvider()

    summary = run_movie_pipeline(
        source_path=pipeline_synthetic_movie,
        output_base_dir=out_dir,
        dry_run=True,
        ai_provider=mock_p,
    )

    assert summary.status == "DRY_RUN"
    assert summary.phases["phase_4_analyze_events"]["status"] == "DRY_RUN"
    assert summary.phases["phase_5_track_characters"]["status"] == "DRY_RUN"
    assert summary.phases["phase_7_reconstruct_story"]["status"] == "DRY_RUN"
    assert mock_p.call_count == 0


def test_pipeline_cli_invocation(pipeline_synthetic_movie: Path, tmp_path: Path):
    """Test CLI execution: python run_pipeline.py <movie_path> --provider mock."""
    out_dir = tmp_path / "analysis_cli"
    cmd = [
        sys.executable,
        "run_pipeline.py",
        str(pipeline_synthetic_movie),
        "--output-dir", str(out_dir),
        "--provider", "mock",
        "--interval", "3.0",
    ]

    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )

    assert res.returncode == 0, f"CLI pipeline execution failed:\nStdout: {res.stdout}\nStderr: {res.stderr}"
    assert "[1/6] Ingesting movie metadata..." in res.stdout
    assert "[2/6] Sampling timeline frames..." in res.stdout
    assert "[3/6] Extracting dialogue and subtitles..." in res.stdout
    assert "[4/6] Analyzing timeline events..." in res.stdout
    assert "[5/6] Tracking character identities & memory..." in res.stdout
    assert "[6/6] Reconstructing movie story..." in res.stdout
    assert "[SUCCESS] PIPELINE COMPLETED" in res.stdout

    summary_json = out_dir / "pipeline_test_movie_2026" / "pipeline_summary.json"
    assert summary_json.exists()

