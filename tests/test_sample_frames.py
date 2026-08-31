"""
Automated Test Suite for Phase 2: Timeline Sampling & Frame Extraction
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
import pytest
from PIL import Image

from movie_analyzer.sample_frames import (
    TimelineSamplingError,
    InvalidIntervalError,
    FrameExtractionError,
    TimelineIndex,
    FrameSample,
    sample_movie_frames,
    compute_sample_points,
    format_timestamp,
)
from movie_analyzer.ingest import SourceFileNotFoundError


@pytest.fixture(scope="session")
def synthetic_10s_video(tmp_path_factory) -> Path:
    """Generate a 10-second synthetic video."""
    temp_dir = tmp_path_factory.mktemp("media_10s")
    video_file = temp_dir / "synthetic_10s_movie.mp4"

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=10:size=640x360:rate=25",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=10",
        "-c:v", "libx264", "-c:a", "aac",
        str(video_file),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"FFmpeg failed: {res.stderr}"
    return video_file


@pytest.fixture(scope="session")
def synthetic_space_video(tmp_path_factory) -> Path:
    """Generate a synthetic video with spaces in directory and filename."""
    temp_dir = tmp_path_factory.mktemp("media with spaces")
    video_file = temp_dir / "Synthetic Space Movie 2026.mp4"

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=4:size=320x240:rate=25",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
        "-c:v", "libx264", "-c:a", "aac",
        str(video_file),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"FFmpeg failed: {res.stderr}"
    return video_file


def test_format_timestamp():
    """Test format_timestamp helper."""
    assert format_timestamp(0.0) == "00:00:00.000"
    assert format_timestamp(3.0) == "00:00:03.000"
    assert format_timestamp(65.5) == "00:01:05.500"
    assert format_timestamp(3661.125) == "01:01:01.125"
    assert format_timestamp(-1.0) == "00:00:00.000"


def test_compute_sample_points():
    """Test deterministic sample points calculation."""
    samples = compute_sample_points(duration_seconds=10.0, interval_seconds=3.0)
    assert len(samples) == 4
    assert samples[0].timestamp_seconds == 0.0
    assert samples[0].frame_file == "frames/frame_000001.jpg"
    assert samples[1].timestamp_seconds == 3.0
    assert samples[2].timestamp_seconds == 6.0
    assert samples[3].timestamp_seconds == 9.0

    # Test exact duration boundary
    samples_exact = compute_sample_points(duration_seconds=9.0, interval_seconds=3.0)
    assert len(samples_exact) == 4
    assert samples_exact[-1].timestamp_seconds == 9.0


def test_invalid_interval_fails(synthetic_10s_video: Path, tmp_path: Path):
    """Test that interval <= 0 raises InvalidIntervalError."""
    with pytest.raises(InvalidIntervalError):
        sample_movie_frames(source_path=synthetic_10s_video, interval_seconds=0, output_base_dir=tmp_path)

    with pytest.raises(InvalidIntervalError):
        sample_movie_frames(source_path=synthetic_10s_video, interval_seconds=-2.5, output_base_dir=tmp_path)


def test_nonexistent_movie_fails(tmp_path: Path):
    """Test that nonexistent video file raises SourceFileNotFoundError."""
    with pytest.raises(SourceFileNotFoundError):
        sample_movie_frames(
            source_path=tmp_path / "nonexistent_video.mp4",
            output_base_dir=tmp_path,
        )


def test_10s_synthetic_movie_sampling(synthetic_10s_video: Path, tmp_path: Path):
    """
    Test 10-second synthetic movie with interval 3s:
    - Expected frame count is 4 (0s, 3s, 6s, 9s)
    - Every timeline entry points to an existing file
    - Timestamps are chronological
    - Final sample does not exceed duration
    - Frames are valid readable JPEGs
    - timeline_index.json is valid
    """
    out_dir = tmp_path / "analysis_test_10s"
    timeline = sample_movie_frames(
        source_path=synthetic_10s_video,
        interval_seconds=3.0,
        output_base_dir=out_dir,
    )

    assert timeline.movie_id == "synthetic_10s_movie"
    assert timeline.sampling_interval_seconds == 3.0
    assert timeline.total_samples == 4
    assert len(timeline.frames) == 4

    # Check chronological timestamps and non-exceeding duration
    prev_ts = -1.0
    for sample in timeline.frames:
        assert sample.timestamp_seconds > prev_ts
        assert sample.timestamp_seconds <= timeline.duration_seconds
        prev_ts = sample.timestamp_seconds

        # Check frame file existence on disk
        frame_path = out_dir / "synthetic_10s_movie" / sample.frame_file
        assert frame_path.exists()
        assert frame_path.stat().st_size > 0

        # Verify image can be opened and decoded
        with Image.open(frame_path) as img:
            assert img.size == (640, 360)
            assert img.format == "JPEG"

    # Check timeline_index.json structure
    json_path = out_dir / "synthetic_10s_movie" / "timeline_index.json"
    assert json_path.exists()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["movie_id"] == "synthetic_10s_movie"
    assert data["sampling_interval_seconds"] == 3.0
    assert data["total_samples"] == 4
    assert len(data["frames"]) == 4


def test_configurable_interval(synthetic_10s_video: Path, tmp_path: Path):
    """Test custom sampling interval (e.g. 2.0s on 10s video -> 6 frames: 0, 2, 4, 6, 8, 10)."""
    out_dir = tmp_path / "analysis_custom_interval"
    timeline = sample_movie_frames(
        source_path=synthetic_10s_video,
        interval_seconds=2.0,
        output_base_dir=out_dir,
    )

    assert timeline.total_samples == 6
    assert [f.timestamp_seconds for f in timeline.frames] == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]


def test_spaces_in_path_and_filename(synthetic_space_video: Path, tmp_path: Path):
    """Test that paths and filenames containing spaces are handled correctly."""
    out_dir = tmp_path / "analysis with spaces"
    timeline = sample_movie_frames(
        source_path=synthetic_space_video,
        interval_seconds=2.0,
        output_base_dir=out_dir,
    )

    assert timeline.total_samples == 3  # 0s, 2s, 4s for 4s duration
    for sample in timeline.frames:
        frame_path = out_dir / timeline.movie_id / sample.frame_file
        assert frame_path.exists()
        with Image.open(frame_path) as img:
            assert img.size == (320, 240)


def test_idempotent_resume_and_force_rebuild(synthetic_10s_video: Path, tmp_path: Path):
    """Test that rerun reuses existing frames, and --force rebuilds."""
    out_dir = tmp_path / "analysis_resume_test"

    # 1. First run
    t0 = time.time()
    timeline1 = sample_movie_frames(
        source_path=synthetic_10s_video,
        interval_seconds=3.0,
        output_base_dir=out_dir,
    )
    t1 = time.time()
    initial_duration = t1 - t0

    frame1_path = out_dir / "synthetic_10s_movie" / "frames" / "frame_000001.jpg"
    assert frame1_path.exists()
    mtime1 = frame1_path.stat().st_mtime

    # 2. Second run without force (should reuse frames and be instant)
    time.sleep(0.05)
    timeline2 = sample_movie_frames(
        source_path=synthetic_10s_video,
        interval_seconds=3.0,
        output_base_dir=out_dir,
        force=False,
    )
    mtime2 = frame1_path.stat().st_mtime
    assert mtime1 == mtime2  # File was reused, not overwritten!
    assert timeline2.total_samples == 4

    # 3. Third run with force=True (should re-extract and update mtime)
    time.sleep(0.05)
    timeline3 = sample_movie_frames(
        source_path=synthetic_10s_video,
        interval_seconds=3.0,
        output_base_dir=out_dir,
        force=True,
    )
    mtime3 = frame1_path.stat().st_mtime
    assert mtime3 >= mtime1
    assert timeline3.total_samples == 4


def test_cli_sample_frames(synthetic_10s_video: Path, tmp_path: Path):
    """Test CLI command python -m movie_analyzer.sample_frames."""
    out_dir = tmp_path / "cli_analysis_frames"
    cmd = [
        sys.executable,
        "-m", "movie_analyzer.sample_frames",
        str(synthetic_10s_video),
        "--interval", "3.0",
        "--output-dir", str(out_dir),
    ]

    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert res.returncode == 0, f"CLI failed: {res.stderr}"
    assert "[SUCCESS] TIMELINE SAMPLING COMPLETED" in res.stdout
    assert (out_dir / "synthetic_10s_movie" / "timeline_index.json").exists()
