"""
Automated Test Suite for Phase 1: Movie Ingest Module
"""

import json
import subprocess
import sys
from pathlib import Path
import pytest

from movie_analyzer.ingest import (
    MovieIngestError,
    SourceFileNotFoundError,
    InvalidMediaFileError,
    ingest_movie,
    extract_metadata,
    save_metadata,
    sanitize_movie_id,
    parse_fps,
    format_file_size,
)


@pytest.fixture(scope="session")
def synthetic_video_path(tmp_path_factory) -> Path:
    """Generate a standard synthetic video for testing using FFmpeg."""
    temp_dir = tmp_path_factory.mktemp("media")
    video_file = temp_dir / "synthetic_test_video.mp4"

    cmd = [
        "ffmpeg",
        "-y",
        "-f", "lavfi", "-i", "testsrc=duration=2:size=1280x720:rate=24",
        "-f", "lavfi", "-i", "sine=frequency=1000:duration=2",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-ac", "2",
        "-ar", "44100",
        "-metadata", "title=Synthetic Test Movie",
        str(video_file),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"FFmpeg failed: {res.stderr}"
    assert video_file.exists()
    return video_file


@pytest.fixture(scope="session")
def space_filename_video_path(tmp_path_factory) -> Path:
    """Generate a synthetic video with spaces in filename and path."""
    temp_dir = tmp_path_factory.mktemp("space folder media")
    video_file = temp_dir / "Synthetic Space Movie Test 2026.mp4"

    cmd = [
        "ffmpeg",
        "-y",
        "-f", "lavfi", "-i", "testsrc=duration=1:size=640x360:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-ac", "1",
        "-ar", "48000",
        str(video_file),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"FFmpeg failed: {res.stderr}"
    assert video_file.exists()
    return video_file


def test_sanitize_movie_id():
    """Test movie_id sanitization."""
    assert sanitize_movie_id("My Great Movie (2024).mp4") == "my_great_movie_2024"
    assert sanitize_movie_id("   Spaces & Special # Characters!  ") == "spaces_special_characters"
    assert sanitize_movie_id("simple_id") == "simple_id"
    assert sanitize_movie_id("") == "unnamed_movie"


def test_format_file_size():
    """Test human-readable file size conversion."""
    assert format_file_size(500) == "500 B"
    assert format_file_size(1024) == "1.00 KB"
    assert format_file_size(1048576) == "1.00 MB"
    assert format_file_size(1073741824) == "1.00 GB"


def test_parse_fps():
    """Test parsing FPS strings."""
    assert parse_fps("24/1") == 24.0
    assert parse_fps("30000/1001") == 29.97
    assert parse_fps("60/1") == 60.0
    assert parse_fps("0/0") == 0.0
    assert parse_fps(None) == 0.0
    assert parse_fps("invalid") == 0.0


def test_ingest_valid_synthetic_video(synthetic_video_path: Path, tmp_path: Path):
    """Test standard ingestion of a valid synthetic video."""
    out_dir = tmp_path / "analysis"
    metadata = ingest_movie(
        source_path=synthetic_video_path,
        output_base_dir=out_dir,
    )

    assert metadata.movie_id == "synthetic_test_video"
    assert metadata.source_filename == "synthetic_test_video.mp4"
    assert metadata.file_extension == ".mp4"
    assert metadata.width == 1280
    assert metadata.height == 720
    assert metadata.resolution == "1280x720"
    assert metadata.fps == 24.0
    assert metadata.video_codec == "h264"
    assert metadata.audio_codec == "aac"
    assert metadata.audio_channel_count == 2
    assert metadata.audio_sample_rate == 44100
    assert metadata.number_of_video_streams == 1
    assert metadata.number_of_audio_streams == 1
    assert metadata.duration_seconds >= 1.9
    assert metadata.file_size_bytes > 0

    # Verify JSON file creation
    expected_json_path = out_dir / "synthetic_test_video" / "movie_metadata.json"
    assert expected_json_path.exists()

    with open(expected_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["movie_id"] == "synthetic_test_video"
    assert data["resolution"] == "1280x720"
    assert data["fps"] == 24.0
    assert data["video_codec"] == "h264"
    assert data["audio_codec"] == "aac"


def test_ingest_nonexistent_path(tmp_path: Path):
    """Test that ingestion of nonexistent file raises SourceFileNotFoundError."""
    fake_path = tmp_path / "nonexistent_movie_file_xyz123.mp4"
    with pytest.raises(SourceFileNotFoundError) as exc_info:
        ingest_movie(source_path=fake_path)
    assert "does not exist" in str(exc_info.value)


def test_ingest_invalid_media_file(tmp_path: Path):
    """Test that ingestion of an invalid corrupt media file raises InvalidMediaFileError."""
    corrupt_file = tmp_path / "corrupt_video.mp4"
    corrupt_file.write_text("THIS IS NOT A VALID VIDEO FILE CONTENT", encoding="utf-8")

    with pytest.raises(InvalidMediaFileError):
        ingest_movie(source_path=corrupt_file, output_base_dir=tmp_path / "analysis")


def test_ingest_filename_with_spaces(space_filename_video_path: Path, tmp_path: Path):
    """Test ingestion with filenames and directory paths containing spaces."""
    out_dir = tmp_path / "analysis space test"
    metadata = ingest_movie(
        source_path=space_filename_video_path,
        output_base_dir=out_dir,
    )

    assert metadata.movie_id == "synthetic_space_movie_test_2026"
    assert metadata.width == 640
    assert metadata.height == 360
    assert metadata.resolution == "640x360"
    assert metadata.fps == 30.0
    assert metadata.audio_channel_count == 1
    assert metadata.audio_sample_rate == 48000

    expected_json = out_dir / "synthetic_space_movie_test_2026" / "movie_metadata.json"
    assert expected_json.exists()


def test_ingest_cli_execution(synthetic_video_path: Path, tmp_path: Path):
    """Test CLI module invocation python -m movie_analyzer.ingest."""
    out_dir = tmp_path / "cli_analysis"
    cmd = [
        sys.executable,
        "-m", "movie_analyzer.ingest",
        str(synthetic_video_path),
        "--output-dir", str(out_dir),
        "--movie-id", "cli_test_movie",
    ]

    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]))
    assert res.returncode == 0, f"CLI failed:\nStdout: {res.stdout}\nStderr: {res.stderr}"
    assert "[SUCCESS] MOVIE INGESTION COMPLETED" in res.stdout
    assert (out_dir / "cli_test_movie" / "movie_metadata.json").exists()


def test_ingest_cli_nonexistent_file_exit_code(tmp_path: Path):
    """Test CLI returns error exit code on nonexistent file."""
    fake_movie = tmp_path / "non_existent_movie_12345.mp4"
    cmd = [
        sys.executable,
        "-m", "movie_analyzer.ingest",
        str(fake_movie),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]))
    assert res.returncode == 1
    assert "[ERROR] INGESTION FAILED" in res.stderr

