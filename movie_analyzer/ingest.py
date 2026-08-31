"""
Movie Review Engine - Phase 1: Movie Metadata Ingestion Module
Extracts comprehensive video/audio/container metadata using ffprobe and persists it to analysis/<movie_id>/movie_metadata.json.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Setup standard logger
logger = logging.getLogger("movie_analyzer.ingest")


# ==============================================================================
# Exceptions
# ==============================================================================

class MovieIngestError(Exception):
    """Base exception for all movie ingestion errors."""
    pass


class FFprobeNotFoundError(MovieIngestError):
    """Raised when ffprobe executable is not available on PATH."""
    pass


class SourceFileNotFoundError(MovieIngestError):
    """Raised when the requested movie source file does not exist."""
    pass


class InvalidMediaFileError(MovieIngestError):
    """Raised when ffprobe fails to parse or identify a valid media file."""
    pass


class MetadataWriteError(MovieIngestError):
    """Raised when saving movie_metadata.json fails."""
    pass


# ==============================================================================
# Data Models
# ==============================================================================

@dataclass
class SubtitleTrack:
    """Detailed information for a subtitle stream."""
    stream_index: int
    codec_name: str
    language: str
    title: Optional[str] = None
    is_default: bool = False
    is_forced: bool = False


@dataclass
class MovieMetadata:
    """Standardized metadata representation of an ingested movie."""
    movie_id: str
    source_filename: str
    absolute_source_path: str
    file_extension: str
    file_size_bytes: int
    file_size_human: str
    container_format: str
    duration_seconds: float
    width: int
    height: int
    resolution: str
    fps: float
    video_codec: Optional[str]
    audio_codec: Optional[str]
    audio_channel_count: int
    audio_sample_rate: Optional[int]
    number_of_video_streams: int
    number_of_audio_streams: int
    number_of_subtitle_streams: int
    subtitle_languages: List[str]
    subtitle_tracks: List[Dict[str, Any]]
    file_created_at: Optional[str]
    file_modified_at: Optional[str]
    media_created_at: Optional[str]
    ingested_at: str
    ffprobe_raw_format_tags: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata dataclass to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert metadata dataclass to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ==============================================================================
# Helper Functions
# ==============================================================================

def sanitize_movie_id(name: str) -> str:
    """Generate a clean filesystem-safe movie_id from a filename or title."""
    # Strip extension if passed
    stem = Path(name).stem
    # Replace non-alphanumeric characters with underscores
    cleaned = re.sub(r"[^\w\-]+", "_", stem).strip("_")
    return cleaned.lower() if cleaned else "unnamed_movie"


def format_file_size(size_in_bytes: int) -> str:
    """Format file size in bytes to human-readable string (KB, MB, GB)."""
    if size_in_bytes < 0:
        return "0 B"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.2f} {unit}" if unit != "B" else f"{size_in_bytes} B"
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.2f} PB"


def parse_fps(rate_str: Optional[str]) -> float:
    """Safely calculate FPS from ffprobe rational fraction (e.g. '24/1', '30000/1001')."""
    if not rate_str or rate_str == "0/0":
        return 0.0
    try:
        if "/" in rate_str:
            num, den = rate_str.split("/", 1)
            denominator = float(den)
            if denominator == 0:
                return 0.0
            return round(float(num) / denominator, 3)
        return round(float(rate_str), 3)
    except Exception as exc:
        logger.warning("Could not parse fps string '%s': %s", rate_str, exc)
        return 0.0


def check_ffprobe_available() -> str:
    """Verify that ffprobe is available on PATH and return its resolved path."""
    ffprobe_path = shutil.which("ffprobe")
    if not ffprobe_path:
        logger.error("ffprobe executable not found in system PATH.")
        raise FFprobeNotFoundError(
            "ffprobe is not installed or not in PATH. Please ensure FFmpeg/ffprobe is installed."
        )
    return ffprobe_path


# ==============================================================================
# Ingestion Logic
# ==============================================================================

def run_ffprobe(source_path: Path) -> Dict[str, Any]:
    """
    Execute ffprobe against the source file and return parsed JSON.
    """
    check_ffprobe_available()

    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(source_path),
    ]

    logger.debug("Executing ffprobe command: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except Exception as exc:
        raise InvalidMediaFileError(f"Failed to execute ffprobe: {exc}") from exc

    if result.returncode != 0:
        stderr_msg = result.stderr.strip()
        logger.error("ffprobe failed with exit code %d: %s", result.returncode, stderr_msg)
        raise InvalidMediaFileError(f"ffprobe failed to analyze file '{source_path}': {stderr_msg}")

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        logger.error("ffprobe returned invalid JSON output: %s", exc)
        raise InvalidMediaFileError(f"ffprobe returned non-JSON output: {exc}") from exc

    if not isinstance(data, dict) or "format" not in data:
        raise InvalidMediaFileError(f"ffprobe output is missing container format information for '{source_path}'")

    return data


def extract_metadata(source_path: Path, movie_id: Optional[str] = None) -> MovieMetadata:
    """
    Extract comprehensive metadata from a media file.
    """
    if not source_path.exists():
        logger.error("Source file does not exist: %s", source_path)
        raise SourceFileNotFoundError(f"Source movie file does not exist: '{source_path}'")

    if not source_path.is_file():
        logger.error("Source path is not a regular file: %s", source_path)
        raise SourceFileNotFoundError(f"Source path is not a file: '{source_path}'")

    resolved_path = source_path.resolve()
    assigned_movie_id = movie_id or sanitize_movie_id(resolved_path.name)

    # Gather filesystem info
    stat_info = resolved_path.stat()
    file_size_bytes = stat_info.st_size
    file_size_human = format_file_size(file_size_bytes)
    file_extension = resolved_path.suffix.lower()

    # Dates from filesystem
    file_created_at = datetime.datetime.fromtimestamp(
        stat_info.st_ctime, tz=datetime.timezone.utc
    ).isoformat()
    file_modified_at = datetime.datetime.fromtimestamp(
        stat_info.st_mtime, tz=datetime.timezone.utc
    ).isoformat()

    # Run ffprobe
    probe_data = run_ffprobe(resolved_path)
    fmt = probe_data.get("format", {})
    streams = probe_data.get("streams", [])

    container_format = fmt.get("format_name", "unknown")
    
    # Duration
    duration_str = fmt.get("duration")
    duration_seconds = round(float(duration_str), 3) if duration_str else 0.0

    # Categorize streams
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    subtitle_streams = [s for s in streams if s.get("codec_type") == "subtitle"]

    # Video stream properties
    primary_video = video_streams[0] if video_streams else None
    width = int(primary_video.get("width", 0)) if primary_video else 0
    height = int(primary_video.get("height", 0)) if primary_video else 0
    resolution = f"{width}x{height}" if width and height else "unknown"
    video_codec = primary_video.get("codec_name") if primary_video else None

    # FPS extraction: try r_frame_rate, fallback to avg_frame_rate
    fps = 0.0
    if primary_video:
        fps = parse_fps(primary_video.get("r_frame_rate"))
        if fps == 0.0:
            fps = parse_fps(primary_video.get("avg_frame_rate"))

    # Audio stream properties
    primary_audio = audio_streams[0] if audio_streams else None
    audio_codec = primary_audio.get("codec_name") if primary_audio else None
    audio_channel_count = int(primary_audio.get("channels", 0)) if primary_audio else 0
    audio_sample_rate = int(primary_audio.get("sample_rate")) if primary_audio and primary_audio.get("sample_rate") else None

    # Subtitles extraction
    subtitle_languages: List[str] = []
    subtitle_tracks: List[Dict[str, Any]] = []

    for sub in subtitle_streams:
        tags = sub.get("tags", {})
        lang = tags.get("language", "und")
        if lang not in subtitle_languages and lang != "und":
            subtitle_languages.append(lang)

        disposition = sub.get("disposition", {})
        track = SubtitleTrack(
            stream_index=int(sub.get("index", 0)),
            codec_name=sub.get("codec_name", "unknown"),
            language=lang,
            title=tags.get("title"),
            is_default=bool(disposition.get("default", 0)),
            is_forced=bool(disposition.get("forced", 0)),
        )
        subtitle_tracks.append(asdict(track))

    # Media container creation timestamp
    format_tags = fmt.get("tags", {})
    media_created_at = format_tags.get("creation_time")

    ingested_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    return MovieMetadata(
        movie_id=assigned_movie_id,
        source_filename=resolved_path.name,
        absolute_source_path=str(resolved_path),
        file_extension=file_extension,
        file_size_bytes=file_size_bytes,
        file_size_human=file_size_human,
        container_format=container_format,
        duration_seconds=duration_seconds,
        width=width,
        height=height,
        resolution=resolution,
        fps=fps,
        video_codec=video_codec,
        audio_codec=audio_codec,
        audio_channel_count=audio_channel_count,
        audio_sample_rate=audio_sample_rate,
        number_of_video_streams=len(video_streams),
        number_of_audio_streams=len(audio_streams),
        number_of_subtitle_streams=len(subtitle_streams),
        subtitle_languages=subtitle_languages,
        subtitle_tracks=subtitle_tracks,
        file_created_at=file_created_at,
        file_modified_at=file_modified_at,
        media_created_at=media_created_at,
        ingested_at=ingested_at,
        ffprobe_raw_format_tags=format_tags,
    )


def save_metadata(metadata: MovieMetadata, output_base_dir: Optional[Path] = None) -> Path:
    """
    Persist metadata to analysis/<movie_id>/movie_metadata.json
    """
    if output_base_dir is None:
        # Default to analysis folder in project root
        workspace_root = Path(__file__).resolve().parents[1]
        output_base_dir = workspace_root / "analysis"

    target_dir = output_base_dir / metadata.movie_id
    target_file = target_dir / "movie_metadata.json"

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        with open(target_file, "w", encoding="utf-8") as f:
            f.write(metadata.to_json(indent=2))
        logger.info("Successfully wrote movie metadata to: %s", target_file)
    except Exception as exc:
        logger.error("Failed to write metadata JSON to %s: %s", target_file, exc)
        raise MetadataWriteError(f"Could not write metadata JSON to '{target_file}': {exc}") from exc

    return target_file


def ingest_movie(
    source_path: str | Path,
    movie_id: Optional[str] = None,
    output_base_dir: Optional[Path] = None,
) -> MovieMetadata:
    """
    High-level ingestion pipeline:
    1. Validates and inspects media file with ffprobe.
    2. Constructs MovieMetadata.
    3. Saves movie_metadata.json into analysis/<movie_id>/.
    4. Returns the MovieMetadata instance.
    """
    path_obj = Path(source_path)
    logger.info("Starting ingestion for movie: %s", path_obj)
    
    metadata = extract_metadata(path_obj, movie_id=movie_id)
    out_file = save_metadata(metadata, output_base_dir=output_base_dir)
    logger.info("Ingestion completed successfully for movie ID: %s (JSON: %s)", metadata.movie_id, out_file)
    
    return metadata


# ==============================================================================
# CLI Interface
# ==============================================================================

def main() -> int:
    """Command-line entry point for movie ingestion."""
    parser = argparse.ArgumentParser(
        description="Movie Review Engine - Phase 1: Ingest movie file and extract metadata."
    )
    parser.add_argument(
        "movie_path",
        type=str,
        help="Path to the source movie file (e.g. 'movie.mp4' or full path).",
    )
    parser.add_argument(
        "--movie-id",
        type=str,
        default=None,
        help="Optional custom movie_id identifier for directory naming.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional override base analysis directory.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable detailed debug logging.",
    )

    args = parser.parse_args()

    # Configure logging format
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    out_base = Path(args.output_dir) if args.output_dir else None

    try:
        metadata = ingest_movie(
            source_path=args.movie_path,
            movie_id=args.movie_id,
            output_base_dir=out_base,
        )
        print("\n" + "=" * 60)
        print(f"[SUCCESS] MOVIE INGESTION COMPLETED: {metadata.movie_id}")
        print("=" * 60)
        print(f"  Source File : {metadata.source_filename}")
        print(f"  Source Path : {metadata.absolute_source_path}")
        print(f"  Duration    : {metadata.duration_seconds}s")
        print(f"  Resolution  : {metadata.resolution} @ {metadata.fps} FPS")
        print(f"  Codecs      : Video [{metadata.video_codec}], Audio [{metadata.audio_codec}]")
        print(f"  Channels/Hz : {metadata.audio_channel_count} ch @ {metadata.audio_sample_rate} Hz")
        print(f"  Streams     : {metadata.number_of_video_streams} Video, {metadata.number_of_audio_streams} Audio, {metadata.number_of_subtitle_streams} Subtitle")
        print(f"  Subtitles   : {metadata.subtitle_languages}")
        print(f"  File Size   : {metadata.file_size_human} ({metadata.file_size_bytes:,} bytes)")
        print("=" * 60 + "\n")
        return 0
    except MovieIngestError as exc:
        print(f"\n[ERROR] INGESTION FAILED: {exc}\n", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected error during movie ingestion")
        print(f"\n[ERROR] UNEXPECTED FAILURE: {exc}\n", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
