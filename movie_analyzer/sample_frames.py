"""
Movie Review Engine - Phase 2: Timeline Sampling & Frame Extraction Module
Extracts representative video frames at configurable intervals and saves timeline_index.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from movie_analyzer.ingest import (
    MovieIngestError,
    SourceFileNotFoundError,
    InvalidMediaFileError,
    FFprobeNotFoundError,
    MovieMetadata,
    extract_metadata,
    ingest_movie,
    sanitize_movie_id,
)

logger = logging.getLogger("movie_analyzer.sample_frames")


# ==============================================================================
# Exceptions
# ==============================================================================

class TimelineSamplingError(MovieIngestError):
    """Base exception for timeline sampling errors."""
    pass


class InvalidIntervalError(TimelineSamplingError):
    """Raised when the sampling interval is invalid (<= 0)."""
    pass


class FFmpegNotFoundError(TimelineSamplingError):
    """Raised when ffmpeg executable is not available on PATH."""
    pass


class FrameExtractionError(TimelineSamplingError):
    """Raised when FFmpeg frame extraction fails."""
    pass


class TimelineWriteError(TimelineSamplingError):
    """Raised when writing timeline_index.json fails."""
    pass


# ==============================================================================
# Data Models
# ==============================================================================

@dataclass
class FrameSample:
    """Represents a single extracted frame sample in the timeline."""
    index: int
    timestamp_seconds: float
    timestamp: str
    frame_file: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TimelineIndex:
    """Full timeline index of sampled frames for a movie."""
    movie_id: str
    sampling_interval_seconds: float
    duration_seconds: float
    total_samples: int
    frames: List[FrameSample] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "movie_id": self.movie_id,
            "sampling_interval_seconds": self.sampling_interval_seconds,
            "duration_seconds": self.duration_seconds,
            "total_samples": self.total_samples,
            "frames": [f.to_dict() if isinstance(f, FrameSample) else f for f in self.frames],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ==============================================================================
# Helper Functions
# ==============================================================================

def format_timestamp(seconds: float) -> str:
    """
    Format floating point seconds into standard 'HH:MM:SS.mmm' format.
    Example: 0.0 -> '00:00:00.000', 65.5 -> '00:01:05.500'
    """
    if seconds < 0:
        seconds = 0.0
    total_millis = int(round(seconds * 1000))
    millis = total_millis % 1000
    total_seconds = total_millis // 1000
    secs = total_seconds % 60
    total_minutes = total_seconds // 60
    mins = total_minutes % 60
    hours = total_minutes // 60
    return f"{hours:02d}:{mins:02d}:{secs:02d}.{millis:03d}"


def check_ffmpeg_available() -> str:
    """Verify that ffmpeg is available on PATH and return its executable path."""
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        logger.error("ffmpeg executable not found on PATH.")
        raise FFmpegNotFoundError(
            "ffmpeg is not installed or not found on PATH. Please ensure FFmpeg is installed."
        )
    return ffmpeg_path


def compute_sample_points(duration_seconds: float, interval_seconds: float) -> List[FrameSample]:
    """
    Compute list of deterministic sample timestamps and frame filenames.
    Samples start at 0.0s and increment by interval_seconds up to duration_seconds.
    """
    if interval_seconds <= 0:
        raise InvalidIntervalError(f"Sampling interval must be greater than 0, got {interval_seconds}")

    samples: List[FrameSample] = []
    
    if duration_seconds <= 0:
        # Edge case: 0s video produces 1 sample at t=0
        samples.append(
            FrameSample(
                index=1,
                timestamp_seconds=0.0,
                timestamp=format_timestamp(0.0),
                frame_file="frames/frame_000001.jpg",
            )
        )
        return samples

    current_t = 0.0
    index = 1
    # Allow a tiny epsilon to avoid floating point truncation issues
    epsilon = 1e-4

    while current_t <= (duration_seconds + epsilon):
        # Clamp to duration if slightly exceeding due to float precision
        clamped_t = round(min(current_t, duration_seconds), 3)
        sample = FrameSample(
            index=index,
            timestamp_seconds=clamped_t,
            timestamp=format_timestamp(clamped_t),
            frame_file=f"frames/frame_{index:06d}.jpg",
        )
        samples.append(sample)
        index += 1
        current_t = round(current_t + interval_seconds, 6)

    return samples


def extract_single_frame(
    source_path: Path,
    timestamp_seconds: float,
    output_frame_path: Path,
    duration_seconds: Optional[float] = None,
    jpeg_quality: int = 2,
) -> None:
    """
    Extract a single frame using fast FFmpeg seek (-ss before -i).
    Clamps seek timestamp to avoid seeking beyond media EOF and includes fallback.
    """
    check_ffmpeg_available()
    output_frame_path.parent.mkdir(parents=True, exist_ok=True)

    # Safe seeking timestamp: leave at least ~0.1s headroom from EOF
    seek_t = max(0.0, timestamp_seconds)
    if duration_seconds is not None and duration_seconds > 0:
        seek_t = min(seek_t, max(0.0, duration_seconds - 0.1))

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{seek_t:.3f}",
        "-i", str(source_path),
        "-frames:v", "1",
        "-pix_fmt", "yuvj420p",
        "-q:v", str(jpeg_quality),
        str(output_frame_path),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    if result.returncode != 0 or not output_frame_path.exists() or output_frame_path.stat().st_size == 0:
        # Fallback to accurate output seek
        fallback_cmd = [
            "ffmpeg",
            "-y",
            "-i", str(source_path),
            "-ss", f"{seek_t:.3f}",
            "-frames:v", "1",
            "-pix_fmt", "yuvj420p",
            "-q:v", str(jpeg_quality),
            str(output_frame_path),
        ]
        fb_result = subprocess.run(
            fallback_cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if fb_result.returncode != 0 or not output_frame_path.exists() or output_frame_path.stat().st_size == 0:
            logger.error("FFmpeg frame extraction failed at %ss: %s", timestamp_seconds, fb_result.stderr)
            raise FrameExtractionError(
                f"Failed to extract frame at timestamp {timestamp_seconds}s for '{source_path}': {fb_result.stderr.strip()}"
            )


def extract_frames_batch(
    source_path: Path,
    interval_seconds: float,
    frames_dir: Path,
    jpeg_quality: int = 2,
) -> None:
    """
    Extract frames in a single streaming FFmpeg pass using select filter.
    Fast and memory-bounded for complete extraction.
    """
    check_ffmpeg_available()
    frames_dir.mkdir(parents=True, exist_ok=True)

    out_pattern = frames_dir / "frame_%06d.jpg"
    filter_expr = f"select=isnan(prev_selected_t)+gte(t-prev_selected_t\\,{interval_seconds})"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(source_path),
        "-vf", filter_expr,
        "-pix_fmt", "yuvj420p",
        "-fps_mode", "vfr",
        "-q:v", str(jpeg_quality),
        str(out_pattern),
    ]

    logger.debug("Running batch frame extraction: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        logger.error("FFmpeg batch extraction failed: %s", result.stderr)
        raise FrameExtractionError(f"FFmpeg batch extraction failed on '{source_path}': {result.stderr.strip()}")


# ==============================================================================
# Main Pipeline Function
# ==============================================================================

def sample_movie_frames(
    source_path: str | Path,
    interval_seconds: float = 3.0,
    movie_id: Optional[str] = None,
    output_base_dir: Optional[Path] = None,
    force: bool = False,
    jpeg_quality: int = 2,
) -> TimelineIndex:
    """
    Execute Phase 2 timeline sampling:
    1. Validate source and ensure Phase 1 metadata exists.
    2. Compute timeline sample points.
    3. Extract frames (resuming existing valid frames unless force=True).
    4. Save analysis/<movie_id>/timeline_index.json.
    """
    resolved_source = Path(source_path).resolve()
    if not resolved_source.exists():
        logger.error("Source file does not exist: %s", resolved_source)
        raise SourceFileNotFoundError(f"Movie source file not found: '{resolved_source}'")

    if interval_seconds <= 0:
        raise InvalidIntervalError(f"Sampling interval must be positive, got {interval_seconds}")

    # Determine analysis directory
    assigned_movie_id = movie_id or sanitize_movie_id(resolved_source.name)
    if output_base_dir is None:
        workspace_root = Path(__file__).resolve().parents[1]
        output_base_dir = workspace_root / "analysis"

    movie_analysis_dir = output_base_dir / assigned_movie_id
    movie_analysis_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = movie_analysis_dir / "frames"
    timeline_index_path = movie_analysis_dir / "timeline_index.json"
    metadata_path = movie_analysis_dir / "movie_metadata.json"

    # Ingest / obtain movie metadata (Phase 1)
    if metadata_path.exists() and not force:
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta_dict = json.load(f)
            duration_seconds = float(meta_dict.get("duration_seconds", 0.0))
        except Exception:
            logger.warning("Could not read existing movie_metadata.json; re-extracting metadata.")
            meta = extract_metadata(resolved_source, movie_id=assigned_movie_id)
            duration_seconds = meta.duration_seconds
    else:
        # Run Phase 1 ingest
        meta = ingest_movie(
            source_path=resolved_source,
            movie_id=assigned_movie_id,
            output_base_dir=output_base_dir,
        )
        duration_seconds = meta.duration_seconds

    # Compute expected sample points
    sample_points = compute_sample_points(duration_seconds, interval_seconds)
    total_samples = len(sample_points)

    logger.info(
        "Movie '%s': Duration=%.2fs, Interval=%.2fs -> Total Samples=%d",
        assigned_movie_id, duration_seconds, interval_seconds, total_samples
    )

    # Check existing frames for idempotent resume
    frames_dir.mkdir(parents=True, exist_ok=True)
    missing_samples: List[FrameSample] = []

    for sample in sample_points:
        frame_abs_path = movie_analysis_dir / sample.frame_file
        if force or not frame_abs_path.exists() or frame_abs_path.stat().st_size == 0:
            missing_samples.append(sample)

    logger.info(
        "Frames status: %d / %d existing. Missing/Need generation: %d",
        total_samples - len(missing_samples), total_samples, len(missing_samples)
    )

    if missing_samples:
        if force or len(missing_samples) == total_samples:
            # Clean extraction via batch pass
            logger.info("Performing full batch frame extraction via FFmpeg...")
            extract_frames_batch(
                source_path=resolved_source,
                interval_seconds=interval_seconds,
                frames_dir=frames_dir,
                jpeg_quality=jpeg_quality,
            )

        # Verify all expected frames exist and fill any residual missing sample points
        for sample in sample_points:
            frame_abs_path = movie_analysis_dir / sample.frame_file
            if not frame_abs_path.exists() or frame_abs_path.stat().st_size == 0:
                logger.debug("Extracting targeted frame %s (t=%.3fs)...", sample.frame_file, sample.timestamp_seconds)
                extract_single_frame(
                    source_path=resolved_source,
                    timestamp_seconds=sample.timestamp_seconds,
                    output_frame_path=frame_abs_path,
                    duration_seconds=duration_seconds,
                    jpeg_quality=jpeg_quality,
                )
    else:
        logger.info("All %d frames already exist and are valid. Reusing existing frames.", total_samples)

    # Build TimelineIndex
    timeline_index = TimelineIndex(
        movie_id=assigned_movie_id,
        sampling_interval_seconds=interval_seconds,
        duration_seconds=duration_seconds,
        total_samples=total_samples,
        frames=sample_points,
    )

    # Persist timeline_index.json
    try:
        with open(timeline_index_path, "w", encoding="utf-8") as f:
            f.write(timeline_index.to_json(indent=2))
        logger.info("Successfully saved timeline index: %s", timeline_index_path)
    except Exception as exc:
        logger.error("Failed to write timeline_index.json: %s", exc)
        raise TimelineWriteError(f"Could not write '{timeline_index_path}': {exc}") from exc

    return timeline_index


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main() -> int:
    """Command-line entry point for timeline sampling."""
    parser = argparse.ArgumentParser(
        description="Movie Review Engine - Phase 2: Timeline Sampling & Frame Extraction."
    )
    parser.add_argument(
        "movie_path",
        type=str,
        help="Path to the movie file (e.g. 'movie.mp4' or full path).",
    )
    parser.add_argument(
        "--interval", "-i",
        type=float,
        default=3.0,
        help="Sampling interval in seconds (default: 3.0).",
    )
    parser.add_argument(
        "--movie-id",
        type=str,
        default=None,
        help="Optional custom movie_id identifier.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional override base analysis directory.",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force rebuild/re-extract all frames even if existing.",
    )
    parser.add_argument(
        "--quality", "-q",
        type=int,
        default=2,
        help="JPEG quality scale for FFmpeg (1-31, 2 = high quality, default: 2).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose debug logging.",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    out_base = Path(args.output_dir) if args.output_dir else None

    try:
        timeline = sample_movie_frames(
            source_path=args.movie_path,
            interval_seconds=args.interval,
            movie_id=args.movie_id,
            output_base_dir=out_base,
            force=args.force,
            jpeg_quality=args.quality,
        )

        print("\n" + "=" * 60)
        print(f"[SUCCESS] TIMELINE SAMPLING COMPLETED: {timeline.movie_id}")
        print("=" * 60)
        print(f"  Duration         : {timeline.duration_seconds}s")
        print(f"  Sample Interval  : {timeline.sampling_interval_seconds}s")
        print(f"  Total Frames     : {timeline.total_samples}")
        if timeline.frames:
            print(f"  First Frame      : {timeline.frames[0].frame_file} @ {timeline.frames[0].timestamp}")
            print(f"  Last Frame       : {timeline.frames[-1].frame_file} @ {timeline.frames[-1].timestamp}")
        print("=" * 60 + "\n")
        return 0
    except TimelineSamplingError as exc:
        print(f"\n[ERROR] SAMPLING FAILED: {exc}\n", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected failure during timeline sampling")
        print(f"\n[ERROR] UNEXPECTED FAILURE: {exc}\n", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
