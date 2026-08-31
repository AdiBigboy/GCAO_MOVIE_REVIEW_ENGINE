"""
Movie Review Engine - Phase 10: Rough Video Preview Renderer
Renders playable 1920x1080 @ 30fps rough review preview video (rough_preview.mp4)
with normalized aspect ratios, visual placeholders, and Malay narration captions.
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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config.paths import ANALYSIS_DIR, MOVIES_SOURCE_DIR, PROJECT_ROOT, get_movie_source_path
from movie_analyzer.ingest import sanitize_movie_id
from preview_engine.models import PreviewManifest, SegmentTimeline, TimelineItem
from preview_engine.timeline_builder import PreviewTimelineBuilder

logger = logging.getLogger("preview_engine.render_preview")


class VideoRenderError(Exception):
    """Base exception for video rendering errors."""
    pass


def _format_srt_time(seconds: float) -> str:
    """Format seconds into SRT timestamp format: HH:MM:SS,mmm."""
    total_ms = int(round(seconds * 1000.0))
    hours = total_ms // 3600000
    total_ms %= 3600000
    minutes = total_ms // 60000
    total_ms %= 60000
    secs = total_ms // 1000
    ms = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _wrap_caption_lines(text: str, max_chars_per_line: int = 55) -> str:
    """Wrap long caption text into clean 2-line subtitles."""
    words = text.split()
    lines: List[str] = []
    current_line: List[str] = []
    current_len = 0

    for w in words:
        if current_len + len(w) + 1 > max_chars_per_line and current_line:
            lines.append(" ".join(current_line))
            current_line = [w]
            current_len = len(w)
        else:
            current_line.append(w)
            current_len += len(w) + 1

    if current_line:
        lines.append(" ".join(current_line))

    # Keep to max 2 lines for clarity
    if len(lines) > 2:
        mid = len(words) // 2
        line1 = " ".join(words[:mid])
        line2 = " ".join(words[mid:])
        return f"{line1}\n{line2}"

    return "\n".join(lines)

def _split_into_caption_chunks(text: str, max_words: int = 15) -> List[str]:
    """Split segment text into short, natural phrase-level caption chunks."""
    # Split on sentence terminals first
    sentences = re.split(r'(?<=[.?!])\s+', text.strip())
    chunks: List[str] = []

    for s in sentences:
        s = s.strip()
        if not s:
            continue
        words = s.split()
        if len(words) <= max_words:
            chunks.append(s)
        else:
            # Subdivide by commas or midpoints
            parts = re.split(r'(?<=[,\-])\s+', s)
            curr: List[str] = []
            curr_len = 0
            for p in parts:
                p_words = p.split()
                if curr_len + len(p_words) > max_words and curr:
                    chunks.append(" ".join(curr))
                    curr = [p]
                    curr_len = len(p_words)
                else:
                    curr.append(p)
                    curr_len += len(p_words)
            if curr:
                chunks.append(" ".join(curr))

    return chunks if chunks else [text]


class RoughPreviewRenderer:
    """
    Renders 1920x1080 rough video preview from PreviewManifest.
    """

    def __init__(
        self,
        movie_id: str,
        analysis_dir: Path,
        movie_source_path: Optional[Path] = None,
    ):
        self.movie_id = movie_id
        self.analysis_dir = analysis_dir
        self.movie_source_path = movie_source_path

        self.preview_dir = analysis_dir / "preview"
        self.chunks_dir = self.preview_dir / "chunks"
        self.output_video_file = self.preview_dir / "rough_preview.mp4"
        self.output_video_v2_file = self.preview_dir / "rough_preview_v2.mp4"
        self.output_srt_file = self.preview_dir / "captions.srt"
        self.manifest_file = self.preview_dir / "preview_manifest.json"

    def generate_srt_captions(self, manifest: PreviewManifest) -> Path:
        """Create phrase-level SRT subtitle file mapping narration segments into timed 1-2 line chunks."""
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        srt_entries: List[str] = []

        sub_idx = 1
        for seg in manifest.segments:
            seg_start = seg.start_time_seconds
            seg_end = seg.end_time_seconds
            seg_dur = max(1.0, seg_end - seg_start)

            chunks = _split_into_caption_chunks(seg.narration_text, max_words=14)
            total_words = sum(len(c.split()) for c in chunks)
            if total_words == 0:
                total_words = 1

            t_cursor = seg_start
            for c_idx, chunk_text in enumerate(chunks):
                chunk_word_count = len(chunk_text.split())
                chunk_fraction = chunk_word_count / total_words
                chunk_dur = seg_dur * chunk_fraction

                c_start = t_cursor
                c_end = seg_end if c_idx == len(chunks) - 1 else round(t_cursor + chunk_dur, 2)
                t_cursor = c_end

                start_str = _format_srt_time(c_start)
                end_str = _format_srt_time(c_end)
                wrapped_text = _wrap_caption_lines(chunk_text, max_chars_per_line=45)

                srt_entries.append(f"{sub_idx}\n{start_str} --> {end_str}\n{wrapped_text}\n")
                sub_idx += 1

        with open(self.output_srt_file, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_entries))

        logger.info("Saved preview captions to %s", self.output_srt_file)
        return self.output_srt_file

    def _render_placeholder_chunk(self, item: TimelineItem, out_path: Path, ref_clip_path: Optional[Path] = None) -> Path:
        """Generate a clean still freeze/hold placeholder with silent audio (no text overlays, no dialogue loops)."""
        dur = max(0.5, item.duration_seconds)

        # If a reference source clip is available, create a clean still frame hold with source aspect ratio
        if ref_clip_path and ref_clip_path.exists() and ref_clip_path.stat().st_size > 0:
            still_img = self.chunks_dir / f"{item.item_id}_still.jpg"
            cmd_extract = [
                "ffmpeg", "-y",
                "-ss", "0.5",
                "-i", str(ref_clip_path),
                "-vframes", "1",
                "-q:v", "2",
                str(still_img),
            ]
            subprocess.run(cmd_extract, capture_output=True)

            if still_img.exists() and still_img.stat().st_size > 0:
                # Clean freeze still hold with silent AAC audio track (no audio loops)
                cmd_still = [
                    "ffmpeg", "-y",
                    "-loop", "1",
                    "-i", str(still_img),
                    "-f", "lavfi",
                    "-i", "anullsrc=r=44100:cl=stereo",
                    "-t", f"{dur:.2f}",
                    "-vf",
                    "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-r", "30",
                    "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                    "-shortest",
                    str(out_path),
                ]
                res_still = subprocess.run(cmd_still, capture_output=True)
                if res_still.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
                    return out_path

        # Clean dark cinematic placeholder with silent AAC audio
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "lavfi",
            "-i", f"color=c=0x0d0d12:s=1920x1080:d={dur:.2f}:r=30",
            "-f", "lavfi",
            "-i", "anullsrc=r=44100:cl=stereo",
            "-t", f"{dur:.2f}",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            "-r", "30",
            "-c:a", "aac",
            "-b:a", "128k",
            "-ar", "44100",
            "-ac", "2",
            "-shortest",
            str(out_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
            cmd_fallback = [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", f"color=c=black:s=1920x1080:d={dur:.2f}:r=30",
                "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                "-t", f"{dur:.2f}",
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-r", "30",
                "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                "-shortest",
                str(out_path),
            ]
            subprocess.run(cmd_fallback, capture_output=True, check=True)

        return out_path

    def _render_source_clip_chunk(self, item: TimelineItem, out_path: Path) -> Path:
        """Normalize source clip to 1920x1080 @ 30fps preserving original audio and aspect ratio."""
        clip_file = Path(item.clip_path) if item.clip_path else None
        if not clip_file or not clip_file.exists() or clip_file.stat().st_size == 0:
            # Fallback to placeholder if clip file is missing
            return self._render_placeholder_chunk(item, out_path)

        dur = max(0.5, item.duration_seconds)
        # Attempt direct normalization with original audio stream
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(clip_file),
            "-t", str(dur),
            "-vf",
            "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            "-r", "30",
            "-c:a", "aac",
            "-b:a", "128k",
            "-ar", "44100",
            "-ac", "2",
            str(out_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
            # If source clip had no audio stream, add silent audio track for consistent stream layout
            cmd_silent_audio = [
                "ffmpeg",
                "-y",
                "-i", str(clip_file),
                "-f", "lavfi",
                "-i", "anullsrc=r=44100:cl=stereo",
                "-t", str(dur),
                "-vf",
                "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-pix_fmt", "yuv420p",
                "-r", "30",
                "-c:a", "aac",
                "-b:a", "128k",
                "-ar", "44100",
                "-ac", "2",
                "-shortest",
                str(out_path),
            ]
            subprocess.run(cmd_silent_audio, capture_output=True)
            if not out_path.exists() or out_path.stat().st_size == 0:
                return self._render_placeholder_chunk(item, out_path, ref_clip_path=clip_file)

        return out_path

    def render(self, manifest: Optional[PreviewManifest] = None) -> Path:
        """
        Execute full clean render pipeline: chunk normalization, original audio preservation,
        and clean concatenation into rough_preview_v3.mp4.
        """
        if manifest is None:
            builder = PreviewTimelineBuilder(
                movie_id=self.movie_id,
                analysis_dir=self.analysis_dir,
            )
            manifest = builder.build_timeline()
            builder.save_manifest(manifest)

        self.preview_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)

        # 1. Generate standalone SRT sidecar captions for inspection
        self.generate_srt_captions(manifest)

        # 2. Render all individual normalized chunks
        chunk_files: List[Path] = []
        all_items: List[TimelineItem] = [it for seg in manifest.segments for it in seg.items]

        for idx, it in enumerate(all_items):
            chunk_file = self.chunks_dir / f"{it.item_id}_{it.item_type}.mp4"
            if it.item_type == "SOURCE_CLIP":
                self._render_source_clip_chunk(it, chunk_file)
            else:
                # Find nearest source clip in timeline to use as clean freeze still background
                ref_clip: Optional[Path] = None
                if idx > 0 and all_items[idx - 1].clip_path:
                    ref_clip = Path(str(all_items[idx - 1].clip_path))
                elif idx < len(all_items) - 1 and all_items[idx + 1].clip_path:
                    ref_clip = Path(str(all_items[idx + 1].clip_path))
                self._render_placeholder_chunk(it, chunk_file, ref_clip_path=ref_clip)
            chunk_files.append(chunk_file)

        # 3. Create concat list
        concat_list_file = self.preview_dir / "concat_list.txt"
        with open(concat_list_file, "w", encoding="utf-8") as f:
            for ch in chunk_files:
                # Use forward slashes for FFmpeg concat safe path
                f.write(f"file '{ch.resolve().as_posix()}'\n")

        # 4. Concatenate chunks into final clean rough_preview_v3.mp4 (video + audio)
        output_v3 = self.preview_dir / "rough_preview_v3.mp4"
        cmd_concat = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list_file),
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "24",
            "-c:a", "aac",
            "-b:a", "128k",
            str(output_v3),
        ]
        res_concat = subprocess.run(cmd_concat, capture_output=True, text=True)
        if res_concat.returncode != 0 or not output_v3.exists():
            raise VideoRenderError(f"FFmpeg concatenation failed: {res_concat.stderr}")

        # Also create copies: rough_preview.mp4 and rough_preview_v2.mp4
        try:
            shutil.copy2(output_v3, self.output_video_file)
            shutil.copy2(output_v3, self.output_video_v2_file)
        except Exception:
            pass

        # Save V3 manifest
        manifest_v3_path = self.preview_dir / "preview_manifest_v3.json"
        with open(manifest_v3_path, "w", encoding="utf-8") as f:
            f.write(manifest.to_json(indent=2))

        logger.info("Successfully rendered clean rough preview video with original audio to %s", output_v3)
        return output_v3


# ==============================================================================
# Public API & Standalone Runner
# ==============================================================================

def build_rough_video_preview(
    source_path: str | Path,
    movie_id: Optional[str] = None,
    output_base_dir: Optional[Path] = None,
    render_video: bool = True,
    force: bool = False,
) -> Tuple[PreviewManifest, Optional[Path]]:
    """
    Public API function to construct timeline manifest and optionally render rough preview video.
    """
    assigned_movie_id = movie_id
    resolved_source: Optional[Path] = None
    try:
        resolved_source = get_movie_source_path(source_path)
        if not assigned_movie_id:
            assigned_movie_id = sanitize_movie_id(resolved_source.name)
    except Exception:
        if not assigned_movie_id:
            assigned_movie_id = sanitize_movie_id(str(source_path))

    out_base = output_base_dir or ANALYSIS_DIR
    analysis_dir = out_base / assigned_movie_id

    if not analysis_dir.exists():
        raise VideoRenderError(
            f"Analysis directory not found for movie '{assigned_movie_id}' at {analysis_dir}"
        )

    builder = PreviewTimelineBuilder(
        movie_id=assigned_movie_id,
        analysis_dir=analysis_dir,
    )
    manifest = builder.build_timeline()
    builder.save_manifest(manifest)

    rendered_video_path: Optional[Path] = None
    if render_video:
        renderer = RoughPreviewRenderer(
            movie_id=assigned_movie_id,
            analysis_dir=analysis_dir,
            movie_source_path=resolved_source,
        )
        rendered_video_path = renderer.render(manifest)

    return manifest, rendered_video_path


def main():
    """CLI entrypoint: python -m preview_engine.render_preview <movie_id>."""
    parser = argparse.ArgumentParser(
        description="Phase 10: Rough Video Preview Builder (temporal timeline + captions)."
    )
    parser.add_argument(
        "source",
        type=str,
        help="Path or name of the movie source file (or movie ID).",
    )
    parser.add_argument(
        "--movie-id",
        type=str,
        default=None,
        help="Optional custom movie identifier.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom base output directory (default: analysis/).",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Only build preview manifest without invoking FFmpeg video rendering.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    out_base = Path(args.output_dir) if args.output_dir else None
    try:
        manifest, video_path = build_rough_video_preview(
            source_path=args.source,
            movie_id=args.movie_id,
            output_base_dir=out_base,
            render_video=(not args.no_render),
        )
        print("\n" + "=" * 60)
        print("ROUGH VIDEO PREVIEW COMPLETE (Phase 10)")
        print("=" * 60)
        print(f"  Movie ID         : {manifest.movie_id}")
        print(f"  Status           : {manifest.status}")
        print(f"  Total Duration   : {manifest.total_duration_seconds:.1f}s")
        print(f"  Resolution       : {manifest.resolution} @ {manifest.fps} fps")
        print(f"  Total Segments   : {manifest.total_segments}")
        print(f"  Source Clips     : {manifest.total_source_clips} ({manifest.total_source_footage_seconds:.1f}s footage)")
        print(f"  Placeholders     : {manifest.total_placeholders} ({manifest.total_placeholder_seconds:.1f}s hold)")
        if video_path and video_path.exists():
            size_mb = video_path.stat().st_size / (1024 * 1024)
            print(f"  Preview Video    : {video_path} ({size_mb:.2f} MB)")
        print("=" * 60 + "\n")
    except Exception as exc:
        print(f"Error during preview rendering: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
