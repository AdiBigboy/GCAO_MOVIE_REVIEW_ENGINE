"""
Movie Review Engine - Phase 3: Subtitle Parsing, Cleaning, and Extraction Helpers
Supports SRT, WebVTT, ASS/SSA with robust timestamp parsing, tag cleanup, and normalization.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("movie_analyzer.subtitles")


# ==============================================================================
# Data Models
# ==============================================================================

@dataclass
class DialogueEntry:
    """Represents a single dialogue line with normalized timestamps and clean text."""
    index: int
    start_seconds: float
    end_seconds: float
    start: str
    end: str
    duration_seconds: float
    text: str
    speaker: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if data["speaker"] is None:
            data.pop("speaker", None)
        return data


# ==============================================================================
# Timestamp & Text Cleaning Helpers
# ==============================================================================

def format_timestamp(seconds: float) -> str:
    """Format floating point seconds into 'HH:MM:SS.mmm'."""
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


def parse_timestamp_to_seconds(ts_str: str) -> float:
    """
    Parse timestamp string into float seconds.
    Supports:
      - 00:01:23,456 (SRT)
      - 00:01:23.456 (VTT)
      - 01:23.456 (Short VTT)
      - 1:02:03.45 (ASS/SSA)
    """
    ts = ts_str.strip().replace(",", ".")
    # Remove any trailing positioning or alignment parameters in VTT (e.g. 'align:start')
    ts = ts.split()[0]
    parts = ts.split(":")
    try:
        if len(parts) == 3:
            h, m, s = parts
            return round(float(h) * 3600 + float(m) * 60 + float(s), 3)
        elif len(parts) == 2:
            m, s = parts
            return round(float(m) * 60 + float(s), 3)
        elif len(parts) == 1:
            return round(float(parts[0]), 3)
    except Exception as exc:
        logger.warning("Failed to parse timestamp '%s': %s", ts_str, exc)
    return 0.0


def clean_subtitle_text(raw_text: str) -> str:
    """
    Clean formatting tags, line breaks, and styling markers from subtitle text.
    - Preserves actual spoken dialogue and punctuation.
    - Strips HTML/VTT tags like <b>, <i>, <font>, <c.yellow>, <v Speaker>.
    - Strips ASS/SSA override tags like {\\an8}, {\\pos(x,y)}, {\\b1}.
    - Converts \\N and newlines into natural spaces.
    """
    if not raw_text:
        return ""

    # Replace ASS / SSA explicit linebreaks
    text = re.sub(r"\\N|\\n", " ", raw_text)
    # Remove ASS style blocks {\...}
    text = re.sub(r"\{[^}]*\}", "", text)
    # Remove HTML/XML/VTT tags <...>
    text = re.sub(r"<[^>]+>", "", text)
    # Remove WebVTT cue position markers if embedded in text
    text = re.sub(r"align:\w+|position:\S+|size:\S+|line:\S+", "", text)
    # Replace all newlines and carriage returns with spaces
    text = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    # Collapse multiple consecutive whitespace characters
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_vtt_speaker(raw_text: str) -> Tuple[str, Optional[str]]:
    """Extract speaker from WebVTT <v SpeakerName> tag if present."""
    match = re.search(r"<v(?:\.[\w-]+)?\s+([^>]+)>", raw_text)
    speaker = match.group(1).strip() if match else None
    cleaned = clean_subtitle_text(raw_text)
    return cleaned, speaker


# ==============================================================================
# Subtitle Format Parsers
# ==============================================================================

def parse_srt_content(content: str) -> List[DialogueEntry]:
    """Parse SubRip (.srt) subtitle string content."""
    entries: List[DialogueEntry] = []
    # Normalize line endings
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    # Split on double newlines
    blocks = re.split(r"\n\s*\n", normalized.strip())

    idx = 1
    for block in blocks:
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        if not lines:
            continue

        # Find line with timestamp arrow -->
        arrow_idx = -1
        for i, line in enumerate(lines):
            if "-->" in line:
                arrow_idx = i
                break

        if arrow_idx == -1:
            continue

        timing_line = lines[arrow_idx]
        timing_parts = timing_line.split("-->")
        if len(timing_parts) != 2:
            continue

        start_sec = parse_timestamp_to_seconds(timing_parts[0])
        end_sec = parse_timestamp_to_seconds(timing_parts[1])

        if end_sec < start_sec:
            logger.warning("SRT entry has end time < start time: %s", timing_line)
            continue

        text_lines = lines[arrow_idx + 1:]
        raw_text = " ".join(text_lines)
        clean_text = clean_subtitle_text(raw_text)

        if not clean_text:
            continue

        duration = round(end_sec - start_sec, 3)
        entries.append(
            DialogueEntry(
                index=idx,
                start_seconds=start_sec,
                end_seconds=end_sec,
                start=format_timestamp(start_sec),
                end=format_timestamp(end_sec),
                duration_seconds=duration,
                text=clean_text,
            )
        )
        idx += 1

    return entries


def parse_vtt_content(content: str) -> List[DialogueEntry]:
    """Parse WebVTT (.vtt) subtitle string content."""
    entries: List[DialogueEntry] = []
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", normalized.strip())

    idx = 1
    for block in blocks:
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        if not lines:
            continue

        # Skip WebVTT header or NOTE blocks
        if lines[0].startswith("WEBVTT") or lines[0].startswith("NOTE") or lines[0].startswith("STYLE"):
            continue

        # Find timestamp line
        arrow_idx = -1
        for i, line in enumerate(lines):
            if "-->" in line:
                arrow_idx = i
                break

        if arrow_idx == -1:
            continue

        timing_line = lines[arrow_idx]
        timing_parts = timing_line.split("-->")
        if len(timing_parts) != 2:
            continue

        start_sec = parse_timestamp_to_seconds(timing_parts[0])
        end_sec = parse_timestamp_to_seconds(timing_parts[1])

        if end_sec < start_sec:
            continue

        text_lines = lines[arrow_idx + 1:]
        raw_text = " ".join(text_lines)
        clean_text, speaker = extract_vtt_speaker(raw_text)

        if not clean_text:
            continue

        duration = round(end_sec - start_sec, 3)
        entries.append(
            DialogueEntry(
                index=idx,
                start_seconds=start_sec,
                end_seconds=end_sec,
                start=format_timestamp(start_sec),
                end=format_timestamp(end_sec),
                duration_seconds=duration,
                text=clean_text,
                speaker=speaker,
            )
        )
        idx += 1

    return entries


def parse_ass_content(content: str) -> List[DialogueEntry]:
    """Parse Advanced SubStation Alpha (.ass / .ssa) subtitle string content."""
    entries: List[DialogueEntry] = []
    lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    in_events = False
    format_fields: List[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue

        if stripped.lower() == "[events]":
            in_events = True
            continue
        elif stripped.startswith("[") and stripped.endswith("]"):
            in_events = False
            continue

        if in_events:
            if stripped.lower().startswith("format:"):
                fields_str = stripped.split(":", 1)[1]
                format_fields = [f.strip().lower() for f in fields_str.split(",")]
                continue

            if stripped.lower().startswith("dialogue:"):
                dialogue_data = stripped.split(":", 1)[1].strip()
                # Split with maxsplit based on number of fields in format line
                if format_fields and "text" in format_fields:
                    num_fields = len(format_fields)
                    parts = dialogue_data.split(",", num_fields - 1)
                    if len(parts) != num_fields:
                        continue

                    field_map = dict(zip(format_fields, parts))
                    start_str = field_map.get("start", "")
                    end_str = field_map.get("end", "")
                    raw_text = field_map.get("text", "")
                    name_field = field_map.get("name", "").strip() or None
                else:
                    # Default ASS format assumption: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
                    parts = dialogue_data.split(",", 9)
                    if len(parts) < 10:
                        continue
                    start_str = parts[1]
                    end_str = parts[2]
                    name_field = parts[4].strip() or None
                    raw_text = parts[9]

                start_sec = parse_timestamp_to_seconds(start_str)
                end_sec = parse_timestamp_to_seconds(end_str)

                if end_sec < start_sec:
                    continue

                clean_text = clean_subtitle_text(raw_text)
                if not clean_text:
                    continue

                duration = round(end_sec - start_sec, 3)
                entries.append(
                    DialogueEntry(
                        index=len(entries) + 1,
                        start_seconds=start_sec,
                        end_seconds=end_sec,
                        start=format_timestamp(start_sec),
                        end=format_timestamp(end_sec),
                        duration_seconds=duration,
                        text=clean_text,
                        speaker=name_field,
                    )
                )

    # Sort chronologically by start_seconds
    entries.sort(key=lambda e: e.start_seconds)
    # Re-index
    for i, entry in enumerate(entries, 1):
        entry.index = i

    return entries


def parse_subtitle_file(file_path: Path) -> List[DialogueEntry]:
    """Parse an external subtitle file based on its extension."""
    if not file_path.exists():
        raise FileNotFoundError(f"Subtitle file not found: {file_path}")

    # Read with utf-8 encoding and fallback to cp1252/latin-1
    content = ""
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            with open(file_path, "r", encoding=enc) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue

    if not content:
        raise ValueError(f"Could not decode subtitle file: {file_path}")

    ext = file_path.suffix.lower()
    if ext == ".srt":
        return parse_srt_content(content)
    elif ext == ".vtt":
        return parse_vtt_content(content)
    elif ext in (".ass", ".ssa"):
        return parse_ass_content(content)
    else:
        # Fallback heuristic: check if content contains WEBVTT or [Events]
        if "WEBVTT" in content[:200]:
            return parse_vtt_content(content)
        elif "[events]" in content.lower():
            return parse_ass_content(content)
        else:
            # Default to SRT parser
            return parse_srt_content(content)


# ==============================================================================
# FFmpeg Subtitle Stream Extraction & Discovery
# ==============================================================================

def extract_embedded_subtitle_stream(
    source_video: Path,
    stream_index: int,
    output_subtitle_path: Path,
) -> Path:
    """
    Extract embedded subtitle stream from video file using FFmpeg.
    """
    output_subtitle_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(source_video),
        "-map", f"0:{stream_index}",
        str(output_subtitle_path),
    ]

    logger.debug("Running FFmpeg subtitle extraction: %s", " ".join(cmd))
    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if res.returncode != 0 or not output_subtitle_path.exists() or output_subtitle_path.stat().st_size == 0:
        logger.error("FFmpeg subtitle extraction failed: %s", res.stderr)
        raise RuntimeError(f"FFmpeg failed to extract subtitle stream {stream_index}: {res.stderr.strip()}")

    return output_subtitle_path


def find_external_subtitles(source_video: Path, movie_id: Optional[str] = None) -> List[Path]:
    """
    Search near source movie and input/subtitles for candidate subtitle files.
    """
    candidates: List[Path] = []
    parent_dir = source_video.parent
    stem = source_video.stem

    if parent_dir.exists() and parent_dir.is_dir():
        # Match exact stem or stem.lang.ext
        for ext in (".srt", ".vtt", ".ass", ".ssa"):
            exact = parent_dir / f"{stem}{ext}"
            if exact.exists() and exact.is_file():
                candidates.append(exact)

            # Match language tagged variants e.g. stem.en.srt, stem.eng.srt
            for match in parent_dir.glob(f"{stem}.*{ext}"):
                if match.is_file() and match not in candidates:
                    candidates.append(match)

    # Also check project input/subtitles if exists
    workspace_root = Path(__file__).resolve().parents[1]
    input_sub_dir = workspace_root / "input" / "subtitles"
    if input_sub_dir.exists():
        search_stems = [stem]
        if movie_id:
            search_stems.append(movie_id)
        for s in search_stems:
            for ext in (".srt", ".vtt", ".ass", ".ssa"):
                for match in input_sub_dir.glob(f"{s}*{ext}"):
                    if match.is_file() and match not in candidates:
                        candidates.append(match)

    return candidates


def detect_subtitle_language_from_filename(file_path: Path) -> str:
    """Extract language tag from filename (e.g. 'movie.en.srt' -> 'en', 'movie.eng.vtt' -> 'eng')."""
    stem = file_path.stem  # e.g. 'movie.en'
    parts = stem.split(".")
    if len(parts) >= 2:
        candidate_lang = parts[-1].lower()
        if len(candidate_lang) in (2, 3) and candidate_lang.isalpha():
            return candidate_lang
    return "und"
