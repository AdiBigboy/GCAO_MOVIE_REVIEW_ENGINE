"""
Movie Review Engine - Phase 3: Subtitle & Dialogue Extraction Module
Detects, extracts, normalizes, and indexes embedded or external movie dialogue into dialogue.json.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import shutil
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
from movie_analyzer.subtitles import (
    DialogueEntry,
    parse_subtitle_file,
    extract_embedded_subtitle_stream,
    find_external_subtitles,
    detect_subtitle_language_from_filename,
)

logger = logging.getLogger("movie_analyzer.extract_dialogue")


# Supported text and image subtitle codecs
TEXT_SUBTITLE_CODECS = {
    "subrip", "srt", "webvtt", "vtt", "ass", "ssa", "mov_text", "text"
}
IMAGE_SUBTITLE_CODECS = {
    "hdmv_pgs_subtitle", "pgssub", "dvd_subtitle", "vobsub", "dvb_subtitle", "xsub"
}
DEFAULT_PREFERRED_LANGUAGES = ["eng", "en", "msa", "ms", "zho", "zh"]


# ==============================================================================
# Exceptions
# ==============================================================================

class DialogueExtractionError(MovieIngestError):
    """Base exception for dialogue extraction failures."""
    pass


class SubtitleFileNotFoundError(DialogueExtractionError):
    """Raised when an explicitly supplied subtitle file does not exist."""
    pass


class DialogueWriteError(DialogueExtractionError):
    """Raised when saving dialogue.json fails."""
    pass


# ==============================================================================
# Data Models
# ==============================================================================

@dataclass
class DialogueDocument:
    """Standardized representation of extracted movie dialogue timeline."""
    movie_id: str
    source_type: str  # 'embedded_subtitle', 'external_subtitle', 'no_dialogue'
    source_file: Optional[str]
    language: str
    status: str  # 'SUCCESS', 'NO_DIALOGUE_SOURCE_AVAILABLE', 'IMAGE_SUBTITLE_UNSUPPORTED_IN_V1'
    total_entries: int
    entries: List[DialogueEntry] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "movie_id": self.movie_id,
            "source_type": self.source_type,
            "source_file": self.source_file,
            "language": self.language,
            "status": self.status,
            "total_entries": self.total_entries,
            "entries": [e.to_dict() if isinstance(e, DialogueEntry) else e for e in self.entries],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ==============================================================================
# Dialogue Extraction Engine
# ==============================================================================

def select_embedded_subtitle_stream(
    subtitle_tracks: List[Dict[str, Any]],
    requested_stream_index: Optional[int] = None,
    requested_language: Optional[str] = None,
    preferred_languages: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Select the best matching embedded text subtitle track.
    Priority:
    1. Explicit requested_stream_index
    2. Explicit requested_language
    3. Default disposition (is_default == True)
    4. Preferred language list (e.g. eng, en)
    5. First available text subtitle stream
    """
    if not subtitle_tracks:
        return None

    # 1. Match explicit stream index
    if requested_stream_index is not None:
        for track in subtitle_tracks:
            if track.get("stream_index") == requested_stream_index:
                return track
        logger.warning("Requested stream index %d not found in subtitle tracks.", requested_stream_index)
        return None

    # Filter to text-based codecs
    text_tracks = [
        t for t in subtitle_tracks
        if t.get("codec_name", "").lower() in TEXT_SUBTITLE_CODECS
    ]
    if not text_tracks:
        return None

    # 2. Match explicit language
    if requested_language:
        req_clean = requested_language.strip().lower()
        for track in text_tracks:
            lang = track.get("language", "").lower()
            if lang == req_clean or lang.startswith(req_clean):
                return track

    # 3. Default disposition
    for track in text_tracks:
        if track.get("is_default"):
            return track

    # 4. Preferred language list
    prefs = preferred_languages or DEFAULT_PREFERRED_LANGUAGES
    for pref in prefs:
        pref_clean = pref.strip().lower()
        for track in text_tracks:
            lang = track.get("language", "").lower()
            if lang == pref_clean or lang.startswith(pref_clean):
                return track

    # 5. First usable text stream
    return text_tracks[0]


def extract_movie_dialogue(
    source_path: str | Path,
    subtitle_file_path: Optional[str | Path] = None,
    subtitle_stream_index: Optional[int] = None,
    language: Optional[str] = None,
    movie_id: Optional[str] = None,
    output_base_dir: Optional[Path] = None,
    force: bool = False,
) -> DialogueDocument:
    """
    Main Phase 3 Pipeline:
    1. Validates movie source and analysis directory.
    2. Idempotent check (reuses valid existing dialogue.json unless force=True).
    3. Detects embedded subtitle streams or discovers external subtitle files.
    4. Parses, cleans, and normalizes subtitles into dialogue.json.
    """
    resolved_source = Path(source_path).resolve()
    if not resolved_source.exists():
        logger.error("Source movie file not found: %s", resolved_source)
        raise SourceFileNotFoundError(f"Source movie file not found: '{resolved_source}'")

    assigned_movie_id = movie_id or sanitize_movie_id(resolved_source.name)
    if output_base_dir is None:
        workspace_root = Path(__file__).resolve().parents[1]
        output_base_dir = workspace_root / "analysis"

    movie_analysis_dir = output_base_dir / assigned_movie_id
    movie_analysis_dir.mkdir(parents=True, exist_ok=True)
    dialogue_json_path = movie_analysis_dir / "dialogue.json"
    subtitles_dir = movie_analysis_dir / "subtitles"
    subtitles_dir.mkdir(parents=True, exist_ok=True)

    # Idempotent reuse check
    if dialogue_json_path.exists() and not force:
        try:
            with open(dialogue_json_path, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
            if "total_entries" in saved_data and "entries" in saved_data:
                logger.info("Reusing valid existing dialogue index: %s", dialogue_json_path)
                entries = [
                    DialogueEntry(
                        index=e["index"],
                        start_seconds=e["start_seconds"],
                        end_seconds=e["end_seconds"],
                        start=e["start"],
                        end=e["end"],
                        duration_seconds=e["duration_seconds"],
                        text=e["text"],
                        speaker=e.get("speaker"),
                    )
                    for e in saved_data.get("entries", [])
                ]
                return DialogueDocument(
                    movie_id=saved_data.get("movie_id", assigned_movie_id),
                    source_type=saved_data.get("source_type", "unknown"),
                    source_file=saved_data.get("source_file"),
                    language=saved_data.get("language", "und"),
                    status=saved_data.get("status", "SUCCESS"),
                    total_entries=len(entries),
                    entries=entries,
                )
        except Exception as exc:
            logger.warning("Existing dialogue.json invalid; re-extracting. Reason: %s", exc)

    # 1. Check explicit subtitle file
    if subtitle_file_path:
        sub_path = Path(subtitle_file_path).resolve()
        if not sub_path.exists():
            raise SubtitleFileNotFoundError(f"Supplied subtitle file does not exist: '{sub_path}'")

        logger.info("Using explicitly supplied subtitle file: %s", sub_path)
        detected_lang = language or detect_subtitle_language_from_filename(sub_path)
        entries = parse_subtitle_file(sub_path)
        doc = DialogueDocument(
            movie_id=assigned_movie_id,
            source_type="external_subtitle",
            source_file=str(sub_path),
            language=detected_lang,
            status="SUCCESS",
            total_entries=len(entries),
            entries=entries,
        )
        _save_dialogue_document(doc, dialogue_json_path)
        return doc

    # 2. Inspect embedded subtitle tracks
    meta = extract_metadata(resolved_source, movie_id=assigned_movie_id)
    sub_tracks = meta.subtitle_tracks

    selected_track = select_embedded_subtitle_stream(
        subtitle_tracks=sub_tracks,
        requested_stream_index=subtitle_stream_index,
        requested_language=language,
    )

    if selected_track:
        stream_idx = selected_track["stream_index"]
        track_lang = selected_track.get("language") or language or "und"
        codec = selected_track.get("codec_name", "srt").lower()
        
        # Decide output extension for extracted raw subtitle
        ext = ".srt"
        if codec in ("webvtt", "vtt"):
            ext = ".vtt"
        elif codec in ("ass", "ssa"):
            ext = ".ass"

        raw_sub_path = subtitles_dir / f"raw_subtitle{ext}"
        logger.info(
            "Extracting embedded subtitle stream %d (%s, lang=%s) -> %s",
            stream_idx, codec, track_lang, raw_sub_path
        )

        try:
            extract_embedded_subtitle_stream(
                source_video=resolved_source,
                stream_index=stream_idx,
                output_subtitle_path=raw_sub_path,
            )
            entries = parse_subtitle_file(raw_sub_path)
            doc = DialogueDocument(
                movie_id=assigned_movie_id,
                source_type="embedded_subtitle",
                source_file=f"subtitles/{raw_sub_path.name}",
                language=track_lang,
                status="SUCCESS",
                total_entries=len(entries),
                entries=entries,
            )
            _save_dialogue_document(doc, dialogue_json_path)
            return doc
        except Exception as exc:
            logger.error("Failed to extract or parse embedded subtitle stream %d: %s", stream_idx, exc)

    # 3. Check for external subtitle discovery near source movie
    external_candidates = find_external_subtitles(resolved_source, movie_id=assigned_movie_id)
    if external_candidates:
        chosen_ext_sub = external_candidates[0]
        logger.info("Discovered external matching subtitle: %s", chosen_ext_sub)
        detected_lang = language or detect_subtitle_language_from_filename(chosen_ext_sub)
        entries = parse_subtitle_file(chosen_ext_sub)
        doc = DialogueDocument(
            movie_id=assigned_movie_id,
            source_type="external_subtitle",
            source_file=str(chosen_ext_sub),
            language=detected_lang,
            status="SUCCESS",
            total_entries=len(entries),
            entries=entries,
        )
        _save_dialogue_document(doc, dialogue_json_path)
        return doc

    # 4. Check if movie only has unsupported image-based subtitles
    image_tracks = [
        t for t in sub_tracks
        if t.get("codec_name", "").lower() in IMAGE_SUBTITLE_CODECS
    ]
    if image_tracks and not sub_tracks:
        pass
    if image_tracks and not any(t.get("codec_name", "").lower() in TEXT_SUBTITLE_CODECS for t in sub_tracks):
        logger.warning("Movie '%s' has only image-based subtitles (PGS/VobSub), unsupported in V1.", assigned_movie_id)
        doc = DialogueDocument(
            movie_id=assigned_movie_id,
            source_type="embedded_subtitle",
            source_file=None,
            language="und",
            status="IMAGE_SUBTITLE_UNSUPPORTED_IN_V1",
            total_entries=0,
            entries=[],
        )
        _save_dialogue_document(doc, dialogue_json_path)
        return doc

    # 5. No subtitle source available
    logger.info("No subtitle source found for movie '%s'.", assigned_movie_id)
    doc = DialogueDocument(
        movie_id=assigned_movie_id,
        source_type="no_dialogue",
        source_file=None,
        language="und",
        status="NO_DIALOGUE_SOURCE_AVAILABLE",
        total_entries=0,
        entries=[],
    )
    _save_dialogue_document(doc, dialogue_json_path)
    return doc


def _save_dialogue_document(doc: DialogueDocument, target_path: Path) -> Path:
    """Save DialogueDocument as JSON."""
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(doc.to_json(indent=2))
        logger.info("Successfully wrote dialogue document: %s", target_path)
        return target_path
    except Exception as exc:
        logger.error("Failed to write dialogue JSON to %s: %s", target_path, exc)
        raise DialogueWriteError(f"Could not save dialogue JSON to '{target_path}': {exc}") from exc


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main() -> int:
    """Command-line entry point for dialogue extraction."""
    parser = argparse.ArgumentParser(
        description="Movie Review Engine - Phase 3: Subtitle & Dialogue Extraction."
    )
    parser.add_argument(
        "movie_path",
        type=str,
        help="Path to the movie file (e.g. 'movie.mkv' or full path).",
    )
    parser.add_argument(
        "--language", "--lang",
        type=str,
        default=None,
        help="Preferred subtitle language code (e.g. 'eng', 'msa', 'zho').",
    )
    parser.add_argument(
        "--subtitle-stream", "--stream",
        type=int,
        default=None,
        help="Specific embedded subtitle stream index to extract.",
    )
    parser.add_argument(
        "--subtitle-file", "--sub-file",
        type=str,
        default=None,
        help="Path to an explicit external subtitle file (.srt, .vtt, .ass).",
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
        help="Force rebuild/re-extract dialogue even if existing.",
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
        doc = extract_movie_dialogue(
            source_path=args.movie_path,
            subtitle_file_path=args.subtitle_file,
            subtitle_stream_index=args.subtitle_stream,
            language=args.language,
            movie_id=args.movie_id,
            output_base_dir=out_base,
            force=args.force,
        )

        print("\n" + "=" * 60)
        print(f"[SUCCESS] DIALOGUE EXTRACTION: {doc.movie_id}")
        print("=" * 60)
        print(f"  Status        : {doc.status}")
        print(f"  Source Type   : {doc.source_type}")
        print(f"  Source File   : {doc.source_file or 'N/A'}")
        print(f"  Language      : {doc.language}")
        print(f"  Total Lines   : {doc.total_entries}")
        if doc.entries:
            print(f"  First Line    : [{doc.entries[0].start}] \"{doc.entries[0].text[:60]}\"")
            print(f"  Last Line     : [{doc.entries[-1].start}] \"{doc.entries[-1].text[:60]}\"")
        print("=" * 60 + "\n")
        return 0
    except DialogueExtractionError as exc:
        print(f"\n[ERROR] DIALOGUE EXTRACTION FAILED: {exc}\n", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected error during dialogue extraction")
        print(f"\n[ERROR] UNEXPECTED FAILURE: {exc}\n", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
