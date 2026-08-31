"""
Movie Review Engine - Movie Analyzer Package
"""

def __getattr__(name: str):
    if name in {
        "MovieIngestError",
        "FFprobeNotFoundError",
        "SourceFileNotFoundError",
        "InvalidMediaFileError",
        "MetadataWriteError",
        "MovieMetadata",
        "ingest_movie",
        "extract_metadata",
        "save_metadata",
        "sanitize_movie_id",
    }:
        from . import ingest
        return getattr(ingest, name)

    if name in {
        "TimelineSamplingError",
        "InvalidIntervalError",
        "FFmpegNotFoundError",
        "FrameExtractionError",
        "TimelineWriteError",
        "FrameSample",
        "TimelineIndex",
        "sample_movie_frames",
        "compute_sample_points",
        "format_timestamp",
    }:
        from . import sample_frames
        return getattr(sample_frames, name)

    if name in {
        "DialogueExtractionError",
        "SubtitleFileNotFoundError",
        "DialogueWriteError",
        "DialogueEntry",
        "DialogueDocument",
        "extract_movie_dialogue",
        "parse_subtitle_file",
        "clean_subtitle_text",
    }:
        from . import extract_dialogue
        from . import subtitles
        if hasattr(extract_dialogue, name):
            return getattr(extract_dialogue, name)
        return getattr(subtitles, name)

    if name in {
        "analyze_timeline_events",
        "match_dialogue_to_timestamp",
        "build_rolling_context",
    }:
        from . import analyze_events
        return getattr(analyze_events, name)

    if name in {
        "track_movie_characters",
        "CharacterMemoryTracker",
        "CharacterRecord",
        "CharactersDocument",
        "CharacterAppearance",
        "VisualProfile",
        "NameEvidence",
        "extract_name_evidence_from_dialogue",
    }:
        from . import track_characters
        return getattr(track_characters, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Ingest (Phase 1)
    "MovieIngestError",
    "FFprobeNotFoundError",
    "SourceFileNotFoundError",
    "InvalidMediaFileError",
    "MetadataWriteError",
    "MovieMetadata",
    "ingest_movie",
    "extract_metadata",
    "save_metadata",
    "sanitize_movie_id",
    # Sample Frames (Phase 2)
    "TimelineSamplingError",
    "InvalidIntervalError",
    "FFmpegNotFoundError",
    "FrameExtractionError",
    "TimelineWriteError",
    "FrameSample",
    "TimelineIndex",
    "sample_movie_frames",
    "compute_sample_points",
    "format_timestamp",
    # Subtitles & Dialogue (Phase 3)
    "DialogueExtractionError",
    "SubtitleFileNotFoundError",
    "DialogueWriteError",
    "DialogueEntry",
    "DialogueDocument",
    "extract_movie_dialogue",
    "parse_subtitle_file",
    "clean_subtitle_text",
    # Timeline Event Analysis (Phase 4)
    "analyze_timeline_events",
    "match_dialogue_to_timestamp",
    "build_rolling_context",
    # Character Tracking & Memory (Phase 5)
    "track_movie_characters",
    "CharacterMemoryTracker",
    "CharacterRecord",
    "CharactersDocument",
    "CharacterAppearance",
    "VisualProfile",
    "NameEvidence",
    "extract_name_evidence_from_dialogue",
]
