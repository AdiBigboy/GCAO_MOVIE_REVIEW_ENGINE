"""
Movie Review Engine - Phase 6: End-to-End Pipeline Runner
Sequentially executes Phase 1 to Phase 5 across a movie file and generates a structured summary.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.paths import (
    ANALYSIS_DIR,
    PROJECT_ROOT,
    get_movie_source_path,
)
from ai import (
    BaseAIProvider,
    EventsDocument,
    get_ai_provider,
    AIProviderError,
    MissingAPIKeyError,
)
from movie_analyzer.ingest import (
    MovieIngestError,
    MovieMetadata,
    SourceFileNotFoundError,
    ingest_movie,
    sanitize_movie_id,
)
from movie_analyzer.sample_frames import (
    TimelineIndex,
    TimelineSamplingError,
    sample_movie_frames,
)
from movie_analyzer.extract_dialogue import (
    DialogueDocument,
    DialogueExtractionError,
    extract_movie_dialogue,
)
from movie_analyzer.analyze_events import (
    analyze_timeline_events,
)
from movie_analyzer.track_characters import (
    CharactersDocument,
    track_movie_characters,
)
from story_engine.reconstruct_story import (
    StoryDocument,
    reconstruct_movie_story,
)
from narration_engine.plan_narration import (
    NarrativePlanDocument,
    generate_movie_narrative_plan,
)
from narration_engine.generate_script import (
    ScriptDocument,
    generate_movie_script,
)
from clip_engine.select_clips import (
    ClipPlanDocument,
    generate_movie_clip_plan,
)

logger = logging.getLogger("pipeline_runner")


# ==============================================================================
# Pipeline Result & Summary Data Models
# ==============================================================================

@dataclass
class PipelinePhaseResult:
    """Status and metadata for an individual pipeline phase."""
    phase_name: str
    status: str  # "SUCCESS", "FAILED", "SKIPPED"
    duration_seconds: float
    artifact_path: Optional[str] = None
    summary: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


@dataclass
class PipelineExecutionSummary:
    """Overall summary of end-to-end pipeline execution."""
    movie_id: str
    source_filename: str
    source_path: str
    status: str  # "SUCCESS", "FAILED", "DRY_RUN"
    started_at: str
    completed_at: str
    total_duration_seconds: float
    phases: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ==============================================================================
# Core Pipeline Execution Logic
# ==============================================================================

def run_movie_pipeline(
    source_path: str | Path,
    movie_id: Optional[str] = None,
    output_base_dir: Optional[Path] = None,
    provider_name: str = "gemini",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    interval_seconds: float = 3.0,
    language: Optional[str] = None,
    subtitle_file_path: Optional[str | Path] = None,
    subtitle_stream_index: Optional[int] = None,
    dialogue_window_seconds: float = 3.0,
    limit: Optional[int] = None,
    force: bool = False,
    dry_run: bool = False,
    jpeg_quality: int = 2,
    ai_provider: Optional[BaseAIProvider] = None,
) -> PipelineExecutionSummary:
    """
    Execute full Phase 1 to Phase 5 pipeline sequentially:
    1. Ingestion & metadata extraction (Phase 1)
    2. Timeline sampling & frame extraction (Phase 2)
    3. Subtitle & dialogue extraction (Phase 3)
    4. Multimodal timeline event analysis (Phase 4)
    5. Character identity tracking & memory (Phase 5)
    6. Generate & save pipeline_summary.json (Phase 6)
    """
    pipeline_start = time.time()
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # 0. Validate and resolve source path
    resolved_source = get_movie_source_path(source_path)
    if not resolved_source.exists():
        logger.error("Movie source file not found: %s", resolved_source)
        raise SourceFileNotFoundError(f"Movie source file not found: '{resolved_source}'")

    if not resolved_source.is_file():
        logger.error("Source path is not a file: %s", resolved_source)
        raise SourceFileNotFoundError(f"Movie source path is not a valid file: '{resolved_source}'")

    out_base = output_base_dir or ANALYSIS_DIR
    out_base.mkdir(parents=True, exist_ok=True)

    assigned_movie_id = movie_id or sanitize_movie_id(resolved_source.name)
    movie_analysis_dir = out_base / assigned_movie_id
    movie_analysis_dir.mkdir(parents=True, exist_ok=True)
    summary_file_path = movie_analysis_dir / "pipeline_summary.json"

    phase_results: Dict[str, Any] = {}

    print("\n" + "=" * 60)
    print("GCAO MOVIE REVIEW ENGINE - PIPELINE EXECUTION")
    print("=" * 60)
    print(f"  Source Movie : {resolved_source.name}")
    print(f"  Source Path  : {resolved_source}")
    print(f"  Movie ID     : {assigned_movie_id}")
    print(f"  Output Base  : {out_base}")
    print(f"  AI Provider  : {provider_name} ({model_name or 'default'})")
    if limit:
        print(f"  Frame Limit  : {limit}")
    if dry_run:
        print("  Mode         : DRY RUN")
    print("=" * 60 + "\n")

    # --------------------------------------------------------------------------
    # Phase 1: Ingest Movie
    # --------------------------------------------------------------------------
    print("[1/8] Ingesting movie metadata...")
    t0 = time.time()
    try:
        metadata: MovieMetadata = ingest_movie(
            source_path=resolved_source,
            movie_id=assigned_movie_id,
            output_base_dir=out_base,
        )
        t_ingest = time.time() - t0
        phase_results["phase_1_ingest"] = {
            "status": "SUCCESS",
            "duration_seconds": round(t_ingest, 3),
            "artifact_file": "movie_metadata.json",
            "resolution": metadata.resolution,
            "fps": metadata.fps,
            "video_codec": metadata.video_codec,
            "audio_codec": metadata.audio_codec,
            "duration_seconds_movie": metadata.duration_seconds,
            "file_size_human": metadata.file_size_human,
            "subtitles_count": metadata.number_of_subtitle_streams,
            "subtitle_languages": metadata.subtitle_languages,
        }
        print(
            f"      -> Success ({t_ingest:.2f}s): {metadata.resolution} @ {metadata.fps} fps, "
            f"Duration: {metadata.duration_seconds:.2f}s, Codec: {metadata.video_codec}/{metadata.audio_codec}"
        )
    except Exception as exc:
        t_ingest = time.time() - t0
        err_msg = f"Phase 1 (Ingest) failed: {exc}"
        logger.error(err_msg, exc_info=True)
        phase_results["phase_1_ingest"] = {
            "status": "FAILED",
            "duration_seconds": round(t_ingest, 3),
            "error": str(exc),
        }
        _write_failed_summary(
            summary_file_path=summary_file_path,
            movie_id=assigned_movie_id,
            source_filename=resolved_source.name,
            source_path=str(resolved_source),
            started_at=started_at,
            duration=time.time() - pipeline_start,
            phase_results=phase_results,
            error=err_msg,
        )
        raise

    # --------------------------------------------------------------------------
    # Phase 2: Sample Movie Frames
    # --------------------------------------------------------------------------
    print("[2/8] Sampling timeline frames...")
    t0 = time.time()
    try:
        timeline: TimelineIndex = sample_movie_frames(
            source_path=resolved_source,
            interval_seconds=interval_seconds,
            movie_id=assigned_movie_id,
            output_base_dir=out_base,
            force=force,
            jpeg_quality=jpeg_quality,
        )
        t_sample = time.time() - t0
        phase_results["phase_2_sample_frames"] = {
            "status": "SUCCESS",
            "duration_seconds": round(t_sample, 3),
            "artifact_file": "timeline_index.json",
            "frames_directory": "frames",
            "sampling_interval_seconds": timeline.sampling_interval_seconds,
            "total_samples": timeline.total_samples,
        }
        print(
            f"      -> Success ({t_sample:.2f}s): {timeline.total_samples} frames extracted "
            f"at interval {timeline.sampling_interval_seconds:.1f}s"
        )
    except Exception as exc:
        t_sample = time.time() - t0
        err_msg = f"Phase 2 (Sample Frames) failed: {exc}"
        logger.error(err_msg, exc_info=True)
        phase_results["phase_2_sample_frames"] = {
            "status": "FAILED",
            "duration_seconds": round(t_sample, 3),
            "error": str(exc),
        }
        _write_failed_summary(
            summary_file_path=summary_file_path,
            movie_id=assigned_movie_id,
            source_filename=resolved_source.name,
            source_path=str(resolved_source),
            started_at=started_at,
            duration=time.time() - pipeline_start,
            phase_results=phase_results,
            error=err_msg,
        )
        raise

    # --------------------------------------------------------------------------
    # Phase 3: Extract Movie Dialogue
    # --------------------------------------------------------------------------
    print("[3/8] Extracting dialogue and subtitles...")
    t0 = time.time()
    try:
        dialogue: DialogueDocument = extract_movie_dialogue(
            source_path=resolved_source,
            subtitle_file_path=subtitle_file_path,
            subtitle_stream_index=subtitle_stream_index,
            language=language,
            movie_id=assigned_movie_id,
            output_base_dir=out_base,
            force=force,
        )
        t_dialogue = time.time() - t0
        phase_results["phase_3_extract_dialogue"] = {
            "status": "SUCCESS",
            "duration_seconds": round(t_dialogue, 3),
            "artifact_file": "dialogue.json",
            "subtitles_directory": "subtitles",
            "source_type": dialogue.source_type,
            "dialogue_status": dialogue.status,
            "language": dialogue.language,
            "total_entries": dialogue.total_entries,
        }
        print(
            f"      -> Success ({t_dialogue:.2f}s): {dialogue.total_entries} dialogue entries extracted "
            f"(source: {dialogue.source_type}, lang: {dialogue.language})"
        )
    except Exception as exc:
        t_dialogue = time.time() - t0
        err_msg = f"Phase 3 (Extract Dialogue) failed: {exc}"
        logger.error(err_msg, exc_info=True)
        phase_results["phase_3_extract_dialogue"] = {
            "status": "FAILED",
            "duration_seconds": round(t_dialogue, 3),
            "error": str(exc),
        }
        _write_failed_summary(
            summary_file_path=summary_file_path,
            movie_id=assigned_movie_id,
            source_filename=resolved_source.name,
            source_path=str(resolved_source),
            started_at=started_at,
            duration=time.time() - pipeline_start,
            phase_results=phase_results,
            error=err_msg,
        )
        raise

    # --------------------------------------------------------------------------
    # Phase 4: Multimodal Timeline Event Analysis
    # --------------------------------------------------------------------------
    print("[4/8] Analyzing timeline events...")
    t0 = time.time()
    try:
        events: EventsDocument = analyze_timeline_events(
            source_path=resolved_source,
            provider_name=provider_name,
            model_name=model_name,
            api_key=api_key,
            movie_id=assigned_movie_id,
            output_base_dir=out_base,
            dialogue_window_seconds=dialogue_window_seconds,
            limit=limit,
            force=force,
            dry_run=dry_run,
            ai_provider=ai_provider,
        )
        t_events = time.time() - t0
        phase_results["phase_4_analyze_events"] = {
            "status": "SUCCESS" if not dry_run else "DRY_RUN",
            "duration_seconds": round(t_events, 3),
            "artifact_file": "events.json" if not dry_run else None,
            "provider": events.provider,
            "model": events.model,
            "total_events": events.total_events,
        }
        print(
            f"      -> Success ({t_events:.2f}s): {events.total_events} timeline events processed "
            f"({events.provider}: {events.model})"
        )
    except Exception as exc:
        t_events = time.time() - t0
        err_msg = f"Phase 4 (Analyze Events) failed: {exc}"
        logger.error(err_msg, exc_info=True)
        phase_results["phase_4_analyze_events"] = {
            "status": "FAILED",
            "duration_seconds": round(t_events, 3),
            "error": str(exc),
        }
        _write_failed_summary(
            summary_file_path=summary_file_path,
            movie_id=assigned_movie_id,
            source_filename=resolved_source.name,
            source_path=str(resolved_source),
            started_at=started_at,
            duration=time.time() - pipeline_start,
            phase_results=phase_results,
            error=err_msg,
        )
        raise

    # --------------------------------------------------------------------------
    # Phase 5: Track Character Identities & Memory
    # --------------------------------------------------------------------------
    print("[5/8] Tracking character identities & memory...")
    t0 = time.time()
    try:
        characters: CharactersDocument = track_movie_characters(
            source_path=resolved_source,
            provider_name=provider_name,
            model_name=model_name,
            api_key=api_key,
            movie_id=assigned_movie_id,
            output_base_dir=out_base,
            limit=limit,
            force=force,
            dry_run=dry_run,
            ai_provider=ai_provider,
        )
        t_characters = time.time() - t0

        named_chars = [c.canonical_name for c in characters.characters if c.canonical_name]
        phase_results["phase_5_track_characters"] = {
            "status": "SUCCESS" if not dry_run else "DRY_RUN",
            "duration_seconds": round(t_characters, 3),
            "artifact_file": "characters.json" if not dry_run else None,
            "total_characters": characters.total_characters,
            "named_characters": named_chars,
            "merges_count": len(characters.merge_history),
        }
        print(
            f"      -> Success ({t_characters:.2f}s): {characters.total_characters} characters tracked "
            f"({len(named_chars)} named: {', '.join(named_chars) if named_chars else 'None'})"
        )
    except Exception as exc:
        t_characters = time.time() - t0
        err_msg = f"Phase 5 (Track Characters) failed: {exc}"
        logger.error(err_msg, exc_info=True)
        phase_results["phase_5_track_characters"] = {
            "status": "FAILED",
            "duration_seconds": round(t_characters, 3),
            "error": str(exc),
        }
        _write_failed_summary(
            summary_file_path=summary_file_path,
            movie_id=assigned_movie_id,
            source_filename=resolved_source.name,
            source_path=str(resolved_source),
            started_at=started_at,
            duration=time.time() - pipeline_start,
            phase_results=phase_results,
            error=err_msg,
        )
        raise

    # --------------------------------------------------------------------------
    # Phase 7: Story Reconstruction Engine
    # --------------------------------------------------------------------------
    print("[6/8] Reconstructing movie story...")
    t0 = time.time()
    try:
        story: StoryDocument = reconstruct_movie_story(
            source_path=resolved_source,
            movie_id=assigned_movie_id,
            output_base_dir=out_base,
            force=force,
        )
        t_story = time.time() - t0
        phase_results["phase_7_reconstruct_story"] = {
            "status": "SUCCESS" if not dry_run else "DRY_RUN",
            "duration_seconds": round(t_story, 3),
            "artifact_file": "story.json" if not dry_run else None,
            "story_status": story.story_status,
            "scenes_count": len(story.scenes),
            "beats_count": len(story.beats),
            "causal_links_count": len(story.causal_links),
            "character_arcs_count": len(story.character_arcs),
            "protagonists_count": len(story.protagonist_candidates),
        }
        print(
            f"      -> Success ({t_story:.2f}s): Story reconstructed "
            f"(status: {story.story_status}, {len(story.scenes)} scenes, {len(story.beats)} beats, "
            f"{len(story.causal_links)} causal links, {len(story.character_arcs)} arcs)"
        )
    except Exception as exc:
        t_story = time.time() - t0
        err_msg = f"Phase 7 (Reconstruct Story) failed: {exc}"
        logger.error(err_msg, exc_info=True)
        phase_results["phase_7_reconstruct_story"] = {
            "status": "FAILED",
            "duration_seconds": round(t_story, 3),
            "error": str(exc),
        }
        _write_failed_summary(
            summary_file_path=summary_file_path,
            movie_id=assigned_movie_id,
            source_filename=resolved_source.name,
            source_path=str(resolved_source),
            started_at=started_at,
            duration=time.time() - pipeline_start,
            phase_results=phase_results,
            error=err_msg,
        )
        raise

    # --------------------------------------------------------------------------
    # Phase 8: Narration & Script Generation Engine
    # --------------------------------------------------------------------------
    print("[7/8] Planning narrative and generating Malay review script...")
    t0 = time.time()
    try:
        plan_doc: NarrativePlanDocument = generate_movie_narrative_plan(
            source_path=resolved_source,
            movie_id=assigned_movie_id,
            output_base_dir=out_base,
            force=force,
        )
        script_doc: ScriptDocument = generate_movie_script(
            source_path=resolved_source,
            movie_id=assigned_movie_id,
            output_base_dir=out_base,
            force=force,
        )
        t_script = time.time() - t0
        phase_results["phase_8_generate_script"] = {
            "status": "SUCCESS" if not dry_run else "DRY_RUN",
            "duration_seconds": round(t_script, 3),
            "plan_artifact_file": "narrative_plan.json" if not dry_run else None,
            "script_artifact_file": "script.json" if not dry_run else None,
            "language": script_doc.language,
            "total_words": script_doc.total_words,
            "estimated_duration_seconds": script_doc.estimated_total_duration_seconds,
            "segments_count": len(script_doc.segments),
        }
        print(
            f"      -> Success ({t_script:.2f}s): Script generated "
            f"({len(script_doc.segments)} segments, {script_doc.total_words} words, "
            f"~{script_doc.estimated_total_duration_seconds:.1f}s duration @ {script_doc.words_per_minute:.0f} WPM)"
        )
    except Exception as exc:
        t_script = time.time() - t0
        err_msg = f"Phase 8 (Generate Script) failed: {exc}"
        logger.error(err_msg, exc_info=True)
        phase_results["phase_8_generate_script"] = {
            "status": "FAILED",
            "duration_seconds": round(t_script, 3),
            "error": str(exc),
        }
        _write_failed_summary(
            summary_file_path=summary_file_path,
            movie_id=assigned_movie_id,
            source_filename=resolved_source.name,
            source_path=str(resolved_source),
            started_at=started_at,
            duration=time.time() - pipeline_start,
            phase_results=phase_results,
            error=err_msg,
        )
        raise

    # --------------------------------------------------------------------------
    # Phase 9: Source Clip Selection Engine
    # --------------------------------------------------------------------------
    print("[8/8] Selecting supporting source clips (<= 3.0s)...")
    t0 = time.time()
    try:
        clip_plan: ClipPlanDocument = generate_movie_clip_plan(
            source_path=resolved_source,
            movie_id=assigned_movie_id,
            output_base_dir=out_base,
            max_clip_duration_seconds=3.0,
            force=force,
        )
        t_clip = time.time() - t0
        phase_results["phase_9_select_clips"] = {
            "status": "SUCCESS" if not dry_run else "DRY_RUN",
            "duration_seconds": round(t_clip, 3),
            "artifact_file": "clip_plan.json" if not dry_run else None,
            "total_clips": clip_plan.total_clips,
            "total_source_duration_seconds": clip_plan.total_source_duration_seconds,
            "max_clip_duration_seconds": clip_plan.max_clip_duration_seconds,
        }
        print(
            f"      -> Success ({t_clip:.2f}s): {clip_plan.total_clips} source clips selected "
            f"(total source footage: {clip_plan.total_source_duration_seconds:.1f}s, max clip: {clip_plan.max_clip_duration_seconds:.1f}s)"
        )
    except Exception as exc:
        t_clip = time.time() - t0
        err_msg = f"Phase 9 (Select Clips) failed: {exc}"
        logger.error(err_msg, exc_info=True)
        phase_results["phase_9_select_clips"] = {
            "status": "FAILED",
            "duration_seconds": round(t_clip, 3),
            "error": str(exc),
        }
        _write_failed_summary(
            summary_file_path=summary_file_path,
            movie_id=assigned_movie_id,
            source_filename=resolved_source.name,
            source_path=str(resolved_source),
            started_at=started_at,
            duration=time.time() - pipeline_start,
            phase_results=phase_results,
            error=err_msg,
        )
        raise

    # --------------------------------------------------------------------------
    # Final Summary Generation
    # --------------------------------------------------------------------------
    total_duration = round(time.time() - pipeline_start, 3)
    completed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    overall_status = "SUCCESS" if not dry_run else "DRY_RUN"

    summary = PipelineExecutionSummary(
        movie_id=assigned_movie_id,
        source_filename=resolved_source.name,
        source_path=str(resolved_source),
        status=overall_status,
        started_at=started_at,
        completed_at=completed_at,
        total_duration_seconds=total_duration,
        phases=phase_results,
        error=None,
    )

    try:
        with open(summary_file_path, "w", encoding="utf-8") as f:
            f.write(summary.to_json(indent=2))
        logger.info("Pipeline summary written: %s", summary_file_path)
    except Exception as exc:
        logger.error("Failed to write pipeline_summary.json: %s", exc)

    print("\n" + "=" * 60)
    print(f"[{overall_status}] PIPELINE COMPLETED IN {total_duration:.2f}s")
    print(f"  Movie ID         : {assigned_movie_id}")
    print(f"  Pipeline Summary : {summary_file_path}")
    print("=" * 60 + "\n")

    return summary


def _write_failed_summary(
    summary_file_path: Path,
    movie_id: str,
    source_filename: str,
    source_path: str,
    started_at: str,
    duration: float,
    phase_results: Dict[str, Any],
    error: str,
) -> None:
    """Save failure record into pipeline_summary.json."""
    try:
        summary_file_path.parent.mkdir(parents=True, exist_ok=True)
        summary = PipelineExecutionSummary(
            movie_id=movie_id,
            source_filename=source_filename,
            source_path=source_path,
            status="FAILED",
            started_at=started_at,
            completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            total_duration_seconds=round(duration, 3),
            phases=phase_results,
            error=error,
        )
        with open(summary_file_path, "w", encoding="utf-8") as f:
            f.write(summary.to_json(indent=2))
    except Exception as exc:
        logger.error("Could not write failure summary: %s", exc)


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main() -> int:
    """Command-line interface for the end-to-end movie review engine pipeline."""
    parser = argparse.ArgumentParser(
        description="GCAO Movie Review Engine - Phase 6: End-to-End Pipeline Runner."
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
        help="Optional custom movie_id identifier.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Optional override base analysis directory (default: project analysis/).",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="gemini",
        help="Multimodal AI provider ('gemini', 'mock', default: 'gemini').",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="AI model name (e.g. 'gemini-flash-latest', 'gemini-2.5-flash').",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional API key override (prefer setting GEMINI_API_KEY environment variable).",
    )
    parser.add_argument(
        "--interval", "-i",
        type=float,
        default=3.0,
        help="Frame sampling interval in seconds (default: 3.0).",
    )
    parser.add_argument(
        "--language", "--lang",
        type=str,
        default=None,
        help="Preferred subtitle language code (e.g. 'eng', 'msa', 'zho').",
    )
    parser.add_argument(
        "--subtitle-file", "--sub-file",
        type=str,
        default=None,
        help="Path to an explicit external subtitle file (.srt, .vtt, .ass).",
    )
    parser.add_argument(
        "--subtitle-stream", "--stream",
        type=int,
        default=None,
        help="Specific embedded subtitle stream index to extract.",
    )
    parser.add_argument(
        "--dialogue-window",
        type=float,
        default=3.0,
        help="Dialogue matching time window in seconds (default: 3.0s).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of frames/events to process (e.g. --limit 5).",
    )
    parser.add_argument(
        "--quality", "-q",
        type=int,
        default=2,
        help="JPEG quality scale (1-31, 2 = high quality, default: 2).",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-extraction and re-analysis of all artifacts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run usage and cost estimation mode without calling AI models.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable detailed debug logging.",
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
        run_movie_pipeline(
            source_path=args.movie_path,
            movie_id=args.movie_id,
            output_base_dir=out_base,
            provider_name=args.provider,
            model_name=args.model,
            api_key=args.api_key,
            interval_seconds=args.interval,
            language=args.language,
            subtitle_file_path=args.subtitle_file,
            subtitle_stream_index=args.subtitle_stream,
            dialogue_window_seconds=args.dialogue_window,
            limit=args.limit,
            force=args.force,
            dry_run=args.dry_run,
            jpeg_quality=args.quality,
        )
        return 0
    except SourceFileNotFoundError as exc:
        print(f"\n[ERROR] SOURCE FILE NOT FOUND: {exc}\n", file=sys.stderr)
        return 1
    except MovieIngestError as exc:
        print(f"\n[ERROR] PIPELINE FAILURE: {exc}\n", file=sys.stderr)
        return 2
    except MissingAPIKeyError as exc:
        print(f"\n[ERROR] MISSING API KEY: {exc}\n", file=sys.stderr)
        return 3
    except AIProviderError as exc:
        print(f"\n[ERROR] AI PROVIDER ERROR: {exc}\n", file=sys.stderr)
        return 4
    except Exception as exc:
        logger.exception("Unexpected failure during pipeline execution")
        print(f"\n[ERROR] UNEXPECTED FAILURE: {exc}\n", file=sys.stderr)
        return 5


if __name__ == "__main__":
    sys.exit(main())
