"""
Movie Review Engine - Phase 4: Multimodal Timeline Event Analysis Module
Combines sampled frames, dialogue timeline, and rolling context to produce structured events.json.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from ai import (
    BaseAIProvider,
    EventsDocument,
    EventRecord,
    EventUncertainty,
    VisualAnalysis,
    get_ai_provider,
    AIProviderError,
    MissingAPIKeyError,
    ModelInvocationError,
    InvalidModelResponseError,
)
from movie_analyzer.ingest import sanitize_movie_id, SourceFileNotFoundError
from movie_analyzer.sample_frames import sample_movie_frames, TimelineIndex
from movie_analyzer.extract_dialogue import extract_movie_dialogue, DialogueDocument

logger = logging.getLogger("movie_analyzer.analyze_events")


# ==============================================================================
# Helper Functions
# ==============================================================================

def match_dialogue_to_timestamp(
    dialogue_entries: List[Dict[str, Any]],
    timestamp_seconds: float,
    window_seconds: float = 3.0,
) -> List[Dict[str, Any]]:
    """
    Find dialogue entries overlapping with the [timestamp - window, timestamp + window] range.
    """
    if not dialogue_entries:
        return []

    win_start = max(0.0, timestamp_seconds - window_seconds)
    win_end = timestamp_seconds + window_seconds

    matched: List[Dict[str, Any]] = []
    for entry in dialogue_entries:
        e_start = float(entry.get("start_seconds", 0.0))
        e_end = float(entry.get("end_seconds", 0.0))

        # Check interval overlap: max(e_start, win_start) <= min(e_end, win_end)
        if e_start <= win_end and e_end >= win_start:
            matched.append(entry)

    return matched


def build_rolling_context(
    completed_events: List[EventRecord],
    max_items: int = 4,
) -> List[str]:
    """
    Create a compact rolling summary list of recent events.
    """
    if not completed_events:
        return []

    recent = completed_events[-max_items:]
    context_lines: List[str] = []
    for evt in recent:
        line = f"- [{evt.timestamp}] {evt.event_summary}"
        context_lines.append(line)
    return context_lines


def save_events_checkpoint(
    events_doc: EventsDocument,
    target_file: Path,
) -> None:
    """Save events document to disk atomically."""
    target_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = target_file.with_suffix(".json.tmp")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(events_doc.to_dict(), f, indent=2, ensure_ascii=False)
    temp_file.replace(target_file)


# ==============================================================================
# Main Event Analysis Engine
# ==============================================================================

def analyze_timeline_events(
    source_path: str | Path,
    provider_name: str = "gemini",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    movie_id: Optional[str] = None,
    output_base_dir: Optional[Path] = None,
    dialogue_window_seconds: float = 3.0,
    limit: Optional[int] = None,
    force: bool = False,
    dry_run: bool = False,
    ai_provider: Optional[BaseAIProvider] = None,
) -> EventsDocument:
    """
    Phase 4 Multimodal Event Analysis:
    1. Loads Phase 2 timeline_index.json and Phase 3 dialogue.json.
    2. Instantiates AI provider.
    3. Handles dry-run estimation.
    4. Resumes from existing events.json checkpoint unless force=True.
    5. Incrementally analyzes frames with dialogue and rolling temporal context.
    6. Persists validated events.json.
    """
    resolved_source = Path(source_path).resolve()
    if not resolved_source.exists():
        logger.error("Movie source file not found: %s", resolved_source)
        raise SourceFileNotFoundError(f"Movie source file not found: '{resolved_source}'")

    assigned_movie_id = movie_id or sanitize_movie_id(resolved_source.name)
    if output_base_dir is None:
        workspace_root = Path(__file__).resolve().parents[1]
        output_base_dir = workspace_root / "analysis"

    movie_analysis_dir = output_base_dir / assigned_movie_id
    movie_analysis_dir.mkdir(parents=True, exist_ok=True)

    timeline_json_path = movie_analysis_dir / "timeline_index.json"
    dialogue_json_path = movie_analysis_dir / "dialogue.json"
    events_json_path = movie_analysis_dir / "events.json"

    # 1. Ensure Phase 2 timeline index exists
    if not timeline_json_path.exists():
        logger.info("timeline_index.json not found; running Phase 2 sampling...")
        timeline_obj = sample_movie_frames(
            source_path=resolved_source,
            movie_id=assigned_movie_id,
            output_base_dir=output_base_dir,
        )
    else:
        with open(timeline_json_path, "r", encoding="utf-8") as f:
            t_data = json.load(f)
        timeline_obj = TimelineIndex(
            movie_id=t_data.get("movie_id", assigned_movie_id),
            sampling_interval_seconds=float(t_data.get("sampling_interval_seconds", 3.0)),
            duration_seconds=float(t_data.get("duration_seconds", 0.0)),
            total_samples=int(t_data.get("total_samples", len(t_data.get("frames", [])))),
            frames=t_data.get("frames", []),
        )

    # 2. Ensure Phase 3 dialogue index exists
    if not dialogue_json_path.exists():
        logger.info("dialogue.json not found; running Phase 3 extraction...")
        dialogue_doc = extract_movie_dialogue(
            source_path=resolved_source,
            movie_id=assigned_movie_id,
            output_base_dir=output_base_dir,
        )
        dialogue_entries = [e.to_dict() for e in dialogue_doc.entries]
    else:
        with open(dialogue_json_path, "r", encoding="utf-8") as f:
            d_data = json.load(f)
        dialogue_entries = d_data.get("entries", [])

    # 3. Setup AI Provider
    debug_dir = movie_analysis_dir / "debug"
    if ai_provider is None:
        provider = get_ai_provider(
            provider_name=provider_name,
            model_name=model_name,
            api_key=api_key,
        )
        if hasattr(provider, "debug_dir") and getattr(provider, "debug_dir") is None:
            setattr(provider, "debug_dir", debug_dir)
    else:
        provider = ai_provider
        if hasattr(provider, "debug_dir") and getattr(provider, "debug_dir") is None:
            setattr(provider, "debug_dir", debug_dir)

    all_frames: List[Dict[str, Any]] = [
        f if isinstance(f, dict) else asdict(f) for f in timeline_obj.frames
    ]
    total_frames = len(all_frames)

    # 4. Check existing checkpoint for resume
    existing_events_by_index: Dict[int, EventRecord] = {}
    if events_json_path.exists() and not force:
        try:
            with open(events_json_path, "r", encoding="utf-8") as f:
                saved_events = json.load(f)
            for item in saved_events.get("events", []):
                idx = int(item["index"])
                vis_d = item.get("visual", {})
                visual = VisualAnalysis(
                    people_count=int(vis_d.get("people_count", 0)),
                    character_labels=list(vis_d.get("character_labels", [])),
                    location=str(vis_d.get("location", "unknown")),
                    action=str(vis_d.get("action", "unknown")),
                    objects=list(vis_d.get("objects", [])),
                    emotion=str(vis_d.get("emotion", "neutral")),
                    interaction=vis_d.get("interaction"),
                    visible_text=vis_d.get("visible_text"),
                )
                unc_d = item.get("uncertainty", {})
                uncertainty = EventUncertainty(
                    character_identity=str(unc_d.get("character_identity", "medium")),
                    event_interpretation=str(unc_d.get("event_interpretation", "medium")),
                    location_certainty=unc_d.get("location_certainty", "medium"),
                )
                rec = EventRecord(
                    index=idx,
                    timestamp_seconds=float(item["timestamp_seconds"]),
                    timestamp=str(item["timestamp"]),
                    frame_file=str(item["frame_file"]),
                    dialogue=list(item.get("dialogue", [])),
                    visual=visual,
                    event_summary=str(item.get("event_summary", "")),
                    plot_significance=int(item.get("plot_significance", 5)),
                    uncertainty=uncertainty,
                )
                existing_events_by_index[idx] = rec
            logger.info("Loaded %d existing events from checkpoint.", len(existing_events_by_index))
        except Exception as exc:
            logger.warning("Could not load existing events.json; starting fresh. Reason: %s", exc)
            existing_events_by_index = {}

    # Determine candidate frames to process
    frames_to_process: List[Dict[str, Any]] = []
    for f in all_frames:
        idx = int(f.get("index", 0))
        if force or idx not in existing_events_by_index:
            frames_to_process.append(f)

    if limit is not None and limit > 0:
        frames_to_process = frames_to_process[:limit]

    # 5. Handle Dry Run Mode
    if dry_run:
        print("\n" + "=" * 60)
        print("[DRY-RUN] PHASE 4 - USAGE & COST ESTIMATE")
        print("=" * 60)
        print(f"  Movie ID            : {assigned_movie_id}")
        print(f"  AI Provider         : {provider.provider_name}")
        print(f"  Model Name          : {provider.model_name}")
        print(f"  Total Video Frames  : {total_frames}")
        print(f"  Completed Events    : {len(existing_events_by_index)}")
        print(f"  Frames to Analyze   : {len(frames_to_process)} (limit={limit})")
        print(f"  Dialogue Window     : +/-{dialogue_window_seconds}s")
        print(f"  Total Dialogue Lines: {len(dialogue_entries)}")
        print(f"  Expected AI Calls   : {len(frames_to_process)}")
        print(f"  Batching Strategy   : Incremental per-frame with rolling context")
        print("=" * 60 + "\n")
        return EventsDocument(
            movie_id=assigned_movie_id,
            provider=provider.provider_name,
            model=provider.model_name,
            total_events=len(existing_events_by_index),
            events=list(existing_events_by_index.values()),
        )

    # 6. Execute Incremental Analysis
    logger.info(
        "Starting AI event analysis on %d frames (Provider: %s, Model: %s)...",
        len(frames_to_process), provider.provider_name, provider.model_name
    )

    processed_events_map = dict(existing_events_by_index)

    for i, frame_item in enumerate(frames_to_process, 1):
        idx = int(frame_item["index"])
        ts_sec = float(frame_item["timestamp_seconds"])
        ts_str = str(frame_item["timestamp"])
        rel_frame_file = str(frame_item["frame_file"])
        abs_frame_path = movie_analysis_dir / rel_frame_file

        if not abs_frame_path.exists():
            logger.error("Frame file missing on disk: %s", abs_frame_path)
            raise FileNotFoundError(f"Sample frame does not exist: '{abs_frame_path}'")

        # Extract matching dialogue
        matched_dialogue = match_dialogue_to_timestamp(
            dialogue_entries=dialogue_entries,
            timestamp_seconds=ts_sec,
            window_seconds=dialogue_window_seconds,
        )

        # Build chronological previous context
        sorted_completed = [
            processed_events_map[k] for k in sorted(processed_events_map.keys()) if k < idx
        ]
        prev_context = build_rolling_context(sorted_completed, max_items=4)

        logger.debug(
            "Analyzing frame %d/%d (Index #%d @ %s, Dialogue Lines: %d)...",
            i, len(frames_to_process), idx, ts_str, len(matched_dialogue)
        )

        # Call AI provider with retry
        max_attempts = 2
        last_error: Optional[Exception] = None
        event_record: Optional[EventRecord] = None

        for attempt in range(1, max_attempts + 1):
            try:
                event_record = provider.analyze_frame_event(
                    image_path=abs_frame_path,
                    timestamp_seconds=ts_sec,
                    timestamp_str=ts_str,
                    dialogue_context=matched_dialogue,
                    previous_context=prev_context,
                )
                break
            except AIProviderError as exc:
                last_error = exc
                logger.warning("AI provider error on frame %d (attempt %d/%d): %s", idx, attempt, max_attempts, exc)
            except Exception as exc:
                last_error = exc
                logger.warning("Unexpected error on frame %d (attempt %d/%d): %s", idx, attempt, max_attempts, exc)

        if event_record is None:
            logger.error("Failed to analyze frame %d after %d attempts.", idx, max_attempts)
            if isinstance(last_error, MissingAPIKeyError):
                raise last_error
            raise ModelInvocationError(f"Could not analyze frame {idx}: {last_error}") from last_error

        # Ensure correct index and relative path
        event_record.index = idx
        event_record.frame_file = rel_frame_file
        event_record.timestamp_seconds = ts_sec
        event_record.timestamp = ts_str
        event_record.dialogue = matched_dialogue

        processed_events_map[idx] = event_record

        # Checkpoint save after every processed event
        sorted_records = [processed_events_map[k] for k in sorted(processed_events_map.keys())]
        current_doc = EventsDocument(
            movie_id=assigned_movie_id,
            provider=provider.provider_name,
            model=provider.model_name,
            total_events=len(sorted_records),
            events=sorted_records,
        )
        save_events_checkpoint(current_doc, events_json_path)

    # 7. Final Document
    final_sorted = [processed_events_map[k] for k in sorted(processed_events_map.keys())]
    final_doc = EventsDocument(
        movie_id=assigned_movie_id,
        provider=provider.provider_name,
        model=provider.model_name,
        total_events=len(final_sorted),
        events=final_sorted,
    )
    save_events_checkpoint(final_doc, events_json_path)
    logger.info("Successfully completed event analysis. Total events: %d (JSON: %s)", len(final_sorted), events_json_path)

    return final_doc


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main() -> int:
    """CLI entry point for Phase 4 timeline event analysis."""
    parser = argparse.ArgumentParser(
        description="Movie Review Engine - Phase 4: Multimodal Timeline Event Analysis."
    )
    parser.add_argument(
        "movie_path",
        type=str,
        help="Path to the movie file (e.g. 'movie.mp4' or full path).",
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
        help="Model name (e.g. 'gemini-2.5-flash', 'gemini-1.5-flash').",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional API key override (prefer setting GEMINI_API_KEY environment variable).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of frames to process in this run (e.g. --limit 5).",
    )
    parser.add_argument(
        "--dialogue-window",
        type=float,
        default=3.0,
        help="Dialogue matching time window in seconds (default: 3.0s).",
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
        "--dry-run",
        action="store_true",
        help="Run usage and cost estimation mode without calling AI models.",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force rebuild/re-analyze all events from scratch.",
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
        doc = analyze_timeline_events(
            source_path=args.movie_path,
            provider_name=args.provider,
            model_name=args.model,
            api_key=args.api_key,
            movie_id=args.movie_id,
            output_base_dir=out_base,
            dialogue_window_seconds=args.dialogue_window,
            limit=args.limit,
            force=args.force,
            dry_run=args.dry_run,
        )

        if not args.dry_run:
            print("\n" + "=" * 60)
            print(f"[SUCCESS] TIMELINE EVENT ANALYSIS COMPLETED: {doc.movie_id}")
            print("=" * 60)
            print(f"  Provider     : {doc.provider} ({doc.model})")
            print(f"  Total Events : {doc.total_events}")
            if doc.events:
                first = doc.events[0]
                last = doc.events[-1]
                print(f"  First Event  : [{first.timestamp}] {first.event_summary[:60]}")
                print(f"  Last Event   : [{last.timestamp}] {last.event_summary[:60]}")
            print("=" * 60 + "\n")
        return 0
    except MissingAPIKeyError as exc:
        print(f"\n[ERROR] MISSING API KEY: {exc}\n", file=sys.stderr)
        return 1
    except AIProviderError as exc:
        print(f"\n[ERROR] AI PROVIDER ERROR: {exc}\n", file=sys.stderr)
        return 2
    except Exception as exc:
        logger.exception("Unexpected failure during timeline event analysis")
        print(f"\n[ERROR] UNEXPECTED FAILURE: {exc}\n", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
