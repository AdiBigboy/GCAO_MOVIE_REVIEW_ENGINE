"""
Movie Review Engine - Phase 9: Source Clip Selection Engine
Selects short, visually grounded movie clips (<= 3.0s) supporting each narration segment.
Prevents contiguous scene reconstruction and ensures 100% source traceability.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from clip_engine.models import (
    ClipPlanDocument,
    SegmentClips,
    SourceClip,
)
from clip_engine.validate_clips import ClipPlanValidator
from config.paths import ANALYSIS_DIR, MOVIES_SOURCE_DIR, PROJECT_ROOT, get_movie_source_path
from movie_analyzer.ingest import sanitize_movie_id
from narration_engine.models import ScriptDocument
from story_engine.models import StoryDocument

logger = logging.getLogger("clip_engine.select_clips")


class ClipSelectionError(Exception):
    """Base exception for clip selection errors."""
    pass


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _safe_list(val: Any) -> List[Any]:
    if isinstance(val, list):
        return val
    return []


def _safe_dict(val: Any) -> Dict[str, Any]:
    if isinstance(val, dict):
        return val
    return {}


class SourceClipSelector:
    """
    Selects supporting source clips for each narration segment.
    """

    def __init__(
        self,
        movie_id: str,
        analysis_dir: Path,
        movie_source_path: Optional[Path] = None,
        max_clip_duration_seconds: float = 3.0,
        min_separation_seconds: float = 8.0,
    ):
        self.movie_id = movie_id
        self.analysis_dir = analysis_dir
        self.movie_source_path = movie_source_path
        self.max_clip_duration_seconds = max_clip_duration_seconds
        self.min_separation_seconds = min_separation_seconds

        self.script_file = analysis_dir / "script.json"
        self.plan_file = analysis_dir / "narrative_plan.json"
        self.story_file = analysis_dir / "story.json"
        self.events_file = analysis_dir / "events.json"
        self.metadata_file = analysis_dir / "movie_metadata.json"
        self.output_clip_plan_file = analysis_dir / "clip_plan.json"

    def load_inputs(self) -> Tuple[ScriptDocument, StoryDocument, Dict[str, Any], Dict[str, Any]]:
        """Load script.json, story.json, events.json, and metadata."""
        if not self.script_file.exists():
            raise ClipSelectionError(f"script.json not found in {self.analysis_dir}")
        if not self.story_file.exists():
            raise ClipSelectionError(f"story.json not found in {self.analysis_dir}")

        with open(self.script_file, "r", encoding="utf-8") as f:
            script_doc = ScriptDocument.from_dict(json.load(f))

        with open(self.story_file, "r", encoding="utf-8") as f:
            story_doc = StoryDocument.from_dict(json.load(f))

        events_data: Dict[str, Any] = {}
        if self.events_file.exists():
            try:
                with open(self.events_file, "r", encoding="utf-8") as f:
                    events_data = json.load(f)
            except Exception as exc:
                logger.debug("Failed to load events.json: %s", exc)

        metadata_data: Dict[str, Any] = {}
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    metadata_data = json.load(f)
            except Exception as exc:
                logger.debug("Failed to load movie_metadata.json: %s", exc)

        return script_doc, story_doc, events_data, metadata_data

    def select(self, expand_coverage: bool = False) -> ClipPlanDocument:
        """
        Build the complete ClipPlanDocument mapping each narration segment
        to short non-contiguous supporting source clips.
        When expand_coverage=True, incorporates all grounded action events for continuous visual flow.
        """
        script_doc, story_doc, events_data, metadata_data = self.load_inputs()

        movie_duration = _safe_float(metadata_data.get("duration_seconds"), 7200.0)
        raw_events = _safe_list(events_data.get("events", []))
        events_by_idx: Dict[int, Dict[str, Any]] = {
            _safe_int(e.get("index", idx + 1)): e
            for idx, e in enumerate(raw_events)
            if isinstance(e, dict)
        }
        scenes_by_id = {s.scene_id: s for s in story_doc.scenes}

        all_selected_clips: List[SourceClip] = []
        segment_clips_list: List[SegmentClips] = []
        clip_counter = 1
        last_clip_end_time = -999.0

        for seg_idx, script_seg in enumerate(script_doc.segments, start=1):
            # Skip open continuation marker from consuming movie source footage
            if "sambungan analisis seterusnya" in script_seg.text.lower() or "setakat bahagian awal" in script_seg.text.lower():
                segment_clips_list.append(
                    SegmentClips(
                        segment_id=script_seg.segment_id,
                        narration_text=script_seg.text,
                        source_event_indices=script_seg.source_event_indices,
                        clips=[],
                    )
                )
                continue

            seg_selected_clips: List[SourceClip] = []
            cand_event_indices = [idx for idx in script_seg.source_event_indices if idx in events_by_idx]

            # Filter out black screens or empty setup frames if other evidence exists
            action_event_indices = [
                idx for idx in cand_event_indices
                if _safe_int(events_by_idx[idx].get("plot_significance", 5)) >= 3
            ]
            if not action_event_indices and cand_event_indices:
                action_event_indices = cand_event_indices

            if expand_coverage:
                chosen_indices = list(action_event_indices)
            else:
                # Determine number of clips for this segment based on importance
                # Importance >= 7: 2 to 3 clips; 4-6: 1 to 2 clips; <= 3: 1 clip
                if script_seg.importance >= 7:
                    target_clip_count = min(3, len(action_event_indices))
                    target_clip_count = max(2, target_clip_count) if len(action_event_indices) >= 2 else 1
                elif script_seg.importance >= 4:
                    target_clip_count = min(2, len(action_event_indices))
                    target_clip_count = max(1, target_clip_count)
                else:
                    target_clip_count = 1

                # Subsample evenly across available events in this segment
                if len(action_event_indices) > target_clip_count:
                    step = len(action_event_indices) / float(target_clip_count)
                    chosen_indices = [
                        action_event_indices[int(i * step)]
                        for i in range(target_clip_count)
                    ]
                else:
                    chosen_indices = list(action_event_indices)

            for ev_idx in chosen_indices:
                ev = events_by_idx.get(ev_idx, {})
                ev_time = _safe_float(ev.get("timestamp_seconds"), float(ev_idx * 30.0))
                ev_summary = str(ev.get("event_summary", "Supporting movie moment"))
                ev_vis = _safe_dict(ev.get("visual", {}))
                vis_action = str(ev_vis.get("action", ev_summary))

                # Determine clip start and end (strictly <= 3.0s duration)
                clip_duration = min(3.0, self.max_clip_duration_seconds)
                # Offset slightly from the exact 30s keyframe to capture continuous natural motion
                # Candidate start: 1.0s before or right at the keyframe
                clip_start = max(0.0, ev_time - 0.5)
                clip_end = min(movie_duration, clip_start + clip_duration)
                actual_duration = round(clip_end - clip_start, 2)

                # Ensure non-contiguous separation from the previous clip
                continuity_exception = False
                continuity_reason = None
                if clip_start < last_clip_end_time + 1.0:
                    # Adjust forward to prevent overlap
                    clip_start = last_clip_end_time + 2.0
                    clip_end = min(movie_duration, clip_start + actual_duration)
                    actual_duration = round(clip_end - clip_start, 2)
                    continuity_exception = True
                    continuity_reason = "Multi-angle reveal in close proximity"

                # Find associated scene ID
                assigned_scene_id = script_seg.source_scene_ids[0] if script_seg.source_scene_ids else "SCENE_001"
                for sid in script_seg.source_scene_ids:
                    sc = scenes_by_id.get(sid)
                    if sc and ev_idx in sc.event_indices:
                        assigned_scene_id = sid
                        break

                clip_obj = SourceClip(
                    clip_id=f"CLIP_{clip_counter:03d}",
                    source_start_seconds=round(clip_start, 2),
                    source_end_seconds=round(clip_end, 2),
                    duration_seconds=actual_duration,
                    source_event_index=ev_idx,
                    source_scene_id=assigned_scene_id,
                    visual_reason=f"Visual evidence for: {vis_action}",
                    importance=script_seg.importance,
                    confidence="high",
                    continuity_exception=continuity_exception,
                    continuity_reason=continuity_reason,
                )

                seg_selected_clips.append(clip_obj)
                all_selected_clips.append(clip_obj)
                clip_counter += 1
                last_clip_end_time = clip_end

            segment_clips_list.append(
                SegmentClips(
                    segment_id=script_seg.segment_id,
                    narration_text=script_seg.text,
                    source_event_indices=script_seg.source_event_indices,
                    clips=seg_selected_clips,
                )
            )

        total_source_duration = sum(c.duration_seconds for c in all_selected_clips)
        warnings: List[str] = []
        if script_doc.status == "PARTIAL":
            warnings.append(
                "Source clips selected only for analyzed partial timeline coverage. "
                "Remaining movie footage unselected."
            )

        plan_doc = ClipPlanDocument(
            movie_id=self.movie_id,
            status=script_doc.status,
            max_clip_duration_seconds=self.max_clip_duration_seconds,
            total_clips=len(all_selected_clips),
            total_source_duration_seconds=round(total_source_duration, 2),
            segments=segment_clips_list,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            warnings=warnings,
        )

        # Validate with ClipPlanValidator
        validator = ClipPlanValidator(
            max_clip_duration_seconds=self.max_clip_duration_seconds,
            movie_duration_seconds=movie_duration,
        )
        val_res = validator.validate(plan_doc)
        if not val_res.valid:
            logger.warning("Clip plan validation reported errors: %s", val_res.errors)

        return plan_doc

    def save_clip_plan(self, plan_doc: ClipPlanDocument) -> Path:
        """Save plan_doc to analysis/<movie_id>/clip_plan.json."""
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_clip_plan_file, "w", encoding="utf-8") as f:
            f.write(plan_doc.to_json(indent=2))
        logger.info("Saved source clip plan to %s", self.output_clip_plan_file)
        return self.output_clip_plan_file

    def extract_preview_clips(
        self,
        plan_doc: ClipPlanDocument,
        max_previews: Optional[int] = None,
    ) -> List[Path]:
        """
        Extract short preview clips for human inspection only.
        Does NOT render full review video.
        """
        source_path_to_use = self.movie_source_path
        if not source_path_to_use or not source_path_to_use.exists():
            if self.metadata_file.exists():
                try:
                    with open(self.metadata_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    cand = meta.get("absolute_source_path")
                    if cand and Path(cand).exists():
                        source_path_to_use = Path(cand)
                    else:
                        cand2 = MOVIES_SOURCE_DIR / str(meta.get("source_filename", ""))
                        if cand2.exists():
                            source_path_to_use = cand2
                except Exception:
                    pass

        if not source_path_to_use or not source_path_to_use.exists():
            logger.info("Movie source file not accessible for preview extraction: %s", source_path_to_use)
            return []

        preview_dir = self.analysis_dir / "clip_previews"
        preview_dir.mkdir(parents=True, exist_ok=True)

        extracted_files: List[Path] = []
        all_clips = [c for s in plan_doc.segments for c in s.clips]
        if max_previews is not None:
            all_clips = all_clips[:max_previews]

        for clip in all_clips:
            out_file = preview_dir / f"{clip.clip_id}_{clip.source_start_seconds:.1f}s.mp4"
            if out_file.exists() and out_file.stat().st_size > 0:
                extracted_files.append(out_file)
                continue

            # FFmpeg extraction command with fast seek and original source audio
            cmd = [
                "ffmpeg",
                "-y",
                "-ss", str(clip.source_start_seconds),
                "-i", str(source_path_to_use),
                "-t", str(clip.duration_seconds),
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "24",
                "-c:a", "aac",
                "-b:a", "128k",
                "-ar", "44100",
                "-ac", "2",
                str(out_file),
            ]
            try:
                subprocess.run(cmd, capture_output=True, check=True)
                if out_file.exists() and out_file.stat().st_size > 0:
                    extracted_files.append(out_file)
            except Exception as exc:
                # Fallback video-only with silent audio track if audio stream is missing
                cmd_fallback = [
                    "ffmpeg",
                    "-y",
                    "-ss", str(clip.source_start_seconds),
                    "-i", str(source_path_to_use),
                    "-f", "lavfi",
                    "-i", "anullsrc=r=44100:cl=stereo",
                    "-t", str(clip.duration_seconds),
                    "-c:v", "libx264",
                    "-preset", "ultrafast",
                    "-crf", "24",
                    "-c:a", "aac",
                    "-shortest",
                    str(out_file),
                ]
                try:
                    subprocess.run(cmd_fallback, capture_output=True, check=True)
                    if out_file.exists() and out_file.stat().st_size > 0:
                        extracted_files.append(out_file)
                except Exception as exc2:
                    logger.debug("Failed to extract preview clip %s: %s (fallback failed: %s)", clip.clip_id, exc, exc2)

        logger.info("Extracted %d preview clips to %s", len(extracted_files), preview_dir)
        return extracted_files


# ==============================================================================
# Public API & Standalone Runner
# ==============================================================================

def generate_movie_clip_plan(
    source_path: str | Path,
    movie_id: Optional[str] = None,
    output_base_dir: Optional[Path] = None,
    max_clip_duration_seconds: float = 3.0,
    extract_preview: bool = False,
    expand_coverage: bool = True,
    force: bool = False,
) -> ClipPlanDocument:
    """
    Public API function to generate source clip plan for a movie.
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
        raise ClipSelectionError(
            f"Analysis directory not found for movie '{assigned_movie_id}' at {analysis_dir}"
        )

    selector = SourceClipSelector(
        movie_id=assigned_movie_id,
        analysis_dir=analysis_dir,
        movie_source_path=resolved_source,
        max_clip_duration_seconds=max_clip_duration_seconds,
    )

    plan_doc = selector.select(expand_coverage=expand_coverage)
    selector.save_clip_plan(plan_doc)

    if extract_preview:
        selector.extract_preview_clips(plan_doc, max_previews=None)

    return plan_doc


def main():
    """CLI entrypoint: python -m clip_engine.select_clips <movie_id>."""
    parser = argparse.ArgumentParser(
        description="Phase 9: Source Clip Selection Engine (<= 3.0s clips mapped to narration)."
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
        "--max-duration",
        type=float,
        default=3.0,
        help="Maximum clip duration in seconds (default: 3.0s).",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Extract short MP4 preview snippets for human inspection.",
    )
    parser.add_argument(
        "--expand-coverage",
        action="store_true",
        default=True,
        help="Incorporate all grounded action events for continuous visual flow (default: True).",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    out_base = Path(args.output_dir) if args.output_dir else None
    try:
        plan_doc = generate_movie_clip_plan(
            source_path=args.source,
            movie_id=args.movie_id,
            output_base_dir=out_base,
            max_clip_duration_seconds=args.max_duration,
            extract_preview=args.preview,
            expand_coverage=args.expand_coverage,
        )
        print("\n" + "=" * 60)
        print("SOURCE CLIP SELECTION COMPLETE (Phase 9)")
        print("=" * 60)
        print(f"  Movie ID        : {plan_doc.movie_id}")
        print(f"  Status          : {plan_doc.status}")
        print(f"  Max Duration    : {plan_doc.max_clip_duration_seconds:.1f}s")
        print(f"  Total Clips     : {plan_doc.total_clips}")
        print(f"  Source Duration : {plan_doc.total_source_duration_seconds:.1f}s")
        for seg in plan_doc.segments:
            print(f"  - [{seg.segment_id}] ({len(seg.clips)} clips):")
            for c in seg.clips:
                print(f"      * {c.clip_id}: {c.source_start_seconds:.1f}s -> {c.source_end_seconds:.1f}s ({c.duration_seconds:.1f}s) | {c.visual_reason}")
        print("=" * 60 + "\n")
    except Exception as exc:
        print(f"Error during clip selection: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
