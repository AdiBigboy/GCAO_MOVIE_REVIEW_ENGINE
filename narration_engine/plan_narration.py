"""
Movie Review Engine - Phase 8: Narrative Planner Module
Transforms story.json into a structured narrative blueprint (narrative_plan.json).
Allocates target durations based on narrative importance and suppresses low-value filler.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from config.paths import ANALYSIS_DIR, PROJECT_ROOT, get_movie_source_path
from movie_analyzer.ingest import sanitize_movie_id
from narration_engine.models import (
    NarrationSegmentPlan,
    NarrativePlanDocument,
)
from story_engine.models import (
    StoryBeat,
    StoryDocument,
)

logger = logging.getLogger("narration_engine.plan_narration")


class NarrativePlanError(Exception):
    """Base exception for narrative planning errors."""
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


class NarrativePlanner:
    """
    Plans narration structure, allocating time proportionally by narrative significance
    while preventing empty scene filler.
    """

    def __init__(
        self,
        movie_id: str,
        analysis_dir: Path,
        full_movie_target_seconds: float = 900.0,  # 15 minutes default target for full movie
        words_per_minute: float = 150.0,
    ):
        self.movie_id = movie_id
        self.analysis_dir = analysis_dir
        self.full_movie_target_seconds = full_movie_target_seconds
        self.words_per_minute = words_per_minute

        self.story_file = analysis_dir / "story.json"
        self.events_file = analysis_dir / "events.json"
        self.timeline_file = analysis_dir / "timeline_index.json"
        self.metadata_file = analysis_dir / "movie_metadata.json"
        self.output_plan_file = analysis_dir / "narrative_plan.json"

    def load_inputs(self) -> Tuple[StoryDocument, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Load story.json and auxiliary metadata."""
        if not self.story_file.exists():
            raise NarrativePlanError(f"story.json not found in {self.analysis_dir}")

        with open(self.story_file, "r", encoding="utf-8") as f:
            story_raw = json.load(f)
        story_doc = StoryDocument.from_dict(story_raw)

        events_data: Dict[str, Any] = {}
        if self.events_file.exists():
            try:
                with open(self.events_file, "r", encoding="utf-8") as f:
                    events_data = json.load(f)
            except Exception as exc:
                logger.debug("Failed to load events.json: %s", exc)

        timeline_data: Dict[str, Any] = {}
        if self.timeline_file.exists():
            try:
                with open(self.timeline_file, "r", encoding="utf-8") as f:
                    timeline_data = json.load(f)
            except Exception as exc:
                logger.debug("Failed to load timeline_index.json: %s", exc)

        metadata_data: Dict[str, Any] = {}
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    metadata_data = json.load(f)
            except Exception as exc:
                logger.debug("Failed to load movie_metadata.json: %s", exc)

        return story_doc, events_data, timeline_data, metadata_data

    def calculate_target_total_duration(
        self,
        story_doc: StoryDocument,
        events_data: Dict[str, Any],
        timeline_data: Dict[str, Any],
        metadata_data: Dict[str, Any],
    ) -> float:
        """
        Calculate total target script duration.
        Full movie defaults to ~15 min (900s).
        Partial movie scales proportionally to coverage.
        """
        if story_doc.story_status == "COMPLETE":
            return float(self.full_movie_target_seconds)

        # For partial movie, compute coverage ratio
        total_samples = _safe_int(timeline_data.get("total_samples", 0))
        analyzed_events = len(_safe_list(events_data.get("events", [])))
        if analyzed_events == 0:
            analyzed_events = sum(len(s.event_indices) for s in story_doc.scenes)

        if total_samples > 0 and analyzed_events > 0:
            coverage_ratio = min(1.0, analyzed_events / total_samples)
            scaled_target = self.full_movie_target_seconds * coverage_ratio
            # Floor at 45 seconds, cap at proportional estimate + buffer
            target = max(45.0, min(scaled_target * 1.25, self.full_movie_target_seconds))
            return round(target, 1)

        # Fallback: estimate from number of events (~4-6 seconds of narration per analyzed event)
        target = max(45.0, min(float(analyzed_events * 5.0), 300.0))
        return round(target, 1)

    def plan(self) -> NarrativePlanDocument:
        """
        Build the complete NarrativePlanDocument from story beats, scenes, and causality.
        """
        story_doc, events_data, timeline_data, metadata_data = self.load_inputs()

        target_total_seconds = self.calculate_target_total_duration(
            story_doc=story_doc,
            events_data=events_data,
            timeline_data=timeline_data,
            metadata_data=metadata_data,
        )

        scenes_by_id = {s.scene_id: s for s in story_doc.scenes}
        beats = story_doc.beats

        # If no story beats present, create synthetic beats from scenes
        if not beats and story_doc.scenes:
            beats = [
                StoryBeat(
                    beat_id="BEAT_001",
                    title="Movie Progression",
                    description=story_doc.story_summary or "Main sequence",
                    supporting_scenes=[s.scene_id for s in story_doc.scenes],
                    characters_involved=list({c for s in story_doc.scenes for c in s.characters}),
                    cause="Initial events unfold.",
                    consequence="Leads to sequential outcomes.",
                    importance=5,
                    uncertainty="medium",
                )
            ]

        # Filter and aggregate beats into narration segments
        # Anti-filler rule: Low importance transition beats (importance <= 3 with no character/plot)
        # get minimal duration or get merged with adjacent beats.
        plan_segments: List[NarrationSegmentPlan] = []
        raw_segment_weights: List[float] = []

        for b_idx, beat in enumerate(beats, start=1):
            supp_scenes = [scenes_by_id[sid] for sid in beat.supporting_scenes if sid in scenes_by_id]
            all_event_indices = [idx for s in supp_scenes for idx in s.event_indices]
            all_event_indices = sorted(list(set(all_event_indices)))

            # Determine purpose and key points
            title_lower = beat.title.lower()
            funcs = [s.scene_function for s in supp_scenes]

            key_points: List[str] = []
            if beat.cause:
                key_points.append(f"Punca/Latar: {beat.cause}")
            if beat.consequence:
                key_points.append(f"Kesan/Perkembangan: {beat.consequence}")

            # Highlight relevant causal links
            for link in story_doc.causal_links:
                if any(ev_id in all_event_indices for ev_id in link.evidence_events):
                    kp = f"Sebab-Akibat: {link.cause} -> {link.effect}"
                    if kp not in key_points:
                        key_points.append(kp)

            if "prologue" in title_lower or ("SETUP" in funcs and "DISCOVERY" in funcs):
                purpose = "opening_hook_and_audio_discovery"
                spoiler_level = "none"
            elif "storm" in title_lower or "escalation" in title_lower or "conflict" in title_lower:
                purpose = "rising_action_and_crisis"
                spoiler_level = "moderate"
            elif "nightclub" in title_lower or "birthday" in title_lower or "introduction" in title_lower:
                purpose = "setting_shift_and_character_intro"
                spoiler_level = "none"
            elif "climax" in title_lower or "CLIMAX" in funcs:
                purpose = "peak_climax"
                spoiler_level = "high"
            elif "transition" in title_lower:
                purpose = "atmospheric_transition"
                spoiler_level = "none"
            else:
                purpose = "story_progression"
                spoiler_level = "none"

            # Assign importance weight (1-10)
            importance = max(1, min(10, beat.importance))
            # If purely atmospheric transition without characters, dampen weight
            if purpose == "atmospheric_transition" and not beat.characters_involved:
                weight = 1.0
            else:
                weight = float(importance)

            raw_segment_weights.append(weight)

            plan_segments.append(
                NarrationSegmentPlan(
                    segment_id=f"SEGMENT_{b_idx:03d}",
                    story_beat_ids=[beat.beat_id],
                    scene_ids=[s.scene_id for s in supp_scenes],
                    event_indices=all_event_indices,
                    purpose=purpose,
                    importance=importance,
                    target_duration_seconds=30.0,  # will be re-proportioned below
                    spoiler_level=spoiler_level,
                    uncertainty=beat.uncertainty,
                    key_points=key_points,
                )
            )

        # Allocate target durations proportionally
        total_weight = sum(raw_segment_weights) if raw_segment_weights else 1.0
        for seg, weight in zip(plan_segments, raw_segment_weights):
            proportional_sec = (weight / total_weight) * target_total_seconds
            # Min 10s for active segments, max bounded
            allocated = max(10.0, proportional_sec)
            seg.target_duration_seconds = round(allocated, 1)

        # Re-normalize to exact target_total_seconds
        sum_allocated = sum(s.target_duration_seconds for s in plan_segments)
        if sum_allocated > 0:
            scale_factor = target_total_seconds / sum_allocated
            for seg in plan_segments:
                seg.target_duration_seconds = round(seg.target_duration_seconds * scale_factor, 1)

        plan_doc = NarrativePlanDocument(
            movie_id=self.movie_id,
            status=story_doc.story_status,
            target_total_duration_seconds=round(target_total_seconds, 1),
            total_segments=len(plan_segments),
            segments=plan_segments,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

        return plan_doc

    def save_narrative_plan(self, plan_doc: NarrativePlanDocument) -> Path:
        """Save plan_doc to analysis/<movie_id>/narrative_plan.json."""
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_plan_file, "w", encoding="utf-8") as f:
            f.write(plan_doc.to_json(indent=2))
        logger.info("Saved narrative plan to %s", self.output_plan_file)
        return self.output_plan_file


# ==============================================================================
# Public API & Standalone Runner
# ==============================================================================

def generate_movie_narrative_plan(
    source_path: str | Path,
    movie_id: Optional[str] = None,
    output_base_dir: Optional[Path] = None,
    full_movie_target_seconds: float = 900.0,
    force: bool = False,
) -> NarrativePlanDocument:
    """
    Public API function to generate narrative plan for a movie.
    """
    assigned_movie_id = movie_id
    if not assigned_movie_id:
        try:
            resolved_source = get_movie_source_path(source_path)
            assigned_movie_id = sanitize_movie_id(resolved_source.name)
        except Exception:
            assigned_movie_id = sanitize_movie_id(str(source_path))

    out_base = output_base_dir or ANALYSIS_DIR
    analysis_dir = out_base / assigned_movie_id

    if not analysis_dir.exists():
        raise NarrativePlanError(
            f"Analysis directory not found for movie '{assigned_movie_id}' at {analysis_dir}"
        )

    planner = NarrativePlanner(
        movie_id=assigned_movie_id,
        analysis_dir=analysis_dir,
        full_movie_target_seconds=full_movie_target_seconds,
    )

    plan_doc = planner.plan()
    planner.save_narrative_plan(plan_doc)
    return plan_doc


def main():
    """CLI entrypoint: python -m narration_engine.plan_narration <movie_id>."""
    parser = argparse.ArgumentParser(
        description="Phase 8 Step 1: Generate structured narrative plan (narrative_plan.json)."
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
        "--target-duration",
        type=float,
        default=900.0,
        help="Target duration in seconds for full movie (default: 900s / 15 mins).",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    out_base = Path(args.output_dir) if args.output_dir else None
    try:
        plan_doc = generate_movie_narrative_plan(
            source_path=args.source,
            movie_id=args.movie_id,
            output_base_dir=out_base,
            full_movie_target_seconds=args.target_duration,
        )
        print("\n" + "=" * 60)
        print("NARRATIVE PLANNING COMPLETE (Phase 8 - Step 1)")
        print("=" * 60)
        print(f"  Movie ID        : {plan_doc.movie_id}")
        print(f"  Status          : {plan_doc.status}")
        print(f"  Target Duration : {plan_doc.target_total_duration_seconds:.1f}s")
        print(f"  Total Segments  : {plan_doc.total_segments}")
        for seg in plan_doc.segments:
            print(f"    - [{seg.segment_id}] {seg.purpose} ({seg.target_duration_seconds:.1f}s | Imp: {seg.importance})")
        print("=" * 60 + "\n")
    except Exception as exc:
        print(f"Error during narrative planning: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
