"""
Movie Review Engine - Phase 10: Timeline Builder Module
Builds chronological preview timelines interleaving source movie clips and visual placeholders
matching estimated narration segment durations.
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from clip_engine.models import ClipPlanDocument, SegmentClips, SourceClip
from config.paths import ANALYSIS_DIR
from narration_engine.models import ScriptDocument, ScriptSegment
from preview_engine.models import PreviewManifest, SegmentTimeline, TimelineItem

logger = logging.getLogger("preview_engine.timeline_builder")


class TimelineBuildError(Exception):
    """Base exception for timeline construction errors."""
    pass


class PreviewTimelineBuilder:
    """
    Constructs editorial preview timeline mapping script segments to clips & placeholders.
    """

    def __init__(
        self,
        movie_id: str,
        analysis_dir: Path,
    ):
        self.movie_id = movie_id
        self.analysis_dir = analysis_dir

        self.script_file = analysis_dir / "script.json"
        self.clip_plan_file = analysis_dir / "clip_plan.json"
        self.preview_dir = analysis_dir / "preview"
        self.clip_previews_dir = analysis_dir / "clip_previews"
        self.output_manifest_file = self.preview_dir / "preview_manifest.json"

    def load_inputs(self) -> Tuple[ScriptDocument, ClipPlanDocument]:
        """Load script.json and clip_plan.json."""
        if not self.script_file.exists():
            raise TimelineBuildError(f"script.json not found in {self.analysis_dir}")
        if not self.clip_plan_file.exists():
            raise TimelineBuildError(f"clip_plan.json not found in {self.analysis_dir}")

        with open(self.script_file, "r", encoding="utf-8") as f:
            script_doc = ScriptDocument.from_dict(json.load(f))

        with open(self.clip_plan_file, "r", encoding="utf-8") as f:
            clip_plan_doc = ClipPlanDocument.from_dict(json.load(f))

        return script_doc, clip_plan_doc

    def _resolve_clip_file(self, clip: SourceClip) -> Optional[Path]:
        """Check if pre-extracted clip file exists in clip_previews directory."""
        if not self.clip_previews_dir.exists():
            return None

        # Look for standard named preview files
        cand1 = self.clip_previews_dir / f"{clip.clip_id}_{clip.source_start_seconds:.1f}s.mp4"
        if cand1.exists() and cand1.stat().st_size > 0:
            return cand1

        for p in self.clip_previews_dir.glob(f"{clip.clip_id}*.mp4"):
            if p.stat().st_size > 0:
                return p

        return None

    def build_timeline(self) -> PreviewManifest:
        """
        Interleave source clips with placeholders to match segment narration durations.
        """
        script_doc, clip_plan_doc = self.load_inputs()

        clips_by_seg: Dict[str, List[SourceClip]] = {
            s.segment_id: s.clips for s in clip_plan_doc.segments
        }

        segment_timelines: List[SegmentTimeline] = []
        current_time = 0.0
        item_counter = 1

        for seg_idx, script_seg in enumerate(script_doc.segments, start=1):
            seg_start = current_time
            seg_duration = max(5.0, script_seg.estimated_duration_seconds)
            seg_end = round(seg_start + seg_duration, 2)

            clips = clips_by_seg.get(script_seg.segment_id, [])
            total_clip_time = sum(c.duration_seconds for c in clips)

            seg_items: List[TimelineItem] = []

            if not clips:
                # No clips (e.g. continuation marker): 100% placeholder card
                item = TimelineItem(
                    item_id=f"ITEM_{item_counter:03d}",
                    item_type="PLACEHOLDER",
                    start_time_seconds=round(seg_start, 2),
                    end_time_seconds=round(seg_end, 2),
                    duration_seconds=round(seg_duration, 2),
                    caption_text=script_seg.text,
                    label=f"Narration Context: {script_seg.segment_id}",
                    has_source_audio=False,
                    audio_track_type="SILENCE",
                    future_narration_point=True,
                    audio_bus_mapping={"narration_bus_ducking_db": 0.0, "source_audio_bus_db": -99.0},
                )
                seg_items.append(item)
                item_counter += 1
            else:
                # Distribute clips and placeholders
                # Placeholders fill the gap between clips
                remaining_placeholder_time = max(0.0, seg_duration - total_clip_time)
                # Divide into intervals: before first clip, between clips, after last clip
                interval_count = len(clips) + 1
                gap_duration = round(remaining_placeholder_time / interval_count, 2)

                t_cursor = seg_start

                for c_idx, clip in enumerate(clips):
                    # 1. Leading/Intermediary Placeholder (Still/Freeze hold)
                    if gap_duration >= 0.5:
                        p_start = t_cursor
                        p_end = round(t_cursor + gap_duration, 2)
                        p_dur = round(p_end - p_start, 2)
                        seg_items.append(
                            TimelineItem(
                                item_id=f"ITEM_{item_counter:03d}",
                                item_type="PLACEHOLDER",
                                start_time_seconds=round(p_start, 2),
                                end_time_seconds=round(p_end, 2),
                                duration_seconds=p_dur,
                                caption_text=script_seg.text,
                                label=f"Scene Hold: {clip.source_scene_id}",
                                has_source_audio=False,
                                audio_track_type="SILENCE",
                                future_narration_point=True,
                                audio_bus_mapping={"narration_bus_ducking_db": 0.0, "source_audio_bus_db": -99.0},
                            )
                        )
                        item_counter += 1
                        t_cursor = p_end

                    # 2. Source Clip (Moving video + original source audio)
                    c_start = t_cursor
                    c_end = round(t_cursor + clip.duration_seconds, 2)
                    c_dur = round(c_end - c_start, 2)
                    clip_file = self._resolve_clip_file(clip)
                    seg_items.append(
                        TimelineItem(
                            item_id=f"ITEM_{item_counter:03d}",
                            item_type="SOURCE_CLIP",
                            start_time_seconds=round(c_start, 2),
                            end_time_seconds=round(c_end, 2),
                            duration_seconds=c_dur,
                            source_clip_id=clip.clip_id,
                            source_event_index=clip.source_event_index,
                            source_scene_id=clip.source_scene_id,
                            caption_text=script_seg.text,
                            clip_path=str(clip_file) if clip_file else None,
                            label=clip.visual_reason,
                            has_source_audio=True,
                            audio_track_type="SOURCE_AUDIO",
                            future_narration_point=True,
                            audio_bus_mapping={"narration_bus_ducking_db": -14.0, "source_audio_bus_db": 0.0},
                        )
                    )
                    item_counter += 1
                    t_cursor = c_end

                # 3. Trailing Placeholder
                if t_cursor < seg_end - 0.2:
                    p_start = t_cursor
                    p_end = seg_end
                    p_dur = round(p_end - p_start, 2)
                    seg_items.append(
                        TimelineItem(
                            item_id=f"ITEM_{item_counter:03d}",
                            item_type="PLACEHOLDER",
                            start_time_seconds=round(p_start, 2),
                            end_time_seconds=round(p_end, 2),
                            duration_seconds=p_dur,
                            caption_text=script_seg.text,
                            label="Narrative Transition",
                            has_source_audio=False,
                            audio_track_type="SILENCE",
                            future_narration_point=True,
                            audio_bus_mapping={"narration_bus_ducking_db": 0.0, "source_audio_bus_db": -99.0},
                        )
                    )
                    item_counter += 1
                    t_cursor = seg_end

            seg_timeline = SegmentTimeline(
                segment_id=script_seg.segment_id,
                start_time_seconds=round(seg_start, 2),
                end_time_seconds=round(seg_end, 2),
                duration_seconds=round(seg_duration, 2),
                narration_text=script_seg.text,
                items=seg_items,
            )
            segment_timelines.append(seg_timeline)
            current_time = seg_end

        total_duration = round(current_time, 2)
        all_items = [it for s in segment_timelines for it in s.items]
        clips_count = sum(1 for it in all_items if it.item_type == "SOURCE_CLIP")
        placeholders_count = sum(1 for it in all_items if it.item_type == "PLACEHOLDER")

        manifest = PreviewManifest(
            movie_id=self.movie_id,
            status=script_doc.status,
            total_duration_seconds=total_duration,
            resolution="1920x1080",
            fps=30.0,
            total_segments=len(segment_timelines),
            total_source_clips=clips_count,
            total_placeholders=placeholders_count,
            total_source_footage_seconds=round(sum(it.duration_seconds for it in all_items if it.item_type == "SOURCE_CLIP"), 2),
            total_placeholder_seconds=round(sum(it.duration_seconds for it in all_items if it.item_type == "PLACEHOLDER"), 2),
            video_output_path=str(self.preview_dir / "rough_preview.mp4"),
            segments=segment_timelines,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            warnings=script_doc.warnings,
        )

        return manifest

    def save_manifest(self, manifest: PreviewManifest) -> Path:
        """Save preview_manifest.json to analysis/<movie_id>/preview/preview_manifest.json."""
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_manifest_file, "w", encoding="utf-8") as f:
            f.write(manifest.to_json(indent=2))
        logger.info("Saved preview manifest to %s", self.output_manifest_file)
        return self.output_manifest_file
