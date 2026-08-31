"""
Movie Review Engine - Phase 9: Clip Plan Validator Module
Performs comprehensive validation of clip duration (<= 3.0s), boundary safety,
anti-contiguous constraints, chronological order, and source traceability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from clip_engine.models import ClipPlanDocument, SourceClip

logger = logging.getLogger("clip_engine.validate_clips")


@dataclass
class ValidationResult:
    """Validation report containing error and warning messages."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ClipPlanValidator:
    """
    Validates clip_plan.json documents against all editorial, safety,
    and anti-contiguous requirements.
    """

    def __init__(
        self,
        max_clip_duration_seconds: float = 3.0,
        min_separation_seconds: float = 5.0,
        movie_duration_seconds: Optional[float] = None,
    ):
        self.max_clip_duration_seconds = max_clip_duration_seconds
        self.min_separation_seconds = min_separation_seconds
        self.movie_duration_seconds = movie_duration_seconds

    def validate(self, plan_doc: ClipPlanDocument) -> ValidationResult:
        errors: List[str] = []
        warnings: List[str] = []

        all_clips: List[SourceClip] = []
        for seg in plan_doc.segments:
            for clip in seg.clips:
                all_clips.append(clip)

        if not all_clips and plan_doc.total_clips > 0:
            errors.append("Plan reports clips but no clips were found in segments.")

        # 1. Validate individual clip properties
        seen_clip_ids = set()
        for idx, clip in enumerate(all_clips):
            # Clip ID uniqueness
            if not clip.clip_id:
                errors.append(f"Clip at index {idx} has an empty clip_id.")
            elif clip.clip_id in seen_clip_ids:
                errors.append(f"Duplicate clip_id found: '{clip.clip_id}'.")
            seen_clip_ids.add(clip.clip_id)

            # Duration constraint (<= 3.0s hard editorial limit)
            if clip.duration_seconds <= 0.0:
                errors.append(f"Clip '{clip.clip_id}' has non-positive duration: {clip.duration_seconds:.2f}s.")
            elif clip.duration_seconds > self.max_clip_duration_seconds + 0.01:
                errors.append(
                    f"Clip '{clip.clip_id}' exceeds maximum editorial duration ({self.max_clip_duration_seconds}s): "
                    f"{clip.duration_seconds:.2f}s."
                )

            # Timestamp bounds safety
            if clip.source_start_seconds < 0.0:
                errors.append(f"Clip '{clip.clip_id}' has negative start timestamp: {clip.source_start_seconds:.2f}s.")

            if clip.source_end_seconds <= clip.source_start_seconds:
                errors.append(
                    f"Clip '{clip.clip_id}' has invalid range: start={clip.source_start_seconds:.2f}s, "
                    f"end={clip.source_end_seconds:.2f}s."
                )

            calc_dur = round(clip.source_end_seconds - clip.source_start_seconds, 2)
            if abs(calc_dur - clip.duration_seconds) > 0.15:
                errors.append(
                    f"Clip '{clip.clip_id}' duration mismatch: recorded {clip.duration_seconds:.2f}s "
                    f"vs calculated {calc_dur:.2f}s."
                )

            if self.movie_duration_seconds and self.movie_duration_seconds > 0.0:
                if clip.source_end_seconds > self.movie_duration_seconds + 1.0:
                    errors.append(
                        f"Clip '{clip.clip_id}' end time ({clip.source_end_seconds:.2f}s) exceeds "
                        f"movie duration ({self.movie_duration_seconds:.2f}s)."
                    )

            # Source Traceability
            if clip.source_event_index <= 0:
                errors.append(f"Clip '{clip.clip_id}' has invalid source_event_index: {clip.source_event_index}.")
            if not clip.source_scene_id:
                errors.append(f"Clip '{clip.clip_id}' has missing source_scene_id.")
            if not clip.visual_reason:
                warnings.append(f"Clip '{clip.clip_id}' has empty visual_reason.")

        # 2. Anti-Contiguous Scene Reconstruction Check
        for i in range(len(all_clips) - 1):
            curr_clip = all_clips[i]
            next_clip = all_clips[i + 1]

            # If adjacent clips overlap or touch without gap (< 0.5s)
            gap = next_clip.source_start_seconds - curr_clip.source_end_seconds
            if gap < 0.0:
                errors.append(
                    f"Clips '{curr_clip.clip_id}' ({curr_clip.source_start_seconds:.1f}-{curr_clip.source_end_seconds:.1f}s) and "
                    f"'{next_clip.clip_id}' ({next_clip.source_start_seconds:.1f}-{next_clip.source_end_seconds:.1f}s) overlap."
                )
            elif gap < 0.5 and not (curr_clip.continuity_exception or next_clip.continuity_exception):
                errors.append(
                    f"Contiguous reconstruction detected between '{curr_clip.clip_id}' and '{next_clip.clip_id}' "
                    f"(gap: {gap:.2f}s) without continuity_exception."
                )
            elif gap < self.min_separation_seconds and not (curr_clip.continuity_exception or next_clip.continuity_exception):
                warnings.append(
                    f"Close temporal proximity ({gap:.1f}s gap) between '{curr_clip.clip_id}' and '{next_clip.clip_id}'."
                )

        # 3. Overall Chronological Ordering
        for i in range(len(all_clips) - 1):
            curr_clip = all_clips[i]
            next_clip = all_clips[i + 1]
            if next_clip.source_start_seconds < curr_clip.source_start_seconds - 5.0:
                warnings.append(
                    f"Non-chronological clip sequence: '{curr_clip.clip_id}' at {curr_clip.source_start_seconds:.1f}s "
                    f"followed by '{next_clip.clip_id}' at {next_clip.source_start_seconds:.1f}s."
                )

        is_valid = (len(errors) == 0)
        return ValidationResult(valid=is_valid, errors=errors, warnings=warnings)
