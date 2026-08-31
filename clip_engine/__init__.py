"""
Movie Review Engine - Clip Engine Module (Phase 9)
Selects and validates short supporting movie clips (<= 3.0s) mapped to narration segments.
"""

from .models import (
    SourceClip,
    SegmentClips,
    ClipPlanDocument,
)
from .validate_clips import (
    ValidationResult,
    ClipPlanValidator,
)
from .select_clips import (
    SourceClipSelector,
    generate_movie_clip_plan,
)

__all__ = [
    "SourceClip",
    "SegmentClips",
    "ClipPlanDocument",
    "ValidationResult",
    "ClipPlanValidator",
    "SourceClipSelector",
    "generate_movie_clip_plan",
]
