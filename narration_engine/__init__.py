"""
Movie Review Engine - Narration Engine Module (Phase 8)
Plans narrative structure and generates natural Malay storytelling / review scripts (script.json).
"""

from .models import (
    NarrationSegmentPlan,
    NarrativePlanDocument,
    ScriptSegment,
    ScriptDocument,
)
from .plan_narration import (
    NarrativePlanner,
    generate_movie_narrative_plan,
)
from .generate_script import (
    MalayStorytellerEngine,
    generate_movie_script,
)

__all__ = [
    "NarrationSegmentPlan",
    "NarrativePlanDocument",
    "ScriptSegment",
    "ScriptDocument",
    "NarrativePlanner",
    "generate_movie_narrative_plan",
    "MalayStorytellerEngine",
    "generate_movie_script",
]
