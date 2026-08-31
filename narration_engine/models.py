"""
Movie Review Engine - Phase 8: Narration & Script Generation Data Models
Defines structured schemas for narrative planning, script segments, and final Malay storytelling script document.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class NarrationSegmentPlan:
    """Planning blueprint for an individual narration segment."""
    segment_id: str
    story_beat_ids: List[str]
    scene_ids: List[str]
    event_indices: List[int]
    purpose: str  # e.g., "hook_and_setup", "reveal_and_threat", "escalation", "character_intro"
    importance: int  # 1 to 10 scale
    target_duration_seconds: float
    spoiler_level: str = "none"  # "none", "moderate", "high"
    uncertainty: str = "low"  # "low", "medium", "high"
    key_points: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NarrationSegmentPlan:
        return cls(
            segment_id=str(data.get("segment_id", "")),
            story_beat_ids=[str(b) for b in data.get("story_beat_ids", [])],
            scene_ids=[str(s) for s in data.get("scene_ids", [])],
            event_indices=[int(idx) for idx in data.get("event_indices", [])],
            purpose=str(data.get("purpose", "story_progression")),
            importance=int(data.get("importance", 5)),
            target_duration_seconds=float(data.get("target_duration_seconds", 30.0)),
            spoiler_level=str(data.get("spoiler_level", "none")),
            uncertainty=str(data.get("uncertainty", "low")),
            key_points=[str(k) for k in data.get("key_points", [])],
        )


@dataclass
class NarrativePlanDocument:
    """Complete narrative blueprint document (narrative_plan.json)."""
    movie_id: str
    status: str  # "PARTIAL" or "COMPLETE"
    target_total_duration_seconds: float
    total_segments: int
    segments: List[NarrationSegmentPlan] = field(default_factory=list)
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "movie_id": self.movie_id,
            "status": self.status,
            "target_total_duration_seconds": round(self.target_total_duration_seconds, 2),
            "total_segments": len(self.segments),
            "segments": [
                s.to_dict() if isinstance(s, NarrationSegmentPlan) else s
                for s in self.segments
            ],
            "created_at": self.created_at,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NarrativePlanDocument:
        return cls(
            movie_id=str(data.get("movie_id", "")),
            status=str(data.get("status", "PARTIAL")),
            target_total_duration_seconds=float(data.get("target_total_duration_seconds", 0.0)),
            total_segments=int(data.get("total_segments", 0)),
            segments=[
                NarrationSegmentPlan.from_dict(s) if isinstance(s, dict) else s
                for s in data.get("segments", [])
            ],
            created_at=data.get("created_at"),
        )


@dataclass
class ScriptSegment:
    """An individual written narration paragraph with strict source mapping."""
    segment_id: str
    text: str
    estimated_duration_seconds: float
    word_count: int
    source_beat_ids: List[str]
    source_scene_ids: List[str]
    source_event_indices: List[int]
    importance: int
    uncertainty: str = "low"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "text": self.text,
            "estimated_duration_seconds": round(self.estimated_duration_seconds, 2),
            "word_count": self.word_count,
            "source_beat_ids": self.source_beat_ids,
            "source_scene_ids": self.source_scene_ids,
            "source_event_indices": self.source_event_indices,
            "importance": self.importance,
            "uncertainty": self.uncertainty,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ScriptSegment:
        return cls(
            segment_id=str(data.get("segment_id", "")),
            text=str(data.get("text", "")),
            estimated_duration_seconds=float(data.get("estimated_duration_seconds", 0.0)),
            word_count=int(data.get("word_count", 0)),
            source_beat_ids=[str(b) for b in data.get("source_beat_ids", [])],
            source_scene_ids=[str(s) for s in data.get("source_scene_ids", [])],
            source_event_indices=[int(idx) for idx in data.get("source_event_indices", [])],
            importance=int(data.get("importance", 5)),
            uncertainty=str(data.get("uncertainty", "low")),
        )


@dataclass
class ScriptDocument:
    """Complete storytelling script document (script.json)."""
    movie_id: str
    status: str  # "PARTIAL" or "COMPLETE"
    language: str  # "ms-MY"
    target_total_duration_seconds: float
    estimated_total_duration_seconds: float
    total_words: int
    words_per_minute: float
    segments: List[ScriptSegment] = field(default_factory=list)
    full_script: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "movie_id": self.movie_id,
            "status": self.status,
            "language": self.language,
            "target_total_duration_seconds": round(self.target_total_duration_seconds, 2),
            "estimated_total_duration_seconds": round(self.estimated_total_duration_seconds, 2),
            "total_words": self.total_words,
            "words_per_minute": self.words_per_minute,
            "segments": [
                s.to_dict() if isinstance(s, ScriptSegment) else s
                for s in self.segments
            ],
            "full_script": self.full_script,
            "warnings": self.warnings,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ScriptDocument:
        return cls(
            movie_id=str(data.get("movie_id", "")),
            status=str(data.get("status", "PARTIAL")),
            language=str(data.get("language", "ms-MY")),
            target_total_duration_seconds=float(data.get("target_total_duration_seconds", 0.0)),
            estimated_total_duration_seconds=float(data.get("estimated_total_duration_seconds", 0.0)),
            total_words=int(data.get("total_words", 0)),
            words_per_minute=float(data.get("words_per_minute", 150.0)),
            segments=[
                ScriptSegment.from_dict(s) if isinstance(s, dict) else s
                for s in data.get("segments", [])
            ],
            full_script=str(data.get("full_script", "")),
            warnings=[str(w) for w in data.get("warnings", [])],
        )
