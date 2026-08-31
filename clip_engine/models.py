"""
Movie Review Engine - Phase 9: Source Clip Selection Data Models
Defines schemas for candidate source clips, segment clip mappings, and the final clip plan document.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SourceClip:
    """An individual bounded source movie clip snippet (max 3.0s duration)."""
    clip_id: str
    source_start_seconds: float
    source_end_seconds: float
    duration_seconds: float
    source_event_index: int
    source_scene_id: str
    visual_reason: str
    importance: int = 5  # 1 to 10 scale
    confidence: str = "high"  # "low", "medium", "high"
    continuity_exception: bool = False
    continuity_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "clip_id": self.clip_id,
            "source_start_seconds": round(self.source_start_seconds, 2),
            "source_end_seconds": round(self.source_end_seconds, 2),
            "duration_seconds": round(self.duration_seconds, 2),
            "source_event_index": self.source_event_index,
            "source_scene_id": self.source_scene_id,
            "visual_reason": self.visual_reason,
            "importance": self.importance,
            "confidence": self.confidence,
        }
        if self.continuity_exception:
            d["continuity_exception"] = True
            d["continuity_reason"] = self.continuity_reason or "Brief multi-angle reveal"
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SourceClip:
        return cls(
            clip_id=str(data.get("clip_id", "")),
            source_start_seconds=float(data.get("source_start_seconds", 0.0)),
            source_end_seconds=float(data.get("source_end_seconds", 0.0)),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            source_event_index=int(data.get("source_event_index", 0)),
            source_scene_id=str(data.get("source_scene_id", "")),
            visual_reason=str(data.get("visual_reason", "")),
            importance=int(data.get("importance", 5)),
            confidence=str(data.get("confidence", "high")),
            continuity_exception=bool(data.get("continuity_exception", False)),
            continuity_reason=data.get("continuity_reason"),
        )


@dataclass
class SegmentClips:
    """Mapping of a narration segment to its selected supporting source clips."""
    segment_id: str
    narration_text: str
    source_event_indices: List[int]
    clips: List[SourceClip] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "narration_text": self.narration_text,
            "source_event_indices": self.source_event_indices,
            "clips": [c.to_dict() if isinstance(c, SourceClip) else c for c in self.clips],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SegmentClips:
        return cls(
            segment_id=str(data.get("segment_id", "")),
            narration_text=str(data.get("narration_text", "")),
            source_event_indices=[int(idx) for idx in data.get("source_event_indices", [])],
            clips=[
                SourceClip.from_dict(c) if isinstance(c, dict) else c
                for c in data.get("clips", [])
            ],
        )


@dataclass
class ClipPlanDocument:
    """Complete source clip selection document (clip_plan.json)."""
    movie_id: str
    status: str  # "PARTIAL" or "COMPLETE"
    max_clip_duration_seconds: float = 3.0
    total_clips: int = 0
    total_source_duration_seconds: float = 0.0
    segments: List[SegmentClips] = field(default_factory=list)
    created_at: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "movie_id": self.movie_id,
            "status": self.status,
            "max_clip_duration_seconds": self.max_clip_duration_seconds,
            "total_clips": sum(len(s.clips) for s in self.segments),
            "total_source_duration_seconds": round(
                sum(c.duration_seconds for s in self.segments for c in s.clips), 2
            ),
            "segments": [
                s.to_dict() if isinstance(s, SegmentClips) else s
                for s in self.segments
            ],
            "created_at": self.created_at,
            "warnings": self.warnings,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ClipPlanDocument:
        return cls(
            movie_id=str(data.get("movie_id", "")),
            status=str(data.get("status", "PARTIAL")),
            max_clip_duration_seconds=float(data.get("max_clip_duration_seconds", 3.0)),
            total_clips=int(data.get("total_clips", 0)),
            total_source_duration_seconds=float(data.get("total_source_duration_seconds", 0.0)),
            segments=[
                SegmentClips.from_dict(s) if isinstance(s, dict) else s
                for s in data.get("segments", [])
            ],
            created_at=data.get("created_at"),
            warnings=[str(w) for w in data.get("warnings", [])],
        )
