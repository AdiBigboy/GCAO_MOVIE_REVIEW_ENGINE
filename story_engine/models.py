"""
Movie Review Engine - Phase 7: Story Reconstruction Data Models
Defines structured schemas for scenes, story beats, causal links, character arcs, and story document.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ProtagonistCandidate:
    """A character identified as a potential protagonist or key narrative lead."""
    character_id: str
    canonical_name: Optional[str]
    prominence_score: float  # 0.0 to 1.0
    screen_time_ratio: float  # fraction of analyzed duration character is present
    event_count: int
    dialogue_count: int
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProtagonistCandidate:
        return cls(
            character_id=str(data.get("character_id", "")),
            canonical_name=data.get("canonical_name"),
            prominence_score=float(data.get("prominence_score", 0.0)),
            screen_time_ratio=float(data.get("screen_time_ratio", 0.0)),
            event_count=int(data.get("event_count", 0)),
            dialogue_count=int(data.get("dialogue_count", 0)),
            rationale=str(data.get("rationale", "")),
        )


@dataclass
class SceneRecord:
    """A cohesive narrative scene grouping adjacent related timeline events."""
    scene_id: str
    start_seconds: float
    end_seconds: float
    event_indices: List[int]
    characters: List[str]
    location: str
    summary: str
    scene_function: str  # SETUP, INTRODUCTION, CONFLICT, DISCOVERY, ESCALATION, REVEAL, TWIST, CLIMAX, RESOLUTION, TRANSITION
    importance: int  # 1 to 10 scale

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SceneRecord:
        return cls(
            scene_id=str(data.get("scene_id", "")),
            start_seconds=float(data.get("start_seconds", 0.0)),
            end_seconds=float(data.get("end_seconds", 0.0)),
            event_indices=[int(idx) for idx in data.get("event_indices", [])],
            characters=[str(c) for c in data.get("characters", [])],
            location=str(data.get("location", "Unspecified")),
            summary=str(data.get("summary", "")),
            scene_function=str(data.get("scene_function", "SETUP")),
            importance=int(data.get("importance", 5)),
        )


@dataclass
class StoryBeat:
    """A macro-level narrative beat combining one or more scenes."""
    beat_id: str
    title: str
    description: str
    supporting_scenes: List[str]
    characters_involved: List[str]
    cause: Optional[str] = None
    consequence: Optional[str] = None
    importance: int = 5  # 1 to 10 scale
    uncertainty: str = "medium"  # low, medium, high

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StoryBeat:
        return cls(
            beat_id=str(data.get("beat_id", "")),
            title=str(data.get("title", "")),
            description=str(data.get("description", "")),
            supporting_scenes=[str(s) for s in data.get("supporting_scenes", [])],
            characters_involved=[str(c) for c in data.get("characters_involved", [])],
            cause=data.get("cause"),
            consequence=data.get("consequence"),
            importance=int(data.get("importance", 5)),
            uncertainty=str(data.get("uncertainty", "medium")),
        )


@dataclass
class CausalLink:
    """An evidence-backed causal relationship connecting actions to outcomes."""
    cause: str
    effect: str
    confidence: str  # low, medium, high
    evidence_events: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CausalLink:
        return cls(
            cause=str(data.get("cause", "")),
            effect=str(data.get("effect", "")),
            confidence=str(data.get("confidence", "medium")),
            evidence_events=[int(e) for e in data.get("evidence_events", [])],
        )


@dataclass
class CharacterArc:
    """Grounded character progression across the movie timeline."""
    character_id: str
    canonical_name: Optional[str]
    introduction: str
    objective: str
    conflict: str
    change: str
    major_decisions: List[str]
    outcome: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CharacterArc:
        return cls(
            character_id=str(data.get("character_id", "")),
            canonical_name=data.get("canonical_name"),
            introduction=str(data.get("introduction", "")),
            objective=str(data.get("objective", "unknown")),
            conflict=str(data.get("conflict", "none identified")),
            change=str(data.get("change", "none identified")),
            major_decisions=[str(d) for d in data.get("major_decisions", [])],
            outcome=str(data.get("outcome", "unresolved")),
        )


@dataclass
class StoryDocument:
    """Complete structured movie story reconstruction document (story.json)."""
    movie_id: str
    story_status: str  # "PARTIAL" or "COMPLETE"
    protagonist_candidates: List[ProtagonistCandidate] = field(default_factory=list)
    scenes: List[SceneRecord] = field(default_factory=list)
    beats: List[StoryBeat] = field(default_factory=list)
    causal_links: List[CausalLink] = field(default_factory=list)
    character_arcs: List[CharacterArc] = field(default_factory=list)
    story_summary: str = ""
    open_uncertainties: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "movie_id": self.movie_id,
            "story_status": self.story_status,
            "protagonist_candidates": [
                p.to_dict() if isinstance(p, ProtagonistCandidate) else p
                for p in self.protagonist_candidates
            ],
            "scenes": [
                s.to_dict() if isinstance(s, SceneRecord) else s
                for s in self.scenes
            ],
            "beats": [
                b.to_dict() if isinstance(b, StoryBeat) else b
                for b in self.beats
            ],
            "causal_links": [
                c.to_dict() if isinstance(c, CausalLink) else c
                for c in self.causal_links
            ],
            "character_arcs": [
                a.to_dict() if isinstance(a, CharacterArc) else a
                for a in self.character_arcs
            ],
            "story_summary": self.story_summary,
            "open_uncertainties": self.open_uncertainties,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StoryDocument:
        return cls(
            movie_id=str(data.get("movie_id", "")),
            story_status=str(data.get("story_status", "PARTIAL")),
            protagonist_candidates=[
                ProtagonistCandidate.from_dict(p) if isinstance(p, dict) else p
                for p in data.get("protagonist_candidates", [])
            ],
            scenes=[
                SceneRecord.from_dict(s) if isinstance(s, dict) else s
                for s in data.get("scenes", [])
            ],
            beats=[
                StoryBeat.from_dict(b) if isinstance(b, dict) else b
                for b in data.get("beats", [])
            ],
            causal_links=[
                CausalLink.from_dict(c) if isinstance(c, dict) else c
                for c in data.get("causal_links", [])
            ],
            character_arcs=[
                CharacterArc.from_dict(a) if isinstance(a, dict) else a
                for a in data.get("character_arcs", [])
            ],
            story_summary=str(data.get("story_summary", "")),
            open_uncertainties=[str(u) for u in data.get("open_uncertainties", [])],
        )
