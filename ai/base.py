"""
Movie Review Engine - Multimodal AI Provider Base Architecture
"""

from __future__ import annotations

import abc
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ==============================================================================
# Exceptions
# ==============================================================================

class AIProviderError(Exception):
    """Base exception for all AI provider operations."""
    pass


class MissingAPIKeyError(AIProviderError):
    """Raised when an API key is required but missing."""
    pass


class ModelInvocationError(AIProviderError):
    """Raised when the multimodal model fails to respond or produces an error."""
    pass


class InvalidModelResponseError(AIProviderError):
    """Raised when the model response fails schema validation."""
    pass


# ==============================================================================
# Data Models
# ==============================================================================

@dataclass
class VisualAnalysis:
    """Detailed visual observation extracted from a single frame."""
    people_count: int
    character_labels: List[str]
    location: str
    action: str
    objects: List[str]
    emotion: str
    interaction: Optional[str] = None
    visible_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if not data.get("interaction"):
            data.pop("interaction", None)
        if not data.get("visible_text"):
            data.pop("visible_text", None)
        return data


@dataclass
class EventUncertainty:
    """Explicit confidence ratings for ambiguity handling."""
    character_identity: str = "high"  # "low", "medium", "high"
    event_interpretation: str = "medium"  # "low", "medium", "high"
    location_certainty: Optional[str] = "medium"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EventRecord:
    """A single analyzed timeline event combining visual, audio dialogue, and plot context."""
    index: int
    timestamp_seconds: float
    timestamp: str
    frame_file: str
    dialogue: List[Dict[str, Any]]
    visual: VisualAnalysis
    event_summary: str
    plot_significance: int  # 1 to 10 scale
    uncertainty: EventUncertainty

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "timestamp_seconds": self.timestamp_seconds,
            "timestamp": self.timestamp,
            "frame_file": self.frame_file,
            "dialogue": self.dialogue,
            "visual": self.visual.to_dict() if isinstance(self.visual, VisualAnalysis) else self.visual,
            "event_summary": self.event_summary,
            "plot_significance": self.plot_significance,
            "uncertainty": self.uncertainty.to_dict() if isinstance(self.uncertainty, EventUncertainty) else self.uncertainty,
        }


@dataclass
class EventsDocument:
    """Complete structured timeline event log for a movie."""
    movie_id: str
    provider: str
    model: str
    total_events: int
    events: List[EventRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "movie_id": self.movie_id,
            "provider": self.provider,
            "model": self.model,
            "total_events": self.total_events,
            "events": [e.to_dict() if isinstance(e, EventRecord) else e for e in self.events],
        }


# ==============================================================================
# Abstract Provider Base
# ==============================================================================

class BaseAIProvider(abc.ABC):
    """Abstract base class for multimodal AI event analyzers."""

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key

    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        """Name of the provider (e.g. 'gemini', 'mock')."""
        pass

    @abc.abstractmethod
    def analyze_frame_event(
        self,
        image_path: Path,
        timestamp_seconds: float,
        timestamp_str: str,
        dialogue_context: List[Dict[str, Any]],
        previous_context: List[str],
    ) -> EventRecord:
        """
        Analyze a single frame image combined with nearby dialogue and temporal history.
        Must return a validated EventRecord instance.
        """
        pass
