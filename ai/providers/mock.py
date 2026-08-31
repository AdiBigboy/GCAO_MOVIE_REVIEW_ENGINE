"""
Movie Review Engine - Mock Multimodal AI Provider
Provides deterministic, fast, schema-compliant responses for automated testing without live API keys.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ai.base import (
    BaseAIProvider,
    EventRecord,
    EventUncertainty,
    VisualAnalysis,
    InvalidModelResponseError,
)


class MockAIProvider(BaseAIProvider):
    """Deterministic mock provider for automated unit and pipeline testing."""

    def __init__(self, model_name: str = "mock-vision-v1", api_key: Optional[str] = None):
        super().__init__(model_name=model_name, api_key=api_key)
        self.call_count = 0
        self.should_fail_invalid_json = False
        self.should_fail_error = False

    @property
    def provider_name(self) -> str:
        return "mock"

    def analyze_frame_event(
        self,
        image_path: Path,
        timestamp_seconds: float,
        timestamp_str: str,
        dialogue_context: List[Dict[str, Any]],
        previous_context: List[str],
    ) -> EventRecord:
        self.call_count += 1

        if self.should_fail_error:
            raise RuntimeError("Simulated mock model invocation failure.")

        if self.should_fail_invalid_json:
            raise InvalidModelResponseError("Simulated invalid model response payload.")

        # Derive intelligent deterministic properties based on context
        dialogue_texts = [d.get("text", "") for d in dialogue_context]
        has_dialogue = len(dialogue_texts) > 0

        people_count = 2 if has_dialogue else 1
        char_labels = ["PERSON_A", "PERSON_B"] if people_count == 2 else ["PERSON_A"]
        
        if has_dialogue:
            action = f"Characters speaking: \"{' '.join(dialogue_texts[:2])}\""
            summary = f"Characters interact while discussing: \"{dialogue_texts[0]}\""
            significance = 6
        else:
            action = "A character is observing the surroundings."
            summary = f"Scene progression at {timestamp_str} showing character observation."
            significance = 4

        visual = VisualAnalysis(
            people_count=people_count,
            character_labels=char_labels,
            location="indoor setting",
            action=action,
            objects=["furniture", "lighting"],
            emotion="neutral",
            interaction="dialogue exchange" if has_dialogue else None,
            visible_text=None,
        )

        uncertainty = EventUncertainty(
            character_identity="high",
            event_interpretation="medium",
            location_certainty="medium",
        )

        # Index will be reassigned or preserved by caller
        return EventRecord(
            index=1,
            timestamp_seconds=timestamp_seconds,
            timestamp=timestamp_str,
            frame_file="",
            dialogue=dialogue_context,
            visual=visual,
            event_summary=summary,
            plot_significance=significance,
            uncertainty=uncertainty,
        )
