"""
Movie Review Engine - Phase 10: Preview Engine Data Models
Defines structured schemas for timeline items, segment timelines, and preview manifests.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TimelineItem:
    """An individual visual block on the preview timeline."""
    item_id: str
    item_type: str  # "SOURCE_CLIP" or "PLACEHOLDER"
    start_time_seconds: float
    end_time_seconds: float
    duration_seconds: float
    source_clip_id: Optional[str] = None
    source_event_index: Optional[int] = None
    source_scene_id: Optional[str] = None
    caption_text: Optional[str] = None
    clip_path: Optional[str] = None
    label: str = ""
    has_source_audio: bool = False
    audio_track_type: str = "SILENCE"  # "SOURCE_AUDIO", "SILENCE", "AMBIENT_TAIL"
    future_narration_point: bool = True
    audio_bus_mapping: Dict[str, Any] = field(default_factory=lambda: {
        "narration_bus_ducking_db": -14.0,
        "source_audio_bus_db": 0.0,
    })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "item_type": self.item_type,
            "start_time_seconds": round(self.start_time_seconds, 2),
            "end_time_seconds": round(self.end_time_seconds, 2),
            "duration_seconds": round(self.duration_seconds, 2),
            "source_clip_id": self.source_clip_id,
            "source_event_index": self.source_event_index,
            "source_scene_id": self.source_scene_id,
            "caption_text": self.caption_text,
            "clip_path": self.clip_path,
            "label": self.label,
            "has_source_audio": self.has_source_audio,
            "audio_track_type": self.audio_track_type,
            "future_narration_point": self.future_narration_point,
            "audio_bus_mapping": self.audio_bus_mapping,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TimelineItem:
        return cls(
            item_id=str(data.get("item_id", "")),
            item_type=str(data.get("item_type", "PLACEHOLDER")),
            start_time_seconds=float(data.get("start_time_seconds", 0.0)),
            end_time_seconds=float(data.get("end_time_seconds", 0.0)),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            source_clip_id=data.get("source_clip_id"),
            source_event_index=data.get("source_event_index"),
            source_scene_id=data.get("source_scene_id"),
            caption_text=data.get("caption_text"),
            clip_path=data.get("clip_path"),
            label=str(data.get("label", "")),
            has_source_audio=bool(data.get("has_source_audio", False)),
            audio_track_type=str(data.get("audio_track_type", "SILENCE")),
            future_narration_point=bool(data.get("future_narration_point", True)),
            audio_bus_mapping=dict(data.get("audio_bus_mapping", {
                "narration_bus_ducking_db": -14.0,
                "source_audio_bus_db": 0.0,
            })),
        )


@dataclass
class SegmentTimeline:
    """Timeline block corresponding to one script narration segment."""
    segment_id: str
    start_time_seconds: float
    end_time_seconds: float
    duration_seconds: float
    narration_text: str
    items: List[TimelineItem] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "start_time_seconds": round(self.start_time_seconds, 2),
            "end_time_seconds": round(self.end_time_seconds, 2),
            "duration_seconds": round(self.duration_seconds, 2),
            "narration_text": self.narration_text,
            "items": [it.to_dict() if isinstance(it, TimelineItem) else it for it in self.items],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SegmentTimeline:
        return cls(
            segment_id=str(data.get("segment_id", "")),
            start_time_seconds=float(data.get("start_time_seconds", 0.0)),
            end_time_seconds=float(data.get("end_time_seconds", 0.0)),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            narration_text=str(data.get("narration_text", "")),
            items=[
                TimelineItem.from_dict(it) if isinstance(it, dict) else it
                for it in data.get("items", [])
            ],
        )


@dataclass
class PreviewManifest:
    """Complete timing and asset manifest for the rough video preview (preview_manifest.json)."""
    movie_id: str
    status: str  # "PARTIAL" or "COMPLETE"
    total_duration_seconds: float
    resolution: str = "1920x1080"
    fps: float = 30.0
    total_segments: int = 0
    total_source_clips: int = 0
    total_placeholders: int = 0
    total_source_footage_seconds: float = 0.0
    total_placeholder_seconds: float = 0.0
    video_output_path: Optional[str] = None
    audio_architecture: Dict[str, Any] = field(default_factory=lambda: {
        "narration_bus": {"role": "primary_voice", "ducking_depth_db": -14.0},
        "source_audio_bus": {"role": "dramatic_sound_and_ambience", "target_loudness_lufs": -24.0},
        "music_bus": {"role": "optional_score", "ducking_depth_db": -18.0},
    })
    segments: List[SegmentTimeline] = field(default_factory=list)
    created_at: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        all_items = [it for s in self.segments for it in s.items]
        clips = [it for it in all_items if it.item_type == "SOURCE_CLIP"]
        placeholders = [it for it in all_items if it.item_type == "PLACEHOLDER"]

        return {
            "movie_id": self.movie_id,
            "status": self.status,
            "total_duration_seconds": round(self.total_duration_seconds, 2),
            "resolution": self.resolution,
            "fps": self.fps,
            "total_segments": len(self.segments),
            "total_source_clips": len(clips),
            "total_placeholders": len(placeholders),
            "total_source_footage_seconds": round(sum(c.duration_seconds for c in clips), 2),
            "total_placeholder_seconds": round(sum(p.duration_seconds for p in placeholders), 2),
            "video_output_path": self.video_output_path,
            "audio_architecture": self.audio_architecture,
            "segments": [s.to_dict() if isinstance(s, SegmentTimeline) else s for s in self.segments],
            "created_at": self.created_at,
            "warnings": self.warnings,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PreviewManifest:
        return cls(
            movie_id=str(data.get("movie_id", "")),
            status=str(data.get("status", "PARTIAL")),
            total_duration_seconds=float(data.get("total_duration_seconds", 0.0)),
            resolution=str(data.get("resolution", "1920x1080")),
            fps=float(data.get("fps", 30.0)),
            total_segments=int(data.get("total_segments", 0)),
            total_source_clips=int(data.get("total_source_clips", 0)),
            total_placeholders=int(data.get("total_placeholders", 0)),
            total_source_footage_seconds=float(data.get("total_source_footage_seconds", 0.0)),
            total_placeholder_seconds=float(data.get("total_placeholder_seconds", 0.0)),
            video_output_path=data.get("video_output_path"),
            audio_architecture=dict(data.get("audio_architecture", {
                "narration_bus": {"role": "primary_voice", "ducking_depth_db": -14.0},
                "source_audio_bus": {"role": "dramatic_sound_and_ambience", "target_loudness_lufs": -24.0},
                "music_bus": {"role": "optional_score", "ducking_depth_db": -18.0},
            })),
            segments=[
                SegmentTimeline.from_dict(s) if isinstance(s, dict) else s
                for s in data.get("segments", [])
            ],
            created_at=data.get("created_at"),
            warnings=[str(w) for w in data.get("warnings", [])],
        )
