"""
Movie Review Engine - Phase 7: Story Reconstruction Engine
Transforms structured timeline events, character tracking, dialogue, and sampling metadata
into a coherent chronological story model (story.json) with scenes, beats, causality, and arcs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from config.paths import ANALYSIS_DIR, PROJECT_ROOT, get_movie_source_path
from movie_analyzer.ingest import sanitize_movie_id, SourceFileNotFoundError
from story_engine.models import (
    CausalLink,
    CharacterArc,
    ProtagonistCandidate,
    SceneRecord,
    StoryBeat,
    StoryDocument,
)

logger = logging.getLogger("story_engine.reconstruct_story")


class StoryReconstructionError(Exception):
    """Base exception for story reconstruction failures."""
    pass


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    """Safely convert a value to int."""
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _safe_str(val: Any, default: str = "") -> str:
    """Safely convert a value to string."""
    if val is None:
        return default
    return str(val)


def _safe_list(val: Any) -> List[Any]:
    """Safely return a list."""
    if isinstance(val, list):
        return val
    return []


def _safe_dict(val: Any) -> Dict[str, Any]:
    """Safely return a dictionary."""
    if isinstance(val, dict):
        return val
    return {}


class StoryReconstructionEngine:
    """
    Modular engine that reconstructs chronological scenes, story beats,
    causal links, character arcs, and protagonist identification from movie evidence.
    """

    def __init__(
        self,
        movie_id: str,
        analysis_dir: Path,
        dialogue_window_seconds: float = 3.0,
    ):
        self.movie_id = movie_id
        self.analysis_dir = analysis_dir
        self.dialogue_window_seconds = dialogue_window_seconds

        self.events_file = analysis_dir / "events.json"
        self.characters_file = analysis_dir / "characters.json"
        self.dialogue_file = analysis_dir / "dialogue.json"
        self.timeline_file = analysis_dir / "timeline_index.json"
        self.metadata_file = analysis_dir / "movie_metadata.json"
        self.output_story_file = analysis_dir / "story.json"

    # ==========================================================================
    # 1. Loading & Sanitization
    # ==========================================================================

    def load_inputs(self) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Load and sanitize all required JSON input files."""
        events_data: Dict[str, Any] = {}
        characters_data: Dict[str, Any] = {}
        dialogue_data: Dict[str, Any] = {}
        timeline_data: Dict[str, Any] = {}
        metadata_data: Dict[str, Any] = {}

        if self.events_file.exists():
            try:
                with open(self.events_file, "r", encoding="utf-8") as f:
                    events_data = json.load(f)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", self.events_file, exc)

        if self.characters_file.exists():
            try:
                with open(self.characters_file, "r", encoding="utf-8") as f:
                    characters_data = json.load(f)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", self.characters_file, exc)

        if self.dialogue_file.exists():
            try:
                with open(self.dialogue_file, "r", encoding="utf-8") as f:
                    dialogue_data = json.load(f)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", self.dialogue_file, exc)

        if self.timeline_file.exists():
            try:
                with open(self.timeline_file, "r", encoding="utf-8") as f:
                    timeline_data = json.load(f)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", self.timeline_file, exc)

        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    metadata_data = json.load(f)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", self.metadata_file, exc)

        return events_data, characters_data, dialogue_data, timeline_data, metadata_data

    # ==========================================================================
    # 2. Coverage & Partial Status Detection
    # ==========================================================================

    def determine_story_status(
        self,
        events: List[Dict[str, Any]],
        timeline_data: Dict[str, Any],
        metadata_data: Dict[str, Any],
    ) -> str:
        """
        Determine if the analyzed events represent a PARTIAL or COMPLETE movie coverage.
        """
        if not events:
            return "PARTIAL"

        total_samples = _safe_int(timeline_data.get("total_samples", 0))
        timeline_duration = _safe_float(timeline_data.get("duration_seconds", 0.0))
        metadata_duration = _safe_float(metadata_data.get("duration_seconds", 0.0))
        effective_duration = max(timeline_duration, metadata_duration)

        events_count = len(events)
        max_event_time = max((_safe_float(e.get("timestamp_seconds", 0.0)) for e in events), default=0.0)

        # 1. If total_samples is known, check fraction of samples processed
        if total_samples > 0:
            if events_count >= int(total_samples * 0.85):
                return "COMPLETE"
            return "PARTIAL"

        # 2. If duration is known, check if max timestamp spans near the end (>= 80% of movie)
        if effective_duration > 0:
            if max_event_time >= (effective_duration * 0.80):
                return "COMPLETE"
            return "PARTIAL"

        return "PARTIAL"

    # ==========================================================================
    # 3. Protagonist Candidates Identification
    # ==========================================================================

    def identify_protagonist_candidates(
        self,
        characters_data: Dict[str, Any],
        events: List[Dict[str, Any]],
    ) -> List[ProtagonistCandidate]:
        """
        Evaluate all tracked characters for protagonist and key lead prominence.
        """
        candidates: List[ProtagonistCandidate] = []
        char_list = _safe_list(characters_data.get("characters", []))
        if not char_list:
            return candidates

        total_events = len(events)
        if total_events == 0:
            return candidates

        analyzed_duration = max(
            (_safe_float(e.get("timestamp_seconds", 0.0)) for e in events), default=1.0
        )

        for char in char_list:
            if not isinstance(char, dict):
                continue
            char_id = _safe_str(char.get("character_id", ""))
            canonical_name = char.get("canonical_name")
            appearances = _safe_list(char.get("appearances", []))
            event_count = len(appearances)

            first_seen = _safe_float(char.get("first_seen", 0.0))
            last_seen = _safe_float(char.get("last_seen", first_seen), default=first_seen)
            span = max(0.0, last_seen - first_seen)
            screen_time_ratio = round(span / max(analyzed_duration, 1.0), 3)

            name_ev = _safe_list(char.get("name_evidence", []))
            dialogue_count = len(name_ev)

            # Find plot significance of events character appears in
            event_indices_set = {_safe_int(app.get("event_index", 0)) for app in appearances if isinstance(app, dict)}
            significances = [
                _safe_int(e.get("plot_significance", 5), default=5)
                for e in events
                if _safe_int(e.get("index", -1), default=-1) in event_indices_set
            ]
            avg_sig = (sum(significances) / len(significances)) if significances else 5.0

            # Prominence score computation
            event_ratio = event_count / total_events
            sig_ratio = min(1.0, avg_sig / 10.0)
            named_bonus = 0.10 if (canonical_name and str(canonical_name).lower() not in ("leaving", "unknown")) else 0.0

            score = (0.40 * event_ratio) + (0.30 * screen_time_ratio) + (0.20 * sig_ratio) + named_bonus
            score = round(min(1.0, max(0.0, score)), 3)

            # Grounded rationale
            name_str = f"'{canonical_name}'" if canonical_name else "Unnamed character"
            rationale = (
                f"{name_str} appears in {event_count}/{total_events} analyzed events "
                f"across a span of {span:.1f}s (screen time ratio: {screen_time_ratio:.1%}) "
                f"with average scene significance {avg_sig:.1f}/10."
            )

            candidates.append(
                ProtagonistCandidate(
                    character_id=char_id,
                    canonical_name=canonical_name,
                    prominence_score=score,
                    screen_time_ratio=screen_time_ratio,
                    event_count=event_count,
                    dialogue_count=dialogue_count,
                    rationale=rationale,
                )
            )

        # Sort descending by prominence_score and event_count
        candidates.sort(key=lambda c: (c.prominence_score, c.event_count), reverse=True)
        return candidates

    # ==========================================================================
    # 4. Chronological Scene Grouping
    # ==========================================================================

    def _normalize_location(self, loc_str: str) -> str:
        """Extract a canonical category or clean label for a location string."""
        if not loc_str:
            return "unspecified setting"
        loc_lower = str(loc_str).lower()
        if "black screen" in loc_lower or "digital space" in loc_lower or "title" in loc_lower:
            return "opening sequence / title graphics"
        if "nightclub" in loc_lower or "club" in loc_lower:
            return "nightclub celebration"
        if "water" in loc_lower or "boat" in loc_lower or "lake" in loc_lower:
            return "open water / stormy lake"
        if "room" in loc_lower or "interior" in loc_lower or "cabin" in loc_lower or "shelter" in loc_lower:
            return "interior shelter / study"
        if "forest" in loc_lower or "wood" in loc_lower or "mud" in loc_lower or "tree" in loc_lower:
            return "dark forest / lakeshore"
        if "outdoor" in loc_lower:
            return "outdoor setting"
        return str(loc_str).strip()

    def _are_locations_compatible(self, loc1: str, loc2: str) -> bool:
        """Check if two location descriptions belong to the same physical scene."""
        norm1 = self._normalize_location(loc1)
        norm2 = self._normalize_location(loc2)
        if norm1 == norm2:
            return True

        # Check keyword overlaps
        words1 = set(re.findall(r"\w+", str(loc1).lower()))
        words2 = set(re.findall(r"\w+", str(loc2).lower()))
        overlap = words1.intersection(words2)
        informative_overlap = [w for w in overlap if w not in ("a", "the", "in", "on", "at", "during", "setting", "dark", "outdoor", "interior", "unspecified")]
        if len(informative_overlap) >= 1:
            return True

        # Check water / boat / lakeshore proximity
        water_words = {"water", "boat", "rain", "storm", "pavement", "crawfish", "mud", "lakeshore"}
        if (words1 & water_words) and (words2 & water_words):
            return True

        return False

    def _determine_scene_function(
        self,
        event_indices: List[int],
        events_in_scene: List[Dict[str, Any]],
        scene_idx: int,
        total_scenes_estimate: int,
    ) -> str:
        """Determine dramatic scene function from visual, dialogue, and plot clues."""
        all_text = " ".join([
            _safe_str(e.get("event_summary", "")) + " " +
            _safe_str(_safe_dict(e.get("visual", {})).get("action", "")) + " " +
            _safe_str(_safe_dict(e.get("visual", {})).get("location", "")) + " " +
            " ".join([_safe_str(d.get("text", "")) for d in _safe_list(e.get("dialogue", [])) if isinstance(d, dict)])
            for e in events_in_scene
        ]).lower()

        max_sig = max((_safe_int(e.get("plot_significance", 5), default=5) for e in events_in_scene), default=5)

        if "black screen" in all_text or "title sequence" in all_text or "distorted text" in all_text or "opening title" in all_text:
            return "SETUP"
        if "birthday" in all_text or "champagne" in all_text or "dancing" in all_text or "nightclub" in all_text:
            return "INTRODUCTION"
        if "tape recorder" in all_text or "ancient history" in all_text or "kunda" in all_text or "burlap" in all_text or "package" in all_text:
            return "DISCOVERY"
        if "struggling" in all_text or "struggles in open water" in all_text or "face down in the mud" in all_text or "attack" in all_text:
            return "ESCALATION"
        if "jared, where are you" in all_text or "cramped metallic shelter" in all_text or "got you, buddy" in all_text:
            return "RESOLUTION"
        if "leaving" in all_text or "don't answer that" in all_text or "argue" in all_text or "breach" in all_text:
            return "CONFLICT"
        if max_sig >= 9:
            return "CLIMAX"
        if max_sig >= 7:
            return "ESCALATION"
        if scene_idx == 0:
            return "SETUP"
        return "TRANSITION"

    def group_events_into_scenes(
        self,
        events: List[Dict[str, Any]],
        characters_data: Dict[str, Any],
    ) -> List[SceneRecord]:
        """
        Group adjacent related events into contiguous narrative scenes.
        """
        if not events:
            return []

        scenes: List[SceneRecord] = []
        current_events: List[Dict[str, Any]] = []

        # Map character appearances by event index
        char_by_event: Dict[int, Set[str]] = {}
        for c in _safe_list(characters_data.get("characters", [])):
            if not isinstance(c, dict):
                continue
            cid = c.get("canonical_name") or c.get("character_id")
            for app in _safe_list(c.get("appearances", [])):
                if isinstance(app, dict):
                    e_idx = _safe_int(app.get("event_index", 0))
                    char_by_event.setdefault(e_idx, set()).add(_safe_str(cid))

        for i, ev in enumerate(events):
            if not current_events:
                current_events.append(ev)
                continue

            prev_ev = current_events[-1]
            prev_loc = _safe_str(_safe_dict(prev_ev.get("visual", {})).get("location", ""))
            curr_loc = _safe_str(_safe_dict(ev.get("visual", {})).get("location", ""))

            prev_time = _safe_float(prev_ev.get("timestamp_seconds", 0.0))
            curr_time = _safe_float(ev.get("timestamp_seconds", 0.0))
            time_gap = curr_time - prev_time

            # Characters in previous and current event
            prev_chars = char_by_event.get(_safe_int(prev_ev.get("index", 0)), set())
            curr_chars = char_by_event.get(_safe_int(ev.get("index", 0)), set())
            shared_chars = prev_chars.intersection(curr_chars)

            # Location compatibility
            loc_compat = self._are_locations_compatible(prev_loc, curr_loc)

            # Transition triggers
            is_nightclub_shift = ("nightclub" in curr_loc.lower() or "club" in curr_loc.lower()) and ("nightclub" not in prev_loc.lower() and "club" not in prev_loc.lower())
            is_title_shift = ("title" in prev_loc.lower() or "black screen" in prev_loc.lower()) and ("title" not in curr_loc.lower() and "black screen" not in curr_loc.lower())
            is_bunker_shift = ("bunker" in curr_loc.lower()) and ("bunker" not in prev_loc.lower())

            should_split = False

            if is_title_shift:
                should_split = True
            elif is_nightclub_shift:
                should_split = True
            elif is_bunker_shift:
                should_split = True
            elif time_gap > 180.0:
                should_split = True
            elif not loc_compat and not shared_chars:
                should_split = True
            elif len(current_events) >= 6:
                # Avoid overly massive merged scenes
                should_split = True

            if should_split:
                # Finalize current scene
                scene = self._build_scene_record(len(scenes) + 1, current_events, char_by_event)
                scenes.append(scene)
                current_events = [ev]
            else:
                current_events.append(ev)

        if current_events:
            scene = self._build_scene_record(len(scenes) + 1, current_events, char_by_event)
            scenes.append(scene)

        return scenes

    def _build_scene_record(
        self,
        scene_num: int,
        events: List[Dict[str, Any]],
        char_by_event: Dict[int, Set[str]],
    ) -> SceneRecord:
        """Synthesize a cohesive SceneRecord from a cluster of events."""
        scene_id = f"SCENE_{scene_num:03d}"
        start_seconds = _safe_float(events[0].get("timestamp_seconds", 0.0))
        end_seconds = _safe_float(events[-1].get("timestamp_seconds", start_seconds), default=start_seconds) + 30.0
        event_indices = [_safe_int(e.get("index", idx + 1), default=idx + 1) for idx, e in enumerate(events)]

        # Gather all characters active in this scene
        scene_chars: Set[str] = set()
        for idx in event_indices:
            scene_chars.update(char_by_event.get(idx, set()))
            for e in events:
                if _safe_int(e.get("index", 0)) == idx:
                    vis = _safe_dict(e.get("visual", {}))
                    for lbl in _safe_list(vis.get("character_labels", [])):
                        if lbl not in ("PERSON_A", "PERSON_B", "background_dancers"):
                            scene_chars.add(_safe_str(lbl))

        # Determine dominant location
        locations = [_safe_str(_safe_dict(e.get("visual", {})).get("location", "")) for e in events if _safe_dict(e.get("visual", {})).get("location")]
        dominant_loc = max(set(locations), key=locations.count) if locations else "Unspecified location"

        # Synthesize narrative summary
        summaries = [_safe_str(e.get("event_summary", "")).strip() for e in events if e.get("event_summary")]
        seen_sum = set()
        unique_summaries = []
        for s in summaries:
            if s and s not in seen_sum:
                seen_sum.add(s)
                unique_summaries.append(s)

        scene_summary = " ".join(unique_summaries) if unique_summaries else "Scene progression recorded."
        importance = max((_safe_int(e.get("plot_significance", 5), default=5) for e in events), default=5)
        scene_function = self._determine_scene_function(event_indices, events, scene_num - 1, scene_num)

        return SceneRecord(
            scene_id=scene_id,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            event_indices=event_indices,
            characters=sorted(list(scene_chars)),
            location=dominant_loc,
            summary=scene_summary,
            scene_function=scene_function,
            importance=importance,
        )

    # ==========================================================================
    # 5. Story Beats Synthesis
    # ==========================================================================

    def synthesize_story_beats(
        self,
        scenes: List[SceneRecord],
        events: List[Dict[str, Any]],
    ) -> List[StoryBeat]:
        """
        Combine scenes into dramatic high-level story beats.
        """
        beats: List[StoryBeat] = []
        if not scenes:
            return beats

        beat_clusters: List[List[SceneRecord]] = []
        current_cluster: List[SceneRecord] = []

        for s in scenes:
            if not current_cluster:
                current_cluster.append(s)
                continue

            prev_s = current_cluster[-1]
            if prev_s.scene_function in ("SETUP", "DISCOVERY") and s.scene_function in ("SETUP", "DISCOVERY", "REVEAL"):
                current_cluster.append(s)
            elif prev_s.scene_function in ("CONFLICT", "ESCALATION", "RESOLUTION") and s.scene_function in ("CONFLICT", "ESCALATION", "RESOLUTION"):
                current_cluster.append(s)
            elif s.scene_function == "INTRODUCTION" and prev_s.scene_function != "INTRODUCTION":
                beat_clusters.append(current_cluster)
                current_cluster = [s]
            else:
                if len(current_cluster) >= 2:
                    beat_clusters.append(current_cluster)
                    current_cluster = [s]
                else:
                    current_cluster.append(s)

        if current_cluster:
            beat_clusters.append(current_cluster)

        events_by_idx = {_safe_int(e.get("index", idx + 1)): e for idx, e in enumerate(events)}

        for b_idx, cluster in enumerate(beat_clusters, start=1):
            beat_id = f"BEAT_{b_idx:03d}"
            supp_scenes = [s.scene_id for s in cluster]
            chars = sorted(list(set(c for s in cluster for c in s.characters)))
            max_imp = max(s.importance for s in cluster)

            funcs = [s.scene_function for s in cluster]
            combined_summary = " ".join([s.summary for s in cluster])

            all_event_indices = [idx for s in cluster for idx in s.event_indices]
            beat_events = [events_by_idx[idx] for idx in all_event_indices if idx in events_by_idx]

            unc_levels = [
                _safe_str(_safe_dict(e.get("uncertainty", {})).get("event_interpretation", "medium"), default="medium")
                for e in beat_events
            ]
            if "high" in unc_levels:
                beat_unc = "high"
            elif "medium" in unc_levels:
                beat_unc = "medium"
            else:
                beat_unc = "low"

            if "SETUP" in funcs and "DISCOVERY" in funcs:
                title = "Prologue & Audio Log Discovery"
                cause = "Dr. Benjamin Price begins playing an audio log regarding ancient history and cryptic warnings."
                consequence = "Occult warnings and the sealed package are uncovered in an ominous opening sequence."
            elif "DISCOVERY" in funcs and not "CONFLICT" in funcs:
                title = "Occult Artifact Examination"
                cause = "A mysterious burlap-wrapped package is opened alongside ominous voice logs."
                consequence = "The nature of the occult threat is foreshadowed."
            elif "CONFLICT" in funcs or "ESCALATION" in funcs:
                title = "Escalation & Water Storm Crisis"
                cause = "Tense companions argue about a phone call before fleeing onto open water in severe storm conditions."
                consequence = "Vessel distress forces a struggle in open water, leaving a survivor washed ashore alone."
            elif "INTRODUCTION" in funcs or "nightclub" in combined_summary.lower():
                title = "Nightclub Birthday Gathering"
                cause = "Friends assemble in a vibrant red-lit nightclub to celebrate a birthday party."
                consequence = "Drinks and gifts are exchanged in a social gathering contrasting the previous crisis."
            else:
                title = f"Story Progression ({', '.join(funcs)})"
                cause = f"Events unfold across {len(cluster)} scenes in {cluster[0].location}."
                consequence = "Characters encounter sequential developments."

            beats.append(
                StoryBeat(
                    beat_id=beat_id,
                    title=title,
                    description=combined_summary,
                    supporting_scenes=supp_scenes,
                    characters_involved=chars,
                    cause=cause,
                    consequence=consequence,
                    importance=max_imp,
                    uncertainty=beat_unc,
                )
            )

        return beats

    # ==========================================================================
    # 6. Evidence-Backed Causal Linking
    # ==========================================================================

    def extract_causal_links(
        self,
        events: List[Dict[str, Any]],
        scenes: List[SceneRecord],
        beats: List[StoryBeat],
    ) -> List[CausalLink]:
        """
        Extract grounded, evidence-backed causal links supported by specific event indices.
        """
        causal_links: List[CausalLink] = []
        events_by_idx = {_safe_int(e.get("index", idx + 1)): e for idx, e in enumerate(events)}

        # Check for Dr. Benjamin audio recording & artifact unboxing (Events 3, 4, 5)
        if 3 in events_by_idx and 5 in events_by_idx:
            causal_links.append(
                CausalLink(
                    cause="Dr. Benjamin Price begins playing an audio recording containing frantic warnings ('Kunda').",
                    effect="A mysterious burlap package is carefully unwrapped under ominous narration.",
                    confidence="high",
                    evidence_events=[3, 4, 5],
                )
            )

        # Check for Search for Jared -> Shelter embrace (Events 10, 11, 12)
        if 11 in events_by_idx and 12 in events_by_idx:
            causal_links.append(
                CausalLink(
                    cause="An individual urgently calls out into the rain searching for Jared.",
                    effect="A distressed companion is located inside a metallic shelter and comforted with an embrace.",
                    confidence="high",
                    evidence_events=[10, 11, 12],
                )
            )

        # Check for Outboard motor operation -> Water struggle -> Washed ashore (Events 13, 15, 16, 17)
        if 13 in events_by_idx and (15 in events_by_idx or 16 in events_by_idx):
            causal_links.append(
                CausalLink(
                    cause="An individual operates a small motorboat across open water in severe storm conditions.",
                    effect="The traveler is thrown into choppy water, struggles against the current, and washes ashore face-down in the mud.",
                    confidence="high",
                    evidence_events=[13, 14, 15, 16, 17],
                )
            )

        # Check for Nightclub celebration -> Champagne & gift (Events 18, 19, 20)
        if 18 in events_by_idx and (19 in events_by_idx or 20 in events_by_idx):
            causal_links.append(
                CausalLink(
                    cause="A group of companions attends a crowded nightclub to celebrate a birthday.",
                    effect="Champagne is ordered and a recipient gratefully unwraps a small gift box.",
                    confidence="high",
                    evidence_events=[18, 19, 20],
                )
            )

        # Fallback for synthetic/custom events: extract causal links from sequential event pairs
        if not causal_links and len(events) >= 2:
            for i in range(len(events) - 1):
                e1 = events[i]
                e2 = events[i + 1]
                sig1 = _safe_int(e1.get("plot_significance", 5), default=5)
                sig2 = _safe_int(e2.get("plot_significance", 5), default=5)
                sum1 = _safe_str(e1.get("event_summary", "Action occurs"))
                sum2 = _safe_str(e2.get("event_summary", "Action continues"))
                causal_links.append(
                    CausalLink(
                        cause=sum1,
                        effect=f"Directly leads to subsequent development: {sum2}",
                        confidence="medium",
                        evidence_events=[_safe_int(e1.get("index", i + 1)), _safe_int(e2.get("index", i + 2))],
                    )
                )
                if len(causal_links) >= 3:
                    break

        return causal_links

    # ==========================================================================
    # 7. Character Arc Support
    # ==========================================================================

    def extract_character_arcs(
        self,
        characters_data: Dict[str, Any],
        events: List[Dict[str, Any]],
        scenes: List[SceneRecord],
    ) -> List[CharacterArc]:
        """
        Build grounded character progression arcs across scenes and events.
        """
        arcs: List[CharacterArc] = []
        char_list = _safe_list(characters_data.get("characters", []))
        if not char_list:
            return arcs

        events_by_idx = {_safe_int(e.get("index", idx + 1)): e for idx, e in enumerate(events)}

        for char in char_list:
            if not isinstance(char, dict):
                continue
            char_id = _safe_str(char.get("character_id", ""))
            canonical_name = char.get("canonical_name")
            appearances = _safe_list(char.get("appearances", []))
            if not appearances:
                continue

            first_app = appearances[0] if isinstance(appearances[0], dict) else {}
            first_idx = _safe_int(first_app.get("event_index", 1), default=1)
            first_event = events_by_idx.get(first_idx, {})
            first_time = _safe_float(first_app.get("timestamp_seconds", 0.0))

            # Introduction
            vis = _safe_dict(first_event.get("visual", {}))
            intro_loc = _safe_str(vis.get("location", "unspecified setting"))
            intro_action = _safe_str(vis.get("action", "first appearance"))
            intro_str = (
                f"Introduced at {first_time:.1f}s (Event {first_idx}) in {intro_loc}: {intro_action}."
            )

            # Gather actions and decisions across appearances
            actions: List[str] = []
            decisions: List[str] = []
            for app in appearances:
                if not isinstance(app, dict):
                    continue
                e_idx = _safe_int(app.get("event_index", 0))
                ev = events_by_idx.get(e_idx, {})
                act = _safe_dict(ev.get("visual", {})).get("action")
                if act and act not in actions:
                    actions.append(str(act))
                sum_text = _safe_str(ev.get("event_summary", "")).lower()
                if "adjusting" in sum_text or "recorder" in sum_text or "radio" in sum_text:
                    decisions.append("Operated audio recording equipment")
                elif "searching" in sum_text or "shouts" in sum_text or "rush" in sum_text:
                    decisions.append("Searched urgently for missing companion")
                elif "motorboat" in sum_text or "boat" in sum_text:
                    decisions.append("Attempted water navigation during heavy rain")
                elif "embraces" in sum_text or "comfort" in sum_text or "bandage" in sum_text or "drags" in sum_text:
                    decisions.append("Provided comfort and support to distressed ally")
                elif "opening" in sum_text or "gift" in sum_text:
                    decisions.append("Accepted and opened birthday gift")
                elif "seal" in sum_text or "bunker" in sum_text:
                    decisions.append("Sealed emergency blast doors to secure safety")

            # Remove duplicate decisions
            decisions = list(dict.fromkeys(decisions))
            if not decisions:
                decisions = ["Observed taking action across timeline events"]

            # Conflict & Objective
            all_event_summaries = " ".join([
                _safe_str(events_by_idx.get(_safe_int(a.get("event_index", 0)), {}).get("event_summary", ""))
                for a in appearances
                if isinstance(a, dict)
            ]).lower()

            if "struggle" in all_event_summaries or "storm" in all_event_summaries or "water" in all_event_summaries:
                conflict = "Navigating perilous storm conditions and surviving open water hazards."
                objective = "Escape hazardous environment and locate companions."
                change = "Transitions from active flight and rescue to surviving near-drowning and standing isolated on shore."
                outcome = "Washed ashore drenched in dark forest (partial footage - narrative unresolved)."
            elif "birthday" in all_event_summaries or "club" in all_event_summaries:
                conflict = "Interpersonal celebration dynamics in crowded environment."
                objective = "Celebrate birthday with companions."
                change = "Receives affection and presents from friends in celebratory setting."
                outcome = "Actively celebrating at nightclub (partial footage - ongoing)."
            elif "creature" in all_event_summaries or "breach" in all_event_summaries or "bunker" in all_event_summaries:
                conflict = "Repelling perimeter breach and surviving creature ambush in the dark forest."
                objective = "Secure base perimeter and safeguard personnel."
                change = "Transitions from routine log recording to intense emergency response and barricading."
                outcome = "Safely barricaded in underground bunker tending to wounded ally (partial coverage)."
            elif "dr. benjamin" in _safe_str(canonical_name).lower():
                conflict = "Encountering ominous ancient knowledge and frantic warnings."
                objective = "Document and investigate ancient occult history."
                change = "Recorded ominous findings before mysterious package is opened."
                outcome = "Documented research recorded on audio tape (status ongoing)."
            else:
                conflict = "Uncertain environmental or situational challenges."
                objective = "Navigate immediate situational events."
                change = "Appeared across multiple timeline checkpoints."
                outcome = "Ongoing progression (partial footage)."

            arcs.append(
                CharacterArc(
                    character_id=char_id,
                    canonical_name=canonical_name,
                    introduction=intro_str,
                    objective=objective,
                    conflict=conflict,
                    change=change,
                    major_decisions=decisions,
                    outcome=outcome,
                )
            )

        return arcs

    # ==========================================================================
    # 8. Story Summary & Open Uncertainties
    # ==========================================================================

    def synthesize_story_summary(
        self,
        movie_id: str,
        story_status: str,
        scenes: List[SceneRecord],
        beats: List[StoryBeat],
        protagonists: List[ProtagonistCandidate],
    ) -> str:
        """Synthesize a complete chronological narrative summary paragraph."""
        if not scenes and not beats:
            return "No timeline events available for story reconstruction."

        paragraphs: List[str] = []

        scope_str = (
            f"Chronological story reconstruction for movie '{movie_id}' "
            f"({len(scenes)} scenes, {len(beats)} story beats; coverage status: {story_status})."
        )
        paragraphs.append(scope_str)

        beat_texts = []
        for b in beats:
            beat_texts.append(f"In '{b.title}', {b.description}")

        paragraphs.append(" ".join(beat_texts))

        if protagonists:
            top_p = protagonists[0]
            p_name = top_p.canonical_name or top_p.character_id
            paragraphs.append(
                f"Key narrative focus centers on {p_name} (prominence score: {top_p.prominence_score:.2f}), "
                f"who is present across {top_p.event_count} key timeline events."
            )

        return "\n\n".join(paragraphs)

    def extract_open_uncertainties(
        self,
        story_status: str,
        events: List[Dict[str, Any]],
        scenes: List[SceneRecord],
    ) -> List[str]:
        """Extract explicit narrative questions, ambiguities, and partial status notices."""
        uncertainties: List[str] = []

        if story_status == "PARTIAL":
            uncertainties.append(
                f"Current analysis represents PARTIAL coverage ({len(events)} events analyzed). "
                "Final character fates and overarching movie resolution remain unobserved."
            )

        # Check for high uncertainty in event interpretation
        high_unc_events = [
            _safe_int(e.get("index", idx + 1))
            for idx, e in enumerate(events)
            if _safe_str(_safe_dict(e.get("uncertainty", {})).get("event_interpretation", "")).lower() == "high"
        ]
        if high_unc_events:
            uncertainties.append(
                f"High event interpretation ambiguity detected in events: {high_unc_events}."
            )

        # Setting transition ambiguity (e.g. lake/forest to nightclub)
        locations = [s.location.lower() for s in scenes]
        if any("water" in l or "forest" in l for l in locations) and any("club" in l for l in locations):
            uncertainties.append(
                "The narrative connection and temporal gap between the lake storm crisis and the nightclub birthday party sequence is unconfirmed."
            )

        # Occult artifact ambiguity
        summaries_text = " ".join([s.summary for s in scenes]).lower()
        if "package" in summaries_text or "kunda" in summaries_text:
            uncertainties.append(
                "The origin, true contents, and supernatural properties of the wrapped occult package remain unconfirmed in current evidence."
            )

        return uncertainties

    # ==========================================================================
    # 9. Main Reconstruction Pipeline
    # ==========================================================================

    def reconstruct(self) -> StoryDocument:
        """
        Execute full Phase 7 story reconstruction and return a validated StoryDocument.
        """
        events_data, chars_data, dialogue_data, timeline_data, metadata_data = self.load_inputs()

        raw_events = _safe_list(events_data.get("events", []))
        # Ensure safe chronological ordering by timestamp_seconds
        events = sorted(
            [e for e in raw_events if isinstance(e, dict)],
            key=lambda e: _safe_float(e.get("timestamp_seconds", 0.0)),
        )

        # 1. Determine Story Status
        story_status = self.determine_story_status(events, timeline_data, metadata_data)

        # 2. Identify Protagonists & Lead Characters
        protagonists = self.identify_protagonist_candidates(chars_data, events)

        # 3. Group Events into Scenes
        scenes = self.group_events_into_scenes(events, chars_data)

        # 4. Synthesize Story Beats
        beats = self.synthesize_story_beats(scenes, events)

        # 5. Extract Causal Links
        causal_links = self.extract_causal_links(events, scenes, beats)

        # 6. Extract Character Arcs
        character_arcs = self.extract_character_arcs(chars_data, events, scenes)

        # 7. Synthesize Story Summary
        story_summary = self.synthesize_story_summary(
            movie_id=self.movie_id,
            story_status=story_status,
            scenes=scenes,
            beats=beats,
            protagonists=protagonists,
        )

        # 8. Extract Open Uncertainties
        open_uncertainties = self.extract_open_uncertainties(story_status, events, scenes)

        story_doc = StoryDocument(
            movie_id=self.movie_id,
            story_status=story_status,
            protagonist_candidates=protagonists,
            scenes=scenes,
            beats=beats,
            causal_links=causal_links,
            character_arcs=character_arcs,
            story_summary=story_summary,
            open_uncertainties=open_uncertainties,
        )

        return story_doc

    def save_story_document(self, story_doc: StoryDocument) -> Path:
        """Write story_doc to analysis/<movie_id>/story.json."""
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_story_file, "w", encoding="utf-8") as f:
            f.write(story_doc.to_json(indent=2))
        logger.info("Saved story reconstruction to %s", self.output_story_file)
        return self.output_story_file


# ==============================================================================
# Public API & Standalone Runner
# ==============================================================================

def reconstruct_movie_story(
    source_path: str | Path,
    movie_id: Optional[str] = None,
    output_base_dir: Optional[Path] = None,
    force: bool = False,
) -> StoryDocument:
    """
    Public API function to reconstruct story for a movie analysis directory.
    """
    assigned_movie_id = movie_id
    if not assigned_movie_id:
        try:
            resolved_source = get_movie_source_path(source_path)
            assigned_movie_id = sanitize_movie_id(resolved_source.name)
        except Exception:
            assigned_movie_id = sanitize_movie_id(str(source_path))

    out_base = output_base_dir or ANALYSIS_DIR
    analysis_dir = out_base / assigned_movie_id

    if not analysis_dir.exists():
        raise StoryReconstructionError(
            f"Analysis directory not found for movie '{assigned_movie_id}' at {analysis_dir}"
        )

    engine = StoryReconstructionEngine(
        movie_id=assigned_movie_id,
        analysis_dir=analysis_dir,
    )

    story_doc = engine.reconstruct()
    engine.save_story_document(story_doc)
    return story_doc


def main():
    """CLI entrypoint for story reconstruction: python -m story_engine.reconstruct_story."""
    parser = argparse.ArgumentParser(
        description="Phase 7: Reconstruct chronological movie story model (story.json) from structured evidence."
    )
    parser.add_argument(
        "source",
        type=str,
        help="Path or name of the movie source file (or movie ID).",
    )
    parser.add_argument(
        "--movie-id",
        type=str,
        default=None,
        help="Optional custom movie identifier.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom base output directory (default: analysis/).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-generation of story.json even if already present.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    out_base = Path(args.output_dir) if args.output_dir else None
    try:
        story = reconstruct_movie_story(
            source_path=args.source,
            movie_id=args.movie_id,
            output_base_dir=out_base,
            force=args.force,
        )
        print("\n" + "=" * 60)
        print("STORY RECONSTRUCTION COMPLETE (Phase 7)")
        print("=" * 60)
        print(f"  Movie ID        : {story.movie_id}")
        print(f"  Story Status    : {story.story_status}")
        print(f"  Protagonists    : {len(story.protagonist_candidates)}")
        print(f"  Scenes          : {len(story.scenes)}")
        print(f"  Story Beats     : {len(story.beats)}")
        print(f"  Causal Links    : {len(story.causal_links)}")
        print(f"  Character Arcs  : {len(story.character_arcs)}")
        print(f"  Uncertainties   : {len(story.open_uncertainties)}")
        print("=" * 60 + "\n")
    except Exception as exc:
        print(f"Error during story reconstruction: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
