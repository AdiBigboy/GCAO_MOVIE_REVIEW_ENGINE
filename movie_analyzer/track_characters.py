"""
Movie Review Engine - Phase 5: Character Identity Tracking & Character Memory Module
Processes timeline events and dialogue to build consistent character identities in characters.json.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ai import (
    BaseAIProvider,
    EventsDocument,
    EventRecord,
    get_ai_provider,
)
from movie_analyzer.ingest import sanitize_movie_id, SourceFileNotFoundError
from movie_analyzer.analyze_events import analyze_timeline_events

logger = logging.getLogger("movie_analyzer.track_characters")


# ==============================================================================
# Data Models
# ==============================================================================

@dataclass
class CharacterAppearance:
    """A verified appearance of a character in a specific timeline event."""
    timestamp_seconds: float
    event_index: int
    frame_file: str
    temporary_label: str
    match_confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VisualProfile:
    """Aggregated visual features observed for a character."""
    gender_presentation: str = "unknown"  # "male", "female", "unknown"
    approx_age_group: str = "adult"      # "child", "teen", "adult", "elderly", "unknown"
    hair: Optional[str] = None
    clothing: List[str] = field(default_factory=list)
    distinctive_features: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gender_presentation": self.gender_presentation,
            "approx_age_group": self.approx_age_group,
            "hair": self.hair,
            "clothing": list(self.clothing),
            "distinctive_features": list(self.distinctive_features),
        }


@dataclass
class NameEvidence:
    """Direct subtitle/dialogue evidence referencing a character's real name."""
    timestamp_seconds: float
    text: str
    candidate_name: str
    speaker: Optional[str] = None
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CharacterRelationship:
    """Observed interaction/relationship between two tracked characters."""
    target_character_id: str
    relationship_type: str
    evidence: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CharacterRecord:
    """Persistent profile for a single tracked character."""
    character_id: str
    canonical_name: Optional[str]
    name_confidence: str  # "unknown", "low", "medium", "high"
    first_seen: float
    last_seen: float
    appearances: List[CharacterAppearance] = field(default_factory=list)
    visual_profile: VisualProfile = field(default_factory=VisualProfile)
    name_evidence: List[NameEvidence] = field(default_factory=list)
    relationships: List[CharacterRelationship] = field(default_factory=list)
    identity_confidence: str = "medium"  # "low", "medium", "high"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "character_id": self.character_id,
            "canonical_name": self.canonical_name,
            "name_confidence": self.name_confidence,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "appearances": [a.to_dict() if isinstance(a, CharacterAppearance) else a for a in self.appearances],
            "visual_profile": self.visual_profile.to_dict() if isinstance(self.visual_profile, VisualProfile) else self.visual_profile,
            "name_evidence": [n.to_dict() if isinstance(n, NameEvidence) else n for n in self.name_evidence],
            "relationships": [r.to_dict() if isinstance(r, CharacterRelationship) else r for r in self.relationships],
            "identity_confidence": self.identity_confidence,
        }


@dataclass
class CharactersDocument:
    """Full character memory database for a movie."""
    movie_id: str
    total_characters: int
    characters: List[CharacterRecord] = field(default_factory=list)
    merge_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "movie_id": self.movie_id,
            "total_characters": self.total_characters,
            "characters": [c.to_dict() if isinstance(c, CharacterRecord) else c for c in self.characters],
            "merge_history": self.merge_history,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ==============================================================================
# Dialogue Name Extraction & Entity Resolution
# ==============================================================================

NAME_PATTERNS = [
    # Direct address e.g. "John, wait!" or "Hey Sarah,"
    re.compile(r"\b(?:Hey|Hi|Hello|Look|Listen|Wait|Please|Thanks),?\s+((?:[A-Z][a-z]{2,15}\s*){1,2})\b"),
    re.compile(r"\b([A-Z][a-z]{2,15}),\s+(?:look|wait|listen|come here|be careful|are you|what|where|why|help)\b", re.IGNORECASE),
    # Self-introduction e.g. "My name is John" or "My name is Detective Miller"
    re.compile(r"\b(?:my name is|i am|i'm|call me)\s+((?:[A-Z][a-z]{2,15}\s*){1,2})\b", re.IGNORECASE),
    # Third-person introduction e.g. "This is John" or "Meet Sarah"
    re.compile(r"\b(?:this is|meet|ask|talk to)\s+((?:[A-Z][a-z]{2,15}\s*){1,2})\b", re.IGNORECASE),
]

DISALLOWED_CANDIDATE_NAMES = {
    "The", "This", "That", "There", "Here", "What", "Where", "When", "Why", "How",
    "Please", "Look", "Listen", "Wait", "Hello", "Yes", "Yeah", "No", "Nope", "Okay",
    "Someone", "Anyone", "Nobody", "Everyone", "Movie", "Engine", "Phase", "Dialogue",
}


def extract_name_evidence_from_dialogue(
    dialogue_lines: List[Dict[str, Any]],
    event_timestamp: float,
) -> List[NameEvidence]:
    """
    Search nearby dialogue for explicit names mentioned in direct address or introductions.
    """
    evidence_list: List[NameEvidence] = []

    for line in dialogue_lines:
        text = line.get("text", "")
        speaker = line.get("speaker")
        ts = float(line.get("start_seconds", event_timestamp))

        # Check speaker attribute from subtitle format
        if speaker and speaker.strip() and speaker.strip() not in DISALLOWED_CANDIDATE_NAMES:
            clean_spk = speaker.strip()
            evidence_list.append(
                NameEvidence(
                    timestamp_seconds=ts,
                    text=f"[Speaker Tag: {clean_spk}] {text}",
                    candidate_name=clean_spk,
                    speaker=clean_spk,
                    confidence=0.90,
                )
            )

        # Check regex patterns in dialogue text
        for pattern in NAME_PATTERNS:
            matches = pattern.findall(text)
            for match in matches:
                candidate = match.strip()
                words = candidate.split()
                # If first word is disallowed, take second word
                if words and words[0] in DISALLOWED_CANDIDATE_NAMES and len(words) > 1:
                    candidate = " ".join(words[1:])

                if candidate not in DISALLOWED_CANDIDATE_NAMES and len(candidate) >= 3:
                    evidence_list.append(
                        NameEvidence(
                            timestamp_seconds=ts,
                            text=text,
                            candidate_name=candidate,
                            speaker=speaker,
                            confidence=0.80,
                        )
                    )
                    # If candidate has title + surname e.g. "Detective Miller", also record "Miller"
                    if len(words) >= 2:
                        surname = words[-1]
                        if surname not in DISALLOWED_CANDIDATE_NAMES:
                            evidence_list.append(
                                NameEvidence(
                                    timestamp_seconds=ts,
                                    text=text,
                                    candidate_name=surname,
                                    speaker=speaker,
                                    confidence=0.80,
                                )
                            )

    return evidence_list


# ==============================================================================
# Character Memory & Identity Tracking Engine
# ==============================================================================

class CharacterMemoryTracker:
    """Maintains consistent character tracking across timeline events."""

    def __init__(self, movie_id: str):
        self.movie_id = movie_id
        self.characters: Dict[str, CharacterRecord] = {}
        self.merge_history: List[Dict[str, Any]] = []
        self._next_id_counter = 1

    def _generate_character_id(self) -> str:
        """Create sequential canonical identifier e.g. CHARACTER_001."""
        char_id = f"CHARACTER_{self._next_id_counter:03d}"
        self._next_id_counter += 1
        return char_id

    def load_from_document(self, doc: CharactersDocument) -> None:
        """Hydrate memory tracker from an existing CharactersDocument."""
        self.movie_id = doc.movie_id
        self.characters = {}
        self.merge_history = list(doc.merge_history)

        max_num = 0
        for char in doc.characters:
            self.characters[char.character_id] = char
            match = re.search(r"CHARACTER_(\d+)", char.character_id)
            if match:
                max_num = max(max_num, int(match.group(1)))

        self._next_id_counter = max_num + 1

    def process_event(self, event: EventRecord) -> List[Tuple[str, str]]:
        """
        Process a single timeline event and update character records.
        Returns mapping of (temporary_label, canonical_character_id).
        """
        ts = event.timestamp_seconds
        vis = event.visual
        if isinstance(vis, dict):
            people_count = int(vis.get("people_count", 0))
            labels = list(vis.get("character_labels", []))
            clothing = list(vis.get("clothing", []))
            action = str(vis.get("action", ""))
        else:
            people_count = vis.people_count
            labels = list(vis.character_labels)
            clothing = list(vis.objects)  # fallback or objects
            action = vis.action

        if people_count <= 0 and not labels:
            return []

        # If people_count > 0 but labels is empty, generate temporary label
        if people_count > 0 and not labels:
            labels = [f"PERSON_{chr(65 + i)}" for i in range(people_count)]

        # Extract name evidence from dialogue
        name_evidences = extract_name_evidence_from_dialogue(event.dialogue, ts)

        event_mappings: List[Tuple[str, str]] = []
        current_event_chars: List[CharacterRecord] = []

        for idx, temp_label in enumerate(labels):
            # Match to existing character
            matched_char = self._match_or_create_character(
                temp_label=temp_label,
                timestamp=ts,
                event_index=event.index,
                frame_file=event.frame_file,
                action=action,
            )

            # Record appearance
            appearance = CharacterAppearance(
                timestamp_seconds=ts,
                event_index=event.index,
                frame_file=event.frame_file,
                temporary_label=temp_label,
                match_confidence=0.88,
            )
            matched_char.appearances.append(appearance)
            matched_char.last_seen = ts

            # Attach name evidence if present
            if name_evidences and idx == 0:
                for ev in name_evidences:
                    # Avoid duplicate evidence
                    if not any(existing.text == ev.text and existing.candidate_name == ev.candidate_name for existing in matched_char.name_evidence):
                        matched_char.name_evidence.append(ev)
                self._update_canonical_name(matched_char)

            event_mappings.append((temp_label, matched_char.character_id))
            current_event_chars.append(matched_char)

        # Record co-occurrence relationships
        if len(current_event_chars) >= 2:
            for i in range(len(current_event_chars)):
                for j in range(len(current_event_chars)):
                    if i != j:
                        c1 = current_event_chars[i]
                        c2 = current_event_chars[j]
                        if not any(r.target_character_id == c2.character_id for r in c1.relationships):
                            c1.relationships.append(
                                CharacterRelationship(
                                    target_character_id=c2.character_id,
                                    relationship_type="co_present",
                                    evidence=f"Appeared together at timestamp {ts:.1f}s",
                                )
                            )

        return event_mappings

    def _match_or_create_character(
        self,
        temp_label: str,
        timestamp: float,
        event_index: int,
        frame_file: str,
        action: str,
    ) -> CharacterRecord:
        """
        Match temporary label to existing character using scene continuity and label consistency.
        """
        # 1. Check if same label was seen very recently (within 10s -> strong continuity)
        for char in self.characters.values():
            time_diff = abs(timestamp - char.last_seen)
            if time_diff <= 10.0 and char.appearances:
                last_app = char.appearances[-1]
                if last_app.temporary_label == temp_label:
                    return char

        # 2. Check label match across timeline if unique
        matching_label_chars = [
            c for c in self.characters.values()
            if any(a.temporary_label == temp_label for a in c.appearances)
        ]
        if len(matching_label_chars) == 1:
            return matching_label_chars[0]

        # 3. Create new canonical character
        new_id = self._generate_character_id()
        gender = "unknown"
        if "male" in temp_label.lower() or "man" in temp_label.lower():
            gender = "male"
        elif "female" in temp_label.lower() or "woman" in temp_label.lower():
            gender = "female"

        new_char = CharacterRecord(
            character_id=new_id,
            canonical_name=None,
            name_confidence="unknown",
            first_seen=timestamp,
            last_seen=timestamp,
            appearances=[],
            visual_profile=VisualProfile(gender_presentation=gender),
            name_evidence=[],
            relationships=[],
            identity_confidence="medium",
        )
        self.characters[new_id] = new_char
        return new_char

    def _update_canonical_name(self, char: CharacterRecord) -> None:
        """
        Evaluate accumulated name evidence and resolve canonical_name with confidence threshold.
        """
        if not char.name_evidence:
            char.canonical_name = None
            char.name_confidence = "unknown"
            return

        name_counts: Dict[str, int] = {}
        max_conf_by_name: Dict[str, float] = {}

        for ev in char.name_evidence:
            name = ev.candidate_name
            name_counts[name] = name_counts.get(name, 0) + 1
            max_conf_by_name[name] = max(max_conf_by_name.get(name, 0.0), ev.confidence)

        # Pick most frequent candidate
        best_name = max(name_counts.keys(), key=lambda k: (name_counts[k], max_conf_by_name[k]))
        count = name_counts[best_name]
        top_conf = max_conf_by_name[best_name]

        # Assign with confidence rating
        if count >= 2 or top_conf >= 0.85:
            char.canonical_name = best_name
            char.name_confidence = "high" if count >= 2 else "medium"
        elif top_conf >= 0.60:
            char.canonical_name = best_name
            char.name_confidence = "low"
        else:
            char.canonical_name = None
            char.name_confidence = "unknown"

    def propose_conservative_merge(self, char_id_1: str, char_id_2: str, reason: str) -> bool:
        """
        Consistently merge two characters if strong evidence demonstrates identical identity.
        """
        if char_id_1 not in self.characters or char_id_2 not in self.characters:
            return False

        c1 = self.characters[char_id_1]
        c2 = self.characters[char_id_2]

        # Merge c2 into c1
        c1.first_seen = min(c1.first_seen, c2.first_seen)
        c1.last_seen = max(c1.last_seen, c2.last_seen)
        c1.appearances.extend(c2.appearances)
        c1.appearances.sort(key=lambda a: a.timestamp_seconds)
        c1.name_evidence.extend(c2.name_evidence)
        self._update_canonical_name(c1)

        # Log merge
        self.merge_history.append({
            "primary_character_id": char_id_1,
            "merged_character_id": char_id_2,
            "reason": reason,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })

        del self.characters[char_id_2]
        return True

    def export_document(self) -> CharactersDocument:
        """Export current state into CharactersDocument."""
        sorted_chars = sorted(self.characters.values(), key=lambda c: c.first_seen)
        return CharactersDocument(
            movie_id=self.movie_id,
            total_characters=len(sorted_chars),
            characters=sorted_chars,
            merge_history=self.merge_history,
        )


# ==============================================================================
# Pipeline Execution
# ==============================================================================

def track_movie_characters(
    source_path: str | Path,
    provider_name: str = "gemini",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    movie_id: Optional[str] = None,
    output_base_dir: Optional[Path] = None,
    limit: Optional[int] = None,
    force: bool = False,
    dry_run: bool = False,
    ai_provider: Optional[BaseAIProvider] = None,
) -> CharactersDocument:
    """
    Phase 5 Pipeline:
    1. Loads Phase 4 events.json (or runs Phase 4 if needed).
    2. Builds/resumes character memory.
    3. Resolves names from dialogue evidence.
    4. Saves analysis/<movie_id>/characters.json.
    """
    resolved_source = Path(source_path).resolve()
    if not resolved_source.exists():
        raise SourceFileNotFoundError(f"Movie file not found: '{resolved_source}'")

    assigned_movie_id = movie_id or sanitize_movie_id(resolved_source.name)
    if output_base_dir is None:
        workspace_root = Path(__file__).resolve().parents[1]
        output_base_dir = workspace_root / "analysis"

    movie_analysis_dir = output_base_dir / assigned_movie_id
    movie_analysis_dir.mkdir(parents=True, exist_ok=True)

    events_json_path = movie_analysis_dir / "events.json"
    characters_json_path = movie_analysis_dir / "characters.json"

    # 1. Ensure events.json exists (Phase 4)
    if not events_json_path.exists() or force:
        logger.info("events.json not found or force=True; executing Phase 4 event analysis...")
        events_doc = analyze_timeline_events(
            source_path=resolved_source,
            provider_name=provider_name,
            model_name=model_name,
            api_key=api_key,
            movie_id=assigned_movie_id,
            output_base_dir=output_base_dir,
            force=force,
            dry_run=dry_run,
            ai_provider=ai_provider,
        )
    else:
        with open(events_json_path, "r", encoding="utf-8") as f:
            e_data = json.load(f)
        events_list: List[EventRecord] = []
        for item in e_data.get("events", []):
            events_list.append(
                EventRecord(
                    index=int(item["index"]),
                    timestamp_seconds=float(item["timestamp_seconds"]),
                    timestamp=str(item["timestamp"]),
                    frame_file=str(item["frame_file"]),
                    dialogue=list(item.get("dialogue", [])),
                    visual=item.get("visual", {}),
                    event_summary=str(item.get("event_summary", "")),
                    plot_significance=int(item.get("plot_significance", 5)),
                    uncertainty=item.get("uncertainty", {}),
                )
            )
        events_doc = EventsDocument(
            movie_id=assigned_movie_id,
            provider=e_data.get("provider", "unknown"),
            model=e_data.get("model", "unknown"),
            total_events=len(events_list),
            events=events_list,
        )

    # 2. Setup Character Memory Tracker
    tracker = CharacterMemoryTracker(movie_id=assigned_movie_id)

    # Check for existing characters.json for resume
    if characters_json_path.exists() and not force:
        try:
            with open(characters_json_path, "r", encoding="utf-8") as f:
                c_data = json.load(f)
            chars_list: List[CharacterRecord] = []
            for c in c_data.get("characters", []):
                apps = [
                    CharacterAppearance(
                        timestamp_seconds=float(a["timestamp_seconds"]),
                        event_index=int(a["event_index"]),
                        frame_file=str(a["frame_file"]),
                        temporary_label=str(a["temporary_label"]),
                        match_confidence=float(a.get("match_confidence", 0.8)),
                    )
                    for a in c.get("appearances", [])
                ]
                evs = [
                    NameEvidence(
                        timestamp_seconds=float(n["timestamp_seconds"]),
                        text=str(n["text"]),
                        candidate_name=str(n["candidate_name"]),
                        speaker=n.get("speaker"),
                        confidence=float(n.get("confidence", 0.5)),
                    )
                    for n in c.get("name_evidence", [])
                ]
                vis_p = c.get("visual_profile", {})
                profile = VisualProfile(
                    gender_presentation=str(vis_p.get("gender_presentation", "unknown")),
                    approx_age_group=str(vis_p.get("approx_age_group", "adult")),
                    hair=vis_p.get("hair"),
                    clothing=list(vis_p.get("clothing", [])),
                    distinctive_features=list(vis_p.get("distinctive_features", [])),
                )
                rels = [
                    CharacterRelationship(
                        target_character_id=str(r["target_character_id"]),
                        relationship_type=str(r["relationship_type"]),
                        evidence=str(r["evidence"]),
                    )
                    for r in c.get("relationships", [])
                ]
                rec = CharacterRecord(
                    character_id=str(c["character_id"]),
                    canonical_name=c.get("canonical_name"),
                    name_confidence=str(c.get("name_confidence", "unknown")),
                    first_seen=float(c["first_seen"]),
                    last_seen=float(c["last_seen"]),
                    appearances=apps,
                    visual_profile=profile,
                    name_evidence=evs,
                    relationships=rels,
                    identity_confidence=str(c.get("identity_confidence", "medium")),
                )
                chars_list.append(rec)
            doc_existing = CharactersDocument(
                movie_id=assigned_movie_id,
                total_characters=len(chars_list),
                characters=chars_list,
                merge_history=list(c_data.get("merge_history", [])),
            )
            tracker.load_from_document(doc_existing)
            logger.info("Loaded %d existing characters from characters.json.", len(chars_list))
        except Exception as exc:
            logger.warning("Could not load existing characters.json; starting fresh: %s", exc)

    # 3. Dry Run Mode
    events_to_process = events_doc.events
    if limit is not None and limit > 0:
        events_to_process = events_to_process[:limit]

    if dry_run:
        print("\n" + "=" * 60)
        print("[DRY-RUN] PHASE 5 - CHARACTER TRACKING ESTIMATE")
        print("=" * 60)
        print(f"  Movie ID               : {assigned_movie_id}")
        print(f"  Total Timeline Events  : {len(events_doc.events)}")
        print(f"  Events to Process      : {len(events_to_process)} (limit={limit})")
        print(f"  Tracked Characters Now : {len(tracker.characters)}")
        print("=" * 60 + "\n")
        return tracker.export_document()

    # 4. Incremental Event Processing
    logger.info("Tracking characters across %d timeline events...", len(events_to_process))
    for evt in events_to_process:
        tracker.process_event(evt)
        # Checkpoint save
        current_doc = tracker.export_document()
        with open(characters_json_path, "w", encoding="utf-8") as f:
            f.write(current_doc.to_json(indent=2))

    final_doc = tracker.export_document()
    with open(characters_json_path, "w", encoding="utf-8") as f:
        f.write(final_doc.to_json(indent=2))

    logger.info("Character tracking complete. Total characters: %d (JSON: %s)", final_doc.total_characters, characters_json_path)
    return final_doc


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main() -> int:
    """CLI entry point for Phase 5 character tracking."""
    parser = argparse.ArgumentParser(
        description="Movie Review Engine - Phase 5: Character Identity Tracking & Memory."
    )
    parser.add_argument(
        "movie_path",
        type=str,
        help="Path to the movie file (e.g. 'movie.mp4' or full path).",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="gemini",
        help="Multimodal AI provider ('gemini', 'mock', default: 'gemini').",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of events to process (e.g. --limit 10).",
    )
    parser.add_argument(
        "--movie-id",
        type=str,
        default=None,
        help="Optional custom movie_id identifier.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional override base analysis directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run character tracking estimate without modifying state.",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force rebuild/re-track all characters from scratch.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable detailed debug logging.",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    out_base = Path(args.output_dir) if args.output_dir else None

    try:
        doc = track_movie_characters(
            source_path=args.movie_path,
            provider_name=args.provider,
            movie_id=args.movie_id,
            output_base_dir=out_base,
            limit=args.limit,
            force=args.force,
            dry_run=args.dry_run,
        )

        if not args.dry_run:
            print("\n" + "=" * 60)
            print(f"[SUCCESS] CHARACTER TRACKING COMPLETED: {doc.movie_id}")
            print("=" * 60)
            print(f"  Total Characters : {doc.total_characters}")
            for char in doc.characters:
                name_str = f"\"{char.canonical_name}\" ({char.name_confidence})" if char.canonical_name else "Unconfirmed"
                print(f"  * {char.character_id}: Name={name_str}, Seen {len(char.appearances)}x ({char.first_seen:.1f}s - {char.last_seen:.1f}s)")
            print("=" * 60 + "\n")
        return 0
    except Exception as exc:
        logger.exception("Unexpected failure during character tracking")
        print(f"\n[ERROR] CHARACTER TRACKING FAILURE: {exc}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
