"""
Automated Test Suite for Phase 7: Story Reconstruction Engine
Tests scene grouping, boundaries, beats, causality, character arcs, uncertainty preservation,
partial movie detection, malformed events, empty dialogue, schema validity, and CLI execution.
"""

import json
import subprocess
import sys
from pathlib import Path
import pytest

from story_engine.models import (
    CausalLink,
    CharacterArc,
    ProtagonistCandidate,
    SceneRecord,
    StoryBeat,
    StoryDocument,
)
from story_engine.reconstruct_story import (
    StoryReconstructionEngine,
    StoryReconstructionError,
    reconstruct_movie_story,
)


@pytest.fixture
def mock_story_environment(tmp_path: Path) -> Path:
    """Create a structured test movie analysis environment with synthetic evidence."""
    analysis_dir = tmp_path / "analysis" / "test_movie_alpha"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # 1. movie_metadata.json
    metadata = {
        "movie_id": "test_movie_alpha",
        "duration_seconds": 1200.0,
        "resolution": "1920x1080",
        "fps": 24.0,
    }
    (analysis_dir / "movie_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # 2. timeline_index.json
    timeline = {
        "movie_id": "test_movie_alpha",
        "sampling_interval_seconds": 30.0,
        "duration_seconds": 1200.0,
        "total_samples": 40,
        "frames": [
            {"index": i, "timestamp_seconds": (i - 1) * 30.0, "timestamp": f"00:00:{(i-1)*30:02d}.000"}
            for i in range(1, 41)
        ],
    }
    (analysis_dir / "timeline_index.json").write_text(json.dumps(timeline, indent=2), encoding="utf-8")

    # 3. events.json (10 events -> covers 300s out of 1200s -> PARTIAL)
    events_data = {
        "movie_id": "test_movie_alpha",
        "provider": "mock",
        "model": "mock-model-v1",
        "total_events": 10,
        "events": [
            {
                "index": 1,
                "timestamp_seconds": 0.0,
                "timestamp": "00:00:00.000",
                "frame_file": "frames/frame_000001.jpg",
                "dialogue": [],
                "visual": {
                    "people_count": 0,
                    "character_labels": [],
                    "location": "Black screen",
                    "action": "Opening title text appears",
                    "objects": [],
                    "emotion": "neutral",
                },
                "event_summary": "Opening title text fades in over a black background.",
                "plot_significance": 1,
                "uncertainty": {"character_identity": "low", "event_interpretation": "low", "location_certainty": "high"},
            },
            {
                "index": 2,
                "timestamp_seconds": 30.0,
                "timestamp": "00:00:30.000",
                "frame_file": "frames/frame_000002.jpg",
                "dialogue": [{"index": 1, "text": "This is Captain Miller recording log one."}],
                "visual": {
                    "people_count": 1,
                    "character_labels": ["PERSON_A"],
                    "location": "Control room",
                    "action": "Adjusting audio transmitter",
                    "objects": ["radio", "log book"],
                    "emotion": "focused",
                },
                "event_summary": "Captain Miller operates radio equipment in the control room while recording a voice log.",
                "plot_significance": 6,
                "uncertainty": {"character_identity": "low", "event_interpretation": "low", "location_certainty": "high"},
            },
            {
                "index": 3,
                "timestamp_seconds": 60.0,
                "timestamp": "00:01:00.000",
                "frame_file": "frames/frame_000003.jpg",
                "dialogue": [{"index": 2, "text": "Warning: perimeter breach detected in sector 4."}],
                "visual": {
                    "people_count": 1,
                    "character_labels": ["PERSON_A"],
                    "location": "Control room",
                    "action": "Examining alarm console",
                    "objects": ["flashing red beacon", "display screen"],
                    "emotion": "tense",
                },
                "event_summary": "An alarm sounds in the control room indicating an active perimeter breach.",
                "plot_significance": 7,
                "uncertainty": {"character_identity": "low", "event_interpretation": "low", "location_certainty": "high"},
            },
            {
                "index": 4,
                "timestamp_seconds": 90.0,
                "timestamp": "00:01:30.000",
                "frame_file": "frames/frame_000004.jpg",
                "dialogue": [{"index": 3, "text": "Grab your gear, we need to move now!"}],
                "visual": {
                    "people_count": 2,
                    "character_labels": ["PERSON_A", "PERSON_B"],
                    "location": "Corridor outside control room",
                    "action": "Rushing down the metallic corridor with equipment",
                    "objects": ["rifles", "emergency lights"],
                    "emotion": "urgent",
                },
                "event_summary": "Captain Miller and Officer Dave rush through the corridor to respond to the breach.",
                "plot_significance": 7,
                "uncertainty": {"character_identity": "low", "event_interpretation": "low", "location_certainty": "medium"},
            },
            {
                "index": 5,
                "timestamp_seconds": 120.0,
                "timestamp": "00:02:00.000",
                "frame_file": "frames/frame_000005.jpg",
                "dialogue": [{"index": 4, "text": "The gate has been torn open from the outside."}],
                "visual": {
                    "people_count": 2,
                    "character_labels": ["PERSON_A", "PERSON_B"],
                    "location": "Outer perimeter gate",
                    "action": "Inspecting mangled metal fence in heavy rain",
                    "objects": ["torn wire", "claw marks"],
                    "emotion": "shocked",
                },
                "event_summary": "The officers discover massive claw marks on the breached outer perimeter gate.",
                "plot_significance": 8,
                "uncertainty": {"character_identity": "low", "event_interpretation": "medium", "location_certainty": "high"},
            },
            {
                "index": 6,
                "timestamp_seconds": 150.0,
                "timestamp": "00:02:30.000",
                "frame_file": "frames/frame_000006.jpg",
                "dialogue": [],
                "visual": {
                    "people_count": 0,
                    "character_labels": [],
                    "location": "Dark woods surrounding base",
                    "action": "Atmospheric shot of swirling fog and heavy rain among pine trees",
                    "objects": ["trees", "rain"],
                    "emotion": "ominous",
                },
                "event_summary": "Swirling fog and darkness envelop the surrounding forest outside the base.",
                "plot_significance": 4,
                "uncertainty": {"character_identity": "high", "event_interpretation": "medium", "location_certainty": "medium"},
            },
            {
                "index": 7,
                "timestamp_seconds": 180.0,
                "timestamp": "00:03:00.000",
                "frame_file": "frames/frame_000007.jpg",
                "dialogue": [{"index": 5, "text": "Look out! Above you!"}],
                "visual": {
                    "people_count": 2,
                    "character_labels": ["PERSON_A", "PERSON_B"],
                    "location": "Dark woods surrounding base",
                    "action": "Taking aim at a shadowy creature leaping from branches",
                    "objects": ["flashlight beam", "creature silhouette"],
                    "emotion": "terrified",
                },
                "event_summary": "A shadowy creature attacks the officers from above in the dark woods.",
                "plot_significance": 9,
                "uncertainty": {"character_identity": "low", "event_interpretation": "low", "location_certainty": "medium"},
            },
            {
                "index": 8,
                "timestamp_seconds": 210.0,
                "timestamp": "00:03:30.000",
                "frame_file": "frames/frame_000008.jpg",
                "dialogue": [{"index": 6, "text": "Dave is down! Fall back to the bunker!"}],
                "visual": {
                    "people_count": 1,
                    "character_labels": ["PERSON_A"],
                    "location": "Dark woods surrounding base",
                    "action": "Dragging wounded comrade toward bunker entrance",
                    "objects": ["mud", "bunker blast door"],
                    "emotion": "desperate",
                },
                "event_summary": "Captain Miller drags his wounded comrade back toward the blast doors under heavy assault.",
                "plot_significance": 9,
                "uncertainty": {"character_identity": "low", "event_interpretation": "low", "location_certainty": "high"},
            },
            {
                "index": 9,
                "timestamp_seconds": 240.0,
                "timestamp": "00:04:00.000",
                "frame_file": "frames/frame_000009.jpg",
                "dialogue": [{"index": 7, "text": "Sealing emergency locks."}],
                "visual": {
                    "people_count": 1,
                    "character_labels": ["PERSON_A"],
                    "location": "Underground bunker interior",
                    "action": "Pulling heavy hydraulic lever to seal blast door",
                    "objects": ["hydraulic levers", "sealed door"],
                    "emotion": "exhausted relief",
                },
                "event_summary": "Captain Miller successfully seals the underground bunker blast doors.",
                "plot_significance": 8,
                "uncertainty": {"character_identity": "low", "event_interpretation": "low", "location_certainty": "high"},
            },
            {
                "index": 10,
                "timestamp_seconds": 270.0,
                "timestamp": "00:04:30.000",
                "frame_file": "frames/frame_000010.jpg",
                "dialogue": [{"index": 8, "text": "We're safe for now, but power is failing."}],
                "visual": {
                    "people_count": 1,
                    "character_labels": ["PERSON_A"],
                    "location": "Underground bunker interior",
                    "action": "Applying emergency bandage to comrade under flickering emergency lights",
                    "objects": ["first aid kit", "flickering lights"],
                    "emotion": "somber",
                },
                "event_summary": "Inside the bunker, Miller tends to Dave's injuries as backup power begins to flicker.",
                "plot_significance": 7,
                "uncertainty": {"character_identity": "low", "event_interpretation": "low", "location_certainty": "high"},
            },
        ],
    }
    (analysis_dir / "events.json").write_text(json.dumps(events_data, indent=2), encoding="utf-8")

    # 4. characters.json
    characters_data = {
        "movie_id": "test_movie_alpha",
        "total_characters": 2,
        "characters": [
            {
                "character_id": "CHARACTER_001",
                "canonical_name": "Captain Miller",
                "name_confidence": "high",
                "first_seen": 30.0,
                "last_seen": 270.0,
                "appearances": [
                    {"timestamp_seconds": t, "event_index": idx, "frame_file": f"frames/frame_{idx:06d}.jpg", "temporary_label": "PERSON_A", "match_confidence": 0.95}
                    for idx, t in [(2, 30.0), (3, 60.0), (4, 90.0), (5, 120.0), (7, 180.0), (8, 210.0), (9, 240.0), (10, 270.0)]
                ],
                "name_evidence": [{"timestamp_seconds": 30.0, "candidate_name": "Captain Miller", "confidence": 0.9}],
            },
            {
                "character_id": "CHARACTER_002",
                "canonical_name": "Officer Dave",
                "name_confidence": "high",
                "first_seen": 90.0,
                "last_seen": 210.0,
                "appearances": [
                    {"timestamp_seconds": t, "event_index": idx, "frame_file": f"frames/frame_{idx:06d}.jpg", "temporary_label": "PERSON_B", "match_confidence": 0.92}
                    for idx, t in [(4, 90.0), (5, 120.0), (7, 180.0)]
                ],
                "name_evidence": [{"timestamp_seconds": 210.0, "candidate_name": "Dave", "confidence": 0.9}],
            },
        ],
    }
    (analysis_dir / "characters.json").write_text(json.dumps(characters_data, indent=2), encoding="utf-8")

    # 5. dialogue.json
    dialogue_data = {
        "movie_id": "test_movie_alpha",
        "language": "eng",
        "total_entries": 8,
        "dialogue": [
            {"index": 1, "start_seconds": 30.0, "end_seconds": 35.0, "text": "This is Captain Miller recording log one."},
            {"index": 2, "start_seconds": 60.0, "end_seconds": 65.0, "text": "Warning: perimeter breach detected in sector 4."},
            {"index": 3, "start_seconds": 90.0, "end_seconds": 95.0, "text": "Grab your gear, we need to move now!"},
            {"index": 4, "start_seconds": 120.0, "end_seconds": 125.0, "text": "The gate has been torn open from the outside."},
            {"index": 5, "start_seconds": 180.0, "end_seconds": 185.0, "text": "Look out! Above you!"},
            {"index": 6, "start_seconds": 210.0, "end_seconds": 215.0, "text": "Dave is down! Fall back to the bunker!"},
            {"index": 7, "start_seconds": 240.0, "end_seconds": 245.0, "text": "Sealing emergency locks."},
            {"index": 8, "start_seconds": 270.0, "end_seconds": 275.0, "text": "We're safe for now, but power is failing."},
        ],
    }
    (analysis_dir / "dialogue.json").write_text(json.dumps(dialogue_data, indent=2), encoding="utf-8")

    return analysis_dir


# ==============================================================================
# Unit & Functional Tests
# ==============================================================================

def test_chronological_scene_grouping(mock_story_environment: Path):
    """Test 1: Chronological event grouping into coherent scenes."""
    engine = StoryReconstructionEngine(
        movie_id="test_movie_alpha",
        analysis_dir=mock_story_environment,
    )
    story = engine.reconstruct()

    assert len(story.scenes) >= 3
    # Check chronological ordering of scenes
    for i in range(len(story.scenes) - 1):
        assert story.scenes[i].start_seconds <= story.scenes[i + 1].start_seconds
        assert story.scenes[i].end_seconds <= story.scenes[i + 1].end_seconds + 30.0


def test_scene_boundaries_on_location_change(mock_story_environment: Path):
    """Test 2: Scene boundaries trigger on distinct location changes."""
    engine = StoryReconstructionEngine(
        movie_id="test_movie_alpha",
        analysis_dir=mock_story_environment,
    )
    story = engine.reconstruct()

    # Event 1 (Black screen) must not be merged with outdoor woods or bunker
    scene_1 = story.scenes[0]
    assert 1 in scene_1.event_indices
    assert "Black screen" in scene_1.location or "opening" in scene_1.location.lower()


def test_story_beat_creation_and_associations(mock_story_environment: Path):
    """Test 3: Beats combine scenes and preserve supporting scene IDs and characters."""
    engine = StoryReconstructionEngine(
        movie_id="test_movie_alpha",
        analysis_dir=mock_story_environment,
    )
    story = engine.reconstruct()

    assert len(story.beats) >= 1
    for beat in story.beats:
        assert beat.beat_id.startswith("BEAT_")
        assert len(beat.supporting_scenes) > 0
        assert beat.title
        assert beat.description
        assert 1 <= beat.importance <= 10


def test_cause_effect_linking_grounded_in_evidence(mock_story_environment: Path):
    """Test 4 & 5: Causal links have explicit evidence event indices and no hallucinated links."""
    engine = StoryReconstructionEngine(
        movie_id="test_movie_alpha",
        analysis_dir=mock_story_environment,
    )
    story = engine.reconstruct()

    assert len(story.causal_links) > 0
    all_event_indices = {e["index"] for e in engine.load_inputs()[0]["events"]}

    for link in story.causal_links:
        assert link.cause
        assert link.effect
        assert link.confidence in ("low", "medium", "high")
        assert len(link.evidence_events) > 0
        # Check every evidence event index really exists in events.json
        for ev_idx in link.evidence_events:
            assert ev_idx in all_event_indices


def test_character_arc_extraction(mock_story_environment: Path):
    """Test 6: Character progression, objectives, and decisions are extracted cleanly."""
    engine = StoryReconstructionEngine(
        movie_id="test_movie_alpha",
        analysis_dir=mock_story_environment,
    )
    story = engine.reconstruct()

    assert len(story.character_arcs) == 2
    miller_arc = next(a for a in story.character_arcs if "Miller" in (a.canonical_name or ""))
    assert miller_arc.introduction
    assert miller_arc.objective
    assert miller_arc.conflict
    assert miller_arc.change
    assert len(miller_arc.major_decisions) > 0
    assert miller_arc.outcome


def test_protagonist_prominence_ranking(mock_story_environment: Path):
    """Test: Protagonist identification ranks Captain Miller first due to frequency and screen time."""
    engine = StoryReconstructionEngine(
        movie_id="test_movie_alpha",
        analysis_dir=mock_story_environment,
    )
    story = engine.reconstruct()

    assert len(story.protagonist_candidates) == 2
    top_p = story.protagonist_candidates[0]
    assert top_p.canonical_name == "Captain Miller"
    assert top_p.prominence_score > story.protagonist_candidates[1].prominence_score
    assert top_p.event_count == 8
    assert "Captain Miller" in top_p.rationale


def test_uncertainty_preservation(mock_story_environment: Path):
    """Test 7: Ambiguities and partial status notices are preserved in open_uncertainties."""
    engine = StoryReconstructionEngine(
        movie_id="test_movie_alpha",
        analysis_dir=mock_story_environment,
    )
    story = engine.reconstruct()

    assert len(story.open_uncertainties) > 0
    # Must record that coverage is partial
    assert any("PARTIAL" in u for u in story.open_uncertainties)


def test_partial_vs_complete_movie_detection(tmp_path: Path):
    """Test 8: Coverage detection correctly marks partial vs complete datasets."""
    # Scenario A: Partial (10 events out of 100 samples)
    dir_partial = tmp_path / "analysis" / "movie_partial"
    dir_partial.mkdir(parents=True, exist_ok=True)
    (dir_partial / "movie_metadata.json").write_text(json.dumps({"duration_seconds": 3000.0}), encoding="utf-8")
    (dir_partial / "timeline_index.json").write_text(json.dumps({"total_samples": 100, "duration_seconds": 3000.0}), encoding="utf-8")
    (dir_partial / "events.json").write_text(json.dumps({
        "events": [{"index": i, "timestamp_seconds": i * 30.0, "event_summary": "Event"} for i in range(1, 11)]
    }), encoding="utf-8")

    engine_partial = StoryReconstructionEngine("movie_partial", dir_partial)
    doc_partial = engine_partial.reconstruct()
    assert doc_partial.story_status == "PARTIAL"

    # Scenario B: Complete (30 events out of 30 samples)
    dir_complete = tmp_path / "analysis" / "movie_complete"
    dir_complete.mkdir(parents=True, exist_ok=True)
    (dir_complete / "movie_metadata.json").write_text(json.dumps({"duration_seconds": 90.0}), encoding="utf-8")
    (dir_complete / "timeline_index.json").write_text(json.dumps({"total_samples": 3, "duration_seconds": 90.0}), encoding="utf-8")
    (dir_complete / "events.json").write_text(json.dumps({
        "events": [
            {"index": 1, "timestamp_seconds": 0.0, "visual": {"location": "Room"}, "event_summary": "Start"},
            {"index": 2, "timestamp_seconds": 30.0, "visual": {"location": "Room"}, "event_summary": "Middle"},
            {"index": 3, "timestamp_seconds": 60.0, "visual": {"location": "Room"}, "event_summary": "End"},
        ]
    }), encoding="utf-8")

    engine_complete = StoryReconstructionEngine("movie_complete", dir_complete)
    doc_complete = engine_complete.reconstruct()
    assert doc_complete.story_status == "COMPLETE"


def test_malformed_event_handling(tmp_path: Path):
    """Test 9: Malformed event objects with missing fields or corrupt types are handled gracefully."""
    dir_malformed = tmp_path / "analysis" / "movie_malformed"
    dir_malformed.mkdir(parents=True, exist_ok=True)
    (dir_malformed / "events.json").write_text(json.dumps({
        "events": [
            {"index": "not_an_int", "timestamp_seconds": "invalid", "event_summary": None},
            {"missing_fields": True},
        ]
    }), encoding="utf-8")

    engine = StoryReconstructionEngine("movie_malformed", dir_malformed)
    doc = engine.reconstruct()
    assert isinstance(doc, StoryDocument)
    assert len(doc.scenes) >= 1


def test_empty_dialogue_handling(tmp_path: Path):
    """Test 10: Empty dialogue file (silent movie or missing subtitles) completes successfully."""
    dir_empty_diag = tmp_path / "analysis" / "movie_silent"
    dir_empty_diag.mkdir(parents=True, exist_ok=True)
    (dir_empty_diag / "events.json").write_text(json.dumps({
        "events": [
            {"index": 1, "timestamp_seconds": 0.0, "dialogue": [], "visual": {"location": "Lake"}, "event_summary": "Boat floats silently."},
            {"index": 2, "timestamp_seconds": 30.0, "dialogue": [], "visual": {"location": "Lake"}, "event_summary": "Sun sets over water."},
        ]
    }), encoding="utf-8")
    (dir_empty_diag / "dialogue.json").write_text(json.dumps({"total_entries": 0, "dialogue": []}), encoding="utf-8")

    engine = StoryReconstructionEngine("movie_silent", dir_empty_diag)
    doc = engine.reconstruct()
    assert isinstance(doc, StoryDocument)
    assert len(doc.scenes) == 1
    assert doc.scenes[0].location == "Lake"


def test_story_json_serialization_and_schema(mock_story_environment: Path):
    """Test 11 & 12: story.json is written, valid JSON, and deserializable into StoryDocument."""
    engine = StoryReconstructionEngine(
        movie_id="test_movie_alpha",
        analysis_dir=mock_story_environment,
    )
    story = engine.reconstruct()
    story_path = engine.save_story_document(story)

    assert story_path.exists()
    assert story_path.is_file()

    with open(story_path, "r", encoding="utf-8") as f:
        loaded_json = json.load(f)

    # Validate top-level schema keys
    expected_keys = {
        "movie_id", "story_status", "protagonist_candidates", "scenes",
        "beats", "causal_links", "character_arcs", "story_summary", "open_uncertainties"
    }
    assert expected_keys.issubset(set(loaded_json.keys()))

    # Roundtrip from_dict validation
    roundtrip_doc = StoryDocument.from_dict(loaded_json)
    assert roundtrip_doc.movie_id == story.movie_id
    assert len(roundtrip_doc.scenes) == len(story.scenes)
    assert len(roundtrip_doc.beats) == len(story.beats)


def test_cli_reconstruct_story(mock_story_environment: Path):
    """Test 13: CLI execution python -m story_engine.reconstruct_story."""
    cmd = [
        sys.executable, "-m", "story_engine.reconstruct_story",
        "test_movie_alpha",
        "--output-dir", str(mock_story_environment.parent),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "STORY RECONSTRUCTION COMPLETE" in res.stdout
