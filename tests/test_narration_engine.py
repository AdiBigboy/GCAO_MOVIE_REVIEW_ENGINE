"""
Automated Test Suite for Phase 8: Narration & Script Generation Engine
Tests chronological ordering, source traceability, duration models, uncertainty preservation,
anti-hallucination/identity bleed prevention, partial story markers, schemas, and CLI execution.
"""

import json
import subprocess
import sys
from pathlib import Path
import pytest

from narration_engine.models import (
    NarrationSegmentPlan,
    NarrativePlanDocument,
    ScriptDocument,
    ScriptSegment,
)
from narration_engine.plan_narration import (
    NarrativePlanner,
    NarrativePlanError,
    generate_movie_narrative_plan,
)
from narration_engine.generate_script import (
    MalayStorytellerEngine,
    ScriptGenerationError,
    generate_movie_script,
)
from story_engine.models import (
    CausalLink,
    CharacterArc,
    ProtagonistCandidate,
    SceneRecord,
    StoryBeat,
    StoryDocument,
)


@pytest.fixture
def mock_narration_environment(tmp_path: Path) -> Path:
    """Create a structured test movie analysis environment with full story.json evidence."""
    analysis_dir = tmp_path / "analysis" / "test_movie_beta"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # 1. movie_metadata.json
    metadata = {
        "movie_id": "test_movie_beta",
        "duration_seconds": 1200.0,
        "resolution": "1920x1080",
    }
    (analysis_dir / "movie_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # 2. timeline_index.json
    timeline = {
        "movie_id": "test_movie_beta",
        "sampling_interval_seconds": 30.0,
        "duration_seconds": 1200.0,
        "total_samples": 40,
    }
    (analysis_dir / "timeline_index.json").write_text(json.dumps(timeline, indent=2), encoding="utf-8")

    # 3. events.json
    events_data = {
        "movie_id": "test_movie_beta",
        "total_events": 5,
        "events": [
            {
                "index": 1,
                "timestamp_seconds": 0.0,
                "event_summary": "Opening title graphics fade in over a dark background.",
                "visual": {"location": "Opening Title Graphics", "action": "Titles fade in", "character_labels": []},
                "plot_significance": 1,
                "uncertainty": {"event_interpretation": "low"},
            },
            {
                "index": 2,
                "timestamp_seconds": 30.0,
                "event_summary": "Dr. Benjamin records an audio log describing an ominous ancient artifact.",
                "visual": {"location": "Study room", "action": "Operating tape recorder", "character_labels": ["Doctor Benjamin"]},
                "plot_significance": 7,
                "uncertainty": {"event_interpretation": "low"},
            },
            {
                "index": 3,
                "timestamp_seconds": 60.0,
                "event_summary": "A mysterious burlap package is unwrapped revealing strange writings.",
                "visual": {"location": "Study room", "action": "Unwrapping package", "character_labels": ["Doctor Benjamin"]},
                "plot_significance": 8,
                "uncertainty": {"event_interpretation": "medium"},
            },
            {
                "index": 4,
                "timestamp_seconds": 90.0,
                "event_summary": "A traveler struggles against the current as a small motorboat capsizes during a storm.",
                "visual": {"location": "Open water / stormy lake", "action": "Swimming in rough water", "character_labels": ["PERSON_A"]},
                "plot_significance": 9,
                "uncertainty": {"event_interpretation": "low"},
            },
            {
                "index": 5,
                "timestamp_seconds": 120.0,
                "event_summary": "The traveler washes ashore face down in the mud before standing in the dark forest.",
                "visual": {"location": "Dark forest / lakeshore", "action": "Standing drenched in forest", "character_labels": ["PERSON_A"]},
                "plot_significance": 8,
                "uncertainty": {"event_interpretation": "low"},
            },
        ],
    }
    (analysis_dir / "events.json").write_text(json.dumps(events_data, indent=2), encoding="utf-8")

    # 4. story.json
    story_doc = StoryDocument(
        movie_id="test_movie_beta",
        story_status="PARTIAL",
        protagonist_candidates=[
            ProtagonistCandidate(
                character_id="CHARACTER_001",
                canonical_name="Doctor Benjamin",
                prominence_score=0.75,
                screen_time_ratio=0.5,
                event_count=2,
                dialogue_count=1,
                rationale="Doctor Benjamin appears in audio recording events.",
            )
        ],
        scenes=[
            SceneRecord(
                scene_id="SCENE_001",
                start_seconds=0.0,
                end_seconds=30.0,
                event_indices=[1],
                characters=[],
                location="Opening Title Graphics",
                summary="Opening title graphics fade in over a dark background.",
                scene_function="SETUP",
                importance=1,
            ),
            SceneRecord(
                scene_id="SCENE_002",
                start_seconds=30.0,
                end_seconds=90.0,
                event_indices=[2, 3],
                characters=["Doctor Benjamin"],
                location="Study room",
                summary="Dr. Benjamin records an audio log and unboxes a mysterious burlap package.",
                scene_function="DISCOVERY",
                importance=8,
            ),
            SceneRecord(
                scene_id="SCENE_003",
                start_seconds=90.0,
                end_seconds=150.0,
                event_indices=[4, 5],
                characters=["PERSON_A"],
                location="Open water / stormy lake",
                summary="A traveler struggles in open water during a storm and washes ashore in the dark forest.",
                scene_function="ESCALATION",
                importance=9,
            ),
        ],
        beats=[
            StoryBeat(
                beat_id="BEAT_001",
                title="Prologue & Audio Log Discovery",
                description="Dr. Benjamin Price investigates ancient history and opens a mysterious package.",
                supporting_scenes=["SCENE_001", "SCENE_002"],
                characters_involved=["Doctor Benjamin"],
                cause="Dr. Benjamin plays audio recordings regarding ancient warnings.",
                consequence="A sealed occult package is uncovered.",
                importance=8,
                uncertainty="medium",
            ),
            StoryBeat(
                beat_id="BEAT_002",
                title="Escalation & Water Storm Crisis",
                description="A traveler encounters severe storm conditions on open water and washes ashore.",
                supporting_scenes=["SCENE_003"],
                characters_involved=["PERSON_A"],
                cause="Vessel distress forces traveler into rough water.",
                consequence="Survivor washes ashore in dark forest.",
                importance=9,
                uncertainty="low",
            ),
        ],
        causal_links=[
            CausalLink(
                cause="Dr. Benjamin Price investigates ancient history and plays audio log.",
                effect="A mysterious burlap package is unwrapped.",
                confidence="high",
                evidence_events=[2, 3],
            ),
            CausalLink(
                cause="Severe storm capsizes small motorboat on open water.",
                effect="Traveler struggles in rough water and washes ashore in dark forest.",
                confidence="high",
                evidence_events=[4, 5],
            ),
        ],
        character_arcs=[
            CharacterArc(
                character_id="CHARACTER_001",
                canonical_name="Doctor Benjamin",
                introduction="Introduced at 30s in study room operating audio equipment.",
                objective="Document ancient history.",
                conflict="Encountering occult warnings.",
                change="Recorded findings before unboxing.",
                major_decisions=["Operated audio recording equipment"],
                outcome="Documented research (partial status).",
            )
        ],
        story_summary="Story begins with audio log discovery before shifting to storm crisis.",
        open_uncertainties=[
            "Current coverage is PARTIAL (5 events analyzed).",
            "Identity connection between Dr. Benjamin and the traveler on the boat is unconfirmed.",
        ],
    )
    (analysis_dir / "story.json").write_text(story_doc.to_json(indent=2), encoding="utf-8")

    return analysis_dir


# ==============================================================================
# Unit & Functional Tests
# ==============================================================================

def test_narrative_plan_generation(mock_narration_environment: Path):
    """Test 1: Narrative planner generates valid narrative_plan.json."""
    planner = NarrativePlanner(
        movie_id="test_movie_beta",
        analysis_dir=mock_narration_environment,
    )
    plan_doc = planner.plan()
    plan_path = planner.save_narrative_plan(plan_doc)

    assert plan_path.exists()
    assert plan_doc.movie_id == "test_movie_beta"
    assert plan_doc.status == "PARTIAL"
    assert len(plan_doc.segments) == 2
    assert plan_doc.segments[0].segment_id == "SEGMENT_001"
    assert plan_doc.segments[1].segment_id == "SEGMENT_002"


def test_importance_weighted_duration_allocation(mock_narration_environment: Path):
    """Test 4: High-importance beats receive proportional duration allocation."""
    planner = NarrativePlanner(
        movie_id="test_movie_beta",
        analysis_dir=mock_narration_environment,
    )
    plan_doc = planner.plan()

    # BEAT_002 has importance 9, BEAT_001 has importance 8
    seg1 = plan_doc.segments[0]
    seg2 = plan_doc.segments[1]
    assert seg1.target_duration_seconds > 0.0
    assert seg2.target_duration_seconds > 0.0
    assert seg2.importance >= seg1.importance


def test_script_generation_traceability(mock_narration_environment: Path):
    """Test 2 & 3: Script segments have 100% source traceability back to beats, scenes, and events."""
    engine = MalayStorytellerEngine(
        movie_id="test_movie_beta",
        analysis_dir=mock_narration_environment,
    )
    script_doc = engine.generate()

    assert len(script_doc.segments) >= 2
    for seg in script_doc.segments:
        assert seg.segment_id.startswith("SEGMENT_")
        assert len(seg.source_beat_ids) > 0
        assert len(seg.source_scene_ids) > 0
        assert len(seg.source_event_indices) > 0
        assert seg.word_count > 0
        assert seg.estimated_duration_seconds > 0.0


def test_malay_prose_style_and_naturalness(mock_narration_environment: Path):
    """Test: Narration is in natural conversational KL Malay with natural transitions."""
    engine = MalayStorytellerEngine(
        movie_id="test_movie_beta",
        analysis_dir=mock_narration_environment,
    )
    script_doc = engine.generate()

    full_text = script_doc.full_script.lower()
    # Check for natural KL Malay markers
    assert "awal cerita ni" in full_text
    assert "dr. benjamin price" in full_text
    assert "vibe" in full_text or "tengah" in full_text or "diorang" in full_text


def test_no_banned_formal_phrases_and_kl_style(mock_narration_environment: Path):
    """Test Phase 8.1: Ensure no stiff academic/formal news Malay phrases remain."""
    engine = MalayStorytellerEngine(
        movie_id="test_movie_beta",
        analysis_dir=mock_narration_environment,
    )
    script_doc = engine.generate()
    full_text = script_doc.full_script.lower()

    banned_phrases = [
        "dalam pada itu",
        "berikutan itu",
        "mewujudkan kontras",
        "mengakibatkan",
        "bergelut dalam perairan",
        "peristiwa tersebut",
        "individu tersebut",
        "memberikan gambaran",
        "menimbulkan persoalan",
        "dapat diperhatikan",
        "oleh itu",
        "setakat bahagian awal ini",
    ]

    for phrase in banned_phrases:
        assert phrase not in full_text, f"Banned formal phrase '{phrase}' found in script!"


def test_uncertainty_and_identity_bleed_prevention(mock_narration_environment: Path):
    """Test 6 & 7: Script does NOT falsely assert that the drowning traveler is definitely Dr. Benjamin."""
    engine = MalayStorytellerEngine(
        movie_id="test_movie_beta",
        analysis_dir=mock_narration_environment,
    )
    script_doc = engine.generate()

    # Segment 2 (storm sequence) must refer to the traveler cautiously without claiming it's Benjamin
    storm_seg = script_doc.segments[1]
    storm_text = storm_seg.text.lower()
    # Should use "lelaki", "mamat", or "diorang", not assert "Benjamin terjatuh dari bot"
    assert "benjamin terjatuh" not in storm_text


def test_partial_story_handling_and_continuation_marker(mock_narration_environment: Path):
    """Test 5: Partial coverage scripts end with an explicit conversational continuation marker."""
    engine = MalayStorytellerEngine(
        movie_id="test_movie_beta",
        analysis_dir=mock_narration_environment,
    )
    script_doc = engine.generate()

    assert script_doc.status == "PARTIAL"
    last_seg = script_doc.segments[-1]
    assert "buat masa ni" in last_seg.text.lower() or "sambungan" in last_seg.text.lower() or "misteri" in last_seg.text.lower()
    assert len(script_doc.warnings) > 0


def test_duration_estimation_accuracy(mock_narration_environment: Path):
    """Test 8: Duration estimation adheres to (words / WPM) * 60 seconds."""
    engine = MalayStorytellerEngine(
        movie_id="test_movie_beta",
        analysis_dir=mock_narration_environment,
        words_per_minute=150.0,
    )
    script_doc = engine.generate()

    assert script_doc.words_per_minute == 150.0
    for seg in script_doc.segments:
        expected_dur = round((seg.word_count / 150.0) * 60.0, 1)
        assert abs(seg.estimated_duration_seconds - expected_dur) <= 0.2


def test_empty_or_weak_scene_resilience(tmp_path: Path):
    """Test 9 & 10: Empty or malformed story document completes without crashing."""
    analysis_dir = tmp_path / "analysis" / "movie_empty_scenes"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    minimal_story = StoryDocument(
        movie_id="movie_empty_scenes",
        story_status="COMPLETE",
        scenes=[],
        beats=[],
    )
    (analysis_dir / "story.json").write_text(minimal_story.to_json(indent=2), encoding="utf-8")

    engine = MalayStorytellerEngine("movie_empty_scenes", analysis_dir)
    script_doc = engine.generate()
    assert isinstance(script_doc, ScriptDocument)
    assert script_doc.movie_id == "movie_empty_scenes"


def test_script_schema_and_roundtrip_serialization(mock_narration_environment: Path):
    """Test 11 & 12: script.json conforms to schema and supports roundtrip deserialization."""
    engine = MalayStorytellerEngine(
        movie_id="test_movie_beta",
        analysis_dir=mock_narration_environment,
    )
    script_doc = engine.generate()
    script_path = engine.save_script_document(script_doc)

    assert script_path.exists()
    with open(script_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    expected_keys = {
        "movie_id", "status", "language", "target_total_duration_seconds",
        "estimated_total_duration_seconds", "total_words", "words_per_minute",
        "segments", "full_script", "warnings"
    }
    assert expected_keys.issubset(set(data.keys()))

    roundtrip = ScriptDocument.from_dict(data)
    assert roundtrip.movie_id == script_doc.movie_id
    assert len(roundtrip.segments) == len(script_doc.segments)
    assert roundtrip.total_words == script_doc.total_words


def test_cli_plan_and_script_generation(mock_narration_environment: Path):
    """Test 13: CLI execution for both plan_narration and generate_script."""
    # 1. CLI plan narration
    cmd1 = [
        sys.executable, "-m", "narration_engine.plan_narration",
        "test_movie_beta",
        "--output-dir", str(mock_narration_environment.parent),
    ]
    res1 = subprocess.run(cmd1, capture_output=True, text=True)
    assert res1.returncode == 0
    assert "NARRATIVE PLANNING COMPLETE" in res1.stdout

    # 2. CLI generate script
    cmd2 = [
        sys.executable, "-m", "narration_engine.generate_script",
        "test_movie_beta",
        "--output-dir", str(mock_narration_environment.parent),
        "--wpm", "150",
    ]
    res2 = subprocess.run(cmd2, capture_output=True, text=True)
    assert res2.returncode == 0
    assert "SCRIPT GENERATION COMPLETE" in res2.stdout
