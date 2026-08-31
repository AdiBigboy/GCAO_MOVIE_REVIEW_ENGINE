"""
Automated Test Suite for Phase 9: Source Clip Selection Engine
Tests maximum 3.0s duration enforcement, boundary safety, anti-contiguous constraints,
source traceability, importance budgeting, partial story bounds, schemas, validator, and CLI.
"""

import json
import subprocess
import sys
from pathlib import Path
import pytest

from clip_engine.models import (
    ClipPlanDocument,
    SegmentClips,
    SourceClip,
)
from clip_engine.validate_clips import (
    ClipPlanValidator,
    ValidationResult,
)
from clip_engine.select_clips import (
    SourceClipSelector,
    ClipSelectionError,
    generate_movie_clip_plan,
)
from narration_engine.models import (
    ScriptDocument,
    ScriptSegment,
)
from story_engine.models import (
    SceneRecord,
    StoryBeat,
    StoryDocument,
)


@pytest.fixture
def mock_clip_environment(tmp_path: Path) -> Path:
    """Create a structured test movie analysis environment with full script and event evidence."""
    analysis_dir = tmp_path / "analysis" / "test_movie_gamma"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # 1. movie_metadata.json
    metadata = {
        "movie_id": "test_movie_gamma",
        "duration_seconds": 600.0,
        "resolution": "1920x1080",
    }
    (analysis_dir / "movie_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # 2. timeline_index.json
    timeline = {
        "movie_id": "test_movie_gamma",
        "sampling_interval_seconds": 30.0,
        "duration_seconds": 600.0,
        "total_samples": 20,
    }
    (analysis_dir / "timeline_index.json").write_text(json.dumps(timeline, indent=2), encoding="utf-8")

    # 3. events.json
    events_data = {
        "movie_id": "test_movie_gamma",
        "total_events": 6,
        "events": [
            {
                "index": 1,
                "timestamp_seconds": 0.0,
                "event_summary": "Opening black screen.",
                "visual": {"location": "Black screen", "action": "Blank screen", "character_labels": []},
                "plot_significance": 1,
            },
            {
                "index": 2,
                "timestamp_seconds": 30.0,
                "event_summary": "Abstract red titles distorting on screen.",
                "visual": {"location": "Title Sequence", "action": "Red lettering", "character_labels": []},
                "plot_significance": 2,
            },
            {
                "index": 3,
                "timestamp_seconds": 60.0,
                "event_summary": "Dr. Benjamin adjusts an audio recorder.",
                "visual": {"location": "Study room", "action": "Operating tape recorder", "character_labels": ["Dr. Benjamin"]},
                "plot_significance": 8,
            },
            {
                "index": 4,
                "timestamp_seconds": 90.0,
                "event_summary": "A stained page with ancient scribbles is revealed.",
                "visual": {"location": "Study room", "action": "Unfolding ancient document", "character_labels": ["Dr. Benjamin"]},
                "plot_significance": 9,
            },
            {
                "index": 5,
                "timestamp_seconds": 150.0,
                "event_summary": "A motorboat flees across choppy water during heavy rain.",
                "visual": {"location": "Stormy lake", "action": "Driving motorboat in storm", "character_labels": ["PERSON_A"]},
                "plot_significance": 9,
            },
            {
                "index": 6,
                "timestamp_seconds": 210.0,
                "event_summary": "A traveler washes ashore face down on the muddy bank.",
                "visual": {"location": "Lakeshore / forest", "action": "Survivor lying in mud", "character_labels": ["PERSON_A"]},
                "plot_significance": 8,
            },
        ],
    }
    (analysis_dir / "events.json").write_text(json.dumps(events_data, indent=2), encoding="utf-8")

    # 4. story.json
    story_doc = StoryDocument(
        movie_id="test_movie_gamma",
        story_status="PARTIAL",
        scenes=[
            SceneRecord(
                scene_id="SCENE_001",
                start_seconds=0.0,
                end_seconds=60.0,
                event_indices=[1, 2],
                characters=[],
                location="Title Sequence",
                summary="Opening titles.",
                scene_function="SETUP",
                importance=2,
            ),
            SceneRecord(
                scene_id="SCENE_002",
                start_seconds=60.0,
                end_seconds=120.0,
                event_indices=[3, 4],
                characters=["Dr. Benjamin"],
                location="Study room",
                summary="Dr. Benjamin discovers ancient writings.",
                scene_function="DISCOVERY",
                importance=8,
            ),
            SceneRecord(
                scene_id="SCENE_003",
                start_seconds=120.0,
                end_seconds=240.0,
                event_indices=[5, 6],
                characters=["PERSON_A"],
                location="Stormy lake",
                summary="Boat flight and washing ashore.",
                scene_function="ESCALATION",
                importance=9,
            ),
        ],
        beats=[],
    )
    (analysis_dir / "story.json").write_text(story_doc.to_json(indent=2), encoding="utf-8")

    # 5. script.json
    script_doc = ScriptDocument(
        movie_id="test_movie_gamma",
        status="PARTIAL",
        language="ms-MY",
        target_total_duration_seconds=60.0,
        estimated_total_duration_seconds=55.0,
        total_words=140,
        words_per_minute=150.0,
        segments=[
            ScriptSegment(
                segment_id="SEGMENT_001",
                text="Awal cerita ni kita nampak Dr. Benjamin sedang mendengar pita rakaman kuno.",
                estimated_duration_seconds=25.0,
                word_count=65,
                source_beat_ids=["BEAT_001"],
                source_scene_ids=["SCENE_002"],
                source_event_indices=[3, 4],
                importance=8,
            ),
            ScriptSegment(
                segment_id="SEGMENT_002",
                text="Kat sinilah keadaan bertukar cemas bila mereka terpaksa meredah ribut dengan bot sebelum terdampar di tebing hutan.",
                estimated_duration_seconds=30.0,
                word_count=75,
                source_beat_ids=["BEAT_002"],
                source_scene_ids=["SCENE_003"],
                source_event_indices=[5, 6],
                importance=9,
            ),
            ScriptSegment(
                segment_id="SEGMENT_003",
                text="Setakat bahagian awal ni, cerita masih tergantung dan misteri baru sahaja bermula.",
                estimated_duration_seconds=15.0,
                word_count=35,
                source_beat_ids=["BEAT_002"],
                source_scene_ids=["SCENE_003"],
                source_event_indices=[5, 6],
                importance=3,
            ),
        ],
        full_script="Sample script text",
    )
    (analysis_dir / "script.json").write_text(script_doc.to_json(indent=2), encoding="utf-8")

    return analysis_dir


# ==============================================================================
# Unit & Functional Tests
# ==============================================================================

def test_maximum_clip_duration_enforcement(mock_clip_environment: Path):
    """Test 1: Every source clip has duration <= 3.0 seconds."""
    selector = SourceClipSelector(
        movie_id="test_movie_gamma",
        analysis_dir=mock_clip_environment,
        max_clip_duration_seconds=3.0,
    )
    plan_doc = selector.select()

    assert plan_doc.total_clips > 0
    for seg in plan_doc.segments:
        for clip in seg.clips:
            assert clip.duration_seconds <= 3.0 + 0.01, f"Clip {clip.clip_id} exceeded 3.0s: {clip.duration_seconds}"
            assert clip.duration_seconds > 0.0


def test_clip_boundary_safety(mock_clip_environment: Path):
    """Test 2 & 3: Start timestamps are non-negative and end timestamps do not overflow movie duration."""
    selector = SourceClipSelector(
        movie_id="test_movie_gamma",
        analysis_dir=mock_clip_environment,
    )
    plan_doc = selector.select()

    for seg in plan_doc.segments:
        for clip in seg.clips:
            assert clip.source_start_seconds >= 0.0
            assert clip.source_end_seconds > clip.source_start_seconds
            assert clip.source_end_seconds <= 600.0  # Movie duration


def test_chronological_ordering_and_traceability(mock_clip_environment: Path):
    """Test 4, 5, 6: Clips maintain chronological order and 100% source traceability."""
    selector = SourceClipSelector(
        movie_id="test_movie_gamma",
        analysis_dir=mock_clip_environment,
    )
    plan_doc = selector.select()

    all_clips = [c for s in plan_doc.segments for c in s.clips]
    assert len(all_clips) >= 4

    for i in range(len(all_clips) - 1):
        assert all_clips[i].source_start_seconds <= all_clips[i + 1].source_start_seconds

    for clip in all_clips:
        assert clip.source_event_index in [3, 4, 5, 6]
        assert clip.source_scene_id.startswith("SCENE_")
        assert len(clip.visual_reason) > 0


def test_anti_contiguous_scene_reconstruction(mock_clip_environment: Path):
    """Test 7: Prevents constructing contiguous movie scenes without gaps."""
    selector = SourceClipSelector(
        movie_id="test_movie_gamma",
        analysis_dir=mock_clip_environment,
    )
    plan_doc = selector.select()
    all_clips = [c for s in plan_doc.segments for c in s.clips]

    for i in range(len(all_clips) - 1):
        c1 = all_clips[i]
        c2 = all_clips[i + 1]
        gap = c2.source_start_seconds - c1.source_end_seconds
        if gap < 0.5:
            assert c1.continuity_exception or c2.continuity_exception


def test_importance_weighted_clip_budgeting(mock_clip_environment: Path):
    """Test 9 & 10: High-importance narration segments get more supporting clips than low-value ones."""
    selector = SourceClipSelector(
        movie_id="test_movie_gamma",
        analysis_dir=mock_clip_environment,
    )
    plan_doc = selector.select()

    # SEGMENT_001 (imp: 8) and SEGMENT_002 (imp: 9) should have 2 clips each
    # SEGMENT_003 (continuation marker, imp: 3) should have 0 clips
    seg1 = plan_doc.segments[0]
    seg2 = plan_doc.segments[1]
    seg3 = plan_doc.segments[2]

    assert len(seg1.clips) == 2
    assert len(seg2.clips) == 2
    assert len(seg3.clips) == 0


def test_partial_story_timeline_containment(mock_clip_environment: Path):
    """Test 11: Clips are selected only from the analyzed timeline region."""
    selector = SourceClipSelector(
        movie_id="test_movie_gamma",
        analysis_dir=mock_clip_environment,
    )
    plan_doc = selector.select()

    assert plan_doc.status == "PARTIAL"
    for seg in plan_doc.segments:
        for clip in seg.clips:
            # Events in mock gamma only go up to 210s
            assert clip.source_start_seconds <= 240.0


def test_validator_detects_violations():
    """Test 13: ClipPlanValidator catches duration overflows, overlaps, and invalid timestamps."""
    validator = ClipPlanValidator(max_clip_duration_seconds=3.0, movie_duration_seconds=100.0)

    # 1. Test clip exceeding 3.0s
    bad_plan = ClipPlanDocument(
        movie_id="test_bad",
        status="COMPLETE",
        segments=[
            SegmentClips(
                segment_id="SEGMENT_001",
                narration_text="Test",
                source_event_indices=[1],
                clips=[
                    SourceClip(
                        clip_id="CLIP_001",
                        source_start_seconds=10.0,
                        source_end_seconds=15.0,  # 5.0s > 3.0s!
                        duration_seconds=5.0,
                        source_event_index=1,
                        source_scene_id="SCENE_001",
                        visual_reason="Test reason",
                    )
                ],
            )
        ],
    )
    res = validator.validate(bad_plan)
    assert not res.valid
    assert any("exceeds maximum editorial duration" in err for err in res.errors)

    # 2. Test negative timestamp
    bad_plan_neg = ClipPlanDocument(
        movie_id="test_bad_neg",
        status="COMPLETE",
        segments=[
            SegmentClips(
                segment_id="SEGMENT_001",
                narration_text="Test",
                source_event_indices=[1],
                clips=[
                    SourceClip(
                        clip_id="CLIP_001",
                        source_start_seconds=-2.0,
                        source_end_seconds=1.0,
                        duration_seconds=3.0,
                        source_event_index=1,
                        source_scene_id="SCENE_001",
                        visual_reason="Test reason",
                    )
                ],
            )
        ],
    )
    res_neg = validator.validate(bad_plan_neg)
    assert not res_neg.valid
    assert any("negative start timestamp" in err for err in res_neg.errors)


def test_schema_and_roundtrip_serialization(mock_clip_environment: Path):
    """Test 12 & 15: clip_plan.json conforms to schema and supports roundtrip deserialization."""
    selector = SourceClipSelector(
        movie_id="test_movie_gamma",
        analysis_dir=mock_clip_environment,
    )
    plan_doc = selector.select()
    out_path = selector.save_clip_plan(plan_doc)

    assert out_path.exists()
    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    expected_keys = {
        "movie_id", "status", "max_clip_duration_seconds", "total_clips",
        "total_source_duration_seconds", "segments", "created_at", "warnings"
    }
    assert expected_keys.issubset(set(data.keys()))

    roundtrip = ClipPlanDocument.from_dict(data)
    assert roundtrip.movie_id == plan_doc.movie_id
    assert roundtrip.total_clips == plan_doc.total_clips
    assert roundtrip.total_source_duration_seconds == plan_doc.total_source_duration_seconds


def test_cli_clip_selection(mock_clip_environment: Path):
    """Test 14: CLI execution for clip_engine.select_clips."""
    cmd = [
        sys.executable, "-m", "clip_engine.select_clips",
        "test_movie_gamma",
        "--output-dir", str(mock_clip_environment.parent),
        "--max-duration", "3.0",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "SOURCE CLIP SELECTION COMPLETE" in res.stdout
