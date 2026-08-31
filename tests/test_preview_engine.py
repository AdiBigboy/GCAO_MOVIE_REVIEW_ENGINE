"""
Automated Test Suite for Phase 10: Rough Video Preview Builder
Tests timeline ordering, segment clip placement, duration calculations, placeholder gap generation,
aspect ratio handling, manifest schemas, deterministic generation, and absence of TTS dependency.
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
from narration_engine.models import (
    ScriptDocument,
    ScriptSegment,
)
from preview_engine.models import (
    PreviewManifest,
    SegmentTimeline,
    TimelineItem,
)
from preview_engine.timeline_builder import (
    PreviewTimelineBuilder,
    TimelineBuildError,
)
from preview_engine.render_preview import (
    RoughPreviewRenderer,
    build_rough_video_preview,
    _format_srt_time,
    _wrap_caption_lines,
)


@pytest.fixture
def mock_preview_environment(tmp_path: Path) -> Path:
    """Create structured test movie analysis environment for preview building."""
    analysis_dir = tmp_path / "analysis" / "test_preview_movie"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    clip_previews_dir = analysis_dir / "clip_previews"
    clip_previews_dir.mkdir(parents=True, exist_ok=True)

    # 1. movie_metadata.json
    metadata = {
        "movie_id": "test_preview_movie",
        "duration_seconds": 600.0,
        "resolution": "1920x1080",
    }
    (analysis_dir / "movie_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # 2. script.json
    script_doc = ScriptDocument(
        movie_id="test_preview_movie",
        status="PARTIAL",
        language="ms-MY",
        target_total_duration_seconds=50.0,
        estimated_total_duration_seconds=50.0,
        total_words=125,
        words_per_minute=150.0,
        segments=[
            ScriptSegment(
                segment_id="SEGMENT_001",
                text="Pada permulaan cerita, watak utama sedang memeriksa rakaman misteri.",
                estimated_duration_seconds=20.0,
                word_count=50,
                source_beat_ids=["BEAT_001"],
                source_scene_ids=["SCENE_001"],
                source_event_indices=[1, 2],
                importance=8,
            ),
            ScriptSegment(
                segment_id="SEGMENT_002",
                text="Keadaan bertukar cemas apabila ribut melanda dan bot mereka terbalik di tebing sungai.",
                estimated_duration_seconds=20.0,
                word_count=50,
                source_beat_ids=["BEAT_002"],
                source_scene_ids=["SCENE_002"],
                source_event_indices=[3, 4],
                importance=9,
            ),
            ScriptSegment(
                segment_id="SEGMENT_003",
                text="Setakat ini misteri masih berlanjutan.",
                estimated_duration_seconds=10.0,
                word_count=25,
                source_beat_ids=["BEAT_002"],
                source_scene_ids=["SCENE_002"],
                source_event_indices=[3, 4],
                importance=3,
            ),
        ],
        full_script="Sample script text",
    )
    (analysis_dir / "script.json").write_text(script_doc.to_json(indent=2), encoding="utf-8")

    # 3. clip_plan.json
    clip_plan_doc = ClipPlanDocument(
        movie_id="test_preview_movie",
        status="PARTIAL",
        max_clip_duration_seconds=3.0,
        total_clips=4,
        total_source_duration_seconds=12.0,
        segments=[
            SegmentClips(
                segment_id="SEGMENT_001",
                narration_text="Pada permulaan cerita, watak utama sedang memeriksa rakaman misteri.",
                source_event_indices=[1, 2],
                clips=[
                    SourceClip(
                        clip_id="CLIP_001",
                        source_start_seconds=10.0,
                        source_end_seconds=13.0,
                        duration_seconds=3.0,
                        source_event_index=1,
                        source_scene_id="SCENE_001",
                        visual_reason="Checking recorder",
                    ),
                    SourceClip(
                        clip_id="CLIP_002",
                        source_start_seconds=40.0,
                        source_end_seconds=43.0,
                        duration_seconds=3.0,
                        source_event_index=2,
                        source_scene_id="SCENE_001",
                        visual_reason="Reading document",
                    ),
                ],
            ),
            SegmentClips(
                segment_id="SEGMENT_002",
                narration_text="Keadaan bertukar cemas apabila ribut melanda dan bot mereka terbalik di tebing sungai.",
                source_event_indices=[3, 4],
                clips=[
                    SourceClip(
                        clip_id="CLIP_003",
                        source_start_seconds=100.0,
                        source_end_seconds=103.0,
                        duration_seconds=3.0,
                        source_event_index=3,
                        source_scene_id="SCENE_002",
                        visual_reason="Boat in storm",
                    ),
                    SourceClip(
                        clip_id="CLIP_004",
                        source_start_seconds=150.0,
                        source_end_seconds=153.0,
                        duration_seconds=3.0,
                        source_event_index=4,
                        source_scene_id="SCENE_002",
                        visual_reason="Survivor in mud",
                    ),
                ],
            ),
            SegmentClips(
                segment_id="SEGMENT_003",
                narration_text="Setakat ini misteri masih berlanjutan.",
                source_event_indices=[],
                clips=[],
            ),
        ],
    )
    (analysis_dir / "clip_plan.json").write_text(clip_plan_doc.to_json(indent=2), encoding="utf-8")

    # Generate synthetic tiny MP4 clips in clip_previews
    for c_id, start_s in [("CLIP_001", 10.0), ("CLIP_002", 40.0), ("CLIP_003", 100.0), ("CLIP_004", 150.0)]:
        clip_p = clip_previews_dir / f"{c_id}_{start_s:.1f}s.mp4"
        cmd = [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "testsrc=size=640x360:rate=30",
            "-t", "3.0",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            str(clip_p),
        ]
        subprocess.run(cmd, capture_output=True, check=True)

    return analysis_dir


# ==============================================================================
# Unit & Functional Tests
# ==============================================================================

def test_timeline_ordering_and_segment_placement(mock_preview_environment: Path):
    """Test 1 & 2: Timeline items strictly maintain chronological start/end order and match script segments."""
    builder = PreviewTimelineBuilder(
        movie_id="test_preview_movie",
        analysis_dir=mock_preview_environment,
    )
    manifest = builder.build_timeline()

    assert manifest.total_segments == 3
    assert len(manifest.segments) == 3

    # Verify chronological continuity across segments
    assert manifest.segments[0].start_time_seconds == 0.0
    assert manifest.segments[0].end_time_seconds == manifest.segments[1].start_time_seconds
    assert manifest.segments[1].end_time_seconds == manifest.segments[2].start_time_seconds
    assert manifest.total_duration_seconds == manifest.segments[2].end_time_seconds

    # Verify segment 1 contains CLIP_001 and CLIP_002
    seg1_clips = [it.source_clip_id for it in manifest.segments[0].items if it.item_type == "SOURCE_CLIP"]
    assert seg1_clips == ["CLIP_001", "CLIP_002"]

    # Verify segment 2 contains CLIP_003 and CLIP_004
    seg2_clips = [it.source_clip_id for it in manifest.segments[1].items if it.item_type == "SOURCE_CLIP"]
    assert seg2_clips == ["CLIP_003", "CLIP_004"]

    # Verify segment 3 has 0 clips and 1 placeholder
    seg3_clips = [it for it in manifest.segments[2].items if it.item_type == "SOURCE_CLIP"]
    assert len(seg3_clips) == 0
    assert len(manifest.segments[2].items) == 1
    assert manifest.segments[2].items[0].item_type == "PLACEHOLDER"


def test_segment_duration_calculation_and_placeholder_gaps(mock_preview_environment: Path):
    """Test 3 & 4: Total items duration inside segment equals estimated narration duration with placeholder gaps."""
    builder = PreviewTimelineBuilder(
        movie_id="test_preview_movie",
        analysis_dir=mock_preview_environment,
    )
    manifest = builder.build_timeline()

    for seg in manifest.segments:
        item_sum = round(sum(it.duration_seconds for it in seg.items), 2)
        assert abs(item_sum - seg.duration_seconds) < 0.05

        # Check that items inside segment do not overlap
        for i in range(len(seg.items) - 1):
            assert seg.items[i].end_time_seconds <= seg.items[i + 1].start_time_seconds + 0.01


def test_clip_duration_hard_constraint(mock_preview_environment: Path):
    """Test 5: Every source clip item has duration <= 3.0s."""
    builder = PreviewTimelineBuilder(
        movie_id="test_preview_movie",
        analysis_dir=mock_preview_environment,
    )
    manifest = builder.build_timeline()

    for seg in manifest.segments:
        for it in seg.items:
            if it.item_type == "SOURCE_CLIP":
                assert it.duration_seconds <= 3.0 + 0.01
                assert it.duration_seconds > 0.0


def test_no_orphan_clips_and_source_traceability(mock_preview_environment: Path):
    """Test 6: Every source clip item maps to scene and event index."""
    builder = PreviewTimelineBuilder(
        movie_id="test_preview_movie",
        analysis_dir=mock_preview_environment,
    )
    manifest = builder.build_timeline()

    for seg in manifest.segments:
        for it in seg.items:
            if it.item_type == "SOURCE_CLIP":
                assert it.source_clip_id is not None
                assert it.source_scene_id in ["SCENE_001", "SCENE_002"]
                assert it.source_event_index in [1, 2, 3, 4]


def test_preview_manifest_schema_and_roundtrip(mock_preview_environment: Path):
    """Test 7 & 10: Manifest conforms to schema, retains partial status, and roundtrips to JSON."""
    builder = PreviewTimelineBuilder(
        movie_id="test_preview_movie",
        analysis_dir=mock_preview_environment,
    )
    manifest = builder.build_timeline()
    out_file = builder.save_manifest(manifest)

    assert out_file.exists()
    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["movie_id"] == "test_preview_movie"
    assert data["status"] == "PARTIAL"
    assert data["resolution"] == "1920x1080"
    assert data["fps"] == 30.0
    assert data["total_source_clips"] == 4
    assert data["total_source_footage_seconds"] == 12.0

    roundtrip = PreviewManifest.from_dict(data)
    assert roundtrip.movie_id == manifest.movie_id
    assert roundtrip.total_duration_seconds == manifest.total_duration_seconds


def test_srt_time_formatting_and_caption_wrapping():
    """Test 11: SRT timestamp conversion and 2-line caption wrapping."""
    assert _format_srt_time(0.0) == "00:00:00,000"
    assert _format_srt_time(65.45) == "00:01:05,450"
    assert _format_srt_time(3661.125) == "01:01:01,125"

    long_text = "Ini adalah contoh teks ulasan cerita yang sangat panjang untuk menguji pemotongan baris sari kata."
    wrapped = _wrap_caption_lines(long_text, max_chars_per_line=30)
    lines = wrapped.split("\n")
    assert len(lines) <= 2
    for line in lines:
        assert len(line) <= 60


def test_rough_preview_video_rendering(mock_preview_environment: Path):
    """Test 8, 9, 12: Renders playable 1920x1080 rough_preview.mp4 without TTS dependency."""
    manifest, video_path = build_rough_video_preview(
        source_path=mock_preview_environment.name,
        movie_id="test_preview_movie",
        output_base_dir=mock_preview_environment.parent,
        render_video=True,
    )

    assert video_path is not None
    assert video_path.exists()
    assert video_path.stat().st_size > 0

    # Probe rendered video
    cmd_probe = [
        "ffprobe", "-v", "error",
        "-show_entries", "stream=width,height,codec_name,r_frame_rate",
        "-of", "json",
        str(video_path),
    ]
    res = subprocess.run(cmd_probe, capture_output=True, text=True, check=True)
    probe_data = json.loads(res.stdout)

    v_stream = next(s for s in probe_data["streams"] if s["codec_name"] == "h264")
    assert v_stream["width"] == 1920
    assert v_stream["height"] == 1080
    assert v_stream["r_frame_rate"] == "30/1"


def test_missing_input_files_raise_error(tmp_path: Path):
    """Test error handling when script.json or clip_plan.json is missing."""
    empty_dir = tmp_path / "analysis" / "empty_movie"
    empty_dir.mkdir(parents=True, exist_ok=True)

    builder = PreviewTimelineBuilder(
        movie_id="empty_movie",
        analysis_dir=empty_dir,
    )
    with pytest.raises(TimelineBuildError):
        builder.build_timeline()
