"""
Automated Test Suite for Phase 3: Subtitle & Dialogue Extraction
"""

import json
import subprocess
import sys
import time
from pathlib import Path
import pytest

from movie_analyzer.extract_dialogue import (
    DialogueDocument,
    DialogueEntry,
    extract_movie_dialogue,
    select_embedded_subtitle_stream,
    SubtitleFileNotFoundError,
)
from movie_analyzer.subtitles import (
    parse_srt_content,
    parse_vtt_content,
    parse_ass_content,
    clean_subtitle_text,
    parse_subtitle_file,
    detect_subtitle_language_from_filename,
)
from movie_analyzer.ingest import SourceFileNotFoundError


@pytest.fixture(scope="session")
def multi_sub_mkv(tmp_path_factory) -> Path:
    """Generate a synthetic MKV movie with multiple embedded subtitle streams."""
    temp_dir = tmp_path_factory.mktemp("media_subs")
    video_file = temp_dir / "Multi Sub Synthetic Movie (2026).mkv"

    # Create English SRT
    en_srt = temp_dir / "en.srt"
    en_srt.write_text(
        "1\n00:00:01,000 --> 00:00:02,500\nHello <b>world</b>!\n\n"
        "2\n00:00:03,000 --> 00:00:05,250\nEnglish embedded <i>dialogue</i>.\n",
        encoding="utf-8"
    )

    # Create Malay SRT
    ms_srt = temp_dir / "ms.srt"
    ms_srt.write_text(
        "1\n00:00:01,000 --> 00:00:02,500\nHelo <b>dunia</b>!\n\n"
        "2\n00:00:03,000 --> 00:00:05,250\nDialog bahasa Melayu.\n",
        encoding="utf-8"
    )

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=6:size=640x360:rate=25",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
        "-i", str(en_srt),
        "-i", str(ms_srt),
        "-c:v", "libx264", "-c:a", "aac",
        "-c:s", "srt",
        "-map", "0:v:0", "-map", "1:a:0",
        "-map", "2:s:0", "-map", "3:s:0",
        "-metadata:s:s:0", "language=eng",
        "-metadata:s:s:0", "title=English SDH",
        "-disposition:s:0", "default",
        "-metadata:s:s:1", "language=msa",
        "-metadata:s:s:1", "title=Bahasa Melayu",
        str(video_file),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"FFmpeg failed: {res.stderr}"
    return video_file


@pytest.fixture(scope="session")
def no_sub_mp4(tmp_path_factory) -> Path:
    """Generate a synthetic video with no subtitle streams."""
    temp_dir = tmp_path_factory.mktemp("media_no_sub")
    video_file = temp_dir / "No Subtitles Movie.mp4"

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=3:size=320x240:rate=25",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-c:v", "libx264", "-c:a", "aac",
        str(video_file),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    return video_file


# ==============================================================================
# Unit Tests for Subtitle Parsers & Cleaning
# ==============================================================================

def test_clean_subtitle_text():
    """Test cleaning of various subtitle formatting tags."""
    assert clean_subtitle_text("<b>Bold</b> and <i>Italic</i>") == "Bold and Italic"
    assert clean_subtitle_text("<font color=\"#ff0000\">Red text</font>") == "Red text"
    assert clean_subtitle_text("{\\an8\\b1}ASS Positional Text{\\b0}") == "ASS Positional Text"
    assert clean_subtitle_text("Line 1\\NLine 2\\nLine 3") == "Line 1 Line 2 Line 3"
    assert clean_subtitle_text("<c.yellow>Color Tag</c>") == "Color Tag"
    assert clean_subtitle_text("<v Narrator>Spoken Words</v>") == "Spoken Words"
    assert clean_subtitle_text("   Lots   of   spaces   \n\n") == "Lots of spaces"


def test_parse_srt_content():
    """Test standard SRT parsing with millisecond accuracy."""
    content = """
1
00:01:23,456 --> 00:01:25,789
Hello detective.

2
00:01:26,000 --> 00:01:28,500
Did you find the <b>clue</b>?
"""
    entries = parse_srt_content(content)
    assert len(entries) == 2
    assert entries[0].index == 1
    assert entries[0].start_seconds == 83.456
    assert entries[0].end_seconds == 85.789
    assert entries[0].duration_seconds == 2.333
    assert entries[0].text == "Hello detective."
    assert entries[0].start == "00:01:23.456"

    assert entries[1].index == 2
    assert entries[1].text == "Did you find the clue?"


def test_parse_vtt_content():
    """Test WebVTT parsing with cue settings and speaker extraction."""
    content = """WEBVTT

NOTE This is a comment

1
00:00:10.500 --> 00:00:13.200 position:10% align:left
<v Agent Smith>We have him cornered.</v>

00:00:14.000 --> 00:00:16.800
<c.green>Understood.</c>
"""
    entries = parse_vtt_content(content)
    assert len(entries) == 2
    assert entries[0].start_seconds == 10.5
    assert entries[0].end_seconds == 13.2
    assert entries[0].text == "We have him cornered."
    assert entries[0].speaker == "Agent Smith"

    assert entries[1].start_seconds == 14.0
    assert entries[1].text == "Understood."


def test_parse_ass_content():
    """Test ASS / SSA parsing with events table and style strip."""
    content = """[Script Info]
Title: Sample ASS
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:05.10,0:00:08.50,Default,Commander,0,0,0,,{\\pos(192,200)}Hold your fire!\\NWait for my command.
Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Look out!
"""
    entries = parse_ass_content(content)
    assert len(entries) == 2
    # Chronological sort check: 1.0s line should come before 5.1s line
    assert entries[0].start_seconds == 1.0
    assert entries[0].text == "Look out!"
    assert entries[1].start_seconds == 5.1
    assert entries[1].end_seconds == 8.5
    assert entries[1].text == "Hold your fire! Wait for my command."
    assert entries[1].speaker == "Commander"


def test_detect_subtitle_language_from_filename():
    """Test extracting language code from filename."""
    assert detect_subtitle_language_from_filename(Path("movie.en.srt")) == "en"
    assert detect_subtitle_language_from_filename(Path("Movie.eng.vtt")) == "eng"
    assert detect_subtitle_language_from_filename(Path("Movie.ms.ass")) == "ms"
    assert detect_subtitle_language_from_filename(Path("Movie.srt")) == "und"


# ==============================================================================
# Pipeline & Integration Tests
# ==============================================================================

def test_extract_embedded_subtitles_default(multi_sub_mkv: Path, tmp_path: Path):
    """Test extracting default embedded subtitle (English)."""
    out_dir = tmp_path / "analysis_embedded"
    doc = extract_movie_dialogue(
        source_path=multi_sub_mkv,
        output_base_dir=out_dir,
    )

    assert doc.status == "SUCCESS"
    assert doc.source_type == "embedded_subtitle"
    assert doc.language == "eng"
    assert doc.total_entries == 2
    assert doc.entries[0].text == "Hello world!"
    assert doc.entries[1].text == "English embedded dialogue."

    # Verify dialogue.json exists
    json_path = out_dir / doc.movie_id / "dialogue.json"
    assert json_path.exists()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["status"] == "SUCCESS"
    assert data["total_entries"] == 2


def test_extract_embedded_subtitles_by_language(multi_sub_mkv: Path, tmp_path: Path):
    """Test extracting specific embedded subtitle by language option (--language msa)."""
    out_dir = tmp_path / "analysis_lang_select"
    doc = extract_movie_dialogue(
        source_path=multi_sub_mkv,
        language="msa",
        output_base_dir=out_dir,
    )

    assert doc.status == "SUCCESS"
    assert doc.language == "msa"
    assert doc.total_entries == 2
    assert doc.entries[0].text == "Helo dunia!"
    assert doc.entries[1].text == "Dialog bahasa Melayu."


def test_extract_explicit_external_subtitle(no_sub_mp4: Path, tmp_path: Path):
    """Test explicitly provided external subtitle file (--subtitle-file)."""
    external_vtt = tmp_path / "custom_movie.en.vtt"
    external_vtt.write_text(
        "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nExternal VTT line 1.\n",
        encoding="utf-8"
    )

    out_dir = tmp_path / "analysis_ext_sub"
    doc = extract_movie_dialogue(
        source_path=no_sub_mp4,
        subtitle_file_path=external_vtt,
        output_base_dir=out_dir,
    )

    assert doc.status == "SUCCESS"
    assert doc.source_type == "external_subtitle"
    assert doc.language == "en"
    assert doc.total_entries == 1
    assert doc.entries[0].text == "External VTT line 1."


def test_external_subtitle_discovery_near_movie(tmp_path: Path):
    """Test auto-discovery of subtitle file placed alongside the movie file."""
    movie_dir = tmp_path / "movie_dir"
    movie_dir.mkdir()
    movie_path = movie_dir / "Adventure Movie 2026.mp4"

    # Create dummy movie
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=25",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
        "-c:v", "libx264", "-c:a", "aac",
        str(movie_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)

    # Place Adventure Movie 2026.srt next to it
    srt_path = movie_dir / "Adventure Movie 2026.srt"
    srt_path.write_text(
        "1\n00:00:00,500 --> 00:00:01,500\nDiscovered subtitle line.\n",
        encoding="utf-8"
    )

    out_dir = tmp_path / "analysis_discovery"
    doc = extract_movie_dialogue(
        source_path=movie_path,
        output_base_dir=out_dir,
    )

    assert doc.status == "SUCCESS"
    assert doc.source_type == "external_subtitle"
    assert doc.total_entries == 1
    assert doc.entries[0].text == "Discovered subtitle line."


def test_no_subtitles_available(no_sub_mp4: Path, tmp_path: Path):
    """Test structured status when movie has no subtitles and no external files."""
    out_dir = tmp_path / "analysis_no_sub"
    doc = extract_movie_dialogue(
        source_path=no_sub_mp4,
        output_base_dir=out_dir,
    )

    assert doc.status == "NO_DIALOGUE_SOURCE_AVAILABLE"
    assert doc.source_type == "no_dialogue"
    assert doc.total_entries == 0
    assert len(doc.entries) == 0

    json_path = out_dir / doc.movie_id / "dialogue.json"
    assert json_path.exists()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["status"] == "NO_DIALOGUE_SOURCE_AVAILABLE"


def test_nonexistent_movie_error(tmp_path: Path):
    """Test error when movie does not exist."""
    with pytest.raises(SourceFileNotFoundError):
        extract_movie_dialogue(
            source_path=tmp_path / "ghost_movie.mp4",
            output_base_dir=tmp_path,
        )


def test_nonexistent_explicit_subtitle_error(no_sub_mp4: Path, tmp_path: Path):
    """Test error when explicitly specified subtitle file does not exist."""
    with pytest.raises(SubtitleFileNotFoundError):
        extract_movie_dialogue(
            source_path=no_sub_mp4,
            subtitle_file_path=tmp_path / "missing.srt",
            output_base_dir=tmp_path,
        )


def test_idempotent_resume_and_force_rebuild(multi_sub_mkv: Path, tmp_path: Path):
    """Test idempotent resume reuses dialogue.json and --force rebuilds."""
    out_dir = tmp_path / "analysis_idempotent"

    # 1. First run
    doc1 = extract_movie_dialogue(
        source_path=multi_sub_mkv,
        output_base_dir=out_dir,
    )
    json_path = out_dir / doc1.movie_id / "dialogue.json"
    assert json_path.exists()
    mtime1 = json_path.stat().st_mtime

    # 2. Second run without force (should reuse)
    time.sleep(0.05)
    doc2 = extract_movie_dialogue(
        source_path=multi_sub_mkv,
        output_base_dir=out_dir,
        force=False,
    )
    mtime2 = json_path.stat().st_mtime
    assert mtime1 == mtime2
    assert doc2.total_entries == 2

    # 3. Third run with force=True (should rebuild)
    time.sleep(0.05)
    doc3 = extract_movie_dialogue(
        source_path=multi_sub_mkv,
        output_base_dir=out_dir,
        force=True,
    )
    mtime3 = json_path.stat().st_mtime
    assert mtime3 >= mtime1
    assert doc3.total_entries == 2


def test_cli_extract_dialogue(multi_sub_mkv: Path, tmp_path: Path):
    """Test CLI execution python -m movie_analyzer.extract_dialogue."""
    out_dir = tmp_path / "cli_analysis_sub"
    cmd = [
        sys.executable,
        "-m", "movie_analyzer.extract_dialogue",
        str(multi_sub_mkv),
        "--output-dir", str(out_dir),
    ]

    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert res.returncode == 0, f"CLI failed: {res.stderr}"
    assert "[SUCCESS] DIALOGUE EXTRACTION" in res.stdout
