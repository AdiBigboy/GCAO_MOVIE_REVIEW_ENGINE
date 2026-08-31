"""
Portability and Paths Test Suite
Validates that PROJECT_ROOT, directory mappings, and MOVIES_SOURCE work portably across machines.
"""

import os
import re
from pathlib import Path
import pytest

import config
from config.paths import (
    PROJECT_ROOT,
    WORKSPACE_DIR,
    ANALYSIS_DIR,
    CACHE_DIR,
    OUTPUT_DIR,
    DATABASE_DIR,
    LOGS_DIR,
    INPUT_DIR,
    get_movie_source_path,
)


def test_project_root_derived_dynamically():
    """Verify that PROJECT_ROOT matches the actual repository root on disk."""
    expected_root = Path(__file__).resolve().parents[1]
    assert PROJECT_ROOT == expected_root
    assert WORKSPACE_DIR == PROJECT_ROOT
    assert PROJECT_ROOT.exists()
    assert (PROJECT_ROOT / "config").exists()
    assert (PROJECT_ROOT / "movie_analyzer").exists()


def test_directories_structure():
    """Verify all standard subdirectories are defined under PROJECT_ROOT."""
    assert ANALYSIS_DIR == PROJECT_ROOT / "analysis"
    assert CACHE_DIR == PROJECT_ROOT / "cache"
    assert OUTPUT_DIR == PROJECT_ROOT / "output"
    assert DATABASE_DIR == PROJECT_ROOT / "database"
    assert LOGS_DIR == PROJECT_ROOT / "logs"
    assert INPUT_DIR == PROJECT_ROOT / "input"

    for d in (ANALYSIS_DIR, CACHE_DIR, OUTPUT_DIR, DATABASE_DIR, LOGS_DIR, INPUT_DIR):
        assert d.exists()
        assert d.is_dir()


def test_movies_source_resolution(monkeypatch, tmp_path: Path):
    """Test resolving paths via MOVIES_SOURCE environment configuration."""
    external_dir = tmp_path / "external_movies"
    external_dir.mkdir()
    sample_movie = external_dir / "sample_external.mp4"
    sample_movie.write_text("dummy video", encoding="utf-8")

    # Set MOVIES_SOURCE in env
    monkeypatch.setenv("MOVIES_SOURCE", str(external_dir))
    
    # Reload or test path resolution
    import config.paths as cp
    cp.MOVIES_SOURCE = str(external_dir)
    cp.MOVIES_SOURCE_DIR = external_dir

    resolved = get_movie_source_path("sample_external.mp4")
    assert resolved == sample_movie.resolve()


def test_no_hardcoded_drive_letters_in_source_code():
    """Audit all python source code files to ensure no hardcoded drive letters exist in logic."""
    drive_pattern = re.compile(r"['\"][A-Za-z]:[\\/]")
    
    violations = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        if any(p in root for p in [".git", ".pytest_cache", "__pycache__"]):
            continue
        for f in files:
            if f.endswith(".py"):
                file_path = Path(root) / f
                with open(file_path, "r", encoding="utf-8", errors="ignore") as src:
                    for line_num, line in enumerate(src, 1):
                        stripped = line.strip()
                        if stripped.startswith("#"):
                            continue
                        if drive_pattern.search(line):
                            violations.append(f"{file_path.relative_to(PROJECT_ROOT)}:{line_num}: {stripped}")

    assert not violations, f"Found hardcoded drive letters in source code:\n" + "\n".join(violations)
