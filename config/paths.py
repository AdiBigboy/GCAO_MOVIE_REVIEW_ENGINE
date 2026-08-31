"""
Central configuration for project paths and environment settings.
Dynamically resolves paths relative to PROJECT_ROOT for complete cross-machine portability.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# Dynamically derive PROJECT_ROOT from project location
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Alias for backward compatibility
WORKSPACE_DIR = PROJECT_ROOT


def _load_env_file(env_file_path: Optional[Path] = None) -> None:
    """Load machine-specific environment variables from .env if present."""
    target_env = env_file_path or (PROJECT_ROOT / ".env")
    if target_env.exists() and target_env.is_file():
        try:
            with open(target_env, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass


# Automatically load machine-specific .env
_load_env_file()

# Standard project subdirectories
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUT_DIR = PROJECT_ROOT / "output"
DATABASE_DIR = PROJECT_ROOT / "database"
LOGS_DIR = PROJECT_ROOT / "logs"
INPUT_DIR = PROJECT_ROOT / "input"

# External movie source path (machine-specific, configurable via MOVIES_SOURCE environment variable)
MOVIES_SOURCE = os.environ.get("MOVIES_SOURCE")
MOVIES_SOURCE_DIR = Path(MOVIES_SOURCE).resolve() if MOVIES_SOURCE else None


def get_movie_source_path(filename_or_path: str | Path) -> Path:
    """
    Resolve a movie path against MOVIES_SOURCE or as an absolute / relative path.
    """
    p = Path(filename_or_path)
    if p.is_absolute() and p.exists():
        return p.resolve()

    # If MOVIES_SOURCE is configured and file exists inside it
    if MOVIES_SOURCE_DIR and (MOVIES_SOURCE_DIR / p.name).exists():
        return (MOVIES_SOURCE_DIR / p.name).resolve()

    # Check relative to PROJECT_ROOT
    if (PROJECT_ROOT / p).exists():
        return (PROJECT_ROOT / p).resolve()

    # Default to resolved path
    return p.resolve()


# Ensure essential workspace directories exist
for path in (ANALYSIS_DIR, CACHE_DIR, OUTPUT_DIR, DATABASE_DIR, LOGS_DIR, INPUT_DIR):
    path.mkdir(parents=True, exist_ok=True)

