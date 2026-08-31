# Git Synchronization Workflow

This document outlines the standard synchronization workflow between machines for the GCAO Movie Review Engine.

---

## Machine Locations

- **Laptop**: `C:\Users\User\Documents\GCAO_MOVIE_REVIEW_ENGINE`
- **Home PC**: `D:\GCAO_MOVIE_REVIEW_ENGINE`

---

## Standard Development Workflow

### 1. Before Starting Work
Always pull the latest changes before beginning work:
```bash
git pull
```

### 2. After Verified Work
Ensure tests pass, check status, stage only source files, commit, and push:
```bash
# 1. Run full test suite
pytest

# 2. Check changed files
git status

# 3. Stage source files
git add <source files>

# 4. Commit verified changes
git commit -m "Describe your changes"

# 5. Push to private remote
git push
```

### 3. On the Other Machine
Pull the new changes and verify tests:
```bash
git pull
pytest
```

---

## What NOT to Sync

The following files and directories must **NEVER** be committed or synced across machines:
- `.env` (contains machine-specific API keys and secrets)
- Raw movie files (`.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`)
- `cache/` (temporary extracted frames, test files, intermediate audio)
- `analysis/` (generated movie metadata, timelines, dialogue indexes, character profiles)
- `output/` (rendered review videos and final outputs)
- `logs/` (local execution logs)
- Local machine absolute paths

All heavy media, cache, analysis, and environment secrets are automatically excluded by `.gitignore`.

---

## Machine-Specific Configuration (`.env`)

Each machine maintains its own independent `.env` file created from `.env.example`.

`MOVIES_SOURCE` is configured per machine based on local drive layout:

- **Laptop Example**:
  ```env
  MOVIES_SOURCE=C:\Users\User\Documents\GCAO_MOVIE_REVIEW_ENGINE\cache
  ```
- **Home PC Example**:
  ```env
  MOVIES_SOURCE=F:\MOVIES_SOURCE
  ```
