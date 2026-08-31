# GCAO Movie Review Engine

An automated multimodal video ingestion, timeline sampling, subtitle extraction, and AI-driven scene analysis engine.

## Architecture

- **`config/`**: Dynamic project root paths, machine-specific environment configurations, and movie source resolution.
- **`movie_analyzer/`**: Core video processing pipeline:
  - Phase 1: Ingestion & metadata extraction (`ingest.py`)
  - Phase 2: Timeline frame sampling (`sample_frames.py`)
  - Phase 3: Subtitle & dialogue extraction (`extract_dialogue.py`, `subtitles.py`)
  - Phase 4: Multimodal timeline event analysis (`analyze_events.py`)
  - Phase 5: Character identity tracking & character memory (`track_characters.py`)
- **`ai/`**: Multimodal AI provider abstractions (`GeminiAIProvider`, `MockAIProvider`).
- **`tests/`**: Automated pytest test suite covering all modules.

## Setup & Portability

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Configure machine-specific environment variables in `.env` (e.g. `GEMINI_API_KEY`, `MOVIES_SOURCE`).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run tests:
   ```bash
   pytest
   ```
