"""
Movie Review Engine - Google Gemini Multimodal AI Provider
Integrates Gemini 2.5/1.5 Flash Vision models for structured timeline event analysis.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

from ai.base import (
    BaseAIProvider,
    EventRecord,
    EventUncertainty,
    VisualAnalysis,
    MissingAPIKeyError,
    ModelInvocationError,
    InvalidModelResponseError,
)

logger = logging.getLogger("ai.providers.gemini")

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"


def _load_dotenv_if_present() -> None:
    """Load key-value pairs from .env if present in workspace or root."""
    for candidate in (
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",
        Path.home() / ".env",
    ):
        if candidate.exists() and candidate.is_file():
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("'\"")
                            if k and k not in os.environ:
                                os.environ[k] = v
            except Exception as exc:
                logger.debug("Failed to read .env from %s: %s", candidate, exc)


class GeminiAIProvider(BaseAIProvider):
    """Google Gemini Multimodal Provider implementation."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 45,
        debug_dir: Optional[Path] = None,
    ):
        _load_dotenv_if_present()
        resolved_model = model_name or os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        if api_key is not None:
            resolved_key = api_key
        else:
            resolved_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

        super().__init__(model_name=resolved_model, api_key=resolved_key)
        self.timeout = timeout
        self.debug_dir = debug_dir
        self.endpoint_url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        )

    @property
    def provider_name(self) -> str:
        return "gemini"

    def _ensure_api_key(self) -> str:
        """Validate API key is available before making network calls."""
        if not self.api_key or not self.api_key.strip():
            raise MissingAPIKeyError(
                "Gemini API key is not configured. Please set the GEMINI_API_KEY or GOOGLE_API_KEY "
                "environment variable, or provide --api-key."
            )
        return self.api_key

    def _build_prompt(
        self,
        timestamp_str: str,
        dialogue_context: List[Dict[str, Any]],
        previous_context: List[str],
    ) -> str:
        """Construct structured multimodal prompt for Gemini."""
        dialogue_section = "None"
        if dialogue_context:
            lines = [f"[{d.get('start', '')} -> {d.get('end', '')}] \"{d.get('text', '')}\"" for d in dialogue_context]
            dialogue_section = "\n".join(lines)

        prev_section = "None (Opening of movie)"
        if previous_context:
            prev_section = "\n".join(previous_context[-4:])

        return f"""You are a professional film analyst and movie review AI.
Analyze this video frame captured at timestamp {timestamp_str} of the movie.

### PREVIOUS RECENT SCENE CONTEXT:
{prev_section}

### DIALOGUE NEAR THIS TIMESTAMP:
{dialogue_section}

### STRICT ANALYSIS GUIDELINES:
1. CHARACTER LABELS: Do NOT hallucinate character names unless explicitly proven in the dialogue or previous context. Use temporary labels like 'PERSON_A', 'PERSON_B', 'male_officer_01', 'female_witness_01'.
2. VISUAL DETAILS: Count visible people, describe specific actions, setting/location, key visible objects, and emotional tone.
3. PLOT SIGNIFICANCE: Rate 1 to 10 (1=unimportant/filler, 5=standard progression, 8=key revelation/action, 10=critical climax).
4. UNCERTAINTY: Rate confidence level ('low', 'medium', 'high') for character identity and event interpretation.
5. Provide a concise factual 1-2 sentence event summary.

Respond ONLY with a JSON object adhering to this schema:
{{
  "visual": {{
    "people_count": <int>,
    "character_labels": ["PERSON_A", ...],
    "location": "<string>",
    "action": "<string>",
    "objects": ["<string>", ...],
    "emotion": "<string>",
    "interaction": "<string or null>",
    "visible_text": "<string or null>"
  }},
  "event_summary": "<string>",
  "plot_significance": <int 1-10>,
  "uncertainty": {{
    "character_identity": "<low|medium|high>",
    "event_interpretation": "<low|medium|high>",
    "location_certainty": "<low|medium|high>"
  }}
}}"""

    def analyze_frame_event(
        self,
        image_path: Path,
        timestamp_seconds: float,
        timestamp_str: str,
        dialogue_context: List[Dict[str, Any]],
        previous_context: List[str],
    ) -> EventRecord:
        """Call Gemini Vision API with image and context, returning a validated EventRecord."""
        api_key = self._ensure_api_key()

        if not image_path.exists():
            raise FileNotFoundError(f"Frame image not found: {image_path}")

        # Encode image to base64
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        prompt_text = self._build_prompt(timestamp_str, dialogue_context, previous_context)

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt_text},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": img_b64,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.2,
                "max_output_tokens": 4096,
            },
        }

        url = f"{self.endpoint_url}?key={api_key}"

        max_retries = 3
        last_response = None
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(url, json=payload, timeout=self.timeout)
                last_response = response
                if response.status_code == 200:
                    break
                if response.status_code in (429, 500, 502, 503, 504):
                    logger.warning(
                        "Gemini API returned transient HTTP %d (attempt %d/%d). Retrying in %ds...",
                        response.status_code, attempt, max_retries, attempt * 2
                    )
                    import time
                    time.sleep(attempt * 2)
                    continue
                # Non-transient error
                break
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as net_err:
                logger.warning(
                    "Gemini API network timeout/error (attempt %d/%d): %s. Retrying...",
                    attempt, max_retries, net_err
                )
                import time
                time.sleep(attempt * 2)
                continue

        response = last_response
        if response is None:
            raise ModelInvocationError("Gemini API failed to return a response after retries.")

        if response.status_code != 200:
            logger.error("Gemini API returned HTTP %d: %s", response.status_code, response.text)
            raise ModelInvocationError(
                f"Gemini API returned error {response.status_code}: {response.text}"
            )

        try:
            res_json = response.json()
            candidates = res_json.get("candidates", [])
            if not candidates:
                raise InvalidModelResponseError("Gemini response contained no candidates.")
            content_part = candidates[0]["content"]["parts"][0]["text"]
            parsed_data = json.loads(content_part)

            # Preserve sanitized raw response for debugging without secrets
            if self.debug_dir:
                try:
                    self.debug_dir.mkdir(parents=True, exist_ok=True)
                    debug_file = self.debug_dir / f"live_gemini_raw_response_{int(timestamp_seconds):04d}s.json"
                    with open(debug_file, "w", encoding="utf-8") as df:
                        json.dump(
                            {
                                "model": self.model_name,
                                "timestamp": timestamp_str,
                                "raw_parsed_content": parsed_data,
                            },
                            df,
                            indent=2,
                            ensure_ascii=False,
                        )
                except Exception as dbg_err:
                    logger.debug("Failed to write debug file: %s", dbg_err)
        except Exception as exc:
            logger.error("Failed to parse Gemini response text: %s", exc)
            raise InvalidModelResponseError(f"Invalid JSON from Gemini: {exc}") from exc

        # Validate schema
        try:
            vis_dict = parsed_data.get("visual", {})
            visual = VisualAnalysis(
                people_count=int(vis_dict.get("people_count", 0)),
                character_labels=list(vis_dict.get("character_labels", [])),
                location=str(vis_dict.get("location", "unknown")),
                action=str(vis_dict.get("action", "unknown action")),
                objects=list(vis_dict.get("objects", [])),
                emotion=str(vis_dict.get("emotion", "neutral")),
                interaction=vis_dict.get("interaction"),
                visible_text=vis_dict.get("visible_text"),
            )

            unc_dict = parsed_data.get("uncertainty", {})
            uncertainty = EventUncertainty(
                character_identity=str(unc_dict.get("character_identity", "medium")),
                event_interpretation=str(unc_dict.get("event_interpretation", "medium")),
                location_certainty=unc_dict.get("location_certainty", "medium"),
            )

            summary = str(parsed_data.get("event_summary", "Event recorded."))
            significance = int(parsed_data.get("plot_significance", 5))
            significance = max(1, min(10, significance))

            return EventRecord(
                index=1,
                timestamp_seconds=timestamp_seconds,
                timestamp=timestamp_str,
                frame_file="",
                dialogue=dialogue_context,
                visual=visual,
                event_summary=summary,
                plot_significance=significance,
                uncertainty=uncertainty,
            )
        except Exception as exc:
            logger.error("Failed to map Gemini response to EventRecord: %s", exc)
            raise InvalidModelResponseError(f"Schema validation error: {exc}") from exc
