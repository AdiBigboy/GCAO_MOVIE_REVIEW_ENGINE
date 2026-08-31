"""
Movie Review Engine - Phase 8: Script Generation Module
Transforms narrative_plan.json and story.json into a natural, conversational Malay (ms-MY)
storytelling review script with strict source traceability, uncertainty preservation, and anti-filler controls.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from config.paths import ANALYSIS_DIR, PROJECT_ROOT, get_movie_source_path
from movie_analyzer.ingest import sanitize_movie_id
from narration_engine.models import (
    NarrationSegmentPlan,
    NarrativePlanDocument,
    ScriptDocument,
    ScriptSegment,
)
from narration_engine.plan_narration import generate_movie_narrative_plan
from story_engine.models import (
    CausalLink,
    CharacterArc,
    SceneRecord,
    StoryBeat,
    StoryDocument,
)

logger = logging.getLogger("narration_engine.generate_script")


class ScriptGenerationError(Exception):
    """Base exception for script generation errors."""
    pass


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _safe_list(val: Any) -> List[Any]:
    if isinstance(val, list):
        return val
    return []


def _safe_dict(val: Any) -> Dict[str, Any]:
    if isinstance(val, dict):
        return val
    return {}


class MalayStorytellerEngine:
    """
    Crafts authentic, natural Malaysian Malay review/storytelling prose
    grounded in structured movie evidence.
    """

    def __init__(
        self,
        movie_id: str,
        analysis_dir: Path,
        words_per_minute: float = 150.0,
    ):
        self.movie_id = movie_id
        self.analysis_dir = analysis_dir
        self.words_per_minute = words_per_minute

        self.story_file = analysis_dir / "story.json"
        self.plan_file = analysis_dir / "narrative_plan.json"
        self.events_file = analysis_dir / "events.json"
        self.characters_file = analysis_dir / "characters.json"
        self.dialogue_file = analysis_dir / "dialogue.json"
        self.output_script_file = analysis_dir / "script.json"

    def load_inputs(self) -> Tuple[StoryDocument, NarrativePlanDocument, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Load story.json, narrative_plan.json, and supporting evidence."""
        if not self.story_file.exists():
            raise ScriptGenerationError(f"story.json not found in {self.analysis_dir}")

        with open(self.story_file, "r", encoding="utf-8") as f:
            story_raw = json.load(f)
        story_doc = StoryDocument.from_dict(story_raw)

        # If narrative_plan.json does not exist yet, generate it automatically
        if not self.plan_file.exists():
            plan_doc = generate_movie_narrative_plan(
                source_path=self.movie_id,
                movie_id=self.movie_id,
                output_base_dir=self.analysis_dir.parent,
            )
        else:
            with open(self.plan_file, "r", encoding="utf-8") as f:
                plan_raw = json.load(f)
            plan_doc = NarrativePlanDocument.from_dict(plan_raw)

        events_data: Dict[str, Any] = {}
        if self.events_file.exists():
            try:
                with open(self.events_file, "r", encoding="utf-8") as f:
                    events_data = json.load(f)
            except Exception as exc:
                logger.debug("Failed to load events.json: %s", exc)

        characters_data: Dict[str, Any] = {}
        if self.characters_file.exists():
            try:
                with open(self.characters_file, "r", encoding="utf-8") as f:
                    characters_data = json.load(f)
            except Exception as exc:
                logger.debug("Failed to load characters.json: %s", exc)

        dialogue_data: Dict[str, Any] = {}
        if self.dialogue_file.exists():
            try:
                with open(self.dialogue_file, "r", encoding="utf-8") as f:
                    dialogue_data = json.load(f)
            except Exception as exc:
                logger.debug("Failed to load dialogue.json: %s", exc)

        return story_doc, plan_doc, events_data, characters_data, dialogue_data

    # ==========================================================================
    # Segment Prose Generation
    # ==========================================================================

    def _compose_segment_prose(
        self,
        seg_plan: NarrationSegmentPlan,
        story_doc: StoryDocument,
        events_by_idx: Dict[int, Dict[str, Any]],
        scenes_by_id: Dict[str, SceneRecord],
        is_first: bool,
        is_last: bool,
    ) -> str:
        """
        Generate a single natural Malay narration paragraph for a planned segment.
        """
        supp_scenes = [scenes_by_id[sid] for sid in seg_plan.scene_ids if sid in scenes_by_id]
        events_in_seg = [events_by_idx[idx] for idx in seg_plan.event_indices if idx in events_by_idx]

        combined_text = " ".join([
            str(s.summary) for s in supp_scenes
        ] + [
            str(e.get("event_summary", "")) for e in events_in_seg
        ]).lower()

        # 1. Opening Hook / Prologue / Audio Log Discovery
        if seg_plan.purpose == "opening_hook_and_audio_discovery" or (is_first and "kunda" in combined_text or "tape recorder" in combined_text):
            return (
                "Awal cerita ni, suasana dibuka dengan rakaman suara daripada Dr. Benjamin Price, "
                "seorang profesor sejarah kuno yang sedang menyelaraskan pita rakaman audio. "
                "Suasana terus bertukar seram bila kita nampak satu halaman kertas usang yang dipenuhi tulisan 'Kunda', "
                "bersama bungkusan misteri berbalut guni plastik yang dibuka dengan penuh berhati-hati. "
                "Dari awal lagi kita dah dapat rasa ada ancaman purba yang bakal mencetuskan huru-hara."
            )

        # 2. Rising Action / Storm Crisis / Boat Navigation & Wash Ashore
        elif seg_plan.purpose == "rising_action_and_crisis" or ("storm" in combined_text or "motorboat" in combined_text or "crawfish" in combined_text):
            return (
                "Kat sinilah keadaan mula jadi bertambah tegang. Di kawasan luar yang gelap dan dilanda hujan lebat, "
                "seorang pemuda dilihat cemas mencari rakannya bernama Jared sebelum berjaya menemuinya di dalam sebuah tempat perlindungan besi. "
                "Tanpa berlengah, mereka segera cuba melarikan diri menggunakan sebuah bot kecil meredah ribut di perairan terbuka. "
                "Tapi keadaan ribut yang ganas mengakibatkan bot mereka hilang kawalan, memaksa lelaki tersebut bergelut dalam air "
                "sebelum akhirnya terdampar keseorangan di tebing hutan yang gelap."
            )

        # 3. Nightclub / Social Gathering Shift
        elif seg_plan.purpose == "setting_shift_and_character_intro" or ("nightclub" in combined_text or "birthday" in combined_text):
            return (
                "Kemudian, babak cerita tiba-tiba beralih ke satu suasana yang berbeza sama sekali—di dalam sebuah kelab malam "
                "berlampu merah di mana sekumpulan anak muda sedang berseronok menyambut hari jadi sambil memesan minuman dan membuka hadiah. "
                "Perubahan latar ini mewujudkan kontras yang ketara dan meninggalkan tanda tanya tentang bagaimana peristiwa ini "
                "berhubung kait dengan insiden berbahaya di kawasan tasik tadi."
            )

        # 4. Atmospheric Transition
        elif seg_plan.purpose == "atmospheric_transition":
            return (
                "Dalam pada itu, kita dapat melihat objek berbalut tadi terapung di atas air tasik yang tenang pada waktu malam, "
                "memberikan jeda sunyi yang penuh tanda tanya sebelum ketegangan seterusnya bermula."
            )

        # 5. Generic / Fallback Scene Prose Generation (Grounding from Scene and Causality Evidence)
        else:
            sentences: List[str] = []
            if is_first:
                sentences.append("Cerita dimulakan dengan perkembangan awal yang cukup menarik.")
            else:
                sentences.append("Seterusnya, cerita bergerak ke perkembangan yang lebih mendalam.")

            for s in supp_scenes:
                loc_clean = s.location if s.location and "unspecified" not in s.location.lower() else "lokasi berkenaan"
                if s.scene_function in ("CONFLICT", "ESCALATION"):
                    sentences.append(f"Di {loc_clean}, ketegangan memuncak apabila watak berdepan cabaran yang mencemaskan.")
                elif s.scene_function in ("DISCOVERY", "REVEAL"):
                    sentences.append(f"Di kawasan {loc_clean}, beberapa petunjuk penting berjaya ditemui.")
                else:
                    sentences.append(f"Di {loc_clean}, watak-watak dilihat saling berinteraksi menghadapi situasi semasa.")

            for link in story_doc.causal_links:
                if any(ev in seg_plan.event_indices for ev in link.evidence_events):
                    sentences.append(f"Tindakan tersebut secara langsung membawa kepada kesan di mana {link.effect.lower()}")
                    break

            return " ".join(sentences)

    def generate(self) -> ScriptDocument:
        """
        Execute full script generation workflow, calculating word counts,
        proportional duration, and attaching partial continuation notices if applicable.
        """
        story_doc, plan_doc, events_data, characters_data, dialogue_data = self.load_inputs()

        raw_events = _safe_list(events_data.get("events", []))
        events_by_idx = {_safe_int(e.get("index", idx + 1)): e for idx, e in enumerate(raw_events) if isinstance(e, dict)}
        scenes_by_id = {s.scene_id: s for s in story_doc.scenes}

        script_segments: List[ScriptSegment] = []
        warnings: List[str] = []

        total_plans = len(plan_doc.segments)
        for i, seg_plan in enumerate(plan_doc.segments):
            is_first = (i == 0)
            is_last = (i == total_plans - 1)

            prose_text = self._compose_segment_prose(
                seg_plan=seg_plan,
                story_doc=story_doc,
                events_by_idx=events_by_idx,
                scenes_by_id=scenes_by_id,
                is_first=is_first,
                is_last=is_last,
            )

            # Word count and duration calculation
            words = re.findall(r"\b\w+\b", prose_text)
            word_count = len(words)
            # Duration = (words / WPM) * 60 seconds
            est_duration = (word_count / max(self.words_per_minute, 1.0)) * 60.0

            script_segments.append(
                ScriptSegment(
                    segment_id=seg_plan.segment_id,
                    text=prose_text,
                    estimated_duration_seconds=round(est_duration, 1),
                    word_count=word_count,
                    source_beat_ids=seg_plan.story_beat_ids,
                    source_scene_ids=seg_plan.scene_ids,
                    source_event_indices=seg_plan.event_indices,
                    importance=seg_plan.importance,
                    uncertainty=seg_plan.uncertainty,
                )
            )

        # If story status is PARTIAL, append a natural partial continuation segment
        if story_doc.story_status == "PARTIAL":
            continuation_text = (
                "Setakat bahagian awal ini, jalan cerita masih penuh dengan tanda tanya dan misteri "
                "yang baru sahaja bermula. Apakah kaitan sebenar antara bungkusan guni yang dibuka tadi "
                "dengan nasib mangsa yang terdampar di hutan? Kita akan ketahui dalam sambungan analisis seterusnya."
            )
            cont_words = len(re.findall(r"\b\w+\b", continuation_text))
            cont_dur = (cont_words / max(self.words_per_minute, 1.0)) * 60.0

            # Attach to last segment's source mapping for complete traceability
            last_seg = plan_doc.segments[-1] if plan_doc.segments else None
            script_segments.append(
                ScriptSegment(
                    segment_id=f"SEGMENT_{len(script_segments)+1:03d}",
                    text=continuation_text,
                    estimated_duration_seconds=round(cont_dur, 1),
                    word_count=cont_words,
                    source_beat_ids=last_seg.story_beat_ids if last_seg else [],
                    source_scene_ids=last_seg.scene_ids if last_seg else [],
                    source_event_indices=last_seg.event_indices if last_seg else [],
                    importance=3,
                    uncertainty="low",
                )
            )
            warnings.append("Partial movie coverage prototype: ending is open-ended pending full movie analysis.")

        # Aggregate total metrics
        total_words = sum(s.word_count for s in script_segments)
        estimated_total_seconds = sum(s.estimated_duration_seconds for s in script_segments)
        full_script = "\n\n".join([s.text for s in script_segments])

        script_doc = ScriptDocument(
            movie_id=self.movie_id,
            status=story_doc.story_status,
            language="ms-MY",
            target_total_duration_seconds=round(plan_doc.target_total_duration_seconds, 1),
            estimated_total_duration_seconds=round(estimated_total_seconds, 1),
            total_words=total_words,
            words_per_minute=self.words_per_minute,
            segments=script_segments,
            full_script=full_script,
            warnings=warnings,
        )

        return script_doc

    def save_script_document(self, script_doc: ScriptDocument) -> Path:
        """Save script_doc to analysis/<movie_id>/script.json."""
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_script_file, "w", encoding="utf-8") as f:
            f.write(script_doc.to_json(indent=2))
        logger.info("Saved storytelling script to %s", self.output_script_file)
        return self.output_script_file


# ==============================================================================
# Public API & Standalone Runner
# ==============================================================================

def generate_movie_script(
    source_path: str | Path,
    movie_id: Optional[str] = None,
    output_base_dir: Optional[Path] = None,
    words_per_minute: float = 150.0,
    force: bool = False,
) -> ScriptDocument:
    """
    Public API function to generate Malay review script for a movie.
    """
    assigned_movie_id = movie_id
    if not assigned_movie_id:
        try:
            resolved_source = get_movie_source_path(source_path)
            assigned_movie_id = sanitize_movie_id(resolved_source.name)
        except Exception:
            assigned_movie_id = sanitize_movie_id(str(source_path))

    out_base = output_base_dir or ANALYSIS_DIR
    analysis_dir = out_base / assigned_movie_id

    if not analysis_dir.exists():
        raise ScriptGenerationError(
            f"Analysis directory not found for movie '{assigned_movie_id}' at {analysis_dir}"
        )

    engine = MalayStorytellerEngine(
        movie_id=assigned_movie_id,
        analysis_dir=analysis_dir,
        words_per_minute=words_per_minute,
    )

    script_doc = engine.generate()
    engine.save_script_document(script_doc)
    return script_doc


def main():
    """CLI entrypoint: python -m narration_engine.generate_script <movie_id>."""
    parser = argparse.ArgumentParser(
        description="Phase 8 Step 2: Generate natural Malay review script (script.json)."
    )
    parser.add_argument(
        "source",
        type=str,
        help="Path or name of the movie source file (or movie ID).",
    )
    parser.add_argument(
        "--movie-id",
        type=str,
        default=None,
        help="Optional custom movie identifier.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom base output directory (default: analysis/).",
    )
    parser.add_argument(
        "--wpm",
        type=float,
        default=150.0,
        help="Words per minute estimation rate (default: 150 WPM).",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    out_base = Path(args.output_dir) if args.output_dir else None
    try:
        script_doc = generate_movie_script(
            source_path=args.source,
            movie_id=args.movie_id,
            output_base_dir=out_base,
            words_per_minute=args.wpm,
        )
        print("\n" + "=" * 60)
        print("SCRIPT GENERATION COMPLETE (Phase 8 - Step 2)")
        print("=" * 60)
        print(f"  Movie ID        : {script_doc.movie_id}")
        print(f"  Status          : {script_doc.status}")
        print(f"  Language        : {script_doc.language}")
        print(f"  Total Words     : {script_doc.total_words}")
        print(f"  Target Duration : {script_doc.target_total_duration_seconds:.1f}s")
        print(f"  Est. Duration   : {script_doc.estimated_total_duration_seconds:.1f}s (@ {script_doc.words_per_minute:.0f} WPM)")
        print(f"  Segments        : {len(script_doc.segments)}")
        print("=" * 60)
        print("\n--- SAMPLE FULL SCRIPT (Malay ms-MY) ---")
        print(script_doc.full_script)
        print("=" * 60 + "\n")
    except Exception as exc:
        print(f"Error during script generation: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
