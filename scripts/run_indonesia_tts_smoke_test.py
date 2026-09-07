from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(r"D:\GCAO MOVIE VOICE ENGINE")
REPO = Path(r"D:\GCAO_MOVIE_REVIEW_ENGINE")
INPUT = REPO / "analysis" / "him" / "variants" / "default" / "narasi_indonesia.json"
REFERENCE = ROOT / "samples" / "voice_reference" / "dii3.mp3"
TEST_TEXTS = [
    "Dari kecil, Cameron memang sudah terpaku dengan sosok quarterback legendaris itu.",
    "Gue kira hidup dia bakal biasa aja, tapi ternyata makin ke sini semuanya malah makin kacau.",
    "Nah, dari sini ceritanya mulai berubah. Awalnya kelihatan biasa, tapi ada sesuatu yang bikin semuanya terasa nggak beres.",
]


def main() -> int:
    load_dotenv(ROOT / ".env")
    base_url = os.getenv("INDONESIA_TTS_URL")
    if not base_url:
        raise RuntimeError("INDONESIA_TTS_URL is required; deployment is intentionally not performed by this script")
    reference_b64 = base64.b64encode(REFERENCE.read_bytes()).decode("ascii")
    reference_text = "Gua lagi gabut parah nih, beneran deh. Lu ada acara gak? Nongkrong yuk, ngopi-ngopi santai lah kita. Males banget gua di rumah mulu, sumpah."
    for index, text in enumerate(TEST_TEXTS, 1):
        response = requests.post(
            f"{base_url.rstrip('/')}/synthesize",
            json={"text": text, "language": "id", "style": "ID_JAKARTA_MILLENNIAL", "reference_audio_b64": reference_b64, "reference_text": reference_text, "speed": 1.0, "seed": 42},
            timeout=300,
        )
        response.raise_for_status()
        payload = response.json()
        output = Path("scratch") / f"indonesia_tts_smoke_{index}.wav"
        output.parent.mkdir(exist_ok=True)
        output.write_bytes(base64.b64decode(payload["audio_b64"]))
        print(json.dumps({"sample": index, "duration_seconds": payload.get("duration_seconds"), "generation_ms": payload.get("generation_ms"), "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
