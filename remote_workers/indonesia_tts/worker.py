from __future__ import annotations

import base64
import hashlib
import io
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_ID = os.getenv("INDONESIAN_MODEL_ID", "Eempostor/F5-TTS-INDO-FINETUNE-V2")
MODEL_REVISION = os.getenv("INDONESIAN_MODEL_REVISION", "ec3e116a9373f83cfc2d53170a72643f596ea6fc")
CHECKPOINT_NAME = "f5_tts_indo_v2.pt"
VOCAB_NAME = "vocab.txt"
REFERENCE_LICENSE = "CC-BY-NC-4.0"
TARGET_SAMPLE_RATE = 24000
MAX_REFERENCE_BYTES = 20 * 1024 * 1024

app = FastAPI(title="GCAO Indonesian Jakarta TTS", version="0.1.0")
_lock = threading.Lock()
_engine: Any = None
_state = "STARTING"
_error: str | None = None
_load_ms = 0.0


class SynthesizePayload(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    language: str = Field(default="id")
    style: str = Field(default="ID_JAKARTA_MILLENNIAL")
    reference_audio_b64: str = Field(..., min_length=4)
    reference_text: str = Field(..., min_length=1, max_length=2000)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    seed: int = Field(default=42, ge=0)


def _load() -> None:
    global _engine, _state, _error, _load_ms
    started = time.perf_counter()
    try:
        import torch
        from huggingface_hub import hf_hub_download
        from f5_tts.api import F5TTS

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA_NOT_AVAILABLE")
        checkpoint = hf_hub_download(MODEL_ID, CHECKPOINT_NAME, revision=MODEL_REVISION)
        vocab = hf_hub_download(MODEL_ID, VOCAB_NAME, revision=MODEL_REVISION)
        _engine = F5TTS(
            model="F5TTS_v1_Base",
            ckpt_file=checkpoint,
            vocab_file=vocab,
            ode_method="euler",
            use_ema=True,
            device="cuda",
        )
        _load_ms = round((time.perf_counter() - started) * 1000, 2)
        _state = "READY"
    except Exception as exc:
        _error = f"{type(exc).__name__}: {exc}"
        _state = "ERROR"


@app.on_event("startup")
def start_loader() -> None:
    threading.Thread(target=_load, daemon=True).start()


@app.get("/health")
def health() -> dict[str, Any]:
    import torch
    return {
        "state": _state,
        "status": "READY" if _state == "READY" else "NOT_READY",
        "gpu_available": bool(torch.cuda.is_available()),
        "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "license": REFERENCE_LICENSE,
        "load_ms": _load_ms,
        "error": _error,
    }


@app.get("/ready")
def ready() -> dict[str, Any]:
    result = health()
    result["ready"] = _state == "READY"
    return result


@app.post("/synthesize")
def synthesize(payload: SynthesizePayload) -> dict[str, Any]:
    if _state != "READY" or _engine is None:
        raise HTTPException(status_code=503, detail="MODEL_NOT_READY")
    if payload.language != "id":
        raise HTTPException(status_code=400, detail="LANGUAGE_MUST_BE_ID")
    try:
        reference = base64.b64decode(payload.reference_audio_b64, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="INVALID_REFERENCE_AUDIO") from exc
    if not reference or len(reference) > MAX_REFERENCE_BYTES:
        raise HTTPException(status_code=400, detail="INVALID_REFERENCE_AUDIO_SIZE")

    import numpy as np
    import soundfile as sf
    started = time.perf_counter()
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ref_file:
            ref_file.write(reference)
            path = ref_file.name
        with _lock:
            waveform, sample_rate, _ = _engine.infer(
                ref_file=path,
                ref_text=payload.reference_text,
                gen_text=payload.text,
                target_rms=0.1,
                cross_fade_duration=0.15,
                sway_sampling_coef=-1.0,
                cfg_strength=2.0,
                nfe_step=32,
                speed=payload.speed,
                seed=payload.seed,
                show_info=lambda _: None,
            )
        samples = np.asarray(waveform, dtype=np.float32).squeeze()
        if samples.ndim != 1 or samples.size == 0 or not np.isfinite(samples).all():
            raise RuntimeError("INVALID_WAVEFORM")
        output = io.BytesIO()
        sf.write(output, samples, int(sample_rate), format="WAV", subtype="PCM_16")
        audio = output.getvalue()
        return {
            "audio_b64": base64.b64encode(audio).decode("ascii"),
            "duration_seconds": round(samples.size / float(sample_rate), 3),
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "language": payload.language,
            "style": payload.style,
            "reference_audio_sha256": hashlib.sha256(reference).hexdigest(),
            "generation_ms": round((time.perf_counter() - started) * 1000, 2),
            "license": REFERENCE_LICENSE,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"INFERENCE_FAILED: {exc}") from exc
    finally:
        if path:
            Path(path).unlink(missing_ok=True)
