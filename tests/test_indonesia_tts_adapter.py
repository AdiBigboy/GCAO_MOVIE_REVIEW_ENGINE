from pathlib import Path

WORKER = Path(__file__).parents[1] / "remote_workers" / "indonesia_tts" / "worker.py"
DOCKERFILE = WORKER.parent / "Dockerfile"


def test_indonesia_worker_isolated_from_malaysian_worker():
    text = WORKER.read_text(encoding="utf-8")
    assert "Eempostor/F5-TTS-INDO-FINETUNE-V2" in text
    assert "remote_workers/malaysian_f5" not in text
    assert "LANGUAGE_MUST_BE_ID" in text


def test_indonesia_worker_contract_is_present():
    text = WORKER.read_text(encoding="utf-8")
    assert '@app.get("/health")' in text
    assert '@app.get("/ready")' in text
    assert '@app.post("/synthesize")' in text
    assert 'ID_JAKARTA_MILLENNIAL' in text
    assert 'reference_text' in text


def test_indonesia_container_is_separate():
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "COPY worker.py ." in text
    assert "malaysian_f5" not in text
