FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models/huggingface \
    INDONESIAN_MODEL_ID=Eempostor/F5-TTS-INDO-FINETUNE-V2 \
    INDONESIAN_MODEL_REVISION=ec3e116a9373f83cfc2d53170a72643f596ea6fc

WORKDIR /app
RUN apt-get update && apt-get install --yes --no-install-recommends ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY worker.py .
EXPOSE 8000
CMD ["uvicorn", "worker:app", "--host", "0.0.0.0", "--port", "8000"]
