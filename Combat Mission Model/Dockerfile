FROM python:3.11-slim

WORKDIR /app

# API + training stack (exclude ipykernel from dev requirements)
COPY requirements.txt .
RUN grep -v '^ipykernel' requirements.txt > /tmp/requirements.api.txt \
    && pip install --no-cache-dir -r /tmp/requirements.api.txt \
    && rm /tmp/requirements.api.txt

# train.py intentionally omitted from the image — build skips retraining so small hosts (e.g.
# Render Free) avoid OOM during sklearn CV + GradientBoosting. Commit `athena.joblib` +
# `models/mission_model_metadata.json` and rebuild the image after local `python train.py`.
COPY model.py schemas.py main.py mission_context.py ./
COPY web ./web/
COPY evaluations_full.csv detailed_full.csv ./
COPY athena.joblib ./
COPY models ./models/

ENV EVAL_PATH=/app/evaluations_full.csv
ENV DETAIL_PATH=/app/detailed_full.csv
ENV MODEL_PATH=/app/athena.joblib
ENV MODEL_METADATA_PATH=/app/models/mission_model_metadata.json

RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

# Render, Fly, Railway, Cloud Run, etc. set PORT; default 8000 for local Docker.
ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=5 \
    CMD python -c "import os,urllib.request; p=os.environ.get('PORT','8000'); urllib.request.urlopen(f'http://127.0.0.1:{p}/health',timeout=4)"

CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
