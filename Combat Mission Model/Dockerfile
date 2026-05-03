FROM python:3.11-slim

WORKDIR /app

# API + training stack (exclude ipykernel from dev requirements)
COPY requirements.txt .
RUN grep -v '^ipykernel' requirements.txt > /tmp/requirements.api.txt \
    && pip install --no-cache-dir -r /tmp/requirements.api.txt \
    && rm /tmp/requirements.api.txt

COPY model.py schemas.py main.py train.py mission_context.py ./
COPY evaluations_full.csv detailed_full.csv ./

ENV EVAL_PATH=/app/evaluations_full.csv
ENV DETAIL_PATH=/app/detailed_full.csv
ENV MODEL_PATH=/app/athena.joblib
ENV MODEL_METADATA_PATH=/app/models/mission_model_metadata.json

RUN python train.py

RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
