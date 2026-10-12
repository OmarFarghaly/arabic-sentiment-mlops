# syntax=docker/dockerfile:1

FROM python:3.12-slim-bookworm AS builder

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir \
    "torch==2.11.0+cpu" \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    "torch==2.11.0+cpu" \
    "transformers==5.18.0" \
    .

RUN pip check
RUN python -c "import torch; assert torch.version.cuda is None, 'Expected CPU-only Torch'"


FROM python:3.12-slim-bookworm AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app

WORKDIR /app

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_DIR="/app/models/production" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app models/production/ /app/models/production/

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).read()"]

CMD ["uvicorn", "arabic_sentiment_mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]