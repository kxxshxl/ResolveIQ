FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf TOKENIZERS_PARALLELISM=false DATA_DIR=/app/data
WORKDIR /app

# CPU-only torch keeps the image small; models run fine on CPU for MiniLM-class sizes. constraints.txt pins every package to the versions
# the recorded evaluation used (scripts/freeze_constraints.py), so an image built later does not silently run newer libraries.
COPY requirements.txt constraints.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==$(sed -n 's/^torch==//p' constraints.txt)" \
 && pip install -r requirements.txt -c constraints.txt

# Bake the embedding + reranker weights into the image so containers start offline and fast.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Severity/sentiment NLI model (~1.7 GB). `--build-arg BAKE_AFFECT=0` skips it (set AFFECT_ENABLED=false: rule/kNN fallback).
ARG BAKE_AFFECT=1
ARG AFFECT_MODEL=MoritzLaurer/deberta-v3-large-zeroshot-v2.0
RUN if [ "$BAKE_AFFECT" = "1" ]; then python -c "import sys; from transformers import AutoTokenizer, AutoModelForSequenceClassification as M; \
AutoTokenizer.from_pretrained(sys.argv[1]); M.from_pretrained(sys.argv[1])" "$AFFECT_MODEL"; fi

COPY app ./app
COPY scripts ./scripts

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 app && chown -R app:app /app /opt/hf
USER app

EXPOSE 8000
# Stateless: scale by running more replicas behind a load balancer (state lives in Postgres/Redis).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "*"]
