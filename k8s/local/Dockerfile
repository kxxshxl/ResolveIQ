# Local-cluster image: the backend image plus the seed data the manifests expect at DATA_DIR=/app/data
# (taxonomy, processed corpus, affect artifact). docker-compose mounts ./data instead; Kubernetes has no host mount, so the
# data is baked in. Build with the data directory as the context (see docs/production.md):
#   docker build -t resolveiq-backend:kind-base ./backend
#   docker build -f k8s/local/Dockerfile --build-arg BASE=resolveiq-backend:kind-base -t resolveiq-backend:kind data
ARG BASE=resolveiq-backend:kind-base
FROM ${BASE}
COPY --chown=app:app . /app/data
