# Deployment guide

Two supported targets, both verified artefacts in this repository:

| Target | Use when | Files |
|---|---|---|
| **Single host, Docker Compose + automatic HTTPS** | one VM / on-prem box, small-to-medium load | `docker-compose.prod.yml`, `deploy/` |
| **Kubernetes** | HA, autoscaling, managed Postgres/Redis | `k8s/base/` (cloud), `k8s/local/` (kind; tested, see [kubernetes.md](kubernetes.md)) |

What was actually exercised (on Docker Desktop, Windows 11): the full production Compose stack (2 backend replicas, TLS proxy,
Prometheus scraping both replicas with a bearer token, Grafana), `deploy/smoke_test.py` (12/12 checks), backup + restore, the
fail-fast insecure-config check, and the read-only / non-root / cap-dropped containers. The Kubernetes manifests were also applied to a
single-node kind cluster and exercised end to end (22 checks, see [kubernetes.md](kubernetes.md)); that is a local cluster, not a managed one. That run used the code of commit `2d32e72`; later changes were not redeployed to the cluster (see kubernetes.md).

## 1. Single-host deployment (Docker Compose)

**Prerequisites:** Docker Engine 24+ with Compose v2, a DNS name pointing at the host (ports 80 and 443 open) for public HTTPS,
4 vCPU / 8 GB RAM minimum (the LLM is the heavy part; a GPU host or an external LLM endpoint is recommended).

```bash
git clone <repo> && cd resolveiq
python deploy/gen_secrets.py --domain support.example.com --email ops@example.com
#   -> writes .env.prod (random DB/Redis/Grafana passwords, 2 API keys) and secrets/metrics_token; prints the API keys ONCE
```
Choose the LLM (edit `.env.prod`):
* **Bundled Ollama (CPU, or GPU with an override):** keep `OLLAMA_BASE_URL=http://ollama:11434`, then
  `docker compose --env-file .env.prod -f docker-compose.prod.yml --profile ollama up -d ollama && docker compose --env-file .env.prod -f docker-compose.prod.yml --profile ollama run --rm ollama-pull`
* **External Ollama / vLLM / any OpenAI-compatible server:** set `OLLAMA_BASE_URL`, or `LLM_PROVIDERS=openai_compat` with
  `OPENAI_COMPAT_BASE_URL/_API_KEY/_MODEL`. A chain such as `LLM_PROVIDERS=ollama,openai_compat` gives automatic fallback.
* **No LLM:** the service still works in evidence-only (`degraded`) mode.

```bash
docker compose --env-file .env.prod -f docker-compose.prod.yml up -d --build        # add --profile monitoring for Prometheus/Grafana
python deploy/smoke_test.py https://support.example.com --api-key <agents key>       # must print SMOKE TEST PASSED
```
Open `https://support.example.com`, enter an API key in the UI's key field, paste a complaint. Grafana (if enabled) is bound to
`127.0.0.1:3001` on the host - use an SSH tunnel. For a local trial use `--domain localhost` (Caddy's internal CA; browsers
warn, `--insecure` for the smoke test) and set `HTTP_PORT` / `HTTPS_PORT` if 80/443 are taken.

**What the stack enforces:** only the proxy publishes ports; Postgres/Redis/backend/Prometheus are internal; the API refuses to
start with `APP_ENV=production` unless API keys (>= 24 chars) are set, CORS is explicit, DB passwords are not placeholders, and
the app connects with the DML-only role; `/metrics` is not served publicly (404 at the proxy, bearer token on the backend);
containers are non-root where the image allows it, read-only root filesystems, `no-new-privileges`, capabilities dropped,
memory limits, log rotation, restart policies.

### Operations
| Task | Command |
|---|---|
| Status | `docker compose --env-file .env.prod -f docker-compose.prod.yml ps` |
| Logs (JSON, with trace id) | `... logs -f backend` |
| Scale API | `BACKEND_REPLICAS=4` in `.env.prod`, then `up -d` (nginx re-resolves Docker DNS, Prometheus discovers replicas) |
| Upgrade | `git pull && ... up -d --build` (the `migrate` job runs first; migrations are forward-only) |
| Rollback | redeploy the previous git tag / `VERSION`; restore a backup if a migration must be undone |
| Backup | `sh deploy/backup.sh` (cron it; keeps the 14 newest in `backups/`; copy off-host) |
| Restore | `sh deploy/backup.sh restore backups/<file>.dump` |
| Rotate secrets | `python deploy/gen_secrets.py --domain ... --email ... --force`, then recreate the stack (DB passwords: change in Postgres first, or recreate the volume) |
| Load new knowledge | `POST /api/v1/ingest/ticket` / `ingest/article` (no restart) |
| Embedding model change | set `EMBEDDING_MODEL`, run `ingestion.reindex()` (see `production.md`), evaluate, then switch |

### Troubleshooting
* `backend` restarts with "refusing to start with an insecure production configuration" - the message lists exactly what to fix.
* `/health/ready` is 503 - the JSON shows which dependency failed (`database`, `models`); models need ~20-40 s on first start.
* Answers are `degraded` - the LLM is unreachable; check `OLLAMA_BASE_URL` and `docker compose ... logs backend | grep llm`.
* 429 - the per-key rate limit (`RATE_LIMIT_PER_MINUTE`) was hit.

## 2. Kubernetes
Prerequisites: ingress-nginx, cert-manager (ClusterIssuer `letsencrypt-prod`), managed Postgres with pgvector (create the
`resolveiq_app` role as in `infra/postgres/init-roles.sh`), managed Redis, an LLM endpoint, a container registry.
```bash
docker build -t <registry>/resolveiq-backend:1.0.0 backend && docker build -t <registry>/resolveiq-frontend:1.0.0 frontend && docker push ...
# the image must contain the seed data at /app/data (k8s/local/Dockerfile shows how); edit image names, host, CORS_ORIGINS and
# OLLAMA_BASE_URL in k8s/base/*.yaml; create the secret (see k8s/base/secret.example.yaml)
kubectl apply -f k8s/base/namespace.yaml
kubectl apply -f k8s/base/configmap.yaml                      # the migrate Job reads it
kubectl apply -f k8s/base/migrate-job.yaml && kubectl -n resolveiq wait --for=condition=complete job/resolveiq-migrate --timeout=15m
kubectl apply -k k8s/base/
python deploy/smoke_test.py https://support.example.com --api-key <key>
```
Tried on a local kind cluster first: `bash k8s/local/deploy.sh --ingress` and `python k8s/local/e2e_test.py --ingress` ([kubernetes.md](kubernetes.md)).
Included: Deployments with startup/readiness/liveness probes, rolling updates (`maxUnavailable: 0`, `preStop` delay), HPA (3-12 on CPU),
PodDisruptionBudget, topology spread, restricted pod security (non-root, read-only root fs, no privilege escalation, dropped
capabilities), default-deny NetworkPolicies, TLS ingress, per-release migration Job. Scrape `/metrics` with the bearer token
(pod annotations are provided); import `infra/grafana/dashboards/resolveiq.json`.

## 3. CI
`.github/workflows/ci.yml`: backend tests against a pgvector service, frontend build, production-compose validation, image
builds, manifest parsing.

## 4. Known deployment limits (be explicit)
* Single-host Compose has no HA: the host, one Postgres and one Redis are single points of failure. Use Kubernetes + managed
  data stores for HA, and PgBouncer when replicas x pool size approaches `max_connections`.
* Authentication is API-key based (per agent / service). Put the UI behind your SSO / VPN for user-level identity; OIDC is future work.
* The LLM (Qwen3-4B) needs GPU for ~5 s answers; on CPU expect much slower generation - size accordingly or use a hosted endpoint.
* No automated alert rules ship with the repo; the Grafana dashboard and metrics are provided, alert thresholds are yours.
