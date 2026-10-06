#!/usr/bin/env bash
# Deploy ResolveIQ to a local kind cluster, end to end and idempotently (re-running updates the images and manifests in place).
#
#   bash k8s/local/deploy.sh [--no-build] [--ingress]
#
# Needs: docker, kind, kubectl, python 3. Memory: the Docker VM needs ~12 GB free (every API/worker pod holds ~2.4 GB of models).
# What it does, in the order that matters (see docs/kubernetes.md for why):
#   1. kind cluster `resolveiq` (skipped if it exists)           5. in-cluster Postgres + Redis + NetworkPolicies
#   2. build backend (+ baked seed data) and frontend images     6. ConfigMap, then the migrate + seed Job (waits for it)
#   3. load the images into the node                             7. API, worker, frontend, Jaeger, HPA, Ingress
#   4. random Secrets (never written to disk, never committed)   8. optional: ingress-nginx (--ingress)
set -euo pipefail
cd "$(dirname "$0")/../.."
CTX=kind-resolveiq
K="kubectl --context $CTX -n resolveiq"
BUILD=1 INGRESS=0
for a in "$@"; do case "$a" in --no-build) BUILD=0 ;; --ingress) INGRESS=1 ;; *) echo "unknown option $a" >&2; exit 2 ;; esac; done

kind get clusters | grep -qx resolveiq || kind create cluster --config k8s/local/kind-config.yaml --wait 180s

if [ "$BUILD" = 1 ]; then
  docker build -t resolveiq-backend:kind-base ./backend
  docker build -f k8s/local/Dockerfile --build-arg BASE=resolveiq-backend:kind-base -t resolveiq-backend:kind data   # context = data/: baked in as /app/data
  docker build -t resolveiq-frontend:kind ./frontend
fi
kind load docker-image resolveiq-backend:kind resolveiq-frontend:kind --name resolveiq

kubectl --context $CTX apply -f k8s/base/namespace.yaml
python k8s/local/make_secrets.py
kubectl --context $CTX apply -f k8s/local/postgres.yaml -f k8s/local/redis.yaml -f k8s/base/networkpolicy.yaml -f k8s/local/networkpolicy-local.yaml
$K rollout status statefulset/postgres --timeout=300s
$K rollout status deployment/redis --timeout=120s

# the migrate Job needs the (overlay-patched) ConfigMap before the rest of the stack exists: apply just that document first
kubectl --context $CTX kustomize k8s/local | python -c "
import sys
docs = sys.stdin.read().split('\n---\n')
print('\n---\n'.join(d for d in docs if 'kind: ConfigMap' in d and 'name: resolveiq-config' in d))" | kubectl --context $CTX apply -f -
$K delete job resolveiq-migrate --ignore-not-found --wait=true
sed 's#registry.example.com/resolveiq-backend:1.0.0#resolveiq-backend:kind#' k8s/base/migrate-job.yaml | kubectl --context $CTX apply -f -
$K wait --for=condition=complete job/resolveiq-migrate --timeout=900s

kubectl --context $CTX apply -k k8s/local
# an image rebuilt under the same tag is only picked up by pods that restart
$K rollout restart deployment/backend && $K rollout status deployment/backend --timeout=600s
$K rollout restart deployment/worker && $K rollout status deployment/worker --timeout=300s
$K rollout status deployment/frontend --timeout=120s
$K rollout status deployment/jaeger --timeout=120s

if [ "$INGRESS" = 1 ]; then
  kubectl --context $CTX apply -f https://kind.sigs.k8s.io/examples/ingress/deploy-ingress-nginx.yaml
  kubectl --context $CTX -n ingress-nginx wait --for=condition=ready pod -l app.kubernetes.io/component=controller --timeout=240s
fi
$K get pods,svc
echo "deployed. test: python k8s/local/e2e_test.py $([ "$INGRESS" = 1 ] && echo --ingress)"
