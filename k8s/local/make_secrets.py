"""Create the Kubernetes Secrets for the LOCAL kind cluster with freshly generated random values.

    python k8s/local/make_secrets.py [--context kind-resolveiq]

Nothing is written to disk or printed: the values exist only in the cluster (read them back with `kubectl get secret ... -o jsonpath`).
By default existing secrets are left alone (the Postgres password is only applied on first initialisation, so silently rotating it would lock the
API out). `--rotate` replaces every value: also delete the postgres PVC so the database is re-initialised, and restart the pods.
Production secrets come from your secret manager instead.
"""
import argparse
import json
import secrets
import subprocess

NS = "resolveiq"


def apply(kubectl: list[str], name: str, data: dict[str, str]) -> None:
    manifest = {"apiVersion": "v1", "kind": "Secret", "metadata": {"name": name, "namespace": NS}, "type": "Opaque", "stringData": data}
    subprocess.run([*kubectl, "apply", "-f", "-"], input=json.dumps(manifest), text=True, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", default="kind-resolveiq")
    ap.add_argument("--rotate", action="store_true", help="replace secrets that already exist")
    a = ap.parse_args()
    kubectl = ["kubectl", "--context", a.context]
    if not a.rotate and subprocess.run([*kubectl, "-n", NS, "get", "secret", "resolveiq-secrets"], capture_output=True).returncode == 0:
        print("secrets already exist; leaving them alone (use --rotate to replace)")
        return
    owner_pw, app_pw, redis_pw = (secrets.token_urlsafe(24) for _ in range(3))
    host = f"postgres.{NS}.svc.cluster.local"
    apply(kubectl, "resolveiq-postgres", {"POSTGRES_PASSWORD": owner_pw, "APP_DB_PASSWORD": app_pw})
    apply(kubectl, "resolveiq-redis", {"REDIS_PASSWORD": redis_pw})
    apply(kubectl, "resolveiq-secrets", {
        "DATABASE_URL": f"postgresql://resolveiq_app:{app_pw}@{host}:5432/resolveiq",
        "DATABASE_ADMIN_URL": f"postgresql://resolveiq_owner:{owner_pw}@{host}:5432/resolveiq",
        "REDIS_URL": f"redis://:{redis_pw}@redis.{NS}.svc.cluster.local:6379/0",
        "API_KEYS": secrets.token_urlsafe(32),
        "METRICS_TOKEN": secrets.token_urlsafe(32),
    })
    print("secrets applied: resolveiq-secrets, resolveiq-postgres, resolveiq-redis")


if __name__ == "__main__":
    main()
