#!/bin/sh
# Logical backup of the production database (tickets, KB, taxonomy, audit trail). Embeddings are included.
#   deploy/backup.sh                  -> backups/resolveiq-YYYYmmdd-HHMMSS.dump (custom format, compressed)
#   deploy/backup.sh restore FILE     -> restores into the running db service (DESTRUCTIVE: replaces objects)
# Schedule with cron / a systemd timer and copy the files off-host; test restores regularly.
set -eu
cd "$(dirname "$0")/.."
DC="docker compose --env-file .env.prod -f docker-compose.prod.yml"
mkdir -p backups
if [ "${1:-}" = "restore" ]; then
  [ -f "${2:?usage: backup.sh restore FILE}" ] || { echo "no such file"; exit 1; }
  $DC exec -T db sh -c 'pg_restore --clean --if-exists --no-owner -U "$POSTGRES_USER" -d "$POSTGRES_DB"' < "$2"
  echo "restored $2"
else
  OUT="backups/resolveiq-$(date +%Y%m%d-%H%M%S).dump"
  $DC exec -T db sh -c 'pg_dump -Fc -U "$POSTGRES_USER" "$POSTGRES_DB"' > "$OUT"
  echo "wrote $OUT ($(wc -c < "$OUT") bytes)"
  # keep the 14 most recent
  ls -1t backups/resolveiq-*.dump | tail -n +15 | xargs -r rm -f
fi
