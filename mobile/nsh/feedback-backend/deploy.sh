#!/usr/bin/env bash
# Deployt das Smude-Feedback-Backend (PocketBase) und seine Site im gemeinsamen
# Caddy. Idempotent; richtet auf einem frischen Server auch den Proxy ein.
set -euo pipefail

HOST="${HOST:-root@2.31.56.235}"
DEST=/opt/smude-feedback
HERE="$(cd "$(dirname "$0")" && pwd)"

ssh "$HOST" "mkdir -p $DEST/pb_data $DEST/pb_migrations $DEST/caddy"
# Ohne --delete: im Dashboard erzeugte Migrationen auf dem Server bleiben erhalten.
rsync -a "$HERE/Dockerfile" "$HERE/docker-compose.yml" "$HERE/Caddyfile.smude" "$HOST:$DEST/"
rsync -a "$HERE/pb_migrations/" "$HOST:$DEST/pb_migrations/"
rsync -a --delete "$HERE/caddy/" "$HOST:$DEST/caddy/"

ssh "$HOST" DEST="$DEST" bash -s <<'REMOTE'
set -euo pipefail
cd "$DEST"
# Zuerst Proxy und Netz `caddy`: der Dienst hängt sich als externes Netz daran.
caddy/caddy-site.sh smude-feedback Caddyfile.smude
docker compose up -d --build --remove-orphans
for _ in $(seq 1 30); do
  [ "$(docker inspect -f '{{.State.Health.Status}}' smude-feedback)" = healthy ] && break
  sleep 2
done
docker inspect -f 'smude-feedback: {{.State.Health.Status}}' smude-feedback
REMOTE
curl -fsS https://smude.notavium.de/feedback/api/health >/dev/null && echo "https://smude.notavium.de/feedback/ ist erreichbar"
