#!/usr/bin/env bash
# Installiert die Caddy-Site eines Projekts im gemeinsamen Proxy /opt/caddy und
# legt Proxy und Docker-Netz `caddy` an, falls sie fehlen. Läuft als root auf dem
# Server, idempotent. Jedes Projekt führt eine identische Kopie dieses Kits mit,
# damit es einen frischen Server allein einrichten kann.
#
#   caddy-site.sh <projekt> <site-datei>
#
# Eine ungültige Site wird nicht übernommen: Validierung vor dem Neuladen, bei
# Fehlern bleibt bzw. wird der vorherige Stand wiederhergestellt.
set -Eeuo pipefail

[[ $# -eq 2 ]] || { printf 'Aufruf: %s <projekt> <site-datei>\n' "$0" >&2; exit 2; }
name=$1
site=$2
[[ "$name" =~ ^[a-z0-9][a-z0-9-]*$ ]] || { printf 'Ungültiger Projektname: %s\n' "$name" >&2; exit 2; }
[[ -f "$site" ]] || { printf 'Site-Datei fehlt: %s\n' "$site" >&2; exit 2; }

kit=$(cd "$(dirname "$0")" && pwd)
root=${CADDY_ROOT:-/opt/caddy}

docker network inspect caddy >/dev/null 2>&1 || docker network create caddy >/dev/null
install -d -m 0755 "$root" "$root/sites" "$root/data" "$root/config"
# Ein bestehender Proxy bleibt, wie er ist; das Kit legt ihn nur an.
[[ -f "$root/compose.yaml" ]] || install -m 0644 "$kit/compose.yaml" "$root/compose.yaml"
[[ -f "$root/Caddyfile" ]] || install -m 0644 "$kit/Caddyfile" "$root/Caddyfile"
docker compose -f "$root/compose.yaml" up --detach --wait

caddy() { docker exec caddy caddy "$@" --config /etc/caddy/Caddyfile --adapter caddyfile; }
target="$root/sites/$name.caddy"
previous=$(mktemp)
trap 'rm -f "$previous"' EXIT
had_previous=false
if [[ -f "$target" ]]; then cp -p "$target" "$previous"; had_previous=true; fi
if [[ "$had_previous" == true ]] && cmp -s "$site" "$target"; then
  printf 'Caddy-Site %s unverändert.\n' "$name"
  exit 0
fi

install -m 0644 "$site" "$target"
if caddy validate >/dev/null && caddy reload; then
  printf 'Caddy-Site %s installiert: %s\n' "$name" "$target"
  exit 0
fi
printf 'Caddy-Site %s ist ungültig; vorheriger Stand wird wiederhergestellt.\n' "$name" >&2
if [[ "$had_previous" == true ]]; then cp -p "$previous" "$target"; else rm -f "$target"; fi
caddy reload || true
exit 1
