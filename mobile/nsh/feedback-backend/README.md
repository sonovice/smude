# Smude Feedback-Backend (PocketBase)

Sammelt TestFlight-Feedback der Smude-App: Scan-Berichte (`reports`) und die
Preisumfrage (`price_votes`). Alles hier im Ordner reicht, um das Backend von
Grund auf neu zu deployen.

| | |
|---|---|
| Öffentliche URL | `https://smude.notavium.de/feedback/` |
| API | `https://smude.notavium.de/feedback/api/...` |
| Dashboard | `https://smude.notavium.de/feedback/_/` |
| Server | `root@2.31.56.235` (Ubuntu, arm64) |
| Compose-Projekt / Container | `smude-feedback` / `smude-feedback` |
| Port | `127.0.0.1:8090` (nur lokal), im Docker-Netz `smude-feedback:8090` |
| Verzeichnis | `/opt/smude-feedback` |
| Daten | `/opt/smude-feedback/pb_data` (SQLite + hochgeladene Dateien) |
| Superuser-Zugang | `/root/smude-feedback-admin.txt` auf dem Server (chmod 600, nicht im Repo) |

## Dateien

- `Dockerfile`: Alpine plus offizielles PocketBase-Release-Binary, Version und SHA256 fest eingetragen.
- `docker-compose.yml`: der Dienst, `restart: unless-stopped`, Healthcheck. Der Container hängt zusätzlich im Docker-Netz `caddy` des gemeinsamen Proxys.
- `Caddyfile.smude`: Site für `smude.notavium.de`. `handle_path /feedback/*` entfernt das Präfix, `request_body max_size 200MB`.
- `caddy/`: Kit des gemeinsamen Reverse Proxys (siehe unten). Identische Kopie liegt im egd-Repo unter `ops/caddy/`.
- `pb_migrations/`: Schema der Collections als JS-Migration. PocketBase wendet neue Migrationen beim Start an.
- `deploy.sh`: rsync auf den Server, Site im Proxy installieren, `docker compose up -d --build`.

## Reverse Proxy und TLS

Auf dem Server läuft ein gemeinsamer Caddy als eigenes Compose-Projekt unter
`/opt/caddy` (Container `caddy`, Ports 80/443, Docker-Netz `caddy`). Er gehört
keinem Projekt. Sein Caddyfile importiert nur `sites/*.caddy`; jedes Projekt
verwaltet genau eine Datei darin (`smude-feedback.caddy`, `egd.caddy`).

`caddy/caddy-site.sh <projekt> <site-datei>` legt Netz und Proxy an, falls sie
fehlen, installiert die Site, validiert und lädt Caddy ohne Neustart neu. Ist die
Site ungültig, bleibt der vorherige Stand. Weil jedes Projekt dieses Kit
mitbringt, richtet sich jedes auf einem frischen Server allein vollständig ein.
Zertifikate holt Caddy per ACME (Let's Encrypt) und speichert sie in
`/opt/caddy/data`. Der A-Eintrag von `smude.notavium.de` muss auf den Server zeigen.

Änderungen am Kit (`compose.yaml`, `Caddyfile`, `caddy-site.sh`) bitte in beiden
Repos gleich halten. Ein bestehender `/opt/caddy` wird vom Kit nicht
überschrieben; dort Änderungen von Hand übernehmen und
`docker compose -f /opt/caddy/compose.yaml up -d` ausführen.

## Deploy

```sh
./deploy.sh                    # Standard-Host root@2.31.56.235
HOST=root@anderer-host ./deploy.sh
```

Beim allerersten Deploy danach einmalig den Superuser anlegen und die Zugangsdaten
nur auf dem Server ablegen:

```sh
ssh root@2.31.56.235
PASS=$(openssl rand -base64 33 | tr -d '/+=' | cut -c1-32)
docker exec smude-feedback pocketbase superuser upsert admin@smude.notavium.de "$PASS" --dir=/pb/pb_data
umask 077; printf 'E-Mail:    admin@smude.notavium.de\nPasswort:  %s\n' "$PASS" > /root/smude-feedback-admin.txt
```

Danach im Dashboard unter *Settings*: Application URL
`https://smude.notavium.de/feedback`, unter *Trusted proxy* den Header
`X-Forwarded-For` eintragen (für korrekte Client-IPs und Rate-Limits).

## Collections

Beide sind vom Typ `base`. List, View, Update und Delete sind nur für Superuser
erlaubt (Regel `null`). Create verlangt den Header
`X-Smude-Token: c415095c1254bc2ebea4fbff1e9629dd` (Regel
`@request.headers.x_smude_token = "..."`). Das Token hält nur Gelegenheits-Posts
ab und ist kein echtes Geheimnis.

- `reports` (multipart): `install`, `status`, `error`, `app`, `device`, `page`, `capture` (JSON), `original` und `result` (je eine Datei, max. 80 MB), `created`/`updated`.
- `price_votes` (JSON): `install`, `price`, `answered`, `active_seconds`, `language`, `app`, `device`, `created`/`updated`.

Hinweis: PocketBase-Zahlenfelder kennen kein `null`. Ein leeres `price` wird als
`0` gespeichert. Ob „keine Angabe“ vorliegt, ergibt sich aus `answered`.

Schemaänderungen kommen als neue Datei nach `pb_migrations/`, danach `./deploy.sh`.
Änderungen im Dashboard erzeugt PocketBase ebenfalls als Migrationsdatei, aber
auf dem Server unter `/opt/smude-feedback/pb_migrations/`. Diese bitte ins Repo
zurückholen (`rsync root@2.31.56.235:/opt/smude-feedback/pb_migrations/ pb_migrations/`).

## PocketBase aktualisieren

1. Neue Version und `checksums.txt` unter <https://github.com/pocketbase/pocketbase/releases> nachsehen, Changelog auf Breaking Changes prüfen.
2. Im `Dockerfile` `PB_VERSION`, `PB_SHA256_ARM64` und `PB_SHA256_AMD64` anpassen, in `docker-compose.yml` den Image-Tag ebenso.
3. Vorher sichern (siehe unten), dann `./deploy.sh`.

## Daten exportieren und sichern

```sh
# Vollständige Kopie (SQLite + Dateien). Kurz stoppen, damit die SQLite-Datei konsistent ist.
ssh root@2.31.56.235 'cd /opt/smude-feedback && docker compose stop && tar czf /root/smude-feedback-$(date +%F).tgz pb_data; docker compose start'
scp root@2.31.56.235:/root/smude-feedback-*.tgz .
```

Alternativ im Dashboard unter *Settings → Backups* ein ZIP erstellen und
herunterladen. Einzelne Collections lassen sich als JSON über die API
exportieren (Superuser-Token aus `/api/collections/_superusers/auth-with-password`):

```sh
curl -s 'https://smude.notavium.de/feedback/api/collections/reports/records?perPage=500' -H "Authorization: $TOKEN"
```

Dateien liegen unter
`https://smude.notavium.de/feedback/api/files/reports/<recordId>/<dateiname>?token=<fileToken>`
(File-Token über `POST /api/files/token`) bzw. direkt in `pb_data/storage/`.

## Prüfen

```sh
curl -s https://smude.notavium.de/feedback/api/health
ssh root@2.31.56.235 'docker ps --filter name=smude-feedback; docker logs --tail 50 smude-feedback'
```
