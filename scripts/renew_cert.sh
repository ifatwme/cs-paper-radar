#!/usr/bin/env bash
# Obtain/renew the Let's Encrypt certificate for this site and hot-reload nginx.
#
# Uses lego's built-in HTTP-01 server: host port 80 is opened only for the few
# seconds of a renewal, so nothing needs to listen on :80 permanently.
# Settings come from .env (never committed):
#   CERT_IDENTIFIER   server IP address or domain the certificate is for (required)
#   CERT_PROFILE      Let's Encrypt profile; "shortlived" (~6 days) is required for IP addresses
#   ACME_SERVER       default: Let's Encrypt production; use the staging URL to test
#   LEGO_EXTRA        extra lego flags
# Usage: scripts/renew_cert.sh
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f .env ]] && set -a && . ./.env && set +a
ID="${CERT_IDENTIFIER:?set CERT_IDENTIFIER (server IP or domain) in .env}"
PROFILE="${CERT_PROFILE:-shortlived}"
SERVER="${ACME_SERVER:-https://acme-v02.api.letsencrypt.org/directory}"
CERTS="${CERTS_DIR:-deploy/certs}"
TLS="${TLS_DIR:-deploy/tls}"
mkdir -p "$CERTS" "$TLS"

# lego v5 `run` obtains the certificate, or renews it when due (short-lived: at half its lifetime).
docker run --rm -p 80:80 -v "$PWD/$CERTS:/certs" goacme/lego:latest run \
  --accept-tos --server "$SERVER" --domains "$ID" \
  --http --path /certs --profile "$PROFILE" ${LEGO_EXTRA:-}

install -o 101 -g 101 -m 0444 "$CERTS/certificates/$ID.crt" "$TLS/fullchain.pem"   # 101 = nginx user
install -o 101 -g 101 -m 0400 "$CERTS/certificates/$ID.key" "$TLS/privkey.pem"
chmod 755 "$TLS"
if docker compose ps --status running --services 2>/dev/null | grep -qx web; then
  docker compose exec -T web nginx -s reload
fi
echo "[$(date -u +%FT%TZ)] certificate valid until: $(openssl x509 -in "$TLS/fullchain.pem" -noout -enddate)"
