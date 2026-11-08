#!/bin/sh
# Recreates Caddy when the config it can see differs from ./Caddyfile.
#
# `docker compose up -d` doesn't notice a change to a bind-mounted file. A reload
# wouldn't help either: git replaces the file rather than rewriting it, and the
# mount stays on the old inode, so the container keeps seeing the old content.
# That stale view is also what this compares, so nothing happens on a deploy
# with no config change, and a deploy that failed halfway is caught by the next.
#
# Usage: recreate-caddy-if-config-changed.sh <compose file>
set -u

compose_file=$1
wanted=$(sha256sum Caddyfile | cut -d' ' -f1)
running=$(docker compose -f "$compose_file" exec -T caddy sha256sum /etc/caddy/Caddyfile 2>/dev/null | cut -d' ' -f1)

if [ -n "$wanted" ] && [ "$wanted" = "$running" ]; then
  echo "Caddy config unchanged"
  exit 0
fi

echo "Caddy config changed, recreating Caddy..."
docker compose -f "$compose_file" up -d --force-recreate --no-deps caddy
