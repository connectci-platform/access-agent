#!/usr/bin/env bash
# Weekly analytics report — run via cron on production server
# Crontab entry: 0 14 * * 1 /home/github/access-agent/scripts/send-weekly-report.sh
# (14:00 UTC = 9am ET / 10am EDT)
set -euo pipefail

cd /home/github/access-agent

docker compose -f docker-compose.prod.yml exec -T access-agent \
  python -m src.reports weekly --email 2>&1 | logger -t access-reports
