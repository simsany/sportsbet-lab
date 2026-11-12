#!/usr/bin/env sh
set -eu
cd /opt/prophitbet
python -c "from web.postgres_store import initialize; initialize()"
python -c "from web.recovery import recover_if_needed; recover_if_needed()"
exec gunicorn --bind "0.0.0.0:${PORT:-8080}" --workers 1 --threads 4 --timeout 1800 --access-logfile - --error-logfile - web.app:app
