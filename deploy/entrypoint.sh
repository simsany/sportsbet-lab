#!/usr/bin/env bash
set -Eeuo pipefail
: "${PORT:=8080}"
: "${VNC_PASSWORD:?Set VNC_PASSWORD in Heroku Config Vars (at least 8 characters).}"
if [ "${#VNC_PASSWORD}" -lt 8 ]; then echo "VNC_PASSWORD must contain at least 8 characters." >&2; exit 1; fi
mkdir -p /tmp/prophitbet
x11vnc -storepasswd "$VNC_PASSWORD" /tmp/prophitbet/vnc.pass >/dev/null
Xvfb :99 -screen 0 "${SCREEN_SIZE:-1600x1000x24}" -ac +extension GLX +render -noreset >/tmp/prophitbet/xvfb.log 2>&1 &
XVFB_PID=$!
cleanup() { kill "${APP_PID:-}" "${VNC_PID:-}" "${WEBSOCKIFY_PID:-}" "${XVFB_PID:-}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
for _ in $(seq 1 30); do [ -S /tmp/.X11-unix/X99 ] && break; sleep 1; done
[ -S /tmp/.X11-unix/X99 ] || { echo "Xvfb failed; inspect /tmp/prophitbet/xvfb.log" >&2; exit 1; }
x11vnc -display :99 -forever -shared -localhost -rfbport 5900 -rfbauth /tmp/prophitbet/vnc.pass -noxdamage -repeat >/tmp/prophitbet/x11vnc.log 2>&1 &
VNC_PID=$!
websockify --web=/usr/share/novnc --heartbeat=30 "$PORT" localhost:5900 >/tmp/prophitbet/websockify.log 2>&1 &
WEBSOCKIFY_PID=$!
sleep 2
echo "Starting original ProphitBet GUI on virtual display :99"
python /app/main.py &
APP_PID=$!
wait "$APP_PID"
