
#!/usr/bin/env bash

set -Eeuo pipefail

# ==================================================
# CONFIGURATION
# ==================================================

export PORT="${PORT:-8080}"
export DISPLAY="${DISPLAY:-:99}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export QT_X11_NO_MITSHM=1
export QT_DEBUG_PLUGINS="${QT_DEBUG_PLUGINS:-0}"

: "${VNC_PASSWORD:?ERROR: VNC_PASSWORD must be configured in Heroku Config Vars}"

if [ "${#VNC_PASSWORD}" -lt 8 ]; then
    echo "ERROR: VNC_PASSWORD must contain at least 8 characters." >&2
    exit 1
fi

echo "=========================================="
echo "Starting ProphitBet Web Application"
echo "=========================================="

echo "PORT=$PORT"
echo "DISPLAY=$DISPLAY"
echo "QT_QPA_PLATFORM=$QT_QPA_PLATFORM"

# ==================================================
# DIRECTORIES
# ==================================================

mkdir -p /tmp/prophitbet

# ==================================================
# PROCESS VARIABLES
# ==================================================

XVFB_PID=""
VNC_PID=""
WEBSOCKIFY_PID=""
APP_PID=""

CLEANUP_DONE=false

# ==================================================
# CLEANUP
# ==================================================

cleanup() {
    if [ "$CLEANUP_DONE" = true ]; then
        return
    fi

    CLEANUP_DONE=true

    echo "Stopping application processes..."

    for pid in \
        "$APP_PID" \
        "$WEBSOCKIFY_PID" \
        "$VNC_PID" \
        "$XVFB_PID"; do

        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done

    for pid in \
        "$APP_PID" \
        "$WEBSOCKIFY_PID" \
        "$VNC_PID" \
        "$XVFB_PID"; do

        if [ -n "$pid" ]; then
            wait "$pid" 2>/dev/null || true
        fi
    done

    echo "Cleanup completed."
}

trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

# ==================================================
# VNC PASSWORD
# ==================================================

echo "Configuring VNC password..."

x11vnc -storepasswd \
    "$VNC_PASSWORD" \
    /tmp/prophitbet/vnc.pass >/dev/null

chmod 600 /tmp/prophitbet/vnc.pass

# ==================================================
# START XVFB
# ==================================================

echo "Starting Xvfb..."

Xvfb "$DISPLAY" \
    -screen 0 "${SCREEN_SIZE:-1600x1000x24}" \
    -ac \
    +extension GLX \
    +render \
    -noreset \
    >/tmp/prophitbet/xvfb.log 2>&1 &

XVFB_PID=$!

echo "Xvfb PID: $XVFB_PID"

# ==================================================
# WAIT FOR XVFB
# ==================================================

echo "Waiting for Xvfb..."

XVFB_READY=false

for i in $(seq 1 30); do

    if ! kill -0 "$XVFB_PID" 2>/dev/null; then
        echo "ERROR: Xvfb process terminated."
        cat /tmp/prophitbet/xvfb.log
        exit 1
    fi

    if [ -S /tmp/.X11-unix/X99 ]; then
        XVFB_READY=true
        break
    fi

    sleep 1
done

if [ "$XVFB_READY" != true ]; then
    echo "ERROR: Xvfb failed to start."
    cat /tmp/prophitbet/xvfb.log
    exit 1
fi

echo "Xvfb started successfully."

# ==================================================
# VERIFY QT / XCB
# ==================================================

echo "Testing Qt XCB initialization..."

python - <<'PY'
import sys

from PySide6.QtCore import QLibraryInfo, qVersion
from PySide6.QtWidgets import QApplication

print("Qt version:", qVersion())

print(
    "Qt plugin directory:",
    QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)
)

try:
    app = QApplication([])
    print("Qt XCB initialization successful.")
    app.quit()

except Exception as exc:
    print("Qt initialization failed:", str(exc))
    sys.exit(1)
PY

echo "Qt verification completed."

# ==================================================
# START X11VNC
# ==================================================

echo "Starting X11VNC..."

x11vnc \
    -display "$DISPLAY" \
    -forever \
    -shared \
    -localhost \
    -rfbport 5900 \
    -rfbauth /tmp/prophitbet/vnc.pass \
    -noxdamage \
    -repeat \
    >/tmp/prophitbet/x11vnc.log 2>&1 &

VNC_PID=$!

sleep 2

if ! kill -0 "$VNC_PID" 2>/dev/null; then
    echo "ERROR: X11VNC failed to start."
    cat /tmp/prophitbet/x11vnc.log
    exit 1
fi

echo "X11VNC started successfully."

# ==================================================
# START NOVNC / WEBSOCKIFY
# ==================================================

echo "Starting noVNC..."

if [ ! -d /usr/share/novnc ]; then
    echo "ERROR: noVNC directory not found."
    exit 1
fi

websockify \
    --web=/usr/share/novnc \
    --heartbeat=30 \
    "$PORT" \
    localhost:5900 \
    >/tmp/prophitbet/websockify.log 2>&1 &

WEBSOCKIFY_PID=$!

sleep 2

if ! kill -0 "$WEBSOCKIFY_PID" 2>/dev/null; then
    echo "ERROR: Websockify failed to start."
    cat /tmp/prophitbet/websockify.log
    exit 1
fi

echo "noVNC started successfully."

# ==================================================
# VERIFY ORIGINAL APPLICATION
# ==================================================

if [ ! -f /opt/prophitbet/main.py ]; then
    echo "ERROR: main.py not found."
    exit 1
fi

# ==================================================
# START PROPHITBET
# ==================================================

echo "=========================================="
echo "Starting original ProphitBet GUI"
echo "=========================================="

cd /opt/prophitbet

python -u /opt/prophitbet/main.py &

APP_PID=$!

echo "Application PID: $APP_PID"
echo "Web interface port: $PORT"

# ==================================================
# MONITOR PROCESSES
# ==================================================

echo "Monitoring application processes..."

set +e

wait -n \
    "$APP_PID" \
    "$VNC_PID" \
    "$WEBSOCKIFY_PID" \
    "$XVFB_PID"

EXIT_CODE=$?

set -e

echo "=========================================="
echo "ERROR: One of the processes exited."
echo "Exit code: $EXIT_CODE"
echo "=========================================="

echo "Xvfb log:"
cat /tmp/prophitbet/xvfb.log 2>/dev/null || true

echo "X11VNC log:"
cat /tmp/prophitbet/x11vnc.log 2>/dev/null || true

echo "Websockify log:"
cat /tmp/prophitbet/websockify.log 2>/dev/null || true

exit "$EXIT_CODE"
