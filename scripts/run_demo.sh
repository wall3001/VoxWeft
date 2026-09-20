#!/bin/bash
# Start bridge (:8133) + serve (:3100) with the seed voices. Adjust the model
# paths for your machine or set VOXWEFT_MODEL_DIR and rely on the defaults.
set -u
cd "$(dirname "$0")/.."
P=${PYTHON:-$(command -v voxweft >/dev/null && dirname "$(command -v voxweft)")/python}
P=${P:-python3}
export VOXWEFT_BRIDGE_URL=${VOXWEFT_BRIDGE_URL:-http://127.0.0.1:8133}
pkill -f "voxweft.cli (bridge|serve)" 2>/dev/null; sleep 1
PYTHONPATH=. nohup "$P" -m voxweft.cli bridge --port 8133 > /tmp/vw_bridge.log 2>&1 &
PYTHONPATH=. nohup "$P" -m voxweft.cli serve  --port 3100 > /tmp/vw_serve.log  2>&1 &
for i in $(seq 1 100); do
  c=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8133/ready 2>/dev/null)
  [ "$c" = "200" ] && { echo "bridge ready (~$((i*3))s)"; break; }; sleep 3
done
for i in $(seq 1 40); do curl -s http://127.0.0.1:3100/health >/dev/null 2>&1 && { echo "serve up"; break; }; sleep 2; done
curl -s http://127.0.0.1:8133/voices | "$P" -c "import sys,json; d=json.load(sys.stdin); print('voices:', len(d['voices']))"
echo "ws://127.0.0.1:3100/ws/voice/v2   logs: /tmp/vw_bridge.log /tmp/vw_serve.log"
