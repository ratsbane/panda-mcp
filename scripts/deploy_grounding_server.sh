#!/bin/bash
# Deploy grounding server as a systemd service on Spark.
#
# Run from the Pi: bash scripts/deploy_grounding_server.sh
#
# Prerequisites: SSH access to spark as user 'claude', sudo on spark.

set -euo pipefail

SPARK_HOST="spark"
SPARK_USER="claude"
REPO_DIR="/home/claude/panda-mcp"
SERVICE_NAME="grounding-server"

echo "=== Deploying grounding server to ${SPARK_HOST} ==="

# 1. Sync the updated server script
echo "1. Syncing grounding_server.py..."
scp scripts/grounding_server.py "${SPARK_USER}@${SPARK_HOST}:${REPO_DIR}/scripts/"

# 2. Copy service file and install
echo "2. Installing systemd service..."
scp scripts/grounding-server.service "${SPARK_USER}@${SPARK_HOST}:/tmp/"
ssh "${SPARK_USER}@${SPARK_HOST}" "
    sudo cp /tmp/grounding-server.service /etc/systemd/system/${SERVICE_NAME}.service
    sudo systemctl daemon-reload
    sudo systemctl enable ${SERVICE_NAME}
"

# 3. Stop any manually-running instances
echo "3. Stopping stale instances..."
ssh "${SPARK_USER}@${SPARK_HOST}" "
    pkill -f 'grounding_server.py' 2>/dev/null || true
    sleep 1
"

# 4. Start the service
echo "4. Starting service..."
ssh "${SPARK_USER}@${SPARK_HOST}" "
    sudo systemctl restart ${SERVICE_NAME}
    sleep 5
    sudo systemctl status ${SERVICE_NAME} --no-pager || true
"

# 5. Health check
echo "5. Health check..."
sleep 20  # Wait for model to load (~15s)
if curl -sf "http://${SPARK_HOST}:8090/health" | python3 -m json.tool; then
    echo ""
    echo "=== Deployment successful! ==="
else
    echo ""
    echo "=== Health check failed (model may still be loading) ==="
    echo "    Check logs: ssh spark 'journalctl -u grounding-server -f'"
fi
