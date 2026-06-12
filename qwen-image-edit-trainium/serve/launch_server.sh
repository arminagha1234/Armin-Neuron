#!/bin/bash
# Launch the FastAPI front-end. Run this AFTER launch_worker.sh has
# printed "pipeline ready, entering serve loop".
#
# Usage:
#   bash /work/path_c/serve/launch_server.sh [PORT]   # default 8000
set -euo pipefail

PORT="${1:-8000}"

source /opt/torch-neuronx/.venv/bin/activate
cd /work/path_c/serve

# fastapi + uvicorn ship in the Beta 2 DLC venv already; if not, install.
python -c "import fastapi, uvicorn" 2>/dev/null || \
    pip install --quiet fastapi uvicorn

export FAL_SOCKET_PATH=/tmp/fal_pipeline.sock
export FAL_REQUEST_TIMEOUT_S=1500

exec uvicorn server:app --host 0.0.0.0 --port "${PORT}" --workers 1 --log-level info
