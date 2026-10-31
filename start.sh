#!/usr/bin/env bash
# Container entrypoint: refuse to run without a token, keep models on the network volume when there is one.
set -euo pipefail

TOKEN="${AGENT_TOKEN:-}"
if [ "${#TOKEN}" -lt 16 ]; then
  echo "AGENT_TOKEN is missing or shorter than 16 characters; refusing to start." >&2
  exit 1
fi

# Without a network volume /workspace is container disk too; models then download again on every new pod.
mkdir -p "${AGENT_MODEL_DIR}" "${AGENT_DATA_DIR}"

cd /opt/agent
exec /opt/venv-agent/bin/python -m gpu_agent.app
