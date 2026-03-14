#!/usr/bin/env bash
# Set up a GPU VM for running the embedding pipeline.
#
# Usage (from the main machine):
#   bash scripts/setup_gpu_vm.sh [VM_NAME] [ZONE]
#
# Copies code, installs deps, verifies GPU + DB connectivity.
# After this completes, SSH in and run:
#   cd ~/sambandh-preprocessing && python scripts/run_embedding.py --device cuda --fp16

set -euo pipefail

VM_NAME="${1:-sambandh-embed-gpu}"
ZONE="${2:-asia-south1-a}"
PROJECT="sulekha-487304"
MAIN_IP=$(hostname -I | awk '{print $1}')
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== GPU VM Setup ==="
echo "  VM:       ${VM_NAME} (${ZONE})"
echo "  Main IP:  ${MAIN_IP}"
echo "  Repo:     ${REPO_DIR}"
echo ""

# Copy code to VM (tar to exclude .venv, .git, __pycache__)
echo "[1/4] Copying code to VM..."
tar czf /tmp/sambandh-code.tar.gz \
    -C "$(dirname "${REPO_DIR}")" \
    --exclude='.venv' --exclude='.git' --exclude='__pycache__' \
    --exclude='*.pyc' --exclude='uv.lock' --exclude='.secrets' \
    "$(basename "${REPO_DIR}")"

gcloud compute scp \
    --zone="${ZONE}" --project="${PROJECT}" \
    --compress --strict-host-key-checking=no \
    /tmp/sambandh-code.tar.gz "${VM_NAME}:~/sambandh-code.tar.gz"

gcloud compute ssh "${VM_NAME}" \
    --zone="${ZONE}" --project="${PROJECT}" \
    --strict-host-key-checking=no \
    --command="cd ~ && tar xzf sambandh-code.tar.gz && rm sambandh-code.tar.gz"

rm -f /tmp/sambandh-code.tar.gz

# Copy GCS credentials
CREDS_SRC="${REPO_DIR}/.secrets/gcs-credentials.json"
if [ -f "${CREDS_SRC}" ]; then
    echo "[2/4] Copying GCS credentials..."
    gcloud compute scp \
        --zone="${ZONE}" --project="${PROJECT}" \
        --strict-host-key-checking=no \
        "${CREDS_SRC}" "${VM_NAME}:~/gcs-credentials.json"
else
    echo "[2/4] No .secrets/gcs-credentials.json found, skipping (VM will use default service account)"
fi

# Install uv + deps
echo "[3/4] Installing uv and dependencies on VM..."
gcloud compute ssh "${VM_NAME}" \
    --zone="${ZONE}" --project="${PROJECT}" \
    --strict-host-key-checking=no \
    --command="
set -e

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=\$HOME/.local/bin:\$PATH

cd ~/sambandh-preprocessing

# Write .env for the embedding script
cat > .env << 'ENVEOF'
DATABASE_URL=postgresql://sambandh:sambandh@${MAIN_IP}:5433/sambandh
GCS_OUTPUT_BUCKET=parsed-sulekha
GOOGLE_APPLICATION_CREDENTIALS=/home/tusharanand/gcs-credentials.json
ENVEOF

# uv will auto-download Python 3.11+ and install deps
uv sync 2>&1 | tail -10
uv run python --version
"

# Verify setup
echo "[4/4] Verifying GPU + DB connectivity..."
gcloud compute ssh "${VM_NAME}" \
    --zone="${ZONE}" --project="${PROJECT}" \
    --strict-host-key-checking=no \
    --command="
cd ~/sambandh-preprocessing
export PATH=\$HOME/.local/bin:\$PATH
uv run python -c \"
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')

import psycopg
conn = psycopg.connect('postgresql://sambandh:sambandh@${MAIN_IP}:5433/sambandh')
row = conn.execute('SELECT COUNT(*) AS n FROM chunk_jobs WHERE status = \'done\'').fetchone()
print(f'DB connected: {row[0]} chunk_jobs done')
conn.close()
print('All checks passed.')
\"
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "SSH in and run:"
echo "  gcloud compute ssh ${VM_NAME} --zone=${ZONE} --project=${PROJECT}"
echo "  cd ~/sambandh-preprocessing"
echo "  ~/.local/bin/uv run python scripts/run_embedding.py --device cuda --fp16"
echo ""
echo "When done, delete the VM:"
echo "  gcloud compute instances delete ${VM_NAME} --zone=${ZONE} --project=${PROJECT} --quiet"
