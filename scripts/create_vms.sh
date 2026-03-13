#!/usr/bin/env bash
# Create N worker VMs for distributed OCR processing.
#
# Usage:
#   bash scripts/create_vms.sh            # Create 1 VM (for testing)
#   bash scripts/create_vms.sh 9          # Create 9 VMs (production)
#
# Prerequisites:
#   - GCP firewall rule: allow-sambandh-workers (tcp:5433,6380 from VPC)
#   - parsed-sulekha GCS bucket exists
#   - This machine's docker-compose is running (Postgres + Redis)

set -euo pipefail

NUM_VMS="${1:-1}"
PROJECT="sulekha-487304"
ZONE="asia-south1-c"
MACHINE_TYPE="n2-standard-8"
IMAGE_FAMILY="debian-12"
IMAGE_PROJECT="debian-cloud"
DISK_SIZE="50GB"

# Internal IP of this machine (the one running Postgres + Redis)
MAIN_IP=$(hostname -I | awk '{print $1}')
echo "Main machine internal IP: ${MAIN_IP}"

# GCS credentials path on the new VMs -- uploaded from this machine
CREDS_SRC="/home/tusharanand/sulekha-scraper/credentials/sulekha-487304-4f5a021dcc1c.json"

STARTUP_SCRIPT=$(cat <<'SCRIPT_EOF'
#!/bin/bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

# System deps
apt-get update -qq
apt-get install -y -qq \
    tesseract-ocr tesseract-ocr-mal tesseract-ocr-eng \
    python3.11 python3.11-venv python3-pip \
    git curl libglib2.0-0 libgl1-mesa-glx libsm6 libxext6 libxrender1

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Clone repo
cd /opt
git clone https://github.com/YOUR_ORG/sambandh-preprocessing.git || true
cd sambandh-preprocessing

# Install Python deps
uv venv
uv pip install -e .
uv pip install celery[redis] psutil psycopg[binary] google-cloud-storage

# Write env file
cat > .env <<ENV
DATABASE_URL=postgresql://sambandh:sambandh@MAIN_MACHINE_IP:5433/sambandh
REDIS_URL=redis://MAIN_MACHINE_IP:6380/0
GOOGLE_APPLICATION_CREDENTIALS=/opt/gcs-key.json
GCS_OUTPUT_BUCKET=parsed-sulekha
ENV

# Start Celery worker
VM_NAME=$(hostname)
cd /opt/sambandh-preprocessing
source .venv/bin/activate

exec celery -A sambandh.celery_app worker \
    --concurrency=8 \
    --pool=prefork \
    -Q ocr \
    -n "worker-${VM_NAME}@%h" \
    --loglevel=INFO
SCRIPT_EOF
)

# Replace placeholder with actual IP
STARTUP_SCRIPT="${STARTUP_SCRIPT//MAIN_MACHINE_IP/${MAIN_IP}}"

echo "Creating ${NUM_VMS} worker VM(s)..."
echo ""

for i in $(seq 1 "$NUM_VMS"); do
    VM_NAME="sambandh-ocr-worker-${i}"
    echo "Creating ${VM_NAME}..."

    gcloud compute instances create "${VM_NAME}" \
        --project="${PROJECT}" \
        --zone="${ZONE}" \
        --machine-type="${MACHINE_TYPE}" \
        --boot-disk-size="${DISK_SIZE}" \
        --image-family="${IMAGE_FAMILY}" \
        --image-project="${IMAGE_PROJECT}" \
        --scopes=storage-full \
        --metadata-from-file=startup-script=<(echo "${STARTUP_SCRIPT}") \
        --tags=sambandh-worker \
        --no-address \
        2>&1 | tail -1

    # Copy GCS credentials to the VM
    echo "  Copying GCS credentials..."
    gcloud compute scp "${CREDS_SRC}" "${VM_NAME}:/opt/gcs-key.json" \
        --zone="${ZONE}" --project="${PROJECT}" 2>/dev/null || \
        echo "  (credentials copy will be retried after VM boots)"

    echo "  ${VM_NAME} created."
    echo ""
done

echo "Done. ${NUM_VMS} VM(s) created."
echo ""
echo "Monitor with:"
echo "  gcloud compute instances list --filter='name~sambandh-ocr-worker' --project=${PROJECT}"
echo ""
echo "SSH into a worker:"
echo "  gcloud compute ssh sambandh-ocr-worker-1 --zone=${ZONE} --project=${PROJECT}"
echo ""
echo "Delete all workers when done:"
echo "  for i in \$(seq 1 ${NUM_VMS}); do"
echo "    gcloud compute instances delete sambandh-ocr-worker-\$i --zone=${ZONE} --project=${PROJECT} --quiet"
echo "  done"
