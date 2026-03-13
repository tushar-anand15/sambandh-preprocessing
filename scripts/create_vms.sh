#!/usr/bin/env bash
# Create worker VMs for distributed OCR processing using Container-Optimized OS.
#
# Usage:
#   bash scripts/create_vms.sh            # Create 1 VM (for testing)
#   bash scripts/create_vms.sh 2          # Create 2 VMs
#
# Prerequisites:
#   - GCP firewall rule allowing tcp:5433,6380 from worker VMs
#   - Image pushed to Artifact Registry
#   - Main VM running Postgres + Redis

set -euo pipefail

NUM_VMS="${1:-1}"
PROJECT="sulekha-487304"
ZONE="asia-south1-c"
MACHINE_TYPE="n2-standard-16"
DISK_SIZE="50GB"

# Artifact Registry image
IMAGE_URL="asia-south1-docker.pkg.dev/sulekha-487304/sambandh/worker:latest"

# Internal IP of this machine (the one running Postgres + Redis)
MAIN_IP=$(hostname -I | awk '{print $1}')
echo "Main machine internal IP: ${MAIN_IP}"

# GCS credentials - will be copied to worker VMs
CREDS_SRC="${CREDS_SRC:-/home/tusharanand/sulekha-scraper/credentials/sulekha-487304-4f5a021dcc1c.json}"

# Worker configuration
WORKER_REPLICAS="${WORKER_REPLICAS:-8}"
WORKER_CONCURRENCY="${WORKER_CONCURRENCY:-2}"

echo "Creating ${NUM_VMS} worker VM(s) with:"
echo "  Machine type: ${MACHINE_TYPE}"
echo "  Image: ${IMAGE_URL}"
echo "  Workers per VM: ${WORKER_REPLICAS} x ${WORKER_CONCURRENCY} concurrency"
echo ""

for i in $(seq 1 "$NUM_VMS"); do
    VM_NAME="sambandh-ocr-worker-${i}"
    MACHINE_NAME="${VM_NAME}"
    
    echo "Creating ${VM_NAME}..."

    # Create docker-compose config for this worker
    COMPOSE_CONFIG=$(cat <<COMPOSE_EOF
version: '3.8'
services:
  worker:
    image: ${IMAGE_URL}
    command: >
      celery -A sambandh.celery_app worker
      --concurrency=${WORKER_CONCURRENCY}
      --pool=prefork
      -Q ocr
      -n worker-%h
    environment:
      - DATABASE_URL=postgresql://sambandh:sambandh@${MAIN_IP}:5433/sambandh
      - REDIS_URL=redis://${MAIN_IP}:6380/0
      - GCS_INPUT_BUCKET=sulekha-pdfs
      - GCS_OUTPUT_BUCKET=parsed-sulekha
      - GOOGLE_APPLICATION_CREDENTIALS=/app/secrets/gcs-credentials.json
      - MACHINE_NAME=${MACHINE_NAME}
    volumes:
      - /home/chronos/gcs-credentials.json:/app/secrets/gcs-credentials.json:ro
    restart: unless-stopped
    deploy:
      replicas: ${WORKER_REPLICAS}
      resources:
        limits:
          memory: 8G
COMPOSE_EOF
)

    # Startup script for Container-Optimized OS
    STARTUP_SCRIPT=$(cat <<'SCRIPT_EOF'
#!/bin/bash
set -euo pipefail

# Wait for Docker to be ready
while ! docker info > /dev/null 2>&1; do
    echo "Waiting for Docker..."
    sleep 2
done

# Configure Docker for Artifact Registry
gcloud auth configure-docker asia-south1-docker.pkg.dev --quiet

# Write docker-compose.yml
cat > /home/chronos/docker-compose.yml << 'COMPOSE_CONTENT'
COMPOSE_PLACEHOLDER
COMPOSE_CONTENT

# Pull the image
docker pull IMAGE_PLACEHOLDER

# Start workers using docker compose
cd /home/chronos
docker compose up -d

echo "Workers started successfully"
SCRIPT_EOF
)

    # Replace placeholders
    STARTUP_SCRIPT="${STARTUP_SCRIPT//COMPOSE_PLACEHOLDER/${COMPOSE_CONFIG}}"
    STARTUP_SCRIPT="${STARTUP_SCRIPT//IMAGE_PLACEHOLDER/${IMAGE_URL}}"

    # Create the VM with Container-Optimized OS
    gcloud compute instances create "${VM_NAME}" \
        --project="${PROJECT}" \
        --zone="${ZONE}" \
        --machine-type="${MACHINE_TYPE}" \
        --boot-disk-size="${DISK_SIZE}" \
        --image-family=cos-stable \
        --image-project=cos-cloud \
        --scopes=storage-full,logging-write,monitoring-write \
        --metadata-from-file=startup-script=<(echo "${STARTUP_SCRIPT}") \
        --tags=sambandh-worker \
        2>&1 | tail -3

    echo "  Waiting for VM to boot..."
    sleep 30

    # Copy GCS credentials to the VM
    echo "  Copying GCS credentials..."
    for attempt in 1 2 3; do
        if gcloud compute scp "${CREDS_SRC}" "${VM_NAME}:/home/chronos/gcs-credentials.json" \
            --zone="${ZONE}" --project="${PROJECT}" 2>/dev/null; then
            echo "  Credentials copied successfully"
            break
        else
            echo "  Attempt ${attempt} failed, retrying in 10s..."
            sleep 10
        fi
    done

    echo "  ${VM_NAME} created."
    echo ""
done

echo "============================================"
echo "Done. ${NUM_VMS} VM(s) created."
echo ""
echo "Monitor workers:"
echo "  gcloud compute instances list --filter='name~sambandh-ocr-worker' --project=${PROJECT}"
echo ""
echo "SSH into a worker:"
echo "  gcloud compute ssh ${VM_NAME} --zone=${ZONE} --project=${PROJECT}"
echo ""
echo "Check worker logs:"
echo "  gcloud compute ssh ${VM_NAME} --zone=${ZONE} --project=${PROJECT} -- 'docker compose -f /home/chronos/docker-compose.yml logs -f'"
echo ""
echo "Delete all workers:"
echo "  for i in \$(seq 1 ${NUM_VMS}); do"
echo "    gcloud compute instances delete sambandh-ocr-worker-\$i --zone=${ZONE} --project=${PROJECT} --quiet"
echo "  done"
