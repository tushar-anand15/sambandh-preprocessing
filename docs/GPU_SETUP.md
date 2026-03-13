# GPU Instance Setup for Batch PDF Parsing

This guide covers setting up a GCP Spot GPU instance to run the batch PDF parser.

## Cost Estimate

| Component | Hourly Cost |
|-----------|-------------|
| T4 GPU (Spot) | ~$0.11 |
| n1-standard-8 (Spot) | ~$0.08 |
| 100GB SSD | ~$0.02 |
| **Total** | **~$0.21/hr** |

For 80k PDFs at ~5 pages/sec on T4: ~45 hours = **~$10**

---

## Prerequisites

### 1. Install and Configure gcloud CLI

```bash
# Check if gcloud is installed
gcloud --version

# Authenticate (if not already)
gcloud auth login

# Set your project
gcloud config set project sulekha-487304
```

### 2. Enable GPU Quota (First Time Only)

GPUs require quota approval. Check your current quota:

```bash
gcloud compute regions describe us-central1 --format="table(quotas.filter(metric:NVIDIA_T4_GPUS))"
```

If quota is 0, request an increase:
1. Go to: https://console.cloud.google.com/iam-admin/quotas
2. Filter for "NVIDIA T4" in `us-central1`
3. Request increase to at least 1

---

## Create the Spot GPU Instance

```bash
gcloud compute instances create sambandh-gpu \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=common-cu121-debian-11-py310 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT \
  --scopes=storage-full,cloud-platform \
  --metadata="install-nvidia-driver=True"
```

**Configuration breakdown:**
- `n1-standard-8`: 8 vCPUs, 30GB RAM
- `nvidia-tesla-t4`: 16GB VRAM GPU
- `common-cu121-debian-11-py310`: Pre-installed CUDA 12.1 + Python 3.10
- `provisioning-model=SPOT`: ~70% cheaper, can be preempted
- `scopes=storage-full`: Full access to GCS buckets

---

## SSH Into the Instance

```bash
gcloud compute ssh sambandh-gpu --zone=us-central1-a
```

---

## Set Up the Environment (On the GPU Instance)

### 1. Verify GPU is detected

```bash
nvidia-smi
```

You should see the T4 GPU listed with 16GB memory.

### 2. Clone the repo and install dependencies

```bash
git clone https://github.com/YOUR_USERNAME/sambandh-preprocessing.git
cd sambandh-preprocessing

python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 3. Set up credentials

**Option A: Copy your GCS key file**

From your local machine:
```bash
gcloud compute scp ~/sulekha-scraper/credentials/sulekha-487304-4f5a021dcc1c.json \
  sambandh-gpu:~/credentials.json --zone=us-central1-a
```

On the GPU instance:
```bash
export GOOGLE_APPLICATION_CREDENTIALS=~/credentials.json
```

**Option B: Use instance's default service account**

If the instance has the right IAM permissions, GCS access works automatically.

### 4. Set up database access

The scraper Postgres needs to be accessible from the GPU instance.

**Option A: Open firewall for GPU instance IP**
```bash
# Get the GPU instance's external IP
gcloud compute instances describe sambandh-gpu --zone=us-central1-a --format='get(networkInterfaces[0].accessConfigs[0].natIP)'

# Add this IP to your scraper's firewall rules
```

**Option B: Use the scraper VM's internal IP (if same VPC)**
```bash
export SCRAPER_DATABASE_URL="postgresql://sulekha:sulekha@INTERNAL_IP:5432/sulekha"
```

**Option C: Export PDF list to a file**

On your scraper machine, export the PDF list and copy it to the GPU instance.

---

## Run the Batch Parser

### Test with a small batch first

```bash
python scripts/batch_parse_gpu.py \
  --year 2025-2026 \
  --limit 10 \
  --local-output ./test_output
```

### Full run (outputs to GCS)

```bash
python scripts/batch_parse_gpu.py \
  --year 2025-2026 \
  --layout-batch 32 \
  --table-batch 4
```

### Filter by district or LB type

```bash
# Process only Ernakulam district
python scripts/batch_parse_gpu.py \
  --year 2025-2026 \
  --district Ernakulam

# Process only Grama Panchayat
python scripts/batch_parse_gpu.py \
  --year 2025-2026 \
  --lb-type "Grama Panchayat"
```

### Resume after interruption

The script automatically checkpoints progress. Just run the same command again:
```bash
python scripts/batch_parse_gpu.py --year 2025-2026
```

It will skip already-processed PDFs and continue from where it left off.

---

## Monitor Progress

### Check state file

```bash
cat parse_state.json | python -m json.tool | head -20
```

### Watch GPU utilization

```bash
watch -n 1 nvidia-smi
```

### Check GCS output

```bash
gsutil ls gs://sulekha-pdfs/parsed/2025-2026/ | head -20
```

---

## Stop/Delete When Done

### Stop (keeps disk, stops billing for compute)

```bash
gcloud compute instances stop sambandh-gpu --zone=us-central1-a
```

### Start again later

```bash
gcloud compute instances start sambandh-gpu --zone=us-central1-a
```

### Delete (removes everything, stops all billing)

```bash
gcloud compute instances delete sambandh-gpu --zone=us-central1-a
```

---

## Troubleshooting

### GPU not detected

```bash
# Check if NVIDIA driver is installed
nvidia-smi

# If not, install it
sudo /opt/deeplearning/install-driver.sh
```

### Out of memory (OOM)

Reduce batch sizes:
```bash
python scripts/batch_parse_gpu.py \
  --year 2025-2026 \
  --layout-batch 16 \
  --table-batch 2
```

### Spot VM preempted

This is expected with Spot VMs. The script checkpoints progress, so just:
1. Recreate the instance with the same command
2. SSH in and run the parser again - it will resume

### Database connection refused

1. Check if scraper Postgres is running
2. Verify firewall allows GPU instance IP
3. Try using internal IP if in same VPC

---

## Alternative: Use tmux for Long-Running Jobs

```bash
# Start a tmux session
tmux new -s parse

# Run the parser
python scripts/batch_parse_gpu.py --year 2025-2026

# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t parse
```

This keeps the job running even if your SSH connection drops.
