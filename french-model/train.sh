#!/bin/bash
set -e

# ============================================================
# Audio8 TTS - French Creole Fine-tuning Setup
# ============================================================
# This script sets up everything for training.
# Training is NOT started automatically.
# ============================================================

# --- Colors for output ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="${REPO_ROOT}/model/audio8_tts_0_6B_preview"
DATA_DIR="${SCRIPT_DIR}/data"
PREPARED_DIR="${SCRIPT_DIR}/prepared_data"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"
LOG_DIR="${SCRIPT_DIR}/logs"

# Dataset
DATASET_PATH="/home/ubuntu/audio8/kreol/data/worldspeech_mfe_ljspeech/wavs"

# Training hyperparameters (can be overridden via env vars)
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
BATCH_SIZE=${BATCH_SIZE:-2}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
MAX_STEPS=${MAX_STEPS:-5000}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
SAVE_STEPS=${SAVE_STEPS:-500}
LOGGING_STEPS=${LOGGING_STEPS:-50}

echo ""
echo "============================================"
echo " Audio8 TTS - French Creole Fine-tuning"
echo "============================================"
echo " Script dir:    ${SCRIPT_DIR}"
echo " Repo root:     ${REPO_ROOT}"
echo " Model dir:     ${MODEL_DIR}"
echo " Dataset path:  ${DATASET_PATH}"
echo " Output dir:    ${OUTPUT_DIR}"
echo "============================================"
echo ""

# ============================================================
# STEP 1: Verify directory structure
# ============================================================
log_info "Step 1/7: Verifying directory structure..."

# Check if we are in the right repo
if [ ! -f "${REPO_ROOT}/audio8_tts_infer.py" ]; then
    log_error "Cannot find audio8_tts_infer.py in ${REPO_ROOT}"
    log_error "Make sure you cloned the Audio8 TTS repo correctly."
    exit 1
fi
log_ok "Found audio8_tts_infer.py"

if [ ! -f "${REPO_ROOT}/audio8_tts_sft.sh" ]; then
    log_error "Cannot find audio8_tts_sft.sh in ${REPO_ROOT}"
    exit 1
fi
log_ok "Found audio8_tts_sft.sh"

if [ ! -f "${REPO_ROOT}/audio8_tts_prepare.py" ]; then
    log_error "Cannot find audio8_tts_prepare.py in ${REPO_ROOT}"
    exit 1
fi
log_ok "Found audio8_tts_prepare.py"

# Check model checkpoint
if [ ! -d "${MODEL_DIR}" ]; then
    log_warn "Model checkpoint not found at ${MODEL_DIR}"
    log_warn "Checking alternative locations..."
    if [ -d "${REPO_ROOT}/model" ]; then
        log_info "Contents of ${REPO_ROOT}/model/:"
        ls -la "${REPO_ROOT}/model/"
    fi
    log_error "Download the model from:"
    log_error "  huggingface-cli download Audio8/Audio8-TTS-Preview-0.6b --local-dir ${MODEL_DIR}"
    exit 1
fi
log_ok "Model checkpoint found at ${MODEL_DIR}"

# List model contents
log_info "Model contents:"
ls -la "${MODEL_DIR}" | head -20
echo ""

# ============================================================
# STEP 2: Verify dataset
# ============================================================
log_info "Step 2/7: Verifying dataset..."

if [ ! -d "${DATASET_PATH}" ]; then
    log_error "Dataset directory not found: ${DATASET_PATH}"
    exit 1
fi
log_ok "Dataset directory exists"

# Count wav files
WAV_COUNT=$(find "${DATASET_PATH}" -name "*.wav" -type f | wc -l)
log_info "Found ${WAV_COUNT} .wav files in ${DATASET_PATH}"

if [ "${WAV_COUNT}" -eq 0 ]; then
    log_error "No .wav files found in ${DATASET_PATH}"
    exit 1
fi
log_ok "Dataset has ${WAV_COUNT} audio files"

# Show first few files
log_info "First 10 .wav files:"
find "${DATASET_PATH}" -name "*.wav" -type f | head -10 | while read f; do
    echo "  $(basename "$f")"
done
echo ""

# Check for metadata files
log_info "Looking for metadata files..."
DATASET_PARENT="$(dirname "${DATASET_PATH}")"
for meta_name in "metadata.csv" "metadata.txt" "metadata.json" "transcripts.csv" "text.csv" "text.txt"; do
    if [ -f "${DATASET_PARENT}/${meta_name}" ]; then
        log_ok "Found: ${DATASET_PARENT}/${meta_name}"
        log_info "First 5 lines:"
        head -5 "${DATASET_PARENT}/${meta_name}"
        echo ""
    fi
    if [ -f "${DATASET_PATH}/${meta_name}" ]; then
        log_ok "Found: ${DATASET_PATH}/${meta_name}"
        log_info "First 5 lines:"
        head -5 "${DATASET_PATH}/${meta_name}"
        echo ""
    fi
done

# Check for text files that might contain transcripts
if [ -f "${DATASET_PARENT}/metadata.csv" ]; then
    log_ok "Using metadata.csv for transcript mapping"
    METADATA_FILE="${DATASET_PARENT}/metadata.csv"
elif [ -f "${DATASET_PATH}/metadata.csv" ]; then
    log_ok "Using metadata.csv for transcript mapping"
    METADATA_FILE="${DATASET_PATH}/metadata.csv"
else
    log_warn "No metadata.csv found!"
    log_warn "Will generate manifest using filenames as IDs"
    log_warn "You will need to add text transcripts manually"
    METADATA_FILE=""
fi

# ============================================================
# STEP 3: Install training dependencies
# ============================================================
log_info "Step 3/7: Installing training dependencies..."

cd "${REPO_ROOT}"

if [ -f "${REPO_ROOT}/requirements-train.txt" ]; then
    log_info "Installing from requirements-train.txt..."
    pip install -r requirements-train.txt 2>&1 | tail -5
    log_ok "Training dependencies installed"
else
    log_warn "requirements-train.txt not found, installing common training deps..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install transformers datasets accelerate
    log_ok "Common training dependencies installed"
fi

# Verify key packages
log_info "Verifying key packages..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')" 2>/dev/null || {
    log_error "PyTorch not installed or CUDA not available"
    exit 1
}
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')" 2>/dev/null || {
    log_error "Transformers not installed"
    exit 1
}
log_ok "All key packages verified"
echo ""

# ============================================================
# STEP 4: Create data directories
# ============================================================
log_info "Step 4/7: Creating data directories..."

mkdir -p "${DATA_DIR}"
mkdir -p "${PREPARED_DIR}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

log_ok "Created: ${DATA_DIR}"
log_ok "Created: ${PREPARED_DIR}"
log_ok "Created: ${OUTPUT_DIR}"
log_ok "Created: ${LOG_DIR}"
echo ""

# ============================================================
# STEP 5: Generate manifest JSONL
# ============================================================
log_info "Step 5/7: Generating training manifest..."

MANIFEST_FILE="${DATA_DIR}/train.jsonl"

python3 << 'PYTHON_SCRIPT'
import os
import json
import sys

dataset_path = os.environ.get("DATASET_PATH", "/home/ubuntu/audio8/kreol/data/worldspeech_mfe_ljspeech/wavs")
data_dir = os.environ.get("DATA_DIR", "data")
output_file = os.path.join(data_dir, "train.jsonl")

# Collect all wav files
wav_files = sorted([f for f in os.listdir(dataset_path) if f.endswith(".wav")])

if not wav_files:
    print(f"ERROR: No .wav files found in {dataset_path}")
    sys.exit(1)

print(f"Found {len(wav_files)} .wav files")

# Check for metadata.csv in parent directory
dataset_parent = os.path.dirname(dataset_path)
metadata_file = os.path.join(dataset_parent, "metadata.csv")

metadata = {}
if os.path.exists(metadata_file):
    print(f"Loading metadata from {metadata_file}")
    with open(metadata_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")  # LJSpeech format uses |
            if len(parts) < 2:
                parts = line.strip().split(",")  # Try CSV format
            if len(parts) >= 2:
                filename = parts[0].strip()
                text = parts[1].strip()
                metadata[filename] = text
    print(f"Loaded {len(metadata)} entries from metadata")
else:
    print(f"WARNING: No metadata.csv found at {metadata_parent}")
    print("Will create manifest with empty text - you MUST add transcripts")

# Generate manifest
records = []
skipped = 0
for wav_file in wav_files:
    wav_path = os.path.abspath(os.path.join(dataset_path, wav_file))
    file_id = os.path.splitext(wav_file)[0]

    # Get text from metadata if available
    text = metadata.get(wav_file, "")
    if not text:
        text = metadata.get(file_id, "")

    if not text:
        skipped += 1
        if skipped <= 5:
            print(f"WARNING: No transcript for {wav_file}")
        continue

    record = {
        "id": f"creole_{file_id}",
        "text": text,
        "audio": wav_path
    }
    records.append(record)

if skipped > 0:
    print(f"\nWARNING: {skipped} files have no transcript")
    if skipped == len(wav_files):
        print("ERROR: No transcripts found. Please provide metadata.csv")
        sys.exit(1)

# Write manifest
with open(output_file, "w", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"\nManifest created: {output_file}")
print(f"Total entries: {len(records)}")
PYTHON_SCRIPT

if [ $? -eq 0 ]; then
    log_ok "Manifest generated successfully"
    log_info "Manifest stats:"
    wc -l "${MANIFEST_FILE}"
    log_info "First 3 entries:"
    head -3 "${MANIFEST_FILE}" | python3 -m json.tool 2>/dev/null || head -3 "${MANIFEST_FILE}"
else
    log_error "Manifest generation failed"
    exit 1
fi
echo ""

# ============================================================
# STEP 6: Precompute codec indices
# ============================================================
log_info "Step 6/7: Precomputing codec indices..."
log_info "This will take a while depending on dataset size..."

cd "${REPO_ROOT}"

python audio8_tts_prepare.py \
    --input-jsonl "${DATA_DIR}/train.jsonl" \
    --output-jsonl "${PREPARED_DIR}/train.jsonl" \
    --batch-size 4 2>&1 | tee "${LOG_DIR}/prepare.log"

if [ $? -eq 0 ]; then
    log_ok "Codec indices precomputed successfully"
    log_info "Prepared manifest stats:"
    wc -l "${PREPARED_DIR}/train.jsonl"
else
    log_error "Codec precomputation failed. Check ${LOG_DIR}/prepare.log"
    exit 1
fi
echo ""

# ============================================================
# STEP 7: Verify everything is ready
# ============================================================
log_info "Step 7/7: Final verification..."

echo ""
echo "============================================"
echo " SETUP COMPLETE - VERIFICATION SUMMARY"
echo "============================================"

# Check all required files
REQUIRED_FILES=(
    "${DATA_DIR}/train.jsonl"
    "${PREPARED_DIR}/train.jsonl"
)

ALL_OK=true
for f in "${REQUIRED_FILES[@]}"; do
    if [ -f "$f" ]; then
        SIZE=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f" 2>/dev/null || echo "unknown")
        log_ok "$(basename $f) exists (${SIZE} bytes)"
    else
        log_error "$(basename $f) MISSING"
        ALL_OK=false
    fi
done

# Check model
if [ -d "${MODEL_DIR}" ]; then
    log_ok "Model checkpoint: ${MODEL_DIR}"
else
    log_error "Model checkpoint: MISSING"
    ALL_OK=false
fi

# Check GPU
log_info "GPU status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || {
    log_warn "nvidia-smi not available"
}

echo ""
echo "============================================"
if [ "${ALL_OK}" = true ]; then
    log_ok "All checks passed!"
else
    log_error "Some checks failed. Fix issues before training."
fi
echo "============================================"
echo ""

# Show training command (but don't run it)
log_info "To start training, run:"
echo ""
echo "  cd ${REPO_ROOT}"
echo "  TRAIN_JSONL=${PREPARED_DIR}/train.jsonl \\"
echo "    NPROC_PER_NODE=${NPROC_PER_NODE} \\"
echo "    BATCH_SIZE=${BATCH_SIZE} \\"
echo "    GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS} \\"
echo "    bash audio8_tts_sft.sh"
echo ""
log_info "Or run the full training script:"
echo "  ./train.sh --start-training"
echo ""
