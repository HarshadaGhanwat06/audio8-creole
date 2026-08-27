#!/bin/bash
set -e

# ============================================================
# Audio8 TTS - French Creole Fine-tuning Setup
# ============================================================
# Project structure on GPU host:
#   ~/audio8/
#   ├── Audio8_TTS/       <- Original model repo (code + model)
#   ├── audio8-creole/    <- This repo (training scripts)
#   └── kreol/            <- Dataset
#       └── data/
#           └── worldspeech_mfe_ljspeech/
#               └── wavs/ <- .wav files
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
CREOLE_DIR="$(dirname "$SCRIPT_DIR")"
AUDIO8_HOME="$(dirname "$CREOLE_DIR")"
REPO_ROOT="${AUDIO8_HOME}/Audio8_TTS"
DATA_DIR="${SCRIPT_DIR}/data"
PREPARED_DIR="${SCRIPT_DIR}/prepared_data"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"
LOG_DIR="${SCRIPT_DIR}/logs"

# Dataset path
DATASET_PATH="${AUDIO8_HOME}/kreol/data/worldspeech_mfe_ljspeech/wavs"

# Model path - check common locations
MODEL_DIR=""
for path in \
    "${REPO_ROOT}/model/audio8_tts_0_6B_preview" \
    "${REPO_ROOT}/model/Audio8-TTS-Preview-0.6b" \
    "${REPO_ROOT}/onnx_runtime/model" \
    "${REPO_ROOT}/model"; do
    if [ -d "$path" ]; then
        MODEL_DIR="$path"
        break
    fi
done

# Training hyperparameters
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
echo " Creole dir:    ${CREOLE_DIR}"
echo " Audio8 repo:   ${REPO_ROOT}"
echo " Dataset:       ${DATASET_PATH}"
echo " Output dir:    ${OUTPUT_DIR}"
echo "============================================"
echo ""

# ============================================================
# STEP 1: Verify directory structure
# ============================================================
log_info "Step 1/7: Verifying directory structure..."

if [ ! -d "${AUDIO8_HOME}" ]; then
    log_error "Audio8 home not found: ${AUDIO8_HOME}"
    exit 1
fi
log_ok "Audio8 home: ${AUDIO8_HOME}"

if [ ! -f "${REPO_ROOT}/audio8_tts_infer.py" ]; then
    log_error "Cannot find audio8_tts_infer.py in ${REPO_ROOT}"
    log_error "Audio8_TTS repo not cloned correctly."
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
if [ -z "${MODEL_DIR}" ] || [ ! -d "${MODEL_DIR}" ]; then
    log_warn "Model checkpoint not found in standard locations"
    log_info "Checking ${REPO_ROOT}/model/..."
    if [ -d "${REPO_ROOT}/model" ]; then
        ls -la "${REPO_ROOT}/model/"
    fi
    echo ""
    log_error "Download the model:"
    log_error "  cd ${REPO_ROOT}"
    log_error "  mkdir -p model"
    log_error "  huggingface-cli download Audio8/Audio8-TTS-Preview-0.6b --local-dir model/audio8_tts_0_6B_preview"
    exit 1
fi
log_ok "Model found: ${MODEL_DIR}"
log_info "Model contents:"
ls "${MODEL_DIR}" | head -10
echo ""

# ============================================================
# STEP 2: Verify dataset
# ============================================================
log_info "Step 2/7: Verifying dataset..."

if [ ! -d "${DATASET_PATH}" ]; then
    log_error "Dataset not found: ${DATASET_PATH}"
    log_info "Checking kreol directory..."
    if [ -d "${AUDIO8_HOME}/kreol" ]; then
        find "${AUDIO8_HOME}/kreol" -type d | head -20
    fi
    exit 1
fi
log_ok "Dataset directory exists"

WAV_COUNT=$(find "${DATASET_PATH}" -name "*.wav" -type f | wc -l)
log_info "Found ${WAV_COUNT} .wav files"

if [ "${WAV_COUNT}" -eq 0 ]; then
    log_error "No .wav files in ${DATASET_PATH}"
    exit 1
fi
log_ok "Dataset has ${WAV_COUNT} audio files"

# Show sample files
log_info "Sample files:"
find "${DATASET_PATH}" -name "*.wav" -type f | head -5 | while read f; do
    SIZE=$(stat -c%s "$f" 2>/dev/null || echo "?")
    echo "  $(basename "$f") (${SIZE} bytes)"
done
echo ""

# Check for metadata
log_info "Looking for transcript files..."
KREOL_DIR="${AUDIO8_HOME}/kreol"
for meta in "metadata.csv" "metadata.txt" "transcripts.csv" "text.csv"; do
    for dir in "${KREOL_DIR}" "${KREOL_DIR}/data" "${DATASET_PATH}"; do
        if [ -f "${dir}/${meta}" ]; then
            log_ok "Found: ${dir}/${meta}"
            log_info "First 3 lines:"
            head -3 "${dir}/${meta}"
            echo ""
        fi
    done
done

# List all files in kreol directory
log_info "Kreol directory structure:"
find "${KREOL_DIR}" -maxdepth 3 -type f | head -20
echo ""

# ============================================================
# STEP 3: Activate venv and install dependencies
# ============================================================
log_info "Step 3/7: Setting up Python environment..."

# Check if venv exists, create if not
VENV_DIR="${SCRIPT_DIR}/.venv"
if [ ! -d "${VENV_DIR}" ]; then
    log_info "Creating virtual environment..."
    python3 -m venv "${VENV_DIR}"
    log_ok "Created venv at ${VENV_DIR}"
fi

log_info "Activating venv..."
source "${VENV_DIR}/bin/activate"
log_ok "Activated: $(which python3)"
log_ok "Python: $(python3 --version)"

# Install dependencies
log_info "Installing training dependencies..."
pip install --upgrade pip -q

if [ -f "${REPO_ROOT}/requirements-train.txt" ]; then
    log_info "Installing from requirements-train.txt..."
    pip install -r "${REPO_ROOT}/requirements-train.txt" -q
    log_ok "Training dependencies installed"
else
    log_warn "requirements-train.txt not found, installing manually..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
    pip install transformers datasets accelerate -q
    log_ok "Dependencies installed"
fi

# Verify packages
log_info "Verifying packages..."
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"
python3 -c "import transformers; print(f'  Transformers: {transformers.__version__}')"
python3 -c "import cv2; print(f'  OpenCV: {cv2.__version__}')" 2>/dev/null || log_warn "OpenCV not installed (not critical for training)"
log_ok "Environment ready"
echo ""

# ============================================================
# STEP 4: Create directories
# ============================================================
log_info "Step 4/7: Creating directories..."

mkdir -p "${DATA_DIR}" "${PREPARED_DIR}" "${OUTPUT_DIR}" "${LOG_DIR}"
log_ok "Created data directories"

# ============================================================
# STEP 5: Generate manifest JSONL
# ============================================================
log_info "Step 5/7: Generating training manifest..."

MANIFEST_FILE="${DATA_DIR}/train.jsonl"

python3 << PYTHON_SCRIPT
import os
import json
import sys

dataset_path = "${DATASET_PATH}"
data_dir = "${DATA_DIR}"
output_file = os.path.join(data_dir, "train.jsonl")

# Collect wav files
wav_files = sorted([f for f in os.listdir(dataset_path) if f.endswith(".wav")])
print(f"Found {len(wav_files)} .wav files")

# Look for metadata
kreol_dir = "${AUDIO8_HOME}/kreol"
metadata = {}
metadata_found = False

for root, dirs, files in os.walk(kreol_dir):
    for fname in files:
        if fname in ["metadata.csv", "metadata.txt", "transcripts.csv"]:
            mpath = os.path.join(root, fname)
            print(f"Loading metadata from: {mpath}")
            with open(mpath, "r", encoding="utf-8") as f:
                for line in f:
                    # Try different separators
                    for sep in ["|", ",", "\t"]:
                        parts = line.strip().split(sep)
                        if len(parts) >= 2:
                            filename = parts[0].strip()
                            text = sep.join(parts[1:]).strip()
                            # Remove .wav extension if present
                            key = os.path.splitext(filename)[0]
                            metadata[key] = text
                            metadata[filename] = text
                            break
            metadata_found = True
            print(f"Loaded {len(metadata)} entries")
            break
    if metadata_found:
        break

if not metadata_found:
    print("WARNING: No metadata file found!")
    print("Please create metadata.csv with format: filename|text")
    print("Example:")
    print("  sample_001.wav|Bonjou kijan ou ye?")
    print("  sample_002.wav|Mwen renmen Kreyol Ayisyen.")
    sys.exit(1)

# Generate manifest
records = []
skipped = 0
for wav_file in wav_files:
    wav_path = os.path.abspath(os.path.join(dataset_path, wav_file))
    file_id = os.path.splitext(wav_file)[0]

    text = metadata.get(wav_file, metadata.get(file_id, ""))
    if not text:
        skipped += 1
        if skipped <= 3:
            print(f"WARNING: No transcript for {wav_file}")
        continue

    records.append({
        "id": f"creole_{file_id}",
        "text": text,
        "audio": wav_path
    })

with open(output_file, "w", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"\nManifest created: {output_file}")
print(f"Total entries: {len(records)}")
if skipped:
    print(f"Skipped (no transcript): {skipped}")
PYTHON_SCRIPT

if [ $? -eq 0 ]; then
    log_ok "Manifest generated"
    log_info "Manifest stats:"
    wc -l "${MANIFEST_FILE}"
else
    log_error "Manifest generation failed"
    exit 1
fi
echo ""

# ============================================================
# STEP 6: Precompute codec indices
# ============================================================
log_info "Step 6/7: Precomputing codec indices..."
log_info "This processes audio files and may take a while..."

cd "${REPO_ROOT}"
python3 audio8_tts_prepare.py \
    --input-jsonl "${DATA_DIR}/train.jsonl" \
    --output-jsonl "${PREPARED_DIR}/train.jsonl" \
    --batch-size 4 2>&1 | tee "${LOG_DIR}/prepare.log"

if [ $? -eq 0 ]; then
    log_ok "Codec indices precomputed"
    log_info "Prepared manifest:"
    wc -l "${PREPARED_DIR}/train.jsonl"
else
    log_error "Precomputation failed. Check ${LOG_DIR}/prepare.log"
    exit 1
fi
echo ""

# ============================================================
# STEP 7: Final verification
# ============================================================
log_info "Step 7/7: Final verification..."

echo ""
echo "============================================"
echo " SETUP COMPLETE"
echo "============================================"

CHECKS_PASSED=0
CHECKS_TOTAL=0

check_item() {
    CHECKS_TOTAL=$((CHECKS_TOTAL + 1))
    if [ "$1" = "ok" ]; then
        log_ok "$2"
        CHECKS_PASSED=$((CHECKS_PASSED + 1))
    else
        log_error "$2"
    fi
}

[ -f "${DATA_DIR}/train.jsonl" ] && check_item "ok" "Raw manifest" || check_item "fail" "Raw manifest MISSING"
[ -f "${PREPARED_DIR}/train.jsonl" ] && check_item "ok" "Prepared manifest" || check_item "fail" "Prepared manifest MISSING"
[ -d "${MODEL_DIR}" ] && check_item "ok" "Model checkpoint" || check_item "fail" "Model checkpoint MISSING"

# Check GPU
if nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null | head -1 > /dev/null; then
    check_item "ok" "GPU available"
else
    check_item "fail" "GPU not detected"
fi

echo ""
echo "Checks passed: ${CHECKS_PASSED}/${CHECKS_TOTAL}"
echo "============================================"

# Print training command
echo ""
log_info "When ready to train, run:"
echo ""
echo "  cd ${REPO_ROOT}"
echo "  source ${VENV_DIR}/bin/activate"
echo "  TRAIN_JSONL=${PREPARED_DIR}/train.jsonl \\"
echo "    NPROC_PER_NODE=${NPROC_PER_NODE} \\"
echo "    BATCH_SIZE=${BATCH_SIZE} \\"
echo "    GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS} \\"
echo "    bash audio8_tts_sft.sh"
echo ""
log_info "Monitor GPU usage: watch -n 1 nvidia-smi"
echo ""
