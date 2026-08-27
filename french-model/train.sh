#!/bin/bash
set -e

# ============================================================
# Audio8 TTS - French Creole Fine-tuning Setup
# ============================================================
# Uses existing venv at: ~/audio8/Audio8_TTS/onnx_runtime/.venv
# ============================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CREOLE_DIR="$(dirname "$SCRIPT_DIR")"
AUDIO8_HOME="$(dirname "$CREOLE_DIR")"
REPO_ROOT="${AUDIO8_HOME}/Audio8_TTS"
VENV_DIR="${REPO_ROOT}/onnx_runtime/.venv"
DATA_DIR="${SCRIPT_DIR}/data"
PREPARED_DIR="${SCRIPT_DIR}/prepared_data"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"
LOG_DIR="${SCRIPT_DIR}/logs"
DATASET_PATH="${AUDIO8_HOME}/kreol/data/worldspeech_mfe_ljspeech/wavs"

# Model path
MODEL_DIR=""
for path in \
    "${REPO_ROOT}/onnx_runtime/model" \
    "${REPO_ROOT}/model/audio8_tts_0_6B_preview" \
    "${REPO_ROOT}/model"; do
    if [ -d "$path" ]; then
        MODEL_DIR="$path"
        break
    fi
done

NPROC_PER_NODE=${NPROC_PER_NODE:-1}
BATCH_SIZE=${BATCH_SIZE:-2}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}

echo ""
echo "============================================"
echo " Audio8 TTS - French Creole Fine-tuning"
echo "============================================"
echo ""

# STEP 1: Verify structure
log_info "Step 1/7: Verifying directory structure..."
[ -f "${REPO_ROOT}/audio8_tts_infer.py" ] && log_ok "audio8_tts_infer.py" || { log_error "Missing audio8_tts_infer.py"; exit 1; }
[ -f "${REPO_ROOT}/audio8_tts_sft.sh" ] && log_ok "audio8_tts_sft.sh" || { log_error "Missing audio8_tts_sft.sh"; exit 1; }
[ -f "${REPO_ROOT}/audio8_tts_prepare.py" ] && log_ok "audio8_tts_prepare.py" || { log_error "Missing audio8_tts_prepare.py"; exit 1; }
[ -d "${MODEL_DIR}" ] && log_ok "Model: ${MODEL_DIR}" || { log_error "Model not found"; exit 1; }
echo ""

# STEP 2: Verify dataset
log_info "Step 2/7: Verifying dataset..."
[ -d "${DATASET_PATH}" ] || { log_error "Dataset not found: ${DATASET_PATH}"; exit 1; }
WAV_COUNT=$(find "${DATASET_PATH}" -name "*.wav" -type f | wc -l)
log_ok "Dataset: ${WAV_COUNT} wav files"
echo ""

# STEP 3: Verify venv
log_info "Step 3/7: Verifying Python environment..."
[ -d "${VENV_DIR}" ] || { log_error "Venv not found: ${VENV_DIR}"; exit 1; }
source "${VENV_DIR}/bin/activate"
log_ok "Venv: ${VENV_DIR}"
log_ok "Python: $(python3 --version)"

python3 -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')" || { log_error "PyTorch issue"; exit 1; }
python3 -c "import transformers; print(f'  Transformers {transformers.__version__}')" || { log_error "Transformers issue"; exit 1; }
log_ok "All packages verified"
echo ""

# STEP 4: Create directories
log_info "Step 4/7: Creating directories..."
mkdir -p "${DATA_DIR}" "${PREPARED_DIR}" "${OUTPUT_DIR}" "${LOG_DIR}"
log_ok "Directories ready"
echo ""

# STEP 5: Generate manifest
log_info "Step 5/7: Generating training manifest..."

python3 << PYTHON_SCRIPT
import os, json, sys

dataset_path = "${DATASET_PATH}"
data_dir = "${DATA_DIR}"
output_file = os.path.join(data_dir, "train.jsonl")
kreol_dir = "${AUDIO8_HOME}/kreol"

wav_files = sorted([f for f in os.listdir(dataset_path) if f.endswith(".wav")])
print(f"Found {len(wav_files)} .wav files")

# Find metadata
metadata = {}
for root, dirs, files in os.walk(kreol_dir):
    for fname in files:
        if fname in ["metadata.csv", "metadata.txt"]:
            mpath = os.path.join(root, fname)
            print(f"Loading: {mpath}")
            with open(mpath, "r", encoding="utf-8") as f:
                for line in f:
                    for sep in ["|", ",", "\t"]:
                        parts = line.strip().split(sep)
                        if len(parts) >= 2:
                            key = os.path.splitext(parts[0].strip())[0]
                            text = sep.join(parts[1:]).strip()
                            metadata[key] = text
                            metadata[parts[0].strip()] = text
                            break
            print(f"Loaded {len(metadata)} entries")
            break
    if metadata:
        break

if not metadata:
    print("ERROR: No metadata.csv found!")
    print("Create metadata.csv with: filename|text")
    sys.exit(1)

records = []
skipped = 0
for wav_file in wav_files:
    wav_path = os.path.abspath(os.path.join(dataset_path, wav_file))
    file_id = os.path.splitext(wav_file)[0]
    text = metadata.get(wav_file, metadata.get(file_id, ""))
    if not text:
        skipped += 1
        continue
    records.append({"id": f"creole_{file_id}", "text": text, "audio": wav_path})

with open(output_file, "w", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"Manifest: {output_file}")
print(f"Entries: {len(records)}, Skipped: {skipped}")
PYTHON_SCRIPT

[ $? -eq 0 ] && log_ok "Manifest generated" || { log_error "Manifest failed"; exit 1; }
echo ""

# STEP 6: Precompute codec indices
log_info "Step 6/7: Precomputing codec indices..."
cd "${REPO_ROOT}"
python3 audio8_tts_prepare.py \
    --input-jsonl "${DATA_DIR}/train.jsonl" \
    --output-jsonl "${PREPARED_DIR}/train.jsonl" \
    --batch-size 4 2>&1 | tee "${LOG_DIR}/prepare.log"

[ $? -eq 0 ] && log_ok "Codec indices ready" || { log_error "Precomputation failed"; exit 1; }
echo ""

# STEP 7: Summary
log_info "Step 7/7: Verification summary..."
echo ""
echo "============================================"
echo " SETUP COMPLETE"
echo "============================================"
[ -f "${DATA_DIR}/train.jsonl" ] && log_ok "Raw manifest: $(wc -l < ${DATA_DIR}/train.jsonl) entries"
[ -f "${PREPARED_DIR}/train.jsonl" ] && log_ok "Prepared manifest: $(wc -l < ${PREPARED_DIR}/train.jsonl) entries"
[ -d "${MODEL_DIR}" ] && log_ok "Model: ${MODEL_DIR}"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null | head -1 && log_ok "GPU available"
echo "============================================"
echo ""
log_info "To start training:"
echo "  cd ${REPO_ROOT}"
echo "  source ${VENV_DIR}/bin/activate"
echo "  TRAIN_JSONL=${PREPARED_DIR}/train.jsonl NPROC_PER_NODE=1 bash audio8_tts_sft.sh"
echo ""
