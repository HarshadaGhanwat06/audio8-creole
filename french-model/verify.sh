#!/bin/bash
set -e

# ============================================================
# Audio8 TTS - Verification Script
# ============================================================
# Run this to verify the setup is working correctly.
# Does NOT start training.
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
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="${REPO_ROOT}/model/audio8_tts_0_6B_preview"
DATA_DIR="${SCRIPT_DIR}/data"
PREPARED_DIR="${SCRIPT_DIR}/prepared_data"

echo ""
echo "============================================"
echo " Audio8 TTS - Setup Verification"
echo "============================================"
echo ""

# --- Check 1: Python and packages ---
log_info "Check 1: Python environment..."
python3 --version
python3 -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" || log_warn "PyTorch issue"
python3 -c "import transformers; print(f'  Transformers {transformers.__version__}')" || log_warn "Transformers issue"
python3 -c "import cv2; print(f'  OpenCV {cv2.__version__}')" || log_warn "OpenCV not installed"
echo ""

# --- Check 2: GPU ---
log_info "Check 2: GPU status..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || log_warn "nvidia-smi not available"
echo ""

# --- Check 3: Model checkpoint ---
log_info "Check 3: Model checkpoint..."
if [ -d "${MODEL_DIR}" ]; then
    log_ok "Model found at ${MODEL_DIR}"
    ls -la "${MODEL_DIR}" | head -10
else
    log_error "Model NOT found at ${MODEL_DIR}"
fi
echo ""

# --- Check 4: Dataset ---
log_info "Check 4: Dataset..."
DATASET_PATH="/home/ubuntu/audio8/kreol/data/worldspeech_mfe_ljspeech/wavs"
if [ -d "${DATASET_PATH}" ]; then
    WAV_COUNT=$(find "${DATASET_PATH}" -name "*.wav" -type f | wc -l)
    log_ok "Dataset found: ${WAV_COUNT} wav files"
    log_info "Sample files:"
    ls "${DATASET_PATH}" | head -5
else
    log_error "Dataset NOT found at ${DATASET_PATH}"
fi
echo ""

# --- Check 5: Generated data ---
log_info "Check 5: Generated data files..."
if [ -f "${DATA_DIR}/train.jsonl" ]; then
    COUNT=$(wc -l < "${DATA_DIR}/train.jsonl")
    log_ok "Raw manifest: ${COUNT} entries"
else
    log_warn "Raw manifest not generated yet (run train.sh)"
fi

if [ -f "${PREPARED_DIR}/train.jsonl" ]; then
    COUNT=$(wc -l < "${PREPARED_DIR}/train.jsonl")
    log_ok "Prepared manifest: ${COUNT} entries"
else
    log_warn "Prepared manifest not generated yet (run train.sh)"
fi
echo ""

# --- Check 6: Repository scripts ---
log_info "Check 6: Required scripts..."
for script in "audio8_tts_infer.py" "audio8_tts_prepare.py" "audio8_tts_sft.sh"; do
    if [ -f "${REPO_ROOT}/${script}" ]; then
        log_ok "${script}"
    else
        log_error "${script} NOT FOUND"
    fi
done
echo ""

# --- Summary ---
echo "============================================"
log_ok "Verification complete!"
echo ""
echo "Next steps:"
echo "  1. Run setup:     ./train.sh"
echo "  2. Start training: cd ${REPO_ROOT} && bash audio8_tts_sft.sh"
echo "============================================"
echo ""
