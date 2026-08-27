#!/bin/bash
set -e

# ============================================================
# Audio8 TTS - Setup Verification
# ============================================================
# Run this to verify everything is ready for training.
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
DATA_DIR="${SCRIPT_DIR}/data"
PREPARED_DIR="${SCRIPT_DIR}/prepared_data"
DATASET_PATH="${AUDIO8_HOME}/kreol/data/worldspeech_mfe_ljspeech/wavs"

echo ""
echo "============================================"
echo " Audio8 TTS - Setup Verification"
echo "============================================"
echo ""

PASS=0
FAIL=0

# --- 1. Python & Packages ---
log_info "1. Python environment"
python3 --version 2>/dev/null && PASS=$((PASS+1)) || { log_error "Python3 not found"; FAIL=$((FAIL+1)); }

python3 -c "import torch; print(f'   PyTorch {torch.__version__}')" 2>/dev/null && PASS=$((PASS+1)) || { log_warn "PyTorch not installed"; FAIL=$((FAIL+1)); }
python3 -c "import torch; print(f'   CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')" 2>/dev/null
python3 -c "import transformers; print(f'   Transformers {transformers.__version__}')" 2>/dev/null && PASS=$((PASS+1)) || { log_warn "Transformers not installed"; FAIL=$((FAIL+1)); }
echo ""

# --- 2. GPU ---
log_info "2. GPU"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null && PASS=$((PASS+1)) || { log_error "nvidia-smi not available"; FAIL=$((FAIL+1)); }
echo ""

# --- 3. Audio8 TTS repo ---
log_info "3. Audio8 TTS repository"
for f in audio8_tts_infer.py audio8_tts_prepare.py audio8_tts_sft.sh requirements-train.txt; do
    if [ -f "${REPO_ROOT}/${f}" ]; then
        log_ok "${f}"
        PASS=$((PASS+1))
    else
        log_error "${f} NOT FOUND in ${REPO_ROOT}"
        FAIL=$((FAIL+1))
    fi
done
echo ""

# --- 4. Model checkpoint ---
log_info "4. Model checkpoint"
MODEL_FOUND=false
for path in \
    "${REPO_ROOT}/model/audio8_tts_0_6B_preview" \
    "${REPO_ROOT}/model/Audio8-TTS-Preview-0.6b" \
    "${REPO_ROOT}/onnx_runtime/model" \
    "${REPO_ROOT}/model"; do
    if [ -d "$path" ]; then
        log_ok "Model found: $path"
        ls "$path" | head -5 | sed 's/^/   /'
        MODEL_FOUND=true
        PASS=$((PASS+1))
        break
    fi
done
if [ "$MODEL_FOUND" = false ]; then
    log_error "Model NOT found in ${REPO_ROOT}/model/"
    log_info "Download with:"
    log_info "  cd ${REPO_ROOT} && mkdir -p model"
    log_info "  huggingface-cli download Audio8/Audio8-TTS-Preview-0.6b --local-dir model/audio8_tts_0_6B_preview"
    FAIL=$((FAIL+1))
fi
echo ""

# --- 5. Dataset ---
log_info "5. Dataset"
if [ -d "${DATASET_PATH}" ]; then
    WAV_COUNT=$(find "${DATASET_PATH}" -name "*.wav" -type f | wc -l)
    log_ok "Dataset found: ${WAV_COUNT} wav files"
    PASS=$((PASS+1))

    log_info "Sample files:"
    find "${DATASET_PATH}" -name "*.wav" -type f | head -5 | sed 's/^/   /'
else
    log_error "Dataset NOT found at ${DATASET_PATH}"
    FAIL=$((FAIL+1))
fi
echo ""

# --- 6. Metadata ---
log_info "6. Metadata/Transcripts"
KREOL_DIR="${AUDIO8_HOME}/kreol"
FOUND_META=false
for meta in "metadata.csv" "metadata.txt" "transcripts.csv" "text.csv"; do
    for dir in "${KREOL_DIR}" "${KREOL_DIR}/data" "${DATASET_PATH}"; do
        if [ -f "${dir}/${meta}" ]; then
            log_ok "Found: ${dir}/${meta}"
            head -3 "${dir}/${meta}" | sed 's/^/   /'
            FOUND_META=true
            PASS=$((PASS+1))
            break 2
        fi
    done
done
if [ "$FOUND_META" = false ]; then
    log_warn "No metadata file found"
    log_info "You'll need to create metadata.csv with: filename|text"
    FAIL=$((FAIL+1))
fi
echo ""

# --- 7. Generated data ---
log_info "7. Generated training data"
if [ -f "${DATA_DIR}/train.jsonl" ]; then
    COUNT=$(wc -l < "${DATA_DIR}/train.jsonl")
    log_ok "Raw manifest: ${COUNT} entries"
    PASS=$((PASS+1))
else
    log_warn "Raw manifest not generated yet (run ./train.sh)"
fi

if [ -f "${PREPARED_DIR}/train.jsonl" ]; then
    COUNT=$(wc -l < "${PREPARED_DIR}/train.jsonl")
    log_ok "Prepared manifest: ${COUNT} entries"
    PASS=$((PASS+1))
else
    log_warn "Prepared manifest not generated yet (run ./train.sh)"
fi
echo ""

# --- Summary ---
TOTAL=$((PASS + FAIL))
echo "============================================"
if [ "${FAIL}" -eq 0 ]; then
    log_ok "All ${TOTAL} checks passed! Ready for training."
else
    log_warn "${PASS}/${TOTAL} checks passed, ${FAIL} need attention"
fi
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Run setup:  ./train.sh"
echo "  2. Then train: cd ${REPO_ROOT} && bash audio8_tts_sft.sh"
echo ""
