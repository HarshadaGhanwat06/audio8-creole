#!/bin/bash
set -e

# ============================================================
# Fix audio8_tts_sft.sh - Disable DeepSpeed for single GPU
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

SFT_SCRIPT="/home/ubuntu/audio8/Audio8_TTS/audio8_tts_sft.sh"
BACKUP="${SFT_SCRIPT}.bak"

echo ""
echo "============================================"
echo " Fixing DeepSpeed Configuration"
echo "============================================"
echo ""

# Step 1: Backup
log_info "Step 1: Creating backup..."
if [ -f "${BACKUP}" ]; then
    log_warn "Backup already exists: ${BACKUP}"
else
    cp "${SFT_SCRIPT}" "${BACKUP}"
    log_ok "Backup created: ${BACKUP}"
fi
echo ""

# Step 2: Show current DeepSpeed lines
log_info "Step 2: Current DeepSpeed configuration:"
echo "---"
grep -n "DEEPSPEED\|deepspeed" "${SFT_SCRIPT}" || true
echo "---"
echo ""

# Step 3: Modify the script
log_info "Step 3: Modifying script..."

# 3a: Change DEEPSPEED_CONFIG default to empty
sed -i 's|DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${PROJECT_ROOT}/configs/deepspeed_zero2.json}"|DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-}"|g' "${SFT_SCRIPT}"
log_ok "Changed DEEPSPEED_CONFIG default to empty"

# 3b: Replace unconditional --deepspeed argument with conditional
# First, find and remove the unconditional deepspeed line
sed -i '/--deepspeed "${DEEPSPEED_CONFIG:-${PROJECT_ROOT}\/configs\/deepspeed_zero2.json}"/d' "${SFT_SCRIPT}"
log_ok "Removed unconditional --deepspeed argument"

# 3c: Add conditional deepspeed block after the training command starts
# We need to find where the training command is and add the conditional
# Look for the line with "audio8_tts_sft.py" and add deepspeed conditionally before it
if grep -q "DEEPSPEED_CONFIG" "${SFT_SCRIPT}"; then
    log_info "Adding conditional DeepSpeed block..."

    # Create a temp file with the conditional logic
    TEMP_FILE=$(mktemp)

    # Read the file and insert conditional deepspeed before the python command
    awk '
    /^python3.*audio8_tts_sft\.py/ || /^python.*audio8_tts_sft\.py/ {
        # Check if DEEPSPEED_CONFIG is set in the line above or nearby
        if (DEEPSPEED_CONFIG_SET == 0) {
            print "DEEPSPEED_ARGS=\"\""
            print "if [ -n \"${DEEPSPEED_CONFIG}\" ]; then"
            print "    DEEPSPEED_ARGS=\"--deepspeed \\\"${DEEPSPEED_CONFIG}\\\"\""
            print "fi"
            print ""
            DEEPSPEED_CONFIG_SET = 1
        }
    }
    { print }
    ' "${SFT_SCRIPT}" > "${TEMP_FILE}"

    # If the awk didn't find it, try a different approach
    if ! grep -q "DEEPSPEED_ARGS" "${TEMP_FILE}"; then
        log_warn "Awk approach didn't work, using sed..."

        # Find the python3 audio8_tts_sft.py line and add conditional before it
        LINE_NUM=$(grep -n "python3.*audio8_tts_sft\.py\|python.*audio8_tts_sft\.py" "${SFT_SCRIPT}" | head -1 | cut -d: -f1)

        if [ -n "${LINE_NUM}" ]; then
            # Insert conditional block before the python command
            sed -i "${LINE_NUM}i\\
DEEPSPEED_ARGS=\"\"\\
if [ -n \"\${DEEPSPEED_CONFIG}\" ]; then\\
    DEEPSPEED_ARGS=\"--deepspeed \\\"\${DEEPSPEED_CONFIG}\\\"\"\\
fi" "${SFT_SCRIPT}"

            log_ok "Inserted conditional DeepSpeed block"
        fi
    else
        mv "${TEMP_FILE}" "${SFT_SCRIPT}"
    fi

    rm -f "${TEMP_FILE}"
fi

# 3d: Now replace any remaining --deepspeed references in the python command with ${DEEPSPEED_ARGS}
# Find lines with audio8_tts_sft.py and add ${DEEPSPEED_ARGS} if not already there
LINE_NUM=$(grep -n "python3.*audio8_tts_sft\.py\|python.*audio8_tts_sft\.py" "${SFT_SCRIPT}" | head -1 | cut -d: -f1)
if [ -n "${LINE_NUM}" ]; then
    # Check if this line already has DEEPSPEED_ARGS
    if ! sed -n "${LINE_NUM}p" "${SFT_SCRIPT}" | grep -q "DEEPSPEED_ARGS"; then
        # Add ${DEEPSPEED_ARGS} at the end of the python command (before the backslash or end)
        sed -i "${LINE_NUM}s|$| \${DEEPSPEED_ARGS}|" "${SFT_SCRIPT}"
        log_ok "Added \${DEEPSPEED_ARGS} to training command"
    fi
fi

echo ""

# Step 4: Verify syntax
log_info "Step 4: Verifying script syntax..."
if bash -n "${SFT_SCRIPT}"; then
    log_ok "Syntax check passed"
else
    log_error "Syntax check failed!"
    log_warn "Restoring backup..."
    cp "${BACKUP}" "${SFT_SCRIPT}"
    exit 1
fi
echo ""

# Step 5: Show modified DeepSpeed section
log_info "Step 5: Modified DeepSpeed configuration:"
echo "---"
grep -n -A2 -B2 "DEEPSPEED" "${SFT_SCRIPT}" || true
echo "---"
echo ""

# Step 6: Show the training command
log_info "Step 6: Training command:"
echo "---"
grep -n "python3.*audio8_tts_sft\|DEEPSPEED_ARGS\|deepspeed" "${SFT_SCRIPT}" || true
echo "---"
echo ""

log_ok "Done! The script is ready for single-GPU training without DeepSpeed."
echo ""
echo "To train, run:"
echo "  cd /home/ubuntu/audio8/Audio8_TTS"
echo "  source /home/ubuntu/audio8/Audio8_TTS/onnx_runtime/.venv/bin/activate"
echo "  TRAIN_JSONL=/home/ubuntu/audio8/audio8-creole/french-model/prepared_data/train.jsonl NPROC_PER_NODE=1 bash audio8_tts_sft.sh"
echo ""
