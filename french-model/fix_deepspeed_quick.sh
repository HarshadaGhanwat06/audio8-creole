#!/bin/bash
set -e

# ============================================================
# Quick DeepSpeed fix - run this on GPU host
# ============================================================

SFT="/home/ubuntu/audio8/Audio8_TTS/audio8_tts_sft.sh"
BAK="/home/ubuntu/audio8/Audio8_TTS/audio8_tts_sft.sh.bak"

echo "=== Backing up ==="
cp "$SFT" "$BAK"
echo "Backup: $BAK"

echo ""
echo "=== Before ==="
grep -n "DEEPSPEED\|deepspeed" "$SFT" || echo "(none)"

echo ""
echo "=== Fixing ==="

# 1. Change default to empty
sed -i 's|DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${PROJECT_ROOT}/configs/deepspeed_zero2.json}"|DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-}"|' "$SFT"

# 2. Remove the unconditional --deepspeed line
sed -i '/--deepspeed "${DEEPSPEED_CONFIG:-${PROJECT_ROOT}\/configs\/deepspeed_zero2.json}"/d' "$SFT"

# 3. Find the python training command line and make deepspeed conditional
# Get line number of the training command
LINE=$(grep -n "python.*audio8_tts_sft\.py" "$SFT" | head -1 | cut -d: -f1)

if [ -n "$LINE" ]; then
    # Insert conditional block before the training command
    sed -i "${LINE}i\\
# DeepSpeed: only use if config is set\\
DS_ARGS=\"\"\\
if [ -n \"\${DEEPSPEED_CONFIG}\" ]; then\\
    DS_ARGS=\"--deepspeed \\\"\${DEEPSPEED_CONFIG}\\\"\"\\
fi" "$SFT"

    # Add ${DS_ARGS} to the training command
    NEW_LINE=$((LINE + 4))
    sed -i "${NEW_LINE}s|python|python|" "${SFT}"
    # Append DS_ARGS at end of that line (before trailing backslash if any)
    sed -i "${NEW_LINE}s|\\\\$| \${DS_ARGS}\\\\|" "$SFT"
fi

echo ""
echo "=== After ==="
grep -n "DEEPSPEED\|deepspeed\|DS_ARGS" "$SFT" || echo "(none)"

echo ""
echo "=== Syntax check ==="
bash -n "$SFT" && echo "PASS" || echo "FAIL"

echo ""
echo "=== Done ==="
