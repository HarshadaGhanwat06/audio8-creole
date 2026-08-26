#!/bin/bash
set -e

# Quick inference test for the fine-tuned French Creole model

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="${SCRIPT_DIR}/outputs"

TEXT=${1:-"Bonjou, sa se yon tès pou modèl Kreyòl Ayisyen."}
REF_AUDIO=${2:-"${REPO_ROOT}/examples/reference.wav"}
REF_TEXT=${3:-"Reference transcript of the voice."}
OUTPUT=${4:-"${SCRIPT_DIR}/outputs/test_output.wav"}

echo "Testing fine-tuned French Creole model..."
echo "  Text:       ${TEXT}"
echo "  Ref audio:  ${REF_AUDIO}"
echo "  Output:     ${OUTPUT}"

cd "${REPO_ROOT}"
python audio8_tts_infer.py \
  --text "${TEXT}" \
  --reference-audio "${REF_AUDIO}" \
  --reference-text "${REF_TEXT}" \
  --output "${OUTPUT}" \
  --model "${MODEL_DIR}"

echo "Done! Output saved to: ${OUTPUT}"
