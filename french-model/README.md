# Audio8 TTS - French Creole Fine-tuning

Fine-tune Audio8 TTS on a custom French Creole (Kreyòl) dataset.

## Setup

```bash
# Update DATASET_PATH in train.sh, then:
chmod +x train.sh
./train.sh
```

## Dataset Format

The script expects this structure:

```
/path/to/creole/dataset/
├── audio/          # .wav files (44.1kHz recommended)
│   ├── sample_001.wav
│   ├── sample_002.wav
│   └── ...
└── metadata.csv    # Columns: filename, text
    filename,text
    sample_001.wav,Bonjou kijan ou ye?
    sample_002.wav,Mwen renmen Kreyòl Ayisyen.
```

Adjust the manifest generation in `train.sh` if your format differs.

## Training Parameters

Override defaults via environment variables:

```bash
NPROC_PER_NODE=1 \
BATCH_SIZE=2 \
GRADIENT_ACCUMULATION_STEPS=8 \
MAX_STEPS=5000 \
LEARNING_RATE=1e-5 \
./train.sh
```

## Output

- Checkpoints: `outputs/`
- Logs: `logs/`
- Prepared data: `prepared_data/`

## Test Fine-tuned Model

```bash
cd ..
python audio8_tts_infer.py \
  --text "Bonjou, sa se yon tès." \
  --reference-audio examples/reference.wav \
  --reference-text "Reference transcript" \
  --output outputs/test.wav \
  --model outputs/
```
