# Audio8 TTS - French Creole Fine-tuning

Fine-tune Audio8 TTS on a custom French Creole (Kreyòl) dataset.

## Project Structure

```
~/audio8/
├── Audio8_TTS/              # Original model repo (code + model weights)
├── audio8-creole/           # This repo (training scripts)
│   └── french-model/
│       ├── train.sh         # Setup + training script
│       ├── verify.sh        # Quick verification
│       └── README.md
└── kreol/                   # Dataset
    └── data/
        └── worldspeech_mfe_ljspeech/
            └── wavs/        # .wav files
```

## Quick Start

```bash
cd ~/audio8/audio8-creole/french-model

# 1. Verify setup
chmod +x verify.sh
./verify.sh

# 2. Run full setup (installs deps, prepares data)
chmod +x train.sh
./train.sh

# 3. Start training
cd ~/audio8/Audio8_TTS
source ~/audio8/audio8-creole/french-model/.venv/bin/activate
TRAIN_JSONL=../audio8-creole/french-model/prepared_data/train.jsonl \
  NPROC_PER_NODE=1 \
  bash audio8_tts_sft.sh
```

## Dataset Format

The dataset should have:
- `wavs/` folder with `.wav` files
- `metadata.csv` with transcript mapping

**metadata.csv format** (use `|` as separator):
```
filename|text
sample_001.wav|Bonjou kijan ou ye?
sample_002.wav|Mwen renmen Kreyol Ayisyen.
```

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

- Checkpoints: `french-model/outputs/`
- Logs: `french-model/logs/`
- Prepared data: `french-model/prepared_data/`

## Test Fine-tuned Model

```bash
cd ~/audio8/Audio8_TTS
python audio8_tts_infer.py \
  --text "Bonjou, sa se yon tès." \
  --reference-audio examples/reference.wav \
  --reference-text "Reference transcript" \
  --output ../audio8-creole/french-model/outputs/test.wav \
  --model ../audio8-creole/french-model/outputs/
```
