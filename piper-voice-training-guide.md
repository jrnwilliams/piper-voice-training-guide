# Train Your Own Voice with Piper TTS — A Complete, Accessible Guide

> **Train a custom text-to-speech voice on any Linux computer, including low-end hardware and CPU-only systems.**
>
> This guide exists because the official Piper training pipeline is **broken and archived** (October 2025). Every fix here was discovered through real troubleshooting on a machine with **no GPU, no NVIDIA graphics, and a standard Fedora Linux installation.**
>
> **Accessible design note:** Written for screen-reader users. All code blocks use plain text (no images). All pitfalls are called out explicitly. Steps are numbered and self-contained. Use screen-reader "jump to heading" navigation to move between sections.

---

## Table of Contents

- [Who This Guide Is For](#who-this-guide-is-for)
- [What You Will End Up With](#what-you-will-end-up-with)
- [Hardware You Need](#hardware-you-need)
- [The Big Picture (How Piper Training Works)](#the-big-picture-how-piper-training-works)
- [Part 1: Install Everything](#part-1-install-everything)
  - [Step 1.1: Install conda](#step-11-install-conda)
  - [Step 1.2: Create the Piper environment](#step-12-create-the-piper-environment)
  - [Step 1.3: Install the Piper training code](#step-13-install-the-piper-training-code)
  - [Step 1.4: Install system dependencies](#step-14-install-system-dependencies)
- [Part 2: Prepare Your Voice Data](#part-2-prepare-your-voice-data)
  - [Step 2.1: Collect and organize audio](#step-21-collect-and-organize-audio)
  - [Step 2.2: Check and convert audio format](#step-22-check-and-convert-audio-format)
  - [Step 2.3: Create the metadata CSV file](#step-23-create-the-metadata-csv-file)
  - [Step 2.4: Verify your data](#step-24-verify-your-data)
- [Part 3: Preprocess (Convert Audio to Training Data)](#part-3-preprocess-convert-audio-to-training-data)
  - [Step 3.1: Choose your language/phonemizer](#step-31-choose-your-languagephonemizer)
  - [Step 3.2: Run the preprocessor](#step-32-run-the-preprocessor)
  - [Step 3.3: Check how many samples are being used](#step-33-check-how-many-samples-are-being-used)
  - [Step 3.4: (Optional) Increase max_phoneme_ids to capture more data](#step-34-optional-increase-max_phoneme_ids-to-capture-more-data)
- [Part 4: Download the Base Model](#part-4-download-the-base-model)
- [Part 5: Train Your Voice](#part-5-train-your-voice)
  - [Step 5.1: Understand how training works on CPU](#step-51-understand-how-training-works-on-cpu)
  - [Step 5.2: Create the resume_training.py script](#step-52-create-the-resume_trainingpy-script)
  - [Step 5.3: Run training](#step-53-run-training)
  - [Step 5.4: Monitor training progress](#step-54-monitor-training-progress)
  - [Step 5.5: When to stop training](#step-55-when-to-stop-training)
  - [Step 5.6: If training dies silently](#step-56-if-training-dies-silently)
- [Part 6: Export and Test Your Voice](#part-6-export-and-test-your-voice)
  - [Step 6.1: Find the latest checkpoint](#step-61-find-the-latest-checkpoint)
  - [Step 6.2: Export to ONNX](#step-62-export-to-onnx)
  - [Step 6.3: Generate test samples](#step-63-generate-test-samples)
- [Part 7: Upload to HuggingFace (Optional)](#part-7-upload-to-huggingface-optional)
  - [Step 7.1: Create a HuggingFace account and token](#step-71-create-a-huggingface-account-and-token)
  - [Step 7.2: Create the repository](#step-72-create-the-repository)
  - [Step 7.3: Upload your models](#step-73-upload-your-models)
- [Part 8: Use Your Voice](#part-8-use-your-voice)
  - [For iOS / iPhone screen readers (VoiceOver)](#for-ios--iphone-screen-readers-voiceover)
  - [For macOS](#for-macos)
  - [For command-line testing on Linux](#for-command-line-testing-on-linux)
- [Troubleshooting](#troubleshooting)
- [Appendix: Every Pitfall in One Place](#appendix-every-pitfall-in-one-place)
- [Appendix: Loss Benchmarks — What to Expect](#appendix-loss-benchmarks--what-to-expect)
- [Glossary](#glossary)

---

## Who This Guide Is For

- **You** have a Linux computer (any distro — Fedora, Ubuntu, Arch, etc.).
- **You** do NOT have an NVIDIA GPU. That is fine. This guide was written on a machine with no GPU at all.
- **You** want to train a custom TTS voice: your own voice, a loved one's voice, a character, or any voice you have recordings of.
- **You** want to use that voice on iPhone with VoiceOver, on a screen reader, or any system that supports Piper ONNX voices.
- **You** may be using a screen reader yourself. This guide is designed to be fully navigable and understandable without visuals.

If you have an NVIDIA GPU with 8GB+ VRAM, some steps will be faster but the process is the same.

---

## What You Will End Up With

By the end of this guide, you will have:

1. **An ONNX model file** — the trained voice, compact enough to share or use anywhere (~63 MB).
2. **A JSON config file** — goes alongside the ONNX. Both files needed to use the voice.
3. **Sample WAV files** — so you can hear what your trained voice sounds like.
4. **(Optional) A HuggingFace repository** — where your voice is stored online for easy downloading.

Your final voice can be used with:
- VoiceOver on iOS/macOS
- eSpeak-NG or any Piper-compatible engine
- Rhasspy / Home Assistant / any open-source TTS system
- The `piper` command-line tool

---

## Hardware You Need

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| **CPU** | Any x86_64 | Quad-core or better | Training is slower on weaker CPUs but still works |
| **RAM** | 4 GB | 8 GB+ | Preprocessing uses little; training needs ~2 GB free |
| **Disk** | 2 GB free | 5 GB free | Base model ~800 MB; checkpoints add up; ONNX files ~63 MB each |
| **GPU** | None required | NVIDIA 8GB+ | Without GPU, training is CPU-only. Works perfectly, just slower |

### Training Time Estimates (CPU-Only)

Based on a real test with 126 samples, Piper "medium" quality:

| Phase | Time |
|-------|------|
| Environment setup | 10-30 minutes |
| Preprocessing | 2-5 minutes |
| Epoch 1-10 | ~35 minutes |
| Epoch 10-50 | ~2.5 hours |
| Epoch 50-100 | ~4.5 hours |
| Epoch 100-500 | ~1.5 days |
| 500 epochs total | ~2-3 days wall time |

You can stop at any point and still have a usable voice. The improvement slows down over time.

---

## The Big Picture (How Piper Training Works)

Think of this like recording a custom voice for a GPS navigator, but smarter. Piper uses a neural network (called VITS) that learns the sound of your voice from short recordings, then can say **anything** — not just the sentences you recorded.

The process is:

1. **Collect audio** — Record or gather clips of the target voice, each with a transcript.
2. **Organize** — Put files in the right folder with a metadata file listing what each clip says.
3. **Preprocess** — Piper converts audio and text into a format the neural network understands (phonemes — the individual sounds of speech).
4. **Fine-tune** — Start from a pre-trained English voice (so Piper already knows how English sounds) and teach it YOUR voice's characteristics. This is the long step.
5. **Export** — Convert the trained model to ONNX, a universal format any device can use.
6. **Test** — Generate sample clips and hear how your voice sounds.

**All of these steps work on CPU.** No GPU required.

---

## Part 1: Install Everything

### Step 1.1: Install conda

conda is a tool that creates isolated Python environments. We need it because Piper requires specific old versions of libraries that conflict with your system Python.

**If you already have conda or miniconda installed, skip to Step 1.2.**

Open a terminal and run:

```bash
# Download miniconda (the lightweight version)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh

# Run the installer (follow the prompts — say "yes" to all defaults)
bash ~/miniconda.sh -b -p $HOME/miniconda3

# Add conda to your shell
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init bash

# Restart your shell or source your profile
source ~/.bashrc
```

Verify it works:

```bash
conda --version
```

You should see something like `conda 24.x.x`.

### Step 1.2: Create the Piper environment

Piper trains with old versions of Python libraries (from 2022-2023). Newer versions break it. conda lets us install these old versions without affecting your system.

```bash
# Create an isolated environment with Python 3.10
conda create -n piper python=3.10 -y

# Activate it
conda activate piper

# THIS IS CRITICAL: downgrade pip before anything else
# pip 24.1 and newer will REFUSE to install the old Piper dependencies
pip install pip==24.0

# Install setuptools that still has pkg_resources (version 72+ removed it)
pip install 'setuptools<72'
```

### Step 1.3: Install the Piper training code

Now install the dependencies in the correct order:

```bash
# You MUST be in the 'piper' conda env for this (your prompt should show "(piper)")

pip install \
    'cython>=0.29.0' \
    'piper-phonemize~=1.1.0' \
    'librosa>=0.9.2' \
    'numpy>=1.19.0,<2' \
    'onnxruntime>=1.11.0' \
    'torch>=1.11.0,<2' \
    'pytorch-lightning==1.7.7' \
    'torchmetrics>=0.11.0,<0.12' \
    'tensorboard>=2.11' \
    'six>=1.15'
```

**If you see ANY errors, do not continue.** Common errors and fixes are at the bottom of this section.

Now clone the Piper source code and build a required C extension:

```bash
# Go to wherever you want to store the Piper source
cd ~
git clone https://github.com/rhasspy/piper.git piper-source

# Build the monotonic_align C extension
cd piper-source/src/python/piper_train/vits/monotonic_align
mkdir -p monotonic_align
cythonize -i core.pyx
mv core*.so monotonic_align/

# Install the piper_train package itself (without re-checking dependencies)
cd ~/piper-source/src/python
pip install -e . --no-deps
```

### Step 1.4: Install system dependencies

You need `ffmpeg` (audio conversion) and `espeak-ng` (text-to-phoneme engine):

**Fedora / RHEL:**
```bash
sudo dnf install ffmpeg espeak-ng inotify-tools -y
```

**Ubuntu / Debian:**
```bash
sudo apt install ffmpeg espeak-ng inotify-tools -y
```

### Verify the installation

Run this to confirm everything is in place:

```bash
conda activate piper
python -m piper_train --help
python -m piper_train.preprocess --help
```

Both should show help text without errors.

### Common Installation Errors (and their fixes)

| Error message | What it means | Fix |
|---|---|---|
| `No matching distribution found for pytorch-lightning==1.7.7` | pip too new | `pip install pip==24.0` then retry |
| `No module named 'pkg_resources'` | setuptools too new | `pip install 'setuptools<72'` then retry |
| `cannot import name '_compare_version'` | torchmetrics version | `pip install 'torchmetrics>=0.11.0,<0.12'` |
| `No module named 'six'` | Missing dependency | `pip install six` |
| `Python.h: No such file or directory` | No Python dev headers | You didn't use conda Python — make sure `conda activate piper` works |
| `cythonize: command not found` | Cython not installed | `pip install cython` then retry building |
| `No module named 'onnxruntime'` | Not installed | `pip install 'onnxruntime>=1.11.0'` |

---

## Part 2: Prepare Your Voice Data

### Step 2.1: Collect and organize audio

You need:
-30 minutes** of audio of the voice (the more the better, but 5 minutes is enough to get started)
- **Transcripts** for every clip — exactly what each clip says

Organize your data like this:

```
my_voice_project/
├── dataset/
│   ├── metadata.csv     # A table of clip IDs and their transcripts
│   └── wav/
│       ├── 001.wav
│       ├── 002.wav
│       ├── 003.wav
│       └── ...
```

Each WAV file should be:
- **One person speaking** (no other voices, no background music)
- **3-15 seconds long** (shorter is fine; very long clips are harder to align)
- Reasonable quality (clean enough to hear clearly)

**Record tips:**
- Quiet room, close to microphone
- Steady volume (not too loud, not too quiet)
- Natural pace — don't rush or draw out syllables
- Vary the content — different sentence types, questions, exclamations

### Step 2.2: Check and convert audio format

Piper requires all audio to be exactly: **22050 Hz, mono (1 channel), 16-bit PCM WAV.**

**Check a file's format:**
```bash
ffprobe -i dataset/wav/001.wav -show_streams -select_streams a -v quiet 2>&1 | grep -E "sample_rate|bits_per_raw_sample|channels"
```

**If it says anything other than:**
- `sample_rate=22050`
- `channels=1`

**Convert all your files at once:**
```bash
cd my_voice_project

# Create output directory for converted files
mkdir -p dataset/wav_mono

# Convert every file at once
for f in dataset/wav/*.wav; do
    ffmpeg -i "$f" -ar 22050 -ac 1 -acodec pcm_s16le "dataset/wav_mono/$(basename "$f")" -y
done
```

The `-ac 1` forces mono (single channel). The `-ar 22050` sets the sample rate. The `-acodec pcm_s16le` sets 16-bit encoding.

**Tip:** If you have very high-quality source recordings, use this for better audio quality during conversion:
```bash
for f in dataset/wav/*.wav; do
    ffmpeg -i "$f" -af "aresample=22050:resampler=soxr" -ac 1 -acodec pcm_s16le "dataset/wav_mono/$(basename "$f")" -y
done
```

After conversion, replace the old `wav/` with the converted one, or point Piper to `wav_mono/` in Step 2.3.

### Step 2.3: Create the metadata CSV file

Create a text file at `dataset/metadata.csv`. This file tells Piper what each audio clip says.

**Format:** Each line contains an ID and a transcript, separated by the pipe character `|`:

```
001|Hello there, this is my voice.
002|I am recording this to train my text to speech model.
003|The quick brown fox jumped over the lazy dog.
```

**CRITICAL RULES for the ID column (most common source of silent failures):**

1. **NO `.wav` extension** — write `001` not `001.wav`
2. **NO directory prefix** — write `001` not `wav/001` and not `dataset/wav/001`
3. **NO quotes** around anything
4. **NO trailing spaces** at end of lines
5. **The transcript** should be plain text — the words spoken in the audio

Your file look exactly like this when viewed with `cat -A` (which shows hidden characters):
```
001|Hello there, this is my voice.$
002|I am recording this to train my text to speech model.$
003|The quick brown fox jumped over the lazy dog.$
```

The `$` at end of each line means "end of line" — it is normal. What you should NOT see is `^I` (tab) or extra spaces before the `$`, or `"` around the line.

**If your existing CSV has problems, fix them with sed:**
```bash
# Remove quotes, .wav extension, directory prefixes, and trailing whitespace:
sed -i 's/^"//;s/"//;s/\.wav//;s|wav/||;s|wav_converted/||;s/[[:space:]]*$//' dataset/metadata.csv
```

### Step 2.4: Verify your data

Run these checks BEFORE proceeding:

```bash
# 1. How many WAV files do you have?
ls dataset/wav/*.wav | wc -l

# 2. How many lines in metadata.csv?
wc -l dataset/metadata.csv

# 3. These two numbers should MATCH. If not, find the missing ones.

# 4. Check first 3 lines of metadata (should show ID|transcript with no quotes)
head -3 dataset/metadata.csv | cat -A
```

---

## Part 3: Preprocess (Convert Audio to Training Data)

"Preprocessing" converts your audio and text into numerical data Piper can learn from. It happens before training and only takes a few minutes.

### Step 3.1: Choose your language/phonemizer

 phonemizer converts your written transcripts into phonemes (the individual sounds of speech). Using the right language setting makes a big difference in your final voice quality.

For **Scottish accents**, use `en-gb-scotland` (NOT plain `en-gb`). espeak-ng has a dedicated Scottish English voice that produces significantly more accurate phonemes.

**Check which English voices are available on your system:**
```bash
espeak-ng --voices=en | grep -i -E "scotland|british|american|english"
```

You should see output like:
```
en-gb-scotland          #5  l/anc M  en-gb-scotland  (...)
en-gb                   #4  l/anc M  en-gb           (...)
en-us                   #3  l/am M  en-us           (...)
```

**Other accent/language options:**
| Language code | For |
|---|---|
| `en-gb-scotland` | Scottish English |
| `en-gb` | British / UK English |
| `en-us` | American English |
| `fr-fr` | French |
| `de` | German |
| `es` | Spanish |
| `it` | Italian |

Find your local voice with `espeak-ng --voices=yt` (replace "yt" with your two-letter language code, e.g. `espeak-ng --voices=fr`).

### Step 3.2: Run the preprocessor

```bash
cd my_voice_project

python -m piper_train.preprocess \
    --input-dir ./dataset/ \
    --output-dir ./preprocessed/ \
    --language en-gb-scotland \
    --sample-rate 22050 \
    --dataset-format ljspeech \
    --single-speaker \
    --max-workers 4
```

Replace `en-gb-scotland` with your chosen language code.

You will see progress output. It should finish in 1-5 minutes depending on your dataset size.

### Step 3.3: Check how many samples are being used

After preprocessing, check the output folder:

```bash
ls -la preprocessed/
```

You should see:
- `config.json` — model configuration
- `dataset.jsonl` — the main training data (one JSON object per line)
- `cache/` — processed audio features

### Step 3.4: (Optional) Increase max_phoneme_ids to capture more data

**This is a common issue:** The default `max_phoneme_ids=400` silently DROPS longer recordings. If you have 126 clips, 20-25 might be thrown away because their transcripts are long.

**Check your distribution first:**
```bash
python3 -c "
import json

with open('./preprocessed/dataset.jsonl') as f:
    lines = f.readlines()

counts = [len(json.loads(l)['phoneme_ids']) for l in lines]
counts.sort()

print(f'Total utterances: {len(counts)}')
print(f'Min phonemes: {min(counts)}')
print(f'Max phonemes: {max(counts)}')
print(f'Median: {counts[len(counts)//2]}')
print()
print('How many files would be DROPPED at each threshold:')
for t in [200, 300, 400, 500, 600, 700]:
    dropped = sum(1 for c in counts if c > t)
    included = len(counts) - dropped
    print(f'  max_phoneme_ids={t}:{included:3d} kept
{3}d dropped')
"
```

**Output example:**
```
max_phoneme_ids=400: 101 kept, 25 dropped
max_phoneme_ids=500: 115 kept, 11 dropped
max_phoneme_ids=600: 122 kept,  4 dropped
```

**To use more of your data, reprocess with a higher limit:**
```bash
# Change 600 to whatever fits your data
python -m piper_train.preprocess \
    --input-dir ./dataset/ \
    --output-dir ./preprocessed/ \
    --language en-gb-scotland \
    --sample-rate 22050 \
    --dataset-format ljspeech \
    --single-speaker \
    --max-workers 4
    # Note: If using the command-line, pass --max-phoneme-ids 600
```

**Reprocessing is very fast** — just a few seconds even for hundreds of files. There is no downside to capturing more data if you have enough samples.

---

## Part 4: Download the Base Model

Piper doesn't learn a voice from scratch — it starts with a pre-trained English voice (that already knows how to speak English) and customizes it to your voice. This is called **fine-tuning.**

**Download the base checkpoint:**
```bash
cd my_voice_project

mkdir -p checkpoints
cd checkpoints

wget "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/lessac/epoch-2164-2023-06-19_21-43-29.ckpt"
```

This checkpoint is ~800MB. The download may take a few minutes depending on your connection.

**The downloaded file will likely be named something like `epoch-2164-ckpt` — rename it properly:**
```bash
cd my_voice_project/checkpoints

python3 -c "
import os, hashlib

# Find the downloaded file
files = os.listdir('.')
for f in files:
    if 'epoch' in f and f.endswith('.ckpt'):
        new_name = 'epoch=2164-step=1355540.ckpt'
        os.rename(f, new_name)
        print(f'Renamed {f} to {new_name}')
"
```

**IMPORTANT:** This file (`epoch=2164-step=1355540.ckpt`) is the **BASE PRETRAINED MODEL**, not your trained voice. When you look for the "latest checkpoint" later, ALWAYS exclude this one. Your trained checkpoints will have different filenames.

---

## Part 5: Train Your Voice

This is the long part. Training reads your data, learns your voice's characteristics, and saves checkpoints periodically. You can stop at any time and resume later.

### Step 5.1: Understand how training works on CPU

**Important context for CPU-only training (no GPU):**

PyTorch Lightning's `--resume_from_checkpoint` flag **HANGS INDEFINITELY on CPU** with large checkpoint files. This is a known bug. The workaround is to load the checkpoint manually in Python and bypass Lightning's resumption system entirely.

Additionally, when loading manually, you MUST restore the optimizer states (the neural network's "momentum") or your loss will spike from ~20 to ~35+. This is not obvious — the training will appear to work but be much worse.

The `resume_training.py` script below handles both of these issues automatically.

### Step 5.2: Create the resume_training.py script

Create a file called `resume_training.py` in your project folder (`~/my_voice_project/resume_training.py`):

```python
#!/usr/bin/env python3
"""
Piper TTS training script for CPU.
Handles checkpoint loading, optimizer restoration, and detached training.

CRITICAL: This script loads checkpoints MANUALLY because
PyTorch Lightning's --resume_from_checkpoint hangs on CPU.
See: https://github.com/rhasspy/piper (archived Oct 2025)
"""

import sys
import os
import json
import torch
from pytorch_lightning import Trainer, callbacks
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from piper_train.vits.lightning import VitsModel

# ============================================================
# CONFIGURATION — change these for your project
# ============================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSED_DIR = os.path.join(PROJECT_DIR, "preprocessed")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "training_output")

# The BASE pretrained model to start from (epoch=2164)
# For the FIRST run, point to this. For RESUMING, point to your latest checkpoint.
BASE_CHECKPOINT = os.path.join(PROJECT_DIR, "checkpoints", "epoch=2164-step=1355540.ckpt")

# Training parameters
BATCH_SIZE = 4          # Reduce if you get memory errors (2 or 1)
MAX_PHONEME_IDS = 600   # Set to 400 to process faster, 600 to capture more data
QUALITY = "medium"      # or "high"
NUM_WORKERS = 0         # MUST be 0 on CPU — any other value causes deadlock
# ============================================================


def find_latest_checkpoint():
    """Find the most recent checkpoint file in the training output."""
    latest_ckpt = None
    latest_mtime = 0

    for root, dirs, files in os.walk(OUTPUT_DIR):
        for f in files:
            if f.endswith(".ckpt") and "epoch=2164" not in f:
                path = os.path.join(root, f)
                mtime = os.path.getmtime(path)
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    latest_ckpt = path

    return latest_ckpt


def get_config():
    """Load config from preprocessed data."""
    config_path = os.path.join(PREPROCESSED_DIR, "config.json")
    with open(config_path) as f:
        return json.load(f)


class RestoreOptimizerStateCallback(Callback):
    """
    Restores optimizer and LR scheduler states at WITHOUT this callback, loss spikes from ~20 to ~35+ when resuming.
    BOTH optimizer_states AND lr_schedulers must be restored.
    """

    def __init__(self, checkpoint):
        self.checkpoint = checkpoint

    def on_train_start(self, trainer, pl_module):
        # Restore optimizer states (there are 2: generator + discriminator)
        if 'optimizer_states' in self.checkpoint and len(self.checkpoint['optimizer_states']) > 0:
            for i, opt_state in enumerate(self.checkpoint['optimizer_states']):
                if i < len(trainer.optimizers):
                    trainer.optimizers[i].load_state_dict(opt_state)
                    print(f"  Restored optimizer {i} state")

        # Restore LR scheduler states (keeps the learning rate schedule correct)
        if 'lr_schedulers' in self.checkpoint and len(self.checkpoint['lr_schedulers']) > 0:
            for i, sched_cfg in enumerate(trainer.lr_scheduler_configs):
                if i < len(self.checkpoint['lr_schedulers']):
                    sched_cfg.scheduler.load_state_dict(self.checkpoint['lr_schedulers'][i])
                    print(f"  Restored LR scheduler {i} state")


def main():
    config = get_config()
    num_symbols = config["num_symbols"]
    num_speakers = config.get("num_speakers", 1)
    sample_rate = config.get("audio", {}).get("sample_rate", 22050)

    # Decide which checkpoint to load
    latest_ckpt = find_latest_checkpoint()

    if latest_ckpt:
        print(f"Resuming from: {latest_ckpt}")
        checkpoint_path = latest_ckpt
    else:
        print(f"Starting from base model: {checkpoint_path}")
        checkpoint_path = BASE_CHECKPOINT

    # Create the model (WITHOUT resume_from_checkpoint — that hangs on CPU)
    model = VitsModel(
        num_symbols=num_symbols,
        num_speakers=num_speakers,
        sample_rate=sample_rate,
        dataset=[os.path.join(PREPROCESSED_DIR, "dataset.jsonl")],
        batch_size=BATCH_SIZE,
        max_phoneme_ids=MAX_PHONEME_IDS,
        num_workers=NUM_WORKERS,
        num_test_examples=0,
        validation_split=0.0,
        quality=QUALITY,
        seed=1234,
    )

    # Load checkpoint manually
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["state)
    print(f"  Loaded {len(ckpt['state_dict'])} parameters")

    # Create the optimizer restoration callback
    restore_callback = RestoreOptimizerStateCallback(ckpt)

    # Create checkpoint saver (saves every 5 epochs, keeps all)
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=5,
        save_top_k=-1,
        save_on_train_epoch_end=True,
    )

    # Create the trainer
    trainer = Trainer(
        max_epochs=-1,  # -1 = infinite (avoids epoch-count edge cases)
        accelerator="cpu",
        precision=32,
        num_sanity_val_steps=0,  # MUST be 0 on CPU — hangs otherwise
        enable_progress_bar=True,
        enable_model_summary=False,
        log_every_n_steps=1,
        default_root_dir=OUTPUT_DIR,
        callbacks=[checkpoint_callback, restore_callback],
    )

    print("Starting training...")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Max phoneme IDs: {MAX_PHONEME_IDS}")
    print(f"  Quality: {QUALITY}")
    print(f"  Output: {OUTPUT_DIR}")
    print()

    trainer.fit(model)


if __name__ == "__main__":
    main()
```

### Step 5.3: Run training

**IMPORTANT: Use the direct Python path, NOT `conda run`.** `conda run` buffers all output and makes training appear frozen when it is actually working.

```bash
cd my_voice_project

# Find your conda Python path
which python
# Should show something like: /home/YOURNAME/miniconda3/envs/piper/bin/python

# Run training (detached so it survives terminal/session closure)
nohup setsid /home/YOURNAME/miniconda3/envs/piper/bin/python -u resume_training.py \
    > training_output/training.log 2>&1 &

echo "Training started. PID: $!"
```

**Replace `/home/YOURNAME/miniconda3/envs/piper/bin/python`** with the actual path from `which python`.

**What this does:**
- `nohup setsid` — detaches training from your terminal so it keeps running even if you close the terminal or log out
- `-u` — unbuffered Python output (so you see progress in real time)
- `> training_output/training.log 2>&1` — saves all output to a log file
- `&` — runs in background

### Step 5.4: Monitor training progress

**Check if training is still running:**
```bash
ps aux | grep resume_training | grep -v grep
```

**See the latest progress (most reliable method):**
```bash
tail -1 training_output/training.log | tr '\r' '\n' | grep "Epoch" | tail -1
```

**Watch the full log in real time:**
```bash
tail -f training_output/training.log
```

**What you should see in the output:**
```
Restored optimizer 0 state
Restored optimizer 1 state
Restored LR scheduler 0 state
Restored LR scheduler 1 state
Epoch 0:  100%|██████████| 31/31 [03:20<00:00,  6.48s/it, loss=20.7]
```

The `loss` number is the key metric. Lower is better. See the benchmarks below.

### Step 5.5: When to stop training

**Loss benchmarks (from a real 126-sample run, medium quality, CPU):**

| Cumulative Epoch | Expected Loss | Quality Level |
|---|---|---|
| 0 (start) | ~36 | Unusable — just starting |
| 1 | ~24-27 | Robotic but recognizable |
| 5 | ~22-23 | Clearly the voice, some artifacts |
| 10 | ~21-22 | Usable, improving |
| 20-50 | ~20-21 | Good quality |
| 50-100 | ~19-20 | Very good |
| 100-200 | ~18-20 | Excellent |
| 200-500 | ~18-20 | Plateau — diminishing returns |

**When to stop:**
- If loss has been oscillating in the same range for **50+ epochs** without a new low, you are near the quality ceiling for your dataset size.
- For small datasets (~120 samples), the model typically plateaus around loss 18-20.
- You can export and test at ANY point — even early checkpoints produce intelligible speech.

**To stop training gracefully:**
```bash
# Find the process
ps aux | grep resume_training | grep -v grep

# Send SIGTERM (lets it finish the current batch and save a checkpoint)
kill -TERM PID
# Replace PID with the actual process number
```

### Step 5.6: If training dies silently

Training processes can sometimes stop without any error message. The log just ends mid-epoch. This is a known issue with PyTorch Lightning on CPU.

**Check if training is alive:**
```bash
ps aux | grep resume_training | grep -v grep
```

**Check if the log is still updating:**
```bash
stat -c '%Y' training_output/training.log
# This shows the last modification time as a Unix timestamp.
# Compare to current time: date +%s
# If the difference is more than ~5 minutes, training has likely died.
```

**If training died, just restart it:**
```bash
cd my_voice_project
nohup setsid /home/YOURNAME/miniconda3/envs/piper/bin/python -u resume_training.py \
    > training_output/training.log 2>&1 &
```

The script automatically finds the latest checkpoint and resumes from there. No data is lost.

---

## Part 6: Export and Test Your Voice

### Step 6.1: Find the latest checkpoint

**IMPORTANT: Do NOT use `epoch=2164-step=1355540.ckpt` — that is the base model, not your trained voice.**

```bash
cd my_voice_project

# Find the latest trained checkpoint (excludes the base model)
find training_output/lightning_logs/ -name "*.ckpt" -not -name "epoch=2164*" -printf '%T@ %p\n' | sort -n | tail -1
```

**If you see duplicate epoch numbers (e.g., `epoch=239-step=14880.ckpt` AND `epoch=239-step=14880-v1.ckpt`):**
- The `-v1` suffix is the NEWER one. Lightning adds this to avoid overwriting.
- Always use the `-v1` (or highest suffix) version.

### Step 6.2: Export to ONNX

```bash
cd my_voice_project

# Create output directory
mkdir -p output

# Replace with your actual checkpoint path
CHECKPOINT="training_output/lightning_logs/version_0/checkpoints/epoch=9-step=520.ckpt"

python -m piper_train.export_onnx \
    "$CHECKPOINT" \
    ./output/my_voice.onnx

# Copy the config file (REQUIRED — the ONNX won't work without it)
cp preprocessed/config.json ./output/my_voice.onnx.json
```

You now have two files:
- `output/my_voice.onnx` — the trained voice model (~63 MB)
- `output/my_voice.onnx.json` — the configuration (~7 KB)

**Both files are needed to use the voice.**

### Step 6.3: Generate test samples

Create a script to generate sample audio clips from your trained voice:

```bash
cd my_voice_project

cat > generate_samples.py << 'SCRIPT'
#!/usr/bin/env python3
"""Generate test audio samples from a trained Piper ONNX model."""

import json
import wave
import numpy as np
import onnxruntime as ort

ONNX_PATH = "./output/my_voice.onnx"
JSON_PATH = "./output/my_voice.onnx.json"
DATASET_PATH = "./preprocessed/dataset.jsonl"
OUTPUT_DIR = "./output/samples"

# Create output directory
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
print(f"Loading model: {ONNX_PATH}")
sess = ort.InferenceSession(ONNX_PATH)

# Load config for sample rate
with open(JSON_PATH) as f:
    config = json.load(f)
sample_rate = config.get("audio", {}).get("sample_rate", 22050)

# Load dataset to get phoneme IDs
with open(DATASET_PATH) as f:
    lines = f.readlines()

# Pick diverse samples (first, quarter, middle, three-quarter, last)
indices = [0, len(lines)//4, len(lines)//2, 3*len(lines)//4, len(lines)-1]
indices = list(set(indices))  # Remove duplicates if dataset is small

for idx in indices:
    d = json.loads(lines[idx])
    phoneme_ids = d["phoneme_ids"]
    text = d.get("text", "")

    # Truncate if needed (shouldn't be necessary with max_phoneme_ids=600)
    phoneme_ids = phoneme_ids[:600]

    # Run inference
    input_dict = {
        "input": np.array([phoneme_ids], dtype=np.int64),
        "input_lengths": np.array([len(phoneme_ids)], dtype=np.int64),
        "scales": np.array([0.667, 1.0, 0.8], dtype=np.float32),
    }

    audio = sess.run(None, input_dict)[0].squeeze()

    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)

    # Save WAV
    out_path = os.path.join(OUTPUT_DIR, f"sample_{idx}.wav")
    with wave.open(out_path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    duration = len(audio_int16) / sample_rate
    print(f"  {out_path} ({duration:.1f}s): {text[:60]}")

print(f"\n{len(indices)} samples saved to {OUTPUT_DIR}/")
SCRIPT
```

Run it:
```bash
/home/YOURNAME/miniconda3/envs/piper/bin/python generate_samples.py
```

**Listen to your samples:**
```bash
# On Linux with PulseAudio:
paplay output/samples/sample_0.wav

# Or copy to your phone/iPhone via email, cloud storage, etc.
```

---

## Part 7: Upload to HuggingFace (Optional)

Sharing your voice on HuggingFace makes it easy for others (or yourself on another device) to download.

### Step 7.1: Create a HuggingFace account and token

1. Go to https://huggingface.co and create a free account
2. Go to https://huggingface.co/settings/tokens
3. Click "Create new token"
4. Name it something like "piper-voices"
5. Select "Write" permission
6. Copy the token (it starts with `hf_`)

### Step 7.2: Create the repository

```bash
# Login (replace with your actual token — no quotes around it in the command)
hf auth login --token hf_YOURACTUALTOKENHERE

# Verify login
hf auth whoami

# Create the repository (replace 'yourusername' with your HF username)
curl -s -X POST "https://huggingface.co/api/repos/create" \
  -H "Authorization: Bearer hf_YOURACTUALTOKENHERE" \
  -H "Content-Type: application/json" \
  -d '{"name": "my-custom-voice", "type": "model", "private": true}'
```

Set `private: false` if you want the voice to be publicly available.

### Step 7.3: Upload your models

```bash
cd my_voice_project

# Create a temp directory with just the files to upload
mkdir -p hf_upload
cp output/my_voice.onnx hf_upload/
cp output/my_voice.onnx.json hf_upload/
cp -r output/samples hf_upload/

# Upload everything
hf upload yourusername/my-custom-voice hf_upload/ --repo-type model --include "*"
```

Your voice is now at: `https://huggingface.co/yourusername/my-custom-voice`

---

## Part 8: Use Your Voice

### For iOS / iPhone screen readers (VoiceOver)

Piper ONNX voices can be used with apps that support custom TTS voices on iOS. The general process:

1. Download your `.onnx` and `.onnx.json` files to your iPhone (via Files app, AirDrop, or download from HuggingFace)
2. Use a TTS app that supports Piper ONNX models (such as "Piper TTS" apps or custom VoiceOver configurations)
3. Select your voice in Settings > Accessibility > Spoken Content > Voices

**Note:** Apple's built-in VoiceOver does not directly support ONNX files. You may need a third-party TTS app or a shortcut that calls a Piper-compatible engine. Check the Piper community for the latest iOS integration options.

### For macOS

1. Download the `.onnx` and `.onnx.json` files
2. Use the `piper` command-line tool on macOS, or integrate with a TTS app that supports Piper
3. In System Settings > Accessibility > Spoken Content, you can add custom voices

### For command-line testing on Linux

```bash
# Install the piper binary (separate from piper_train)
# Download from: https://github.com/rhasspy/piper/releases

# Generate speech from text
echo "Hello, this is my custom voice speaking." | \
    ./piper -m output/my_voice.onnx --output_file test.wav

# Play it
paplay test.wav
```

---

## Troubleshooting

### "Training hangs / no output for a long time"

**Cause:** You used `conda run` instead of the direct Python path.
**Fix:** Kill the process and restart with the direct path:
```bash
killall -9 python
nohup setsid /path/to/conda/envs/piper/bin/python -u resume_training.py > training_output/training.log 2>&1 &
```

### "Loss is ~36 and not decreasing"

**Cause:** You are training from the base model (epoch=2164) instead of your trained checkpoint, OR the optimizer state was not restored.
**Fix:** Check that the script is finding your latest checkpoint. Look for "Resuming from:" in the log. If it says "Starting from base model" when you expected to resume, check that your `training_output/lightning_logs/` directory has checkpoint files.

### "Loss jumped to ~35 after resuming"

**Cause:** Optimizer states were not restored. The `RestoreOptimizerStateCallback` is not working.
**Fix:** Check the log for "Restored optimizer 0 state" messages. If they are missing, the checkpoint may not contain optimizer states (some early checkpoints don't). Try a later checkpoint.

### "ModuleNotFoundError: No module named 'piper_train'"

**Cause:** You are not in the `piper` conda environment, or `pip install -e . --no-deps` was not run.
**Fix:**
```bash
conda activate piper
cd ~/piper-source/src/python
pip install -e . --no-deps
```

### "RuntimeError: CUDA out of memory"

**Cause:** Batch size is too large for your:** Reduce `BATCH_SIZE` in `resume_training.py` from 4 to 2 or 1.

### "DataLoader worker exited unexpectedly" or training hangs during data loading

**Cause:** `num_workers` is not set to 0 on CPU.
**Fix:** Ensure `NUM_WORKERS = 0` in `resume_training.py`.

### "struct.error: ushort format requires 0 <= number <= 65535" when saving WAV

**Cause:** Using Piper's built-in `wavfile.write()` which has a bug with certain audio values.
**Fix:** Use Python's standard `wave` module instead (as shown in the sample generation script).

### "All phoneme IDs are 0" / "Model produces silence or noise"

**Cause:** Using espeak-ng IPA output instead of the phoneme IDs from `dataset.jsonl`.
**Fix:** Always use `phoneme_ids` from the preprocessed `dataset.jsonl` file. The IPA symbols from espeak-ng are NOT the same as Piper's internal phoneme IDs.

### "FileNotFoundError: cache/something.pt"

**Cause:** You renamed the preprocessed directory but the paths inside `dataset.jsonl` still point to the old location.
**Fix:** Either rename back, or re-run preprocessing, or fix paths in `dataset.jsonl`:
```python
import json

with open('preprocessed/dataset.jsonl') as f:
    lines = f.readlines()

with open('preprocessed/dataset.jsonl', 'w') as f:
    for line in lines:
        obj = json.loads(line)
        obj['audio_norm_path'] = obj['audio_norm_path'].replace('old_dir/', 'preprocessed/')
        obj['audio_spec_path'] = obj['audio_spec_path'].replace('old_dir/', 'preprocessed/')
        f.write(json.dumps(obj) + '\n')
```

### "Duplicate epoch numbers — which checkpoint is newer?"

**Cause:** Lightning resets the epoch counter to 0 on each resume. Two different runs can produce checkpoints with the same epoch number.
**Fix:** The `-v1` suffix is the newer one. Or check modification time:
```bash
ls -la training_output/lightning_logs/version_*/checkpoints/epoch=239*.ckpt
```

---

## Appendix: Every Pitfall in One Place

This is a summary of every known issue, collected in one place for quick reference.

### Installation Pitfalls

1. **pip 24.1+ rejects old metadata** — `pytorch-lightning==1.7.7` uses `torch (>=1.9.*)` which newer pip rejects. Fix: `pip install pip==24.0` FIRST.
2. **setuptools ≥72 removed pkg_resources** — pytorch-lightning imports this at startup. Fix: `pip install 'setuptools<72'`.
3. **torchmetrics version mismatch** — `_compare_version` moved in newer versions. Fix: `pip install 'torchmetrics>=0.11.0,<0.12'`.
4. **Python 3.12+ incompatible** — Piper targets 3.7-3.9. Fix: Use conda with Python 3.10.
5. **monotonic_align needs Python.h** — The C extension needs dev headers. Fix: Use conda Python (includes headers).
6. **six not auto-installed** — tensorboard needs it but doesn't declare it. Fix: `pip install six`.

### Data Preparation Pitfalls

7. **CSV format must be bare IDs** — `001|text` NOT `wav/001.wav|text`. The preprocessor prepends `wav/` internally.
8. **Quotes break parsing** — `"001|text"` will not parse correctly. Strip all quotes.
9. **Trailing whitespace causes path issues** — Strip trailing spaces from CSV lines.
10. **Audio must be 22050 Hz mono 16-bit** — Anything else will cause errors or poor quality.
11. **max_phoneme_ids=400 drops long utterances** — 20-25% of your data may be silently excluded. Increase to 600.

### Training Pitfalls

12. **NEVER use `--resume_from_checkpoint` on CPU** — It hangs indefinitely. Load manually with `torch.load()`.
13. **Must restore optimizer states** — Without them, loss spikes from ~20 to ~35+. Use the `RestoreOptimizerStateCallback`.
14. **Must restore LR scheduler states** — Without them, the learning rate schedule resets.
15. **num_workers MUST be 0 on CPU** — Any other value causes DataLoader deadlock.
16. **num_sanity_val_steps MUST be 0 on CPU** — Sanity validation hangs.
17. **conda run buffers all output** — Use direct Python path: `/path/to/conda/envs/piper/bin/python`
18. **Training can die silently** — No error message, just stops. Check with `ps aux | grep python`.
19. **Training dies when terminal closes** — Use `nohup setsid` to fully detach.
20. **epoch=2164 is the BASE MODEL** — Never confuse this with your trained checkpoints.
21. **Duplicate epoch numbers across restarts** — The `-v1` suffix is the newer checkpoint.
22. **ModelCheckpoint(every_n_epochs=N) is unreliable** — May not fire. Use `every_n_epochs=5` with `save_top_k=-1`.
23. **max_epochs must be > checkpoint epoch** — Or Lightning throws MisconfigurationException. Use `max_epochs=-1` (infinite).

### Inference/Export Pitfalls

24. **ONNX output shape is (1,1,1,N)** — Must `.squeeze()` before saving.
25. **Use standard wave module** — NOT `piper_train.vits.wavfile.write()` (has struct.error bug).
26. **Multiply by 32767 for int16** — Audio is float [-1,1], must convert to 16-bit range.
27. **Use phoneme_ids from dataset.jsonl** — NOT espeak-ng IPA output (different symbol set).
28. **ONNX input names are `input`, `input_lengths`, `scales`** — NOT `phoneme_ids` or `text`.

### HuggingFace Pitfalls

29. **Use `hf` CLI, not `huggingface-cli`** — The latter is deprecated.
30. **Token must be direct argument** — `hf auth login --token "hf_xxx"` NOT via stdin pipe.
31. **Tokens get truncated by chat interfaces** — If your token contains `...`, it has been cut off.
32. **Upload ONNX + JSON pairs** — Raw .ckpt files (~800MB) are usually not needed; ONNX (~63MB) is what users need.

---

## Appendix: Loss Benchmarks — What to Expect

These numbers are from a real training run: 126 samples, Piper medium quality, CPU-only, batch_size=4, starting from the en_US base model (epoch=2164).

| Cumulative Epoch | End-of-Epoch Loss | Wall Time (cumulative) | What to Expect |
|---|---|---|---|
| 0 (first batch) | ~36 | — | Random noise |
| 1 | ~24-27 | minutes | Robotic but recognizable as the voice |
| 5 | ~22-23 | ~17 minutes | Clear voice, some artifacts |
| 10 | ~21-22 | ~33 minutes | Usable quality |
| 15 | ~22.7 | ~50 minutes | Oscillation is NORMAL — look at the trend |
| 29 | ~21.0 | ~90 minutes | Good baseline |
| 50 | ~20-21 | ~3 hours | Solid quality |
| 84 | ~20.4 | ~8 hours | Very good |
| 100 | ~19-20 | ~1 day | Excellent |
| 200 | ~18-20 | ~2 days | Near ceiling for this dataset |
| 500 | ~18.5-19.5 | ~4-5 days | Plateau — diminishing returns |

**Key points:**
- Loss OSCILLATES epoch to epoch. This is normal. Look at the overall trend, not individual values.
- A single-batch dip to ~17.8 after resuming is a TRANSIENT ARTIFACT of optimizer restoration, not real improvement.
- For small datasets (~120 samples), loss plateaus around 18-20. Going lower requires dramatically more time.
- You can export and test at ANY point. Even epoch 10 produces intelligible speech.

---

## Glossary

| Term | Meaning |
|---|---|
| **Checkpoint** | A saved snapshot of the model at a specific point in training. Lets you resume or go back to an earlier version. |
| **ckpt file** | A checkpoint file (PyTorch format). Large (~800MB). Contains model weights, optimizer state, and scheduler state. |
| **ONNX** | Open Neural Network Exchange — a universal model format. The `.onnx` file is your final trained voice (~63MB). |
| **Epoch** | One complete pass through all your training data. Training runs for many epochs. |
| **Loss** | A number measuring how wrong the model is. Lower = better. The main progress indicator. |
| **Phoneme** | An individual sound of speech. "Hello" = /h/ /ə/ /l/ /oʊ/. Piper converts text to phonemes before generating audio. |
| **Phonemizer** | The tool that converts written text to phonemes. Piper uses espeak-ng. |
| **Fine-tuning** | Starting from a pre-trained model and adapting it to your specific voice. Much faster than training from scratch. |
| **Base model** | The pre-trained English voice you start from. NOT your trained voice. |
| **conda** | A tool for creating isolated Python environments. Lets you install old library versions without breaking your system. |
| **espeak-ng** | The speech synthesis engine Piper uses for phoneme conversion. |
| **VITS** | The neural network architecture Piper uses (Variational Inference with adversarial learning for end-to-end Text-to-Speech). |
| **Lightning** | PyTorch Lightning — the training framework Piper uses. Version 1.7.7 has several CPU bugs. |
| **max_phoneme_ids** | Maximum number of phonemes per utterance. Higher = more data included, but longer utterances use more memory. |
| **batch_size** | How many samples are processed at once. Lower = less memory, slightly slower. |
| **num_workers** | Number of parallel data loading processes. MUST be 0 on CPU. |
| **HuggingFace** | An online platform for sharing models and datasets. Where you can upload your trained voice. |

---

## Credits and License

This guide was developed through extensive real-world troubleshooting on a CPU-only Linux system (Fedora 44, no GPU). Every fix was discovered empirically because the official Piper repository was archived in October 2025 and all official documentation is now broken.

**You are free to:**
- Share this guide
- Modify it for your needs
- Post it on GitHub, forums, or anywhere else
- Translate it into other languages
- Use it to help others train their own voices

**Credit appreciated but not required.** If you do credit, you can credit the Piper TTS community and the original Piper project by rhasspy.

**License:** This guide is released into the public domain (CC0). Use it however you want.

---

## Further Resources

- **Piper source code (archived):** https://github.com/rhasspy/piper
- **Piper checkpoints:** https://huggingface.co/datasets/rhasspy/piper-checkpoints
- **espeak-ng voices:** Run `espeak-ng --voices=en` to see available English voices
- **HuggingFace Hub:** https://huggingface.co (for sharing your trained voice)
- **Community:** Search for "Piper TTS" on GitHub Issues for community support

---

*Last updated: June 2026. Tested on Fedora 44, Python 3.10, Piper medium quality, CPU-only training with 126 samples.*
