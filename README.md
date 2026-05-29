# finetuning_injector_file

A Python script that fine-tunes **Gemma 4 E2B** on an Igbo-English dictionary dataset using **QLoRA** (Quantized Low-Rank Adaptation).

---

## What it does

1. Loads a cleaned Igbo-English dictionary CSV file
2. Formats each row into a translation training example
3. Loads Gemma 4 E2B compressed in 4-bit to save GPU memory
4. Applies LoRA adapters on top of the model
5. Fine-tunes the model on the Igbo data
6. Saves the fine-tuned model to a local folder

After fine-tuning, Gemma becomes significantly better at translating between English and Igbo.

---

## Requirements

Run this once in Colab before using the script:

```bash
pip install transformers datasets trl peft accelerate bitsandbytes pyarrow
```

> This script is designed to run on **Google Colab** (free tier) with a T4 GPU.

---

## Dataset Format

Your CSV file must have exactly two columns — `en` for English and `ig` for Igbo:

| en | ig |
|---|---|
| when one enters | a ba a |
| good night | abali oma |
| I love you | a fum gi nanya |
| name of a State in Igbo land | a bia |

### Preparing your dataset
If your CSV has an unwanted index column, clean it first:

```python
import pandas as pd
df = pd.read_csv("your_file.csv")
df = df.drop(columns=["Unnamed: 0"])
df.to_csv("igbo_clean.csv", index=False)
```

---

## How to use it

**Step 1** — Upload your `corpus file` to Colab using the Files panel on the left

**Step 2** — Run the install cell:
```bash
pip install transformers datasets trl peft accelerate bitsandbytes pyarrow
```

**Step 3** — Run the script and wait for training to complete

**Step 4** — The fine-tuned model is saved to `./igbo-gemma4-finetuned/`

---

## How it works (step by step)

### Step 1 — Load data
```python
df = pd.read_csv(input corpus file dir)
df = df.dropna()
```
Reads the CSV and removes any empty rows that could confuse the model.

### Step 2 — Format data
Each row is turned into a training example:
```
Translate English to Igbo.
English: good night
Igbo: abali oma
```
Gemma reads thousands of these examples and learns the translation pattern between English and Igbo.

### Step 3 — Compress model
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
```
Loads Gemma in 4-bit mode. This reduces memory usage by 4x so the model fits in Colab's free GPU.

### Step 4 — Apply LoRA
```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    exclude_modules=["vision_tower", "multi_modal_projector", "audio_tower"]
)
```
Instead of retraining all 5 billion parameters — which would take days — LoRA adds small trainable layers on top of Gemma and only trains those. The result is less than 1% of parameters being trained, but with similar results to full training.

> **Note:** `target_modules="all-linear"` is required for Gemma 4. Using specific layer names like `q_proj` or `v_proj` causes a `Gemma4ClippableLinear` error due to Gemma 4's custom layer architecture.

### Step 5 — Train
```python
trainer.train()
```
Runs through all examples 3 times (3 epochs). During training you will see:

```
Step    Training Loss
50      2.869023
100     1.954745
150     1.623401
```

The loss number should drop over time. Lower loss = the model is learning better.

### Step 6 — Save
```python
model.save_pretrained("./igbo-gemma4-finetuned")
tokenizer.save_pretrained("./igbo-gemma4-finetuned")
```
Saves the LoRA adapter and tokenizer to a folder. This is your fine-tuned model — ready to be used for translation.

---

## Key Configuration Settings

| Setting | Value | Meaning |
|---|---|---|
| `num_train_epochs` | 3 | Goes through all data 3 times |
| `per_device_train_batch_size` | 2 | Processes 2 examples at a time |
| `gradient_accumulation_steps` | 4 | Effectively 8 examples per update |
| `learning_rate` | 2e-4 | How fast the model adjusts |
| `lora_r` | 16 | Size of the LoRA layers |
| `lora_alpha` | 32 | Strength of LoRA influence |
| `lora_dropout` | 0.05 | Prevents overfitting |

---

## Why QLoRA?

Training all 5 billion parameters of Gemma from scratch would require:
- 80GB+ of GPU memory
- Days of compute time
- Expensive hardware

QLoRA solves this by combining two techniques:
- **Quantization** — compresses the model to 4-bit (4x less memory)
- **LoRA** — only trains a small set of extra layers (less than 1% of parameters)

The result is fine-tuning that runs on a free Colab GPU in a few hours.

---

## Hardware Requirements

| Resource | Minimum |
|---|---|
| GPU | NVIDIA T4 (15GB) — Colab free tier |
| RAM | 12GB |
| Disk | 20GB free |
---

## Built With

- [Gemma 4 E2B](https://huggingface.co/google/gemma-4-e2b-it) — Google's open language model
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) — model loading
- [PEFT](https://huggingface.co/docs/peft) — LoRA fine-tuning
- [TRL](https://huggingface.co/docs/trl) — supervised fine-tuning trainer
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) — 4-bit quantization
- [Google Colab](https://colab.research.google.com) — GPU training environment

---
