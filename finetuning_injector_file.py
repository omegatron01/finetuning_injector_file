import torch
import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from google.colab import userdata

HF_TOKEN = userdata.get('Mytoken')
os.environ['HF_TOKEN'] = HF_TOKEN

df = pd.read_csv(input dir of training file)
df = df.dropna()

print(f"Training on {len(df)} examples")

def format_row(row):
    return {
        "text": f"Translate English to Igbo.\nEnglish: {row['en']}\nIgbo: {row['ig']}"
    }

dataset = Dataset.from_list([format_row(row) for _, row in df.iterrows()])
print("Dataset ready!")
print(dataset[0])

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = "google/gemma-4-E4B-it"
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
print("Model loaded!")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
    exclude_modules=["vision_tower", "multi_modal_projector", "audio_tower"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

sft_config = SFTConfig(
    output_dir="./igbo-gemma4-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=50,
    save_strategy="epoch",
    bf16=True,
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=sft_config,
)

print("Starting training...")
trainer.train()
print("Training complete!")

model.save_pretrained("./igbo-gemma4-finetuned")
tokenizer.save_pretrained("./igbo-gemma4-finetuned")
print("Model saved to igbo-gemma4-finetuned folder!")

