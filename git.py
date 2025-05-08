import json
import torch
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, GitForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
import os

# Disable tokenizer parallelism to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load GIT processor
processor = AutoProcessor.from_pretrained("microsoft/git-large", use_fast=True)

# Paths
img_dir = "./images/"
csv_path = "./images/metadata.csv" 

# Load CSV data
df = pd.read_csv(csv_path)

# Process dataset into training format
processed_data = []

# Iterate over rows in the CSV
for index, row in df.iterrows():
    image_path = img_dir + row["file_name"] 
    text = row["text"] 

    try:
        # Open the image
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Image not found: {image_path}")
        continue

    # Tokenize inputs (both image and text)
    inputs = processor(
        images=image,
        text=text,  # Include the text input
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        legacy=False,  # Avoid deprecation warnings
    )

    # Add labels for training
    inputs["labels"] = inputs["input_ids"]

    # Append processed data
    processed_data.append({k: v.squeeze(0) for k, v in inputs.items()})

# Convert list to Dataset
dataset = Dataset.from_list(processed_data)

# Split dataset into train and validation sets
dataset = dataset.train_test_split(test_size=0.15)
train_dataset = dataset["train"]
val_dataset = dataset["test"]

# Load pre-trained GIT model
model = GitForCausalLM.from_pretrained("microsoft/git-large")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./git-finetuned",
    per_device_train_batch_size=8, 
    per_device_eval_batch_size=8,
    num_train_epochs=64,
    save_steps=256,
    save_total_limit=2,
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    weight_decay=0.01,
    fp16=True,  
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",  # Disable external logging (e.g., Weights & Biases)
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Start fine-tuning
trainer.train()
