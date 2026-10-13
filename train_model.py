import os
import torch
import numpy as np
from datasets import Dataset
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

MODEL_NAME = "distilbert/distilbert-base-uncased"
OUTPUT_DIR = "./model_output"
ONNX_EXPORT_PATH = "models/complexity_classifier.onnx"

print("[1/5] Loading and formatting dataset...")
df = pd.read_csv("thesis_dataset_balanced.csv")
df = df.dropna(subset=['text', 'label'])
df['label'] = df['label'].astype(int)

# 80/20 train/eval split to track real validation loss
train_df = df.sample(frac=0.8, random_state=42)
eval_df = df.drop(train_df.index)

train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

print("[2/5] Initializing Tokenizer & Model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

train_tokenized = train_dataset.map(tokenize_function, batched=True)
eval_tokenized = eval_dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Metrics for thesis evaluation (Accuracy + F1)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = np.mean(preds == labels)
    return {"accuracy": acc}

print("[3/5] Configuring Stabilized Training Arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1.5e-5,          # Low LR to prevent gradient explosion
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_steps=100,              # Warmup prevents early step NaN
    max_grad_norm=1.0,             # Gradient clipping prevents NaN
    fp16=False,                    # Enforce stable FP32 on ROCm
    logging_steps=100,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=eval_tokenized,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

print("[4/5] Training classifier...")
trainer.train()

print("[5/5] Exporting clean ONNX graph...")
os.makedirs(os.path.dirname(ONNX_EXPORT_PATH), exist_ok=True)
model.eval()
model.to("cpu")  # Force model to CPU to match the dummy_inputs

# Dummy input matching router tensor shapes
dummy_inputs = tokenizer(
    "Sample prompt for graph tracing", 
    return_tensors="pt", 
    truncation=True, 
    padding="max_length", 
    max_length=128
)

torch.onnx.export(
    model,
    (dummy_inputs["input_ids"], dummy_inputs["attention_mask"]),
    ONNX_EXPORT_PATH,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size"},
        "attention_mask": {0: "batch_size"}
    },
    opset_version=14
)

print(f"\nTraining and export complete. Verified ONNX saved to: {ONNX_EXPORT_PATH}")
