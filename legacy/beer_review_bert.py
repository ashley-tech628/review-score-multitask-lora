#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Solution A: DistilBERT + LoRA Multi-Task Regression
- Input: review/text
- Output: 5 normalized scores (appearance/aroma/palate/taste/overall ∈ [0,1])
- Training: Trained on train_data.json, with a portion internally reserved for validation
- Testing: Evaluated on test_data.json, printing MSE and average MSE for each score

Execution:
    python schemeA_train_beer.py
"""

import os
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import mean_squared_error
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
)

from peft import LoraConfig, get_peft_model


# -----------------------------
# Basic Settings
# -----------------------------

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128

TRAIN_FILE = "train_data.json"
TEST_FILE = "test_data.json"

TARGET_KEYS = [
    "review/appearance",
    "review/aroma",
    "review/palate",
    "review/taste",
    "review/overall",
]


# -----------------------------
# Utility Functions
# -----------------------------

def parse_fraction(s: Any) -> float:
    """
    Parse ‘3/5’ into 0.6 (float).
    If the format is invalid, return 0.0.
    Also compatible with cases where the input is already a number.
    """
    if s is None:
        return 0.0
    s = str(s)
    if "/" not in s:
        try:
            return float(s)
        except Exception:
            return 0.0
    num, den = s.split("/")
    try:
        num = float(num.strip())
        den = float(den.strip())
        if den == 0:
            return 0.0
        return num / den
    except Exception:
        return 0.0


# -----------------------------
# Model Definition: DistilBERT + MLP Regression Head
# -----------------------------

class MultiTaskDistilBERTRegressor(nn.Module):
    """
      Use DistilBERT to extract text features, then add a regression head on top to output five scores.
    """

    def __init__(self, base_model_name: str, num_labels: int = 5, dropout: float = 0.1):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.transformer.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_labels),
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ):
        # DistilBERT: last_hidden_state shape = (B, T, H)
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_emb = outputs.last_hidden_state[:, 0, :]  # Take the first token as the sentence vector
        x = self.dropout(cls_emb)
        preds = self.regressor(x)  # (B, 5)

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(preds, labels)

        return {
            "loss": loss,
            "logits": preds,
        }


# -----------------------------
# Data Preprocessing
# -----------------------------

def get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def preprocess_function(examples: Dict[str, Any], tokenizer: AutoTokenizer) -> Dict[str, Any]:
    # 1) First, safely extract the text and convert it entirely into a string.
    raw_texts = examples.get("review/text", [])
    texts = []
    for t in raw_texts:
        if t is None:
            texts.append("")    # If there's truly no content, return an empty string.
        elif isinstance(t, (list, dict)):
            # If any dirty data happens to be a list or dictionary, convert it directly to a string to prevent tokenizer errors.
            texts.append(str(t))
        else:
            texts.append(str(t))

     # 2) Parse 5 scores and normalize them to 0-1 (compatible with missing values / None in the test set)
    all_labels = []
    n = len(texts)
    for i in range(n):
        row_labels = []
        for key in TARGET_KEYS:
            col = examples.get(key, None)
            val = None
            if col is not None and i < len(col):
                val = col[i]
            row_labels.append(parse_fraction(val))  # parse_fraction 已经对 None 做兜底
        all_labels.append(row_labels)

    # 3) Call the tokenizer (at this point, `texts` is guaranteed to be a List[str])
    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    tokenized["labels"] = all_labels
    return tokenized



# -----------------------------
# Evaluation Metrics
# -----------------------------

def compute_metrics(eval_pred):
    preds, labels = eval_pred  # numpy arrays
    # preds / labels: (N, 5)
    mse_list = []
    for i in range(5):
        mse_i = mean_squared_error(labels[:, i], preds[:, i])
        mse_list.append(mse_i)
    avg_mse = float(np.mean(mse_list))
    return {
        "mse_appearance": mse_list[0],
        "mse_aroma": mse_list[1],
        "mse_palate": mse_list[2],
        "mse_taste": mse_list[3],
        "mse_overall": mse_list[4],
        "mse_avg": avg_mse,
    }


# -----------------------------
# Main Process
# -----------------------------

def main():
    print("✅ Loading tokenizer and raw datasets...")

    tokenizer = get_tokenizer()

    # Read train_data.json and test_data.json
    # Assume each file is a list[dict], with the same structure as previously provided
    raw_datasets = load_dataset(
        "json",
        data_files={
            "train": TRAIN_FILE,
            "test": TEST_FILE,
        },
    )

    raw_train = raw_datasets["train"]
    raw_test = raw_datasets["test"]

    print("Raw train size:", len(raw_train))
    print("Raw test  size:", len(raw_test))

    # Preprocess the train data before splitting it into train/valid sets
    def train_preprocess(batch):
        return preprocess_function(batch, tokenizer)

    print("✅ Preprocessing train dataset...")
    processed_train = raw_train.map(
        train_preprocess,
        batched=True,
        remove_columns=raw_train.column_names,
    )

    # Split into train / valid sets
    train_valid = processed_train.train_test_split(
        test_size=0.1,
        seed=42,
    )
    train_ds = train_valid["train"]
    valid_ds = train_valid["test"]

    # Preprocessing the test set
    print("✅ Preprocessing test dataset...")
    def test_preprocess(batch):
        return preprocess_function(batch, tokenizer)

    processed_test = raw_test.map(
        test_preprocess,
        batched=True,
        remove_columns=raw_test.column_names,
    )

    # Set as PyTorch tensor output
    train_ds.set_format(type="torch")
    valid_ds.set_format(type="torch")
    processed_test.set_format(type="torch")

    print("Train size:", len(train_ds))
    print("Valid size:", len(valid_ds))
    print("Test  size:", len(processed_test))

    # -------------------------
    # Model Building + LoRA
    # -------------------------
    print("✅ Building model with LoRA...")

    base_model = MultiTaskDistilBERTRegressor(MODEL_NAME)

    # LoRA Configuration: Apply LoRA only to q_lin / v_lin in the attention layer
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_lin", "v_lin"],
    )

    model = get_peft_model(base_model, lora_config)
    print(model)

    # Print whether MPS is currently enabled
    print("MPS available:", torch.backends.mps.is_available())

    # -------------------------
    # Training Parameters
    # -------------------------
    print("✅ Setting up Trainer and TrainingArguments...")

    batch_size = 32
    epochs = 3
    lr = 2e-5

    training_args = TrainingArguments(
        output_dir="./beer_multitask_distilbert_mps",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,

        logging_steps=100,   # Log every X steps
        save_steps=1000,     # Save a checkpoint every X steps
        save_total_limit=2,  # Maximum number of checkpoints to retain

        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # -------------------------
    # Training
    # -------------------------
    print("🚀 Start training...")
    trainer.train()

    print("✅ Training finished.")
    print("📊 Validation performance on the best checkpoint:")
    val_metrics = trainer.evaluate()
    print(val_metrics)

    # -------------------------
    # Evaluate on test_data.json
    # -------------------------
    print("📊 Evaluating on test_data.json ...")
    test_metrics = trainer.evaluate(eval_dataset=processed_test)
    print("✅ Test metrics:")
    for k, v in test_metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()