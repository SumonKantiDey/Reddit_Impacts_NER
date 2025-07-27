import logging
import torch
import ast, gc
import pdb
import json
import random
import os
import itertools
import pandas as pd 
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback)

from transformers import RobertaTokenizerFast, RobertaForTokenClassification

import src.settings as settings
from .evaluation import calculate_f1_per_entity_covering_all, compute_metrics
from .dataloader import reddit_impacts_dataset, NERDataset, tokenize_and_align_labels_with_dict
from .utils import CustomLoggingCallback, get_parser, setup_device, seed_everything

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# torch.cuda.empty_cache()
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # Use GPU 0
# print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Available GPUs:", torch.cuda.device_count())
# if device.type == "cuda":
#     print(f"Using GPU: {device} {torch.cuda.get_device_name(0)}")

# Parse arguments
parser = get_parser()
args = parser.parse_args()
logger = logging.getLogger(__name__)

seed_everything(seed=args.seed)
print(args)

# Load the model configuration
save_model_name = args.model_name
model_name = settings.plm_models_config[args.model_name]
logger.info(f"{model_name=}, {save_model_name=}")

# Parse string like "1,2" into list of ints [1, 2]
if args.gpus:
    gpu_list = [int(x) for x in args.gpus.split(",")]
else:
    gpu_list = None

device = setup_device(gpus=gpu_list)

if "deberta" in save_model_name:
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    # config=config,
    id2label=settings.id2label,
    label2id=settings.label2id,
    hidden_dropout_prob=args.dropout,
    attention_probs_dropout_prob=args.dropout,
    ignore_mismatched_sizes=True
)

model = model.to(device)

logger.info(f"Model is on device: {next(model.parameters()).device}")

# Load and prepare dataset
ner_dataset = NERDataset(tokenizer_name=model_name)

train_dataset, dev_dataset, test_dataset = reddit_impacts_dataset(args.seed)

# Tokenize and align labels using the Hugging Face dataset.map()
train_tokenized_datasets = train_dataset.map(
    lambda examples: tokenize_and_align_labels_with_dict(examples, ner_dataset),
    batched=True,
    # remove_columns=train_dataset.column_names
)
dev_tokenized_datasets = dev_dataset.map(
    lambda examples: tokenize_and_align_labels_with_dict(examples, ner_dataset),
    batched=True, 
    # remove_columns=test_dataset.column_names
)
# Data collator for token classification
data_collator = DataCollatorForTokenClassification(ner_dataset.tokenizer)

logger.info(f"{train_tokenized_datasets[12]=}")

training_args = TrainingArguments(
    output_dir=f"models_and_checkpoints/{save_model_name}-finetuned-ner",  # Directory for saving models
    evaluation_strategy="steps",       # Must be steps or epoch
    eval_steps=args.eval_steps,               # Evaluate every 100 steps
    learning_rate=args.learning_rate,           # Fine-tuning learning rate
    logging_strategy="steps",     # Log progress during training
    logging_steps=args.logging_steps,             # Log every 50 steps
    per_device_train_batch_size=args.batch_size,  # Adjusted batch size for better performance
    per_device_eval_batch_size=args.batch_size,   # Same as training batch size for evaluation
    num_train_epochs=args.num_epochs,           # Train for 3 complete passes through the dataset
    save_strategy="steps",        # Save model based on steps
    save_steps=args.save_steps,               # Save model every 100 steps
    save_total_limit=2,           # Keep the last 2 checkpoints only
    load_best_model_at_end=True,  # Automatically load the best model at the end
    metric_for_best_model="eval_loss",  # Optimize for validation loss

    # metric_for_best_model="F1-Score",
    # greater_is_better=True,

    report_to=["tensorboard"],
    weight_decay=args.weight_decay,            # Regularization to reduce overfitting
    #logging_dir="custom_logs",    # Custom directory for training logs
    fp16=True,                    # Enable mixed precision for faster training
    gradient_accumulation_steps=2, # Accumulate gradients to simulate larger batch size
    max_grad_norm=args.max_grad_norm,  # Clips gradients to ensure their L2 norm does not exceed the specified value.
    lr_scheduler_type=args.lr_scheduler_type, # Linear scheduler
    # warmup_steps=100, # Gradually increase the learning rate during the first 100 steps
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized_datasets,
    eval_dataset=dev_tokenized_datasets,
    data_collator=data_collator,
    tokenizer=ner_dataset.tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[CustomLoggingCallback(),EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
)

# Train the model
logger.info(f"Starting the model training with {model_name}")
train_result = trainer.train()
trainer.save_model(f"models_and_checkpoints/{save_model_name}_{args.seed}.model")
logger.info(f"Model saved to models_and_checkpoints/{save_model_name}_{args.seed}.model.")

logging.info("Evaluation based on dev data...")
eval_result = trainer.evaluate()

# Log train and evaluation results
logging.info(f"Training Results: {train_result}")
logging.info(f"Evaluation Results: {eval_result}")
logger.info(args)

# Cleanup unused memory
gc.collect()
torch.cuda.empty_cache()