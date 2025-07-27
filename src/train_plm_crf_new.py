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
    RobertaTokenizerFast,
    AutoConfig, 
    AutoModelForTokenClassification, 
    RobertaForTokenClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback)

import src.settings as settings
from .evaluation import calculate_f1_per_entity_covering_all, compute_metrics
from .dataloader import reddit_impacts_dataset, NERDataset, tokenize_and_align_labels_with_dict
from .utils import setup_device, CustomLoggingCallback, get_parser, load_tokenizer, seed_everything

from .models import RobertaCRFForNER, BertCRFForNER, BERTForTokenClassificationWithCRF, EncoderForTokenClassificationWithBiLSTMCRF
import time


# Parse arguments
parser = get_parser()
args = parser.parse_args()
logger = logging.getLogger(__name__)

seed_everything(seed=args.seed)
print(args)

# Load the model configuration
save_model_name = settings.models_config[args.model_name]
model_storage_path = f"models_and_checkpoints/{save_model_name}-crf_{args.seed}"

logger.info(f"{args.model_name=}, {save_model_name=}")

# Parse string like "1,2" into list of ints [1, 2]
if args.gpus:
    gpu_list = [int(x) for x in args.gpus.split(",")]
else:
    gpu_list = None

device = setup_device(gpus=gpu_list)
tokenizer = load_tokenizer(args.model_name, args.use_crf)

def tokenize_and_align_labels(examples):
    # if "deberta" in args.model_name.lower():
    #     examples_tokens = [[word.lower() for word in sentence] for sentence in examples["tokens"]]
    # else:
    #     examples_tokens = examples["tokens"]

    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True, 
        is_split_into_words=True  # Important for word-level labels
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to original words
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special token (CLS, SEP, PAD)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])  # New word
            else:
                label_ids.append(-100)  # Same word (subword token)
            previous_word_idx = word_idx
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# load the dataset 
train_dataset, dev_dataset, test_dataset = reddit_impacts_dataset(args.seed) 

# Apply preprocessing
train_tokenized_datasets = train_dataset.map(tokenize_and_align_labels, batched=True)
dev_tokenized_datasets = dev_dataset.map(tokenize_and_align_labels, batched=True)
test_tokenized_datasets = test_dataset.map(tokenize_and_align_labels, batched=True)

logger.info(f"{settings.label_names} {settings.id2label} {settings.label2id}")

if "roberta" in save_model_name:
    model = RobertaCRFForNER(num_labels=len(settings.label_names), id2label=settings.id2label, label2id=settings.label2id, device=device)
# elif save_model_name == "bert-large-uncased":
#     model = BertCRFForNER(num_labels=len(settings.label_names), id2label=settings.id2label, label2id=settings.label2id, device=device)
else: 
    print("DEBERTA>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> train")
    # Load config with label info
    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=len(settings.label2id),
        id2label=settings.id2label,
        label2id=settings.label2id
    )
    #model = BERTForTokenClassificationWithCRF(config, device=device)
    model = EncoderForTokenClassificationWithBiLSTMCRF(config, device=device)

model = model.to(device)

logger.info(f"Model is on device: {next(model.parameters()).device}")

data_collator = DataCollatorForTokenClassification(tokenizer)


training_args = TrainingArguments(
    output_dir=model_storage_path,  # Directory for saving models
    evaluation_strategy="steps",
    eval_steps=args.eval_steps, 
    learning_rate=args.learning_rate,
    logging_strategy="steps",
    logging_steps=args.logging_steps,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.num_epochs,
    save_strategy="steps",
    save_steps=args.save_steps,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",  # Optimize for validation loss
    report_to=["tensorboard"],
    weight_decay=args.weight_decay,
    fp16=True,
    gradient_accumulation_steps=2,
    max_grad_norm=args.max_grad_norm,
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
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[CustomLoggingCallback(),EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
)

# Train the model
logger.info(f"Starting the model training with {save_model_name}")
train_result = trainer.train()

# Save the entire model (state_dict)
torch.save(model.state_dict(), f"{model_storage_path}/final_model_{args.seed}.pt")

# Also save the tokenizer and config for easy loading later
tokenizer.save_pretrained(model_storage_path)
# model.roberta.config.save_pretrained(model_storage_path)
# torch.save(model.crf.state_dict(), f"{model_storage_path}/crf.bin")
logger.info(f"Model saved to {model_storage_path}")

logging.info("Evaluation based on dev data...")
eval_result = trainer.evaluate()

# Log train and evaluation results
logging.info(f"Training Results: {train_result}")
logging.info(f"Evaluation Results: {eval_result}")
logger.info(
    f"Hyperparameters: model_name = {args.model_name}, learning_rate = {args.learning_rate}, batch_size = {args.batch_size}, num_epochs = {args.num_epochs},weight_decay = {args.weight_decay}, "
    f"seed = {args.seed}, dropout = {args.dropout}, lr_scheduler_type = {args.lr_scheduler_type}, max_grad_norm = {args.max_grad_norm}, early_stopping_patience = {args.early_stopping_patience}"
)

# Cleanup unused memory
gc.collect()
torch.cuda.empty_cache()