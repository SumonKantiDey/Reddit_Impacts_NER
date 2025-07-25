import logging
import os
import argparse
from transformers import (
    AutoTokenizer,
    RobertaTokenizerFast,
    AutoConfig, 
    AutoModelForTokenClassification,
    DebertaV2TokenizerFast, 
    RobertaForTokenClassification,
    TrainerCallback
)
from typing import Optional, Union, List
import random
import numpy as np

logger = logging.getLogger(__name__)

# utils.py
import os
import torch
from typing import Union, List, Optional

def setup_device(
    gpus: Optional[Union[int, List[int]]] = 0,
    verbose: bool = True
) -> torch.device:
    """
    Set CUDA device(s), clear cache, and return primary torch.device.

    Args:
      gpus: int or list of ints (GPU IDs) — e.g., 0 or [0,1,2]
      verbose: whether to print device setup information

    Returns:
      torch.device: 'cuda' if available, else 'cpu'
    """
    # 1) Normalize input
    if isinstance(gpus, int):
        gpus = [gpus]  # make it a list
    if not gpus:
        gpus = []  # empty list means all GPUs

    # 2) Set environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()

    if gpus:
        gpus_str = ",".join(map(str, gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus_str
    # if gpus is empty, don't set CUDA_VISIBLE_DEVICES (all GPUs available)

    # 3) Pick device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all GPUs")
        print(f"CUDA_VISIBLE_DEVICES = {visible}")
        print(f"Available GPUs = {torch.cuda.device_count()}")
        if device.type == "cuda":
            print(f"Using device: {device} → {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU")

    return device


def load_tokenizer(model_name: str, use_crf: bool = False):
    if "roberta" in model_name.lower() and use_crf == True:
        tokenizer = RobertaTokenizerFast.from_pretrained(model_name, add_prefix_space=True)
    elif "deberta" in model_name.lower() and use_crf == True:
        tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


# Define a custom callback to log metrics to console
class CustomLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # Log training loss and eval loss if available
            if 'loss' in logs and 'eval_loss' in logs:
                logger.info(f"Step {state.global_step} - Train loss: {logs['loss']:.4f} - Eval loss: {logs['eval_loss']:.4f}")
            elif 'loss' in logs:
                logger.info(f"Step {state.global_step} - Train loss: {logs['loss']:.4f}")
            elif 'eval_loss' in logs:
                logger.info(f"Step {state.global_step} - Eval loss: {logs['eval_loss']:.4f}")


def seed_everything(seed: int):
    print("everything seeded")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_parser():
    # Argument parser
    parser = argparse.ArgumentParser(description="Fine-tune a NER model with LoRA")

    # Model and data parameters
    parser.add_argument("--model_name", type=str, default="TinyLlama", help="Name for saving the fine-tuned model")

    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for training")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate (0.0 for no dropout)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--logging_steps", type=int, default=10, help="Steps for logging metrics")
    parser.add_argument("--eval_steps", type=int, default=50, help="Steps for evaluation during training")
    parser.add_argument("--save_steps", type=int, default=50, help="Steps for saving the model")

    # New arguments
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", choices=["linear", "cosine", "polynomial", "constant", "constant_with_warmup"], help="Learning rate scheduler type.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for gradient clipping.")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Number of evaluations with no improvement to wait before early stopping.")
    parser.add_argument("--seed", type=int, default=42)

    # LoRA-specific parameters
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout rate")
    parser.add_argument("--lora_target_linear", type=bool, default=False, help="Apply LoRA to linear layers.")

    # GPUs parameters
    parser.add_argument(
        "--gpus",
        type=str,
        help="Comma-separated GPU IDs (e.g., 0,1,2)"
    )
    parser.add_argument("--use_crf", type=bool, default=False, help="Apply CRF layer")
    return parser