import logging
from datasets import load_dataset
from transformers import RobertaTokenizerFast
from .dataloader import reddit_impacts_dataset, NERDataset
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
import torch
from transformers import RobertaForTokenClassification, DataCollatorForTokenClassification, RobertaConfig, AutoConfig
from .evaluation import calculate_f1_per_entity_covering_all, compute_metrics
from .utils import setup_device, CustomLoggingCallback, get_parser, load_tokenizer, seed_everything
from torchcrf import CRF
import os
import pandas as pd
import src.settings as settings
from .models import RobertaCRFForNER, BertCRFForNER, BERTForTokenClassificationWithCRF, EncoderForTokenClassificationWithBiLSTMCRF


# Parse arguments
parser = get_parser()
args = parser.parse_args()
logger = logging.getLogger(__name__)
print(args)

logger.info("...........Evaluate Test Dataset with Bilstm CRF...........")

# Load the model configuration
save_model_name = settings.models_config[args.model_name]
model_storage_path = f"models_and_checkpoints/{save_model_name}-crf_{args.seed}"

# Parse string like "1,2" into list of ints [1, 2]
if args.gpus:
    gpu_list = [int(x) for x in args.gpus.split(",")]
else:
    gpu_list = None

device = setup_device(gpus=gpu_list) 

# Used for LLM+CRF
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


def evaluate_on_test_set(model, tokenizer, test_dataset):
    test_tokenized = test_dataset.map(tokenize_and_align_labels, batched=True)
    print(test_tokenized)

    # Create data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # Create test trainer
    test_trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./temp",
            per_device_eval_batch_size=8,
            fp16=True
        ),
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    # Run evaluation
    results = test_trainer.evaluate(test_tokenized)
    return results


# Initialize model architecture
if save_model_name == "roberta-large":
    print("roberta model loaded")
    tokenizer = load_tokenizer(args.model_name, args.use_crf)
    model = RobertaCRFForNER(num_labels=len(settings.label_names), id2label=settings.id2label, label2id=settings.label2id, device=device)
    model.load_state_dict(torch.load(f"{model_storage_path}/final_model_{args.seed}.pt"))
    model.eval()

# elif save_model_name == "bert-large-uncased":
#     tokenizer = load_tokenizer(args.model_name, args.use_crf)
#     model = BertCRFForNER(num_labels=len(settings.label_names), id2label=settings.id2label, label2id=settings.label2id, device=device)
#     model.load_state_dict(torch.load(f"{model_storage_path}/final_model_{args.seed}.pt"))
#     model.eval()
else:
    if save_model_name == "debarta-large":
        tokenizer = load_tokenizer(args.model_name, args.use_crf)
    else:
        tokenizer = load_tokenizer(args.model_name, args.use_crf)

    # Load config with label info
    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=len(settings.label2id),
        id2label=settings.id2label,
        label2id=settings.label2id
    )
    # model = BERTForTokenClassificationWithCRF(config, device=device)
    model = EncoderForTokenClassificationWithBiLSTMCRF(config, device=device)

    model.load_state_dict(torch.load(f"{model_storage_path}/final_model_{args.seed}.pt"))
    model.eval()

    
model.to(device)

# load dataset 
train_dataset, dev_dataset, test_dataset = reddit_impacts_dataset(args.seed) 

logger.info("...........test_data_evaluation...........")
test_results = evaluate_on_test_set(model, tokenizer, test_dataset)
logger.info(test_results)


# def predict_custom_ner(test_dataset):
#     output_lines = []
#     prediction_label = []
#     for i, post_tokens in enumerate(test_dataset['tokens']):
#         # print(post_tokens)
#         # Tokenize the post with word alignment
#         inputs = tokenizer(
#             post_tokens,
#             is_split_into_words=True,
#             return_tensors="pt",
#             truncation=True,
#             padding=True
#         ).to(device)

#         # Get predictions
#         with torch.no_grad():
#             # 1) run Roberta to get emissions
#             if "roberta" in save_model_name:
#                 outputs = model.roberta(
#                     input_ids=inputs["input_ids"],
#                     attention_mask=inputs["attention_mask"]
#                 )
#                 emissions = outputs.logits # (B, T, num_labels)
#             # elif "bert" in save_model_name:
#             #     outputs = model.bert(
#             #         input_ids=inputs["input_ids"],
#             #         attention_mask=inputs["attention_mask"]
#             #     )
#             #     emissions = outputs.logits # (B, T, num_labels)
#             else:
#                 # Step 1: Get BERT embeddings
#                 outputs = model.bert(
#                     input_ids=inputs["input_ids"],
#                     attention_mask=inputs["attention_mask"],
#                     token_type_ids=inputs.get("token_type_ids", None)
#                 )
#                 # Step 2: Apply dropout and classification layer
#                 sequence_output = model.dropout(outputs[0])
#                 emissions = model.classifier(sequence_output)  # shape: (B, T, num_labels)
    
#             mask = inputs["attention_mask"].bool()  # (B, T)
#             # print(emissions.shape)
#             # print(mask.shape)
#             # 2) decode with CRF directly
#             decoded_tagseqs = model.crf.decode(emissions, mask=mask)

#         predictions = decoded_tagseqs[0]
#         # Map predictions back to word-level
#         word_ids = inputs.word_ids(batch_index=0)  # Get alignment between word and token

#         post_prediction_label = []
#         previous_word_idx = None
#         for idx, word_idx in enumerate(word_ids):
#             if word_idx is None:
#                 continue  # Skip [CLS], [SEP], or padding
#             if word_idx != previous_word_idx: # If a word is split into multiple subword tokens (e.g., "hospitalization" → ["hospital", "##ization"]), they will all have the same word_idx.
#                 pred_label_id = predictions[idx]
#                 pred_label = settings.label_names[pred_label_id] if pred_label_id != 0 else "O"
#                 post_prediction_label.append(pred_label)
#             previous_word_idx = word_idx

#         prediction_label.append(post_prediction_label)

#     results_per_entity = calculate_f1_per_entity_covering_all(test_dataset['ner_tags_str'], prediction_label)
#     # Print results
#     logger.info(f"Relaxed F1 Score for each Entity for {save_model_name} + CRF for seed {args.seed}")
#     for entity, metrics in results_per_entity.items():
#         result_str = f"Entity Type: {entity}\n"
#         for metric, value in metrics.items():
#             # print(f"  {metric}: {value}")
#             result_str += f"  {metric}: {value}\n"
#         logger.info(f"\n {result_str}")

#     logger.info("Save the prediction result as dataframe")
#     test_df = pd.DataFrame(test_dataset)
#     test_df['prediction'] = prediction_label
#     test_df = test_df[["tokens","labels","ner_tags_str","prediction"]]

#     file_loc = "/labs/sarkerlab/sdey26/Reddit_Impacts_NER/test_pred_files"
#     test_df.to_excel(f"{file_loc}/{save_model_name}_{args.seed}_crf_pred.xlsx", index=False)

def predict_custom_ner(test_dataset):
    output_lines = []
    prediction_label = []
    for i, post_tokens in enumerate(test_dataset['tokens']):
        # Tokenize the post with word alignment
        inputs = tokenizer(
            post_tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(device)

        # Get predictions
        with torch.no_grad():
            # 1) run Roberta to get emissions
            if "roberta" in save_model_name:
                outputs = model.roberta(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                )
                emissions = outputs.logits # (B, T, num_labels)
            
            # elif "bert" in save_model_name:
            #     outputs = model.bert(
            #         input_ids=inputs["input_ids"],
            #         attention_mask=inputs["attention_mask"]
            #     )
            #     emissions = outputs.logits # (B, T, num_labels)

            else:
                # Step 1: Get encoder embeddings
                outputs = model.encoder(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    token_type_ids=inputs.get("token_type_ids", None)
                )
                # Step 2: Apply dropout and classification layer
                sequence_output = model.dropout(outputs[0])
                lstm_output, _ = model.bilstm(sequence_output)
                emissions = model.classifier(lstm_output)
    
            mask = inputs["attention_mask"].bool()  # (B, T)
            # print(emissions.shape)
            # print(mask.shape)
            
            # 2) decode with CRF directly
            decoded_tagseqs = model.crf.decode(emissions, mask=mask)

        predictions = decoded_tagseqs[0]
        # Map predictions back to word-level
        word_ids = inputs.word_ids(batch_index=0)  # Get alignment between word and token

        post_prediction_label = []
        previous_word_idx = None
        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue  # Skip [CLS], [SEP], or padding
            if word_idx != previous_word_idx: # If a word is split into multiple subword tokens (e.g., "hospitalization" → ["hospital", "##ization"]), they will all have the same word_idx.
                pred_label_id = predictions[idx]
                pred_label = settings.label_names[pred_label_id] if pred_label_id != 0 else "O"
                post_prediction_label.append(pred_label)
            previous_word_idx = word_idx

        prediction_label.append(post_prediction_label)

    results_per_entity = calculate_f1_per_entity_covering_all(test_dataset['ner_tags_str'], prediction_label)
    # Print results
    logger.info(f"Relaxed F1 Score for each Entity for {save_model_name} + CRF for seed {args.seed}")
    for entity, metrics in results_per_entity.items():
        result_str = f"Entity Type: {entity}\n"
        for metric, value in metrics.items():
            # print(f"  {metric}: {value}")
            result_str += f"  {metric}: {value}\n"
        logger.info(f"\n {result_str}")

    logger.info("Save the prediction result as dataframe")
    test_df = pd.DataFrame(test_dataset)
    test_df['prediction'] = prediction_label
    test_df = test_df[["tokens","labels","ner_tags_str","prediction"]]

    file_loc = "/labs/sarkerlab/sdey26/Reddit_Impacts_NER/test_pred_files"
    test_df.to_excel(f"{file_loc}/{save_model_name}_{args.seed}_bilstm_crf_pred.xlsx", index=False)


predict_custom_ner(test_dataset)