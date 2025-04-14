import os
import torch
import logging
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForTokenClassification
import src.settings as settings
from .dataloader import reddit_impacts_dataset
from .evaluation import calculate_f1_per_entity_covering_all
from .utils import get_parser

os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # Use GPU 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

path = "/labs/sarkerlab/sdey26/Reddit_Impacts_NER/models_and_checkpoints/"
# Parse arguments
parser = get_parser()
args = parser.parse_args()
logger = logging.getLogger(__name__)

logger.info(f"Test Data Evaluation: Model {args.model_name}")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(f'{path}{args.model_name}.model')
model = AutoModelForTokenClassification.from_pretrained(
    f'{path}{args.model_name}.model',
    id2label=settings.id2label,
    label2id=settings.label2id,
)
model = model.to(device)
train_dataset, dev_dataset, test_dataset = reddit_impacts_dataset()

# test_dataset = test_dataset[9:11]
# Predict labels for each token in paragraph
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

    # print("-----------------------------------------------------------------")
    # print("Input IDs:", tokenizer.decode(inputs["input_ids"][0]))
    # print("input = ",input)
    # print("post_tokens = ", post_tokens)
    # print(tokenizer.decode(inputs["input_ids"][0]))

    # Perform token classification
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()  # Full prediction including [CLS], [SEP]
    # print("predictions = ",predictions)

    # Map predictions back to word-level
    word_ids = inputs.word_ids(batch_index=0)  # Get alignment between word and token
    # print("word_ids = ",word_ids)

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
    # print("Predicted labels:", post_prediction_label)
    # print("True labels:     ", test_dataset['ner_tags_str'][i])


# print(prediction_label)
# Calculate relaxed F1 score per entity
results_per_entity = calculate_f1_per_entity_covering_all(test_dataset['ner_tags_str'], prediction_label)

# Print results
logger.info(f"Relaxed F1 Score Results Per Entity for model {args.model_name}")
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
test_df.to_excel(f"/labs/sarkerlab/sdey26/Reddit_Impacts_NER/test_pred_files/{args.model_name}_pred.xlsx", index=False)