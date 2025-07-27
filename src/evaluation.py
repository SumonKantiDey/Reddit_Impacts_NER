import logging
import src.settings as settings

from typing import List, NamedTuple, Dict
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import evaluate
import numpy as np

logger = logging.getLogger(__name__)

# Define the Entity NamedTuple
class Entity(NamedTuple):
    e_type: str
    start_offset: int
    end_offset: int

# Convert BIO tags to entities
def bio_to_entities(bio_tags: List[str]) -> List[Entity]:
    """
    Convert a single BIO-tagged sequence to a list of Entity objects.
    Handles fragmented entities correctly.
    """
    entities = []
    start = None
    entity_type = None

    for i, tag in enumerate(bio_tags):
        if tag.startswith("B-"):
            if entity_type is not None:  # Close previous entity
                entities.append(Entity(e_type=entity_type, start_offset=start, end_offset=i - 1))
            entity_type = tag[2:]  # Extract type after "B-"
            start = i
        elif tag.startswith("I-") and entity_type == tag[2:]:
            # Continuation of the same entity
            continue
        elif tag.startswith("I-") and entity_type != tag[2:]:
            # Fragmented entity, treat as a new one
            if entity_type is not None:
                entities.append(Entity(e_type=entity_type, start_offset=start, end_offset=i - 1))
            entity_type = tag[2:]
            start = i
        elif tag == "O":
            if entity_type is not None:  # Close current entity
                entities.append(Entity(e_type=entity_type, start_offset=start, end_offset=i - 1))
                entity_type = None
                start = None

    if entity_type is not None:  # Handle last entity
        entities.append(Entity(e_type=entity_type, start_offset=start, end_offset=len(bio_tags) - 1))

    return entities

# Calculate relaxed overlap
def relaxed_overlap(entity1: Entity, entity2: Entity) -> float:
    """
    Calculate token overlap between two entities.
    Returns the absolute number of overlapping tokens.
    """
    if entity1.e_type != entity2.e_type:
        return 0  # Different types, no overlap

    return max(0, min(entity1.end_offset, entity2.end_offset) - max(entity1.start_offset, entity2.start_offset) + 1)


# Calculate F1 score per entity using absolute overlaps
def calculate_f1_per_entity_covering_all(gold_labels: List[List[str]], pred_labels: List[List[str]]) -> dict:
    """
    Calculate precision, recall, and F1 score for each entity type using absolute token overlap.
    Ensures all predicted entities contribute to evaluation.
    """
    aggregated_results = defaultdict(lambda: {"TP_overlap": 0, "Total_True_Length": 0, "Total_Pred_Length": 0})

    for gold, pred in zip(gold_labels, pred_labels):
        # Convert BIO tags to entities
        true_entities = bio_to_entities(gold)
        pred_entities = bio_to_entities(pred)

        # Process each true entity
        matched_pred_indices = set()
        for true_entity in true_entities:
            # print(f"{true_entity=}")
            for i, pred_entity in enumerate(pred_entities):
                # print(f"{pred_entity=}")
                if i in matched_pred_indices:
                    continue  # Skip already matched predictions
                overlap = relaxed_overlap(true_entity, pred_entity)
                if overlap > 0:
                    aggregated_results[true_entity.e_type]["TP_overlap"] += overlap
                    matched_pred_indices.add(i)
            aggregated_results[true_entity.e_type]["Total_True_Length"] += (true_entity.end_offset - true_entity.start_offset + 1)

        # Count lengths of all predicted entities
        for pred_entity in pred_entities:
            aggregated_results[pred_entity.e_type]["Total_Pred_Length"] += (pred_entity.end_offset - pred_entity.start_offset + 1)

    # Calculate precision, recall, and F1 for each entity type
    final_results = {}
    overall_tp_overlap = 0
    overall_true_length = 0
    overall_pred_length = 0
    
    for entity_type, values in aggregated_results.items():
        precision = values["TP_overlap"] / values["Total_Pred_Length"] if values["Total_Pred_Length"] > 0 else 0.0
        recall = values["TP_overlap"] / values["Total_True_Length"] if values["Total_True_Length"] > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        overall_tp_overlap += values["TP_overlap"]
        overall_true_length += values["Total_True_Length"]
        overall_pred_length += values["Total_Pred_Length"]

        final_results[entity_type] = {
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "F1-Score": round(f1, 3),
            "Coverage": f"{values['TP_overlap']}/{values['Total_True_Length']}"
        }
    # Calculate overall precision, recall, and F1 - Micro
    overall_precision = overall_tp_overlap / overall_pred_length if overall_pred_length > 0 else 0.0
    overall_recall = overall_tp_overlap / overall_true_length if overall_true_length > 0 else 0.0
    overall_f1 = (2 * overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0

    final_results["Overall"] = {
        "Precision": round(overall_precision, 3),
        "Recall": round(overall_recall, 3),
        "F1-Score": round(overall_f1, 3),
        "Coverage": f"{overall_tp_overlap}/{overall_true_length}"
    }
    return final_results


metric = evaluate.load("seqeval")
def compute_metrics(p):
    predictions, labels = p

    # print(predictions.shape)
    # print(labels.shape)

    predictions = np.argmax(predictions, axis=2) # (batch_size, sequence_length, num_labels) -> (batch_size, sequence_length)

    true_predictions = [
        [settings.label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [settings.label_names[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # print(true_predictions)
    # print(true_labels)

    # all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
  
    # # Flatten the lists of lists
    # flat_true_predictions = [item for sublist in true_predictions for item in sublist]
    # flat_true_labels = [item for sublist in true_labels for item in sublist]

    # # Calculate precision, recall, and F1 using sklearn
    # precision = precision_score(flat_true_labels, flat_true_predictions, average='macro')
    # recall = recall_score(flat_true_labels, flat_true_predictions, average='macro')
    # f1 = f1_score(flat_true_labels, flat_true_predictions, average='macro')
    # accuracy = accuracy_score(flat_true_labels, flat_true_predictions)
    
    # # Print the detailed results
    # print("Flattened Metrics:")
    # print(f"Precision: {precision:.4f}")
    # print(f"Recall: {recall:.4f}")
    # print(f"F1 Score: {f1:.4f}")
    # print(f"Accuracy: {accuracy:.4f}")

    # return {
    #     "precision": precision,
    #     "recall": recall,
    #     "f1": f1,
    #     "accuracy": accuracy
    # }
    final_results = calculate_f1_per_entity_covering_all(true_labels, true_predictions)
    return final_results



def compute_metrics_ner(p):
    label_names = ['O', 'B-ClinicalImpacts', 'I-ClinicalImpacts', 'B-SocialImpacts', 'I-SocialImpacts']
    label2id = {label: i for i, label in enumerate(label_names)}
    id2label = {i: label for label, i in label2id.items()}
    predictions, labels = p
    print("------------------------------")
    print(type(p))
    print("p = ", predictions)
    print("label = ", labels)
    # true_predictions = [
    #     [settings.label_names[p] for p in prediction]  # No filtering needed
    #     for prediction in predictions
    # ]
    # true_labels = [
    #     [settings.label_names[l] for l in label]  # No -100 masking needed
    #     for label in labels
    # ]

     # labels and predictions are already aligned and contain only valid IDs
    true_predictions = [
        [id2label[p] for p in prediction]
        for prediction in predictions
    ]
    true_labels = [
        [id2label[l] for l in label]
        for label in labels
    ]
    flat_preds = [item for sublist in true_predictions for item in sublist]
    flat_labels = [item for sublist in true_labels for item in sublist]

    # 🛑 Add sanity check
    if len(flat_preds) != len(flat_labels):
        print("⚠️ Length mismatch:")
        print("flat_preds:", len(flat_preds))
        print("flat_labels:", len(flat_labels))
        print("Sample prediction:", true_predictions[0] if true_predictions else "None")
        print("Sample label:", true_labels[0] if true_labels else "None")
        return {"precision": 0, "recall": 0, "f1": 0, "accuracy": 0}

    precision = precision_score(flat_labels, flat_preds, average="macro", zero_division=0)
    recall = recall_score(flat_labels, flat_preds, average="macro", zero_division=0)
    f1 = f1_score(flat_labels, flat_preds, average="macro", zero_division=0)
    accuracy = accuracy_score(flat_labels, flat_preds)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }



# Example input
# gold_labels = [
#     ["O", "B-Social_Impacts", "I-Social_Impacts", "I-Social_Impacts", "O", "O", "B-Clinical_Impacts", "I-Clinical_Impacts"],
#     ["O", "O", "O", "O", "O", "O"]
# ]

# pred_labels = [
#     ["O", "I-Social_Impacts", "O", "I-Social_Impacts", "O", "O", "B-Clinical_Impacts", "I-Clinical_Impacts"],
#     ["I-Social_Impacts", "I-Social_Impacts", "O", "O", "O", "O"]
# ]

# Compute F1 score using absolute overlaps
# results_per_entity = calculate_f1_per_entity_covering_all(gold_labels, pred_labels)

# Output results
# print("F1 Score Results Per Entity:")
# for entity, metrics in results_per_entity.items():
#     print(f"Entity Type: {entity}")
#     for metric, value in metrics.items():
#         print(f"  {metric}: {value}")
#     print()