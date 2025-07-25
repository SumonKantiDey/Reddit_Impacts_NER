import numpy as np
import logging
from sklearn.metrics import f1_score
from typing import List, Dict, Tuple
import pandas as pd
import ast
from .evaluation import calculate_f1_per_entity_covering_all 
from .utils import get_parser
import src.settings as settings


parser = get_parser()
args = parser.parse_args()
logger = logging.getLogger(__name__)

def bootstrap_f1_ci_all(
    preds: List[List[str]],
    labels: List[List[str]],
    num_samples: int = 1000,
    seed: int = 3
) -> Dict[str, Dict[str, float]]:
    """
    Compute bootstrap-based 95% CI for F1 scores of:
    - ClinicalImpacts
    - SocialImpacts
    - Overall (macro average of the above two)

    Uses stratified sampling to handle imbalanced entity distributions.

    Returns:
        {
            'ClinicalImpacts': {'mean': ..., 'lower': ..., 'upper': ...},
            'SocialImpacts':   {'mean': ..., 'lower': ..., 'upper': ...},
            'Overall':         {'mean': ..., 'lower': ..., 'upper': ...}
        }
    """
    np.random.seed(seed)
    assert len(preds) == len(labels), "Predictions and labels must be the same length."
    sample_size = len(preds)

    entity_types = ['ClinicalImpacts', 'SocialImpacts', 'Overall']
    f1_records = {etype: [] for etype in entity_types}

    # Stratify: index rows with at least one Clinical or Social tag
    def contains_target_entity(seq):
        return any(tag.endswith("ClinicalImpacts") or tag.endswith("SocialImpacts") for tag in seq)

    entity_indices = [i for i, seq in enumerate(labels) if contains_target_entity(seq)]

    if len(entity_indices) < 5:
        logger.info("[Warning] Very few sequences contain target entities. Bootstrap CI may be unstable.")

    for _ in range(num_samples):
        clinical_idxs = [i for i, seq in enumerate(labels) if any("ClinicalImpacts" in tag for tag in seq)]
        social_idxs = [i for i, seq in enumerate(labels) if any("SocialImpacts" in tag for tag in seq)]
        clinical_sample = np.random.choice(clinical_idxs, size=len(clinical_idxs), replace=True)
        social_sample = np.random.choice(social_idxs, size=len(social_idxs), replace=True)
        indices = list(clinical_sample) + list(social_sample)

        pred_sample = [preds[i] for i in indices]
        label_sample = [labels[i] for i in indices]

        metrics = calculate_f1_per_entity_covering_all(pred_sample, label_sample)

        for etype in entity_types:
            f1 = metrics.get(etype, {}).get("F1-Score", 0.0)
            f1_records[etype].append(f1)

    # Compute mean and 95% CI
    ci_results = {}
    for etype in entity_types:
        scores = sorted(f1_records[etype])
        ci_results[etype] = {
            "mean": np.mean(scores),
            "lower": np.percentile(scores, 2.5),
            "upper": np.percentile(scores, 97.5)
        }
    return ci_results

if __name__ == "__main__":
    path = "/labs/sarkerlab/sdey26/Reddit_Impacts_NER/"

    if args.use_crf == True:
        save_model_name = settings.models_config[args.model_name]
        file_path = f"{path}test_pred_files/{save_model_name}_{args.seed}_bilstm_crf_pred.xlsx"  # _crf_pred
        print("come here >>>>>>>>>",file_path)
    else: 
        file_path = f"{path}test_pred_files/{args.model_name}_{args.seed}_pred.xlsx"


    # file_path = "./test_pred_files/llms/gpt4_3_shot.xlsx"

    df = pd.read_excel(file_path)
    columns_to_convert = ['ner_tags_str', 'prediction']
    df[columns_to_convert] = df[columns_to_convert].applymap(ast.literal_eval)

    labels = df['ner_tags_str'].tolist()
    preds = df['prediction'].tolist()
    results = bootstrap_f1_ci_all(preds, labels, num_samples=2000, seed=args.seed)
    for name, stats in results.items():
        logger.info(f"{name} F1 = {stats['mean']:.3f}, 95% CI = [{stats['lower']:.3f}, {stats['upper']:.3f}]")