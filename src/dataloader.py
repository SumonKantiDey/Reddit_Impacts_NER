import logging
import pandas as pd 
from transformers import AutoTokenizer
from datasets import Dataset
from sklearn.model_selection import train_test_split
import src.settings as settings
import ast


logger = logging.getLogger(__name__)

# test how performance (e.g., F1 score) improves as you increase the number of labeled training examples.
def label_impact(tags):
    has_clin = any("ClinicalImpacts" in tag for tag in tags)
    has_soc = any("SocialImpacts" in tag for tag in tags)
    
    if has_clin and has_soc:
        return "both"
    elif has_clin:
        return "clinical_only"
    elif has_soc:
        return "social_only"
    else:
        return "none"
    
def get_balanced_sample(df, total_n=10, n_clinical=2, n_social=2, n_both=2, seed=42):

    df["impact_type"] = df["ner_tags"].apply(label_impact)
    clinical_only = df[df["impact_type"] == "clinical_only"]
    social_only = df[df["impact_type"] == "social_only"]
    both_impacts = df[df["impact_type"] == "both"]
    no_impact = df[df["impact_type"] == "none"]

    # Adjust sample sizes based on availability
    n_clinical = min(n_clinical, len(clinical_only))
    n_social = min(n_social, len(social_only))
    n_both = min(n_both, len(both_impacts))

    # Compute remaining samples for 'none' after adjustment
    n_allocated = n_clinical + n_social + n_both
    n_none = max(0, total_n - n_allocated)
    n_none = min(n_none, len(no_impact))  # Cap to available 'none' samples
    
    sampled = pd.concat([
        clinical_only.sample(n=n_clinical, random_state=seed),
        social_only.sample(n=n_social, random_state=seed),
        both_impacts.sample(n=n_both, random_state=seed),
        no_impact.sample(n=n_none, random_state=seed) if n_none > 0 else pd.DataFrame()
    ]).sample(frac=1, random_state=seed)  # Shuffle

    return sampled.reset_index(drop=True)

def compute_allocation(total_n):
    # Adjust as needed; this gives you a rule of thumb
    n_both = max(1, total_n // 5)
    n_clin = max(1, total_n // 4)
    n_soc = max(1, total_n // 4)
    n_none = total_n - (n_both + n_clin + n_soc)
    return n_clin, n_soc, n_both, n_none


class NERDataset:
    def __init__(self, tokenizer_name, max_length=512):
        if "deberta" in tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, add_prefix_space=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def tokenize_and_align_labels(self, examples):
        label_all_tokens = True
        tokenized_inputs = self.tokenizer(
            list(examples["tokens"]), 
            truncation=True, 
            is_split_into_words=True,
            max_length=self.max_length, # Covers most sequences
            padding="longest",  # Save resources for short sequences
            #padding=True,
        )

        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_id = None
            label_ids = []
            
            for word_id in word_ids:
                if word_id is None:  # Special tokens like [CLS], [SEP]
                    label_ids.append(-100)  # Ignored in the loss calculation
                    
                elif word_id != previous_word_id: # First token of a word
                    label_id = -100 if word_id is None else label[word_id]
                    label_ids.append(label_id)
                    
                else:  # Subword tokens
                    label_ids.append(label[word_id])
                previous_word_id = word_id
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs
        
def tokenize_and_align_labels_with_dict(examples, ner_dataset):
    """
    Wrapper function to call tokenize_and_align_labels with the ner_dataset instance.
    """
    return ner_dataset.tokenize_and_align_labels(examples)

# Function to convert NER tag IDs to string labels
def convert_ner_tags_str_tag_num(example):
    example['ner_tags'] = [settings.label2id[tag] for tag in example['ner_tags_str']]
    return example

def reddit_impacts_dataset(seed):

    columns_to_convert = ['tokens', 'labels', 'ner_tags']
    train_df = pd.read_csv('new_data/new_train_data.csv')
    test_df = pd.read_csv('new_data/new_test_data.csv')
    dev_df = pd.read_csv('new_data/new_dev_data.csv')

    merged_df = pd.concat([train_df, dev_df], ignore_index=False)
    merged_df = merged_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


    # Splitting 90% train, 10% dev
    train_df, dev_df = train_test_split(merged_df, test_size=0.1, random_state=seed)
    train_df[columns_to_convert] = train_df[columns_to_convert].applymap(ast.literal_eval) # convert string to list 
    test_df[columns_to_convert] = test_df[columns_to_convert].applymap(ast.literal_eval) # convert string to list 
    dev_df[columns_to_convert] = dev_df[columns_to_convert].applymap(ast.literal_eval) # convert string to list 

    #-----------------------------------------------------------------------------------------------------------------
    # check model performance with different data size -- activate this section if need to test the model with minimum data
    # logger.info("------------------------- 740 Samples (75%) -----------------------------")
    # n_clin, n_soc, n_both, n_none = compute_allocation(740)
    # print(n_clin, n_soc, n_both, n_none)
    # train_df = get_balanced_sample(train_df, total_n=740, n_clinical=n_clin, n_social=n_soc, n_both=n_both, seed=seed)
    # print(train_df)
    #-----------------------------------------------------------------------------------------------------------------
    train_df = train_df.reset_index(drop=True)
    dev_df = dev_df.reset_index(drop=True)
  

    train_df.rename(columns={'ner_tags': 'ner_tags_str'}, inplace=True)
    dev_df.rename(columns={'ner_tags': 'ner_tags_str'}, inplace=True)
    test_df.rename(columns={'ner_tags': 'ner_tags_str'}, inplace=True)

    # print(test_df)
    # print(type(test_df['ner_tags_str'][0]))

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    dev_dataset = Dataset.from_pandas(dev_df)

    logger.info(f"train = {train_dataset}, dev = {dev_dataset}, test = {test_dataset}")

    # Apply the conversion function to the dataset
    train_dataset = train_dataset.map(convert_ner_tags_str_tag_num)
    dev_dataset = dev_dataset.map(convert_ner_tags_str_tag_num)
    test_dataset = test_dataset.map(convert_ner_tags_str_tag_num)
    return train_dataset, dev_dataset, test_dataset

if __name__ == "__main__":
    train_dataset, dev_dataset, test_dataset = reddit_impacts_dataset(42)
    print(train_dataset)
    dd = pd.DataFrame(train_dataset)
    print(dd.loc[0])
    print(type(dd['ID'][0]))

    # print("----------------------------")
    # for val in dd[dd['ID'] == 'Train_1']['labels']:
    #     print(val)
    
    # clinical, social = 0,0
    # for row in pd.DataFrame(test_dataset)['labels']:
    #     for r in row:
    #         if r != '_':
    #             if r == 'ClinicalImpacts':
    #                 clinical += 1
    #             elif r == 'SocialImpacts':
    #                 social += 1 
    # print(clinical, social)