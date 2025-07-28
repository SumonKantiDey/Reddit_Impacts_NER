import json, ast
import pandas as pd 
import re
from .evaluation import calculate_f1_per_entity_covering_all

# Show full column content without truncation
pd.set_option('display.max_colwidth', None)

def preprocess_ner_labels(df, column='ner_tags'):
    """
    Clean and standardize NER labels in a pandas column with list of tags.
    Handles case, whitespace, quote issues, and known label variations.
    """
    label_map = {
        "socialimpact": "SocialImpacts",
        "socialimpacts": "SocialImpacts",
        "clinicalimpact": "ClinicalImpacts",
        "clinicalimpacts": "ClinicalImpacts",
        "o": "O"
    }

    def normalize(label):
        # Remove any leading/trailing quotes, whitespace, make lowercase
        clean = label.strip().strip("'\"").lower()
        return label_map.get(clean, label.strip().strip("'\""))

    def normalize_labels(label_list):
        return [normalize(label) for label in label_list]

    df = df.copy()
    df[column] = df[column].apply(normalize_labels)
    return df

def clean_O_tags(df, column='ner_tags'):
    """
    Cleans up tags like "O')" and ensures they're treated as 'O'.
    """

    def clean_label(label):
        # cleaned = re.sub(r"[\\'\"\`)\s]+$", "", label.strip())
        cleaned = re.sub(r"^[\'\"\`\s]+|[\'\"\`\)\s]+$", "", label.strip())
        # cleaned = re.sub(r"[\\'\")\s]+$", "", label.strip())
        return "O" if cleaned == "o" else cleaned

    df = df.copy()
    df[column] = df[column].apply(lambda tag_list: [clean_label(tag) for tag in tag_list])
    return df

def to_bio_tags(label_sequence):
    bio_tags = []
    prev = "O"

    for label in label_sequence:
        if label == "O":
            bio_tags.append("O")
            prev = "O"
        elif label != prev:
            bio_tags.append(f"B-{label}")
            prev = label
        else:
            bio_tags.append(f"I-{label}")
    return bio_tags

 
shot = 0 # 3,5

# llama
llama_json_file = f"./test_pred_files/llms/llama_{shot}_shot.json"
llama_xlsx = f"./test_pred_files/llms/llama_{shot}_shot.xlsx"

# Gemma
gemma_json_file = f"./test_pred_files/llms/gemma_{shot}_shot.json"
gemma_xlsx = f"./test_pred_files/llms/gemma_{shot}_shot.xlsx"

# gpt4
gpt4_json_file = f"./test_pred_files/llms/gpt4_{shot}_shot.json"
gpt4_xlsx = f"./test_pred_files/llms/gpt4_{shot}_shot.xlsx"

save_xlsx = gpt4_xlsx

def read_json_file():
    with open(gpt4_json_file, 'r') as f:
        data = json.load(f)
    return data

data = read_json_file()

# Convert to DataFrame
df = pd.DataFrame(data)
# print(df)

# Convert string representations of lists to actual Python lists
df['tokens'] = df['tokens'].apply(ast.literal_eval)
df['ner_tags_str'] = df['truth_label'].apply(ast.literal_eval)

pred_list = []
tag_map = {}
for index, row  in df.iterrows():
    print("index = ", index)
    ind = row['index']
    tokens = row['tokens']
    # print(row['pred_label'])
    items = row['pred_label'].strip().split(' ') # split the prediction label
    # print(index, items, len(items))
    for item in items:
        if '-' in item:
            idx = item.rfind('-')  # Handles tokens like "anti-inflammatory-O"
            token = item[:idx]
            tag = item[idx+1:]
            tag_map[token] = tag    
    output = [(token, tag_map.get(token, 'O')) for token in tokens] 
    pred_list.append(output)

df['preds_tuple'] = pred_list
df['pred_label'] = df['preds_tuple'].apply(lambda pairs: [label for _, label in pairs])
df = preprocess_ner_labels(df, column='pred_label')
df = clean_O_tags(df,column='pred_label')
print(type(df['pred_label'][0]))

# Add a new column that checks if lengths match
df['label_length_match'] = df.apply(lambda row: len(row['pred_label']) == len(row['ner_tags_str']), axis=1)
df['length_difference'] = df.apply(lambda row: len(row['pred_label']) - len(row['ner_tags_str']), axis=1)

# Print rows where lengths do not match (if any)
mismatched = df[~df['label_length_match']]

if mismatched.empty:
    print("✅ All rows have matching lengths for pred_label and ner_tags_str.")
else:
    print("❌ Mismatched rows found:")
    print(mismatched[['index', 'tokens', 'ner_tags_str', 'pred_label', 'length_difference']])

df = df[['index','tokens', 'ner_tags_str','pred_label']]
df['prediction'] = df['pred_label'].apply(to_bio_tags)
# df = df[:275]
# df.to_excel('temp.xlsx', index=False)

kk = df[df['index'] == 247]
for index, row in kk.iterrows(): 
    print(row['tokens'])
    print(row['ner_tags_str'])
    print(row['pred_label'])
    print(row['prediction'])

results_per_entity = calculate_f1_per_entity_covering_all(df['ner_tags_str'], df['prediction'])
print(results_per_entity)
print("-------"*20)

print(f"Relaxed F1 Score Results Per Entity for model {save_xlsx}")
for entity, metrics in results_per_entity.items():
    result_str = f"Entity Type: {entity}\n"
    for metric, value in metrics.items():
        # print(f"  {metric}: {value}")
        result_str += f"  {metric}: {value}\n"
    print(f"\n {result_str}")

df.to_excel(f"{save_xlsx}", index=False)