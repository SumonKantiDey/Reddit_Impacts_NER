import os
import json
import ast
import torch
import requests
import pandas as pd 
from typing import List, Tuple, Dict
from datetime import datetime
import time
import openai
import torch
from sentence_transformers import SentenceTransformer, util

pd.set_option('display.max_colwidth', None)

model = SentenceTransformer('all-MiniLM-L6-v2')
output_json_file = "./test_pred_files/llms/"

# configure API version
openai.api_type = "azure"
openai.api_base = ""
openai.api_version = ""
openai.api_key = ""
deployment_name = "gpt-4o"

def save_results(results, json_file):
    """
    Save results to a JSON file.
    """
    if os.path.exists(json_file):
        with open(json_file, 'r',encoding="utf-8") as file:
            data = json.load(file)
    else:
        data = []

    data.append(results)
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def read_json_file(gpt4_json_file):
    with open(gpt4_json_file, 'r') as f:
        data = json.load(f)
    return data

# Function to check if both labels are present
def has_both_impacts(ner_tags):
    return ('ClinicalImpacts' in ner_tags) and ('ClinicalImpacts' in ner_tags)

def few_shot_examples(): 
    data = pd.read_csv('new_data/new_train_data.csv')
    data = data[['tokens', 'ner_tags']]

    data[['tokens', 'ner_tags']] = data[['tokens', 'ner_tags']].applymap(ast.literal_eval) # convert string to list 
    both_impacts = data[data['ner_tags'].astype(str).str.contains('SocialImpacts') & data['ner_tags'].astype(str).str.contains('ClinicalImpacts')]

    # Create a new column with token-label format
    both_impacts['token_label'] = both_impacts.apply(
        lambda row: [
            f"{token}-{label.split('-')[-1]}" if '-' in label else f"{token}-{label}" 
            for token, label in zip(row['tokens'], row['ner_tags'])
        ],
        axis=1
    )
    both_impacts['token_length'] = both_impacts['tokens'].apply(len)
    both_impacts = both_impacts.sort_values(by='token_length', ascending=True)
    both_impacts = both_impacts[1:4]
    examples = [(row['tokens'], row['token_label']) for index, row in both_impacts.iterrows()]
    return examples

# few_shot_examples = few_shot_examples()
def get_top_n_matches(shot, input_tokens): 
    data = pd.read_csv('new_data/new_train_data.csv')
    data = data[['tokens', 'ner_tags']]

    data[['tokens', 'ner_tags']] = data[['tokens', 'ner_tags']].applymap(ast.literal_eval) # convert string to list 
    both_impacts = data[data['ner_tags'].astype(str).str.contains('SocialImpacts') & data['ner_tags'].astype(str).str.contains('ClinicalImpacts')]
    both_impacts['text'] = both_impacts['tokens'].apply(lambda x: ' '.join(x))

    # print(both_impacts[10:14])

    # Load model once
    embeddings = model.encode(both_impacts['text'].tolist(), convert_to_tensor=True)
    
    input_text = ' '.join(input_tokens)
    input_embedding = model.encode(input_text, convert_to_tensor=True)
    
    # Cosine similarity
    cosine_scores = util.cos_sim(input_embedding, embeddings)[0]
    top_k = torch.topk(cosine_scores, k=shot)
    
    top_indices = top_k.indices.cpu().numpy()
    top_scores = [score.item() for score in top_k.values]

    top_n_df = both_impacts.iloc[top_indices].copy() # fine because .iloc is position-based
    top_n_df['similarity_score'] = top_scores

    #print(top_n_df) # print top3 from train_df that are matched based on test tokens
    
    # Create a new column with token-label format
    top_n_df['token_label'] = top_n_df.apply(
        lambda row: [
            f"{token}-{label.split('-')[-1]}" if '-' in label else f"{token}-{label}" 
            for token, label in zip(row['tokens'], row['ner_tags'])
        ],
        axis=1
    )
    top_n_df['token_length'] = top_n_df['tokens'].apply(len)
    examples = [(row['tokens'], row['token_label']) for index, row in top_n_df.iterrows()]
    return examples

def build_prompt(shot: int, tokens: List[str]) -> str:
    annotation_guidelines = (
        "=== Strict Annotation Rules ===\n"
        "1. Annotate ONLY first-person experiences. Ignore third-party reports.\n"
        "2. Label all drug names (e.g., 'heroin', 'fentanyl') as 'O'.\n"
        "3. Label personal pronouns (e.g., 'I', 'my') as 'O' — they are not part of entity spans.\n"
        "4. ASSUME opioid involvement UNLESS a non-opioid cause is clearly stated.\n"
        "5. If multiple substances are mentioned, DEFAULT to opioid-related impact when unsure.\n"
        "6. Label mental health terms (e.g., 'depression') as ClinicalImpacts unless context clearly shows a non-opioid cause.\n"
        "7. Label non-integral words (e.g., adjectives, adverbs, or temporal words like 'very', 'suddenly') as 'O' if they are not essential to the entity span.\n"
        "8. For corrupted or unreadable tokens (e.g., 'Ìm', '?', '##', etc.), ALWAYS label as 'O'.\n"
        "9. Maintain the exact token order and label each token with either\n"
        "10. Continue labeling all tokens until the end. Do not output '?' or stop early. If unsure about any token, label it as 'O'.\n"

        "=== Output Format ===\n"
        "Return token-label pairs like: 'token-Label token-Label ...'\n"
    )

    prompt = (
        "You are a medical AI assistant that classifies tokens into:\n"
        "- ClinicalImpacts: Health/well-being effects.\n"
        "- SocialImpacts: Societal/community effects.\n"
        "- O: Tokens outside these categories.\n\n"
        f"{annotation_guidelines}\n"
        # "=== Examples ===\n"
    )
    
    # examples = few_shot_examples()
    if shot != 0:
        examples = get_top_n_matches(shot, tokens)

        for i, (toks, labels) in enumerate(examples, 1):
            combined = " ".join(labels)
            prompt += (
                f"Example {i}:\n"
                f"{combined}\n\n"
            )
    
    prompt += (
        "=== New Input ===\n"
        f"Tokens: {tokens}\n"
        "Output:"
    )
    return prompt

def classify_with_gpt4o(shot: int, tokens: List[str]):
    response_text = None
    prompt = build_prompt(shot, tokens)
    # print("prompt = ", prompt)

    try:
        response = openai.ChatCompletion.create(
            # model="gpt-4o",
            engine=deployment_name,  # Azure deployment name
            messages=[
                {"role": "system", "content": "You are an expert medical NER annotator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=300
        )
        output_text = response["choices"][0]["message"]["content"].strip()
        # print("output text = ", output_text)
    except Exception as e:
        print(f"Error processing API response: {e}")
        output_text = None

    return output_text

# def top_3_similarity_check():
    # test_df = pd.read_csv('new_data/new_test_data.csv')
    # test_df = test_df[['tokens', 'ner_tags']]
    # test_df[['tokens', 'ner_tags']] = test_df[['tokens', 'ner_tags']].applymap(ast.literal_eval) # convert string to list 

    # for index, row in test_df.iterrows():
    #     tokens = row['tokens']
    #     truth_label = row['ner_tags']
    #     # print(index, tokens, truth_label)
    #     data = get_top3_matches(tokens)
    #     print("tokens = ", tokens)
    #     for i, (toks, labels) in enumerate(data, 1):
    #         combined = " ".join(labels)
    #         prompt = (
    #             f"Example {i}:\n"
    #             f"{combined}\n\n"
    #         )
    #         print(prompt)
    #         print("-"*100)
    #     break

def main(shot):
    test_df = pd.read_csv('new_data/new_test_data.csv')
    test_df = test_df[['tokens', 'ner_tags']]
    test_df[['tokens', 'ner_tags']] = test_df[['tokens', 'ner_tags']].applymap(ast.literal_eval) # convert string to list 

    gpt4_json_file = f"{output_json_file}gpt4_{shot}_shot.json" 

    stored_data = read_json_file(gpt4_json_file)
    stored_indices = [d['index'] for d in stored_data]
    print(stored_indices)
    for index, row in test_df.iterrows():
        if index in stored_indices: 
            continue

        tokens = row['tokens']
        truth_label = row['ner_tags']
        # print(index, tokens, truth_label)

        prediction = classify_with_gpt4o(shot,tokens)
        if prediction == None: 
            prediction = []

        pred_results = {
            "index": index,
            "tokens": str(tokens), 
            "truth_label": str(truth_label), 
            "pred_label": prediction
        }
        print(pred_results)
        save_results(pred_results, gpt4_json_file)
        time.sleep(5)

#top_3_similarity_check()
main(shot=0) # shot = 0, 3, 5
