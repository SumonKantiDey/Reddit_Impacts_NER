import pandas as pd
import ast 

df = pd.read_csv("/labs/sarkerlab/sdey26/Reddit_Impacts_NER/new_data/new_train_data.csv")
columns_to_convert = ['tokens', 'labels']
df[columns_to_convert] = df[columns_to_convert].applymap(ast.literal_eval)



# Flatten and count total tokens
total_tokens = df['tokens'].apply(len).sum()

# Flatten and count entity occurrences
flat_labels = [label for sublist in df['labels'] for label in sublist]
social_count = flat_labels.count('SocialImpacts')
clinical_count = flat_labels.count('ClinicalImpacts')

# Output summary
summary = {
    'Total Rows': len(df),
    'Total Tokens': total_tokens,
    'Total SocialImpacts': social_count,
    'Total ClinicalImpacts': clinical_count
}

print(summary)