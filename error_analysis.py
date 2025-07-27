import pandas as pd 
import os, json

def save_results(data, json_file):
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


path = '/Reddit_Impacts_NER'
deberta = pd.read_excel(f"{path}/test_pred_files/debarta-large_6_pred.xlsx")
gpt4 = pd.read_excel(f"{path}/labs/sarkerlab/sdey26/Reddit_Impacts_NER/gpt4_temp.xlsx")


data = []
for (index1, row1), (index2, row2) in zip(deberta.iterrows(), gpt4.iterrows()):

    if row1["tokens"] == row2["tokens"]:
        data.append(
            {
                "index": row2["index"], 
                "tokens" : row2["tokens"],
                "truth_label" : row1["ner_tags_str"],
                "deberta_pred" : row1["prediction"], 
                "gpt4_pred" : row2["prediction"]
            }
        )

json_file = "error_analysis.json"

save_results(data, json_file)
