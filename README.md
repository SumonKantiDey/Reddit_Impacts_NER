### Inference Gap in Domain Expertise and Machine Intelligence in Named Entity Recognition: Creation of and Insights from a Substance Use-related Dataset

This repository contains the code and supplementary materials for the paper. It will be updated with more reproducible resources, such as models, training scripts, and more.

### Folder structure ###
- `./new_data` - Contains all dataset-related files.
- `./results` - Stores evaluation results generated from the test dataset.
- `./notebooks` - Includes notebooks for data preprocessing and exploration.
- `./src` - Contains source code for Named Entity Recognition (NER) using LLMs and PLMs.

### Code usage instructions ### 
First clone this repo and move to the directory. Then, install necessary libraries. Also, following commands can be used:
```bash
$ git clone https://github.com/SumonKantiDey/Reddit_Impacts_NER.git
$ cd Reddit_Impacts_NER/ 
$ sudo pip install -r requirements.txt
```

### Model Training and Inference ### 
Train and evaluate **PLM-based** models across multiple random seeds using the following scripts:
```bash
# Train and evaluate PLM model
src/train_plm.sh
# Train and evaluate PLM + CRF model
src/train_plm_crf.sh
```
####  These scripts will:
- Train the model using different seeds
- Evaluate on the test dataset
- Compute confidence intervals


> ✨ **Tip:** You can modify parameters (e.g., model name, GPU ID) directly within the script files.

### Few-shot Inference with LLMs ### 
We support few-shot prompting using several large language models. Run the desired script from below:
```bash
# Few-shot inference using GPT-4
python src/few_shot_gpt4.py
# Few-shot inference using LLaMA
python src/few_shot_llama.py
# Few-shot inference using Gemma
python src/few_shot_gemma.py
```
These scripts will:
- Perform NER via in-context learning with few-shot examples
- Generate predictions using LLMs
> ⚠️ Make sure you have appropriate API access or local setup for each LLM.

### Methods ###
![Model Architecture](/Reddit_Impacts_NER/figs/method.png)
 
### Dataset ###
The dataset will be made available upon reasonable request.

### Results ###
![results](/Reddit_Impacts_NER/figs/results.png)
![entity-level](/Reddit_Impacts_NER/figs/entity-level.png)
