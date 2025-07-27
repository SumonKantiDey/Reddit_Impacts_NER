# Reddit_Impacts_NER

# Dataset 

# Methodology

# Results
Relaxed F1 Score Results Per Entity for model `bert-large-uncased`

| Entity Type       | Precision    | Recall    | F1-Score    | Coverage     |
|-------------------|--------------|-----------|-------------|--------------|
| ClinicalImpacts   | 0.639        | 0.457     | 0.533       | 117 / 256    |
| SocialImpacts     | 0.767        | 0.306     | 0.437       | 33 / 108     |
| **Overall**       | **0.664**    | **0.412** | **0.508**   | **150 / 364**|

Relaxed F1 Score Results Per Entity for model `bert-large-uncased-mlm`
| Entity Type      | Precision | Recall | F1-Score | Coverage    |
|------------------|-----------|--------|----------|-------------|
| ClinicalImpacts  | 0.719     | 0.539  | 0.616    | 138 / 256   |
| SocialImpacts    | 0.579     | 0.204  | 0.301    | 22 / 108    |
| Overall          | 0.696     | 0.440  | 0.539    | 160 / 364   |


Relaxed F1 Score Results Per Entity for model **xlm-roberta-large**
| Entity Type       | Precision    |  Recall   | F1-Score    |  Coverage  |
|-------------------|--------------|-----------|-------------|--------------|
| ClinicalImpacts   | 0.659        | 0.445     | 0.531       | 114 / 256    |
| SocialImpacts     | 0.476        | 0.278     | 0.351       | 30 / 108     |
| **Overall**       | **0.610**    | **0.396** | **0.480**   | **144 / 364** |


Relaxed F1 Score Results Per Entity for model `biobert-large-cased`
|   Entity Type     |   Precision  |   Recall  |  F1-Score   |  Coverage   |
|-------------------|--------------|-----------|-------------|--------------|
| ClinicalImpacts   | 0.703        | 0.379     | 0.492       | 97 / 256     |
| SocialImpacts     | 0.488        | 0.194     | 0.278       | 21 / 108     |
| **Overall**       | **0.652**    | **0.324** | **0.433**   | **118 / 364** |

Relaxed F1 Score Results Per Entity for model `roberta-large-ner-english`
| Entity Type      | Precision | Recall | F1-Score | Coverage    |
|------------------|-----------|--------|----------|-------------|
| ClinicalImpacts  | 0.633     | 0.539  | 0.582    | 138 / 256   |
| SocialImpacts    | 0.578     | 0.241  | 0.340    | 26 / 108    |
| **Overall**      | 0.624     | 0.451  | 0.523    | 164 / 364   |

# CRF based results
Relaxed F1 Score Results Per Entity for model `roberta-base + CRF`
| Entity Type       | Precision | Recall | F1-Score | Coverage  |
|-------------------|-----------|--------|----------|-----------|
| ClinicalImpacts   | 0.637     | 0.508  | 0.565    | 130/256   |
| SocialImpacts     | 0.468     | 0.333  | 0.389    | 36/108    |
| Overall           | 0.591     | 0.456  | 0.515    | 166/364   |

Relaxed F1 Score Results Per Entity for model `roberta-large + CRF`
| Entity Type       | Precision | Recall | F1-Score | Coverage  |
|-------------------|-----------|--------|----------|-----------|
| ClinicalImpacts   | 0.685     | 0.543  | 0.606    | 139/256   |
| SocialImpacts     | 0.627     | 0.343  | 0.443    | 37/108    |
| Overall           | 0.672     | 0.484  | 0.562    | 176/364   |

Relaxed F1 Score Results Per Entity for model `bert-large-uncased + CRF`
| Entity Type       | Precision | Recall | F1-Score | Coverage |
|-------------------|-----------|--------|----------|----------|
| ClinicalImpacts   | 0.534     | 0.582  | 0.557    | 149/256  |
| SocialImpacts     | 0.432     | 0.444  | 0.438    | 48/108   |
| Overall           | 0.505     | 0.541  | 0.523    | 197/364  |

Relaxed F1 Score Results Per Entity for model `bert-large-uncased-mlm + CRF`
| Entity Type       | Precision | Recall | F1-Score | Coverage  |
|-------------------|-----------|--------|----------|-----------|
| ClinicalImpacts   | 0.630     | 0.633  | 0.632    | 162/256   |
| SocialImpacts     | 0.411     | 0.426  | 0.418    | 46/108    |
| Overall           | 0.564     | 0.571  | 0.568    | 208/364   |

Relaxed F1 Score Results Per Entity for model `roberta-large-ner-english + CRF`
| Entity Type     | Precision | Recall | F1-Score | Coverage  |
|-----------------|-----------|--------|----------|-----------|
| ClinicalImpacts | 0.668     | 0.566  | 0.613    | 145/256   |
| SocialImpacts   | 0.584     | 0.417  | 0.486    | 45/108    |
| Overall         | 0.646     | 0.522  | 0.578    | 190/364   |


Relaxed F1 Score Results Per Entity for model `xlm-roberta-large + CRF`
| Entity Type      | Precision | Recall | F1-Score | Coverage  |
|------------------|-----------|--------|----------|-----------|
| ClinicalImpacts  | 0.511     | 0.551  | 0.530    | 141/256   |
| SocialImpacts    | 0.500     | 0.352  | 0.413    | 38/108    |
| Overall          | 0.509     | 0.492  | 0.500    | 179/364   |


### Relaxed F1 Score Results Per Entity – LLaMa 70B (with dynamic prompting)

| Entity Type      | Precision | Recall | F1-Score | Coverage     |
|------------------|-----------|--------|----------|--------------|
| ClinicalImpacts  | 0.513     | 0.473  | 0.492    | 121 / 256    |
| SocialImpacts    | 0.283     | 0.120  | 0.169    | 13 / 108     |
| **Overall**      | 0.475     | 0.368  | 0.415    | 134 / 364    |

### Relaxed F1 Score Results Per Entity – LLaMa 70B (static prompting)
| Entity Type     | Precision | Recall | F1-Score | Coverage    |
|-----------------|-----------|--------|----------|-------------|
| ClinicalImpacts | 0.410     | 0.453  | 0.430    | 116 / 256   |
| SocialImpacts   | 0.389     | 0.065  | 0.111    | 7 / 108     |
| **Overall**     | 0.409     | 0.338  | 0.370    | 123 / 364   |


## Relaxed F1 Score Results Per Entity – GPT4o (with dynamic prompting)

| Entity Type      | Precision | Recall | F1-Score | Coverage    |
|------------------|-----------|--------|----------|-------------|
| ClinicalImpacts  | 0.464     | 0.555  | 0.505    | 142 / 256   |
| SocialImpacts    | 0.276     | 0.250  | 0.262    | 27 / 108    |
| **Overall**      | 0.418     | 0.464  | 0.440    | 169 / 364   |