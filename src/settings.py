import os
import logging
import logging.config

DEBUG = os.environ.get("DEBUG", '').lower()

DEBUG_LOG_FILE = os.environ.get("DEBUG_LOG_FILE", '').strip() or "/labs/sarkerlab/sdey26/Reddit_Impacts_NER/logs/debug.log"
INFO_LOG_FILE = os.environ.get("INFO_LOG_FILE", '').strip() or "/labs/sarkerlab/sdey26/Reddit_Impacts_NER/logs/info.log"
ERROR_LOG_FILE = os.environ.get("ERROR_LOG_FILE", '').strip() or "/labs/sarkerlab/sdey26/Reddit_Impacts_NER/logs/error.log"

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },

    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "stream": "ext://sys.stdout"
        },

        "debug_file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "filename": DEBUG_LOG_FILE,
            "maxBytes": 10485760,
            "backupCount": 20,
            "encoding": "utf8"
        },

        "info_file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "simple",
            "filename": INFO_LOG_FILE,
            "maxBytes": 10485760,
            "backupCount": 20,
            "encoding": "utf8"
        },

        "error_file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "simple",
            "filename": ERROR_LOG_FILE,
            "maxBytes": 10485760,
            "backupCount": 20,
            "encoding": "utf8"
        }
    },

    "loggers": {
        "my_module": {
            "level": "ERROR",
            "handlers": ["console"],
            "propagate": False
        }
    },

    "root": {
        "level": "INFO" if DEBUG else "INFO",
        "handlers": ["console", "info_file_handler", "error_file_handler"]
    }
}

logging.config.dictConfig(LOGGING_CONFIG)

# Define the mapping for labels
label_names = ['O', 'B-ClinicalImpacts','I-ClinicalImpacts','B-SocialImpacts', 'I-SocialImpacts']
id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}

plm_models_config = {
    "distilbert-base-uncased":"distilbert-base-uncased",
    "roberta-large-ner-english": "Jean-Baptiste/roberta-large-ner-english",
    "xlm-roberta-large":"xlm-roberta-large-finetuned-conll03-english",
    "roberta-base":"roberta-base",
    "roberta-large":"roberta-large",
    "debarta-large": "microsoft/deberta-large", # microsoft/deberta-v3-large" not giving good score
    "bert-large-uncased":"bert-large-uncased",
    "biobert-large-cased":"dmis-lab/biobert-large-cased-v1.1",
    "OpenBioLLM8B": "aaditya/OpenBioLLM-Llama3-8B", # 'deepseek-ai/deepseek-coder-1.3b-base'
    "TinyLlama":"TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "BiomedicalLLM":"ContactDoctor/Bio-Medical-Llama-3-8B",
    "bert-large-uncased-mlm": "/labs/sarkerlab/sdey26/reddit_impacts/mlm_training/bert-large-uncased"
    # "BiomedicalLLM":"meta-llama/Meta-Llama-3-8B"
}

models_config = {
    "distilbert-base-uncased": "distilbert-base-uncased",
    "microsoft/deberta-large":"debarta-large", # microsoft/deberta-v3-large" not giving good score
    "Jean-Baptiste/roberta-large-ner-english": "roberta-large-ner-english",
    "/labs/sarkerlab/sdey26/reddit_impacts/mlm_training/roberta-large-ner": "roberta-large-ner-mlm-english",
    "xlm-roberta-large-finetuned-conll03-english": "xlm-roberta-large",
    "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract": "BiomedBERT",
    "roberta-base": "roberta-base",
    "roberta-large": "roberta-large",
    "bert-large-uncased": "bert-large-uncased",
    "dmis-lab/biobert-large-cased-v1.1": "BioBERT-large-cased",
    "aaditya/OpenBioLLM-Llama3-8B": "OpenBioLLM8B",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "TinyLlama",
    "ContactDoctor/Bio-Medical-Llama-3-8B": "BiomedicalLLM",
    "/labs/sarkerlab/sdey26/reddit_impacts/mlm_training/bert-large-uncased": "bert-large-uncased-mlm"
}




