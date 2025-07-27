import logging
import torch
import os
import itertools
import pandas as pd 
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    RobertaTokenizerFast,
    AutoConfig, 
    AutoModel,
    AutoModelForTokenClassification, 
    RobertaForTokenClassification,
    TrainingArguments, 
    Trainer,
    PreTrainedModel,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback)

from .utils import setup_device, CustomLoggingCallback, get_parser, load_tokenizer
from torchcrf import CRF
from torch import nn

parser = get_parser()
args = parser.parse_args()


class RobertaCRFForNER(torch.nn.Module):
    def __init__(self, num_labels, id2label, label2id, device):
        super().__init__()
        self.roberta = RobertaForTokenClassification.from_pretrained(
            args.model_name, 
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
            return_dict=True
        )
        self.crf = CRF(num_labels, batch_first=True)
        self.device = device

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        if labels is not None:
            # Move everything to same device
            logits = logits.to(self.device)
            labels = labels.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            # Mask invalid labels (optional)
            labels[labels < 0] = 0  # Replace -100 with 0 (or handle differently)
              
            loss = -self.crf(logits, labels, mask=attention_mask.bool())
            return {"loss": loss, "logits": logits}
        else:
            return self.crf.decode(logits, mask=attention_mask.bool())

class BertCRFForNER(nn.Module):
    def __init__(self, num_labels, id2label, label2id, device):
        super().__init__()
        self.bert = AutoModelForTokenClassification.from_pretrained(
            args.model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,

            hidden_dropout_prob=args.dropout,
            attention_probs_dropout_prob=args.dropout,
            ignore_mismatched_sizes=True,

            return_dict=True
        )
        self.crf = CRF(num_labels, batch_first=True)
        self.device = device

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        if labels is not None:
            # Move everything to the correct device
            logits = logits.to(self.device)
            labels = labels.to(self.device)
            attention_mask = attention_mask.to(self.device)

            labels = labels.clone()
            labels[labels < 0] = 0  # Replace ignored labels

            loss = -self.crf(logits, labels, mask=attention_mask.bool())
            return {"loss": loss, "logits": logits}
        else:
            decode_mask = attention_mask.bool()
            pred_ids = self.crf.decode(logits, mask=decode_mask)
            return pred_ids


class BERTForTokenClassificationWithCRF(PreTrainedModel):
    def __init__(self, config, device=None):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = AutoModel.from_pretrained(args.model_name)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)
        self._device = device
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = self.dropout(outputs[0])
        emissions = self.classifier(sequence_output)

        if labels is not None:
            emissions = emissions.to(self._device)
            labels = labels.to(self._device)
            attention_mask = attention_mask.to(self._device)
            
            labels = labels.clone()
            labels[labels < 0] = 0  # Replace ignored labels

            loss = -self.crf(emissions, labels, mask=attention_mask.bool(), reduction='mean')
            return {"loss": loss, "logits": emissions}
        else:
            prediction = self.crf.decode(emissions, mask=attention_mask.bool())
            return prediction

class EncoderForTokenClassificationWithBiLSTMCRF(PreTrainedModel):
    def __init__(self, config, device=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self._device = device

        # BERT encoder
        self.encoder = AutoModel.from_pretrained(args.model_name)

        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # BiLSTM layer (input = hidden size of BERT, output = same size by default)
        self.lstm_hidden_size = config.hidden_size
        self.bilstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=self.lstm_hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Classifier layer for CRF emissions
        self.classifier = nn.Linear(self.lstm_hidden_size, self.num_labels)

        # CRF layer
        self.crf = CRF(self.num_labels, batch_first=True)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None, token_type_ids=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = self.dropout(outputs[0])  # (batch_size, seq_len, hidden_size)

        # BiLSTM layer
        lstm_out, _ = self.bilstm(sequence_output)  # (batch_size, seq_len, hidden_size)

        # Linear layer to get emission scores for CRF
        emissions = self.classifier(lstm_out)  # (batch_size, seq_len, num_labels)

        if labels is not None:
            # Move to correct device
            emissions = emissions.to(self._device)
            labels = labels.to(self._device)
            attention_mask = attention_mask.to(self._device)

            # Replace -100 or ignored labels with 0 to avoid CRF crash
            labels = labels.clone()
            labels[labels < 0] = 0

            loss = -self.crf(emissions, labels, mask=attention_mask.bool(), reduction='mean')
            return {"loss": loss, "logits": emissions}
        else:
            prediction = self.crf.decode(emissions, mask=attention_mask.bool())
            return prediction