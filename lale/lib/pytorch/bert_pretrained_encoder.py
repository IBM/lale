# Copyright 2019 IBM Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import numpy as np
import pandas as pd
import torch
import torch.cuda
from pytorch_pretrained_bert import BertModel, BertTokenizer

import lale.docstrings
import lale.operators

logging.basicConfig(level=logging.INFO)


class _BertPretrainedEncoderImpl:
    def __init__(self, batch_size=32):
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_seq_length = self.tokenizer.max_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.batch_size = batch_size

    # def fit(self, X, y):
    #     # TODO: Find the right value for max sequence length
    #     return _BertPretrainedEncoderImpl()

    def transform(self, X):
        if isinstance(X, np.ndarray) or isinstance(X, pd.DataFrame):
            X = X.squeeze()
        self.model.eval()
        self.model.to(self.device)
        transformed_X = None
        num_batches = len(X) // self.batch_size
        last_batch_size = len(X) % self.batch_size
        for batch_idx in range(num_batches + 1):
            min_idx = batch_idx * self.batch_size
            if batch_idx + 1 <= num_batches:
                max_idx = (batch_idx + 1) * self.batch_size
            else:
                max_idx = batch_idx * self.batch_size + last_batch_size
            batch_data = X[min_idx:max_idx]
            indexed_tokenized_X = []
            segments_ids = []
            # Convert token to vocabulary indices
            for line in batch_data:
                tokenized_text = self.tokenizer.tokenize(f"[CLS] {line} [SEP]")
                tokenized_ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)
                if len(tokenized_ids) > self.max_seq_length - 2:
                    tokenized_ids = tokenized_ids[: (self.max_seq_length)]

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1] * len(tokenized_ids)
                # Zero-pad up to the sequence length.
                padding = [0] * (self.max_seq_length - len(tokenized_ids))
                padded_tokenized_ids = tokenized_ids + padding
                input_mask += padding
                indexed_tokenized_X.append(padded_tokenized_ids)

                # This transformer is only applicable for single sentences as we are passing all
                # segment ids as 1. BERT has a notion of a sentence pair, say for a question-answering task
                # that would need a different handling of segments_ids
                segments_ids.append([1 for token in padded_tokenized_ids])

            # Convert inputs to PyTorch tensors
            tokens_tensor = torch.tensor(indexed_tokenized_X)
            segments_tensors = torch.tensor(segments_ids)

            # Predict hidden states features for each layer
            with torch.no_grad():
                encoded_layers, _ = self.model(
                    tokens_tensor.to(self.device), segments_tensors.to(self.device)
                )
            batch_X = encoded_layers[-1][:, 0, :].detach().cpu().numpy()
            if transformed_X is None:
                transformed_X = batch_X
            else:
                transformed_X = np.vstack((transformed_X, batch_X))
        return transformed_X


_input_schema_fit = {
    "description": "Input data schema for training.",
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "anyOf": [
                {"type": "array", "items": {"type": "string"}},
                {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 1,
                        "items": {"type": "string"},
                    },
                },
            ],
        },
        "y": {"description": "Labels, optional."},
    },
}

_input_transform_schema = {
    "description": "Input data schema for predictions.",
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "anyOf": [
                {"type": "array", "items": {"type": "string"}},
                {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 1,
                        "items": {"type": "string"},
                    },
                },
            ],
        }
    },
}

_output_transform_schema = {
    "description": "Output data schema for transformed data.",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}


_hyperparams_schema = {
    "description": "Hyperparameter schema.",
    "allOf": [
        {
            "description": "This first sub-object lists all constructor arguments with their "
            "types, one at a time, omitting cross-argument constraints.",
            "type": "object",
            "additionalProperties": False,
            "relevantToOptimizer": [],
            "properties": {
                "batch_size": {
                    "description": "Batch size used for transform.",
                    "type": "integer",
                    "default": 64,
                    "minimum": 1,
                    "distribution": "uniform",
                    "minimumForOptimizer": 32,
                    "maximumForOptimizer": 128,
                }
            },
        }
    ],
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters for a transformer for"
    " a text data transformer based on pre-trained BERT model "
    "(https://github.com/huggingface/pytorch-pretrained-BERT).",
    "type": "object",
    "tags": {
        "pre": ["text"],
        "op": ["transformer", "~interpretable"],
        "post": ["embedding"],
    },
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_schema_fit,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}


BertPretrainedEncoder = lale.operators.make_operator(
    _BertPretrainedEncoderImpl, _combined_schemas
)

lale.docstrings.set_docstrings(BertPretrainedEncoder)
