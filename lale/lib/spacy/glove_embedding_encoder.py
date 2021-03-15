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
import os
import urllib.request
import zipfile

import numpy as np
from sklearn.utils.validation import _is_arraylike
from spacy.lang.en import English

import lale.docstrings
import lale.operators

logging.basicConfig(level=logging.INFO)


class _GloveEmbeddingEncoderImpl(object):
    """
    _GloveEmbeddingEncoderImpl is a module allows simple generation of sentence embeddings using
    glove word embeddings

    Parameters
    ----------
    dim: int, (default=300), dimension of word embeddings to use
    combiner: string, (default=mean), apply mean pooling or max pooling on word embeddings to generate
        sentence embedding

    References
    ----------
    R. JeffreyPennington and C. Manning. Glove: Global vectors for word representation. 2014

    """

    def __init__(self, dim=300, combiner="mean"):

        self.combiner = combiner
        self.dim = dim

        if self.combiner not in ["mean", "max"]:
            raise ValueError("Combiner must be either mean or max")

        if self.dim not in [50, 100, 200, 300]:
            raise ValueError("dim must be in 50, 100, 200, or 300")

        glove_path = os.path.join(
            os.path.dirname(__file__), "resources", "glove.6B.{}d.txt".format(dim)
        )

        if not os.path.exists(glove_path):
            download_url = "http://nlp.stanford.edu/data/glove.6B.zip"
            download_dir = os.path.join(
                os.path.dirname(__file__), "resources", "glove.6B.zip"
            )
            if not os.path.exists(os.path.join(os.path.dirname(__file__), "resources")):
                os.mkdir(os.path.join(os.path.dirname(__file__), "resources"))
            urllib.request.urlretrieve(download_url, download_dir)
            with zipfile.ZipFile(download_dir, "r") as zip_ref:
                zip_ref.extractall(os.path.join(os.path.dirname(__file__), "resources"))
            os.remove(download_dir)

        self.glove_dict = self._load_glove(glove_path)
        nlp = English()
        self.tokenizer = nlp.Defaults.create_tokenizer(nlp)
        print("finish initialization")

    def fit(self, X, y=None):
        return self

    def _load_glove(self, glove_path):
        glove_dict = dict()
        with open(glove_path, "r", encoding="utf-8") as file_handle:
            for line in file_handle:
                line = line.split()
                glove_dict[line[0]] = np.array(line[1:], dtype=np.float32)
        return glove_dict

    def transform(self, X):
        if not _is_arraylike(X):
            raise TypeError("X is not iterable")

        transformed_X = list()
        for text in X:
            temp_vec = list()
            for token in self.tokenizer(text.lower()):
                temp_vec.append(self.glove_dict.get(token, self.glove_dict["unk"]))

            if self.combiner == "mean":
                sentence_vec = np.mean(temp_vec, axis=0)
            else:
                sentence_vec = np.amax(temp_vec, axis=0)

            transformed_X.append(sentence_vec)
        return transformed_X


_input_schema_fit = {
    "description": "Input data schema for training.",
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Input Text",
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

_input_transform_schema = {
    "description": "Input data schema for training.",
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Input Text",
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
                "combiner": {"enum": ["mean", "max"], "default": "mean"},
                "dim": {"enum": [50, 100, 200, 300], "default": 300},
            },
        }
    ],
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters for a transformer for"
    " a text data transformer based on pre-trained glove embedding",
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


GloveEmbeddingEncoder = lale.operators.make_operator(
    _GloveEmbeddingEncoderImpl(), _combined_schemas
)

lale.docstrings.set_docstrings(GloveEmbeddingEncoder)
