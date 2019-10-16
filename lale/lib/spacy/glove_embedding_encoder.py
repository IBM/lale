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
import spacy
import numpy as np
from lale.operators import make_operator
import lale.helpers
from sklearn.utils.validation import _is_arraylike

logging.basicConfig(level=logging.INFO)


class GloveEmbeddingEncoderImpl(object):
    """
    GloveEmbeddingEncoderImpl is a module allows simple generation of sentence embeddings using
    glove word embeddings

    Parameters
    ----------
    combiner: string, (default=mean), apply mean pooling or max pooling on word embeddings to generate
        sentence embedding

    References
    ----------
    R. JeffreyPennington and C. Manning. Glove: Global vectors for word representation. 2014

    """
    def __init__(self, combiner='mean'):
        self.nlp = spacy.load("en_vectors_web_lg")
        self.combiner = combiner
        if self.combiner not in ["mean", "max"]:
            raise ValueError("Combiner must be either mean or max")

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        if not _is_arraylike(X):
            raise TypeError("X is not iterable")

        if self.combiner == 'mean':
            transformed_x = self._mean(X)
        else:
            transformed_x = self._max(X)

        return transformed_x

    def _mean(self, X):
        transformed_X = list()
        for text in X:
            doc = self.nlp(text)
            transformed_X.append(doc.vector)

        return transformed_X

    def _max(self, X):
        transformed_X = list()
        for text in X:
            doc = self.nlp(text)
            temp_vec = list()
            for token in doc:
                temp_vec.append(token.vector)
            temp_vec = np.amax(temp_vec, axis=0)
            transformed_X.append(temp_vec)

        return transformed_X


_input_schema_fit = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Input data schema for training.',
    'type': 'object',
    'required': ['X'],
    'additionalProperties': False,
    'properties': {
        'X': {
            'description': 'Input Text',
            'anyOf': [
                {'type': 'array',
                 'items': {'type': 'string'}},
                {'type': 'array',
                 'items': {
                     'type': 'array', 'minItems': 1, 'maxItems': 1,
                     'items': {'type': 'string'}}}]}
    }}

_input_schema_predict = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Input data schema for training.',
    'type': 'object',
    'required': ['X'],
    'additionalProperties': False,
    'properties': {
        'X': {
            'description': 'Input Text',
            'anyOf': [
                {'type': 'array',
                 'items': {'type': 'string'}},
                {'type': 'array',
                 'items': {
                     'type': 'array', 'minItems': 1, 'maxItems': 1,
                     'items': {'type': 'string'}}}]}
    }}

_output_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Output data schema for transformed data.',
    'type': 'array',
    'items': {'type': 'array', 'items': {'type': 'number'}}}

_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Hyperparameter schema.',
    'allOf': [
        {'description':
             'This first sub-object lists all constructor arguments with their '
             'types, one at a time, omitting cross-argument constraints.',
         'type': 'object',
         'additionalProperties': False,
         'relevantToOptimizer': [],
         'properties': {
             'combiner': {
                'enum': ['mean', 'max'],
                'default': 'mean'}}}]}

_combined_schemas = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Combined schema for expected data and hyperparameters for a transformer for'
                   ' a text data transformer based on pre-trained glove embedding',
    'type': 'object',
    'tags': {
        'pre': ['text'],
        'op': ['transformer', '~interpretable'],
        'post': ['embedding']},
    'properties': {
        'input_fit': _input_schema_fit,
        'input_predict': _input_schema_predict,
        'output': _output_schema,
        'hyperparams': _hyperparams_schema}}


if __name__ == "__main__":
    lale.helpers.validate_is_schema(_combined_schemas)

GloveEmbeddingEncoder = make_operator(GloveEmbeddingEncoderImpl(), _combined_schemas)
