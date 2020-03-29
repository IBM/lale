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

import sklearn.feature_selection
import lale.helpers
import lale.operators
import pandas as pd

class SelectKBestImpl():
    def __init__(self, score_func=None, k=10):
        if score_func:
            self._hyperparams = {
                'score_func': score_func,
                'k': k}
        else:
            self._hyperparams = {
                'k': k
            }
        self._sklearn_model = sklearn.feature_selection.SelectKBest(**self._hyperparams)

    def fit(self, X, y=None):
        self._sklearn_model.fit(X, y)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            keep_indices = self._sklearn_model.get_support(indices=True)
            keep_columns = [X.columns[i] for i in keep_indices]
            result = X[keep_columns]
        else:
            result = self._sklearn_model.transform(X)
        return result

_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Select features according to the k highest scores.',
    'allOf': [{
        'type': 'object',
        'required': ['score_func', 'k'],
        'relevantToOptimizer': ['k'],
        'additionalProperties': False,
        'properties': {
            'score_func':  {
                'anyOf': [{}, {
                    'enum': [None]}],
                'default': None,
                'description': 'Function taking two arrays X and y, and returning a pair of arrays (scores, pvalues) or a single array with scores.'},
            'k': {
                'anyOf': [{
                    'type': 'integer',
                    'minimumForOptimizer': 2,
                    'maximumForOptimizer': 15}, {
                    'enum': ['all']}],
                'default': 10,
                'description': 'Number of top features to select'},
        }}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Run score function on (X, y) and get the appropriate features.',
    'type': 'object',
    'required': ['X', 'y'],
    'additionalProperties': False,
    'properties': {
        'X': {
            'description': 'Training input samples.',
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {'type': 'number'}}},
        'y': {
            'type': 'array',
            'items': {'type': 'number'},
            'description': 'Target values.'}}}

_input_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Reduce X to the selected features.',
    'required': ['X'],
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            },
            'description': 'The input samples'},
    },
}
_output_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'The input samples with only the selected features.',
    'type': 'array',
    'items': {
        'type': 'array',
        'items': {'type': 'number'}}}

_combined_schemas = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Combined schema for expected data and hyperparameters.',
    'documentation_url': 'https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html',
    'type': 'object',
    'tags': {
        'pre': [],
        'op': ['transformer'],
        'post': []},
    'properties': {
        'hyperparams': _hyperparams_schema,
        'input_fit': _input_fit_schema,
        'input_transform': _input_transform_schema,
        'output_transform': _output_transform_schema}}

if (__name__ == '__main__'):
    lale.helpers.validate_is_schema(_combined_schemas)
SelectKBest = lale.operators.make_operator(SelectKBestImpl, _combined_schemas)
