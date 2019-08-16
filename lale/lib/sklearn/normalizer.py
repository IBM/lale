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

import sklearn.preprocessing.data
import lale.helpers
import lale.operators

class NormalizerImpl():

    def __init__(self, norm=None, copy=True):
        self._hyperparams = {
            'norm': norm,
            'copy': copy}

    def fit(self, X, y=None):
        self._sklearn_model = sklearn.preprocessing.data.Normalizer(**self._hyperparams)
        self._sklearn_model.fit(X, y)
        return self

    def transform(self, X):
        return self._sklearn_model.transform(X)
_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Normalize samples individually to unit norm.',
    'allOf': [{
        'type': 'object',
        'required': ['norm'],
        'relevantToOptimizer': ['norm'],
        'additionalProperties': False,
        'properties': {
            'norm': {
                'enum': ['l1', 'l2', 'max'],
                'default': 'l2',
                'description': 'The norm to use to normalize each non zero sample.'},
            'copy': {
                'type': 'boolean',
                'default': True,
                'description': 'set to False to perform inplace row normalization and avoid a'},
        }}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Do nothing and return the estimator unchanged',
    'type': 'object',
    'required': ['X'],
    'additionalProperties': False,
    'properties': {
        'X': {
            'description': 'Features; the outer array is over samples.',
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {'type': 'number'}}},
        'y': {
            'description': 'Target class labels; the array is over samples.'}}}

_input_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Scale each non zero row of X to unit norm',
    'type': 'object',
    'required': ['X'],
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            },
            'description': 'The data to normalize, row by row. scipy.sparse matrices should be'},
        'copy': {
            'anyOf': [{
                'type': 'boolean'}, {
                'enum': [None]}],
            'default': None,
            'description': 'Copy the input X or not.'},
    },
}
_output_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Scale each non zero row of X to unit norm',
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
        'input_predict': _input_transform_schema,
        'output': _output_transform_schema},
}
if (__name__ == '__main__'):
    lale.helpers.validate_is_schema(_combined_schemas)
Normalizer = lale.operators.make_operator(NormalizerImpl, _combined_schemas)
